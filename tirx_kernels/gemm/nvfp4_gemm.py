from __future__ import annotations

from enum import IntEnum

import flashinfer
import torch
from flashinfer import SfLayout, nvfp4_quantize

import tvm
from tvm.backend.cuda.operator.tile_primitive.gemm_async.tcgen05 import sf_smem_layout
from tvm.backend.cuda.operator.tile_primitive.tma_utils import SwizzleMode
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.bench import bench
from tvm.tirx.lang.pipeline import MBarrier, Pipeline, PipelineState, TMABar
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D


class WarpRole(IntEnum):
    MMA = 0
    TMA = 2
    EPILOGUE = 4


def prepare_data(M: int, N: int, K: int, *, return_origin: bool = False):
    torch.manual_seed(0)
    A_origin = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B_origin = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    A_global_sf = 448 * 6 / A_origin.float().abs().nan_to_num().max()
    B_global_sf = 448 * 6 / B_origin.float().abs().nan_to_num().max()
    A_fp4, A_sf = nvfp4_quantize(
        A_origin, A_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    B_fp4, B_sf = nvfp4_quantize(
        B_origin, B_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    alpha = 1.0 / (A_global_sf * B_global_sf)
    C_ref = torch.mm(A_origin, B_origin.T)
    if return_origin:
        return (A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref, A_origin, B_origin)
    return (A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref)


_CUBLASLT_EXT = None


def _load_cublaslt_nvfp4_ext():
    """Load the cuBLASLt NVFP4 baseline as a PyTorch inline extension."""
    global _CUBLASLT_EXT
    if _CUBLASLT_EXT is not None:
        return _CUBLASLT_EXT

    from torch.utils.cpp_extension import CUDA_HOME, load_inline

    source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#define CHECK_CUBLAS_THROW(call)                                             \
  do {                                                                       \
    cublasStatus_t status = call;                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                   \
      throw std::runtime_error("cuBLASLt error status=" +                    \
                               std::to_string(static_cast<int>(status)));     \
    }                                                                        \
  } while (0)

#define CHECK_CUDA_THROW(call)                                               \
  do {                                                                       \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error: ") +                \
                               cudaGetErrorString(err));                     \
    }                                                                        \
  } while (0)

struct Nvfp4Plan {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_c = nullptr;
  cublasLtMatrixLayout_t layout_d = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  void* workspace = nullptr;
  size_t workspace_size = 128 * 1024 * 1024;

  Nvfp4Plan(int M, int N, int K) {
    CHECK_CUBLAS_THROW(cublasLtCreate(&handle));
    CHECK_CUBLAS_THROW(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));

    // TN layout mapping for the cuBLASLt NVFP4 matmul:
    // cuBLAS "A" is logical B, cuBLAS "B" is logical A, TN writes row-major D
    // through a column-major NxM view.
    CHECK_CUBLAS_THROW(cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_4F_E2M1, K, N, K));
    CHECK_CUBLAS_THROW(cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_4F_E2M1, K, M, K));
    CHECK_CUBLAS_THROW(cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_16BF, N, M, N));
    CHECK_CUBLAS_THROW(cublasLtMatrixLayoutCreate(&layout_d, CUDA_R_16BF, N, M, N));

    CHECK_CUDA_THROW(cudaMalloc(&workspace, workspace_size));
    CHECK_CUBLAS_THROW(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS_THROW(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size)));

    void* dummy_scale = workspace;
    CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dummy_scale, sizeof(dummy_scale)));
    CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dummy_scale, sizeof(dummy_scale)));

    int returned = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        handle, desc, layout_a, layout_b, layout_c, layout_d, preference,
        1, &heuristic, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0) {
      throw std::runtime_error("cuBLASLt NVFP4 heuristic returned no algorithm");
    }
  }

  ~Nvfp4Plan() {
    if (workspace) cudaFree(workspace);
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (layout_a) cublasLtMatrixLayoutDestroy(layout_a);
    if (layout_b) cublasLtMatrixLayoutDestroy(layout_b);
    if (layout_c) cublasLtMatrixLayoutDestroy(layout_c);
    if (layout_d) cublasLtMatrixLayoutDestroy(layout_d);
    if (desc) cublasLtMatmulDescDestroy(desc);
    if (handle) cublasLtDestroy(handle);
  }
};

static std::mutex g_mu;
static std::unordered_map<std::string, std::unique_ptr<Nvfp4Plan>> g_plans;

static Nvfp4Plan* get_plan(int M, int N, int K) {
  std::lock_guard<std::mutex> lock(g_mu);
  std::string key = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
  auto it = g_plans.find(key);
  if (it == g_plans.end()) {
    it = g_plans.emplace(key, std::make_unique<Nvfp4Plan>(M, N, K)).first;
  }
  return it->second.get();
}

void nvfp4_cublaslt(torch::Tensor A, torch::Tensor B, torch::Tensor A_scale,
                    torch::Tensor B_scale, double alpha, torch::Tensor D,
                    int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A_scale.is_cuda() && B_scale.is_cuda() && D.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kByte && B.scalar_type() == at::kByte,
              "A and B must be uint8 packed FP4 tensors");
  TORCH_CHECK(A_scale.scalar_type() == at::kByte && B_scale.scalar_type() == at::kByte,
              "scale tensors must be uint8 FP8 payloads");
  TORCH_CHECK(D.scalar_type() == at::kBFloat16, "D must be bf16");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && A_scale.is_contiguous() &&
              B_scale.is_contiguous() && D.is_contiguous(), "all tensors must be contiguous");

  Nvfp4Plan* plan = get_plan(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));

  auto* A_ptr = reinterpret_cast<const __nv_fp4x2_e2m1*>(A.data_ptr<uint8_t>());
  auto* B_ptr = reinterpret_cast<const __nv_fp4x2_e2m1*>(B.data_ptr<uint8_t>());
  auto* A_scale_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(A_scale.data_ptr<uint8_t>());
  auto* B_scale_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(B_scale.data_ptr<uint8_t>());
  auto* D_ptr = reinterpret_cast<__nv_bfloat16*>(D.data_ptr<at::BFloat16>());

  const void* cublas_a_scale = B_scale_ptr;
  const void* cublas_b_scale = A_scale_ptr;
  CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
      plan->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &cublas_a_scale, sizeof(cublas_a_scale)));
  CHECK_CUBLAS_THROW(cublasLtMatmulDescSetAttribute(
      plan->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &cublas_b_scale, sizeof(cublas_b_scale)));

  float alpha_f = static_cast<float>(alpha);
  float beta = 0.0f;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  CHECK_CUBLAS_THROW(cublasLtMatmul(
      plan->handle, plan->desc, &alpha_f,
      B_ptr, plan->layout_a,
      A_ptr, plan->layout_b,
      &beta,
      D_ptr, plan->layout_c,
      D_ptr, plan->layout_d,
      &plan->heuristic.algo,
      plan->workspace, plan->workspace_size,
      stream));
}
"""
    extra_include_paths = []
    extra_ldflags = ["-lcublas", "-lcublasLt"]
    if CUDA_HOME:
        extra_include_paths.append(f"{CUDA_HOME}/include")
        extra_ldflags.insert(0, f"-L{CUDA_HOME}/lib64")
    _CUBLASLT_EXT = load_inline(
        name="nvfp4_cublaslt_baseline_ext",
        cpp_sources=[source],
        functions=["nvfp4_cublaslt"],
        with_cuda=True,
        extra_include_paths=extra_include_paths,
        extra_cflags=["-O3"],
        extra_ldflags=extra_ldflags,
        verbose=False,
    )
    return _CUBLASLT_EXT


def _tma_g2c_args(bar_r0, stage, cta_mask, cta_group):
    """Shared kwargs for the A/B and SF TMA g2c loads; only the mbarrier and
    cta_mask vary.

    ``bar_r0`` is the barrier ALREADY mapped to cluster rank 0 (the pair leader)
    once via ``bar.remote_view(0)`` — its ``.buf`` is the rank-0-mapped base, so
    ``bar_r0.ptr_to([stage])`` is a plain ``base + stage*8`` add, NOT a per-call
    ``mapa.shared::cluster`` (the `0x654` PRMT/UPRMT/SEL/IMAD.WIDE remap chain).
    The old form re-mapped ``bar.ptr_to([stage])`` on EVERY g2c copy (4 copies x
    K_TILES x tasks), emitting that `mapa` ALU per copy; hoisting the rank-0 map
    out of the k-loop (mirroring nymph's prologue ``smem_full_cta0_ptr``) sheds
    it — leaving only the cheap stage index add."""
    return {
        "dispatch": "tma",
        "cta_group": cta_group,
        "mbar": T.reinterpret("handle", bar_r0.ptr_to([stage])),
        "cta_mask": cta_mask,
        "cache_hint": "evict_normal",
        "prefetch_tensormap": True,
    }


@T.jit
def _kernel(
    A_packed: T.Buffer((M, K // 2), "uint8"),
    B_packed: T.Buffer((N, K // 2), "uint8"),
    SFA_in: T.Buffer((M, K // 16), "uint8", layout=sf_smem_layout(M, K // 16, sf_per_mma=4)),
    SFB_in: T.Buffer((N, K // 16), "uint8", layout=sf_smem_layout(N, K // 16, sf_per_mma=4)),
    alpha: T.Buffer((1,), "float32"),
    D: T.Buffer((M, N), "bfloat16"),
    *,
    M: T.constexpr,
    N: T.constexpr,
    K: T.constexpr,
    # Fixed hardware + tile/cluster/pipeline choices (tir_ws_kernel never
    # overrides these). Derived quantities are computed from them below.
    SM_COUNT: T.constexpr = 148,
    CTA_GROUP: T.constexpr = 2,
    CLUSTER_M: T.constexpr = 2,
    CLUSTER_N: T.constexpr = 1,
    CTA_M: T.constexpr = 128,
    CTA_N: T.constexpr = 128,
    CTA_K: T.constexpr = 256,
    MMA_K: T.constexpr = 64,
    EPI_TILE: T.constexpr = 64,
    TMEM_LD_SIZE: T.constexpr = 64,
    WB_PIPE_DEPTH: T.constexpr = 2,
    PIPE_DEPTH: T.constexpr = 5,
    TMEM_PIPE_DEPTH: T.constexpr = 1,
    L2_GROUP_SIZE: T.constexpr = 8,
    NUM_WARPS: T.constexpr = 8,
    OVERLAP_EPI: T.constexpr = True,
    # Epilogue reg->smem store datapath:
    #   "stsm" — the tcgen05.ld-atom fragment (`alloc_tcgen05_ldst_frag` "16x256b")
    #            stored with `dispatch="ldstmatrix"` → STSM. The narrow per-store-tile
    #            EPI_TILE-wide tcgen05.ld atoms force a tcgen05.ld per chunk (the more
    #            store tiles, the more tcgen05.ld) plus the STSM datapath's address ALU.
    #   "sts"  — nymph's / fp16_bf16_gemm's datapath: a plain `T.wg_reg_tile` (thread-axis
    #            (128, W) reg tile) drained from TMEM with ONE wide tcgen05.ld per EPI_TILE
    #            band, stored reg->smem with plain `Tx.wg.copy` in 8-bf16 (128b) sub-slices
    #            → STS.128. Sheds the per-chunk STSM/tcgen05.ld address ALU (ncu @16384:
    #            canon's STSM datapath issued +7.7M ALU + 8x the tcgen05.ld vs nymph's STS).
    EPI_STORE: T.constexpr = "stsm",
    # MMA/consumer k-loop structure:
    #   False — the flat `T.serial(K_TILES)` form with a RUNTIME `accum` register
    #           (accum=0 then accum=1 inside the loop), feeding gemm_async an
    #           accumulate flag the tcgen05 MMA reads from a register every issue.
    #   True  — nymph's PEELED software-pipeline form: peel k=0 with a COMPILE-TIME
    #           `accum=False` (cold accumulate), then a `T.serial(K_TILES-1)` steady
    #           state with a COMPILE-TIME `accum=True`. Dropping the runtime accum
    #           variable sheds the per-k-tile `accum=1` store + the register read on
    #           the gemm issue, and lets the peeled prologue (k=0) issue its
    #           cp/cp/gemm before the steady-state loop's per-iteration ring counter
    #           runs — matching nymph's emitted k-loop (its `mma_ktile(0, False)` +
    #           `for ti in serial(63): mma_ktile(ti+1, True)`). Trims the consumer
    #           warp's ALU/CBU pipe pressure (ncu @16384: canon's flat form executes
    #           +5.3M ALU vs nymph at equal instruction count; under the shared ~990W
    #           power cap that extra per-cycle ALU intensity pins canon's sustained
    #           clock ~100 MHz below nymph). Gated per-shape in TIRX_CONFIGS.
    MMA_PEEL: T.constexpr = False,
):
    # Derived shapes (formulas, so they track the params above).
    CLUSTER_SIZE = T.meta_var(CLUSTER_M * CLUSTER_N)
    MMA_N = T.meta_var(CTA_N * CTA_GROUP)
    SFB_N = T.meta_var(MMA_N)
    MMA_K_BLOCKS = T.meta_var(CTA_K // MMA_K)
    SF_CTA_K = T.meta_var(CTA_K // 16)
    NUM_CLUSTERS = T.meta_var(SM_COUNT // CLUSTER_SIZE)
    D_SWIZZLE_MODE = T.meta_var(
        SwizzleMode.SWIZZLE_32B_ATOM
        if EPI_TILE == 16
        else SwizzleMode.SWIZZLE_64B_ATOM
        if EPI_TILE == 32
        else SwizzleMode.SWIZZLE_128B_ATOM
    )
    A_BYTES = T.meta_var(CTA_M * (CTA_K // 2) * CTA_GROUP)
    B_BYTES = T.meta_var(CTA_N * (CTA_K // 2) * CTA_GROUP)
    SFA_BYTES = T.meta_var(CTA_M * SF_CTA_K * CTA_GROUP)
    SFB_BYTES = T.meta_var(SFB_N * SF_CTA_K * CTA_GROUP)
    K_TILES = T.meta_var(K // CTA_K)
    CLUSTER_M_TILES = T.meta_var(M // CTA_M // CLUSTER_M)
    CLUSTER_N_TILES = T.meta_var(N // MMA_N // CLUSTER_N)
    T.device_entry()
    cluster_rank = T.cta_id_in_cluster([CLUSTER_SIZE], preferred=[CLUSTER_SIZE])
    cta_idx = T.cta_id([SM_COUNT])
    tid_in_cta = T.thread_id([NUM_WARPS * 32])
    lane_id = T.lane_id([32])
    tid_in_wg = T.thread_id_in_wg([128])
    wg_id = T.warpgroup_id([NUM_WARPS // 4])
    warp_id = T.warp_id([NUM_WARPS])
    cb_m: T.let = cluster_rank % CLUSTER_M
    cb_n: T.let = cluster_rank // CLUSTER_M
    pair_id: T.let = cluster_rank // CTA_GROUP
    id_in_pair: T.let = cluster_rank % CTA_GROUP
    pair_leader_rank: T.let = pair_id * CTA_GROUP
    tile_scheduler = ClusterPersistentScheduler2D(
        "tile_scheduler",
        num_m_tiles=CLUSTER_M_TILES,
        num_n_tiles=CLUSTER_N_TILES,
        num_clusters=NUM_CLUSTERS,
        l2_group_size=L2_GROUP_SIZE,
    )
    tile_scheduler.init(cta_idx // CLUSTER_SIZE)
    m_idx = T.meta_var(tile_scheduler.m_idx)
    n_idx = T.meta_var(tile_scheduler.n_idx)
    cta_m = T.meta_var(m_idx * CLUSTER_M + cb_m)
    cta_n = T.meta_var(n_idx * CLUSTER_N + cb_n)
    a_m = T.meta_var(cta_m * CTA_M)
    d_m = T.meta_var(cta_m * CTA_M)
    b_n = T.meta_var(cta_n * MMA_N + id_in_pair * CTA_N)
    d_n = T.meta_var(cta_n * MMA_N)
    pool = T.SMEMPool()
    A_smem_packed = pool.alloc_mma((PIPE_DEPTH, CTA_M, CTA_K // 2), "uint8")
    B_smem_packed = pool.alloc_mma((PIPE_DEPTH, CTA_N, CTA_K // 2), "uint8")
    SFA_smem = pool.alloc(
        (PIPE_DEPTH, CTA_M, SF_CTA_K),
        "uint8",
        layout=sf_smem_layout(128, 16, sf_per_mma=4, pipe_depth=PIPE_DEPTH),
        align=1024,
    )
    SFB_smem = pool.alloc(
        (PIPE_DEPTH, SFB_N, SF_CTA_K),
        "uint8",
        layout=sf_smem_layout(SFB_N, 16, sf_per_mma=4, pipe_depth=PIPE_DEPTH),
        align=1024,
    )
    output_smem = pool.alloc_mma(
        (WB_PIPE_DEPTH, CTA_M, EPI_TILE), "bfloat16", swizzle_mode=D_SWIZZLE_MODE
    )
    tmem_addr = pool.alloc([1], "uint32", align=4)
    mbar_leader = tid_in_cta == 32
    smem_pipe = Pipeline(pool, PIPE_DEPTH, full="tma", empty="tcgen05", leader=mbar_leader)
    tile_full_bar = TMABar(pool, PIPE_DEPTH, leader=mbar_leader)
    tile_full_bar.init(1)
    scale_full_bar = TMABar(pool, PIPE_DEPTH, leader=mbar_leader)
    scale_full_bar.init(1)
    # Hoist the cluster-rank-0 (pair-leader) barrier mapping out of the per-k-tile
    # g2c loads. `remote_view(0)` computes `map_shared_rank(buf.ptr_to([0]), 0)`
    # ONCE; the TMA copies then index `[stage]` (a plain add) instead of re-issuing
    # the `mapa.shared::cluster` remap (`0x654` PRMT/UPRMT/SEL + IMAD.WIDE) on every
    # copy. Mirrors nymph's prologue `smem_full_cta0_ptr` (ncu @16384: the per-copy
    # mapa is ~1.05M each of SEL/UPRMT/PRMT/IMAD.WIDE in the producer warps).
    tile_full_bar_r0 = tile_full_bar.remote_view(0)
    scale_full_bar_r0 = scale_full_bar.remote_view(0)
    tmem_pipe = Pipeline(
        pool,
        TMEM_PIPE_DEPTH,
        full="tcgen05",
        empty="mbar",
        init_empty=CTA_GROUP,
        leader=mbar_leader,
    )
    tmem_finished = MBarrier(pool, 1, leader=mbar_leader)
    tmem_finished.init(1)
    pool.commit()
    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=CTA_GROUP, tmem_addr=tmem_addr)
    tmem = tmem_pool.alloc((CTA_M, 512), "float32")
    A_smem = A_smem_packed.view("float4_e2m1fn")
    B_smem = B_smem_packed.view("float4_e2m1fn")
    sf_mma_k = T.meta_var(4)
    SFB_n_chunks = T.meta_var(SFB_N // 128)
    tmem_pool.move_base_to(448)
    SFA_tmem = tmem_pool.alloc_sf(
        (128, sf_mma_k * MMA_K_BLOCKS), "float8_e4m3fn", sf_per_mma=sf_mma_k
    )
    tmem_pool.move_base_to(464)
    SFB_tmem = tmem_pool.alloc_sf(
        (128 * SFB_n_chunks, sf_mma_k * MMA_K_BLOCKS), "float8_e4m3fn", sf_per_mma=sf_mma_k
    )
    T.ptx.barrier.cluster.arrive(sem="release", aligned=True)
    T.ptx.barrier.cluster.wait(acquire=True, aligned=False)
    # Alloc TMEM after the cluster sync, warp-0-only, before the role split, so
    # the TMA warp overlaps its first loads with the alloc.
    tmem_pool.commit()
    if tid_in_cta < 32:
        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
    pair_mask: T.int32
    pair_mask = 0
    pair_mask = pair_mask | 1 << pair_leader_rank
    pair_mask = pair_mask | 1 << pair_leader_rank + 1
    tma_cur = PipelineState(PIPE_DEPTH, 1)
    mma_smem = PipelineState(PIPE_DEPTH, 0)
    mma_tmem = PipelineState(TMEM_PIPE_DEPTH, 1)
    accum: T.int32
    accum = 0
    epi_cur = PipelineState(TMEM_PIPE_DEPTH, 0)
    epi_wb_state = PipelineState(WB_PIPE_DEPTH, 1)
    alpha_local: T.float32
    alpha_local = alpha[0]
    if warp_id == int(WarpRole.TMA):

        @T.inline
        def issue_tma_load(k_tile: T.int32):
            stage = tma_cur.stage
            k = T.meta_var(k_tile * CTA_K // 2)
            smem_pipe.empty.wait(tma_cur.stage, tma_cur.phase)
            if id_in_pair == 0:
                tile_bytes = T.meta_var(A_BYTES + B_BYTES)
                # id_in_pair == 0 ⇒ the executing CTA *is* pair_leader_rank (cluster
                # rank 0), so this arrives on its OWN barrier. Use the LOCAL
                # `mbarrier.arrive.expect_tx.shared::cta` form (no cta_id) instead of
                # the `.shared::cluster` + `mapa` self-map — the cross-CTA op emitted a
                # per-k-tile mapa (the `0x654` PRMT/SEL/IMAD.WIDE remap) to map rank 0
                # to itself. The local form drops that address ALU entirely.
                T.ptx.mbarrier.arrive.expect_tx(tile_full_bar.ptr_to([stage]), tile_bytes)
            single_cta_mask: T.int32 = 1 << id_in_pair
            # Barrier pre-mapped to the cluster leader (the g2c primitive maps
            # neither the barrier nor expect_tx — both handled above).
            tile_copy = T.meta_var(_tma_g2c_args(tile_full_bar_r0, stage, single_cta_mask, CTA_GROUP))
            Tx.copy_async(
                A_smem_packed[stage, 0:CTA_M, 0 : CTA_K // 2],
                A_packed[a_m : a_m + CTA_M, k : k + CTA_K // 2],
                **tile_copy,
            )
            Tx.copy_async(
                B_smem_packed[stage, 0:CTA_N, 0 : CTA_K // 2],
                B_packed[b_n : b_n + CTA_N, k : k + CTA_K // 2],
                **tile_copy,
            )

        if T.ptx.elect_sync():
            while tile_scheduler.valid():
                for k_tile in T.serial(K_TILES):
                    issue_tma_load(k_tile)
                    tma_cur.advance()
                tile_scheduler.next_tile()
    elif warp_id == int(WarpRole.TMA) + 1:

        @T.inline
        def issue_scale_tma_load(k_tile: T.int32):
            stage = tma_cur.stage
            sf_k = T.meta_var(k_tile * SF_CTA_K)
            sf_m = T.meta_var((a_m // 128) * 128)
            sf_n = T.meta_var((d_n // 128) * 128)
            smem_pipe.empty.wait(tma_cur.stage, tma_cur.phase)
            if id_in_pair == 0:
                scale_bytes = T.meta_var(SFA_BYTES + SFB_BYTES)
                # Self-arrive (id_in_pair == 0 ⇒ this CTA is pair_leader_rank); local
                # `shared::cta` form drops the per-k-tile self-`mapa` (see issue_tma_load).
                T.ptx.mbarrier.arrive.expect_tx(scale_full_bar.ptr_to([stage]), scale_bytes)
            single_cta_mask: T.int32 = 1 << id_in_pair
            # SFA: each CTA loads its half (single_cta_mask). SFB: multicast to
            # both CTAs (pair_mask).
            sfa_copy = T.meta_var(_tma_g2c_args(scale_full_bar_r0, stage, single_cta_mask, CTA_GROUP))
            Tx.copy_async(
                SFA_smem[stage, 0:CTA_M, 0:SF_CTA_K],
                SFA_in[sf_m : sf_m + CTA_M, sf_k : sf_k + SF_CTA_K],
                **sfa_copy,
            )
            sfb_copy = T.meta_var(_tma_g2c_args(scale_full_bar_r0, stage, pair_mask, CTA_GROUP))
            if SFB_N == 128:
                if id_in_pair == 0:
                    Tx.copy_async(
                        SFB_smem[stage, 0:SFB_N, 0:SF_CTA_K],
                        SFB_in[sf_n : sf_n + SFB_N, sf_k : sf_k + SF_CTA_K],
                        **sfb_copy,
                    )
            else:
                Tx.copy_async(
                    SFB_smem[stage, cb_m * 128 : cb_m * 128 + 128, 0:SF_CTA_K],
                    SFB_in[sf_n + cb_m * 128 : sf_n + cb_m * 128 + 128, sf_k : sf_k + SF_CTA_K],
                    **sfb_copy,
                )

        if T.ptx.elect_sync():
            while tile_scheduler.valid():
                for k_tile in T.serial(K_TILES):
                    issue_scale_tma_load(k_tile)
                    tma_cur.advance()
                tile_scheduler.next_tile()
    elif (warp_id == int(WarpRole.MMA)) & (id_in_pair == 0):

        @T.inline
        def execute_mma():
            stage = mma_smem.stage
            scale_full_bar.wait(mma_smem.stage, mma_smem.phase)
            tile_full_bar.wait(mma_smem.stage, mma_smem.phase)
            Tx.copy_async(SFA_tmem, SFA_smem[stage], cta_group=CTA_GROUP)
            Tx.copy_async(SFB_tmem, SFB_smem[stage], cta_group=CTA_GROUP)
            Tx.gemm_async(
                tmem[:, 0:MMA_N],
                A_smem[stage],
                B_smem[stage],
                SFA=SFA_tmem,
                SFB=SFB_tmem,
                accum=accum,
                dispatch="tcgen05",
                cta_group=CTA_GROUP,
            )
            accum = 1
            smem_pipe.empty.arrive(mma_smem.stage, cta_group=CTA_GROUP, cta_mask=pair_mask)

        # Peeled-pipeline variant of execute_mma: the accumulate flag is a
        # COMPILE-TIME constant (`accum_flag`), so the gemm issues with a literal
        # accum and there is no runtime `accum` register / per-iteration `accum=1`
        # store. k=0 issues with accum_flag=False (cold), k>=1 with accum_flag=True.
        @T.inline
        def execute_mma_peeled(accum_flag: T.constexpr):
            stage = mma_smem.stage
            scale_full_bar.wait(mma_smem.stage, mma_smem.phase)
            tile_full_bar.wait(mma_smem.stage, mma_smem.phase)
            Tx.copy_async(SFA_tmem, SFA_smem[stage], cta_group=CTA_GROUP)
            Tx.copy_async(SFB_tmem, SFB_smem[stage], cta_group=CTA_GROUP)
            Tx.gemm_async(
                tmem[:, 0:MMA_N],
                A_smem[stage],
                B_smem[stage],
                SFA=SFA_tmem,
                SFB=SFB_tmem,
                accum=accum_flag,
                dispatch="tcgen05",
                cta_group=CTA_GROUP,
            )
            smem_pipe.empty.arrive(mma_smem.stage, cta_group=CTA_GROUP, cta_mask=pair_mask)

        if T.ptx.elect_sync():
            if MMA_PEEL:
                # nymph's peeled software pipeline: peel k=0 (cold accum), then a
                # T.serial(K_TILES-1) steady state with a compile-time hot accum.
                while tile_scheduler.valid():
                    tmem_pipe.empty.wait(mma_tmem.stage, mma_tmem.phase)
                    execute_mma_peeled(False)
                    mma_smem.advance()
                    for k_tile in T.serial(K_TILES - 1):
                        execute_mma_peeled(True)
                        mma_smem.advance()
                    tmem_pipe.full.arrive(mma_tmem.stage, cta_group=CTA_GROUP, cta_mask=pair_mask)
                    mma_tmem.advance()
                    tile_scheduler.next_tile()
            else:
                while tile_scheduler.valid():
                    tmem_pipe.empty.wait(mma_tmem.stage, mma_tmem.phase)
                    accum = 0
                    for k_tile in T.serial(K_TILES):
                        execute_mma()
                        mma_smem.advance()
                    tmem_pipe.full.arrive(mma_tmem.stage, cta_group=CTA_GROUP, cta_mask=pair_mask)
                    mma_tmem.advance()
                    tile_scheduler.next_tile()
    elif warp_id >= int(WarpRole.EPILOGUE):

        @T.inline
        def regs_to_smem(reg_ldst_16b):
            if EPI_STORE == "sts":
                # nymph / fp16_bf16_gemm datapath: plain `Tx.wg.copy` of the thread-axis
                # reg tile in 8-bf16 (128b) sub-slices → STS.128 (one swizzle chunk each),
                # no ldstmatrix dispatch. Sheds the STSM address-ALU overhead.
                for cj in T.unroll(EPI_TILE // 8):
                    cc = T.meta_var(cj * 8)
                    Tx.wg.copy(
                        output_smem[epi_wb_state.stage, 0:CTA_M, cc : cc + 8],
                        reg_ldst_16b[:, cc : cc + 8],
                    )
            else:
                # R->S in 16-col chunks to match stmatrix.x4 granularity (one wide
                # copy schedules worse).
                for cj in T.unroll(EPI_TILE // 16):
                    cc = T.meta_var(cj * 16)
                    Tx.wg.copy(
                        output_smem[epi_wb_state.stage, 0:CTA_M, cc : cc + 16],
                        reg_ldst_16b[:, cc : cc + 16],
                        dispatch="ldstmatrix",
                    )

        @T.inline
        def epilogue():
            tmem_pipe.full.wait(epi_cur.stage, epi_cur.phase)

            # Per-chunk store: R->S (stmatrix) then S->G (TMA). Shared by both schedules.
            @T.inline
            def store_epi_chunk(reg_ldst_16b, linear_n: T.constexpr):
                T.ptx.cp_async.bulk.wait_group(WB_PIPE_DEPTH - 1, read=True)
                T.cuda.warpgroup_sync(1)
                regs_to_smem(reg_ldst_16b)
                T.cuda.warpgroup_sync(1)
                d_n_out: T.int32
                d_n_out = d_n + linear_n
                if tid_in_wg == 0:
                    T.ptx.fence.proxy_async("shared::cta")
                    Tx.copy_async(
                        D[d_m : d_m + CTA_M, d_n_out : d_n_out + EPI_TILE],
                        output_smem[epi_wb_state.stage, 0:CTA_M, 0:EPI_TILE],
                        dispatch="tma",
                        cache_hint="evict_first",
                        prefetch_tensormap=True,
                    )
                    T.ptx.cp_async.bulk.commit_group()
                epi_wb_state.advance()

            # Fusion vs fission of {load; scale+cast; store}: overlap fuses and reuses
            # a small (128, EPI_TILE) frag; non-overlap splits the loops, needing a big
            # (128, MMA_N) frag (all chunks live between load and store).
            if OVERLAP_EPI and EPI_STORE == "sts":
                # nymph's overlap datapath (its default at 16384): per EPI_TILE band, ONE
                # wide tcgen05.ld into a plain (128, EPI_TILE) thread-axis reg tile, ONE
                # wait.ld, scale+cast, then plain-STS store. The wide drain keeps the
                # tcgen05.ld count at MMA_N/EPI_TILE (4 @ EPI_TILE=64) instead of the STSM
                # frag's per-chunk count, and the store is STS not STSM — shedding canon's
                # +7.7M ALU / 8x tcgen05.ld epilogue overhead (the residual @16384).
                reg_ldst = T.wg_reg_tile(EPI_TILE, dtype="float32")
                reg_ldst_16b = T.wg_reg_tile(EPI_TILE, dtype="bfloat16")
                for no in T.unroll(MMA_N // EPI_TILE):
                    linear_n = T.meta_var(no * EPI_TILE)
                    Tx.wg.copy_async(reg_ldst[:, :], tmem[:, linear_n : linear_n + EPI_TILE])
                    T.ptx.tcgen05.wait.ld()
                    Tx.wg.mul(reg_ldst, reg_ldst, alpha_local)
                    Tx.wg.cast(reg_ldst_16b, reg_ldst)
                    if no == MMA_N // EPI_TILE - 1 and tid_in_wg == 0:
                        tmem_pipe.empty.arrive(
                            epi_cur.stage, cta_id=pair_leader_rank, pred=True, count=1
                        )
                    store_epi_chunk(reg_ldst_16b, linear_n)
            elif OVERLAP_EPI:
                reg_ldst = T.alloc_tcgen05_ldst_frag("16x256b", (128, EPI_TILE), "float32")
                reg_ldst_16b = T.alloc_cast_frag(reg_ldst, "bfloat16")
                for no in T.unroll(MMA_N // EPI_TILE):
                    linear_n = T.meta_var(no * EPI_TILE)
                    Tx.wg.copy_async(reg_ldst[:, :], tmem[:, linear_n : linear_n + EPI_TILE])
                    if no == MMA_N // EPI_TILE - 1:
                        T.ptx.tcgen05.wait.ld()
                        if tid_in_wg == 0:
                            tmem_pipe.empty.arrive(
                                epi_cur.stage, cta_id=pair_leader_rank, pred=True, count=1
                            )
                    Tx.wg.mul(reg_ldst, reg_ldst, alpha_local)
                    Tx.wg.cast(reg_ldst_16b, reg_ldst)
                    store_epi_chunk(reg_ldst_16b, linear_n)
            else:
                # Keep the 2D frag so it can be column-sliced for the chunked store.
                reg_all = T.alloc_tcgen05_ldst_frag("16x256b", (128, MMA_N), "float32")
                reg_all_16b = T.alloc_cast_frag(reg_all, "bfloat16")
                for no in T.unroll(MMA_N // EPI_TILE):
                    ln = T.meta_var(no * EPI_TILE)
                    Tx.wg.copy_async(reg_all[:, ln : ln + EPI_TILE], tmem[:, ln : ln + EPI_TILE])
                T.ptx.tcgen05.wait.ld()
                # scale + cast the whole frag
                Tx.wg.mul(reg_all, reg_all, alpha_local)
                Tx.wg.cast(reg_all_16b, reg_all)
                if tid_in_wg == 0:
                    tmem_pipe.empty.arrive(
                        epi_cur.stage, cta_id=pair_leader_rank, pred=True, count=1
                    )
                T.cuda.warpgroup_sync(1)
                for no in T.unroll(MMA_N // EPI_TILE):
                    ln = T.meta_var(no * EPI_TILE)
                    store_epi_chunk(reg_all_16b[:, ln : ln + EPI_TILE], ln)

        while tile_scheduler.valid():
            epilogue()
            epi_cur.advance()
            tile_scheduler.next_tile()
        if tid_in_wg == 0:
            T.ptx.cp_async.bulk.wait_group(0, read=True)
        T.cuda.warpgroup_sync(1)
    if warp_id == int(WarpRole.EPILOGUE):
        if T.ptx.elect_sync():
            T.ptx.mbarrier.arrive.cluster_count(
                tmem_finished.ptr_to([0]), pair_leader_rank + 1 - id_in_pair, 1
            )
        T.ptx.mbarrier.try_wait_acquire_cluster(tmem_finished.ptr_to([0]), 0)
        T.ptx.tcgen05.dealloc(tmem_pool.addr, n_cols=512, cta_group=CTA_GROUP)


def tir_ws_kernel(M: int, N: int, K: int):
    assert M % 128 == 0 and N % 256 == 0 and K % 256 == 0
    assert (M // 128) % 2 == 0
    assert (K // 16) % 4 == 0
    config = dict(TIRX_CONFIGS.get((M, N, K), {}))
    return _kernel.specialize(M=M, N=N, K=K, **config)


TIRX_CONFIGS = {
    # Per-shape launch/pipeline tuning. The cluster N tile spans CTA_GROUP CTAs,
    # so CTA_N = (cluster N tile) / CTA_GROUP.
    (1024, 1024, 1024): {
        "SM_COUNT": 64,
        "CTA_N": 64,
        "EPI_TILE": 32,
        "PIPE_DEPTH": 5,
        "L2_GROUP_SIZE": 12,
        "OVERLAP_EPI": True,
    },
    (2048, 2048, 2048): {
        "SM_COUNT": 128,
        "CTA_N": 128,
        "EPI_TILE": 32,
        "PIPE_DEPTH": 5,
        "L2_GROUP_SIZE": 4,
        "OVERLAP_EPI": True,
    },
    (4096, 4096, 4096): {
        "SM_COUNT": 148,
        "CTA_N": 128,
        "EPI_TILE": 32,
        "PIPE_DEPTH": 5,
        "L2_GROUP_SIZE": 4,
        "OVERLAP_EPI": False,
    },
    (8192, 8192, 8192): {
        "SM_COUNT": 148,
        "CTA_N": 128,
        "EPI_TILE": 16,
        "PIPE_DEPTH": 4,
        "L2_GROUP_SIZE": 1,
        "OVERLAP_EPI": False,
    },
    (16384, 16384, 16384): {
        "SM_COUNT": 148,
        "CTA_N": 128,
        # Epilogue datapath @16384 — INVESTIGATED, the no_overlap/STSM/EPI_TILE=16 form below
        # is the measured best; the nymph-style sts/overlap form was built and ruled out:
        #
        # Fresh ncu (clean B200, fixed clocks) showed canon is ALREADY structurally faster
        # than nymph here: 1.95M vs 2.00M cycles, tensor pipe 96.0% vs 93.3%, SM throughput
        # 97.9% vs 94.0%. The wall-clock residual (canon/nymph ~1.06) is therefore NOT a
        # canon-internal stall; the two kernels' instruction counts are within 1% (canon
        # 86.5M vs nymph 85.6M) and at equal clock canon wins. The gap is the SUSTAINED
        # BOOST CLOCK during the free-running bench (canon ~1850 MHz vs nymph ~1900+,
        # measured per-kernel via nvidia-smi under sustained load), which the bench captures
        # and ncu (locked clocks) does not — hence the ±0.05 run-to-run ratio noise.
        #
        # The prompt's hypothesis was that canon's epilogue (1.05M tcgen05.ld = 8x nymph's
        # 131K; 1.05M STSM vs nymph's plain STS) steals MMA issue slots. That was tested:
        # the EPI_STORE="sts" + OVERLAP_EPI=True + EPI_TILE=64 path below reproduces nymph's
        # epilogue datapath EXACTLY (ncu after: tcgen05.ld 131K, STSM 0, STS 1.05M — all
        # byte-identical to nymph) and draws LESS power (583-686 W vs the STSM form's
        # 684-707 W). But it does NOT improve wall-clock: a tightly-paired A/B (NEW vs this
        # OLD form timed back-to-back in the SAME boost-clock window, so clock cancels) gave
        # NEW/OLD median 1.007 — i.e. the STSM form is marginally FASTER at equal clock, and
        # NEW won only 8/20 reps. So the epilogue datapath is a wall-clock wash; the
        # tcgen05.ld/STSM counts are off the critical path here (canon's tensor pipe is the
        # wall and it is already higher than nymph's). The sts overlap path is kept as the
        # EPI_STORE="sts" capability (matches fp16_bf16_gemm) but is not selected at 16384.
        # A clock-controlled paired sweep of the remaining levers (PIPE_DEPTH 4, L2_GROUP
        # 8/32, TMEM_PIPE_DEPTH 2) all tied-or-regressed vs this config, confirming it is the
        # measured optimum reachable from this kernel's construct space.
        "EPI_TILE": 16,
        "EPI_STORE": "stsm",
        "OVERLAP_EPI": False,
        # PIPE_DEPTH 4 -> 5: deeper A/B/SF SMEM ring. At 16384 the kernel is tensor-bound
        # but the MMA periodically stalls waiting on the next k-tile's TMA loads — a 5th
        # ring stage hides that load latency so the tcgen05 pipe stops bubbling. ncu: the
        # tensor pipe active fraction rises 91.9% -> 93.5% on the profiled launch. EPI_TILE=16
        # keeps the D store ring small enough that the extra A/B/SF stage still fits SMEM.
        "PIPE_DEPTH": 5,
        # L2_GROUP_SIZE 12 -> 16: the persistent scheduler walks each cluster's consecutive
        # tasks down an L2_GROUP_SIZE-row band of the cluster-tile grid. At 16384 that grid is
        # CLUSTER_M_TILES = M/CTA_M/CLUSTER_M = 64 rows; 64 % 12 = 4 leaves a 4-row tail group,
        # so the scheduler takes the _gm_emit_full_and_tail path (an extra runtime tail branch +
        # its own row/col index arithmetic every next_tile). 64 % 16 == 0 makes FULL_GROUPS=4 /
        # TAIL_ROWS=0, so it takes the leaner _gm_emit_full_only path (no tail branch) AND the
        # 16-row band is the wider A-row/B-col working set that fits L2 best here (sweep peak:
        # L2=12 -> ~5.40 PFLOP/s, L2=16 -> ~5.50 PFLOP/s on a clean B200). nymph's default
        # l2_group_size is 16 (TILE_GROUPS_ROW_SIZE) for this shape; this matches it.
        "L2_GROUP_SIZE": 16,
        # Peel the MMA k-loop into nymph's software-pipeline form (compile-time accum,
        # k=0 peeled). Sheds the runtime `accum` register + per-k-tile ALU on the
        # consumer warp; see MMA_PEEL doc above. Scoped to 16384 (the shape where the
        # consumer warp's ALU pressure pins the sustained clock below nymph).
        "MMA_PEEL": True,
    },
}


KERNEL_META = {"name": "nvfp4_gemm", "category": "gemm", "compute_capability": 10}
CONFIGS = [
    {"M": s, "N": s, "K": s, "label": f"{s}x{s}x{s}"} for s in [1024, 2048, 4096, 8192, 16384]
]


def get_kernel(M, N, K):
    return tir_ws_kernel(M, N, K)


def run_test(M=1024, N=1024, K=1024):
    """Compile, run, and verify kernel."""
    import torch
    import torch.nn.functional as F

    kernel = tir_ws_kernel(M, N, K)
    A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref = prepare_data(M, N, K)
    alpha_tensor = torch.tensor([alpha], device="cuda", dtype=torch.float)
    out = torch.empty_like(C_ref).to("cuda").to(torch.bfloat16)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        ex.mod(A_fp4, B_fp4, A_sf, B_sf, alpha_tensor, out)
    cosine_sim = F.cosine_similarity(
        out.reshape(-1).float(), C_ref.to("cuda").reshape(-1).float(), dim=0
    )
    assert cosine_sim > 0.97, f"nvfp4_gemm cosine_sim {cosine_sim:.6f} <= 0.97"


# timer=None inherits the global default (proton). Proton matters here: the
# flashinfer/cublaslt references carry heavy per-call host dispatch (Python + internal
# cudaDeviceSynchronize), and since the nvfp4 kernel (~28µs) is faster than that dispatch,
# event wall-clock is host-starved and over-credits us ~4x. Proton measures pure GPU
# kernel time -> honest ~parity (verified 0.996 vs event 4.11).
def run_bench(M=1024, N=1024, K=1024, *, warmup=None, repeat=None, timer=None, **kwargs):
    """Benchmark."""
    import torch

    metadata = {}
    kernel = tir_ws_kernel(M, N, K)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref = prepare_data(M, N, K)
    alpha_value = float(alpha.item())
    alpha_tensor = torch.tensor([alpha_value], device="cuda", dtype=torch.float)
    out_tir = torch.empty_like(C_ref).to("cuda").to(torch.bfloat16)

    funcs = {"tir": lambda: ex.mod(A_fp4, B_fp4, A_sf, B_sf, alpha_tensor, out_tir)}

    def _flashinfer():
        out_fi = torch.empty_like(out_tir)

        def run():
            with flashinfer.autotune(False):  # time with the tuned config, no re-tune
                return flashinfer.mm_fp4(
                    A_fp4,
                    B_fp4.T,
                    A_sf,
                    B_sf.T,
                    alpha,
                    out=out_fi,
                    block_size=16,
                    backend="auto",
                    use_nvfp4=True,
                )

        # Autotune once so the timed runs use the tuned config.
        with flashinfer.autotune(True):
            run()
        torch.cuda.synchronize()
        return run

    def _cublaslt():
        ext = _load_cublaslt_nvfp4_ext()
        out_cublaslt = torch.empty_like(out_tir)
        return lambda: ext.nvfp4_cublaslt(
            A_fp4, B_fp4, A_sf, B_sf, alpha_value, out_cublaslt, M, N, K
        )

    result = bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashinfer": _flashinfer, "cublaslt_nvfp4": _cublaslt},
        **kwargs,
    )
    result["metadata"] = {**result.get("metadata", {}), **metadata}
    return result
