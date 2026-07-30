from __future__ import annotations

import hashlib
import json
import os
from enum import IntEnum
from pathlib import Path

import flashinfer
import torch
from flashinfer import SfLayout, nvfp4_quantize

import tvm
from tvm.backend.cuda.op import cuda_func_call
from tvm.backend.cuda.operator.tile_primitive.gemm_async.tcgen05 import sf_smem_layout
from tvm.backend.cuda.operator.tile_primitive.tma_utils import SwizzleMode
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.bench import bench
from tvm.tirx.lang.pipeline import MBarrier, PipelineState, TCGen05Bar, TMABar
from tvm.tirx.lang.tile_scheduler import ClusterLaunchControlScheduler, ClusterPersistentScheduler2D

_NVFP4_COPY_SCALE_AND_MMA_K256_2CTA_SRC = r"""
__forceinline__ __device__ void tirx_nvfp4_copy_scale_and_mma_k256_2cta(
    uint32_t d_tmem_addr,
    void* a_smem,
    void* b_smem,
    void* sfa_smem,
    void* sfb_smem,
    uint32_t accum) {
    constexpr uint64_t mma_desc_hi = uint64_t{0x40004040} << 32;
    constexpr uint64_t cp_desc_hi = uint64_t{0x00004008} << 32;
    constexpr uint32_t instr_desc = 0x10400480;
    uint64_t a_desc =
        mma_desc_hi | ((__cvta_generic_to_shared(a_smem) >> 4) & uint32_t{0x3fff});
    uint64_t b_desc =
        mma_desc_hi | ((__cvta_generic_to_shared(b_smem) >> 4) & uint32_t{0x3fff});
    uint64_t sfa_desc =
        cp_desc_hi | ((__cvta_generic_to_shared(sfa_smem) >> 4) & uint32_t{0x3fff});
    uint64_t sfb_desc =
        cp_desc_hi | ((__cvta_generic_to_shared(sfb_smem) >> 4) & uint32_t{0x3fff});
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        ".reg .b64 ad, bd, sad, sbd;\n"
        ".reg .b32 sfat, sfbt;\n"
        "mov.b64 ad, %1;\n"
        "mov.b64 bd, %2;\n"
        "mov.b64 sad, %3;\n"
        "mov.b64 sbd, %4;\n"
        "mov.b32 sfat, 448;\n"
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [sfat], sad;\n"
        "add.u32 sfat, sfat, 4;\n"
        "add.u64 sad, sad, 32;\n"
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [sfat], sad;\n"
        "add.u32 sfat, sfat, 4;\n"
        "add.u64 sad, sad, 32;\n"
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [sfat], sad;\n"
        "add.u32 sfat, sfat, 4;\n"
        "add.u64 sad, sad, 32;\n"
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [sfat], sad;\n"
        "mov.b32 sfbt, 464;\n"
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [sfbt], sbd;\n"
        "add.u32 sfbt, sfbt, 8;\n"
        "add.u64 sbd, sbd, 32;\n"
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [sfbt], sbd;\n"
        "add.u32 sfbt, sfbt, 8;\n"
        "add.u64 sbd, sbd, 32;\n"
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [sfbt], sbd;\n"
        "add.u32 sfbt, sfbt, 8;\n"
        "add.u64 sbd, sbd, 32;\n"
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [sfbt], sbd;\n"
        "add.u64 sbd, sbd, 32;\n"
        "mov.b32 sfbt, 468;\n"
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [sfbt], sbd;\n"
        "add.u32 sfbt, sfbt, 8;\n"
        "add.u64 sbd, sbd, 32;\n"
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [sfbt], sbd;\n"
        "add.u32 sfbt, sfbt, 8;\n"
        "add.u64 sbd, sbd, 32;\n"
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [sfbt], sbd;\n"
        "add.u32 sfbt, sfbt, 8;\n"
        "add.u64 sbd, sbd, 32;\n"
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [sfbt], sbd;\n"
        "mov.b32 sfat, 448;\n"
        "mov.b32 sfbt, 464;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%0], ad, bd, %5, [sfat], [sfbt], p;\n"
        "add.u64 ad, ad, 2;\n"
        "add.u64 bd, bd, 2;\n"
        "add.u32 sfat, sfat, 4;\n"
        "add.u32 sfbt, sfbt, 8;\n"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%0], ad, bd, %5, [sfat], [sfbt], 1;\n"
        "add.u64 ad, ad, 2;\n"
        "add.u64 bd, bd, 2;\n"
        "add.u32 sfat, sfat, 4;\n"
        "add.u32 sfbt, sfbt, 8;\n"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%0], ad, bd, %5, [sfat], [sfbt], 1;\n"
        "add.u64 ad, ad, 2;\n"
        "add.u64 bd, bd, 2;\n"
        "add.u32 sfat, sfat, 4;\n"
        "add.u32 sfbt, sfbt, 8;\n"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%0], ad, bd, %5, [sfat], [sfbt], 1;\n"
        "}\n"
        :
        : "r"(d_tmem_addr),
          "l"(a_desc),
          "l"(b_desc),
          "l"(sfa_desc),
          "l"(sfb_desc),
          "r"(instr_desc),
          "r"(accum));
}
"""


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


def _tma_g2s_args(bar, stage, cta_mask, cta_group, cache_hint):
    """Shared kwargs for the A/B and SF TMA g2s loads."""
    args = {
        "dispatch": "tma_auto",
        "cta_group": cta_group,
        "mbar": T.reinterpret("handle", T.ptx.map_shared_rank(bar.ptr_to([stage]), 0)),
        "cta_mask": cta_mask,
        "prefetch_tensormap": True,
    }
    if cache_hint is not None:
        args["cache_hint"] = cache_hint
    return args


@T.jit
def _kernel(
    A_packed: T.Buffer((M, K // 2), "uint8"),
    B_packed: T.Buffer((N, K // 2), "uint8"),
    SFA_in: T.Buffer((M, K // 16), "uint8", layout=sf_smem_layout(M, K // 16, sf_per_mma=4)),
    SFB_in: T.Buffer((N, K // 16), "uint8", layout=sf_smem_layout(N, K // 16, sf_per_mma=4)),
    alpha: T.float32,
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
    CLC_SCHEDULER: T.constexpr = False,
    NUM_WARPS: T.constexpr = 8,
    OVERLAP_EPI: T.constexpr = True,
    TMA_SPLIT: T.constexpr = 1,
    LOAD_CACHE_HINT: T.constexpr = "evict_normal",
    MERGE_TMA_BARRIERS: T.constexpr = False,
    EARLY_TMEM_TEARDOWN: T.constexpr = False,
    DIRECT_EPI: T.constexpr = False,
    FUSED_SCALE_MMA: T.constexpr = False,
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
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    cluster_rank = T.cta_id_in_cluster([CLUSTER_SIZE], preferred=[CLUSTER_SIZE])
    cta_idx = T.cta_id(
        [CLUSTER_M_TILES * CLUSTER_N_TILES * CLUSTER_SIZE if CLC_SCHEDULER else SM_COUNT]
    )
    tid_in_cta = T.thread_id([NUM_WARPS * 32])
    lane_id = T.lane_id([32])
    tid_in_wg = T.thread_id_in_wg([128])
    wg_id = T.warpgroup_id([NUM_WARPS // 4])
    warp_id = T.warp_id([NUM_WARPS])
    warp_id_in_wg = T.warp_id_in_wg([4])
    cb_m: T.let = cluster_rank % CLUSTER_M
    cb_n: T.let = cluster_rank // CLUSTER_M
    pair_id: T.let = cluster_rank // CTA_GROUP
    id_in_pair: T.let = cluster_rank % CTA_GROUP
    pair_leader_rank: T.let = pair_id * CTA_GROUP
    pool = T.SMEMPool()
    if CLC_SCHEDULER:
        clc_sched = ClusterLaunchControlScheduler(
            pool,
            num_m_tiles=CLUSTER_M_TILES,
            num_n_tiles=CLUSTER_N_TILES,
            l2_group_size=L2_GROUP_SIZE,
            cta_group=CTA_GROUP,
            # scheduler (2 CTAs), A/B loaders (2), SF loaders (2),
            # leader-CTA MMA (1), and epilogue warpgroups (2)
            finish_arrivals=9,
        )
        tile_scheduler = clc_sched.worker("tile_scheduler")
    else:
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
    A_smem_packed = pool.alloc_tcgen05_mma_AB((PIPE_DEPTH, CTA_M, CTA_K // 2), "uint8")
    B_smem_packed = pool.alloc_tcgen05_mma_AB((PIPE_DEPTH, CTA_N, CTA_K // 2), "uint8")
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
    output_smem = pool.alloc_tcgen05_mma_AB(
        (WB_PIPE_DEPTH, CTA_M, EPI_TILE), "bfloat16", swizzle_mode=D_SWIZZLE_MODE
    )
    tmem_addr = pool.alloc([1], "uint32", align=4)
    mbar_leader = tid_in_cta == 32
    smem_full = TMABar(pool, PIPE_DEPTH, leader=mbar_leader)
    smem_empty = TCGen05Bar(pool, PIPE_DEPTH, leader=mbar_leader)
    if not MERGE_TMA_BARRIERS:
        tile_full_bar = TMABar(pool, PIPE_DEPTH, leader=mbar_leader)
        scale_full_bar = TMABar(pool, PIPE_DEPTH, leader=mbar_leader)
    tmem_full = TCGen05Bar(pool, TMEM_PIPE_DEPTH, leader=mbar_leader)
    tmem_empty = MBarrier(pool, TMEM_PIPE_DEPTH, leader=mbar_leader)
    tmem_finished = MBarrier(pool, 1, leader=mbar_leader)
    if mbar_leader:
        for stage in T.unroll(PIPE_DEPTH):
            T.ptx.mbarrier.init(smem_full.ptr_to([stage]), 1)
            T.ptx.mbarrier.init(smem_empty.ptr_to([stage]), 1)
            if not MERGE_TMA_BARRIERS:
                T.ptx.mbarrier.init(tile_full_bar.ptr_to([stage]), 1)
                T.ptx.mbarrier.init(scale_full_bar.ptr_to([stage]), 1)
        for stage in T.unroll(TMEM_PIPE_DEPTH):
            T.ptx.mbarrier.init(tmem_full.ptr_to([stage]), 1)
            T.ptx.mbarrier.init(tmem_empty.ptr_to([stage]), CTA_GROUP)
        T.ptx.mbarrier.init(tmem_finished.ptr_to([0]), 1)
    pool.commit()
    tmem_pool = T.TMEMPool(
        pool, total_cols=512, cta_group=CTA_GROUP, tmem_addr=tmem_addr, sync_after_alloc=False
    )
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
    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    if OVERLAP_EPI:
        T.ptx.barrier.cluster.arrive(sem="relaxed", aligned=True)
    else:
        T.cuda.cluster_sync()
    # Split the overlap prologue so the first TMA flight and TMEM allocation
    # can run in parallel. Each active role performs the acquire wait below.
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
    epi_store_iter: T.int32
    epi_store_iter = 0

    @T.inline
    def issue_scale_tma_load(k_tile: T.int32):
        stage = tma_cur.stage
        sf_k = T.meta_var(k_tile * SF_CTA_K)
        sf_m = T.meta_var((a_m // 128) * 128)
        sf_n = T.meta_var((d_n // 128) * 128)
        smem_empty.wait(tma_cur.stage, tma_cur.phase)
        if not MERGE_TMA_BARRIERS:
            if id_in_pair == 0:
                scale_bytes = T.meta_var(SFA_BYTES + SFB_BYTES)
                T.ptx.mbarrier.arrive.expect_tx(
                    scale_full_bar.ptr_to([stage]), scale_bytes, remote=pair_leader_rank, pred=True
                )
        single_cta_mask: T.int32 = 1 << id_in_pair
        # SFA: each CTA loads its half (single_cta_mask). SFB: multicast to
        # both CTAs (pair_mask).
        if MERGE_TMA_BARRIERS:
            sfa_copy = T.meta_var(
                _tma_g2s_args(smem_full, stage, single_cta_mask, CTA_GROUP, LOAD_CACHE_HINT)
            )
        else:
            sfa_copy = T.meta_var(
                _tma_g2s_args(scale_full_bar, stage, single_cta_mask, CTA_GROUP, LOAD_CACHE_HINT)
            )
        Tx.copy_async(
            SFA_smem[stage, 0:CTA_M, 0:SF_CTA_K],
            SFA_in[sf_m : sf_m + CTA_M, sf_k : sf_k + SF_CTA_K],
            **sfa_copy,
        )
        if MERGE_TMA_BARRIERS:
            sfb_copy = T.meta_var(
                _tma_g2s_args(smem_full, stage, pair_mask, CTA_GROUP, LOAD_CACHE_HINT)
            )
        else:
            sfb_copy = T.meta_var(
                _tma_g2s_args(scale_full_bar, stage, pair_mask, CTA_GROUP, LOAD_CACHE_HINT)
            )
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

    if warp_id == int(WarpRole.TMA):
        if OVERLAP_EPI:
            T.ptx.barrier.cluster.wait(acquire=True, aligned=False)

        @T.inline
        def issue_tma_load(k_tile: T.int32):
            stage = tma_cur.stage
            k = T.meta_var(k_tile * CTA_K // 2)
            smem_empty.wait(tma_cur.stage, tma_cur.phase)
            if id_in_pair == 0:
                if MERGE_TMA_BARRIERS:
                    tile_bytes = T.meta_var(A_BYTES + B_BYTES + SFA_BYTES + SFB_BYTES)
                    T.ptx.mbarrier.arrive.expect_tx(
                        smem_full.ptr_to([stage]), tile_bytes, remote=pair_leader_rank, pred=True
                    )
                else:
                    tile_bytes = T.meta_var(A_BYTES + B_BYTES)
                    T.ptx.mbarrier.arrive.expect_tx(
                        tile_full_bar.ptr_to([stage]),
                        tile_bytes,
                        remote=pair_leader_rank,
                        pred=True,
                    )
            single_cta_mask: T.int32 = 1 << id_in_pair
            # Barrier pre-mapped to the cluster leader (the g2s primitive maps
            # neither the barrier nor expect_tx — both handled above).
            if MERGE_TMA_BARRIERS:
                tile_copy = T.meta_var(
                    _tma_g2s_args(smem_full, stage, single_cta_mask, CTA_GROUP, LOAD_CACHE_HINT)
                )
            else:
                tile_copy = T.meta_var(
                    _tma_g2s_args(tile_full_bar, stage, single_cta_mask, CTA_GROUP, LOAD_CACHE_HINT)
                )
            for split_idx in T.unroll(TMA_SPLIT):
                split_m = T.meta_var(split_idx * (CTA_M // TMA_SPLIT))
                Tx.copy_async(
                    A_smem_packed[stage, split_m : split_m + CTA_M // TMA_SPLIT, 0 : CTA_K // 2],
                    A_packed[
                        a_m + split_m : a_m + split_m + CTA_M // TMA_SPLIT, k : k + CTA_K // 2
                    ],
                    **tile_copy,
                )
            for split_idx in T.unroll(TMA_SPLIT):
                split_n = T.meta_var(split_idx * (CTA_N // TMA_SPLIT))
                Tx.copy_async(
                    B_smem_packed[stage, split_n : split_n + CTA_N // TMA_SPLIT, 0 : CTA_K // 2],
                    B_packed[
                        b_n + split_n : b_n + split_n + CTA_N // TMA_SPLIT, k : k + CTA_K // 2
                    ],
                    **tile_copy,
                )

        if T.ptx.elect_sync():
            while tile_scheduler.valid():
                for k_tile in T.serial(K_TILES):
                    issue_tma_load(k_tile)
                    tma_cur.advance()
                if CLC_SCHEDULER:
                    tile_scheduler.consume()
                    tile_scheduler.advance_coords()
                    tile_scheduler.mark_done_if_drained()
                else:
                    tile_scheduler.next_tile()
    elif warp_id == int(WarpRole.TMA) + 1:
        if OVERLAP_EPI:
            T.ptx.barrier.cluster.wait(acquire=True, aligned=False)
        if T.ptx.elect_sync():
            while tile_scheduler.valid():
                for k_tile in T.serial(K_TILES):
                    issue_scale_tma_load(k_tile)
                    tma_cur.advance()
                if CLC_SCHEDULER:
                    tile_scheduler.consume()
                    tile_scheduler.advance_coords()
                    tile_scheduler.mark_done_if_drained()
                else:
                    tile_scheduler.next_tile()
    elif warp_id == int(WarpRole.MMA) + 1:
        if CLC_SCHEDULER:
            if OVERLAP_EPI:
                T.ptx.barrier.cluster.wait(acquire=True, aligned=False)
            clc_sched.run_scheduler(id_in_pair)
            # Match nvjet's PREEXIT tail: once CLC has handed out every tile,
            # let the next same-stream launch enter while the current launch's
            # MMA/epilogue roles drain their final task.
            if id_in_pair == 0:
                T.ptx.griddepcontrol.launch_dependents()
    elif (warp_id == int(WarpRole.MMA)) & (id_in_pair == 0):
        if OVERLAP_EPI:
            T.ptx.barrier.cluster.wait(acquire=True, aligned=False)

        @T.inline
        def execute_mma():
            stage = mma_smem.stage
            if MERGE_TMA_BARRIERS:
                smem_full.wait(mma_smem.stage, mma_smem.phase)
            else:
                scale_full_bar.wait(mma_smem.stage, mma_smem.phase)
                tile_full_bar.wait(mma_smem.stage, mma_smem.phase)
            if FUSED_SCALE_MMA:
                T.evaluate(
                    cuda_func_call(
                        "tirx_nvfp4_copy_scale_and_mma_k256_2cta",
                        tmem.allocated_addr[0],
                        A_smem.ptr_to([stage, 0, 0]),
                        B_smem.ptr_to([stage, 0, 0]),
                        SFA_smem.ptr_to([stage, 0, 0]),
                        SFB_smem.ptr_to([stage, 0, 0]),
                        accum,
                        source_code=_NVFP4_COPY_SCALE_AND_MMA_K256_2CTA_SRC,
                    )
                )
            else:
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
            smem_empty.arrive(mma_smem.stage, cta_group=CTA_GROUP, cta_mask=pair_mask)

        if T.ptx.elect_sync():
            while tile_scheduler.valid():
                if CLC_SCHEDULER:
                    tile_scheduler.consume()
                tmem_empty.wait(mma_tmem.stage, mma_tmem.phase)
                accum = 0
                for k_tile in T.serial(K_TILES):
                    execute_mma()
                    mma_smem.advance()
                tmem_full.arrive(mma_tmem.stage, cta_group=CTA_GROUP, cta_mask=pair_mask)
                mma_tmem.advance()
                if CLC_SCHEDULER:
                    tile_scheduler.mark_done_if_drained()
                else:
                    tile_scheduler.next_tile()
    elif warp_id >= int(WarpRole.EPILOGUE):
        if OVERLAP_EPI:
            T.ptx.barrier.cluster.wait(acquire=True, aligned=False)
        alpha_local: T.float32
        alpha_local = alpha

        @T.inline
        def regs_to_smem(reg_ldst_16b):
            if DIRECT_EPI:
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
        def epilogue(d_m_cur, d_n_cur):
            tmem_full.wait(epi_cur.stage, epi_cur.phase)

            # Per-chunk store: R->S (stmatrix) then S->G (TMA). Shared by both schedules.
            @T.inline
            def store_epi_chunk(reg_ldst_16b, linear_n: T.constexpr):
                if epi_store_iter >= WB_PIPE_DEPTH:
                    T.ptx.cp_async.bulk.wait_group(WB_PIPE_DEPTH - 1, read=True)
                    T.cuda.warpgroup_sync(1)
                regs_to_smem(reg_ldst_16b)
                T.cuda.warpgroup_sync(1)
                d_n_out: T.int32
                d_n_out = d_n_cur + linear_n
                if tid_in_wg == 0:
                    T.ptx.fence.proxy_async("shared::cta")
                    Tx.copy_async(
                        D[d_m_cur : d_m_cur + CTA_M, d_n_out : d_n_out + EPI_TILE],
                        output_smem[epi_wb_state.stage, 0:CTA_M, 0:EPI_TILE],
                        dispatch="tma_auto",
                        cache_hint="evict_first",
                        prefetch_tensormap=True,
                    )
                    T.ptx.cp_async.bulk.commit_group()
                epi_wb_state.advance()
                epi_store_iter = epi_store_iter + 1

            # Fusion vs fission of {load; scale+cast; store}: overlap fuses and reuses
            # a small (128, EPI_TILE) frag; non-overlap splits the loops, needing a big
            # (128, MMA_N) frag (all chunks live between load and store).
            if OVERLAP_EPI:
                if DIRECT_EPI:
                    reg_ldst = T.wg_reg_tile(EPI_TILE)
                    reg_ldst_16b = T.wg_reg_tile(EPI_TILE, dtype="bfloat16")
                else:
                    reg_ldst = T.alloc_tcgen05_ldst_frag("16x256b", (128, EPI_TILE), "float32")
                    reg_ldst_16b = T.alloc_cast_frag(reg_ldst, "bfloat16")
                for no in T.unroll(MMA_N // EPI_TILE):
                    linear_n = T.meta_var(no * EPI_TILE)
                    Tx.wg.copy_async(reg_ldst[:, :], tmem[:, linear_n : linear_n + EPI_TILE])
                    if no == MMA_N // EPI_TILE - 1:
                        T.ptx.tcgen05.wait.ld()
                        if EARLY_TMEM_TEARDOWN:
                            T.cuda.warpgroup_sync(1)
                        if tid_in_wg == 0:
                            tmem_empty.arrive(
                                epi_cur.stage, remote=pair_leader_rank, pred=True, count=1
                            )
                            if EARLY_TMEM_TEARDOWN:
                                if (
                                    tile_scheduler.linear_idx + NUM_CLUSTERS
                                    >= CLUSTER_M_TILES * CLUSTER_N_TILES
                                ):
                                    T.ptx.mbarrier.arrive(
                                        tmem_finished.ptr_to([0]),
                                        remote=pair_leader_rank + 1 - id_in_pair,
                                        pred=True,
                                        count=1,
                                    )
                    Tx.wg.mul(reg_ldst, reg_ldst, alpha_local)
                    Tx.wg.cast(reg_ldst_16b, reg_ldst)
                    store_epi_chunk(reg_ldst_16b, linear_n)
            else:
                # Keep the 2D frag so it can be column-sliced for the chunked store.
                if DIRECT_EPI:
                    reg_all = T.wg_reg_tile(MMA_N)
                    reg_all_16b = T.wg_reg_tile(MMA_N, dtype="bfloat16")
                else:
                    reg_all = T.alloc_tcgen05_ldst_frag("16x256b", (128, MMA_N), "float32")
                    reg_all_16b = T.alloc_cast_frag(reg_all, "bfloat16")
                for no in T.unroll(MMA_N // EPI_TILE):
                    ln = T.meta_var(no * EPI_TILE)
                    Tx.wg.copy_async(reg_all[:, ln : ln + EPI_TILE], tmem[:, ln : ln + EPI_TILE])
                T.ptx.tcgen05.wait.ld()
                if EARLY_TMEM_TEARDOWN:
                    T.cuda.warpgroup_sync(1)
                    if tid_in_wg == 0:
                        tmem_empty.arrive(
                            epi_cur.stage, remote=pair_leader_rank, pred=True, count=1
                        )
                        if (
                            tile_scheduler.linear_idx + NUM_CLUSTERS
                            >= CLUSTER_M_TILES * CLUSTER_N_TILES
                        ):
                            T.ptx.mbarrier.arrive(
                                tmem_finished.ptr_to([0]),
                                remote=pair_leader_rank + 1 - id_in_pair,
                                pred=True,
                                count=1,
                            )
                # scale + cast the whole frag
                Tx.wg.mul(reg_all, reg_all, alpha_local)
                Tx.wg.cast(reg_all_16b, reg_all)
                if not EARLY_TMEM_TEARDOWN:
                    if tid_in_wg == 0:
                        tmem_empty.arrive(
                            epi_cur.stage, remote=pair_leader_rank, pred=True, count=1
                        )
                    T.cuda.warpgroup_sync(1)
                for no in T.unroll(MMA_N // EPI_TILE):
                    ln = T.meta_var(no * EPI_TILE)
                    store_epi_chunk(reg_all_16b[:, ln : ln + EPI_TILE], ln)

        if CLC_SCHEDULER:
            cur_d_m: T.int32
            cur_d_n: T.int32
            while tile_scheduler.valid():
                cur_d_m = d_m
                cur_d_n = d_n
                tile_scheduler.consume_wg(wg_id, warp_id_in_wg, lane_id)
                tile_scheduler.advance_coords()
                epilogue(cur_d_m, cur_d_n)
                epi_cur.advance()
                tile_scheduler.mark_done_if_drained()
        else:
            while tile_scheduler.valid():
                epilogue(d_m, d_n)
                epi_cur.advance()
                tile_scheduler.next_tile()
        if not EARLY_TMEM_TEARDOWN:
            if tid_in_wg == 0:
                T.ptx.cp_async.bulk.wait_group(0, read=True)
            T.cuda.warpgroup_sync(1)
    if warp_id == int(WarpRole.EPILOGUE):
        if not EARLY_TMEM_TEARDOWN:
            if T.ptx.elect_sync():
                T.ptx.mbarrier.arrive(
                    tmem_finished.ptr_to([0]),
                    remote=pair_leader_rank + 1 - id_in_pair,
                    pred=True,
                    count=1,
                )
        T.ptx.mbarrier.try_wait_acquire_cluster(tmem_finished.ptr_to([0]), 0)
        T.ptx.tcgen05.dealloc(tmem_pool.addr, n_cols=512, cta_group=CTA_GROUP)


def tir_ws_kernel(M: int, N: int, K: int):
    assert M % 128 == 0 and N % 256 == 0 and K % 256 == 0
    assert (M // 128) % 2 == 0
    assert (K // 16) % 4 == 0
    config = dict(TIRX_CONFIGS.get((M, N, K), {}))
    tma_split = config.get("TMA_SPLIT", 1)
    cta_n = config.get("CTA_N", 128)
    assert 128 % tma_split == 0 and cta_n % tma_split == 0
    assert 128 // tma_split >= 8 and cta_n // tma_split >= 8
    return _kernel.specialize(M=M, N=N, K=K, **config)


TIRX_CONFIGS = {
    # Per-shape launch/pipeline tuning. The cluster N tile spans CTA_GROUP CTAs,
    # so CTA_N = (cluster N tile) / CTA_GROUP.
    (1024, 1024, 1024): {
        "SM_COUNT": 64,
        "CTA_N": 64,
        "EPI_TILE": 32,
        "WB_PIPE_DEPTH": 3,
        "PIPE_DEPTH": 5,
        "L2_GROUP_SIZE": 12,
        "OVERLAP_EPI": True,
        "LOAD_CACHE_HINT": None,
        "MERGE_TMA_BARRIERS": True,
        "EARLY_TMEM_TEARDOWN": True,
        "DIRECT_EPI": True,
    },
    (2048, 2048, 2048): {
        "SM_COUNT": 128,
        "CTA_N": 128,
        "EPI_TILE": 32,
        "WB_PIPE_DEPTH": 3,
        "PIPE_DEPTH": 5,
        "L2_GROUP_SIZE": 4,
        "OVERLAP_EPI": True,
        "TMA_SPLIT": 8,
        "MERGE_TMA_BARRIERS": True,
        "EARLY_TMEM_TEARDOWN": True,
        "DIRECT_EPI": True,
    },
    (4096, 4096, 4096): {
        "SM_COUNT": 148,
        "CTA_N": 128,
        "EPI_TILE": 32,
        "PIPE_DEPTH": 5,
        "L2_GROUP_SIZE": 4,
        "OVERLAP_EPI": False,
        "LOAD_CACHE_HINT": None,
        "DIRECT_EPI": True,
    },
    (8192, 8192, 8192): {
        "SM_COUNT": 148,
        "CTA_N": 128,
        "EPI_TILE": 32,
        "PIPE_DEPTH": 5,
        "L2_GROUP_SIZE": 1,
        "CLC_SCHEDULER": True,
        "OVERLAP_EPI": False,
        "LOAD_CACHE_HINT": None,
        "MERGE_TMA_BARRIERS": True,
        "DIRECT_EPI": True,
        "FUSED_SCALE_MMA": True,
    },
    (16384, 16384, 16384): {
        "SM_COUNT": 148,
        "CTA_N": 128,
        "EPI_TILE": 16,
        "PIPE_DEPTH": 5,
        "L2_GROUP_SIZE": 16,
        "CLC_SCHEDULER": True,
        "OVERLAP_EPI": False,
        "LOAD_CACHE_HINT": "evict_normal",
        "DIRECT_EPI": True,
        "FUSED_SCALE_MMA": True,
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
    alpha_value = float(alpha.item())
    out = torch.empty_like(C_ref).to("cuda").to(torch.bfloat16)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        ex.mod(A_fp4, B_fp4, A_sf, B_sf, alpha_value, out)
    cosine_sim = F.cosine_similarity(
        out.reshape(-1).float(), C_ref.to("cuda").reshape(-1).float(), dim=0
    )
    assert cosine_sim > 0.97, f"nvfp4_gemm cosine_sim {cosine_sim:.6f} <= 0.97"


def _flashinfer_autotune_cache_path(
    M: int, N: int, K: int, *, backend: str = "auto"
) -> Path | None:
    """Return an environment-specific, per-shape FlashInfer cache path."""
    cache_root = os.environ.get("TIRX_BENCH_CACHE_DIR")
    if not cache_root:
        return None

    # FlashInfer rejects cache metadata from a different software/GPU stack.
    # Put each stack in its own directory as well, so an obsolete file cannot
    # prevent the current process from saving its newly tuned result.
    from flashinfer.autotuner import _collect_metadata

    environment = _collect_metadata()
    digest = hashlib.sha256(
        json.dumps(environment, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    version = str(getattr(flashinfer, "__version__", "unknown")).replace("/", "_")
    backend_suffix = "" if backend == "auto" else f"_{backend}"
    return (
        Path(cache_root)
        / "flashinfer"
        / f"{version}-{digest}"
        / f"nvfp4_gemm{backend_suffix}_{M}x{N}x{K}.json"
    )


def _flashinfer_tuned_choice(
    M: int, N: int, K: int, cache_path: Path | None, *, expected_runner: str | None = None
) -> tuple[str, object]:
    """Read the exact-shape runner/tactic selected by FlashInfer's autotuner."""
    choices: list[tuple[str, object]] = []
    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text())
        packed_shape_prefix = f"(({M}, {K // 2}), ({K // 2}, {N})"
        choices.extend(
            (value[0], value[1])
            for key, value in payload.items()
            if key.startswith("('fp4_gemm', ") and packed_shape_prefix in key
        )
    else:
        from flashinfer.autotuner import AutoTuner

        for key, (_, tactic, _) in AutoTuner.get().profiling_cache.items():
            if (
                key.custom_op == "fp4_gemm"
                and len(key.nearest_profile) >= 2
                and key.nearest_profile[0] == (M, K // 2)
                and key.nearest_profile[1] == (K // 2, N)
            ):
                choices.append((key.runner_class_name, tactic))

    if expected_runner is not None:
        choices = [choice for choice in choices if choice[0] == expected_runner]

    unique = list(
        dict.fromkeys((runner, json.dumps(tactic, sort_keys=True)) for runner, tactic in choices)
    )
    if len(unique) != 1:
        raise RuntimeError(
            "FlashInfer autotune did not produce exactly one fp4_gemm choice "
            f"for M={M}, N={N}, K={K}, expected_runner={expected_runner}: {choices}"
        )
    runner, tactic_json = unique[0]
    tactic = json.loads(tactic_json)
    if tactic == -1:
        raise RuntimeError(
            f"FlashInfer autotune fell back to {runner} tactic=-1 for M={M}; "
            "refusing to benchmark an untuned fallback"
        )
    return runner, tactic


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
    out_tir = torch.empty_like(C_ref).to("cuda").to(torch.bfloat16)

    funcs = {"tir": lambda: ex.mod(A_fp4, B_fp4, A_sf, B_sf, alpha_value, out_tir)}
    flashinfer_backend = os.environ.get("TIRX_NVFP4_FLASHINFER_BACKEND", "auto")
    if flashinfer_backend not in {"auto", "cutlass"}:
        raise ValueError(
            f"TIRX_NVFP4_FLASHINFER_BACKEND must be 'auto' or 'cutlass', got {flashinfer_backend!r}"
        )
    expected_flashinfer_runner = "CutlassFp4GemmRunner" if flashinfer_backend == "cutlass" else None
    flashinfer_cache_path = _flashinfer_autotune_cache_path(M, N, K, backend=flashinfer_backend)
    flashinfer_context_kwargs = {"tuning_buckets": (M,), "round_up": False}
    if flashinfer_cache_path is not None:
        flashinfer_context_kwargs["cache"] = str(flashinfer_cache_path)

    def _flashinfer():
        out_fi = torch.empty_like(out_tir)
        cache_hit_before_tune = False
        if flashinfer_cache_path is not None and flashinfer_cache_path.exists():
            try:
                _flashinfer_tuned_choice(
                    M, N, K, flashinfer_cache_path, expected_runner=expected_flashinfer_runner
                )
                cache_hit_before_tune = True
            except (json.JSONDecodeError, RuntimeError):
                pass

        def run():
            return flashinfer.mm_fp4(
                A_fp4,
                B_fp4.T,
                A_sf,
                B_sf.T,
                alpha,
                out=out_fi,
                block_size=16,
                backend=flashinfer_backend,
                use_nvfp4=True,
            )

        # Tune/load exactly this benchmark shape and persist the selection in
        # the suite cache. Both profiling and all cache I/O happen before the
        # launch closure is handed to bench().
        with flashinfer.autotune(True, **flashinfer_context_kwargs):
            run()
        torch.cuda.synchronize()

        # Exercise the normal non-tuning lookup once before timing and reject
        # a silent heuristic fallback. Keep the exact same bucket override that
        # was used while tuning: cuDNN runner cache keys include its mapper.
        with flashinfer.autotune(False, **flashinfer_context_kwargs):
            run()
        torch.cuda.synchronize()
        runner, tactic = _flashinfer_tuned_choice(
            M, N, K, flashinfer_cache_path, expected_runner=expected_flashinfer_runner
        )
        sample_rows = min(M, 256)
        sample_cols = min(N, 256)
        cosine_similarity = torch.nn.functional.cosine_similarity(
            out_fi[:sample_rows, :sample_cols].reshape(-1).float(),
            C_ref[:sample_rows, :sample_cols].reshape(-1).float(),
            dim=0,
        ).item()
        if cosine_similarity <= 0.97:
            raise RuntimeError(
                "FlashInfer tuned NVFP4 output failed validation: "
                f"cosine_similarity={cosine_similarity:.6f}"
            )
        metadata.update(
            {
                "flashinfer_autotune_cache": (
                    "hit"
                    if cache_hit_before_tune
                    else "miss"
                    if flashinfer_cache_path is not None
                    else "memory"
                ),
                "flashinfer_tuning_bucket": M,
                "flashinfer_requested_backend": flashinfer_backend,
                "flashinfer_runner": runner,
                "flashinfer_tactic": tactic,
                "flashinfer_cosine_similarity": cosine_similarity,
            }
        )
        return run

    def _cublaslt():
        ext = _load_cublaslt_nvfp4_ext()
        out_cublaslt = torch.empty_like(out_tir)
        return lambda: ext.nvfp4_cublaslt(
            A_fp4, B_fp4, A_sf, B_sf, alpha_value, out_cublaslt, M, N, K
        )

    # FlashInfer is a required reference for this benchmark. Prepare and
    # validate its tuned launch before entering bench() so a bad/missing cache
    # fails the workload instead of being downgraded to an optional baseline
    # construction error.
    flashinfer_run = _flashinfer()
    # Load the file and install the exact-M mapper once, outside all timer
    # calls. Timed FlashInfer launches then perform only an in-memory cache
    # lookup and the selected kernel launch.
    with flashinfer.autotune(False, **flashinfer_context_kwargs):
        result = bench(
            funcs,
            warmup=warmup,
            repeat=repeat,
            timer=timer,
            references={"flashinfer": lambda: flashinfer_run, "cublaslt_nvfp4": _cublaslt},
            **kwargs,
        )
    result["metadata"] = {**result.get("metadata", {}), **metadata}
    return result
