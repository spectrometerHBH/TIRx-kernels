from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.lang.pipeline import MBarrier, TCGen05Bar, TMABar
from tvm.tirx.layout import ComposeLayout, Iter, S, SwizzleLayout, TCol, TileLayout, TLane

B_H = 128
B_TOPK = 128
D_V = 512
NUM_BUFS = 2
NUM_THREADS = 512
MAX_INIT_VAL = -1.0e30
LOG_2_E = math.log2(math.e)
LN_2 = math.log(2.0)

HEAD128_LAUNCH_PARAM_TAGS = (
    "blockIdx.x",
    "clusterCtaIdx.x",
    "threadIdx.x",
    "tirx.use_dyn_shared_memory",
)

TMEM_COL_O = 0
TMEM_COL_P = 256
TMEM_COL_Q = 320

NAMED_BARRIER_WG0_SYNC = 0

BF16_BYTES = 2
D_TQ = 384
NUM_TQ_TILES = D_TQ // 64
Q_FULL_DESC_SDO = 64
K_MAJOR_SWIZZLED_DESC_LDO = 1
P_TMEM_ELEMENTS = B_TOPK // 2
B_EPI = 64
WG1_NUM_WARPS = 4
WG1_NUM_LOCAL_ROWS_PER_WARP = (B_TOPK // 2) // 4 // WG1_NUM_WARPS
WG2_NUM_WARPS = 4
WG2_NUM_LOCAL_ROWS_PER_PART = (B_TOPK // 2) // 4 // WG2_NUM_WARPS


@dataclass(frozen=True)
class SparseFlashMLAPrefillHead128Config:
    label: str
    s_q: int
    s_kv: int
    topk: int
    d_qk: int
    h_q: int = B_H
    h_kv: int = 1
    d_v: int = D_V
    have_attn_sink: bool = False
    have_topk_length: bool = False
    inject_invalid_indices: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.h_q != B_H:
            raise ValueError("head128 regular phase1 requires h_q == 128")
        if self.h_kv != 1:
            raise ValueError("head128 regular phase1 requires h_kv == 1")
        if self.d_qk not in (512, 576):
            raise ValueError("d_qk must be 512 or 576")
        if self.d_v != D_V:
            raise ValueError("d_v must be 512")
        if self.topk % B_TOPK != 0:
            raise ValueError("topk must be a multiple of 128")


CONFIGS = [
    {
        "label": "regular_dqk512_s1_kv2048_topk1408",
        "s_q": 1,
        "s_kv": 2048,
        "topk": 1408,
        "d_qk": 512,
    },
    {
        "label": "regular_dqk576_s1_kv2048_topk1408",
        "s_q": 1,
        "s_kv": 2048,
        "topk": 1408,
        "d_qk": 576,
    },
    {
        "label": "regular_features_dqk512_s17_kv4096_topk1536",
        "s_q": 17,
        "s_kv": 4096,
        "topk": 1536,
        "d_qk": 512,
        "have_attn_sink": True,
        "have_topk_length": True,
    },
    {
        "label": "regular_features_dqk576_s17_kv4096_topk1536",
        "s_q": 17,
        "s_kv": 4096,
        "topk": 1536,
        "d_qk": 576,
        "have_attn_sink": True,
        "have_topk_length": True,
    },
    {
        "label": "regular_invalid_indices_dqk512_s3_kv2304_topk1408",
        "s_q": 3,
        "s_kv": 2304,
        "topk": 1408,
        "d_qk": 512,
        "inject_invalid_indices": True,
    },
    {
        "label": "regular_invalid_indices_dqk576_s3_kv2304_topk1408",
        "s_q": 3,
        "s_kv": 2304,
        "topk": 1408,
        "d_qk": 576,
        "inject_invalid_indices": True,
    },
]

BENCH_CONFIGS = [
    {
        "label": f"bench_regular_dqk{d_qk}_hq128_s4096_kv{s_kv}_topk2048",
        "s_q": 4096,
        "s_kv": s_kv,
        "topk": 2048,
        "d_qk": d_qk,
        "h_q": B_H,
        "have_attn_sink": True,
    }
    for d_qk in (512, 576)
    for s_kv in (8192, 32768, 65536)
]

KERNEL_META = {
    "name": "sparse_flashmla_prefill_head128_phase1",
    "category": "flashmla",
    "compute_capability": 10,
}


def _cfg(**kwargs: Any) -> SparseFlashMLAPrefillHead128Config:
    cfg_fields = {field.name for field in fields(SparseFlashMLAPrefillHead128Config)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    if "label" not in cfg_kwargs:
        cfg_kwargs["label"] = "custom"
    cfg = SparseFlashMLAPrefillHead128Config(**cfg_kwargs)
    cfg.validate()
    return cfg


def _flashmla_regular_dispatch_reason(cfg: SparseFlashMLAPrefillHead128Config) -> str:
    if cfg.h_q != B_H:
        return "out_of_scope: h_q != 128 dispatches to head64 or unsupported path"
    if cfg.d_qk == 512 and cfg.topk <= 1280:
        return "out_of_scope: sm100 head128 D_QK=512 topk<=1280 dispatches small-topk"
    return f"regular: sm100 head128 run_fwd_phase1_kernel<{cfg.d_qk}>"


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = kwargs.get("device", "cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)

    q = torch.randn(
        (cfg.s_q, cfg.h_q, cfg.d_qk), device=device, dtype=torch.bfloat16, generator=gen
    )
    kv = torch.randn(
        (cfg.s_kv, cfg.h_kv, cfg.d_qk), device=device, dtype=torch.bfloat16, generator=gen
    )
    out = torch.empty((cfg.s_q, cfg.h_q, cfg.d_v), device=device, dtype=torch.bfloat16)
    max_logits = torch.empty((cfg.s_q, cfg.h_q), device=device, dtype=torch.float32)
    lse = torch.empty((cfg.s_q, cfg.h_q), device=device, dtype=torch.float32)

    indices = torch.randint(
        low=0,
        high=cfg.s_kv,
        size=(cfg.s_q, cfg.h_kv, cfg.topk),
        device=device,
        dtype=torch.int32,
        generator=gen,
    )
    if cfg.inject_invalid_indices:
        indices[:, :, 0] = -1
        indices[:, :, 1] = cfg.s_kv
        indices[:, :, 2] = cfg.s_kv + 17
        indices[:, :, -1] = -7
    attn_sink = (
        torch.randn((cfg.h_q,), device=device, dtype=torch.float32, generator=gen)
        if cfg.have_attn_sink
        else torch.empty((cfg.h_q,), device=device, dtype=torch.float32)
    )
    if cfg.have_topk_length:
        topk_length = torch.randint(
            low=0,
            high=cfg.topk + 1,
            size=(cfg.s_q,),
            device=device,
            dtype=torch.int32,
            generator=gen,
        )
    else:
        topk_length = torch.empty((cfg.s_q,), device=device, dtype=torch.int32)

    sm_scale = 1.0 / math.sqrt(cfg.d_qk)
    return {
        "config": cfg,
        "q": q,
        "kv": kv,
        "indices": indices,
        "attn_sink": attn_sink,
        "topk_length": topk_length,
        "out": out,
        "max_logits": max_logits,
        "lse": lse,
        "sm_scale": sm_scale,
        "sm_scale_div_log2": sm_scale * LOG_2_E,
        "dispatch_reason": _flashmla_regular_dispatch_reason(cfg),
    }


def _reference_sparse_prefill(
    case: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg: SparseFlashMLAPrefillHead128Config = case["config"]
    q = case["q"].float()
    kv = case["kv"][:, 0, :].float()
    indices = case["indices"][:, 0, :].to(torch.long)
    sm_scale = case["sm_scale"]
    ref_out = torch.zeros((cfg.s_q, cfg.h_q, cfg.d_v), device=q.device, dtype=torch.float32)
    ref_max_logits = torch.full((cfg.s_q, cfg.h_q), -float("inf"), device=q.device)
    ref_lse = torch.full((cfg.s_q, cfg.h_q), float("inf"), device=q.device)

    for s_q_idx in range(cfg.s_q):
        length = int(case["topk_length"][s_q_idx].item()) if cfg.have_topk_length else cfg.topk
        row_indices = indices[s_q_idx]
        pos = torch.arange(cfg.topk, device=q.device)
        valid = (pos < length) & (row_indices >= 0) & (row_indices < cfg.s_kv)
        if not torch.any(valid):
            continue
        selected = row_indices.clamp(0, cfg.s_kv - 1)
        k_full = kv[selected]
        logits = torch.matmul(q[s_q_idx], k_full[:, : cfg.d_qk].T) * sm_scale
        logits[:, ~valid] = -float("inf")
        max_logits = torch.max(logits, dim=-1).values
        exp_logits = torch.exp(logits - max_logits[:, None])
        exp_logits[:, ~valid] = 0.0
        denom = torch.sum(exp_logits, dim=-1)
        if cfg.have_attn_sink:
            sink = case["attn_sink"].float()
            denom_with_sink = denom + torch.exp(sink - max_logits)
        else:
            denom_with_sink = denom
        ref_out[s_q_idx] = torch.matmul(exp_logits, k_full[:, : cfg.d_v]) / denom_with_sink[:, None]
        ref_max_logits[s_q_idx] = max_logits
        ref_lse[s_q_idx] = max_logits + torch.log(denom)
    return ref_out.to(torch.bfloat16), ref_max_logits, ref_lse


def _make_tirx_launch(case: dict[str, Any]) -> dict[str, Any]:
    import ctypes

    cfg: SparseFlashMLAPrefillHead128Config = case["config"]
    attn_sink_ptr = (
        ctypes.c_void_p(int(case["attn_sink"].data_ptr()))
        if cfg.have_attn_sink
        else ctypes.c_void_p(0)
    )
    topk_length_ptr = (
        ctypes.c_void_p(int(case["topk_length"].data_ptr()))
        if cfg.have_topk_length
        else ctypes.c_void_p(0)
    )
    return {
        "case": case,
        "args": (
            case["q"],
            case["kv"].reshape(-1),
            case["indices"].reshape(-1),
            attn_sink_ptr,
            topk_length_ptr,
            case["out"],
            case["max_logits"],
            case["lse"],
        ),
    }


def _build_tirx_launches(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [_make_tirx_launch(case)]


def _run_tirx_launches(
    executable: Any, launches: list[dict[str, Any]], *, output_case: dict[str, Any] | None = None
) -> None:
    for launch in launches:
        executable(*launch["args"])


def _tirx_benchmark_tensors(
    case: dict[str, Any], launches: list[dict[str, Any]]
) -> tuple[Any, ...]:
    return (
        case["q"],
        case["kv"],
        case["indices"],
        case["attn_sink"],
        case["topk_length"],
        case["out"],
        case["max_logits"],
        case["lse"],
    )


def _pack_valid_mask8(
    lane_indices: Any, abs_pos_start: Any, lane_idx: Any, topk_len: Any, s_kv: Any
) -> Any:
    terms = []
    for i in range(8):
        valid = (
            (lane_indices[i] >= 0)
            & (lane_indices[i] < s_kv)
            & (abs_pos_start + lane_idx * 8 + i < topk_len)
        )
        terms.append(T.Select(valid, T.int32(1 << i), T.int32(0)))
    while len(terms) > 1:
        terms = [T.bitwise_or(terms[i], terms[i + 1]) for i in range(0, len(terms), 2)]
    return T.cast(terms[0], "int8")


@T.jit
def _kernel(
    q: T.Buffer((s_q, h_q, d_qk), "bfloat16"),
    kv: T.Buffer((s_kv * stride_kv_s_kv,), "bfloat16"),
    indices: T.Buffer((s_q * stride_indices_s_q,), "int32"),
    attn_sink: T.handle("float32"),
    topk_length: T.handle("int32"),
    out: T.Buffer((s_q, h_q, D_V), "bfloat16"),
    max_logits: T.Buffer((s_q, h_q), "float32"),
    lse: T.Buffer((s_q, h_q), "float32"),
    *,
    s_q: T.constexpr,
    s_kv: T.constexpr,
    topk: T.constexpr,
    d_qk: T.constexpr,
    h_q: T.constexpr,
    stride_kv_s_kv: T.constexpr,
    stride_indices_s_q: T.constexpr,
    have_attn_sink: T.constexpr,
    have_topk_length: T.constexpr,
    sm_scale_div_log2: T.constexpr,
):
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    # CUDA_TRANSCRIBE_START: run_fwd_phase1_kernel line 622, then sparse_attn_fwd_kernel_devfunc line 68.
    # Transcription note: match upstream FlashMLA phase1's one CTA pair per query-row launch.
    # Transcription note: preserve upstream source-order roles and mixed TMA/MMA/softmax warp layout.
    block_idx = T.cta_id([2 * s_q])
    T.cta_id_in_cluster([2])
    cta_idx: T.let = block_idx % 2
    s_q_idx: T.let = block_idx // 2
    thread_idx = T.thread_id([NUM_THREADS])
    warp_idx: T.let = T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), thread_idx // 32, 0, 32)
    lane_idx: T.let = thread_idx % 32
    topk_len: T.let = (
        T.cuda.ldg(T.handle_add_byte_offset(topk_length, s_q_idx * 4), "int32")
        if have_topk_length
        else topk
    )
    num_k_blocks: T.let = T.max((topk_len + B_TOPK - 1) // B_TOPK, 1)
    warpgroup_idx: T.let = T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), thread_idx // 128, 0, 32)
    idx_in_warpgroup: T.let = thread_idx % 128
    d_sq = T.meta_var(d_qk - D_TQ)
    num_sq_tiles = T.meta_var((d_qk - D_TQ) // 64)
    num_qk_tiles = T.meta_var(d_qk // 64)
    shared_u_elems = T.meta_var((B_H // 2) * d_sq + (D_V // 2) * B_TOPK + (B_TOPK // 2) * d_qk)
    tiled_mma_smem_desc = T.meta_var(
        "recompute"
        if (d_qk == 512 and s_kv == 8192)
        else "local_hoist"
        if (d_qk == 576 and s_kv != 65536)
        else "hoist"
        if ((d_qk == 512 and s_kv == 32768) or (d_qk == 576 and s_kv == 65536))
        else "encode"
    )

    # CUDA phase1.cuh:84-90, config.h:93-118.  Preserve SharedMemoryPlan's
    # union offsets: q_full, {sq, v, k}, and o alias the same base.
    pool = T.SMEMPool()
    u_base = T.meta_var(pool.offset)
    q_full = pool.alloc_mma((B_H // 2, d_qk), "bfloat16")
    pool.move_base_to(u_base)
    sq_smem = pool.alloc_mma((B_H // 2, d_sq), "bfloat16")
    pool.move_base_to(u_base + (B_H // 2) * d_sq * BF16_BYTES)
    v_smem = pool.alloc_mma((D_V // 2, B_TOPK), "bfloat16")
    pool.move_base_to(u_base + ((B_H // 2) * d_sq + (D_V // 2) * B_TOPK) * BF16_BYTES)
    k_smem = pool.alloc_mma((B_TOPK // 2, d_qk), "bfloat16")
    pool.move_base_to(u_base)
    o_smem = pool.alloc_mma((B_H // 2, D_V), "bfloat16")
    pool.move_base_to(u_base + shared_u_elems * BF16_BYTES)
    s_smem = pool.alloc(((B_H // 2) * B_TOPK,), "bfloat16")
    is_k_valid = pool.alloc((NUM_BUFS, B_TOPK // 8), "int8")
    bar_prologue_q = TMABar(pool, 1)
    bar_prologue_utccp = TCGen05Bar(pool, 1)
    bar_qk_part_done = TCGen05Bar(pool, NUM_BUFS)
    bar_qk_done = TCGen05Bar(pool, NUM_BUFS)
    bar_sv_part_done = TCGen05Bar(pool, NUM_BUFS)
    bar_sv_done = TCGen05Bar(pool, NUM_BUFS)
    bar_k_part0_ready = TMABar(pool, NUM_BUFS)
    bar_k_part1_ready = TMABar(pool, NUM_BUFS)
    bar_v_part0_ready = TMABar(pool, NUM_BUFS)
    bar_v_part1_ready = TMABar(pool, NUM_BUFS)
    bar_p_free = MBarrier(pool, NUM_BUFS)
    bar_so_ready = MBarrier(pool, NUM_BUFS)
    bar_k_valid_ready = MBarrier(pool, NUM_BUFS)
    bar_k_valid_free = MBarrier(pool, NUM_BUFS)
    tmem_start_addr = pool.alloc((1,), "uint32", align=4)
    rowwise_max_buf = pool.alloc((128,), "float32")
    rowwise_li_buf = pool.alloc((128,), "float32")
    pool.commit()
    q_tma = q.view(d_qk, h_q, s_q, layout=TileLayout(S[(d_qk, h_q, s_q) : (1, d_qk, h_q * d_qk)]))
    out_tma = out.view(D_V, h_q, s_q, layout=TileLayout(S[(D_V, h_q, s_q) : (1, D_V, h_q * D_V)]))
    kv_tma = kv.view(s_kv, d_qk, layout=TileLayout(S[(s_kv, d_qk) : (stride_kv_s_kv, 1)]))

    g_indices_base: T.let = s_q_idx * stride_indices_s_q
    tiled_mma_p_accumulate = T.alloc_local((1,), "uint32")
    tiled_mma_o_accumulate = T.alloc_local((1,), "uint32")
    tiled_mma_p_accumulate[0] = T.uint32(0)
    tiled_mma_o_accumulate[0] = T.uint32(0)

    # CUDA phase1.cuh:87-146.  Warp 0 owns barrier init, Q TMA launch,
    # and the cta_group::2 TMEM allocation.
    if warp_idx == 0:
        if T.ptx.elect_sync():
            bar_prologue_q.init(1)
            bar_prologue_utccp.init(1)
            for init_stage in T.unroll(NUM_BUFS):
                T.ptx.mbarrier.init(bar_qk_part_done.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_qk_done.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_sv_part_done.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_sv_done.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_k_part0_ready.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_k_part1_ready.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_v_part0_ready.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_v_part1_ready.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_p_free.ptr_to([init_stage]), 128 * 2)
                T.ptx.mbarrier.init(bar_so_ready.ptr_to([init_stage]), 128 * 2)
                T.ptx.mbarrier.init(bar_k_valid_ready.ptr_to([init_stage]), 16)
                T.ptx.mbarrier.init(bar_k_valid_free.ptr_to([init_stage]), 128)
            T.ptx.fence.mbarrier_init()

    T.cuda.cluster_sync()

    if warp_idx == 0:
        if T.ptx.elect_sync():
            for q_tma_tile in T.unroll(num_qk_tiles):
                Tx.copy_async(
                    q_full[:, q_tma_tile * 64 : (q_tma_tile + 1) * 64],
                    q_tma[
                        q_tma_tile * 64 : (q_tma_tile + 1) * 64,
                        cta_idx * (B_H // 2) : (cta_idx + 1) * (B_H // 2),
                        s_q_idx : s_q_idx + 1,
                    ],
                    dispatch="tma",
                    mbar=bar_prologue_q.ptr_to([0]),
                    cta_group=2,
                    cache_hint=T.uint64(0x12F0000000000000),
                    tensor_map_dim_order="natural",
                    prefetch_tensormap=True,
                    tensormap_l2_promotion="L2::256B",
                )

        T.ptx.tcgen05.alloc(T.address_of(tmem_start_addr[0]), n_cols=512, cta_group=2)
        T.cuda.trap_when_assert_failed(tmem_start_addr[0] == T.uint32(0))
        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)

    T.cuda.cta_sync()

    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=2, tmem_addr=tmem_start_addr)
    tmem_ldst = tmem_pool.alloc((128, 512), "float32", datapath="D")
    tmem_pool.move_base_to(TMEM_COL_P)
    tmem_p = tmem_pool.alloc(
        (B_H // 2, B_TOPK),
        "float32",
        layout=TileLayout(S[(B_H // 2, 2, B_TOPK // 2) : (1 @ TLane, 64 @ TLane, 1 @ TCol)]),
    )
    tmem_pool.move_base_to(TMEM_COL_Q)
    q_tmem = tmem_pool.alloc(
        (B_H // 2, D_TQ), "bfloat16", layout=TileLayout(S[(B_H // 2, D_TQ) : (1 @ TLane, 1 @ TCol)])
    )
    tmem_pool.move_base_to(TMEM_COL_O)
    tmem_o_lo = tmem_pool.alloc(
        (B_H // 2, D_V // 2),
        "float32",
        layout=TileLayout(S[(B_H // 2, 2, D_V // 4) : (1 @ TLane, 64 @ TLane, 1 @ TCol)]),
    )
    tmem_o_hi = tmem_pool.alloc(
        (B_H // 2, D_V // 2),
        "float32",
        layout=TileLayout(S[(B_H // 2, 2, D_V // 4) : (1 @ TLane, 64 @ TLane, 1 @ TCol)]),
    )
    s_smem_gemm = s_smem.view(
        B_H // 2, B_TOPK, layout=TileLayout(S[(B_H // 2, B_TOPK) : (1, B_H // 2)])
    )
    v_smem_gemm = v_smem.view(
        B_TOPK,
        D_V // 2,
        layout=ComposeLayout(
            SwizzleLayout(3, 3, 3, swizzle_inner=True),
            TileLayout(S[(B_TOPK, (D_V // 2) // 64, 64) : (64, B_TOPK * 64, 1)]),
        ),
    )

    if warpgroup_idx == 0:
        # CUDA phase1.cuh:150-386.  Scale/exp warpgroup and epilogue.
        T.ptx.setmaxnreg(True, 144)
        mi = T.local_scalar("float32")
        mi = MAX_INIT_VAL
        li = T.local_scalar("float32")
        li = 0.0
        real_mi = T.local_scalar("float32")
        real_mi = T.float32(-float("inf"))
        scale_pair: T.let = T.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)

        for k in T.serial(0, num_k_blocks, unroll=False):
            cur_buf: T.let = k % NUM_BUFS
            cur_phase: T.let = (k // NUM_BUFS) & 1
            bar_qk_done.wait(cur_buf, cur_phase)
            T.ptx.tcgen05.fence.after_thread_sync()

            p_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, P_TMEM_ELEMENTS), "uint32")
            Tx.wg.copy_async(
                p_frag[:, :],
                tmem_ldst.with_dtype("uint32")[:, TMEM_COL_P : TMEM_COL_P + P_TMEM_ELEMENTS],
            )
            p = p_frag.local()
            T.ptx.tcgen05.wait.ld()
            T.ptx.tcgen05.fence.before_thread_sync()
            bar_p_free.arrive(cur_buf, cta_id=T.uint32(0))

            bar_k_valid_ready.wait(cur_buf, cur_phase)
            valid_word_offset: T.let = T.if_then_else(idx_in_warpgroup >= 64, B_TOPK // 8 // 2, 0)
            is_k_valid_lo: T.let = T.ptx.ld(
                is_k_valid.ptr_to([cur_buf, valid_word_offset]), "uint32", "u32", space="shared"
            )
            is_k_valid_hi: T.let = T.ptx.ld(
                is_k_valid.ptr_to([cur_buf, valid_word_offset + 4]), "uint32", "u32", space="shared"
            )
            for p_i in T.unroll(P_TMEM_ELEMENTS // 2):
                invalid_p_predicate: T.let = T.bitwise_and(
                    T.shift_right(is_k_valid_lo, T.uint32(p_i)), T.uint32(1)
                ) == T.uint32(0)
                p[p_i] = T.if_then_else(invalid_p_predicate, T.uint32(0xFF800000), p[p_i])
            for p_i in T.unroll(P_TMEM_ELEMENTS // 2):
                invalid_p_predicate: T.let = T.bitwise_and(
                    T.shift_right(is_k_valid_hi, T.uint32(p_i)), T.uint32(1)
                ) == T.uint32(0)
                p[p_i + P_TMEM_ELEMENTS // 2] = T.if_then_else(
                    invalid_p_predicate, T.uint32(0xFF800000), p[p_i + P_TMEM_ELEMENTS // 2]
                )

            cur_pi_max = T.local_scalar("float32")
            cur_pi_max = T.float32(-float("inf"))
            for p_i in T.unroll(P_TMEM_ELEMENTS):
                cur_pi_max = T.max(cur_pi_max, T.cuda.uint_as_float(p[p_i]))
            cur_pi_max = cur_pi_max * sm_scale_div_log2
            bar_k_valid_free.arrive(cur_buf)

            T.ptx.bar.sync(NAMED_BARRIER_WG0_SYNC, 128)
            rowwise_max_buf[idx_in_warpgroup] = cur_pi_max
            T.ptx.bar.sync(NAMED_BARRIER_WG0_SYNC, 128)
            cur_pi_max = T.max(cur_pi_max, rowwise_max_buf[idx_in_warpgroup ^ 64])
            real_mi = T.max(real_mi, cur_pi_max)
            should_scale_o = T.local_scalar("bool")
            should_scale_o = T.ptx.any_sync(T.uint32(0xFFFFFFFF), cur_pi_max - mi > 6.0) != 0

            new_max = T.local_scalar("float32")
            scale_for_old = T.local_scalar("float32")
            if not should_scale_o:
                scale_for_old = 1.0
                new_max = mi
            else:
                new_max = T.max(cur_pi_max, mi)
                scale_for_old = T.ptx.exp2(mi - new_max)
            mi = new_max
            li = li * scale_for_old

            s_pack = T.alloc_local((P_TMEM_ELEMENTS // 2,), "uint32")
            neg_new_max_pair: T.let = T.cuda.make_float2(-new_max, -new_max)
            for s_i in T.unroll(P_TMEM_ELEMENTS // 2):
                p_pair: T.let = T.cuda.make_float2(
                    T.cuda.uint_as_float(p[s_i * 2]), T.cuda.uint_as_float(p[s_i * 2 + 1])
                )
                fma_pair: T.let = T.ptx.fma_f32x2(p_pair, scale_pair, neg_new_max_pair, dps=False)
                s_x: T.let = T.ptx.exp2(T.cuda.float2_x(fma_pair))
                s_y: T.let = T.ptx.exp2(T.cuda.float2_y(fma_pair))
                li = li + s_x + s_y
                s_pack[s_i] = T.cuda.float22bfloat162_rn(s_x, s_y)

            if k > 0:
                prev_buf: T.let = (k - 1) % NUM_BUFS
                prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                bar_sv_done.wait(prev_buf, prev_phase)

            for s_store_i in T.unroll(P_TMEM_ELEMENTS // 8):
                s_store_offset = (
                    (idx_in_warpgroup % 64) * 8
                    + (idx_in_warpgroup // 64) * ((B_H // 2) * (B_TOPK // 2))
                    + s_store_i * (B_H // 2) * 8
                )
                Tx.copy(
                    s_smem.view("uint32")[s_store_offset // 2 : s_store_offset // 2 + 4],
                    s_pack[s_store_i * 4 : s_store_i * 4 + 4],
                )

            if (k > 0) & should_scale_o:
                T.ptx.tcgen05.fence.after_thread_sync()
                scale_for_old_pair: T.let = T.cuda.make_float2(scale_for_old, scale_for_old)
                o_rescale_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, 32), "float32")
                o_rescale = o_rescale_frag.local()
                for chunk_idx in T.unroll((D_V // 2) // 32):
                    Tx.wg.copy_async(
                        o_rescale_frag[:, :],
                        tmem_ldst[
                            :, TMEM_COL_O + chunk_idx * 32 : TMEM_COL_O + (chunk_idx + 1) * 32
                        ],
                    )
                    T.ptx.tcgen05.wait.ld()
                    for o_i in T.unroll(16):
                        o_pair: T.let = T.cuda.make_float2(
                            o_rescale[o_i * 2], o_rescale[o_i * 2 + 1]
                        )
                        o_scaled_pair: T.let = T.ptx.mul_f32x2(
                            o_pair, scale_for_old_pair, dps=False
                        )
                        o_rescale[o_i * 2] = T.cuda.float2_x(o_scaled_pair)
                        o_rescale[o_i * 2 + 1] = T.cuda.float2_y(o_scaled_pair)
                    Tx.wg.copy_async(
                        tmem_ldst[
                            :, TMEM_COL_O + chunk_idx * 32 : TMEM_COL_O + (chunk_idx + 1) * 32
                        ],
                        o_rescale_frag[:, :],
                    )
                    T.ptx.tcgen05.wait.st()
                T.ptx.tcgen05.fence.before_thread_sync()

            T.ptx.fence.proxy_async("shared::cta")
            bar_so_ready.arrive(cur_buf, cta_id=T.uint32(0))

        if real_mi == T.float32(-float("inf")):
            li = 0.0
            mi = T.float32(-float("inf"))

        rowwise_li_buf[idx_in_warpgroup] = li
        T.ptx.bar.sync(NAMED_BARRIER_WG0_SYNC, 128)
        li = li + rowwise_li_buf[idx_in_warpgroup ^ 64]

        if idx_in_warpgroup < B_H // 2:
            global_head: T.let = cta_idx * (B_H // 2) + idx_in_warpgroup
            cur_lse = T.local_scalar("float32")
            cur_lse_log: T.let = T.log(li)
            T.ptx.fma_f32(T.address_of(cur_lse), mi, LN_2, cur_lse_log)
            cur_lse = T.if_then_else(
                cur_lse == T.float32(-float("inf")), T.float32(float("inf")), cur_lse
            )
            max_logits[s_q_idx, global_head] = real_mi * LN_2
            lse[s_q_idx, global_head] = cur_lse

        last_k: T.let = num_k_blocks - 1
        last_buf: T.let = last_k % NUM_BUFS
        last_phase: T.let = (last_k // NUM_BUFS) & 1
        bar_sv_done.wait(last_buf, last_phase)
        T.ptx.tcgen05.fence.after_thread_sync()

        attn_sink_log2: T.let = (
            T.cuda.ldg(
                T.handle_add_byte_offset(
                    attn_sink, (cta_idx * (B_H // 2) + (idx_in_warpgroup % 64)) * 4
                ),
                "float32",
            )
            * LOG_2_E
            if have_attn_sink
            else T.float32(-float("inf"))
        )
        output_scale = T.local_scalar("float32")
        output_scale = T.cuda.fdividef(T.float32(1.0), li + T.ptx.exp2(attn_sink_log2 - mi))
        o_epi_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, B_EPI), "float32")
        o_epi = o_epi_frag.local()
        have_valid_indices: T.let = T.ptx.any_sync(T.uint32(0xFFFFFFFF), li != 0.0) != 0
        if not have_valid_indices:
            for o_zero_i in T.unroll(B_EPI):
                o_epi[o_zero_i] = 0.0
            output_scale = 1.0
        output_scale_pair: T.let = T.cuda.make_float2(output_scale, output_scale)
        for epi_k in T.unroll((D_V // 2) // B_EPI):
            if have_valid_indices:
                Tx.wg.copy_async(
                    o_epi_frag[:, :],
                    tmem_ldst[:, TMEM_COL_O + epi_k * B_EPI : TMEM_COL_O + (epi_k + 1) * B_EPI],
                )
                T.ptx.tcgen05.wait.ld()
            for o_i in T.unroll(B_EPI // 8):
                o_epi_bf16 = T.alloc_local((4,), "uint32")
                for o_j in T.unroll(4):
                    o_pair_idx: T.let = o_i * 8 + o_j * 2
                    o_pair: T.let = T.cuda.make_float2(o_epi[o_pair_idx], o_epi[o_pair_idx + 1])
                    o_epi_pair: T.let = T.ptx.mul_f32x2(o_pair, output_scale_pair, dps=False)
                    o_epi_bf16[o_j] = T.cuda.float22bfloat162_rn(
                        T.cuda.float2_x(o_epi_pair), T.cuda.float2_y(o_epi_pair)
                    )
                o_base_col: T.let = (idx_in_warpgroup // 64) * (D_V // 2) + epi_k * B_EPI + o_i * 8
                Tx.copy(
                    o_smem.view("uint32")[
                        idx_in_warpgroup % 64, o_base_col // 2 : o_base_col // 2 + 4
                    ],
                    o_epi_bf16[:],
                )

            T.ptx.fence.proxy_async("shared::cta")
            T.ptx.bar.sync(NAMED_BARRIER_WG0_SYNC, 128)
            if warp_idx == 0:
                if T.ptx.elect_sync():
                    Tx.copy_async(
                        out_tma[
                            epi_k * B_EPI : (epi_k + 1) * B_EPI,
                            cta_idx * (B_H // 2) : (cta_idx + 1) * (B_H // 2),
                            s_q_idx : s_q_idx + 1,
                        ],
                        o_smem[:, epi_k * B_EPI : (epi_k + 1) * B_EPI],
                        dispatch="tma",
                        tensor_map_dim_order="natural",
                        prefetch_tensormap=True,
                        tensormap_l2_promotion="L2::256B",
                    )
            if warp_idx == 1:
                if T.ptx.elect_sync():
                    epi_k2: T.let = epi_k + (D_V // B_EPI // 2)
                    Tx.copy_async(
                        out_tma[
                            epi_k2 * B_EPI : (epi_k2 + 1) * B_EPI,
                            cta_idx * (B_H // 2) : (cta_idx + 1) * (B_H // 2),
                            s_q_idx : s_q_idx + 1,
                        ],
                        o_smem[:, epi_k2 * B_EPI : (epi_k2 + 1) * B_EPI],
                        dispatch="tma",
                        tensor_map_dim_order="natural",
                        prefetch_tensormap=True,
                        tensormap_l2_promotion="L2::256B",
                    )

        if warp_idx == 0:
            T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=512, cta_group=2)

    elif warpgroup_idx == 1:
        # CUDA phase1.cuh:387-446.  K producer warpgroup.
        T.ptx.setmaxnreg(False, 96)
        wg1_warp_idx: T.let = warp_idx - 4
        if T.ptx.elect_sync():
            for k in T.serial(0, num_k_blocks, unroll=False):
                indices_int4 = T.alloc_local((WG1_NUM_LOCAL_ROWS_PER_WARP, 4), "int32")
                max_indices = T.local_scalar("int32")
                min_indices = T.local_scalar("int32")
                max_indices = -1
                min_indices = s_kv

                for local_row in T.unroll(WG1_NUM_LOCAL_ROWS_PER_WARP):
                    row_base: T.let = (
                        g_indices_base
                        + k * B_TOPK
                        + cta_idx * (B_TOPK // 2)
                        + (local_row * WG1_NUM_WARPS + wg1_warp_idx) * 4
                    )
                    T.cuda.ldg(
                        indices.ptr_to([row_base]),
                        "int32",
                        dst=(
                            indices_int4.ptr_to([local_row, 0]),
                            indices_int4.ptr_to([local_row, 1]),
                            indices_int4.ptr_to([local_row, 2]),
                            indices_int4.ptr_to([local_row, 3]),
                        ),
                        vec="v4",
                    )
                    local_max: T.let = T.max(
                        T.max(indices_int4[local_row, 0], indices_int4[local_row, 1]),
                        T.max(indices_int4[local_row, 2], indices_int4[local_row, 3]),
                    )
                    local_min: T.let = T.min(
                        T.min(indices_int4[local_row, 0], indices_int4[local_row, 1]),
                        T.min(indices_int4[local_row, 2], indices_int4[local_row, 3]),
                    )
                    max_indices = T.max(max_indices, local_max)
                    min_indices = T.min(min_indices, local_min)

                is_all_rows_invalid: T.let = (min_indices == s_kv) | (max_indices == -1)
                should_skip_tma: T.let = is_all_rows_invalid & (k >= NUM_BUFS)
                cur_buf: T.let = k % NUM_BUFS
                cur_phase: T.let = (k // NUM_BUFS) & 1

                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_qk_part_done.wait(prev_buf, prev_phase)
                if not should_skip_tma:
                    for local_col in T.unroll(num_sq_tiles):
                        raw_k_offset: T.let = wg1_warp_idx * 4 * 64 + local_col * (B_TOPK // 2) * 64
                        k_gather_tile = T.decl_buffer(
                            (WG1_NUM_LOCAL_ROWS_PER_WARP * 4, 64),
                            "bfloat16",
                            k_smem.data,
                            elem_offset=k_smem.elem_offset + raw_k_offset,
                            scope="shared.dyn",
                            layout=ComposeLayout(
                                SwizzleLayout(3, 3, 3, swizzle_inner=True),
                                TileLayout.from_iters(
                                    [
                                        Iter(
                                            WG1_NUM_LOCAL_ROWS_PER_WARP, WG1_NUM_WARPS * 4 * 64, "m"
                                        ),
                                        Iter(4, 64, "m"),
                                        Iter(64, 1, "m"),
                                    ]
                                ),
                            ),
                        )
                        Tx.copy_async(
                            k_gather_tile[:, :],
                            kv_tma[:, local_col * 64 : (local_col + 1) * 64],
                            dispatch="tma",
                            mbar=bar_k_part0_ready.ptr_to([cur_buf]),
                            cta_group=2,
                            cta_mask=T.uint16(1),
                            cache_hint=T.uint64(0x14F0000000000000),
                            gather_axis=0,
                            indexer=[
                                indices_int4[row, lane]
                                for row in range(WG1_NUM_LOCAL_ROWS_PER_WARP)
                                for lane in range(4)
                            ],
                            prefetch_tensormap=True,
                            tensormap_l2_promotion="L2::256B",
                        )
                else:
                    T.ptx.mbarrier.complete_tx(
                        bar_k_part0_ready.ptr_to([cur_buf]),
                        T.uint32(WG1_NUM_LOCAL_ROWS_PER_WARP * 4 * d_sq * BF16_BYTES),
                        T.uint32(0),
                        T.uint32(1),
                    )

                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_qk_done.wait(prev_buf, prev_phase)
                if not should_skip_tma:
                    for local_col_inner in T.unroll(num_qk_tiles - num_sq_tiles):
                        local_col: T.let = num_sq_tiles + local_col_inner
                        raw_k_offset: T.let = wg1_warp_idx * 4 * 64 + local_col * (B_TOPK // 2) * 64
                        k_gather_tile = T.decl_buffer(
                            (WG1_NUM_LOCAL_ROWS_PER_WARP * 4, 64),
                            "bfloat16",
                            k_smem.data,
                            elem_offset=k_smem.elem_offset + raw_k_offset,
                            scope="shared.dyn",
                            layout=ComposeLayout(
                                SwizzleLayout(3, 3, 3, swizzle_inner=True),
                                TileLayout.from_iters(
                                    [
                                        Iter(
                                            WG1_NUM_LOCAL_ROWS_PER_WARP, WG1_NUM_WARPS * 4 * 64, "m"
                                        ),
                                        Iter(4, 64, "m"),
                                        Iter(64, 1, "m"),
                                    ]
                                ),
                            ),
                        )
                        Tx.copy_async(
                            k_gather_tile[:, :],
                            kv_tma[:, local_col * 64 : (local_col + 1) * 64],
                            dispatch="tma",
                            mbar=bar_k_part1_ready.ptr_to([cur_buf]),
                            cta_group=2,
                            cta_mask=T.uint16(1),
                            cache_hint=T.uint64(0x14F0000000000000),
                            gather_axis=0,
                            indexer=[
                                indices_int4[row, lane]
                                for row in range(WG1_NUM_LOCAL_ROWS_PER_WARP)
                                for lane in range(4)
                            ],
                            prefetch_tensormap=True,
                            tensormap_l2_promotion="L2::256B",
                        )
                else:
                    T.ptx.mbarrier.complete_tx(
                        bar_k_part1_ready.ptr_to([cur_buf]),
                        T.uint32(WG1_NUM_LOCAL_ROWS_PER_WARP * 4 * D_TQ * BF16_BYTES),
                        T.uint32(0),
                        T.uint32(1),
                    )

    elif warpgroup_idx == 2:
        # CUDA phase1.cuh:447-489.  V producer warpgroup.
        T.ptx.setmaxnreg(False, 96)
        wg2_warp_idx: T.let = warp_idx - 8
        if T.ptx.elect_sync():
            bar_prologue_utccp.wait(0, 0)
            for k in T.serial(0, num_k_blocks, unroll=False):
                cur_buf: T.let = k % NUM_BUFS
                cur_phase: T.let = (k // NUM_BUFS) & 1
                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_sv_part_done.wait(prev_buf, prev_phase)
                token_idxs_part0 = T.alloc_local((WG2_NUM_LOCAL_ROWS_PER_PART, 4), "int32")
                for local_row in T.unroll(WG2_NUM_LOCAL_ROWS_PER_PART):
                    row_base: T.let = (
                        g_indices_base + k * B_TOPK + (local_row * WG2_NUM_WARPS + wg2_warp_idx) * 4
                    )
                    T.cuda.ldg(
                        indices.ptr_to([row_base]),
                        "int32",
                        dst=(
                            token_idxs_part0.ptr_to([local_row, 0]),
                            token_idxs_part0.ptr_to([local_row, 1]),
                            token_idxs_part0.ptr_to([local_row, 2]),
                            token_idxs_part0.ptr_to([local_row, 3]),
                        ),
                        vec="v4",
                    )
                for local_col in T.unroll((D_V // 2) // 64):
                    src_col: T.let = local_col * 64 + cta_idx * 256
                    raw_v_offset: T.let = wg2_warp_idx * 4 * 64 + local_col * B_TOPK * 64
                    v_gather_tile = T.decl_buffer(
                        (WG2_NUM_LOCAL_ROWS_PER_PART * 4, 64),
                        "bfloat16",
                        v_smem_gemm.data,
                        elem_offset=v_smem_gemm.elem_offset + raw_v_offset,
                        scope="shared.dyn",
                        layout=ComposeLayout(
                            SwizzleLayout(3, 3, 3, swizzle_inner=True),
                            TileLayout.from_iters(
                                [
                                    Iter(WG2_NUM_LOCAL_ROWS_PER_PART, WG2_NUM_WARPS * 4 * 64, "m"),
                                    Iter(4, 64, "m"),
                                    Iter(64, 1, "m"),
                                ]
                            ),
                        ),
                    )
                    Tx.copy_async(
                        v_gather_tile[:, :],
                        kv_tma[:, src_col : src_col + 64],
                        dispatch="tma",
                        mbar=bar_v_part0_ready.ptr_to([cur_buf]),
                        cta_group=2,
                        cta_mask=T.uint16(1),
                        cache_hint=T.uint64(0x14F0000000000000),
                        gather_axis=0,
                        indexer=[
                            token_idxs_part0[row, lane]
                            for row in range(WG2_NUM_LOCAL_ROWS_PER_PART)
                            for lane in range(4)
                        ],
                        prefetch_tensormap=True,
                        tensormap_l2_promotion="L2::256B",
                    )

                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_sv_done.wait(prev_buf, prev_phase)
                token_idxs_part1 = T.alloc_local((WG2_NUM_LOCAL_ROWS_PER_PART, 4), "int32")
                for local_row_inner in T.unroll(WG2_NUM_LOCAL_ROWS_PER_PART):
                    local_row: T.let = WG2_NUM_LOCAL_ROWS_PER_PART + local_row_inner
                    row_base: T.let = (
                        g_indices_base + k * B_TOPK + (local_row * WG2_NUM_WARPS + wg2_warp_idx) * 4
                    )
                    T.cuda.ldg(
                        indices.ptr_to([row_base]),
                        "int32",
                        dst=(
                            token_idxs_part1.ptr_to([local_row_inner, 0]),
                            token_idxs_part1.ptr_to([local_row_inner, 1]),
                            token_idxs_part1.ptr_to([local_row_inner, 2]),
                            token_idxs_part1.ptr_to([local_row_inner, 3]),
                        ),
                        vec="v4",
                    )
                for local_col in T.unroll((D_V // 2) // 64):
                    src_col: T.let = local_col * 64 + cta_idx * 256
                    raw_v_offset: T.let = (
                        wg2_warp_idx * 4 + WG2_NUM_LOCAL_ROWS_PER_PART * WG2_NUM_WARPS * 4
                    ) * 64 + local_col * B_TOPK * 64
                    v_gather_tile = T.decl_buffer(
                        (WG2_NUM_LOCAL_ROWS_PER_PART * 4, 64),
                        "bfloat16",
                        v_smem_gemm.data,
                        elem_offset=v_smem_gemm.elem_offset + raw_v_offset,
                        scope="shared.dyn",
                        layout=ComposeLayout(
                            SwizzleLayout(3, 3, 3, swizzle_inner=True),
                            TileLayout.from_iters(
                                [
                                    Iter(WG2_NUM_LOCAL_ROWS_PER_PART, WG2_NUM_WARPS * 4 * 64, "m"),
                                    Iter(4, 64, "m"),
                                    Iter(64, 1, "m"),
                                ]
                            ),
                        ),
                    )
                    Tx.copy_async(
                        v_gather_tile[:, :],
                        kv_tma[:, src_col : src_col + 64],
                        dispatch="tma",
                        mbar=bar_v_part1_ready.ptr_to([cur_buf]),
                        cta_group=2,
                        cta_mask=T.uint16(1),
                        cache_hint=T.uint64(0x14F0000000000000),
                        gather_axis=0,
                        indexer=[
                            token_idxs_part1[row, lane]
                            for row in range(WG2_NUM_LOCAL_ROWS_PER_PART)
                            for lane in range(4)
                        ],
                        prefetch_tensormap=True,
                        tensormap_l2_promotion="L2::256B",
                    )

    else:
        # CUDA phase1.cuh:490-606.  MMA warp and KV-valid loading warp.
        T.ptx.setmaxnreg(True, 168)
        if (cta_idx == 0) & (warp_idx == 12):
            if T.ptx.elect_sync():
                bar_prologue_q.arrive(0, tx_count=B_H * d_qk * BF16_BYTES)
                bar_prologue_q.wait(0, 0)
                T.ptx.tcgen05.fence.after_thread_sync()
                Tx.copy_async(
                    q_tmem[:, :],
                    q_full[:, d_sq : d_sq + D_TQ],
                    shape="64x128b",
                    cta_group=2,
                    multicast="warpx2::02_13",
                    desc_ldo=K_MAJOR_SWIZZLED_DESC_LDO,
                    desc_sdo=Q_FULL_DESC_SDO,
                    desc_swizzle=3,
                    tile_count=NUM_TQ_TILES,
                    subtile_count=8,
                    tmem_tile_stride_32b=32,
                    tmem_subtile_stride_32b=4,
                    desc_tile_stride_16b=(B_H // 2) * 128 // 16,
                    desc_subtile_stride_16b=1,
                )
                bar_prologue_utccp.arrive(0, cta_group=2, cta_mask=3)

                for k in T.serial(0, num_k_blocks + 1, unroll=False):
                    if k < num_k_blocks:
                        cur_buf: T.let = k % NUM_BUFS
                        cur_phase: T.let = (k // NUM_BUFS) & 1

                        bar_k_part0_ready.arrive(cur_buf, tx_count=B_TOPK * d_sq * BF16_BYTES)
                        bar_k_part0_ready.wait(cur_buf, cur_phase)
                        if k > 0:
                            prev_buf: T.let = (k - 1) % NUM_BUFS
                            prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                            bar_p_free.wait(prev_buf, prev_phase)
                        T.ptx.tcgen05.fence.after_thread_sync()

                        tiled_mma_p_accumulate[0] = T.uint32(0)
                        if d_sq > 0:
                            Tx.gemm_async(
                                tmem_p[:, :],
                                sq_smem[:, :d_sq],
                                k_smem[:, :d_sq],
                                accum=tiled_mma_p_accumulate[0],
                                dispatch="tcgen05",
                                cta_group=2,
                                smem_desc=tiled_mma_smem_desc,
                            )
                            tiled_mma_p_accumulate[0] = T.uint32(1)
                        bar_qk_part_done.arrive(cur_buf, cta_group=2, cta_mask=3)

                        bar_k_part1_ready.arrive(
                            cur_buf, tx_count=B_TOPK * (d_qk - d_sq) * BF16_BYTES
                        )
                        bar_k_part1_ready.wait(cur_buf, cur_phase)
                        T.ptx.tcgen05.fence.after_thread_sync()

                        Tx.gemm_async(
                            tmem_p[:, :],
                            q_tmem[:, :D_TQ],
                            k_smem[:, d_sq : d_sq + D_TQ],
                            accum=tiled_mma_p_accumulate[0],
                            dispatch="tcgen05",
                            cta_group=2,
                            smem_desc=tiled_mma_smem_desc,
                        )
                        tiled_mma_p_accumulate[0] = T.uint32(1)
                        bar_qk_done.arrive(cur_buf, cta_group=2, cta_mask=3)

                    if k > 0:
                        cur_buf_prev: T.let = (k - 1) % NUM_BUFS
                        cur_phase_prev: T.let = ((k - 1) // NUM_BUFS) & 1
                        bar_so_ready.wait(cur_buf_prev, cur_phase_prev)

                        bar_v_part0_ready.arrive(
                            cur_buf_prev, tx_count=(B_TOPK // 2) * D_V * BF16_BYTES
                        )
                        bar_v_part0_ready.wait(cur_buf_prev, cur_phase_prev)
                        T.ptx.tcgen05.fence.after_thread_sync()
                        tiled_mma_o_accumulate[0] = T.if_then_else(k == 1, T.uint32(0), T.uint32(1))
                        Tx.gemm_async(
                            tmem_o_lo[:, :],
                            s_smem_gemm[:, 0 : B_TOPK // 2],
                            v_smem_gemm[0 : B_TOPK // 2, 0 : D_V // 4],
                            transB=True,
                            accum=tiled_mma_o_accumulate[0],
                            dispatch="tcgen05",
                            cta_group=2,
                            smem_desc=tiled_mma_smem_desc,
                        )
                        Tx.gemm_async(
                            tmem_o_hi[:, :],
                            s_smem_gemm[:, 0 : B_TOPK // 2],
                            v_smem_gemm[0 : B_TOPK // 2, D_V // 4 : D_V // 2],
                            transB=True,
                            accum=tiled_mma_o_accumulate[0],
                            dispatch="tcgen05",
                            cta_group=2,
                            smem_desc=tiled_mma_smem_desc,
                        )
                        tiled_mma_o_accumulate[0] = T.uint32(1)
                        bar_sv_part_done.arrive(cur_buf_prev, cta_group=2, cta_mask=3)

                        bar_v_part1_ready.arrive(
                            cur_buf_prev, tx_count=(B_TOPK // 2) * D_V * BF16_BYTES
                        )
                        bar_v_part1_ready.wait(cur_buf_prev, cur_phase_prev)
                        T.ptx.tcgen05.fence.after_thread_sync()
                        Tx.gemm_async(
                            tmem_o_lo[:, :],
                            s_smem_gemm[:, B_TOPK // 2 : B_TOPK],
                            v_smem_gemm[B_TOPK // 2 : B_TOPK, 0 : D_V // 4],
                            transB=True,
                            accum=tiled_mma_o_accumulate[0],
                            dispatch="tcgen05",
                            cta_group=2,
                            smem_desc=tiled_mma_smem_desc,
                        )
                        Tx.gemm_async(
                            tmem_o_hi[:, :],
                            s_smem_gemm[:, B_TOPK // 2 : B_TOPK],
                            v_smem_gemm[B_TOPK // 2 : B_TOPK, D_V // 4 : D_V // 2],
                            transB=True,
                            accum=tiled_mma_o_accumulate[0],
                            dispatch="tcgen05",
                            cta_group=2,
                            smem_desc=tiled_mma_smem_desc,
                        )
                        tiled_mma_o_accumulate[0] = T.uint32(1)
                        bar_sv_done.arrive(cur_buf_prev, cta_group=2, cta_mask=3)

        elif warp_idx == 13:
            if lane_idx < B_TOPK // 8:
                lane_indices = T.alloc_local((8,), "int32")
                for k in T.serial(0, num_k_blocks, unroll=False):
                    T.ptx.ld_global_nc(
                        indices.ptr_to([g_indices_base + k * B_TOPK + lane_idx * 8]),
                        "int32",
                        "s32",
                        dst=lane_indices.ptr_to([0]),
                        vec="v8",
                        l1_evict="L1::evict_normal",
                        l2_evict="L2::evict_normal",
                        prefetch_size="L2::256B",
                    )
                    abs_pos_start: T.let = k * B_TOPK
                    is_ks_valid_mask: T.let = _pack_valid_mask8(
                        lane_indices, abs_pos_start, lane_idx, topk_len, s_kv
                    )
                    cur_buf: T.let = k % NUM_BUFS
                    cur_phase: T.let = (k // NUM_BUFS) & 1
                    bar_k_valid_free.wait(cur_buf, cur_phase ^ 1)
                    is_k_valid[cur_buf, lane_idx] = is_ks_valid_mask
                    bar_k_valid_ready.arrive(cur_buf)


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    stride_kv_s_kv = int(kwargs.get("stride_kv_s_kv", cfg.d_qk * cfg.h_kv))
    stride_indices_s_q = int(kwargs.get("stride_indices_s_q", cfg.topk * cfg.h_kv))
    kernel = _kernel.specialize(
        s_q=cfg.s_q,
        s_kv=cfg.s_kv,
        topk=cfg.topk,
        d_qk=cfg.d_qk,
        h_q=cfg.h_q,
        stride_kv_s_kv=stride_kv_s_kv,
        stride_indices_s_q=stride_indices_s_q,
        have_attn_sink=cfg.have_attn_sink,
        have_topk_length=cfg.have_topk_length,
        sm_scale_div_log2=(1.0 / math.sqrt(cfg.d_qk)) * LOG_2_E,
    )
    return kernel.with_attr("tirx.kernel_launch_params", list(HEAD128_LAUNCH_PARAM_TAGS))


def run_test(**kwargs: Any) -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA head128 phase1")

    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    cfg: SparseFlashMLAPrefillHead128Config = case["config"]
    if not case["dispatch_reason"].startswith("regular:"):
        raise SkipTest(case["dispatch_reason"])
    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)
    launches = _build_tirx_launches(case)
    _run_tirx_launches(ex, launches, output_case=case)
    torch.cuda.synchronize()
    ref_out, ref_max_logits, ref_lse = _reference_sparse_prefill(case)
    torch.testing.assert_close(case["out"], ref_out, rtol=4.01 / 128, atol=5e-3)
    torch.testing.assert_close(case["max_logits"], ref_max_logits, rtol=2.01 / 65536, atol=1e-6)
    torch.testing.assert_close(case["lse"], ref_lse, rtol=2.01 / 65536, atol=1e-6)
    cfg.validate()


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    _rounds = kwargs.pop("rounds", 1)
    _round_cooldown_s = kwargs.pop("round_cooldown_s", 1.0)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA head128 phase1 benchmark")

    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    case = prepare_data(**kwargs)
    if not case["dispatch_reason"].startswith("regular:"):
        raise SkipTest(case["dispatch_reason"])
    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    case["launches"] = _build_tirx_launches(case)

    funcs = {"tirx": lambda: _run_tirx_launches(ex, case["launches"])}

    from tirx_kernels.flashmla._flashmla_bench import flashmla_reference_builder

    def _flashmla_ref():
        run = flashmla_reference_builder()
        return lambda: run(case)

    return bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashmla": _flashmla_ref},
        rounds=_rounds,
        round_cooldown_s=_round_cooldown_s,
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
