"""FlashAttention-4 TIRx kernel with benchmark and NVIDIA IKET entry points."""

from __future__ import annotations

import argparse
import math
import os
from functools import partial

import numpy as np
import torch

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.bench import bench
from tvm.tirx.cuda import iket
from tvm.tirx.cuda.iket import IketProfiler
from tvm.tirx.lang.pipeline import MBarrier, Pipeline, PipelineState, TCGen05Bar
from tvm.tirx.lang.tile_scheduler import FlashAttentionLinearScheduler, FlashAttentionLPTScheduler
from tvm.tirx.layout import wg_local_layout

M_CLUSTER = 1
N_CLUSTER = 1
IKET_EVENT_NAMES = (
    "correction",
    "epi-ld-tmem",
    "issue-tma-k",
    "issue-tma-q",
    "issue-tma-v",
    "softmax-exp2",
    "softmax-fma",
    "softmax-max",
    "softmax-sum",
    "softmax-tmem-st",
    "tma-store",
    "softmax-baseline",
    "softmax-phase-0",
    "softmax-phase-1",
    "softmax-phase-2",
    "softmax-phase-3",
    "softmax-phase-4",
    "softmax-phase-5",
)
# Shape-tuned ex2 emulation selects pairs where i*2%16 >= 16-2*PAIRS in [START, 3).
# Causal includes fragment 0; non-causal starts at fragment 1.
EMU_PAIRS_CAUSAL = 2
EMU_START_CAUSAL = 0
EMU_PAIRS_NC = 2
EMU_START_NC = 1


def ceildiv(a, b):
    return (a + b - 1) // b


def combine_int_frac_ex2(x_rounded, frac_ex2):
    x_rounded_bits = T.cuda.float_as_uint(x_rounded)
    frac_ex2_bits = T.cuda.float_as_uint(frac_ex2)
    return T.cuda.uint_as_float(T.shift_left(x_rounded_bits, T.uint32(23)) + frac_ex2_bits)


def get_n_block_max(m_block_idx, causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE):
    """Maximum KV block index (exclusive) for this Q block."""
    n_block_max = ceildiv(SEQ_LEN_KV, BLK_N)
    if not causal:
        return n_block_max
    m_idx_max = (m_block_idx + 1) * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
    n_idx = m_idx_max + SEQ_LEN_KV - SEQ_LEN_Q
    return T.min(n_block_max, ceildiv(n_idx, BLK_N))


def get_n_block_min_causal_mask(m_block_idx, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE):
    """KV block index where causal masking stops being needed."""
    m_idx_min = m_block_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
    n_idx = m_idx_min + SEQ_LEN_KV - SEQ_LEN_Q
    return T.max(0, n_idx // BLK_N)


@T.inline
def ex2_emulation_2(out, idx, x, y):
    poly_ex2_deg3 = T.meta_var((1.0, 0.6951461434364319, 0.22756439447402954, 0.07711908966302872))
    fp32_round_int = T.meta_var(float(2**23 + 2**22))
    xy_clamped: T.f32[2]
    xy_clamped[0] = T.max(x, -127.0)
    xy_clamped[1] = T.max(y, -127.0)
    xy_rounded: T.f32[2]
    Tx.add(xy_rounded, xy_clamped, fp32_round_int, rounding_mode="rm")
    xy_rounded_back: T.f32[2]
    Tx.sub(xy_rounded_back, xy_rounded, fp32_round_int, rounding_mode="rn")
    xy_frac: T.f32[2]
    Tx.sub(xy_frac, xy_clamped, xy_rounded_back, rounding_mode="rn")
    xy_frac_ex2: T.f32[2]
    xy_frac_ex2[0] = poly_ex2_deg3[3]
    xy_frac_ex2[1] = poly_ex2_deg3[3]
    Tx.fma(xy_frac_ex2, xy_frac_ex2, xy_frac, poly_ex2_deg3[2])
    Tx.fma(xy_frac_ex2, xy_frac_ex2, xy_frac, poly_ex2_deg3[1])
    Tx.fma(xy_frac_ex2, xy_frac_ex2, xy_frac, poly_ex2_deg3[0])
    out[idx] = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0])
    out[idx + 1] = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1])


WG_NUMBER = 4
WARP_NUMBER = 4
NUM_THREADS = 32 * WARP_NUMBER * WG_NUMBER
N_COLS_TMEM = 512
TMEM_PIPE_DEPTH = 2
SMEM_PIPE_DEPTH_Q = 2
SMEM_PIPE_DEPTH_KV = 3
BLK_M = 128
BLK_N = 128
BLK_K = 64
SOFTMAX_LD_CHUNK = 32
SOFTMAX_ST_CHUNK = 32
EPI_TILE = 64
TMEM_EPI_LD_SIZE = 16
USE_S0_S1_BARRIER = False
MMA_M = 128
MMA_N = 128
MMA_K = 16
F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16
a_type_qk = tvm.DataType("float16")
b_type_qk = tvm.DataType("float16")
d_type_qk = tvm.DataType("float32")
a_type_pv = tvm.DataType("float16")
b_type_pv = tvm.DataType("float16")
d_type_pv = tvm.DataType("float32")


@T.jit
def _kernel(
    Q: T.Buffer((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"),
    K: T.Buffer((BATCH_SIZE, SEQ_LEN_KV, NUM_KV_HEADS, HEAD_DIM), "float16"),
    V: T.Buffer((BATCH_SIZE, SEQ_LEN_KV, NUM_KV_HEADS, HEAD_DIM), "float16"),
    O: T.Buffer((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"),
    *,
    BATCH_SIZE: T.constexpr,
    SEQ_LEN_Q: T.constexpr,
    SEQ_LEN_KV: T.constexpr,
    NUM_QO_HEADS: T.constexpr,
    NUM_KV_HEADS: T.constexpr,
    HEAD_DIM: T.constexpr,
    is_causal: T.constexpr = False,
    CTA_GROUP: T.constexpr = 1,
    TMEM_PIPE_DEPTH: T.constexpr = TMEM_PIPE_DEPTH,
    SMEM_PIPE_DEPTH_KV: T.constexpr = SMEM_PIPE_DEPTH_KV,
):
    GQA_RATIO = T.meta_var(NUM_QO_HEADS // NUM_KV_HEADS)
    SEQ_Q_PER_TILE = T.meta_var(BLK_M // GQA_RATIO)
    # Use pairwise 64-thread stats barriers for GQA=1; packed GQA keeps a
    # 256-thread collective. sScale slot reuse remains on softmax_corr.empty.
    STATS_BAR_PAIRWISE = T.meta_var(GQA_RATIO == 1)
    L2_SIZE = T.meta_var(50 * 1024 * 1024)
    SIZE_ONE_KV_HEAD = T.meta_var(SEQ_LEN_KV * HEAD_DIM * 2 * F16_BYTES)
    L2_SWIZZLE = T.meta_var(
        1 if L2_SIZE < SIZE_ONE_KV_HEAD else 1 << int(math.log2(L2_SIZE // SIZE_ONE_KV_HEAD))
    )
    SSCALE_TOTAL_SIZE = T.meta_var(2 * SMEM_PIPE_DEPTH_Q * BLK_M)
    assert TMEM_PIPE_DEPTH * MMA_N <= N_COLS_TMEM, "TMEM columns exceeded"
    num_q_blocks_total = T.meta_var(ceildiv(SEQ_LEN_Q, SEQ_Q_PER_TILE))
    num_q_blocks = T.meta_var(ceildiv(num_q_blocks_total, SMEM_PIPE_DEPTH_Q))
    num_total_tasks = T.meta_var(BATCH_SIZE * NUM_KV_HEADS * num_q_blocks)
    # Causal CTAs run one task, so their exposed tail uses stage-parallel
    # softmax epilogues; persistent non-causal CTAs overlap correction epilogues.
    EPI_ON_SOFTMAX = T.meta_var(is_causal)
    EARLY_Q_RELEASE = T.meta_var(not is_causal)
    max_ctas: T.let = 148
    cta_count: T.let = T.min(max_ctas, num_total_tasks) if not is_causal else num_total_tasks
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    bx = T.cta_id([cta_count])
    wg_id = T.warpgroup_id([4])
    warp_id = T.warp_id_in_wg([4])
    tid_in_wg = T.thread_id_in_wg([128])
    pool = T.SMEMPool()
    Q_smem = pool.alloc_tcgen05_mma_AB((SMEM_PIPE_DEPTH_Q, BLK_M, HEAD_DIM), "float16")
    K_smem = pool.alloc_tcgen05_mma_AB((SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM), "float16")
    V_smem = K_smem.view(SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM)
    O_smem = pool.alloc_tcgen05_mma_AB((TMEM_PIPE_DEPTH, BLK_M, HEAD_DIM), "float16")
    sScale = pool.alloc((SSCALE_TOTAL_SIZE,), "float32", align=1024)
    tmem_addr = pool.alloc([1], "uint32")
    ACC_SCALE_BASE: T.let = 0
    ROW_SUM_BASE: T.let = 0
    kv_pipe = PipelineState(SMEM_PIPE_DEPTH_KV)
    phase_q: T.int32
    phase_s_full: T.int32
    phase_tmem: T.int32
    phase_s0_s1: T.int32
    phase_q_load: T.int32
    phase_oepi: T.int32
    q_load = Pipeline(pool, SMEM_PIPE_DEPTH_Q, full="tma", empty="tcgen05", empty_phase_offset=1)
    kv_load = Pipeline(pool, SMEM_PIPE_DEPTH_KV, full="tma", empty="tcgen05", empty_phase_offset=1)
    p_o_rescale = MBarrier(pool, 2)
    p_o_rescale.init(256)
    s_ready = TCGen05Bar(pool, 2)
    s_ready.init(1)
    o_ready = TCGen05Bar(pool, 2)
    o_ready.init(1)
    softmax_corr = Pipeline(
        pool, 2, full="mbar", empty="mbar", init_full=128, init_empty=128, empty_phase_offset=1
    )
    corr_epi = Pipeline(
        pool,
        TMEM_PIPE_DEPTH,
        full="mbar",
        empty="mbar",
        init_full=128,
        init_empty=32,
        empty_phase_offset=1,
    )
    p_ready_2 = MBarrier(pool, 2)
    p_ready_2.init(128)
    bar_s0_s1_sequence = MBarrier(pool, 8)
    # Initialize in the first group so the single prologue fence covers it.
    bar_s0_s1_sequence.init(32)
    pool.commit()
    iket = IketProfiler()
    # MMA warp 12 allocates TMEM after the prologue sync; dependent barriers gate
    # all other users, so TMA load warps need no extra CTA-wide allocation sync.
    tmem_pool = T.TMEMPool(
        pool,
        total_cols=N_COLS_TMEM,
        cta_group=CTA_GROUP,
        tmem_addr=tmem_addr,
        alloc_warp=12,
        dealloc_warp=0,
    )
    tmem = tmem_pool.alloc((128, N_COLS_TMEM), "float32")
    tmem_pool.move_base_to(0)
    tmem_as_f16 = tmem_pool.alloc((128, N_COLS_TMEM * 2), "float16")
    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    T.cuda.cta_sync()
    # S and O share low/high MMA_N-wide TMEM stages; constant stage indexing keeps
    # the region base address at zero.
    S_region = T.meta_var(tmem.rearrange("m (s n) -> s m n", n=MMA_N))
    O_region = S_region
    # P overlays the f16 view of the S stages: the high MMA_N-wide half (two=1)
    # of each stage's 2*MMA_N f16 columns, indexed [stage, 1, :, cols].
    P_region = T.meta_var(tmem_as_f16.rearrange("m (s two n) -> s two m n", two=2, n=MMA_N))
    scheduler = (
        FlashAttentionLPTScheduler(
            "fa_scheduler",
            num_batches=BATCH_SIZE,
            num_heads=NUM_KV_HEADS,
            num_m_blocks=num_q_blocks,
            l2_swizzle=L2_SWIZZLE,
        )
        if is_causal
        else FlashAttentionLinearScheduler(
            "fa_scheduler",
            num_batches=BATCH_SIZE,
            num_heads=NUM_KV_HEADS,
            num_m_blocks=num_q_blocks,
            num_ctas=cta_count,
        )
    )
    scheduler.init(bx)
    kv_pipe.init(0)
    phase_q = 0
    phase_oepi = 0
    phase_tmem = 0
    phase_s_full = 0
    if USE_S0_S1_BARRIER:
        phase_s0_s1 = T.if_then_else(wg_id == 1, 0, 1)
    phase_q_load = 0
    tmem_pool.commit()
    if (wg_id == 3) & (warp_id == 0):
        T.cuda.trap_when_assert_failed(tmem_addr[0] == T.uint32(0))
    if wg_id == 2:
        for i_q in T.unroll(2):
            p_o_rescale.arrive(i_q)
    num_kv_blocks: T.let = ceildiv(SEQ_LEN_KV, BLK_N)
    while scheduler.valid():
        m_block_idx = T.meta_var(scheduler.m_block_idx)
        batch_idx = T.meta_var(scheduler.batch_idx)
        kv_head_idx = T.meta_var(scheduler.head_idx)
        m_start = T.meta_var(m_block_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q)
        if wg_id == 3:
            T.ptx.setmaxnreg(False, 48)
            if warp_id == 1:

                @T.inline
                def load_q(i_q):
                    q_load.empty.wait(i_q, phase_q_load)
                    tma_copy_q = T.meta_var(
                        {
                            "dispatch": "tma_auto",
                            "mbar": q_load.full.buf.ptr_to([i_q]),
                            "cta_group": CTA_GROUP,
                        }
                    )
                    tma_q_token = iket.range_start("issue-tma-q")
                    Q_smem_4d = Q_smem.view(SMEM_PIPE_DEPTH_Q, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
                    if T.ptx.elect_sync():
                        Tx.copy_async(
                            Q_smem_4d[i_q, :, :, :],
                            Q[
                                batch_idx,
                                m_start + i_q * SEQ_Q_PER_TILE : m_start
                                + (i_q + 1) * SEQ_Q_PER_TILE,
                                kv_head_idx * GQA_RATIO : (kv_head_idx + 1) * GQA_RATIO,
                                :,
                            ],
                            **tma_copy_q,
                        )
                        q_load.full.arrive(i_q, CTA_GROUP * BLK_M * HEAD_DIM * F16_BYTES)
                    iket.range_end(tma_q_token)

                @T.inline
                def load_k(i_kv):
                    kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase)
                    tma_copy_k = T.meta_var(
                        {
                            "dispatch": "tma_auto",
                            "mbar": kv_load.full.buf.ptr_to([kv_pipe.stage]),
                            "cta_group": CTA_GROUP,
                        }
                    )
                    tma_k_token = iket.range_start("issue-tma-k")
                    if T.ptx.elect_sync():
                        Tx.copy_async(
                            K_smem[kv_pipe.stage, :, :],
                            K[batch_idx, i_kv * BLK_N : (i_kv + 1) * BLK_N, kv_head_idx, :],
                            **tma_copy_k,
                        )
                        kv_load.full.arrive(kv_pipe.stage, CTA_GROUP * BLK_N * HEAD_DIM * F16_BYTES)
                    iket.range_end(tma_k_token)
                    kv_pipe.advance()

                @T.inline
                def load_v(i_kv):
                    kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase)
                    tma_copy_v = T.meta_var(
                        {
                            "dispatch": "tma_auto",
                            "mbar": kv_load.full.buf.ptr_to([kv_pipe.stage]),
                            "cta_group": CTA_GROUP,
                        }
                    )
                    tma_v_token = iket.range_start("issue-tma-v")
                    if T.ptx.elect_sync():
                        Tx.copy_async(
                            V_smem[kv_pipe.stage, :, :],
                            V[batch_idx, i_kv * BLK_N : (i_kv + 1) * BLK_N, kv_head_idx, :],
                            **tma_copy_v,
                        )
                        kv_load.full.arrive(kv_pipe.stage, CTA_GROUP * BLK_N * HEAD_DIM * F16_BYTES)
                    iket.range_end(tma_v_token)
                    kv_pipe.advance()

                load_trip_count: T.int32
                load_trip_count = (
                    get_n_block_max(m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE)
                    if is_causal
                    else num_kv_blocks
                )
                load_q(0)
                load_k(load_trip_count - 1)
                load_q(1)
                phase_q_load ^= 1
                load_v(load_trip_count - 1)
                for _i in T.serial(load_trip_count - 1, unroll=False):
                    i_kv: T.let = load_trip_count - 2 - _i
                    load_k(i_kv)
                    load_v(i_kv)
            if warp_id == 2:
                corr_epi.full.wait(0, phase_tmem)
                tma_store_token = iket.range_start("tma-store")
                for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                    if i_q != 0:
                        corr_epi.full.wait(i_q, phase_tmem)
                    m_start_global = T.meta_var(m_start + i_q * SEQ_Q_PER_TILE)
                    O_smem_4d = O_smem.view(TMEM_PIPE_DEPTH, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
                    if T.ptx.elect_sync():
                        Tx.copy_async(
                            O[
                                batch_idx,
                                m_start_global : m_start_global + SEQ_Q_PER_TILE,
                                kv_head_idx * GQA_RATIO : (kv_head_idx + 1) * GQA_RATIO,
                                :,
                            ],
                            O_smem_4d[i_q, :, :, :],
                            dispatch="tma_auto",
                        )
                    T.ptx.cp_async.bulk.commit_group()
                for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                    T.ptx.cp_async.bulk.wait_group(1 - i_q)
                    corr_epi.empty.arrive(i_q)
                iket.range_end(tma_store_token)
                phase_tmem ^= 1
            if warp_id == 0:
                acc: T.int32
                acc = 0

                @T.inline
                def gemm_qk(q_stage, kv_stage):
                    Tx.warp.gemm_async(
                        S_region[q_stage, :, :],
                        Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
                        K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
                        dispatch="tcgen05",
                        cta_group=CTA_GROUP,
                    )
                    if T.ptx.elect_sync():
                        s_ready.arrive(q_stage)

                # Use a 64-column causal PV split to release MMA earlier;
                # non-causal keeps 96 columns after small-shape regressions.
                K_SPLIT = T.meta_var((4 if is_causal else 6) * MMA_K)

                @T.inline
                def gemm_pv_part1(i_q, kv_stage, should_accumulate):
                    Tx.warp.gemm_async(
                        O_region[SMEM_PIPE_DEPTH_Q + i_q, :, :],
                        P_region[i_q, 1, :, 0:K_SPLIT],
                        V_smem[kv_stage, 0:K_SPLIT, 0:HEAD_DIM],
                        transB=True,
                        accum=should_accumulate,
                        dispatch="tcgen05",
                        cta_group=CTA_GROUP,
                    )

                @T.inline
                def gemm_pv_part2(i_q, kv_stage):
                    p_ready_2.wait(i_q, phase_tmem)
                    Tx.warp.gemm_async(
                        O_region[SMEM_PIPE_DEPTH_Q + i_q, :, :],
                        P_region[i_q, 1, :, K_SPLIT:BLK_N],
                        V_smem[kv_stage, K_SPLIT:BLK_N, 0:HEAD_DIM],
                        transB=True,
                        accum=True,
                        dispatch="tcgen05",
                        cta_group=CTA_GROUP,
                    )

                @T.inline
                def gemm_pv(i_q, kv_stage, should_accumulate):
                    gemm_pv_part1(i_q, kv_stage, should_accumulate)
                    gemm_pv_part2(i_q, kv_stage)

                for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                    q_load.full.wait(i_q, phase_q_load)
                    if i_q == 0:
                        kv_load.full.wait(kv_pipe.stage, kv_pipe.phase)
                    gemm_qk(i_q, kv_pipe.stage)
                    if i_q == 1:
                        if T.ptx.elect_sync():
                            kv_load.empty.arrive(kv_pipe.stage)
                kv_pipe.advance()
                mma_trip_count: T.int32
                mma_trip_count = (
                    get_n_block_max(m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE)
                    if is_causal
                    else num_kv_blocks
                )
                for i_kv in T.serial(mma_trip_count - 1, unroll=False):
                    stage_v: T.let = kv_pipe.stage
                    phase_v: T.let = kv_pipe.phase
                    kv_pipe.advance()
                    stage_k = T.meta_var(kv_pipe.stage)
                    phase_k = T.meta_var(kv_pipe.phase)
                    for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                        if i_q == 0:
                            kv_load.full.wait(stage_v, phase_v)
                        p_o_rescale.wait(i_q, phase_tmem)
                        gemm_pv(i_q, stage_v, acc)
                        if i_q == 1:
                            if T.ptx.elect_sync():
                                kv_load.empty.arrive(stage_v)
                        if i_q == 0:
                            kv_load.full.wait(stage_k, phase_k)
                        gemm_qk(i_q, stage_k)
                        # Non-causal CTAs release Q after its final QK reader to overlap
                        # the next load; causal one-task/one-trip cases keep tail release.
                        if EARLY_Q_RELEASE:
                            if i_kv == mma_trip_count - 2:
                                if T.ptx.elect_sync():
                                    q_load.empty.arrive(i_q)
                        if i_q == 1:
                            if T.ptx.elect_sync():
                                kv_load.empty.arrive(stage_k)
                    acc = 1
                    kv_pipe.advance()
                    phase_tmem ^= 1
                for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                    if i_q == 0:
                        kv_load.full.wait(kv_pipe.stage, kv_pipe.phase)
                    p_o_rescale.wait(i_q, phase_tmem)
                    gemm_pv(i_q, kv_pipe.stage, acc)
                    if i_q == 1:
                        if T.ptx.elect_sync():
                            kv_load.empty.arrive(kv_pipe.stage)
                    if T.ptx.elect_sync():
                        o_ready.arrive(i_q)
                kv_pipe.advance()
                phase_tmem ^= 1
                if not EARLY_Q_RELEASE:
                    for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                        if T.ptx.elect_sync():
                            q_load.empty.arrive(i_q)
                phase_q_load ^= 1
        elif wg_id < 2:
            T.ptx.setmaxnreg(True, 200)
            scale_log2 = T.meta_var(math.log2(math.e) / math.sqrt(HEAD_DIM))
            rescale_threshold = T.meta_var(8.0)
            row_max: T.f32[1]
            row_sum: T.f32[1]
            if warp_id == 0:
                iket.mark("softmax-baseline")

            @T.inline
            def mask_r2p(s_chunk_buf, col_limit, ncol: T.int32):
                """Apply a 32-column mask using R2P-style bit manipulation."""
                ncol = T.meta_var(ncol)
                # The subtract-free low-k mask uses PTX shl clamping and ANDN to avoid
                # VIADD/VIMNMX before the bit test compiles to R2P.
                CHUNK_SIZE: T.let = 32
                num_chunks: T.let = ceildiv(ncol, CHUNK_SIZE)
                s_chunk_local = s_chunk_buf.local(ncol)
                for s in T.unroll(num_chunks):
                    k_keep: T.let = T.max(col_limit - s * CHUNK_SIZE, 0)
                    mask_inv: T.uint32
                    mask_inv = T.ptx.shl(
                        T.uint32(0xFFFFFFFF), T.cast(k_keep, "uint32"), ptx_type="b32"
                    )
                    for i in T.unroll(CHUNK_SIZE):
                        if i < ncol - s * CHUNK_SIZE:
                            c: T.let = s * CHUNK_SIZE + i
                            in_bound: T.let = T.bitwise_and(
                                T.bitwise_not(mask_inv), T.shift_left(T.uint32(1), i)
                            )
                            s_chunk_local[c] = T.Select(
                                T.cast(in_bound, "bool"), s_chunk_local[c], T.float32(-float("inf"))
                            )

            @T.inline
            def apply_causal_mask(s_chunk_buf, m_blk_idx, n_blk_idx):
                """Map packed GQA rows to their causal limit and apply the R2P mask."""
                seq_pos_in_wg: T.let = tid_in_wg // GQA_RATIO
                row_idx: T.let = (
                    m_blk_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
                    + wg_id * SEQ_Q_PER_TILE
                    + seq_pos_in_wg
                )
                causal_row_offset: T.let = 1 + SEQ_LEN_KV - n_blk_idx * BLK_N - SEQ_LEN_Q
                col_limit_right: T.let = row_idx + causal_row_offset
                mask_r2p(s_chunk_buf, col_limit_right, BLK_N)

            @T.inline
            def softmax_step(i_kv, apply_mask=False, is_first=False):
                s_chunk_buf: T.f32[BLK_N]
                s_chunk = s_chunk_buf.view(128, BLK_N, layout=wg_local_layout(BLK_N))
                p_chunk_buf_f32: T.f32[BLK_N // 2]
                p_chunk_buf = T.decl_buffer((BLK_N,), dtype="float16", data=p_chunk_buf_f32.data)
                p_chunk = p_chunk_buf.view(128, BLK_N, layout=wg_local_layout(BLK_N))
                s_ready.wait(wg_id, phase_s_full)
                if warp_id == 0:
                    iket.mark("softmax-phase-0")
                softmax_max_token = iket.sentinel_token("softmax-max")
                if warp_id == 0:
                    softmax_max_token = iket.range_start("softmax-max")
                tile_max: T.f32[1]
                for chunk_idx in T.unroll(BLK_N // SOFTMAX_LD_CHUNK):
                    Tx.wg.copy_async(
                        s_chunk[
                            :, chunk_idx * SOFTMAX_LD_CHUNK : (chunk_idx + 1) * SOFTMAX_LD_CHUNK
                        ],
                        S_region[
                            wg_id,
                            :,
                            chunk_idx * SOFTMAX_LD_CHUNK : (chunk_idx + 1) * SOFTMAX_LD_CHUNK,
                        ],
                    )
                if apply_mask:
                    apply_causal_mask(s_chunk_buf, m_block_idx, i_kv)
                row_max_old: T.f32
                row_max_old = row_max[0]
                if is_first:
                    Tx.max(tile_max, s_chunk_buf)
                else:
                    tile_max[0] = row_max_old
                    Tx.max(tile_max, s_chunk_buf, accum=True)
                row_max_new: T.f32
                acc_scale: T.f32
                acc_scale_: T.f32
                row_max_safe: T.f32
                row_max_new = tile_max[0]
                row_max_safe = T.if_then_else(tile_max[0] == -float("inf"), 0.0, tile_max[0])
                if is_first:
                    acc_scale = T.float32(1.0)
                else:
                    acc_scale_ = (row_max_old - row_max_safe) * scale_log2
                    if acc_scale_ >= -rescale_threshold:
                        row_max_new = row_max_old
                        row_max_safe = row_max_old
                        acc_scale = T.float32(1.0)
                    else:
                        acc_scale = T.ptx.exp2(acc_scale_)
                row_max[0] = row_max_new
                row_max_scaled: T.let = row_max_safe * scale_log2
                if warp_id == 0:
                    iket.mark("softmax-phase-1")
                iket.range_end(softmax_max_token)
                if tid_in_wg < BLK_M and (not is_first):
                    sScale_idx: T.let = ACC_SCALE_BASE + tid_in_wg + wg_id * BLK_M
                    sScale[sScale_idx] = acc_scale
                # Signal correction with pairwise/collective HW barriers;
                # sScale slot reuse remains on softmax_corr.empty.
                if STATS_BAR_PAIRWISE:
                    tvm.backend.cuda.op.ptx_bar_arrive(1 + wg_id * 4 + warp_id, 64)
                else:
                    tvm.backend.cuda.op.ptx_bar_arrive(1 + wg_id, 256)
                softmax_fma_token = iket.sentinel_token("softmax-fma")
                if warp_id == 0:
                    softmax_fma_token = iket.range_start("softmax-fma")
                Tx.wg.fma(s_chunk, s_chunk, scale_log2, -row_max_scaled)
                iket.range_end(softmax_fma_token)
                if USE_S0_S1_BARRIER:
                    bar_s0_s1_sequence.wait(wg_id * 4 + warp_id, phase_s0_s1)
                softmax_exp2_token = iket.sentinel_token("softmax-exp2")
                if warp_id == 0:
                    softmax_exp2_token = iket.range_start("softmax-exp2")
                for frag_idx in T.unroll(4):
                    s_chunk_local = s_chunk_buf.local(BLK_N)
                    for i in T.unroll(BLK_N // 4 // 2):
                        idx = T.meta_var(frag_idx * BLK_N // 4 + 2 * i)
                        # Select the shape-tuned ex2 window from the module-level knobs.
                        emu_pairs = T.meta_var(EMU_PAIRS_CAUSAL if is_causal else EMU_PAIRS_NC)
                        emu_start = T.meta_var(EMU_START_CAUSAL if is_causal else EMU_START_NC)
                        if (
                            i * 2 % 16 < 16 - 2 * emu_pairs
                            or frag_idx >= 4 - 1
                            or frag_idx < emu_start
                            or apply_mask
                        ):
                            s_chunk_local[idx] = T.ptx.exp2(s_chunk_local[idx])
                            s_chunk_local[idx + 1] = T.ptx.exp2(s_chunk_local[idx + 1])
                        else:
                            ex2_emulation_2(
                                s_chunk_local, idx, s_chunk_local[idx], s_chunk_local[idx + 1]
                            )
                    Tx.wg.cast(
                        p_chunk[:, frag_idx * BLK_N // 4 : (frag_idx + 1) * BLK_N // 4],
                        s_chunk[:, frag_idx * BLK_N // 4 : (frag_idx + 1) * BLK_N // 4],
                    )
                if USE_S0_S1_BARRIER:
                    bar_s0_s1_sequence.arrive((1 - wg_id) * 4 + warp_id)
                iket.range_end(softmax_exp2_token)
                softmax_tmem_st_token = iket.sentinel_token("softmax-tmem-st")
                if warp_id == 0:
                    softmax_tmem_st_token = iket.range_start("softmax-tmem-st")
                P_SPLIT_Q = T.meta_var(2 if is_causal else 3)
                for i in T.unroll(P_SPLIT_Q):
                    Tx.wg.copy_async(
                        P_region[wg_id, 1, :, i * BLK_N // 4 : (i + 1) * BLK_N // 4],
                        p_chunk[:, i * BLK_N // 4 : (i + 1) * BLK_N // 4],
                    )
                T.ptx.tcgen05.wait.st()
                p_o_rescale.arrive(wg_id)
                for i in T.unroll(4 - P_SPLIT_Q):
                    Tx.wg.copy_async(
                        P_region[
                            wg_id,
                            1,
                            :,
                            (P_SPLIT_Q + i) * BLK_N // 4 : (P_SPLIT_Q + 1 + i) * BLK_N // 4,
                        ],
                        p_chunk[:, (P_SPLIT_Q + i) * BLK_N // 4 : (P_SPLIT_Q + 1 + i) * BLK_N // 4],
                    )
                if warp_id == 0:
                    iket.mark("softmax-phase-2")
                T.ptx.tcgen05.wait.st()
                p_ready_2.arrive(wg_id)
                if warp_id == 0:
                    iket.mark("softmax-phase-3")
                iket.range_end(softmax_tmem_st_token)
                softmax_corr.empty.wait(wg_id, phase_q)
                if warp_id == 0:
                    iket.mark("softmax-phase-4")
                softmax_sum_token = iket.sentinel_token("softmax-sum")
                if warp_id == 0:
                    softmax_sum_token = iket.range_start("softmax-sum")
                phase_s_full ^= 1
                phase_q ^= 1
                if is_first:
                    Tx.sum(row_sum, s_chunk_buf)
                else:
                    row_sum[0] = row_sum[0] * acc_scale
                    Tx.sum(row_sum, s_chunk_buf, accum=True)
                if warp_id == 0:
                    iket.mark("softmax-phase-5")
                iket.range_end(softmax_sum_token)
                if USE_S0_S1_BARRIER:
                    phase_s0_s1 ^= 1

            # Guard the first sScale write against the previous task's tail read;
            # causal softmax epilogues have neither and skip this wait.
            if not EPI_ON_SOFTMAX:
                softmax_corr.empty.wait(wg_id, phase_q)
            # Flip phase even without the wait so step 1 consumes the prologue credit;
            # starting at phase 0 can outrun the non-sticky credit and deadlock.
            phase_q ^= 1
            n_block_max: T.let = get_n_block_max(
                m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE
            )
            n_block_min_causal: T.let = (
                get_n_block_min_causal_mask(m_block_idx, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE)
                if is_causal
                else n_block_max
            )
            softmax_step(n_block_max - 1, apply_mask=is_causal, is_first=True)
            n_block_max_after_p1: T.let = n_block_max - 1
            num_phase2_blocks: T.let = T.max(n_block_max_after_p1 - n_block_min_causal, 0)
            for i in T.serial(num_phase2_blocks, unroll=False):
                n_block: T.let = n_block_max_after_p1 - 1 - i
                softmax_step(n_block, apply_mask=True)
            n_block_max_after_p2: T.let = T.min(n_block_max_after_p1, n_block_min_causal)
            for i in T.serial(n_block_max_after_p2, unroll=False):
                n_block: T.let = n_block_max_after_p2 - 1 - i
                softmax_step(n_block, apply_mask=False)
            if EPI_ON_SOFTMAX:
                # Run each causal stage epilogue on its softmax WG with
                # register-resident row sums and 32-wide TMEM loads.
                EPI_LD_SM = T.meta_var(32)
                o_ready.wait(wg_id, phase_oepi)
                corr_epi.empty.wait(wg_id, phase_oepi)
                epi_ld_tmem_token = iket.sentinel_token("epi-ld-tmem")
                if warp_id == 0:
                    epi_ld_tmem_token = iket.range_start("epi-ld-tmem")
                acc_O_row_is_zero_or_nan: T.let = tvm.tirx.any(
                    row_sum[0] == T.float32(0.0), row_sum[0] != row_sum[0]
                )
                norm_scale_sm: T.let = T.ptx.rcp(
                    T.Select(acc_O_row_is_zero_or_nan, T.float32(1.0), row_sum[0])
                )
                o_row_f32_sm = T.wg_reg_tile(EPI_LD_SM)
                o_row_f16_sm = T.wg_reg_tile(EPI_LD_SM, "float16")
                for epi_q in T.unroll(2):
                    if wg_id == epi_q:
                        for d_tile in T.unroll(ceildiv(HEAD_DIM, EPI_LD_SM)):
                            Tx.wg.copy_async(
                                o_row_f32_sm,
                                O_region[
                                    SMEM_PIPE_DEPTH_Q + epi_q,
                                    :,
                                    d_tile * EPI_LD_SM : (d_tile + 1) * EPI_LD_SM,
                                ],
                            )
                            Tx.wg.mul(o_row_f32_sm, o_row_f32_sm, norm_scale_sm)
                            Tx.wg.cast(o_row_f16_sm, o_row_f32_sm)
                            Tx.wg.copy(
                                O_smem[
                                    epi_q, 0:BLK_M, d_tile * EPI_LD_SM : (d_tile + 1) * EPI_LD_SM
                                ],
                                o_row_f16_sm,
                                vec_len=8,
                            )
                iket.range_end(epi_ld_tmem_token)
                T.ptx.fence.proxy_async("shared::cta")
                corr_epi.full.arrive(wg_id)
                p_o_rescale.arrive(wg_id)
                phase_oepi ^= 1
            else:
                if tid_in_wg < BLK_M:
                    sScale[ROW_SUM_BASE + tid_in_wg + wg_id * BLK_M] = row_sum[0]
                if STATS_BAR_PAIRWISE:
                    tvm.backend.cuda.op.ptx_bar_arrive(1 + wg_id * 4 + warp_id, 64)
                else:
                    tvm.backend.cuda.op.ptx_bar_arrive(1 + wg_id, 256)
        if wg_id == 2:
            T.ptx.setmaxnreg(False, 64)
            if STATS_BAR_PAIRWISE:
                tvm.backend.cuda.op.ptx_bar_sync(1 + 0 * 4 + warp_id, 64)
            else:
                tvm.backend.cuda.op.ptx_bar_sync(1 + 0, 256)
            softmax_corr.empty.arrive(0)
            if STATS_BAR_PAIRWISE:
                tvm.backend.cuda.op.ptx_bar_sync(1 + 1 * 4 + warp_id, 64)
            else:
                tvm.backend.cuda.op.ptx_bar_sync(1 + 1, 256)
            phase_q ^= 1
            corr_trip_count: T.let = (
                get_n_block_max(m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE)
                if is_causal
                else num_kv_blocks
            )
            for i_kv in T.serial(corr_trip_count - 1, unroll=False):
                for i_q in T.unroll(2):
                    if STATS_BAR_PAIRWISE:
                        tvm.backend.cuda.op.ptx_bar_sync(1 + i_q * 4 + warp_id, 64)
                    else:
                        tvm.backend.cuda.op.ptx_bar_sync(1 + i_q, 256)
                    correction_token = iket.sentinel_token("correction")
                    if warp_id == 0:
                        correction_token = iket.range_start("correction")
                    acc_scale: T.f32
                    should_rescale: T.i32
                    if tid_in_wg < BLK_M:
                        acc_scale = sScale[ACC_SCALE_BASE + tid_in_wg + i_q * BLK_M]
                        should_rescale = T.Select(acc_scale < T.float32(1.0), 1, 0)
                    else:
                        should_rescale = 0
                    any_needs_rescale: T.let = T.ptx.any_sync(4294967295, should_rescale)
                    if any_needs_rescale != 0:
                        if tid_in_wg < BLK_M:
                            RESCALE_TILE = T.meta_var(16)
                            o_row = T.wg_reg_tile(RESCALE_TILE)
                            for d_tile in T.unroll(ceildiv(HEAD_DIM, RESCALE_TILE)):
                                d_start: T.let = d_tile * RESCALE_TILE
                                if d_start < HEAD_DIM:
                                    Tx.wg.copy_async(
                                        o_row,
                                        O_region[
                                            SMEM_PIPE_DEPTH_Q + i_q,
                                            :,
                                            d_start : d_start + RESCALE_TILE,
                                        ],
                                    )
                                    Tx.wg.mul(o_row, o_row, acc_scale)
                                    Tx.wg.copy_async(
                                        O_region[
                                            SMEM_PIPE_DEPTH_Q + i_q,
                                            :,
                                            d_start : d_start + RESCALE_TILE,
                                        ],
                                        o_row,
                                    )
                            T.ptx.tcgen05.wait.st()
                    p_o_rescale.arrive(i_q)
                    softmax_corr.empty.arrive(1 - i_q)
                    iket.range_end(correction_token)
                phase_q ^= 1
            softmax_corr.empty.arrive(1)
            if not EPI_ON_SOFTMAX:
                for i_q in T.unroll(2):
                    if STATS_BAR_PAIRWISE:
                        tvm.backend.cuda.op.ptx_bar_sync(1 + i_q * 4 + warp_id, 64)
                    else:
                        tvm.backend.cuda.op.ptx_bar_sync(1 + i_q, 256)
                    row_sum: T.let = sScale[ROW_SUM_BASE + tid_in_wg + i_q * BLK_M]
                    softmax_corr.empty.arrive(i_q)
                    o_ready.wait(i_q, phase_tmem)
                    corr_epi.empty.wait(i_q, phase_tmem)
                    epi_ld_tmem_token = iket.sentinel_token("epi-ld-tmem")
                    if warp_id == 0:
                        epi_ld_tmem_token = iket.range_start("epi-ld-tmem")
                    acc_O_mn_row_is_zero_or_nan: T.let = tvm.tirx.any(
                        row_sum == T.float32(0.0), row_sum != row_sum
                    )
                    norm_scale: T.let = T.ptx.rcp(
                        T.Select(acc_O_mn_row_is_zero_or_nan, T.float32(1.0), row_sum)
                    )
                    o_row_f32 = T.wg_reg_tile(TMEM_EPI_LD_SIZE)
                    o_row_f16 = T.wg_reg_tile(TMEM_EPI_LD_SIZE, "float16")
                    for d_tile in T.unroll(ceildiv(HEAD_DIM, TMEM_EPI_LD_SIZE)):
                        d_start: T.let = d_tile * TMEM_EPI_LD_SIZE
                        if d_start < HEAD_DIM:
                            Tx.wg.copy_async(
                                o_row_f32,
                                O_region[
                                    SMEM_PIPE_DEPTH_Q + i_q, :, d_start : d_start + TMEM_EPI_LD_SIZE
                                ],
                            )
                            Tx.wg.mul(o_row_f32, o_row_f32, norm_scale)
                            Tx.wg.cast(o_row_f16, o_row_f32)
                            Tx.wg.copy(
                                O_smem[
                                    i_q,
                                    0:BLK_M,
                                    d_tile * TMEM_EPI_LD_SIZE : d_tile * TMEM_EPI_LD_SIZE
                                    + TMEM_EPI_LD_SIZE,
                                ],
                                o_row_f16,
                                vec_len=8,
                            )
                    iket.range_end(epi_ld_tmem_token)
                    T.ptx.fence.proxy_async("shared::cta")
                    corr_epi.full.arrive(i_q)
                    p_o_rescale.arrive(i_q)
                phase_tmem ^= 1
            phase_q ^= 1
        scheduler.next_tile()
    tmem_pool.dealloc()
    # No final CTA sync is required after TMEM deallocation; warps exit independently.


def get_flash_attention4_kernel(
    batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim, is_causal=False
):
    # Shape-tuned ptxas levels avoid spills for causal GQA=1.
    # Use 4 at s1024 and 5 otherwise; FA4_REG_LEVEL overrides this selection.
    _reg_override = os.environ.get("FA4_REG_LEVEL", "")
    if _reg_override:
        _reg_level = _reg_override
    elif is_causal and num_qo_heads == num_kv_heads:
        _reg_level = "4" if seq_len_q <= 1024 else "5"
    else:
        _reg_level = "10"
    os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = _reg_level
    # Single-wave causal s1024 uses TMEM/KV depths 3/2; multi-wave shapes keep 2/3.
    # KV depth 1 deadlocks, while 3/2 regresses multi-wave cases.
    _deep_o = is_causal and seq_len_q <= 1024
    _tmem_depth = 3 if _deep_o else TMEM_PIPE_DEPTH
    _kv_depth = 2 if _deep_o else SMEM_PIPE_DEPTH_KV
    return _kernel.specialize(
        BATCH_SIZE=batch_size,
        SEQ_LEN_Q=seq_len_q,
        SEQ_LEN_KV=seq_len_kv,
        NUM_QO_HEADS=num_qo_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        is_causal=is_causal,
        TMEM_PIPE_DEPTH=_tmem_depth,
        SMEM_PIPE_DEPTH_KV=_kv_depth,
    )


def prepare_data(batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim):
    torch.manual_seed(0)
    Q = torch.randn((batch_size, seq_len_q, num_qo_heads, head_dim), dtype=torch.float16)
    K = torch.randn((batch_size, seq_len_kv, num_kv_heads, head_dim), dtype=torch.float16)
    V = torch.randn((batch_size, seq_len_kv, num_kv_heads, head_dim), dtype=torch.float16)
    O = torch.zeros((batch_size, seq_len_q, num_qo_heads, head_dim), dtype=torch.float16)
    return (Q, K, V, O)


KERNEL_META = {"name": "flash_attention4", "category": "attention", "compute_capability": 10}
CONFIGS = [
    {
        "batch_size": 1,
        "seq_len": sl,
        "num_qo_heads": 32,
        "num_kv_heads": kv,
        "head_dim": 128,
        "is_causal": causal,
        "label": f"s{sl}_h32kv{kv}{('_causal' if causal else '')}",
    }
    for sl in [1024, 2048, 4096, 8192]
    for kv in [4, 8, 16, 32]
    for causal in [False, True]
]


def get_kernel(
    batch_size, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=False, **kwargs
):
    return get_flash_attention4_kernel(
        batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=is_causal
    )


def run_test(batch_size, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=False, **kwargs):
    """Compile, run, and verify flash attention 4 kernel."""
    from tirx_kernels.runner import compile_kernel

    Q, K, V, _ = prepare_data(batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim)
    prim_func = get_flash_attention4_kernel(
        batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=is_causal
    )
    ex = compile_kernel(prim_func)
    Q_tir = Q.cuda()
    K_tir = K.cuda()
    V_tir = V.cuda()
    O_tir = torch.empty(
        (batch_size, seq_len, num_qo_heads, head_dim), dtype=torch.float16, device="cuda"
    )
    ex(Q_tir, K_tir, V_tir, O_tir)
    torch.cuda.synchronize()
    Q_t = Q.float().transpose(1, 2)
    K_t = K.float().transpose(1, 2)
    V_t = V.float().transpose(1, 2)
    if num_qo_heads != num_kv_heads:
        repeat_factor = num_qo_heads // num_kv_heads
        K_t = K_t.repeat_interleave(repeat_factor, dim=1)
        V_t = V_t.repeat_interleave(repeat_factor, dim=1)
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) * scale
    if is_causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    ref = torch.matmul(attn, V_t).transpose(1, 2).to(torch.float16)
    np.testing.assert_allclose(O_tir.cpu().numpy(), ref.cpu().numpy(), rtol=0.01, atol=0.01)


def run_bench(
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
    warmup=None,
    repeat=None,
    timer=None,  # None inherits the global default (proton); the CuTeDSL flashattn
    # reference cannot be CUDA-graph-captured, so proton (not cudagraph_proton) is what
    # gives an honest ratio here (verified 0.994 vs event's unstable 0.97-1.38).
    **kwargs,
):
    """Benchmark flash attention 4."""
    from tirx_kernels.runner import compile_kernel

    prim_func = get_flash_attention4_kernel(
        batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=is_causal
    )
    ex = compile_kernel(prim_func)

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    Q, K, V, _ = prepare_data(batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim)
    Q_cuda = Q.cuda()
    K_cuda = K.cuda()
    V_cuda = V.cuda()
    O_tir = torch.empty(
        (batch_size, seq_len, num_qo_heads, head_dim), dtype=torch.float16, device="cuda"
    )
    funcs = {"tir": lambda: ex(Q_cuda, K_cuda, V_cuda, O_tir)}

    def _flashattn_sm100():
        # Build the CuTeDSL reference through TVM-FFI, converting tensors before compile.
        # The run closure passes Torch tensors directly and creates no new CuTe wrappers.
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        from flash_attn.cute.cute_dsl_utils import to_cute_tensor
        from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
        from flash_attn.cute.utils import AuxData

        Qi, Ki, Vi, _ = prepare_data(
            batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim
        )
        Qf = Qi.cuda().contiguous()
        Kf = Ki.cuda().contiguous()
        Vf = Vi.cuda().contiguous()
        Of = torch.zeros_like(Qf)
        q_t = to_cute_tensor(Qf)
        k_t = to_cute_tensor(Kf)
        v_t = to_cute_tensor(Vf)
        o_t = to_cute_tensor(Of)

        fa_fwd = FlashAttentionForwardSm100(
            head_dim=head_dim,
            head_dim_v=head_dim,
            qhead_per_kvhead=num_qo_heads // num_kv_heads,
            is_causal=is_causal,
            is_local=False,
            pack_gqa=False,
            m_block_size=128,
            n_block_size=128,
            is_persistent=True,
        )
        _stream_fa = cutlass_torch.default_stream()
        _scale_fa = 1.0 / math.sqrt(head_dim)
        compiled_fa = cute.compile(
            fa_fwd,
            q_t,
            k_t,
            v_t,
            o_t,
            None,  # mLSE
            _scale_fa,  # softmax_scale
            None,  # mCuSeqlensQ
            None,  # mCuSeqlensK
            None,  # mSeqUsedQ
            None,  # mSeqUsedK
            None,  # mPageTable
            None,  # window_size_left
            None,  # window_size_right
            None,  # learnable_sink
            None,  # descale_tensors
            None,  # blocksparse_tensors
            AuxData(),  # aux_data
            _stream_fa,  # stream (compile-time only under tvm-ffi)
            options="--enable-tvm-ffi",
        )

        def run():
            compiled_fa(
                Qf,
                Kf,
                Vf,
                Of,
                None,  # mLSE
                _scale_fa,
                None,  # mCuSeqlensQ
                None,  # mCuSeqlensK
                None,  # mSeqUsedQ
                None,  # mSeqUsedK
                None,  # mPageTable
                None,  # window_size_left
                None,  # window_size_right
                None,  # learnable_sink
                None,  # descale_tensors
                None,  # blocksparse_tensors
                AuxData(),  # aux_data
            )

        # Keep compile-time CuTe wrappers and backing Torch storage alive with the closure.
        run._fa_keep_alive = (q_t, k_t, v_t, o_t, Qf, Kf, Vf, Of)
        return run

    return bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashattn_sm100": _flashattn_sm100},
        **kwargs,
    )


def _parse_iket_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the annotated FA4 kernel with NVIDIA IKET"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of traced FA4 launches; setup and compilation remain outside the loop",
    )
    parser.add_argument("--output-dir", default="/tmp/fa4-iket")
    parser.add_argument(
        "--postprocess", choices=("perfetto", "json", "html", "none", "all"), default="all"
    )
    parser.add_argument("--clobber", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--keep", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-ts-cnt-per-warp", type=int, default=None)
    return parser.parse_args()


def _profile_iket_workload(args: argparse.Namespace) -> None:
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    func = get_flash_attention4_kernel(
        args.batch_size,
        args.seq_len,
        args.seq_len,
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
        is_causal=args.causal,
    )
    executable = IketProfiler().compile(
        tvm.IRModule({"main": func}),
        target=tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}),
        tir_pipeline="tirx",
    )

    q, k, v, _ = prepare_data(
        args.batch_size,
        args.seq_len,
        args.seq_len,
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
    )
    q, k, v = q.cuda(), k.cuda(), v.cuda()
    out = torch.empty(
        (args.batch_size, args.seq_len, args.num_qo_heads, args.head_dim),
        dtype=torch.float16,
        device="cuda",
    )

    for _ in range(args.repeat):
        executable(q, k, v, out)
    torch.cuda.synchronize()


def _print_iket_result(result: iket.IketProfileResult) -> None:
    print(f"IKET output directory: {result.output_dir}")
    for path in (*result.json_traces, *result.perfetto_traces, *result.html_reports):
        print(f"IKET artifact: {path}")


def main() -> None:
    """Profile FA4 when this kernel module is executed directly."""
    args = _parse_iket_args()
    result = iket.run(
        partial(_profile_iket_workload, args),
        output_dir=args.output_dir,
        postprocess=args.postprocess,
        clobber=args.clobber,
        timeout=args.timeout,
        keep=args.keep,
        max_ts_cnt_per_warp=args.max_ts_cnt_per_warp,
    )
    _print_iket_result(result)


if __name__ == "__main__":
    main()
