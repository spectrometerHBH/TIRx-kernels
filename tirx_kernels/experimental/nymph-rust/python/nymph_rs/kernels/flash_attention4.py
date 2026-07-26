"""FlashAttention4-shaped protocol kernel expressed in Nymph IR.

The kernel keeps the tirx-kernels FA4 stream shape under the per-warp
execution model: one custom scheduler stream (warp 15, lane 0) broadcasts task
metadata through a two-stage SMEM mailbox, and six consumer streams drain each
task (TMA load, MMA, two softmax warpgroups, correction epilogue, TMA store).
mbarrier.arrive is per-thread, so the mailbox mbarrier count is derived from
the consumer composition below as one arrival per mailbox-reading thread —
adding a consumer cannot silently leave task_empty with an obsolete count.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ..builder import IRBuilder, TmemBand
from ..nymph_rs import (
    DType,
    FenceKind,
    FenceScope,
    Kernel,
    LaunchShape,
    MBar,
    MBarKind,
    MemorySpace,
    ScalarDType,
    SmemSwizzleLayout,
    Swizzle,
    Tensor,
    TensorSlice,
)

BLK_M = 128
BLK_N = 128
HEAD_DIM = 128
MMA_K = 16
SMEM_PIPE_DEPTH_Q = 2
SMEM_PIPE_DEPTH_KV = 3
TMEM_PIPE_DEPTH = 2
N_COLS_TMEM = 512
# Softmax consumes one 128-column score tile as four 32-cell register chunks.
SOFTMAX_CHUNK_CELLS = 32
SOFTMAX_NUM_CHUNKS = BLK_N // SOFTMAX_CHUNK_CELLS
P_STORE_CELLS = SOFTMAX_CHUNK_CELLS // 2
# The PV MMA role starts after the first 6 k-groups are stored to p_tmem. This
# matches the baseline overlap point: 6 groups * MMA_K cells covers the first
# three 32-cell softmax chunks, so p_first_ready fires after chunk index 2.
PV_SPLIT_GROUPS = 6
P_FIRST_READY_CHUNKS = (PV_SPLIT_GROUPS * MMA_K) // SOFTMAX_CHUNK_CELLS
O_CHUNK_CELLS = 16
F16_BYTES = 2
F32_BYTES = 4
I32_BYTES = 4
TASK_BROADCAST_STAGES = 2
WARPGROUP_THREADS = 128
# (role, streams, mailbox-reading threads per stream). mbarrier.arrive is
# per-thread under the per-warp execution model: single-thread streams
# contribute one task_empty arrival per slot; warpgroup streams let every
# thread arrive after its own mailbox read, so the scheduler cannot overwrite
# a slot a lagging warp of the group is still reading.
TASK_CONSUMER_ROLES = (
    ("tma_load", 1, 1),
    ("mma", 1, 1),
    ("softmax", SMEM_PIPE_DEPTH_Q, WARPGROUP_THREADS),
    ("correction_epilogue", 1, WARPGROUP_THREADS),
    ("tma_store", 1, 1),
)
TASK_CONSUMER_COUNT = sum(streams * threads for _, streams, threads in TASK_CONSUMER_ROLES)
TASK_BROADCAST_FIELDS_DEF = (
    "task_id",
    "task_iter",
    "pipe_base",
    "kv_stage0_base",
    "kv_stage1_base",
    "kv_stage2_base",
    "m_block_idx",
    "head_idx",
    "batch_idx",
)
TASK_BROADCAST_FIELDS = len(TASK_BROADCAST_FIELDS_DEF)
(
    TASK_FIELD_ID,
    TASK_FIELD_ITER,
    TASK_FIELD_PIPE_BASE,
    TASK_FIELD_KV_STAGE0_BASE,
    TASK_FIELD_KV_STAGE1_BASE,
    TASK_FIELD_KV_STAGE2_BASE,
    TASK_FIELD_M_BLOCK,
    TASK_FIELD_HEAD,
    TASK_FIELD_BATCH,
) = range(TASK_BROADCAST_FIELDS)


@dataclass(frozen=True, slots=True)
class FlashAttention4Config:
    batch_size: int = 1
    seq_len: int = 1024
    num_qo_heads: int = 32
    num_kv_heads: int = 4
    head_dim: int = HEAD_DIM
    is_causal: bool = False
    cta_group: int = 1
    num_warps: int = 16
    launch_shape: LaunchShape | None = None


CONFIGS = [
    {
        "batch_size": 1,
        "seq_len": seq_len,
        "num_qo_heads": 32,
        "num_kv_heads": num_kv_heads,
        "head_dim": HEAD_DIM,
        "is_causal": False,
        "label": f"s{seq_len}_h32kv{num_kv_heads}",
    }
    for seq_len in [1024, 2048, 4096, 8192]
    for num_kv_heads in [4, 8, 16, 32]
]


def _assert_task_broadcast_contract() -> None:
    field_ids = (
        TASK_FIELD_ID,
        TASK_FIELD_ITER,
        TASK_FIELD_PIPE_BASE,
        TASK_FIELD_KV_STAGE0_BASE,
        TASK_FIELD_KV_STAGE1_BASE,
        TASK_FIELD_KV_STAGE2_BASE,
        TASK_FIELD_M_BLOCK,
        TASK_FIELD_HEAD,
        TASK_FIELD_BATCH,
    )
    if field_ids != tuple(range(TASK_BROADCAST_FIELDS)):
        raise AssertionError("task broadcast field ids must be dense and match field count")
    if TASK_BROADCAST_FIELDS != len(TASK_BROADCAST_FIELDS_DEF):
        raise AssertionError("task broadcast field count must match field definitions")
    if TASK_CONSUMER_COUNT != sum(streams * threads for _, streams, threads in TASK_CONSUMER_ROLES):
        raise AssertionError("task_empty arrive count must match task consumer role composition")


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def build_flash_attention4(config: FlashAttention4Config = FlashAttention4Config()) -> Kernel:
    _validate_config(config)
    _assert_task_broadcast_contract()
    gqa_ratio = config.num_qo_heads // config.num_kv_heads
    seq_q_per_tile = BLK_M // gqa_ratio
    num_q_blocks_total = _ceil_div(config.seq_len, seq_q_per_tile)
    num_q_blocks = _ceil_div(num_q_blocks_total, SMEM_PIPE_DEPTH_Q)
    num_kv_blocks = _ceil_div(config.seq_len, BLK_N)
    task_count = config.batch_size * config.num_kv_heads * num_q_blocks
    launch_shape = config.launch_shape or (min(148, task_count),)
    _validate_launch_shape(launch_shape, config.cta_group)

    q_tile_bytes = BLK_M * HEAD_DIM * F16_BYTES
    kv_tile_bytes = BLK_N * HEAD_DIM * F16_BYTES
    o_tile_bytes = BLK_M * HEAD_DIM * F16_BYTES
    q_bytes = SMEM_PIPE_DEPTH_Q * q_tile_bytes
    kv_bytes = SMEM_PIPE_DEPTH_KV * kv_tile_bytes
    o_bytes = TMEM_PIPE_DEPTH * o_tile_bytes
    scale_bytes = 4 * BLK_M * F32_BYTES
    q_offset = 0
    kv_offset = q_offset + q_bytes
    o_offset = kv_offset + kv_bytes
    scale_offset = o_offset + o_bytes
    task_offset = scale_offset + scale_bytes
    tensor_end = task_offset + TASK_BROADCAST_STAGES * TASK_BROADCAST_FIELDS * I32_BYTES
    mbar_stages = [
        *((f"q_full_{stage}", 1) for stage in range(2)),
        *((f"q_empty_{stage}", 1) for stage in range(2)),
        ("k_full", SMEM_PIPE_DEPTH_KV),
        ("k_empty", SMEM_PIPE_DEPTH_KV),
        ("v_full", SMEM_PIPE_DEPTH_KV),
        ("v_empty", SMEM_PIPE_DEPTH_KV),
        ("s_ready", 2),
        ("p_first_ready", 2),
        ("p_second_ready", 2),
        ("p_o_rescale", 2),
        ("softmax_corr", 2),
        ("row_sum_ready", 2),
        ("o_ready", 2),
        ("corr_epi_full", 2),
        ("corr_epi_empty", 2),
        ("task_full", TASK_BROADCAST_STAGES),
        ("task_empty", TASK_BROADCAST_STAGES),
        ("softmax_corr_empty", 2),
    ]
    cursor = _align(tensor_end, 8)
    mbar_offsets = {}
    for name, stages in mbar_stages:
        mbar_offsets[name] = cursor
        cursor += stages * 8
    tmem_addr_offset = _align(cursor, 4)
    smem_size_bytes = tmem_addr_offset + 4

    k = IRBuilder(
        "nymph_flash_attention4",
        num_warps=config.num_warps,
        smem_size_bytes=smem_size_bytes,
        launch_shape=launch_shape,
        cluster_shape=(config.cta_group,),
    )
    q_gmem = k.arg(
        space=MemorySpace.GMEM,
        dtype=DType.F16,
        shape=(config.batch_size, config.seq_len, config.num_qo_heads, config.head_dim),
    )
    k_gmem = k.arg(
        space=MemorySpace.GMEM,
        dtype=DType.F16,
        shape=(config.batch_size, config.seq_len, config.num_kv_heads, config.head_dim),
    )
    v_gmem = k.arg(
        space=MemorySpace.GMEM,
        dtype=DType.F16,
        shape=(config.batch_size, config.seq_len, config.num_kv_heads, config.head_dim),
    )
    o_gmem = k.arg(
        space=MemorySpace.GMEM,
        dtype=DType.F16,
        shape=(config.batch_size, config.seq_len, config.num_qo_heads, config.head_dim),
    )

    smem_layout = SmemSwizzleLayout(Swizzle.B128)
    q_smem = tuple(
        _smem_tile(k, q_offset + stage * q_tile_bytes, smem_layout) for stage in range(2)
    )
    k_smem = tuple(
        _smem_tile(k, kv_offset + stage * kv_tile_bytes, smem_layout) for stage in range(3)
    )
    # FA4 aliases K_smem and V_smem; the protocol checker sees the same SMEM bytes
    # through different tensor ids because the byte_offset is identical.
    v_smem = tuple(
        _smem_tile(k, kv_offset + stage * kv_tile_bytes, smem_layout) for stage in range(3)
    )
    o_smem = tuple(
        _smem_tile(k, o_offset + stage * o_tile_bytes, smem_layout)
        for stage in range(TMEM_PIPE_DEPTH)
    )
    s_scale = k.tensor(
        space=MemorySpace.SMEM, dtype=DType.F32, shape=(4, BLK_M), byte_offset=scale_offset
    )
    task_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.I32,
        shape=(TASK_BROADCAST_STAGES, TASK_BROADCAST_FIELDS),
        byte_offset=task_offset,
    )
    # TMEM column plan: pure-Python bands over the 512-cell alloc (TMEM is not
    # a tensor; every use is a flat absolute physical TmemOperand, so the bands
    # only keep the column plan named). S stages at cols 0/128, O at 256/384.
    s_tmem = tuple(
        TmemBand(col0=stage * BLK_N, dtype=DType.F32) for stage in range(SMEM_PIPE_DEPTH_Q)
    )
    o_tmem = tuple(
        TmemBand(col0=BLK_N * SMEM_PIPE_DEPTH_Q + stage * BLK_N, dtype=DType.F32)
        for stage in range(TMEM_PIPE_DEPTH)
    )
    # TmemBand col0 is a physical 32-bit cell column. This matches baseline
    # P_region's f16 logical col_start=MMA_N and stride=TMEM_STAGE_STRIDE * 2
    # as physical starts 64 and 192 (f16 packs 2 elements per cell, so the
    # 128-wide P tile spans 64 cells, aliased over the S stage's cols).
    p_tmem = tuple(
        TmemBand(col0=(BLK_N // 2) + stage * BLK_N, dtype=DType.F16)
        for stage in range(SMEM_PIPE_DEPTH_Q)
    )

    s_frags = tuple(
        k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(SOFTMAX_CHUNK_CELLS,))
        for _ in range(SOFTMAX_NUM_CHUNKS)
    )
    p_frags = tuple(
        k.tensor(space=MemorySpace.REG, dtype=DType.F16, shape=(SOFTMAX_CHUNK_CELLS,))
        for _ in range(SOFTMAX_NUM_CHUNKS)
    )
    o_frag = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(O_CHUNK_CELLS,))
    o_frag_f16 = k.tensor(space=MemorySpace.REG, dtype=DType.F16, shape=(O_CHUNK_CELLS,))
    row_tmp = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(1,))
    row_scale = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(1,))
    tile_tmp = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(1,))
    row_bias = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(1,))
    row_max = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(1,))
    row_max_old = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(1,))
    row_max_safe = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(1,))
    row_sum_acc = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(1,))
    rounded = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(SOFTMAX_CHUNK_CELLS,))
    rounded_back = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(SOFTMAX_CHUNK_CELLS,))
    frac = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(SOFTMAX_CHUNK_CELLS,))
    frac_ex2 = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(SOFTMAX_CHUNK_CELLS,))

    q_full = tuple(
        k.mbar(kind=MBarKind.TMA, byte_offset=mbar_offsets[f"q_full_{stage}"], stages=1)
        for stage in range(2)
    )
    q_empty = tuple(
        k.mbar(kind=MBarKind.THREAD, byte_offset=mbar_offsets[f"q_empty_{stage}"], stages=1)
        for stage in range(2)
    )
    k_full = k.mbar(
        kind=MBarKind.TMA, byte_offset=mbar_offsets["k_full"], stages=SMEM_PIPE_DEPTH_KV
    )
    k_empty = k.mbar(
        kind=MBarKind.THREAD, byte_offset=mbar_offsets["k_empty"], stages=SMEM_PIPE_DEPTH_KV
    )
    v_full = k.mbar(
        kind=MBarKind.TMA, byte_offset=mbar_offsets["v_full"], stages=SMEM_PIPE_DEPTH_KV
    )
    v_empty = k.mbar(
        kind=MBarKind.THREAD, byte_offset=mbar_offsets["v_empty"], stages=SMEM_PIPE_DEPTH_KV
    )
    s_ready = k.mbar(kind=MBarKind.TCGEN05, byte_offset=mbar_offsets["s_ready"], stages=2)
    p_first_ready = k.mbar(
        kind=MBarKind.THREAD, byte_offset=mbar_offsets["p_first_ready"], stages=2
    )
    p_second_ready = k.mbar(
        kind=MBarKind.THREAD, byte_offset=mbar_offsets["p_second_ready"], stages=2
    )
    p_o_rescale = k.mbar(kind=MBarKind.THREAD, byte_offset=mbar_offsets["p_o_rescale"], stages=2)
    softmax_corr = k.mbar(kind=MBarKind.THREAD, byte_offset=mbar_offsets["softmax_corr"], stages=2)
    row_sum_ready = k.mbar(
        kind=MBarKind.THREAD, byte_offset=mbar_offsets["row_sum_ready"], stages=2
    )
    o_ready = k.mbar(kind=MBarKind.TCGEN05, byte_offset=mbar_offsets["o_ready"], stages=2)
    corr_epi_full = k.mbar(
        kind=MBarKind.THREAD, byte_offset=mbar_offsets["corr_epi_full"], stages=2
    )
    corr_epi_empty = k.mbar(
        kind=MBarKind.THREAD, byte_offset=mbar_offsets["corr_epi_empty"], stages=2
    )
    task_full = k.mbar(
        kind=MBarKind.THREAD, byte_offset=mbar_offsets["task_full"], stages=TASK_BROADCAST_STAGES
    )
    task_empty = k.mbar(
        kind=MBarKind.THREAD, byte_offset=mbar_offsets["task_empty"], stages=TASK_BROADCAST_STAGES
    )
    softmax_corr_empty = k.mbar(
        kind=MBarKind.THREAD, byte_offset=mbar_offsets["softmax_corr_empty"], stages=2
    )

    task_space = k.task_space(
        grid=(num_q_blocks, config.num_kv_heads, config.batch_size),
        fields=("m_block_idx", "head_idx", "batch_idx"),
    )
    scheduler = k.scheduler(task_space, policy="custom")

    with k.if_warp(0):
        # tmem_alloc is warp-collective (exactly one full warp); mbarrier.init
        # is per-thread, so a single elected thread runs every init.
        k.tmem_alloc(0, N_COLS_TMEM, addr_byte_offset=tmem_addr_offset, cta_group=config.cta_group)
        with k.if_elected():
            # count=1 barriers: armed by the TMA engine (tx bytes), a tcgen05
            # commit, or a single-thread stream's lone arrive.
            for mbar in [*q_full, *q_empty]:
                _init_stages(k, mbar, stages=1, count=1)
            _init_stages(k, k_full, stages=SMEM_PIPE_DEPTH_KV, count=1)
            _init_stages(k, k_empty, stages=SMEM_PIPE_DEPTH_KV, count=1)
            _init_stages(k, v_full, stages=SMEM_PIPE_DEPTH_KV, count=1)
            _init_stages(k, v_empty, stages=SMEM_PIPE_DEPTH_KV, count=1)
            for mbar in [s_ready, o_ready, corr_epi_empty, task_full]:
                _init_stages(k, mbar, stages=2, count=1)
            # Warpgroup-published barriers: the producing warpgroup's threads
            # each arrive once after their own per-row work (tcgen05_st rows /
            # s_scale lanes / o_smem rows), so a completed phase proves all
            # 128 rows landed without any wg_sync on the producer side.
            for mbar in [
                p_first_ready,
                p_second_ready,
                softmax_corr,
                softmax_corr_empty,
                row_sum_ready,
                corr_epi_full,
            ]:
                _init_stages(k, mbar, stages=2, count=WARPGROUP_THREADS)
            # p_o_rescale gates each PV MMA on BOTH the softmax warpgroup's
            # P-store and the correction warpgroup's O-rescale: 128 + 128.
            _init_stages(k, p_o_rescale, stages=2, count=2 * WARPGROUP_THREADS)
            _init_stages(k, task_empty, stages=TASK_BROADCAST_STAGES, count=TASK_CONSUMER_COUNT)

    # No implicit barrier between top-level statements: this fence + sync IS
    # the publish of the TMEM alloc and mbarrier cells to every stream.
    k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
    k.cta_sync()

    # Warpgroup 3 hosts the four single-warp streams (MMA / TMA load / TMA
    # store / scheduler); one warpgroup-wide setmaxnreg replaces the old
    # per-role maxnreg=48 annotations (set_maxnreg must statically cover a
    # whole warpgroup, which the single-warp branches below cannot).
    with k.if_warpgroup(3):
        k.set_maxnreg(48)

        _emit_scheduler_role(
            k,
            config,
            scheduler,
            task_smem,
            task_full,
            task_empty,
            num_q_blocks,
            num_kv_blocks,
            seq_q_per_tile,
        )
        _emit_tma_load_role(
            k,
            config,
            task_smem,
            task_full,
            task_empty,
            q_gmem,
            k_gmem,
            v_gmem,
            q_smem,
            k_smem,
            v_smem,
            q_full,
            q_empty,
            k_full,
            k_empty,
            v_full,
            v_empty,
            num_q_blocks,
            num_kv_blocks,
            q_tile_bytes,
            kv_tile_bytes,
            seq_q_per_tile,
        )
        _emit_mma_role(
            k,
            config,
            task_smem,
            task_full,
            task_empty,
            q_smem,
            k_smem,
            v_smem,
            s_tmem,
            p_tmem,
            o_tmem,
            q_full,
            q_empty,
            k_full,
            k_empty,
            v_full,
            v_empty,
            s_ready,
            p_first_ready,
            p_second_ready,
            p_o_rescale,
            o_ready,
            num_q_blocks,
            num_kv_blocks,
            seq_q_per_tile,
        )
        _emit_tma_store_role(
            k,
            config,
            task_smem,
            task_full,
            task_empty,
            o_gmem,
            o_smem,
            corr_epi_full,
            corr_epi_empty,
            num_q_blocks,
            seq_q_per_tile,
        )
    _emit_softmax_roles(
        k,
        config,
        task_smem,
        task_full,
        task_empty,
        s_tmem,
        p_tmem,
        s_scale,
        s_frags,
        p_frags,
        row_tmp,
        row_scale,
        tile_tmp,
        row_bias,
        row_max,
        row_max_old,
        row_max_safe,
        row_sum_acc,
        rounded,
        rounded_back,
        frac,
        frac_ex2,
        s_ready,
        p_first_ready,
        p_second_ready,
        p_o_rescale,
        softmax_corr,
        softmax_corr_empty,
        row_sum_ready,
        num_q_blocks,
        num_kv_blocks,
        seq_q_per_tile,
    )
    _emit_correction_epilogue_role(
        k,
        config,
        task_smem,
        task_full,
        task_empty,
        o_tmem,
        o_smem,
        s_scale,
        o_frag,
        o_frag_f16,
        row_tmp,
        row_scale,
        softmax_corr,
        softmax_corr_empty,
        row_sum_ready,
        p_o_rescale,
        o_ready,
        corr_epi_full,
        corr_epi_empty,
        num_q_blocks,
        num_kv_blocks,
        seq_q_per_tile,
    )
    # Teardown: every stream's pipeline work happens-before the dealloc.
    k.cta_sync()
    with k.if_warp(0):
        k.tmem_relinquish(config.cta_group)
        k.tmem_dealloc(0, N_COLS_TMEM, config.cta_group)

    return k.build()


def _emit_scheduler_role(
    k: IRBuilder,
    config: FlashAttention4Config,
    scheduler,
    task_smem: Tensor,
    task_full: MBar,
    task_empty: MBar,
    num_q_blocks: int,
    num_kv_blocks: int,
    seq_q_per_tile: int,
) -> None:
    # Single-thread scheduler stream: sched_next must sit in a statically
    # single-warp branch, and the mailbox stores + task_full arrive are
    # per-thread, so exactly one thread runs the whole loop.
    with k.if_warp(15), k.if_elected():
        sched_iter = k.scalar(initial=0, dtype=ScalarDType.I32)
        pipe_base = k.scalar(initial=0, dtype=ScalarDType.I32)
        kv_stage0_base = k.scalar(initial=0, dtype=ScalarDType.I32)
        kv_stage1_base = k.scalar(initial=0, dtype=ScalarDType.I32)
        kv_stage2_base = k.scalar(initial=0, dtype=ScalarDType.I32)
        with k.scheduler_impl(scheduler):
            with k.loop():
                stage = _task_broadcast_stage(sched_iter)
                phase = _task_broadcast_phase(sched_iter)
                k.mbarrier_wait(task_empty, stage=stage, phase=(phase + 1) % 2)
                task = k.sched_next(scheduler)
                k.store_scalar(_task_slot(task_smem, stage, TASK_FIELD_ID), task.task_id)
                k.store_scalar(_task_slot(task_smem, stage, TASK_FIELD_ITER), sched_iter)
                k.store_scalar(_task_slot(task_smem, stage, TASK_FIELD_PIPE_BASE), pipe_base)
                k.store_scalar(
                    _task_slot(task_smem, stage, TASK_FIELD_KV_STAGE0_BASE), kv_stage0_base
                )
                k.store_scalar(
                    _task_slot(task_smem, stage, TASK_FIELD_KV_STAGE1_BASE), kv_stage1_base
                )
                k.store_scalar(
                    _task_slot(task_smem, stage, TASK_FIELD_KV_STAGE2_BASE), kv_stage2_base
                )
                with k.if_(task.task_id >= 0):
                    k.store_scalar(
                        _task_slot(task_smem, stage, TASK_FIELD_M_BLOCK),
                        _scheduled_task_m_block_idx(config, task.task_id, num_q_blocks),
                    )
                    k.store_scalar(
                        _task_slot(task_smem, stage, TASK_FIELD_HEAD),
                        _scheduled_task_head_idx(config, task.task_id, num_q_blocks),
                    )
                    k.store_scalar(
                        _task_slot(task_smem, stage, TASK_FIELD_BATCH),
                        _scheduled_task_batch_idx(config, task.task_id, num_q_blocks),
                    )
                with k.if_(task.task_id < 0):
                    k.store_scalar(_task_slot(task_smem, stage, TASK_FIELD_M_BLOCK), 0)
                    k.store_scalar(_task_slot(task_smem, stage, TASK_FIELD_HEAD), 0)
                    k.store_scalar(_task_slot(task_smem, stage, TASK_FIELD_BATCH), 0)
                k.mbarrier_arrive(task_full, stage=stage)
                k.break_if(task.task_id < 0)
                m_block_idx = _scheduled_task_m_block_idx(config, task.task_id, num_q_blocks)
                n_block_count = _n_block_count(config, m_block_idx, num_kv_blocks, seq_q_per_tile)
                k.scalar_store(pipe_base, pipe_base + n_block_count)
                k.scalar_store(
                    kv_stage0_base, kv_stage0_base + _kv_stage_use_count_expr(n_block_count, 0)
                )
                k.scalar_store(
                    kv_stage1_base, kv_stage1_base + _kv_stage_use_count_expr(n_block_count, 1)
                )
                k.scalar_store(
                    kv_stage2_base, kv_stage2_base + _kv_stage_use_count_expr(n_block_count, 2)
                )
                k.scalar_store(sched_iter, sched_iter + 1)


@contextmanager
def _persistent_task_loop(
    k: IRBuilder, task_smem: Tensor, task_full: MBar, task_empty: MBar
) -> Iterator[tuple[object, object, object, tuple[object, object, object], object, object, object]]:
    """Consume scheduler mailbox entries until the sentinel task id is seen.

    Every executing thread reads the slot and arrives task_empty once
    (per-thread arrive), so TASK_CONSUMER_ROLES must account one arrival per
    mailbox-reading thread of every consumer stream.
    """
    consumer_iter = k.scalar(initial=0, dtype=ScalarDType.I32)
    with k.loop():
        stage = _task_broadcast_stage(consumer_iter)
        phase = _task_broadcast_phase(consumer_iter)
        k.mbarrier_wait(task_full, stage=stage, phase=phase)
        task_id = k.scalar(
            initial=_task_slot(task_smem, stage, TASK_FIELD_ID), dtype=ScalarDType.I32
        )
        task_iter = k.scalar(
            initial=_task_slot(task_smem, stage, TASK_FIELD_ITER), dtype=ScalarDType.I32
        )
        pipe_base = k.scalar(
            initial=_task_slot(task_smem, stage, TASK_FIELD_PIPE_BASE), dtype=ScalarDType.I32
        )
        kv_stage0_base = k.scalar(
            initial=_task_slot(task_smem, stage, TASK_FIELD_KV_STAGE0_BASE), dtype=ScalarDType.I32
        )
        kv_stage1_base = k.scalar(
            initial=_task_slot(task_smem, stage, TASK_FIELD_KV_STAGE1_BASE), dtype=ScalarDType.I32
        )
        kv_stage2_base = k.scalar(
            initial=_task_slot(task_smem, stage, TASK_FIELD_KV_STAGE2_BASE), dtype=ScalarDType.I32
        )
        m_block_idx = k.scalar(
            initial=_task_slot(task_smem, stage, TASK_FIELD_M_BLOCK), dtype=ScalarDType.I32
        )
        head_idx = k.scalar(
            initial=_task_slot(task_smem, stage, TASK_FIELD_HEAD), dtype=ScalarDType.I32
        )
        batch_idx = k.scalar(
            initial=_task_slot(task_smem, stage, TASK_FIELD_BATCH), dtype=ScalarDType.I32
        )
        k.mbarrier_arrive(task_empty, stage=stage)
        k.break_if(task_id < 0)
        yield (
            task_id,
            task_iter,
            pipe_base,
            (kv_stage0_base, kv_stage1_base, kv_stage2_base),
            m_block_idx,
            head_idx,
            batch_idx,
        )
        k.scalar_store(consumer_iter, consumer_iter + 1)


def _task_broadcast_stage(task_iter):
    return task_iter % TASK_BROADCAST_STAGES


def _task_broadcast_phase(task_iter):
    return (task_iter // TASK_BROADCAST_STAGES) % 2


def _task_m_block_idx(task_id, num_q_blocks: int):
    return task_id % num_q_blocks


def _task_head_idx(task_id, num_q_blocks: int, num_kv_heads: int):
    return (task_id // num_q_blocks) % num_kv_heads


def _task_batch_idx(task_id, num_q_blocks: int, num_kv_heads: int, batch_size: int):
    if batch_size == 1:
        return 0
    return (task_id // (num_q_blocks * num_kv_heads)) % batch_size


def _scheduled_task_m_block_idx(config: FlashAttention4Config, task_id, num_q_blocks: int):
    return _task_m_block_idx(task_id, num_q_blocks)


def _scheduled_task_head_idx(config: FlashAttention4Config, task_id, num_q_blocks: int):
    return _task_head_idx(task_id, num_q_blocks, config.num_kv_heads)


def _scheduled_task_batch_idx(config: FlashAttention4Config, task_id, num_q_blocks: int):
    return _task_batch_idx(task_id, num_q_blocks, config.num_kv_heads, config.batch_size)


def _emit_tma_load_role(
    k: IRBuilder,
    config: FlashAttention4Config,
    task_smem: Tensor,
    task_full: MBar,
    task_empty: MBar,
    q_gmem: Tensor,
    k_gmem: Tensor,
    v_gmem: Tensor,
    q_smem: tuple[Tensor, Tensor],
    k_smem: tuple[Tensor, Tensor, Tensor],
    v_smem: tuple[Tensor, Tensor, Tensor],
    q_full: tuple[MBar, MBar],
    q_empty: tuple[MBar, MBar],
    k_full: MBar,
    k_empty: MBar,
    v_full: MBar,
    v_empty: MBar,
    num_q_blocks: int,
    num_kv_blocks: int,
    q_tile_bytes: int,
    kv_tile_bytes: int,
    seq_q_per_tile: int,
) -> None:
    # Single-thread TMA producer stream (lane 0 of warp 13): mbarrier waits
    # are fine from one thread, arrive_expect_tx is per-thread (one arm per
    # load), and tma_load requires a single-thread mask.
    with k.if_warp(13), k.if_elected():
        with _persistent_task_loop(k, task_smem, task_full, task_empty) as (
            _task_id,
            local_iter,
            _pipe_base,
            kv_stage_bases,
            m_block_idx,
            head_idx,
            batch_idx,
        ):
            n_block_count = _n_block_count(config, m_block_idx, num_kv_blocks, seq_q_per_tile)
            m_start = m_block_idx * seq_q_per_tile * SMEM_PIPE_DEPTH_Q
            q_head = head_idx * (config.num_qo_heads // config.num_kv_heads)
            gqa_ratio = config.num_qo_heads // config.num_kv_heads
            q_empty_phase = (local_iter + 1) % 2

            def load_q(i_q: int) -> None:
                k.mbarrier_wait(q_empty[i_q], phase=q_empty_phase)
                k.mbarrier_arrive_expect_tx(q_full[i_q], bytes=q_tile_bytes)
                k.tma_load(
                    q_smem[i_q],
                    q_gmem,
                    mbar=q_full[i_q],
                    coords=(batch_idx, m_start + i_q * seq_q_per_tile, q_head, 0),
                    shape=(BLK_M, HEAD_DIM),
                    gmem_shape=(1, seq_q_per_tile, gqa_ratio, HEAD_DIM),
                    cta_group=config.cta_group,
                )

            def load_k(kv_idx: int) -> None:
                kv_block_idx = _scheduled_kv_block_idx(config, kv_idx, n_block_count, num_kv_blocks)
                stage = kv_idx % SMEM_PIPE_DEPTH_KV
                use = _kv_stage_use(kv_stage_bases[stage], kv_idx)
                empty_phase = (use + 1) % 2
                k.mbarrier_wait(v_empty, stage=stage, phase=empty_phase)
                k.mbarrier_arrive_expect_tx(k_full, stage=stage, bytes=kv_tile_bytes)
                k.tma_load(
                    k_smem[stage],
                    k_gmem,
                    mbar=k_full,
                    mbar_stage=stage,
                    coords=(batch_idx, kv_block_idx * BLK_N, head_idx, 0),
                    shape=(BLK_N, HEAD_DIM),
                    gmem_shape=(1, BLK_N, 1, HEAD_DIM),
                    cta_group=config.cta_group,
                )

            def load_v(kv_idx: int) -> None:
                kv_block_idx = _scheduled_kv_block_idx(config, kv_idx, n_block_count, num_kv_blocks)
                stage = kv_idx % SMEM_PIPE_DEPTH_KV
                use = _kv_stage_use(kv_stage_bases[stage], kv_idx)
                full_phase = use % 2
                k.mbarrier_wait(k_empty, stage=stage, phase=full_phase)
                k.mbarrier_arrive_expect_tx(v_full, stage=stage, bytes=kv_tile_bytes)
                k.tma_load(
                    v_smem[stage],
                    v_gmem,
                    mbar=v_full,
                    mbar_stage=stage,
                    coords=(batch_idx, kv_block_idx * BLK_N, head_idx, 0),
                    shape=(BLK_N, HEAD_DIM),
                    gmem_shape=(1, BLK_N, 1, HEAD_DIM),
                    cta_group=config.cta_group,
                )

            load_q(0)
            with _kv_in_bounds(k, 0, n_block_count):
                load_k(0)
            load_q(1)
            with _kv_in_bounds(k, 0, n_block_count):
                load_v(0)
            for kv_idx in range(1, num_kv_blocks):
                with _kv_in_bounds(k, kv_idx, n_block_count):
                    load_k(kv_idx)
                    load_v(kv_idx)


def _emit_mma_role(
    k: IRBuilder,
    config: FlashAttention4Config,
    task_smem: Tensor,
    task_full: MBar,
    task_empty: MBar,
    q_smem: tuple[Tensor, Tensor],
    k_smem: tuple[Tensor, Tensor, Tensor],
    v_smem: tuple[Tensor, Tensor, Tensor],
    s_tmem: tuple[TmemBand, TmemBand],
    p_tmem: tuple[TmemBand, TmemBand],
    o_tmem: tuple[TmemBand, TmemBand],
    q_full: tuple[MBar, MBar],
    q_empty: tuple[MBar, MBar],
    k_full: MBar,
    k_empty: MBar,
    v_full: MBar,
    v_empty: MBar,
    s_ready: MBar,
    p_first_ready: MBar,
    p_second_ready: MBar,
    p_o_rescale: MBar,
    o_ready: MBar,
    num_q_blocks: int,
    num_kv_blocks: int,
    seq_q_per_tile: int,
) -> None:
    # Single-thread MMA issuer stream (lane 0 of warp 12): tcgen05_mma and
    # tcgen05_commit require a single-thread mask, and the k/v/q release
    # arrives are per-thread (each buffer's empty barrier counts exactly one
    # arrival per k-tile).
    with k.if_warp(12), k.if_elected():
        with _persistent_task_loop(k, task_smem, task_full, task_empty) as (
            _task_id,
            local_iter,
            pipe_base,
            kv_stage_bases,
            m_block_idx,
            _head_idx,
            _batch_idx,
        ):
            n_block_count = _n_block_count(config, m_block_idx, num_kv_blocks, seq_q_per_tile)
            q_phase = local_iter % 2
            stage = 0
            full_phase = _kv_stage_use(kv_stage_bases[stage], 0) % 2
            with _kv_in_bounds(k, 0, n_block_count):
                k.mbarrier_wait(k_full, stage=stage, phase=full_phase)
                for i_q in range(SMEM_PIPE_DEPTH_Q):
                    k.mbarrier_wait(q_full[i_q], phase=q_phase)
                    _emit_qk_mma(k, config, s_tmem[i_q], q_smem[i_q], k_smem[stage])
                    k.tcgen05_commit(s_ready, stage=i_q, cta_group=config.cta_group)
                k.mbarrier_arrive(k_empty, stage=stage)
            for kv_idx in range(num_kv_blocks):
                with _kv_in_bounds(k, kv_idx, n_block_count):
                    stage_v = kv_idx % SMEM_PIPE_DEPTH_KV
                    full_phase_v = _kv_stage_use(kv_stage_bases[stage_v], kv_idx) % 2
                    pipe_phase = _pipe_phase(pipe_base, kv_idx)
                    next_kv_idx = kv_idx + 1
                    has_next_k = next_kv_idx < num_kv_blocks
                    if has_next_k:
                        stage_k = next_kv_idx % SMEM_PIPE_DEPTH_KV
                        full_phase_k = _kv_stage_use(kv_stage_bases[stage_k], next_kv_idx) % 2
                    for i_q in range(SMEM_PIPE_DEPTH_Q):
                        if i_q == 0:
                            k.mbarrier_wait(v_full, stage=stage_v, phase=full_phase_v)
                        k.mbarrier_wait(p_o_rescale, stage=i_q, phase=pipe_phase)
                        k.mbarrier_wait(p_first_ready, stage=i_q, phase=pipe_phase)
                        for k_group in range(PV_SPLIT_GROUPS):
                            _emit_pv_mma(
                                k,
                                config,
                                o_tmem[i_q],
                                p_tmem[i_q],
                                v_smem[stage_v],
                                k_group,
                                kv_idx,
                            )
                        k.mbarrier_wait(p_second_ready, stage=i_q, phase=pipe_phase)
                        for k_group in range(PV_SPLIT_GROUPS, BLK_N // MMA_K):
                            _emit_pv_mma(
                                k,
                                config,
                                o_tmem[i_q],
                                p_tmem[i_q],
                                v_smem[stage_v],
                                k_group,
                                kv_idx,
                            )
                        k.tcgen05_commit(o_ready, stage=i_q, cta_group=config.cta_group)
                        if has_next_k:
                            with _kv_in_bounds(k, next_kv_idx, n_block_count):
                                if i_q == 0:
                                    k.mbarrier_wait(k_full, stage=stage_k, phase=full_phase_k)
                                k.mbarrier_wait(q_full[i_q], phase=q_phase)
                                _emit_qk_mma(k, config, s_tmem[i_q], q_smem[i_q], k_smem[stage_k])
                                k.tcgen05_commit(s_ready, stage=i_q, cta_group=config.cta_group)
                    k.mbarrier_arrive(v_empty, stage=stage_v)
                    if has_next_k:
                        with _kv_in_bounds(k, next_kv_idx, n_block_count):
                            k.mbarrier_arrive(k_empty, stage=stage_k)
            for i_q in range(SMEM_PIPE_DEPTH_Q):
                k.mbarrier_arrive(q_empty[i_q])


def _emit_pv_mma(
    k: IRBuilder,
    config: FlashAttention4Config,
    o_stage: TmemBand,
    p_stage: TmemBand,
    v_stage: Tensor,
    k_group: int,
    kv_idx: int,
) -> None:
    k_offset = k_group * MMA_K
    k.tcgen05_mma(
        o_stage.at(0, 0),
        p_stage.elem(0, k_offset),
        v_stage[k_offset : k_offset + MMA_K, :],
        m=BLK_M,
        n=HEAD_DIM,
        k=MMA_K,
        accum=kv_idx != 0 or k_group != 0,
        trans_b=True,
        cta_group=config.cta_group,
    )


def _emit_qk_mma(
    k: IRBuilder, config: FlashAttention4Config, s_stage: TmemBand, q_stage: Tensor, k_stage: Tensor
) -> None:
    for k_group in range(HEAD_DIM // MMA_K):
        k_offset = k_group * MMA_K
        k.tcgen05_mma(
            s_stage.at(0, 0),
            q_stage[:, k_offset : k_offset + MMA_K],
            k_stage[:, k_offset : k_offset + MMA_K],
            m=BLK_M,
            n=BLK_N,
            k=MMA_K,
            accum=k_group != 0,
            cta_group=config.cta_group,
        )


def _emit_softmax_roles(
    k: IRBuilder,
    config: FlashAttention4Config,
    task_smem: Tensor,
    task_full: MBar,
    task_empty: MBar,
    s_tmem: tuple[TmemBand, TmemBand],
    p_tmem: tuple[TmemBand, TmemBand],
    s_scale: Tensor,
    s_frags: tuple[Tensor, ...],
    p_frags: tuple[Tensor, ...],
    row_tmp: Tensor,
    row_scale: Tensor,
    tile_tmp: Tensor,
    row_bias: Tensor,
    row_max: Tensor,
    row_max_old: Tensor,
    row_max_safe: Tensor,
    row_sum_acc: Tensor,
    rounded: Tensor,
    rounded_back: Tensor,
    frac: Tensor,
    frac_ex2: Tensor,
    s_ready: MBar,
    p_first_ready: MBar,
    p_second_ready: MBar,
    p_o_rescale: MBar,
    softmax_corr: MBar,
    softmax_corr_empty: MBar,
    row_sum_ready: MBar,
    num_q_blocks: int,
    num_kv_blocks: int,
    seq_q_per_tile: int,
) -> None:
    scale_log2 = math.log2(math.e) / math.sqrt(config.head_dim)
    # Softmax warpgroup streams: the body stays warpgroup-wide (tcgen05_ld/st
    # spread the 128 rows across the group's 128 threads; reg ops are
    # warp-collective). Every mbarrier_arrive below is per-thread — each
    # thread signals after ITS row's tcgen05_st / s_scale store — so the
    # barriers are initialized with count=WARPGROUP_THREADS and a completed
    # phase proves all 128 rows landed without a producer-side wg_sync.
    for i_q in range(SMEM_PIPE_DEPTH_Q):
        with k.if_warpgroup(i_q):
            k.set_maxnreg(200)
            with _persistent_task_loop(k, task_smem, task_full, task_empty) as (
                _task_id,
                _local_iter,
                pipe_base,
                _kv_stage_bases,
                m_block_idx,
                _head_idx,
                _batch_idx,
            ):
                n_block_count = _n_block_count(config, m_block_idx, num_kv_blocks, seq_q_per_tile)
                k.reg_fill(row_max, -float("inf"))
                k.reg_fill(row_sum_acc, 0.0)
                for kv_idx in range(num_kv_blocks):
                    with _kv_in_bounds(k, kv_idx, n_block_count):
                        pipe_phase = _pipe_phase(pipe_base, kv_idx)
                        kv_block_idx = _scheduled_kv_block_idx(
                            config, kv_idx, n_block_count, num_kv_blocks
                        )
                        query_start = (
                            m_block_idx * seq_q_per_tile * SMEM_PIPE_DEPTH_Q + i_q * seq_q_per_tile
                        )
                        k.mbarrier_wait(s_ready, stage=i_q, phase=pipe_phase)
                        for chunk in range(SOFTMAX_NUM_CHUNKS):
                            col = chunk * SOFTMAX_CHUNK_CELLS
                            k.tcgen05_ld(
                                s_frags[chunk], s_tmem[i_q].at(0, col), num=SOFTMAX_CHUNK_CELLS
                            )
                        k.tcgen05_wait_ld()
                        if config.is_causal:
                            for chunk in range(SOFTMAX_NUM_CHUNKS):
                                col = chunk * SOFTMAX_CHUNK_CELLS
                                k.reg_causal_mask(
                                    s_frags[chunk],
                                    s_frags[chunk],
                                    query_start=query_start,
                                    key_start=kv_block_idx * BLK_N + col,
                                    group_size=config.num_qo_heads // config.num_kv_heads,
                                )
                        k.reg_reduce(row_tmp, s_frags[0], op="max")
                        for chunk in range(1, SOFTMAX_NUM_CHUNKS):
                            k.reg_reduce(tile_tmp, s_frags[chunk], op="max")
                            k.reg_max(row_tmp, row_tmp, tile_tmp)
                        k.reg_add(row_max_old, row_max, 0.0)
                        if kv_idx == 0:
                            k.reg_add(row_max, row_tmp, 0.0)
                            if config.is_causal:
                                k.reg_causal_mask(
                                    row_max_safe,
                                    row_tmp,
                                    query_start=query_start,
                                    key_start=kv_block_idx * BLK_N,
                                    group_size=config.num_qo_heads // config.num_kv_heads,
                                    mask_value=0.0,
                                )
                            else:
                                k.reg_add(row_max_safe, row_tmp, 0.0)
                            k.reg_fill(row_scale, 1.0)
                        else:
                            k.reg_max(row_max, row_max_old, row_tmp)
                            k.reg_softmax_rescale(
                                row_max, row_scale, row_max_old, row_max, scale_log2, threshold=8.0
                            )
                            k.reg_add(row_max_safe, row_max, 0.0)
                        k.reg_mul(row_bias, row_max_safe, -scale_log2)
                        k.reg_store(_scale_slot(s_scale, i_q, k.tid_in_wg()), row_scale)
                        k.mbarrier_arrive(softmax_corr, stage=i_q)
                        for chunk in range(SOFTMAX_NUM_CHUNKS):
                            k.reg_fma(s_frags[chunk], s_frags[chunk], scale_log2, row_bias)
                            _emit_exp2_emulation(
                                k, s_frags[chunk], rounded, rounded_back, frac, frac_ex2
                            )
                            if config.is_causal:
                                col = chunk * SOFTMAX_CHUNK_CELLS
                                k.reg_causal_mask(
                                    s_frags[chunk],
                                    s_frags[chunk],
                                    query_start=query_start,
                                    key_start=kv_block_idx * BLK_N + col,
                                    group_size=config.num_qo_heads // config.num_kv_heads,
                                    mask_value=0.0,
                                )
                            k.reg_cvt(p_frags[chunk], s_frags[chunk])
                        for chunk in range(SOFTMAX_NUM_CHUNKS):
                            k.tcgen05_st(
                                p_tmem[i_q].at(0, chunk * P_STORE_CELLS),
                                p_frags[chunk][0:P_STORE_CELLS],
                                num=P_STORE_CELLS,
                            )
                            if chunk == P_FIRST_READY_CHUNKS - 1:
                                k.tcgen05_wait_st()
                                k.mbarrier_arrive(p_o_rescale, stage=i_q)
                                k.mbarrier_arrive(p_first_ready, stage=i_q)
                        k.tcgen05_wait_st()
                        k.mbarrier_arrive(p_second_ready, stage=i_q)
                        k.mbarrier_wait(softmax_corr_empty, stage=i_q, phase=pipe_phase)
                        k.reg_reduce(row_tmp, s_frags[0], op="sum")
                        for chunk in range(1, SOFTMAX_NUM_CHUNKS):
                            k.reg_reduce(tile_tmp, s_frags[chunk], op="sum")
                            k.reg_add(row_tmp, row_tmp, tile_tmp)
                        if kv_idx == 0:
                            k.reg_add(row_sum_acc, row_tmp, 0.0)
                        else:
                            k.reg_fma(row_sum_acc, row_sum_acc, row_scale, row_tmp)
                k.reg_store(_scale_slot(s_scale, 2 + i_q, k.tid_in_wg()), row_sum_acc)
                k.mbarrier_arrive(row_sum_ready, stage=i_q)


def _emit_correction_epilogue_role(
    k: IRBuilder,
    config: FlashAttention4Config,
    task_smem: Tensor,
    task_full: MBar,
    task_empty: MBar,
    o_tmem: tuple[TmemBand, TmemBand],
    o_smem: tuple[Tensor, Tensor],
    s_scale: Tensor,
    o_frag: Tensor,
    o_frag_f16: Tensor,
    row_tmp: Tensor,
    row_scale: Tensor,
    softmax_corr: MBar,
    softmax_corr_empty: MBar,
    row_sum_ready: MBar,
    p_o_rescale: MBar,
    o_ready: MBar,
    corr_epi_full: MBar,
    corr_epi_empty: MBar,
    num_q_blocks: int,
    num_kv_blocks: int,
    seq_q_per_tile: int,
) -> None:
    # Correction warpgroup stream: same per-thread arrive discipline as the
    # softmax warpgroups — p_o_rescale / softmax_corr_empty / corr_epi_full
    # each collect 128 arrivals, one per thread after its own row work.
    with k.if_warpgroup(2):
        k.set_maxnreg(64)
        with _persistent_task_loop(k, task_smem, task_full, task_empty) as (
            _task_id,
            local_iter,
            pipe_base,
            _kv_stage_bases,
            m_block_idx,
            _head_idx,
            _batch_idx,
        ):
            n_block_count = _n_block_count(config, m_block_idx, num_kv_blocks, seq_q_per_tile)
            for kv_idx in range(num_kv_blocks):
                with _kv_in_bounds(k, kv_idx, n_block_count):
                    pipe_phase = _pipe_phase(pipe_base, kv_idx)
                    for i_q in range(SMEM_PIPE_DEPTH_Q):
                        k.mbarrier_wait(softmax_corr, stage=i_q, phase=pipe_phase)
                        if kv_idx != 0:
                            k.mbarrier_wait(
                                o_ready, stage=i_q, phase=_pipe_phase(pipe_base, kv_idx - 1)
                            )
                            k.reg_load(row_scale, _scale_slot(s_scale, i_q, k.tid_in_wg()))
                            _emit_o_rescale(k, o_tmem[i_q], o_frag, row_scale)
                        k.mbarrier_arrive(p_o_rescale, stage=i_q)
                        k.mbarrier_arrive(softmax_corr_empty, stage=i_q)
            final_phase = _pipe_phase(pipe_base, n_block_count - 1)
            for i_q in range(SMEM_PIPE_DEPTH_Q):
                k.mbarrier_wait(row_sum_ready, stage=i_q, phase=local_iter % 2)
                k.mbarrier_wait(o_ready, stage=i_q, phase=final_phase)
                k.mbarrier_wait(corr_epi_empty, stage=i_q, phase=(local_iter + 1) % 2)
                k.reg_load(row_tmp, _scale_slot(s_scale, 2 + i_q, k.tid_in_wg()))
                k.reg_unary(row_scale, row_tmp, op="rcp")
                k.reg_min(row_scale, 1.0, row_scale)
                for d_tile in range(HEAD_DIM // O_CHUNK_CELLS):
                    col = d_tile * O_CHUNK_CELLS
                    k.tcgen05_ld(o_frag, o_tmem[i_q].at(0, col), num=O_CHUNK_CELLS)
                    k.tcgen05_wait_ld()
                    k.reg_mul(o_frag, o_frag, row_scale)
                    k.reg_cvt(o_frag_f16, o_frag)
                    k.reg_store(
                        TensorSlice(
                            tensor=o_smem[i_q],
                            offsets=(k.tid_in_wg(), col),
                            shape=(1, O_CHUNK_CELLS),
                        ),
                        o_frag_f16,
                    )
                k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                k.mbarrier_arrive(corr_epi_full, stage=i_q)


def _emit_o_rescale(k: IRBuilder, o_stage: TmemBand, o_frag: Tensor, row_scale: Tensor) -> None:
    for d_tile in range(HEAD_DIM // O_CHUNK_CELLS):
        col = d_tile * O_CHUNK_CELLS
        k.tcgen05_ld(o_frag, o_stage.at(0, col), num=O_CHUNK_CELLS)
        k.tcgen05_wait_ld()
        # A register reduction spans one warp; the group-any is only a skip
        # optimization here (rows that need no rescale carry row_scale == 1.0,
        # an exact float identity), so warp scope is value-equivalent.
        k.reg_cond_rescale(o_frag, o_frag, row_scale, threshold=1.0, scope="warp")
        k.tcgen05_st(o_stage.at(0, col), o_frag, num=O_CHUNK_CELLS)
    k.tcgen05_wait_st()


def _emit_tma_store_role(
    k: IRBuilder,
    config: FlashAttention4Config,
    task_smem: Tensor,
    task_full: MBar,
    task_empty: MBar,
    o_gmem: Tensor,
    o_smem: tuple[Tensor, Tensor],
    corr_epi_full: MBar,
    corr_epi_empty: MBar,
    num_q_blocks: int,
    seq_q_per_tile: int,
) -> None:
    # Single-thread TMA store stream (lane 0 of warp 14). cp_async_bulk
    # groups are per-stream: the commit and the wait_group_read must run on
    # the SAME stream that issued the tma_store, so all three live in this
    # one-thread branch; the correction warpgroup observes the drain through
    # the corr_epi_empty mbarrier, not through the group.
    with k.if_warp(14), k.if_elected():
        with _persistent_task_loop(k, task_smem, task_full, task_empty) as (
            _task_id,
            local_iter,
            _pipe_base,
            _kv_stage_bases,
            m_block_idx,
            head_idx,
            batch_idx,
        ):
            m_start = m_block_idx * seq_q_per_tile * SMEM_PIPE_DEPTH_Q
            gqa_ratio = config.num_qo_heads // config.num_kv_heads
            q_head = head_idx * gqa_ratio
            for i_q in range(TMEM_PIPE_DEPTH):
                k.mbarrier_wait(corr_epi_full, stage=i_q, phase=local_iter % 2)
                k.tma_store(
                    o_gmem,
                    o_smem[i_q],
                    coords=(batch_idx, m_start + i_q * seq_q_per_tile, q_head, 0),
                    shape=(BLK_M, HEAD_DIM),
                    gmem_shape=(1, seq_q_per_tile, gqa_ratio, HEAD_DIM),
                )
                k.cp_async_bulk_commit_group()
            for i_q in range(TMEM_PIPE_DEPTH):
                k.cp_async_bulk_wait_group_read(1 - i_q)
                k.mbarrier_arrive(corr_epi_empty, stage=i_q)


def _emit_exp2_emulation(
    k: IRBuilder,
    src_dst: Tensor,
    rounded: Tensor,
    rounded_back: Tensor,
    frac: Tensor,
    frac_ex2: Tensor,
) -> None:
    """Approximate exp2 in registers with the FA4 polynomial + exponent bit path.

    `rounded = floor(x + 1.5 * 2**23) - 1.5 * 2**23` extracts an integer exponent
    under fp32 mantissa rounding. The polynomial approximates exp2(frac), and
    RegCombineIntFracEx2 rebuilds `2**rounded * exp2(frac)` by shifting the
    integer exponent contribution into the fp32 exponent field.
    """
    fp32_round_int = float(2**23 + 2**22)
    k.reg_add(rounded, src_dst, fp32_round_int, rounding="rm")
    k.reg_sub(rounded_back, rounded, fp32_round_int)
    k.reg_sub(frac, src_dst, rounded_back)
    k.reg_fill(frac_ex2, 0.07711908966302872)
    k.reg_fma(frac_ex2, frac_ex2, frac, 0.22756439447402954)
    k.reg_fma(frac_ex2, frac_ex2, frac, 0.6951461434364319)
    k.reg_fma(frac_ex2, frac_ex2, frac, 1.0)
    k.reg_combine_int_frac_ex2(src_dst, rounded, frac_ex2)


def _smem_tile(k: IRBuilder, byte_offset: int, layout: SmemSwizzleLayout | None) -> Tensor:
    return k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.F16,
        shape=(BLK_M, HEAD_DIM),
        layout=layout,
        byte_offset=byte_offset,
    )


def _scale_slot(tensor: Tensor, row: int, col) -> TensorSlice:
    return TensorSlice(tensor=tensor, offsets=(row, col), shape=(1, 1))


def _task_slot(tensor: Tensor, stage, field: int) -> TensorSlice:
    return TensorSlice(tensor=tensor, offsets=(stage, field), shape=(1, 1))


def _init_stages(k: IRBuilder, mbar: MBar, *, stages: int, count: int) -> None:
    for stage in range(stages):
        k.mbarrier_init(mbar, count=count, stage=stage)


def _kv_stage_use(stage_base, kv_idx: int):
    return stage_base + (kv_idx // SMEM_PIPE_DEPTH_KV)


def _kv_stage_use_count_expr(n_block_count, stage: int):
    return (n_block_count + (SMEM_PIPE_DEPTH_KV - 1 - stage)) // SMEM_PIPE_DEPTH_KV


def _pipe_phase(pipe_base, kv_idx):
    return (pipe_base + kv_idx) % 2


def _scheduled_kv_block_idx(
    config: FlashAttention4Config, kv_idx: int, n_block_count, num_kv_blocks: int
):
    if config.is_causal:
        return n_block_count - 1 - kv_idx
    return (num_kv_blocks - 1) - kv_idx


def _n_block_count(
    config: FlashAttention4Config, m_block_idx, num_kv_blocks: int, seq_q_per_tile: int
):
    if config.is_causal:
        q_rows_per_task = seq_q_per_tile * SMEM_PIPE_DEPTH_Q
        return (m_block_idx * q_rows_per_task + q_rows_per_task + (BLK_N - 1)) // BLK_N
    return num_kv_blocks


@contextmanager
def _kv_in_bounds(k: IRBuilder, kv_idx: int, n_block_count) -> Iterator[None]:
    if isinstance(n_block_count, int):
        yield
    else:
        with k.if_(n_block_count > kv_idx):
            yield


def _seq_q_per_tile(config: FlashAttention4Config) -> int:
    return BLK_M // (config.num_qo_heads // config.num_kv_heads)


def _validate_config(config: FlashAttention4Config) -> None:
    if config.batch_size != 1:
        raise ValueError("flash_attention4 currently supports batch_size=1")
    if config.head_dim != HEAD_DIM:
        raise ValueError("flash_attention4 currently supports head_dim=128")
    if config.num_qo_heads <= 0 or config.num_kv_heads <= 0:
        raise ValueError("flash_attention4 head counts must be positive")
    if config.num_qo_heads % config.num_kv_heads != 0:
        raise ValueError("flash_attention4 num_qo_heads must be divisible by num_kv_heads")
    gqa_ratio = config.num_qo_heads // config.num_kv_heads
    if BLK_M % gqa_ratio != 0:
        raise ValueError("flash_attention4 BLK_M must be divisible by GQA ratio")
    if config.seq_len <= 0 or config.seq_len % BLK_N != 0:
        raise ValueError("flash_attention4 seq_len must be a positive multiple of 128")
    if config.cta_group != 1:
        raise ValueError("flash_attention4 currently supports CTA_GROUP=1")
    if config.num_warps < 16:
        raise ValueError("flash_attention4 num_warps must be at least 16")


def _validate_launch_shape(launch_shape: LaunchShape, cta_group: int) -> None:
    if not isinstance(launch_shape, tuple) or len(launch_shape) != 1:
        raise ValueError("flash_attention4 requires a 1D launch_shape")
    if launch_shape[0] < 1:
        raise ValueError("flash_attention4 launch_shape[0] must be positive")
    if launch_shape[0] % cta_group != 0:
        raise ValueError("flash_attention4 launch_shape[0] must be divisible by CTA_GROUP")


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs
