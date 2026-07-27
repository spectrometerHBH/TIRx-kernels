"""Static-shape fp16/bf16 GEMM expressed in clean Nymph IR."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from ..builder import IRBuilder
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
    TensorSlice,
)

_ELEMENT_BYTES = {DType.F16: 2, DType.BF16: 2}

CTA_GROUP = 2
CTA_M = 256  # cluster M tile (2-SM MMA combines the pair)
BLK_M = CTA_M // CTA_GROUP  # per-CTA A rows = 128
MMA_K = 16  # the dense fp16/bf16 atomic MMA K (a full-K IR MMA is an ordered run of k/16 atoms)
# Per-warpgroup register rebalance for the no-overlap epilogue (canon's setmaxnreg).
PRODUCER_MAXNREG = 56
CONSUMER_MAXNREG = 224
TMEM_LD_SIZE = 8  # tcgen05.ld fragment width (cols per warp-collective load)
NOL = 2 * TMEM_LD_SIZE  # no-overlap tmem->reg f32 read band (canonical `NOL`=16)
N_COLS_TMEM = 512
# CLC (Cluster Launch Control) response handle.
CLC_HANDLE_WORDS = 4
CLC_HANDLE_BYTES = CLC_HANDLE_WORDS * 4

# Per-shape tuning knobs (selected by N), mirroring TIRx GEMM_CONFIGS. CTA_M=256 always.
GEMM_CONFIGS = {
    # 1024: mma_n=64 (16 N tiles × 4 M tiles = 64 cluster tasks). nvjet's
    # shape: blk_k=64 with a deep 10-stage ring, and the STATIC scheduler —
    # one launch tile per cluster, so the CLC steal machinery is pure overhead.
    1024: {
        "mma_n": 64,
        "blk_k": 64,
        "l2_group_size": 4,
        "overlap_epilogue": True,
        "pipe_depth": 10,
        "wb_pipe_depth": 2,
        "scheduler": "static",
    },
    2048: {
        "mma_n": 256,
        "blk_k": 64,
        # l2>=num_m_tiles => plain M-major raster (nvjet's order); beats the
        # grouped swizzle here (0.984 -> 0.989).
        "l2_group_size": 16,
        "overlap_epilogue": True,
        # depth 6 (nvjet's 64x6): bf16_2048 0.985 -> 1.002; fp16 stays 1.00.
        "pipe_depth": 6,
        # wb_pipe_depth=8 (EPI_N=32, not 64): measured better than canon's 4
        # (0.971 vs 0.921) — kept deliberately.
        "wb_pipe_depth": 8,
    },
    # 4096: nvjet's shape — single consumer (OVERLAP) with a 6-stage ring
    # (SMEM: (16+16)KB x 6 = 192KB); cublas/tirx 0.969 (2-cons depth-4) ->
    # 0.983 here.
    4096: {
        "mma_n": 256,
        "blk_k": 64,
        # l2>=num_m_tiles => plain M-major raster (nvjet's order):
        # cublas/tirx 0.983 -> 1.006 fp16 / 1.002 bf16 measured.
        "l2_group_size": 16,
        "overlap_epilogue": True,
        "pipe_depth": 6,
        "wb_pipe_depth": 8,
    },
    8192: {
        "mma_n": 256,
        # blk_k=64/pipe_depth=4.
        "blk_k": 64,
        "l2_group_size": 8,
        "overlap_epilogue": False,
        "pipe_depth": 4,
        "wb_pipe_depth": 8,
    },
    16384: {
        "mma_n": 256,
        # Canon verbatim: blk_k=64/pipe_depth=4 (was 128/2 "same SMEM";
        # fp16_16384 measured 0.959 there). l2_group_size=4 (not canon's 8):
        # cublas-ratio 0.962 -> 0.992 measured; 16 collapses to 0.943.
        "blk_k": 64,
        "l2_group_size": 4,
        "overlap_epilogue": False,
        "pipe_depth": 4,
        "wb_pipe_depth": 8,
    },
}
_DEFAULT_GEMM_CONFIG = {
    "mma_n": 256,
    "blk_k": 64,
    "l2_group_size": 8,
    "overlap_epilogue": False,
    "pipe_depth": 4,
    "wb_pipe_depth": 8,
}


@dataclass(frozen=True, slots=True)
class Fp16Bf16GemmConfig:
    m: int = 512
    n: int = 256
    k: int = 64
    dtype: DType = DType.F16
    # Independent knobs. Left as None they resolve from N via GEMM_CONFIGS.
    mma_n: int | None = None
    blk_k: int | None = None
    pipe_depth: int | None = None
    wb_pipe_depth: int | None = None
    l2_group_size: int | None = None
    overlap_epilogue: bool | None = None
    launch_shape: LaunchShape | None = None
    # "clc" (default) or "static" (no CLC: one launch tile per cluster).
    scheduler: str | None = None
    # Teardown form: None=canon handshake+dealloc; "dealloc_only" skips the
    # tmem_fin cluster handshake; "none" exits immediately (nvjet's form —
    # the HW frees TMEM at CTA exit).
    teardown: str | None = None


class _Resolved:
    """Resolved geometry for one (config)."""

    __slots__ = (
        "blk_k",
        "blk_n",
        "epi_n",
        "k_tiles",
        "l2_group_size",
        "mma_n",
        "num_consumer",
        "num_d_tiles",
        "num_m_tiles",
        "num_n_tiles",
        "num_warps",
        "overlap",
        "pair_tasks",
        "pipe_depth",
        "producer_wg_base",
        "scheduler",
        "tmem_slots",
        "wb_pipe_depth",
    )

    def __init__(self, config: Fp16Bf16GemmConfig) -> None:
        knob = GEMM_CONFIGS.get(config.n, _DEFAULT_GEMM_CONFIG)

        def pick(name, default):
            value = getattr(config, name)
            return default if value is None else value

        self.mma_n = pick("mma_n", knob["mma_n"])
        self.blk_k = pick("blk_k", knob["blk_k"])
        self.pipe_depth = pick("pipe_depth", knob["pipe_depth"])
        self.wb_pipe_depth = pick("wb_pipe_depth", knob["wb_pipe_depth"])
        self.l2_group_size = pick("l2_group_size", knob["l2_group_size"])
        self.overlap = pick("overlap_epilogue", knob["overlap_epilogue"])
        self.scheduler = pick("scheduler", knob.get("scheduler", "clc"))

        self.num_consumer = 1 if self.overlap else 2
        # OVERLAP double-buffers the single consumer's TMEM (MMA_PIPE=2).
        self.tmem_slots = 2 if self.overlap else self.num_consumer
        self.num_d_tiles = 2 if self.wb_pipe_depth > 1 else 1
        self.blk_n = self.mma_n // CTA_GROUP  # per-CTA B rows
        self.epi_n = self.mma_n // self.wb_pipe_depth
        self.num_m_tiles = config.m // (CTA_M * self.num_consumer)
        self.num_n_tiles = config.n // self.mma_n
        self.pair_tasks = self.num_m_tiles * self.num_n_tiles
        self.k_tiles = config.k // self.blk_k
        self.num_warps = (self.num_consumer + 1) * 4
        self.producer_wg_base = self.num_consumer * 4  # first warp of producer wg


def build_fp16_bf16_gemm(config: Fp16Bf16GemmConfig = Fp16Bf16GemmConfig()) -> Kernel:
    r = _Resolved(config)
    _validate_config(config, r)

    # Scheduling: the hardware CLC (Cluster Launch Control) work-stealing scheduler.
    num_clusters = r.pair_tasks
    launch_shape = config.launch_shape or (num_clusters * CTA_GROUP,)
    _validate_launch_shape(launch_shape)

    elem = _ELEMENT_BYTES[config.dtype]
    a_tile_bytes = BLK_M * r.blk_k * elem
    b_tile_bytes = r.blk_n * r.blk_k * elem
    d_tile_bytes = BLK_M * r.epi_n * elem
    # Per-consumer A rings, one shared B ring, per-consumer D writeback rings.
    a_offsets = tuple(c * r.pipe_depth * a_tile_bytes for c in range(r.num_consumer))
    b_off = r.num_consumer * r.pipe_depth * a_tile_bytes
    d_base = b_off + r.pipe_depth * b_tile_bytes
    d_offsets = tuple(d_base + c * r.num_d_tiles * d_tile_bytes for c in range(r.num_consumer))
    pool_off = d_base + r.num_consumer * r.num_d_tiles * d_tile_bytes
    # Scheduling: the hardware CLC.
    clc_off = (pool_off + 15) // 16 * 16  # the CLC response handle is 16B-aligned
    data_end = clc_off + CLC_HANDLE_BYTES
    metadata_cursor = (data_end + 7) // 8 * 8
    smem_full_off = metadata_cursor
    metadata_cursor += r.pipe_depth * 8
    smem_empty_off = metadata_cursor
    metadata_cursor += r.pipe_depth * 8
    tmem_full_off = metadata_cursor
    metadata_cursor += r.tmem_slots * 8
    tmem_empty_off = metadata_cursor
    metadata_cursor += r.tmem_slots * 8
    sched_arr_full_off = metadata_cursor
    metadata_cursor += 8
    sched_fin_off = metadata_cursor
    metadata_cursor += 8
    tmem_fin_off = metadata_cursor if r.overlap else None
    if r.overlap:
        metadata_cursor += 8
    alloc_done_off = metadata_cursor
    metadata_cursor += 8
    tmem_addr_off = (metadata_cursor + 3) // 4 * 4
    smem_size_bytes = tmem_addr_off + 4

    k = IRBuilder(
        "nymph_fp16_bf16_gemm",
        num_warps=r.num_warps,
        smem_size_bytes=smem_size_bytes,
        launch_shape=launch_shape,
        cluster_shape=(CTA_GROUP,),
    )
    a_gmem = k.arg(space=MemorySpace.GMEM, dtype=config.dtype, shape=(config.m, config.k))
    b_gmem = k.arg(space=MemorySpace.GMEM, dtype=config.dtype, shape=(config.n, config.k))
    d_gmem = k.arg(space=MemorySpace.GMEM, dtype=config.dtype, shape=(config.m, config.n))

    ab_layout = SmemSwizzleLayout(Swizzle.B128)
    a_smem = tuple(
        k.tensor(
            space=MemorySpace.SMEM,
            dtype=config.dtype,
            shape=(r.pipe_depth, BLK_M, r.blk_k),
            layout=ab_layout,
            byte_offset=off,
        )
        for off in a_offsets
    )
    b_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=config.dtype,
        shape=(r.pipe_depth, r.blk_n, r.blk_k),
        layout=ab_layout,
        byte_offset=b_off,
    )
    # Swizzle the epilogue write-back tile as wide as its row allows (canon's
    # `_swizzle_for_row_bytes(EPI_N * elem)`): the per-thread row stores
    # otherwise land on the same bank group — the measured 96x SMEM
    # store-bank-conflict gap at 2048³.
    d_row_bytes = r.epi_n * elem
    d_layout = None
    for swizzle, atom in ((Swizzle.B128, 128), (Swizzle.B64, 64), (Swizzle.B32, 32)):
        if d_row_bytes >= atom and d_row_bytes % atom == 0:
            d_layout = SmemSwizzleLayout(swizzle)
            break
    d_smem = tuple(
        k.tensor(
            space=MemorySpace.SMEM,
            dtype=config.dtype,
            shape=(r.num_d_tiles, BLK_M, r.epi_n),
            layout=d_layout,
            byte_offset=off,
        )
        for off in d_offsets
    )
    # CLC response handle.
    clc_handle = k.tensor(
        space=MemorySpace.SMEM, dtype=DType.U32, shape=(CLC_HANDLE_WORDS,), byte_offset=clc_off
    )

    # TMEM: one 512-col allocation.
    accum = k.tmem_tensor(0)

    smem_full = k.mbar(kind=MBarKind.TMA, byte_offset=smem_full_off, stages=r.pipe_depth)
    smem_full_cta0 = k.mbar_ref(smem_full, remote_coord=0)
    smem_empty = k.mbar(kind=MBarKind.TCGEN05, byte_offset=smem_empty_off, stages=r.pipe_depth)
    tmem_full = k.mbar(kind=MBarKind.TCGEN05, byte_offset=tmem_full_off, stages=r.tmem_slots)
    tmem_empty = k.mbar(kind=MBarKind.THREAD, byte_offset=tmem_empty_off, stages=r.tmem_slots)
    # ---- CLC scheduler objects.
    sched_space = k.task_space(grid=(r.num_m_tiles, r.num_n_tiles), fields=("m_idx", "n_idx"))
    sched = k.scheduler(sched_space, policy="custom")
    # sched_arr: the 16B handle barrier.
    sched_arr_full = k.mbar(kind=MBarKind.TMA, byte_offset=sched_arr_full_off, stages=1)
    # sched_fin: every worker loop arrives once per task at CTA-0's barrier.
    finish_arrivals = (2 + r.num_consumer) * 2 + r.num_consumer
    sched_fin = k.mbar(kind=MBarKind.THREAD, byte_offset=sched_fin_off, stages=1)
    sched_fin_leader = k.mbar_ref(sched_fin, remote_coord=0)
    # tmem_fin: canon's lightweight 2-CTA teardown handshake (overlap path).
    tmem_fin = (
        k.mbar(kind=MBarKind.THREAD, byte_offset=tmem_fin_off, stages=1) if r.overlap else None
    )
    # alloc_done: warp 0 arrives after the pair's tcgen05.alloc; the MMA warps
    # wait it before their first tcgen05 op (see the prologue reorder below).
    alloc_done = k.mbar(kind=MBarKind.THREAD, byte_offset=alloc_done_off, stages=1)
    # No sched_sync rendezvous.
    tmem_empty_leader = k.mbar_ref(tmem_empty, remote_coord=0)

    cta_id = k.cta_id()
    cta_rank = k.ctaid_in_cluster()
    # Per-cluster identity: cluster_id is this cluster's launch tile.
    cluster_id = cta_id // CTA_GROUP

    # Per-role task source: the CLC consume loop (each worker steals tiles via
    # the handle), or STATIC — this cluster's own launch tile (nvjet's scheme:
    # zero scheduler traffic when the grid already has one task per cluster).
    static_sched = r.scheduler == "static"
    if static_sched:

        @contextmanager
        def task_loop(wg_bar=None, warp_sync=False):
            yield k.let(cluster_id), k.let(0)

    else:

        def task_loop(wg_bar=None, warp_sync=False):
            return _clc_worker_loop(
                k,
                sched,
                clc_handle,
                sched_arr_full,
                sched_fin_leader,
                cluster_id,
                wg_bar=wg_bar,
                warp_sync=warp_sync,
            )

    def work_coords(task_id):
        """ClusterLaunchControlScheduler group-major L2 raster (cluster_m=cluster_n=1)."""
        if r.num_m_tiles <= r.l2_group_size:
            return task_id % r.num_m_tiles, task_id // r.num_m_tiles
        group_span = r.l2_group_size * r.num_n_tiles
        group_id = task_id // group_span
        within = task_id % group_span
        m_idx = group_id * r.l2_group_size + within % r.l2_group_size
        n_idx = within // r.l2_group_size
        return m_idx, n_idx

    # mbarrier inits are split across warps 1/2 (halves the serial init chain
    # on the cluster-barrier critical path); warp 0's tcgen05.alloc (below,
    # post-barrier) starts at kernel-entry and hides under the producers'
    # first TMA flight entirely.
    with k.if_warp(1):
        with k.if_elected():
            _init_stages(k, smem_full, stages=r.pipe_depth, count=1)
            _init_stages(k, smem_empty, stages=r.pipe_depth, count=r.num_consumer)
    with k.if_warp(2):
        with k.if_elected():
            _init_stages(k, tmem_full, stages=r.tmem_slots, count=1)
            _init_stages(k, tmem_empty, stages=r.tmem_slots, count=CTA_GROUP)
            # CLC handshake (canon's barrier set).
            _init_stages(k, sched_arr_full, stages=1, count=1)
            _init_stages(k, sched_fin, stages=1, count=finish_arrivals)
            _init_stages(k, alloc_done, stages=1, count=1)
            if r.overlap:
                _init_stages(k, tmem_fin, stages=1, count=1)  # canon's tmem_fin (init_full=1)

    # Seal the mbarrier-init epoch.
    k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
    k.fence(kind=FenceKind.MBARRIER_INIT)

    # Cross-CTA prologue barrier.
    if r.overlap:
        k.cluster_barrier_arrive()
    else:
        k.cluster_sync()

    # tmem_alloc AFTER the prologue barrier (warp-collective on full warp 0):
    # the producers' first TMA flights overlap the pair's ~0.9us tcgen05.alloc
    # rendezvous instead of serializing behind it (nvjet hides it too — its
    # UTCATOMSWS atomic is effectively free). alloc_done releases the MMA warps.
    # Alloc only what the accumulator slots actually span (mma_n x tmem_slots
    # columns; 1024's 2x64=128 vs the full 512 pool — a smaller FIND_AND_SET
    # window shortens the pair-collective latency the MMA warp waits on).
    tmem_cols = max(32, r.tmem_slots * r.mma_n)
    with k.if_warp(0):
        k.tmem_alloc(0, tmem_cols, addr_byte_offset=tmem_addr_off, cta_group=CTA_GROUP)
        with k.if_elected():
            k.mbarrier_arrive(alloc_done, stage=0)

    # Producer warpgroup register reduction.
    with k.if_warpgroup(r.num_consumer):
        k.set_maxnreg(PRODUCER_MAXNREG)

        # ---- CLC scheduler warp (skipped under the static scheduler: there is
        # exactly one launch tile per cluster, nothing to steal).
        if not static_sched:
            with k.if_warp(r.producer_wg_base + 2):
                # cluster_barrier_wait is WARP-COLLECTIVE.
                if r.overlap:
                    k.cluster_barrier_wait()
                with k.if_elected():
                    with k.scheduler_impl(sched):
                        sf_phase = k.scalar(initial=1, dtype=ScalarDType.I32)
                        sa_phase = k.scalar(initial=0, dtype=ScalarDType.I32)
                        sched_done = k.scalar(initial=0, dtype=ScalarDType.I32)
                        with k.loop():
                            k.break_if(sched_done.ne(0))
                            with k.if_(cta_rank.eq(0)):
                                k.mbarrier_wait(sched_fin, stage=0, phase=sf_phase)
                                k.scalar_store(sf_phase, (sf_phase + 1) % 2)
                            k.mbarrier_arrive_expect_tx(
                                sched_arr_full, bytes=CLC_HANDLE_BYTES, stage=0
                            )
                            with k.if_(cta_rank.eq(0)):
                                k.clc_try_cancel(sched, clc_handle, sched_arr_full, stage=0)
                            k.mbarrier_wait(sched_arr_full, stage=0, phase=sa_phase)
                            k.scalar_store(sa_phase, (sa_phase + 1) % 2)
                            raw = k.clc_query_cancel(sched, clc_handle)
                            k.mbarrier_arrive(sched_fin_leader, stage=0)
                            with k.if_(raw < 0):
                                k.scalar_store(sched_done, 1)

        # ---- TMA loader (TIRx producer-wg warp 3) — single-elect.
        with k.if_warp(r.producer_wg_base + 3):
            if r.overlap:
                k.cluster_barrier_wait()
            with k.if_elected():
                # Loop-carried SMEM-ring.
                ld_stage = k.scalar(initial=0, dtype=ScalarDType.I32)
                ld_phase = k.scalar(initial=0, dtype=ScalarDType.I32)
                with task_loop() as (task_id, local_iter):
                    # Hoist the per-task tile coords into registers ONCE per task.
                    m_idx_e, n_idx_e = work_coords(task_id)
                    m_idx = k.let(m_idx_e)
                    n_idx = k.let(n_idx_e)
                    b_n = (n_idx * CTA_GROUP + cta_rank) * r.blk_n
                    with k.for_loop(stop=r.k_tiles) as kt:
                        k.mbarrier_wait(smem_empty, stage=ld_stage, phase=(ld_phase + 1) % 2)
                        tx = r.num_consumer * a_tile_bytes + b_tile_bytes
                        kc = kt * r.blk_k
                        for c in range(r.num_consumer):
                            a_m = ((m_idx * CTA_GROUP + cta_rank) * r.num_consumer + c) * BLK_M
                            k.tma_load(
                                TensorSlice(
                                    tensor=a_smem[c],
                                    offsets=(ld_stage, 0, 0),
                                    shape=(1, BLK_M, r.blk_k),
                                ),
                                a_gmem,
                                mbar=smem_full_cta0,
                                coords=(a_m, kc),
                                shape=(1, BLK_M, r.blk_k),
                                gmem_shape=(BLK_M, r.blk_k),
                                mbar_stage=ld_stage,
                                cta_group=CTA_GROUP,
                            )
                        k.tma_load(
                            TensorSlice(
                                tensor=b_smem, offsets=(ld_stage, 0, 0), shape=(1, r.blk_n, r.blk_k)
                            ),
                            b_gmem,
                            mbar=smem_full_cta0,
                            coords=(b_n, kc),
                            shape=(1, r.blk_n, r.blk_k),
                            gmem_shape=(r.blk_n, r.blk_k),
                            mbar_stage=ld_stage,
                            cta_group=CTA_GROUP,
                        )
                        # Canonical fp16 ordering.
                        with k.if_(cta_rank.eq(0)):
                            k.mbarrier_arrive_expect_tx(
                                smem_full_cta0, bytes=CTA_GROUP * tx, stage=ld_stage
                            )
                        _advance_ring(k, ld_stage, ld_phase, r.pipe_depth)

        # ---- MMA (TIRx producer-wg warps 0..NUM_CONSUMER-1, cluster leader only).
        for c in range(r.num_consumer):
            with k.if_warp(r.producer_wg_base + c):
                if r.overlap:
                    k.cluster_barrier_wait()
                # The tcgen05.alloc (post-barrier, on warp 0) must land before
                # this warp's first tcgen05 op — see the prologue reorder.
                k.mbarrier_wait(alloc_done, stage=0, phase=0)
                with k.if_elected():
                    with k.if_(cta_rank.eq(0)):
                        # Loop-carried SMEM-ring induction counter.
                        mma_stage = k.scalar(initial=0, dtype=ScalarDType.I32)
                        mma_phase = k.scalar(initial=0, dtype=ScalarDType.I32)
                        with task_loop() as (task_id, local_iter):
                            if r.overlap:
                                slot = local_iter % r.tmem_slots
                                tmem_empty_phase = (local_iter // r.tmem_slots + 1) % 2
                            else:
                                slot = c
                                tmem_empty_phase = (local_iter + 1) % 2
                            k.mbarrier_wait(tmem_empty, stage=slot, phase=tmem_empty_phase)
                            acc_op = accum.at(0, slot * r.mma_n)

                            # The ROLLED k-loop with a RUNTIME accum cell.
                            accum_flag = k.scalar(initial=0, dtype=ScalarDType.I32)
                            with k.for_loop(stop=r.k_tiles, unroll=False) as _kt:
                                k.mbarrier_wait(smem_full, stage=mma_stage, phase=mma_phase)
                                k.tcgen05_mma(
                                    acc_op,
                                    k.mma_a_smem(
                                        k.smem_tile(
                                            a_smem[c],
                                            prefix_indices=(mma_stage,),
                                            row_offset=0,
                                            col_offset=0,
                                            rows=BLK_M,
                                            cols=r.blk_k,
                                        )
                                    ),
                                    k.smem_tile(
                                        b_smem,
                                        prefix_indices=(mma_stage,),
                                        row_offset=0,
                                        col_offset=0,
                                        rows=r.blk_n,
                                        cols=r.blk_k,
                                    ),
                                    mma_m=CTA_M,
                                    mma_n=r.mma_n,
                                    format=("f16" if config.dtype == DType.F16 else "bf16"),
                                    block_scale=None,
                                    accum=accum_flag,
                                    trans_a=False,
                                    trans_b=False,
                                    ws=False,
                                    cta_group=CTA_GROUP,
                                )
                                k.scalar_store(accum_flag, 1)
                                k.tcgen05_commit(
                                    smem_empty,
                                    stage=mma_stage,
                                    cta_group=CTA_GROUP,
                                    multicast_cta_mask=0b11,
                                )
                                _advance_ring(k, mma_stage, mma_phase, r.pipe_depth)
                            k.tcgen05_commit(
                                tmem_full, stage=slot, cta_group=CTA_GROUP, multicast_cta_mask=0b11
                            )

    # ---- epilogue (TIRx consumer warpgroups 0..NUM_CONSUMER-1).
    for c in range(r.num_consumer):
        wg_bar = 10 + c  # one named warpgroup barrier per epilogue role
        with k.if_warpgroup(c):
            # Consumer epilogue register raise (no-overlap path only).
            if not r.overlap:
                k.set_maxnreg(CONSUMER_MAXNREG)
            if r.overlap:
                k.cluster_barrier_wait()
            with task_loop(wg_bar=wg_bar) as (task_id, local_iter):
                # The writeback fragments are declared FRESH per wb-tile inside the epilogue.
                local_iter = k.let(local_iter)
                m_idx_e, n_idx_e = work_coords(task_id)
                m_idx = k.let(m_idx_e)
                n_idx = k.let(n_idx_e)
                if r.overlap:
                    slot = local_iter % r.tmem_slots
                    tmem_full_phase = (local_iter // r.tmem_slots) % 2
                else:
                    # Use the RUNTIME warpgroup id.
                    slot = k.warpgroup_id()
                    tmem_full_phase = local_iter % 2
                k.mbarrier_wait(tmem_full, stage=slot, phase=tmem_full_phase)
                tmem_col0 = slot * r.mma_n
                d_m = ((m_idx * CTA_GROUP + cta_rank) * r.num_consumer + c) * BLK_M

                if r.overlap:
                    # Fused per-tile load+cast+store, overlapping the next MMA.
                    for ot in range(r.wb_pipe_depth):
                        # Fresh per-wb-tile cast fragment (canon's loop-local Dreg_16b).
                        out_wide = k.tensor(
                            space=MemorySpace.REG, dtype=config.dtype, shape=(r.epi_n,)
                        )
                        store_iter = local_iter * r.wb_pipe_depth + ot
                        # The ring-guard wait+sync is needed ONLY when the
                        # d_smem ring actually wraps (store_iter >= depth) —
                        # an unconditional per-band sync is pure latency at
                        # wb_pipe_depth <= num_d_tiles (e.g. 1024's 2 bands).
                        with k.if_(store_iter >= r.num_d_tiles):
                            k.cp_async_bulk_wait_group_read(r.num_d_tiles - 1)
                            k.wg_sync(barrier_id=wg_bar)
                        # Fold the D-smem buffer index the same way the SMEM ring does.
                        db = _ring_index(local_iter, ot, r.wb_pipe_depth, r.num_d_tiles)[0]
                        # Read the EPI_N band in read_w=32 chunks.
                        read_w = min(r.epi_n, 32)
                        for kc in range(r.epi_n // read_w):
                            chunk_frag = k.tensor(
                                space=MemorySpace.REG, dtype=DType.F32, shape=(read_w,)
                            )
                            col = tmem_col0 + ot * r.epi_n + kc * read_w
                            k.tcgen05_ld(
                                TensorSlice(tensor=chunk_frag, offsets=(0,), shape=(read_w,)),
                                accum.at(0, col),
                                num=read_w,
                            )
                            k.tcgen05_wait_ld()
                            k.reg_cvt(
                                TensorSlice(
                                    tensor=out_wide, offsets=(kc * read_w,), shape=(read_w,)
                                ),
                                TensorSlice(tensor=chunk_frag, offsets=(0,), shape=(read_w,)),
                            )
                        for kc in range(r.epi_n // TMEM_LD_SIZE):
                            k.reg_store(
                                TensorSlice(
                                    tensor=d_smem[c],
                                    offsets=(db, k.tid_in_wg(), kc * TMEM_LD_SIZE),
                                    shape=(1, 1, TMEM_LD_SIZE),
                                ),
                                TensorSlice(
                                    tensor=out_wide,
                                    offsets=(kc * TMEM_LD_SIZE,),
                                    shape=(TMEM_LD_SIZE,),
                                ),
                            )
                        # Warpgroup-sync FIRST.
                        k.wg_sync(barrier_id=wg_bar)
                        if ot == r.wb_pipe_depth - 1:
                            # Free the accumulator slot AFTER the wg_sync.
                            with k.if_(k.tid_in_wg().eq(0)):
                                k.mbarrier_arrive(tmem_empty_leader, stage=slot)
                        with k.if_(k.tid_in_wg().eq(0)):
                            k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                            k.tma_store(
                                d_gmem,
                                TensorSlice(
                                    tensor=d_smem[c], offsets=(db, 0, 0), shape=(1, BLK_M, r.epi_n)
                                ),
                                coords=(d_m, n_idx * r.mma_n + ot * r.epi_n),
                                shape=(1, BLK_M, r.epi_n),
                                gmem_shape=(BLK_M, r.epi_n),
                            )
                        k.cp_async_bulk_commit_group()
                else:
                    # No-overlap interleaves each writeback tile's load, cast, and store.
                    bands_per_tile = r.epi_n // NOL
                    for ot in range(r.wb_pipe_depth):
                        out_tile = k.tensor(
                            space=MemorySpace.REG, dtype=config.dtype, shape=(r.epi_n,)
                        )
                        for sub in range(bands_per_tile):
                            band_frag = k.tensor(
                                space=MemorySpace.REG, dtype=DType.F32, shape=(NOL,)
                            )
                            col = tmem_col0 + (ot * bands_per_tile + sub) * NOL
                            k.tcgen05_ld(
                                TensorSlice(tensor=band_frag, offsets=(0,), shape=(NOL,)),
                                accum.at(0, col),
                                num=NOL,
                            )
                            k.tcgen05_wait_ld()
                            k.reg_cvt(
                                TensorSlice(tensor=out_tile, offsets=(sub * NOL,), shape=(NOL,)),
                                TensorSlice(tensor=band_frag, offsets=(0,), shape=(NOL,)),
                            )
                        store_iter = local_iter * r.wb_pipe_depth + ot
                        with k.if_(store_iter >= r.num_d_tiles):
                            k.cp_async_bulk_wait_group_read(r.num_d_tiles - 1)
                        # wg_sync OUTSIDE the runtime branch.
                        k.wg_sync(barrier_id=wg_bar)
                        db = _ring_index(local_iter, ot, r.wb_pipe_depth, r.num_d_tiles)[0]
                        for kc in range(r.epi_n // TMEM_LD_SIZE):
                            k.reg_store(
                                TensorSlice(
                                    tensor=d_smem[c],
                                    offsets=(db, k.tid_in_wg(), kc * TMEM_LD_SIZE),
                                    shape=(1, 1, TMEM_LD_SIZE),
                                ),
                                TensorSlice(
                                    tensor=out_tile,
                                    offsets=(kc * TMEM_LD_SIZE,),
                                    shape=(TMEM_LD_SIZE,),
                                ),
                            )
                        # Warpgroup-sync FIRST, THEN a single-thread proxy fence.
                        k.wg_sync(barrier_id=wg_bar)
                        with k.if_(k.tid_in_wg().eq(0)):
                            k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                            k.tma_store(
                                d_gmem,
                                TensorSlice(
                                    tensor=d_smem[c], offsets=(db, 0, 0), shape=(1, BLK_M, r.epi_n)
                                ),
                                coords=(d_m, n_idx * r.mma_n + ot * r.epi_n),
                                shape=(1, BLK_M, r.epi_n),
                                gmem_shape=(BLK_M, r.epi_n),
                            )
                        k.cp_async_bulk_commit_group()
                    with k.if_(k.tid_in_wg().eq(0)):
                        k.mbarrier_arrive(tmem_empty_leader, stage=slot)
            # No-overlap keeps the drain before its cluster-wide teardown;
            # the overlap path drops it — the last task's TMA stores drain
            # on the HW side at kernel exit (D visibility is
            # completion-ordered), so the pair's dealloc rendezvous below
            # overlaps the final store flight instead of serializing behind
            # a full drain (nvjet's PREEXIT-style tail).
            if not r.overlap:
                k.cp_async_bulk_wait_group_read(0)
                k.wg_sync(barrier_id=wg_bar)

    # Cluster-wide barrier before freeing TMEM.
    teardown = config.teardown or GEMM_CONFIGS.get(config.n, {}).get("teardown")
    if r.overlap:
        if teardown != "none":
            if teardown != "dealloc_only":
                # canon's tmem_fin teardown.
                with k.if_warp(0):
                    with k.if_elected():
                        k.mbarrier_arrive(k.mbar_ref(tmem_fin, remote_coord=1 - cta_rank), stage=0)
                    k.mbarrier_wait(tmem_fin, stage=0, phase=0)
            with k.if_warp(0):
                k.tmem_relinquish(CTA_GROUP)
                k.tmem_dealloc(0, tmem_cols, CTA_GROUP)
        # "none": nvjet's form — plain EXIT, the HW frees TMEM at CTA exit.
    else:
        k.cluster_sync()
        with k.if_warp(0):
            k.tmem_relinquish(CTA_GROUP)
            k.tmem_dealloc(0, tmem_cols, CTA_GROUP)

    return k.build()


def _ring_index(local_iter, kt: int, k_tiles: int, pipe_depth: int):
    """SMEM-ring slot and occupancy parity for global k-step ``seq = local_iter*k_tiles + kt``."""
    q, rem = divmod(k_tiles, pipe_depth)
    stage_carry = (local_iter * rem) if rem else 0
    stage = (stage_carry + kt) % pipe_depth
    cross = ((local_iter * rem + kt) // pipe_depth) if rem else (kt // pipe_depth)
    qp = q % 2
    phase_carry = (local_iter * qp) if qp else 0
    occ_parity = (phase_carry + cross) % 2
    return stage, occ_parity


def _advance_ring(k: IRBuilder, stage_sc, phase_sc, pipe_depth: int) -> None:
    """Advance a persistent SMEM-ring."""
    # Snapshot stage+1 into its own scalar FIRST.
    nxt = k.scalar(initial=stage_sc + 1, dtype=ScalarDType.I32)
    with k.if_(nxt >= pipe_depth):
        k.scalar_store(nxt, 0)
        k.scalar_store(phase_sc, (phase_sc + 1) % 2)
    k.scalar_store(stage_sc, nxt)


@contextmanager
def _clc_worker_loop(
    k: IRBuilder,
    sched,
    clc_handle,
    sched_arr_full,
    sched_fin_leader,
    cluster_id,
    wg_bar=None,
    warp_sync=False,
):
    """A CLC worker's consume loop, written out explicitly in nymph IR."""
    task_id = k.scalar(initial=cluster_id, dtype=ScalarDType.I32)
    local_iter = k.scalar(initial=0, dtype=ScalarDType.I32)
    with k.loop():
        yield task_id, local_iter
        k.mbarrier_wait(sched_arr_full, stage=0, phase=local_iter % 2)
        raw = k.clc_query_cancel(sched, clc_handle)
        # Warpgroup consumers (epilogue).
        if wg_bar is not None:
            k.wg_sync(barrier_id=wg_bar)
            with k.if_(k.tid_in_wg().eq(0)):
                k.mbarrier_arrive(sched_fin_leader, stage=0)
        else:
            if warp_sync:
                # Full-warp consumers.
                k.warp_sync()
            k.mbarrier_arrive(sched_fin_leader, stage=0)
        k.break_if(raw < 0)
        k.scalar_store(task_id, raw // CTA_GROUP)
        k.scalar_store(local_iter, local_iter + 1)


def _init_stages(k: IRBuilder, mbar: MBar, *, stages: int, count: int) -> None:
    # Emit the per-stage `mbarrier.init` run (canon's grouped-init block).
    for i in range(stages):
        k.mbarrier_init(mbar, count=count, stage=i)


def _validate_config(config: Fp16Bf16GemmConfig, r: _Resolved) -> None:
    if config.dtype not in _ELEMENT_BYTES:
        raise ValueError("fp16_bf16_gemm dtype must be f16 or bf16")
    for name in ("m", "n", "k"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"fp16_bf16_gemm {name} must be a positive integer")
    if r.mma_n % CTA_GROUP != 0:
        raise ValueError("fp16_bf16_gemm mma_n must be divisible by cta_group")
    if r.blk_k % MMA_K != 0:
        raise ValueError("fp16_bf16_gemm blk_k must be a multiple of 16")
    if config.k % r.blk_k != 0:
        raise ValueError("fp16_bf16_gemm k must be divisible by blk_k")
    if config.m % (CTA_M * r.num_consumer) != 0:
        raise ValueError("fp16_bf16_gemm m must be divisible by 256 * num_consumer")
    if config.n % r.mma_n != 0:
        raise ValueError("fp16_bf16_gemm n must be divisible by mma_n")
    if r.epi_n % TMEM_LD_SIZE != 0 or r.mma_n % r.wb_pipe_depth != 0:
        raise ValueError("fp16_bf16_gemm mma_n // wb_pipe_depth must be a multiple of 8")
    if r.tmem_slots * r.mma_n > N_COLS_TMEM:
        raise ValueError("fp16_bf16_gemm accumulator exceeds the 512-column TMEM budget")
    if r.num_m_tiles > r.l2_group_size and r.num_m_tiles % r.l2_group_size != 0:
        raise ValueError("fp16_bf16_gemm num_m_tiles must be a multiple of l2_group_size")
    if r.pair_tasks < 1:
        raise ValueError("fp16_bf16_gemm resolves to zero tiles; check m/n vs the tile sizes")
    if r.scheduler not in ("clc", "static"):
        raise ValueError("fp16_bf16_gemm scheduler must be 'clc' or 'static'")


def _validate_launch_shape(launch_shape: LaunchShape) -> None:
    if not isinstance(launch_shape, tuple) or len(launch_shape) != 1:
        raise ValueError("fp16_bf16_gemm requires a 1D launch_shape")
    count = launch_shape[0]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("fp16_bf16_gemm launch_shape[0] must be a positive integer")
    if count % CTA_GROUP != 0:
        raise ValueError("fp16_bf16_gemm launch_shape[0] must be divisible by cta_group")


# Bench-suite interface (see bench/nymph_bench_guide.md).

KERNEL_META = {"name": "nymph_fp16_bf16_gemm", "category": "experimental", "compute_capability": 10}
CONFIGS = [
    {"dtype": d, "M": s, "N": s, "K": s, "label": f"{d}_{s}x{s}x{s}"}
    for d in ["fp16", "bf16"]
    for s in [1024, 2048, 4096, 8192, 16384]
]


def _compile_nymph(dtype, M, N, K):
    import importlib.util
    import os
    import tempfile

    import tvm

    from ..nymph_rs import kernel_to_tirx_source

    ndt = DType.F16 if dtype == "fp16" else DType.BF16
    src = kernel_to_tirx_source(build_fp16_bf16_gemm(Fp16Bf16GemmConfig(m=M, n=N, k=K, dtype=ndt)))
    p = os.path.join(tempfile.mkdtemp(prefix="nymph_gemm_"), "g.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nymph_gemm_emitted", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return tvm.compile(
        tvm.IRModule({"main": m.main}), tvm.target.Target("cuda"), tir_pipeline="tirx"
    )


def run_bench(dtype, M, N, K, *, warmup=None, repeat=None, timer=None, **kwargs):
    """Bench canon vs nymph with the bench-suite's exact methodology."""
    import torch

    from tirx_kernels.gemm.fp16_bf16_gemm import prepare_data, tir_kernel
    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    canon = compile_kernel(tir_kernel(dtype, M, N, K))
    nymph = _compile_nymph(dtype, M, N, K)
    a, b, c = prepare_data(dtype, M, N, K)
    oc, on, obl = (
        torch.zeros_like(c, device="cuda"),
        torch.zeros_like(c, device="cuda"),
        torch.zeros_like(c, device="cuda"),
    )
    funcs = {
        "tir": lambda: canon(a, b, oc),
        "tirx": lambda: nymph(a, b, on),
        # cuBLAS as a first-class impl (pure launch, preallocated out).
        "cublas": lambda: torch.matmul(a, b.T, out=obl),
    }
    for fn in funcs.values():
        fn()
    torch.cuda.synchronize()
    ref = torch.mm(a, b.T)
    for name, out in (("tir", oc), ("tirx", on), ("cublas", obl)):
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.float().flatten(), dim=0
        )
        if cos < 0.99:
            raise AssertionError(f"{name} output diverges from reference (cosine={cos:.4f})")
    return bench(funcs, warmup=warmup, repeat=repeat, timer=timer, **kwargs)


def register_bench_interface() -> None:
    """Self-register into the bench-suite kernel cache (see bench/nymph_bench_guide.md)."""
    import sys

    from tirx_kernels.registry import _KERNEL_CACHE

    _KERNEL_CACHE[KERNEL_META["name"]] = sys.modules[__name__]
