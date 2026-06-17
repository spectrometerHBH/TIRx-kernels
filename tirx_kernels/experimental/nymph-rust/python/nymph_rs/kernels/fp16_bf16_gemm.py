"""Static-shape fp16/bf16 GEMM expressed in Nymph IR.

Faithful port of ``tirx_kernels/gemm/fp16_bf16_gemm.py`` (the latest CLC +
overlapped-epilogue implementation), at the same modeling granularity as the
nvfp4 / fp8-blockwise / FA4 ports. The port keeps:

- the role split: a dedicated **scheduler** warp, a **TMA-load** warp, one
  **MMA** warp per consumer (issuing from the cluster leader only), and one
  **epilogue** warpgroup per consumer. The producer warpgroup is ``wg_id ==
  NUM_CONSUMER`` (scheduler = its warp 2, loader = its warp 3, MMA = its
  warps ``0 .. NUM_CONSUMER-1``); the epilogues are warpgroups
  ``0 .. NUM_CONSUMER-1`` — the exact TIRx ``wg_id`` / ``warp_id`` partition.

- the tile scheduler: TIRx drives this with a CLC (Cluster Launch Control)
  work-stealing scheduler. CLC has no executable semantics in the nymph value
  simulator, so — exactly like the FA4 port — the dedicated scheduler warp is
  modeled as a round-robin broadcast through an SMEM task mailbox (``task_full``
  / ``task_empty`` handshake). Unlike FA4 (cta_group=1), this is a *cluster*
  kernel: ``k.sched_next``'s cursor is keyed per-cluster, so both CTAs of a pair
  would share it and pull DIFFERENT (interleaved) tasks, desyncing the cluster
  MMA. The scheduler warp therefore walks the grid-stride sequence with a local
  counter (``task_id = task_start + n * task_step``) so both CTAs broadcast the
  SAME task stream. The group-major L2 tile rasterization (``L2_GROUP_SIZE``) is
  reproduced in ``work_coords``; the only thing abstracted is that the broadcast
  hands out a deterministic round-robin task instead of a dynamically stolen one.

- the pipeline protocol: a continuous-sequence multi-stage SMEM ring
  (``smem_full`` = TMA arrive-expect-tx; ``smem_empty`` = the leader's
  ``tcgen05_commit`` multicast to both CTAs, ``NUM_CONSUMER`` arrivals) and a
  ``TMEM_SLOTS``-deep accumulator pipeline (``tmem_full`` = ``tcgen05_commit``
  multicast; ``tmem_empty`` = both CTAs' epilogues arrive at the leader). The
  ring sequence ``seq = local_iter * k_tiles + k_tile`` is never reset per task,
  so ``stage = seq % PIPE_DEPTH`` stays aligned with the ``occ`` phase even when
  ``k_tiles % PIPE_DEPTH != 0``.

- the data path: the (m=256, cta_group=2) cluster MMA. Each CTA TMAs its own A
  rows and its own half of the B band; the leader issues the cluster MMA reading
  both CTAs' tiles (it waits its own and the peer's ``smem_full``), writing each
  CTA's 128-row accumulator half.

- the two epilogue paths, resolved per shape by ``OVERLAP_EPILOGUE`` (just like
  TIRx): OVERLAP (``NUM_CONSUMER=1``, double-buffered TMEM) fuses each EPI_N
  chunk's load+cast+store and frees the accumulator after the last chunk;
  no-overlap (``NUM_CONSUMER=2``) drains the whole accumulator into registers,
  frees it, then streams the store tiles.

Physical-layout details below the value model (SMEM swizzles, the stmatrix
register staging, the runtime epilogue scale) are modeled logically, exactly
like the nvfp4 / fp8 / FA4 ports. ``setmaxnreg`` is omitted (a codegen budget).
"""

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
    Tensor,
    TensorSlice,
    TmemLayout,
    TmemLayoutKind,
)

_ELEMENT_BYTES = {DType.F16: 2, DType.BF16: 2}

CTA_GROUP = 2
CTA_M = 256  # cluster M tile (2-SM MMA combines the pair)
BLK_M = CTA_M // CTA_GROUP  # per-CTA A rows = 128
MMA_K = 16  # fp16/bf16 MMA instruction K
TMEM_LD_SIZE = 8  # tcgen05.ld fragment width (cols per warp-collective load)
N_COLS_TMEM = 512
SM_NUMBER = 148
I32_BYTES = 4

# Task-broadcast mailbox (the FA4 custom-scheduler pattern). The scheduler warp
# fills slot `sched_iter % TASK_BROADCAST_STAGES`; every consumer role drains it.
# Two slots suffice: the mailbox only bounds how far the loader prefetches ahead
# of the epilogue (a throughput knob); the SMEM/TMEM ring barriers — not the
# mailbox — are what govern correctness, so a shallow ring never deadlocks.
TASK_BROADCAST_STAGES = 2
TASK_BROADCAST_FIELDS = 2
TASK_FIELD_ID, TASK_FIELD_ITER = range(TASK_BROADCAST_FIELDS)

# Per-shape tuning knobs (selected by N), mirroring TIRx GEMM_CONFIGS. CTA_M=256
# always; everything else is derived from these.
GEMM_CONFIGS = {
    1024: {
        "mma_n": 64,
        "blk_k": 128,
        "l2_group_size": 4,
        "overlap_epilogue": True,
        "pipe_depth": 5,
        "wb_pipe_depth": 2,
    },
    2048: {
        "mma_n": 256,
        "blk_k": 64,
        "l2_group_size": 8,
        "overlap_epilogue": True,
        "pipe_depth": 5,
        "wb_pipe_depth": 4,
    },
    4096: {
        "mma_n": 256,
        "blk_k": 64,
        "l2_group_size": 4,
        "overlap_epilogue": False,
        "pipe_depth": 4,
        "wb_pipe_depth": 8,
    },
    8192: {
        "mma_n": 256,
        "blk_k": 64,
        "l2_group_size": 8,
        "overlap_epilogue": False,
        "pipe_depth": 4,
        "wb_pipe_depth": 8,
    },
    16384: {
        "mma_n": 256,
        "blk_k": 64,
        "l2_group_size": 8,
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


class _Resolved:
    """Resolved geometry for one (config) — all the derived constants TIRx
    computes from the independent knobs."""

    __slots__ = (
        "blk_k",
        "blk_n",
        "epi_n",
        "k_groups",
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

        self.num_consumer = 1 if self.overlap else 2
        # OVERLAP double-buffers the single consumer's TMEM (MMA_PIPE=2);
        # no-overlap gives each consumer its own slot. Both come to 2 slots.
        self.tmem_slots = 2 if self.overlap else self.num_consumer
        self.num_d_tiles = 2 if self.wb_pipe_depth > 1 else 1
        self.blk_n = self.mma_n // CTA_GROUP  # per-CTA B rows
        self.epi_n = self.mma_n // self.wb_pipe_depth
        self.num_m_tiles = config.m // (CTA_M * self.num_consumer)
        self.num_n_tiles = config.n // self.mma_n
        self.pair_tasks = self.num_m_tiles * self.num_n_tiles
        self.k_tiles = config.k // self.blk_k
        self.k_groups = self.blk_k // MMA_K
        self.num_warps = (self.num_consumer + 1) * 4
        self.producer_wg_base = self.num_consumer * 4  # first warp of producer wg


def build_fp16_bf16_gemm(config: Fp16Bf16GemmConfig = Fp16Bf16GemmConfig()) -> Kernel:
    r = _Resolved(config)
    _validate_config(config, r)

    num_clusters = max(1, min(SM_NUMBER // CTA_GROUP, r.pair_tasks))
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
    task_off = d_base + r.num_consumer * r.num_d_tiles * d_tile_bytes
    smem_size_bytes = task_off + TASK_BROADCAST_STAGES * TASK_BROADCAST_FIELDS * I32_BYTES

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
    d_smem = tuple(
        k.tensor(
            space=MemorySpace.SMEM,
            dtype=config.dtype,
            shape=(r.num_d_tiles, BLK_M, r.epi_n),
            byte_offset=off,
        )
        for off in d_offsets
    )
    task_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.I32,
        shape=(TASK_BROADCAST_STAGES, TASK_BROADCAST_FIELDS),
        byte_offset=task_off,
    )

    # TMEM: one 512-col allocation; the accumulator is the leading TMEM_SLOTS *
    # MMA_N column band (each CTA holds its own 128 M-rows x MMA_N per slot).
    tmem_base = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F32,
        shape=(128, N_COLS_TMEM),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=0),
    )
    accum = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F32,
        shape=(128, r.tmem_slots * r.mma_n),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=0),
    )
    accum_frag = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(TMEM_LD_SIZE,))
    out_frag = k.tensor(space=MemorySpace.REG, dtype=config.dtype, shape=(TMEM_LD_SIZE,))
    # No-overlap holds the whole MMA_N result row in registers (TIRx Dreg_16b)
    # so the accumulator can be freed before any store.
    out_wide = k.tensor(space=MemorySpace.REG, dtype=config.dtype, shape=(r.mma_n,))

    smem_full = k.mbar(kind=MBarKind.TMA, stages=r.pipe_depth)
    smem_empty = k.mbar(kind=MBarKind.TCGEN05, stages=r.pipe_depth)
    tmem_full = k.mbar(kind=MBarKind.TCGEN05, stages=r.tmem_slots)
    tmem_empty = k.mbar(kind=MBarKind.THREAD, stages=r.tmem_slots)
    task_full = k.mbar(kind=MBarKind.THREAD, stages=TASK_BROADCAST_STAGES)
    task_empty = k.mbar(kind=MBarKind.THREAD, stages=TASK_BROADCAST_STAGES)
    # The cluster MMA reads the peer CTA's tiles too: the leader waits its own
    # and the peer's smem_full before issuing. The epilogues arrive at the
    # leader's tmem_empty.
    peer_smem_full = k.mbar_ref(smem_full, remote_coord=1)
    tmem_empty_leader = k.mbar_ref(tmem_empty, remote_coord=0)

    cta_id = k.cta_id()
    cta_rank = k.ctaid_in_cluster()
    # Persistent cluster grid-stride: cluster `task_start` strides by the cluster
    # count. The scheduler warp walks `task_id = task_start + n * task_step` (the
    # same round-robin `sched_next` computes) — but computed locally per CTA, NOT
    # via `k.sched_next`: its cursor is keyed per-cluster, so for cta_group=2 both
    # CTAs would share it and get DIFFERENT (interleaved) tasks, desyncing the
    # cluster MMA. A local counter makes both CTAs broadcast the SAME task stream.
    task_start = cta_id // CTA_GROUP
    task_step = k.launch_cta_count // CTA_GROUP
    # task_empty arrivals: loader + NUM_CONSUMER MMA warps + NUM_CONSUMER
    # epilogue warpgroups (the MMA warp consumes on both CTAs; only its issue is
    # leader-gated, so the count stays uniform across the cluster).
    task_consumer_count = 1 + 2 * r.num_consumer

    def work_coords(task_id):
        """ClusterLaunchControlScheduler group-major L2 raster
        (cluster_m=cluster_n=1): m tiles walk within an l2_group_size-row group,
        groups row-major. Returns (m_idx, n_idx)."""
        if r.num_m_tiles <= r.l2_group_size:
            return task_id % r.num_m_tiles, task_id // r.num_m_tiles
        group_span = r.l2_group_size * r.num_n_tiles
        group_id = task_id // group_span
        within = task_id % group_span
        m_idx = group_id * r.l2_group_size + within % r.l2_group_size
        n_idx = within // r.l2_group_size
        return m_idx, n_idx

    with k.kernel_init(warp=0):
        k.tmem_alloc(tmem_base, n_cols=N_COLS_TMEM, cta_group=CTA_GROUP)
        _init_stages(k, smem_full, stages=r.pipe_depth, count=1)
        _init_stages(k, smem_empty, stages=r.pipe_depth, count=r.num_consumer)
        _init_stages(k, tmem_full, stages=r.tmem_slots, count=1)
        _init_stages(k, tmem_empty, stages=r.tmem_slots, count=CTA_GROUP)
        _init_stages(k, task_full, stages=TASK_BROADCAST_STAGES, count=1)
        _init_stages(k, task_empty, stages=TASK_BROADCAST_STAGES, count=task_consumer_count)

    # ---- scheduler warp (TIRx producer-wg warp 2 / CLC run_scheduler) ----
    with k.role(warp=r.producer_wg_base + 2, elected=True):
        sched_iter = k.scalar(initial=0, dtype=ScalarDType.I32)
        bcast_id = k.scalar(initial=0, dtype=ScalarDType.I32)
        with k.loop():
            stage = sched_iter % TASK_BROADCAST_STAGES
            phase = (sched_iter // TASK_BROADCAST_STAGES) % 2
            k.mbarrier_wait(task_empty, stage=stage, phase=(phase + 1) % 2)
            # The cluster's sched_iter-th grid-stride task, or the sentinel once the
            # cluster has drained its share. Computed locally so both CTAs agree.
            k.scalar_store(bcast_id, task_start + sched_iter * task_step)
            with k.if_(bcast_id >= r.pair_tasks):
                k.scalar_store(bcast_id, -1)
            k.store_scalar(_task_slot(task_smem, stage, TASK_FIELD_ID), bcast_id)
            k.store_scalar(_task_slot(task_smem, stage, TASK_FIELD_ITER), sched_iter)
            k.mbarrier_arrive(task_full, stage=stage)
            k.break_if(bcast_id < 0)
            k.scalar_store(sched_iter, sched_iter + 1)

    # ---- TMA loader (TIRx producer-wg warp 3) ----
    with k.role(warp=r.producer_wg_base + 3):
        with _persistent_task_loop(k, task_smem, task_full, task_empty) as (task_id, local_iter):
            m_idx, n_idx = work_coords(task_id)
            b_n = (n_idx * CTA_GROUP + cta_rank) * r.blk_n
            for kt in range(r.k_tiles):
                seq = local_iter * r.k_tiles + kt
                stage = seq % r.pipe_depth
                occ = seq // r.pipe_depth
                k.mbarrier_wait(smem_empty, stage=stage, phase=(occ + 1) % 2)
                tx = r.num_consumer * a_tile_bytes + b_tile_bytes
                k.mbarrier_arrive_expect_tx(smem_full, bytes=tx, stage=stage)
                kc = kt * r.blk_k
                for c in range(r.num_consumer):
                    a_m = ((m_idx * CTA_GROUP + cta_rank) * r.num_consumer + c) * BLK_M
                    k.tma_load(
                        TensorSlice(
                            tensor=a_smem[c], offsets=(stage, 0, 0), shape=(1, BLK_M, r.blk_k)
                        ),
                        a_gmem,
                        mbar=smem_full,
                        bytes=a_tile_bytes,
                        coords=(a_m, kc),
                        shape=(1, BLK_M, r.blk_k),
                        gmem_shape=(BLK_M, r.blk_k),
                        mbar_stage=stage,
                    )
                k.tma_load(
                    TensorSlice(tensor=b_smem, offsets=(stage, 0, 0), shape=(1, r.blk_n, r.blk_k)),
                    b_gmem,
                    mbar=smem_full,
                    bytes=b_tile_bytes,
                    coords=(b_n, kc),
                    shape=(1, r.blk_n, r.blk_k),
                    gmem_shape=(r.blk_n, r.blk_k),
                    mbar_stage=stage,
                )

    # ---- MMA (TIRx producer-wg warps 0..NUM_CONSUMER-1, cluster leader only) ----
    for c in range(r.num_consumer):
        with k.role(warp=r.producer_wg_base + c):
            with _persistent_task_loop(k, task_smem, task_full, task_empty) as (
                task_id,
                local_iter,
            ):
                with k.if_(cta_rank.eq(0)):
                    if r.overlap:
                        slot = local_iter % r.tmem_slots
                        tmem_empty_phase = (local_iter // r.tmem_slots + 1) % 2
                    else:
                        slot = c
                        tmem_empty_phase = (local_iter + 1) % 2
                    k.mbarrier_wait(tmem_empty, stage=slot, phase=tmem_empty_phase)
                    acc_slice = TensorSlice(
                        tensor=accum, offsets=(0, slot * r.mma_n), shape=(128, r.mma_n)
                    )
                    for kt in range(r.k_tiles):
                        seq = local_iter * r.k_tiles + kt
                        stage = seq % r.pipe_depth
                        occ = seq // r.pipe_depth
                        k.mbarrier_wait(smem_full, stage=stage, phase=occ % 2)
                        k.mbarrier_wait(peer_smem_full, stage=stage, phase=occ % 2)
                        for kg in range(r.k_groups):
                            ko = kg * MMA_K
                            k.tcgen05_mma(
                                acc_slice,
                                TensorSlice(
                                    tensor=a_smem[c],
                                    offsets=(stage, 0, ko),
                                    shape=(1, BLK_M, MMA_K),
                                ),
                                TensorSlice(
                                    tensor=b_smem, offsets=(stage, 0, ko), shape=(1, r.blk_n, MMA_K)
                                ),
                                m=CTA_M,
                                n=r.mma_n,
                                k=MMA_K,
                                accum=(kt > 0 or kg > 0),
                                cta_group=CTA_GROUP,
                            )
                        k.tcgen05_commit(
                            smem_empty, stage=stage, cta_group=CTA_GROUP, multicast_cta_mask=0b11
                        )
                    k.tcgen05_commit(
                        tmem_full, stage=slot, cta_group=CTA_GROUP, multicast_cta_mask=0b11
                    )

    # ---- epilogue (TIRx consumer warpgroups 0..NUM_CONSUMER-1) ----
    for c in range(r.num_consumer):
        wg_bar = 10 + c  # one named warpgroup barrier per epilogue role
        with k.role(warpgroup=c):
            with _persistent_task_loop(k, task_smem, task_full, task_empty) as (
                task_id,
                local_iter,
            ):
                m_idx, n_idx = work_coords(task_id)
                if r.overlap:
                    slot = local_iter % r.tmem_slots
                    tmem_full_phase = (local_iter // r.tmem_slots) % 2
                else:
                    slot = c
                    tmem_full_phase = local_iter % 2
                k.mbarrier_wait(tmem_full, stage=slot, phase=tmem_full_phase)
                tmem_col0 = slot * r.mma_n
                d_m = ((m_idx * CTA_GROUP + cta_rank) * r.num_consumer + c) * BLK_M

                if r.overlap:
                    # Fused per-tile load+cast+store, overlapping the next MMA.
                    for ot in range(r.wb_pipe_depth):
                        store_iter = local_iter * r.wb_pipe_depth + ot
                        with k.if_(store_iter >= r.num_d_tiles):
                            k.cp_async_bulk_wait_group_read(r.num_d_tiles - 1)
                            k.wg_sync(barrier_id=wg_bar)
                        db = store_iter % r.num_d_tiles
                        for kc in range(r.epi_n // TMEM_LD_SIZE):
                            col = tmem_col0 + ot * r.epi_n + kc * TMEM_LD_SIZE
                            k.tcgen05_ld(accum_frag, accum, num=TMEM_LD_SIZE, row=0, col=col)
                            k.tcgen05_wait_ld()
                            k.reg_cvt(out_frag, accum_frag)
                            k.reg_store(
                                TensorSlice(
                                    tensor=d_smem[c],
                                    offsets=(db, k.tid_in_wg(), kc * TMEM_LD_SIZE),
                                    shape=(1, 1, TMEM_LD_SIZE),
                                ),
                                out_frag,
                            )
                        if ot == r.wb_pipe_depth - 1:
                            k.mbarrier_arrive(tmem_empty_leader, stage=slot)
                        k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                        k.wg_sync(barrier_id=wg_bar)
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
                    # No-overlap: drain the whole accumulator into registers,
                    # free it, then stream the store tiles.
                    for kc in range(r.mma_n // TMEM_LD_SIZE):
                        col = tmem_col0 + kc * TMEM_LD_SIZE
                        k.tcgen05_ld(accum_frag, accum, num=TMEM_LD_SIZE, row=0, col=col)
                        k.tcgen05_wait_ld()
                        k.reg_cvt(
                            TensorSlice(
                                tensor=out_wide, offsets=(kc * TMEM_LD_SIZE,), shape=(TMEM_LD_SIZE,)
                            ),
                            accum_frag,
                        )
                    k.mbarrier_arrive(tmem_empty_leader, stage=slot)
                    for ot in range(r.wb_pipe_depth):
                        store_iter = local_iter * r.wb_pipe_depth + ot
                        with k.if_(store_iter >= r.num_d_tiles):
                            k.cp_async_bulk_wait_group_read(r.num_d_tiles - 1)
                            k.wg_sync(barrier_id=wg_bar)
                        db = store_iter % r.num_d_tiles
                        for kc in range(r.epi_n // TMEM_LD_SIZE):
                            k.reg_store(
                                TensorSlice(
                                    tensor=d_smem[c],
                                    offsets=(db, k.tid_in_wg(), kc * TMEM_LD_SIZE),
                                    shape=(1, 1, TMEM_LD_SIZE),
                                ),
                                TensorSlice(
                                    tensor=out_wide,
                                    offsets=(ot * r.epi_n + kc * TMEM_LD_SIZE,),
                                    shape=(TMEM_LD_SIZE,),
                                ),
                            )
                        k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                        k.wg_sync(barrier_id=wg_bar)
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
            # Drain in-flight TMA stores once after the persistent loop (TIRx
            # drains before the tmem teardown). A per-task drain would leave the
            # NEXT task's pacing wait_group with no committed group to release on.
            k.cp_async_bulk_wait_group_read(0)
            k.wg_sync(barrier_id=wg_bar)

    with k.kernel_finalize(warp=0):
        k.tmem_dealloc(tmem_base, n_cols=N_COLS_TMEM, cta_group=CTA_GROUP)

    return k.build()


@contextmanager
def _persistent_task_loop(k: IRBuilder, task_smem: Tensor, task_full: MBar, task_empty: MBar):
    """Consume scheduler-mailbox entries until the sentinel task id is seen."""
    consumer_iter = k.scalar(initial=0, dtype=ScalarDType.I32)
    with k.loop():
        stage = consumer_iter % TASK_BROADCAST_STAGES
        phase = (consumer_iter // TASK_BROADCAST_STAGES) % 2
        k.mbarrier_wait(task_full, stage=stage, phase=phase)
        task_id = k.scalar(
            initial=_task_slot(task_smem, stage, TASK_FIELD_ID), dtype=ScalarDType.I32
        )
        local_iter = k.scalar(
            initial=_task_slot(task_smem, stage, TASK_FIELD_ITER), dtype=ScalarDType.I32
        )
        k.mbarrier_arrive(task_empty, stage=stage)
        k.break_if(task_id < 0)
        yield task_id, local_iter
        k.scalar_store(consumer_iter, consumer_iter + 1)


def _task_slot(tensor: Tensor, stage, field: int) -> TensorSlice:
    return TensorSlice(tensor=tensor, offsets=(stage, field), shape=(1, 1))


def _init_stages(k: IRBuilder, mbar: MBar, *, stages: int, count: int) -> None:
    for stage in range(stages):
        k.mbarrier_init(mbar, count=count, stage=stage)


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


def _validate_launch_shape(launch_shape: LaunchShape) -> None:
    if not isinstance(launch_shape, tuple) or len(launch_shape) != 1:
        raise ValueError("fp16_bf16_gemm requires a 1D launch_shape")
    count = launch_shape[0]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("fp16_bf16_gemm launch_shape[0] must be a positive integer")
    if count % CTA_GROUP != 0:
        raise ValueError("fp16_bf16_gemm launch_shape[0] must be divisible by cta_group")
