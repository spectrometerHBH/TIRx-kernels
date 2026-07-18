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

- the tile scheduler: the **CLC (Cluster Launch Control)** work-stealing
  scheduler, written out EXPLICITLY in nymph IR (policy="custom") with the same
  primitive sequence as the canonical kernel — no software mailbox, no library
  macro. The dedicated scheduler warp issues ``clc_try_cancel`` into a 16B SMEM
  handle; the scheduler and every worker ``clc_query_cancel`` it for the next
  stolen tile, handshaking through ``sched_arr`` (the 16B response barrier, filled
  by try_cancel's tx, multicast to both CTAs) and ``sched_fin`` (every participant
  arrives once per task at the leader before the next steal). Each cluster
  processes its own launch tile (``cluster_id``) first, then steals via CLC. The
  two CLC ops are first-class nymph IR leaf ops: the value simulator returns the
  canonical round-robin task for ``clc_query_cancel`` (the trusted seam) and runs
  the real handshake, while codegen translates each op 1:1 to ``T.ptx.clc_*`` — so
  the emitted TIRx reproduces canon's CLC (``UGETNEXTWORKID.BROADCAST``) verbatim.
  The group-major L2 tile rasterization (``L2_GROUP_SIZE``) is in ``work_coords``.

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
like the nvfp4 / fp8 / FA4 ports. The no-overlap path carries canon's
``setmaxnreg`` register rebalance (see PRODUCER_MAXNREG / CONSUMER_MAXNREG) so
the consumer epilogue holds its MMA_N-wide writeback fragment without spilling.
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
    TensorSlice,
    TmemLayout,
    TmemLayoutKind,
)

_ELEMENT_BYTES = {DType.F16: 2, DType.BF16: 2}

CTA_GROUP = 2
CTA_M = 256  # cluster M tile (2-SM MMA combines the pair)
BLK_M = CTA_M // CTA_GROUP  # per-CTA A rows = 128
MMA_K = 16  # fp16/bf16 MMA instruction K
# Per-warpgroup register rebalance for the no-overlap epilogue (canon's setmaxnreg):
# drop the producer warpgroup, raise the consumers so the MMA_N-wide writeback fragment
# stays in registers (no LDL/STL spill). (56, 224) is canon's exact pair; with two
# consumer warpgroups it sums to 128*(56+224+224) = 64512 regs, inside the 65536/SM
# file. This pair is what every bench-suite run and GPU-numerics check has validated.
PRODUCER_MAXNREG = 56
CONSUMER_MAXNREG = 224
TMEM_LD_SIZE = 8  # tcgen05.ld fragment width (cols per warp-collective load)
NOL = 2 * TMEM_LD_SIZE  # no-overlap tmem->reg f32 read band (canonical `NOL`=16)
N_COLS_TMEM = 512
SM_NUMBER = 148
I32_BYTES = 4
# CLC (Cluster Launch Control) response handle: a 16-byte (uint32[4]) SMEM cell that
# `clusterlaunchcontrol.try_cancel` writes and `query_cancel` decodes.
CLC_HANDLE_WORDS = 4
CLC_HANDLE_BYTES = CLC_HANDLE_WORDS * 4

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
        # wb_pipe_depth=8 (EPI_N=32, not 64): the wider 64-col writeback fragment spilled
        # under the overlap path's 255-reg ceiling (216 B); the 32-col tile fits in registers.
        "wb_pipe_depth": 8,
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
        # blk_k=64/pipe_depth=4 — matches the canonical fp16/bf16 GEMM_CONFIGS[8192] (cta_k=64,
        # pipe_depth=4). The earlier blk_k=128/pipe_depth=2 was tuned under the OLD data-asymmetric
        # bench (canon fed real data, nymph random); under the FAIR bench (both fed identical data)
        # the canon-matching 64/4 benches ~1.00 vs the 128/2 form's ~0.979.
        "blk_k": 64,
        "l2_group_size": 8,
        "overlap_epilogue": False,
        "pipe_depth": 4,
        "wb_pipe_depth": 8,
    },
    16384: {
        "mma_n": 256,
        # blk_k=128/pipe_depth=2 (same SMEM as 64/4): fed-bound at ~96% tensor-active, the
        # wider-K ring keeps the tensor core fed — 1.000 -> 1.007 (min 0.989 -> 0.997). Same
        # lever as 8192.
        "blk_k": 128,
        "l2_group_size": 8,
        "overlap_epilogue": False,
        "pipe_depth": 2,
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

    # Scheduling: the hardware CLC (Cluster Launch Control) work-stealing scheduler — the
    # SAME scheduler canon uses, written out explicitly in nymph IR (a dedicated scheduler
    # warp issues clc_try_cancel/query_cancel; workers steal tiles). One cluster per tile
    # (oversubscribed, like canon), so num_clusters == pair_tasks.
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
    # Scheduling: the hardware CLC (Cluster Launch Control) scheduler, written out
    # explicitly in nymph IR exactly like the canonical TIRx kernel (policy="custom").
    # The scheduler warp issues `clc_try_cancel` into a 16B handle; the scheduler and
    # every worker `clc_query_cancel` it for the next tile. No software mailbox, no
    # grid-stride formula — the same primitive sequence canon emits, so codegen (a pure
    # 1:1 translator) reproduces canon's SASS (UGETNEXTWORKID.BROADCAST).
    clc_off = (pool_off + 15) // 16 * 16  # the CLC response handle is 16B-aligned
    smem_size_bytes = clc_off + CLC_HANDLE_BYTES

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
    # CLC response handle: a 16B (uint32[4]) SMEM cell the scheduler's `clc_try_cancel`
    # fills and `clc_query_cancel` decodes. The kernel's own object (policy="custom").
    clc_handle = k.tensor(
        space=MemorySpace.SMEM, dtype=DType.U32, shape=(CLC_HANDLE_WORDS,), byte_offset=clc_off
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
    # tmem->reg epilogue fragments, sized to the canonical's register footprint so
    # the generated `wg_reg_tile` matches (a wider f32 read spills; a wider cast frag
    # drops STSM dispatch -> STS.128). A whole band's tcgen05.ld atoms are read into
    # distinct sub-slices BEFORE a single wait_ld (one wait per band, not per 8-col
    # atom, matching the canonical fence structure). OVERLAP reads/casts one EPI_N
    # band at a time (Dreg/Dreg_16b both EPI_N); no-overlap stages the f32 read in
    # NOL=16 sub-bands (Dreg=NOL) but accumulates the whole MMA_N cast row
    # (Dreg_16b=MMA_N) so the accumulator can be freed before any store.
    # NOTE: these REG fragments are declared INSIDE the epilogue task loop (below), NOT
    # here at function scope — canon declares its `r_local_ptr[256]`/`local_32b_ptr[16]`
    # inside the `while(1)` task loop, so ptxas sees them as loop-local (fresh each task)
    # and promotes them to registers; a function-scope array is whole-kernel-lived and
    # stays in LOCAL memory (the LDL/STL spill, ~12x canon's). Verified via the nvrtc CUDA C.
    _frag_f32_width = r.epi_n if r.overlap else NOL
    _frag_out_width = r.epi_n if r.overlap else r.mma_n

    smem_full = k.mbar(kind=MBarKind.TMA, stages=r.pipe_depth)
    smem_empty = k.mbar(kind=MBarKind.TCGEN05, stages=r.pipe_depth)
    tmem_full = k.mbar(kind=MBarKind.TCGEN05, stages=r.tmem_slots)
    tmem_empty = k.mbar(kind=MBarKind.THREAD, stages=r.tmem_slots)
    # ---- CLC scheduler objects (policy="custom", written out in nymph IR) ----
    # The task space the CLC scheduler rasters: one task per cluster tile (m, n).
    sched_space = k.task_space(grid=(r.num_m_tiles, r.num_n_tiles), fields=("m_idx", "n_idx"))
    sched = k.scheduler(sched_space, policy="custom")
    # sched_arr: the 16B handle barrier. `clc_try_cancel` completes-tx it (multicast to
    # both cluster CTAs); the scheduler and every worker wait it before decoding. A TMA
    # barrier (the response lands as a 16B tx), single-stage — the scheduler is lockstep
    # with the consumers via sched_fin (exactly canon's stages=1 sched_arr).
    sched_arr_full = k.mbar(kind=MBarKind.TMA, stages=1)
    # sched_fin: every worker loop arrives once per task at CTA-0's barrier; the leader
    # scheduler waits it before issuing the next try_cancel (the slot-reuse gate). The
    # MMA runs on the cluster LEADER only (canon's `warp_id < NC & cbx == 0`); the loader,
    # epilogue, and scheduler run on both CTAs — so arrivals/task = scheduler(2) +
    # loader(2) + epilogue(2*NC) + MMA(NC) = (2 + NC) * 2 + NC (canon's `finish_arrivals`).
    finish_arrivals = (2 + r.num_consumer) * 2 + r.num_consumer
    sched_fin = k.mbar(kind=MBarKind.THREAD, stages=1)
    sched_fin_leader = k.mbar_ref(sched_fin, remote_coord=0)
    # tmem_fin: canon's lightweight 2-CTA teardown handshake (overlap path) — each CTA's
    # warp0/lane0 arrives at the PEER CTA's barrier, then warp0 waits its own, so both CTAs
    # finished their epilogue's TMEM reads before either deallocs. Lighter than a full
    # cluster_sync (only warp0 waits; the other 7 warps exit immediately).
    tmem_fin = k.mbar(kind=MBarKind.THREAD, stages=1) if r.overlap else None
    # (No sched_sync rendezvous — canon's barrier set is just sched_arr + sched_fin. The
    # scheduler arms expect_tx on both CTAs before the leader's try_cancel, so the response
    # tx never races ahead of the peer's expect_tx.)
    # The cluster MMA reads the peer CTA's tiles too: the leader waits its own
    # and the peer's smem_full before issuing. The epilogues arrive at the
    # leader's tmem_empty.
    peer_smem_full = k.mbar_ref(smem_full, remote_coord=1)
    tmem_empty_leader = k.mbar_ref(tmem_empty, remote_coord=0)

    cta_id = k.cta_id()
    cta_rank = k.ctaid_in_cluster()
    # Per-cluster identity: cluster_id is this cluster's launch tile; the CLC scheduler steals
    # the rest of the tiles via the handle (both CTAs share cluster_id, so the cluster MMA
    # stays in lockstep).
    cluster_id = cta_id // CTA_GROUP

    # Per-role task source: the CLC consume loop (each worker steals tiles via the handle).
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
        # CLC handshake (canon's barrier set): sched_arr.full takes one thread-arrive (the
        # scheduler's arrive.expect_tx) plus the 16B tx from try_cancel; sched_fin gathers
        # every participant's per-task arrival before the leader issues the next try_cancel.
        _init_stages(k, sched_arr_full, stages=1, count=1)
        _init_stages(k, sched_fin, stages=1, count=finish_arrivals)
        if r.overlap:
            _init_stages(k, tmem_fin, stages=1, count=1)  # canon's tmem_fin (init_full=1)

    # Seal the mbarrier-init epoch: a `fence.mbarrier_init` makes the inits above visible to
    # the async engines before any role touches a barrier (canon's `T.ptx.fence.mbarrier_init`).
    # Written EXPLICITLY here — codegen does not fabricate it.
    k.fence(kind=FenceKind.MBARRIER_INIT)

    # Cross-CTA prologue barrier — written EXPLICITLY here (codegen never fabricates it or
    # branches on overlap). OVERLAP splits it: a collective `cluster_barrier_arrive` here (all
    # threads, after the mbarrier inits) + a per-role `cluster_barrier_wait`, so the barrier
    # latency overlaps each role's setup and idle warps skip the wait. NO-OVERLAP uses a fused
    # `proxy_async + cluster_sync` (a full cluster barrier converges every CTA thread before
    # any role touches a peer mbar). The split is verified by the protocol checker (each role
    # must wait before any cross-CTA peer-mbarrier access).
    if r.overlap:
        k.cluster_barrier_arrive()
    else:
        k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
        k.cluster_sync()

    # Producer warpgroup register reduction. canon emits `setmaxnreg(False, 56)` on the
    # producer warpgroup in BOTH the overlap and no-overlap paths (it gates only the
    # consumer RAISE on `not OVERLAP_EPILOGUE`, never the producer drop). Dropping the
    # producer frees register-file budget → higher occupancy (matters most on the small
    # latency-bound shapes 1024/2048, which use the overlap path).
    with k.role(warpgroup=r.num_consumer, maxnreg=PRODUCER_MAXNREG):
        pass

    # ---- CLC scheduler warp (producer-wg warp 2, both CTAs) ----
    # Written out explicitly in nymph IR (policy="custom"), the same primitive sequence as
    # canon's `run_scheduler`: the dedicated scheduler warp issues clc_try_cancel and hands
    # tiles to the workers through sched_arr / sched_fin.
    with k.role(warp=r.producer_wg_base + 2, elected=True):
        if r.overlap:
            k.cluster_barrier_wait()
        with k.scheduler_impl(sched):
            sched_iter = k.scalar(initial=0, dtype=ScalarDType.I32)
            with k.loop():
                # canon's CLC scheduler handshake (barrier set = sched_arr + sched_fin, NO extra
                # sched_sync rendezvous). The leader waits the finished barrier; BOTH CTAs arm
                # expect_tx; the leader then issues try_cancel (its multicast response tx lands
                # in the already-armed sched_arr); both wait the handle. (sched_fin starts
                # satisfied at phase 1 so iter 0 runs.) The earlier nymph "expect_tx/tx race on
                # 16384" that motivated the sched_sync rendezvous was actually the elect-only
                # cluster.wait deadlock (now fixed) — verified race-free on 16384x5 without it.
                with k.if_(cta_rank.eq(0)):
                    k.mbarrier_wait(sched_fin, stage=0, phase=(sched_iter + 1) % 2)
                k.mbarrier_arrive_expect_tx(sched_arr_full, bytes=CLC_HANDLE_BYTES, stage=0)
                with k.if_(cta_rank.eq(0)):
                    k.clc_try_cancel(sched, clc_handle, sched_arr_full, stage=0)
                k.mbarrier_wait(sched_arr_full, stage=0, phase=sched_iter % 2)
                raw = k.clc_query_cancel(sched, clc_handle)
                k.mbarrier_arrive(sched_fin_leader, stage=0)
                k.break_if(raw < 0)
                k.scalar_store(sched_iter, sched_iter + 1)

    # ---- TMA loader (TIRx producer-wg warp 3) — single-elect (canon's `if elect_sync():
    # while: tma_load`): one issuing thread runs the whole loop, collapsing the per-op
    # elect guards (ELECT/BRA) that nymph otherwise emits at every TMA. No tcgen05 here,
    # so single-elect is checker-legal. ----
    with k.role(warp=r.producer_wg_base + 3, elected=True):
        if r.overlap:
            k.cluster_barrier_wait()
        # Loop-carried SMEM-ring (stage, phase) induction counter (canon's PipelineState.advance):
        # incremented by 1 per k-tile and wrapped at pipe_depth. Cheaper than recomputing
        # `(local_iter*k_tiles+kt) % pipe_depth` each iteration (which regressed: the per-iter
        # modulo/shift went vector + spilled). Walked inside a T.serial loop (uniform iteration).
        ld_stage = k.scalar(initial=0, dtype=ScalarDType.I32)
        ld_phase = k.scalar(initial=0, dtype=ScalarDType.I32)
        with task_loop() as (task_id, local_iter):
            # Hoist the per-task tile coords into registers ONCE per task (canon's
            # `buffer_5`/`buffer_6`), instead of re-deriving the L2-raster expression
            # `(task>>6)*4 + (task&63)&3` inline in every TMA address (3x/k-tile here).
            # shuffle_sync: the task id comes from a scheduler-mailbox LDS (a vector
            # load ptxas can't prove warp-uniform), but the TMA issue is a UNIFORM-pipe
            # instruction (UTMALDG) whose coords live in URs — unwrapped, every TMA
            # issue pays vector math + R2UR moves (ncu fp16 1024: R2UR 13.5K vs canon
            # 1.8K, clustered on the two g2cluster lines). Same treatment as nvfp4's
            # task loops (which wrap under elected=True as well).
            m_idx_e, n_idx_e = work_coords(task_id)
            m_idx = k.scalar(initial=m_idx_e, dtype=ScalarDType.I32)
            n_idx = k.scalar(initial=n_idx_e, dtype=ScalarDType.I32)
            b_n = (n_idx * CTA_GROUP + cta_rank) * r.blk_n
            with k.for_loop(stop=r.k_tiles) as kt:
                k.mbarrier_wait(smem_empty, stage=ld_stage, phase=(ld_phase + 1) % 2)
                tx = r.num_consumer * a_tile_bytes + b_tile_bytes
                k.mbarrier_arrive_expect_tx(smem_full, bytes=tx, stage=ld_stage)
                kc = kt * r.blk_k
                for c in range(r.num_consumer):
                    a_m = ((m_idx * CTA_GROUP + cta_rank) * r.num_consumer + c) * BLK_M
                    k.tma_load(
                        TensorSlice(
                            tensor=a_smem[c], offsets=(ld_stage, 0, 0), shape=(1, BLK_M, r.blk_k)
                        ),
                        a_gmem,
                        mbar=smem_full,
                        bytes=a_tile_bytes,
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
                    mbar=smem_full,
                    bytes=b_tile_bytes,
                    coords=(b_n, kc),
                    shape=(1, r.blk_n, r.blk_k),
                    gmem_shape=(r.blk_n, r.blk_k),
                    mbar_stage=ld_stage,
                    cta_group=CTA_GROUP,
                )
                _advance_ring(k, ld_stage, ld_phase, r.pipe_depth)

    # ---- MMA (TIRx producer-wg warps 0..NUM_CONSUMER-1, cluster leader only) ----
    # Single-elect the whole MMA worker loop (canon's `if elect_sync(): while ...`): one
    # elect at the role top, so every single-issue op inside (mbar waits, gemm, commits)
    # inherits it instead of carrying its own ELECT/VOTEU/PLOP3 guard.
    for c in range(r.num_consumer):
        with k.role(warp=r.producer_wg_base + c, elected=True):
            if r.overlap:
                k.cluster_barrier_wait()
            with k.if_(cta_rank.eq(0)):
                # Loop-carried SMEM-ring induction counter (canon's PipelineState), walked by
                # the same +1/wrap advance as the loader so the rolled k-loop's operand-
                # descriptor address math is one increment per iter (NOT a per-iter modulo).
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
                    acc_slice = TensorSlice(
                        tensor=accum, offsets=(0, slot * r.mma_n), shape=(128, r.mma_n)
                    )

                    def _mma_kstep(first_ktile):
                        # One k-tile: wait both smem_full barriers, issue the k_groups k=16
                        # MMAs (codegen collapses them to one gemm_async over BLK_K), commit
                        # smem_empty, advance the ring. accum=False ONLY on the first MMA of the
                        # first k-tile (overwrite the accumulator), else True — kept a Python
                        # bool by peeling the first k-tile.
                        k.mbarrier_wait(smem_full, stage=mma_stage, phase=mma_phase)
                        k.mbarrier_wait(peer_smem_full, stage=mma_stage, phase=mma_phase)
                        for kg in range(r.k_groups):
                            ko = kg * MMA_K
                            k.tcgen05_mma(
                                acc_slice,
                                TensorSlice(
                                    tensor=a_smem[c],
                                    offsets=(mma_stage, 0, ko),
                                    shape=(1, BLK_M, MMA_K),
                                ),
                                TensorSlice(
                                    tensor=b_smem,
                                    offsets=(mma_stage, 0, ko),
                                    shape=(1, r.blk_n, MMA_K),
                                ),
                                m=CTA_M,
                                n=r.mma_n,
                                k=MMA_K,
                                accum=((not first_ktile) or kg > 0),
                                cta_group=CTA_GROUP,
                            )
                        k.tcgen05_commit(
                            smem_empty,
                            stage=mma_stage,
                            cta_group=CTA_GROUP,
                            multicast_cta_mask=0b11,
                        )
                        _advance_ring(k, mma_stage, mma_phase, r.pipe_depth)

                    # Peel the first k-tile (accum overwrite); ROLL the remaining k_tiles-1
                    # with accum=True via a serial loop (uniform iteration count).
                    _mma_kstep(first_ktile=True)
                    with k.for_loop(stop=r.k_tiles - 1) as _kt:
                        _mma_kstep(first_ktile=False)
                    k.tcgen05_commit(
                        tmem_full, stage=slot, cta_group=CTA_GROUP, multicast_cta_mask=0b11
                    )

    # ---- epilogue (TIRx consumer warpgroups 0..NUM_CONSUMER-1) ----
    for c in range(r.num_consumer):
        wg_bar = 10 + c  # one named warpgroup barrier per epilogue role
        # Consumer epilogue register raise (no-overlap path only): holds the MMA_N-wide
        # writeback fragment in registers instead of spilling. Overlap keeps a narrow
        # EPI_N fragment and needs no raise (canon gates this on `not OVERLAP_EPILOGUE`).
        cons_maxnreg = CONSUMER_MAXNREG if not r.overlap else None
        with k.role(warpgroup=c, maxnreg=cons_maxnreg):
            if r.overlap:
                k.cluster_barrier_wait()
            with task_loop(wg_bar=wg_bar) as (task_id, local_iter):
                # The writeback fragments are declared FRESH per wb-tile inside the epilogue
                # readback loops below (canon's loop-local Dreg/Dreg_16b) so ptxas promotes them
                # to registers; a task- or function-scope fragment held across the store
                # warpgroup-syncs spills to local memory (measured: 489K LDL -> 0).
                # Hoist tile coords once/task (canon's `cur_m`/`cur_n`) — the epilogue
                # store address re-derives the same L2-raster expression ×8 otherwise.
                # shuffle_sync makes them compiler-provably warp-UNIFORM so the derived
                # epilogue address/phase math lowers to the uniform datapath (UIADD3/
                # UISETP/USEL like canon) instead of vector IMAD/ISETP/SEL + per-use
                # R2UR conversions — the nvfp4 local_iter treatment (ncu fp16 1024:
                # nymph R2UR 13.5K vs canon 1.8K, IMAD +9.4K, ISETP +8.3K; isolated
                # duration 16% over canon). Wrap only the CONSUMER's derived values
                # (wrapping raw task_id in the producer paths was the June bf16-8192
                # pathological-compile case).
                local_iter = k.shuffle_sync(local_iter)
                m_idx_e, n_idx_e = work_coords(task_id)
                m_idx = k.shuffle_sync(m_idx_e)
                n_idx = k.shuffle_sync(n_idx_e)
                if r.overlap:
                    slot = local_iter % r.tmem_slots
                    tmem_full_phase = (local_iter // r.tmem_slots) % 2
                else:
                    # Use the RUNTIME warpgroup id (numerically == c under this role's wg guard)
                    # for the TMEM column base, matching canon's `tmem[:, wg_id*MMA_N + i*NOL]`.
                    # A COMPILE-TIME-CONSTANT column (slot=c) lets ptxas hoist+cluster all 16
                    # epilogue tcgen05.ld reads (256 f32 live → LDL/STL spill that idles the MMA
                    # + throttles); canon's runtime base creates the address dependency that makes
                    # ptxas SERIALIZE the band reads (16 f32 live, no spill) — the trigger for
                    # canon's spill-free epilogue. Same value, just not constant-folded.
                    slot = k.warpgroup_id()
                    tmem_full_phase = local_iter % 2
                k.mbarrier_wait(tmem_full, stage=slot, phase=tmem_full_phase)
                tmem_col0 = slot * r.mma_n
                d_m = ((m_idx * CTA_GROUP + cta_rank) * r.num_consumer + c) * BLK_M

                if r.overlap:
                    # Fused per-tile load+cast+store, overlapping the next MMA. Match the
                    # canonical fence structure: read the whole EPI_N band's tcgen05.ld
                    # atoms into a wide fragment, then ONE wait_ld + ONE wide cast (the
                    # per-atom wait used before over-fenced and serialized the reads).
                    for ot in range(r.wb_pipe_depth):
                        # Fresh per-wb-tile cast fragment (canon's loop-local Dreg_16b): reusing
                        # one across the wb tiles has WAR deps that keep it in local memory
                        # (spill); a fresh single-use one promotes to registers.
                        out_wide = k.tensor(
                            space=MemorySpace.REG, dtype=config.dtype, shape=(r.epi_n,)
                        )
                        store_iter = local_iter * r.wb_pipe_depth + ot
                        with k.if_(store_iter >= r.num_d_tiles):
                            k.cp_async_bulk_wait_group_read(r.num_d_tiles - 1)
                            k.wg_sync(barrier_id=wg_bar)
                        # Fold the D-smem buffer index the same way the SMEM ring does:
                        # db = (local_iter*wb_pipe_depth + ot) % num_d_tiles collapses to the
                        # compile-time constant ot % num_d_tiles when num_d_tiles | wb_pipe_depth,
                        # so each d_smem store address is a literal instead of a runtime local_iter
                        # expression that ptxas spills (the IMAD*2 + STL/LDL storm of SMEM addrs).
                        db = _ring_index(local_iter, ot, r.wb_pipe_depth, r.num_d_tiles)[0]
                        # Read the EPI_N band in read_w=32 chunks (raw .x64 @ 2048's EPI_N=64
                        # overflows the 255-reg ceiling), each into a FRESH small f32 fragment
                        # that is cast and freed before the next chunk — so the live f32
                        # footprint is read_w (not the whole EPI_N), which keeps it in registers.
                        read_w = min(r.epi_n, 32)
                        for kc in range(r.epi_n // read_w):
                            chunk_frag = k.tensor(
                                space=MemorySpace.REG, dtype=DType.F32, shape=(read_w,)
                            )
                            col = tmem_col0 + ot * r.epi_n + kc * read_w
                            k.tcgen05_ld(
                                TensorSlice(tensor=chunk_frag, offsets=(0,), shape=(read_w,)),
                                accum,
                                num=read_w,
                                row=0,
                                col=col,
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
                        if ot == r.wb_pipe_depth - 1:
                            k.mbarrier_arrive(tmem_empty_leader, stage=slot)
                        # Warpgroup-sync FIRST (makes all threads' reg->smem writes CTA-visible),
                        # THEN a SINGLE-THREAD proxy fence (codegen guards it `if tid_in_wg==0`) —
                        # canon's order. The old all-128-thread fence-before-sync was the dominant
                        # epilogue stall (canon's own note); one thread's fence suffices post-sync.
                        k.wg_sync(barrier_id=wg_bar)
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
                    # No-overlap: interleave read+cast+store PER wb tile so the live writeback
                    # fragment is just ONE EPI_N tile (16 regs), not the whole MMA_N=256 row
                    # (128 regs). A 256-wide fragment held live across all 8 store warpgroup-syncs
                    # is what ptxas spills (measured: 60 LDL/STL bracketing the bar.sync vs canon's
                    # 2). The f32 read frag is a FRESH per-band single-use array (canon's loop-local
                    # `Dreg`) so it promotes to registers. tmem_empty is arrived AFTER the last
                    # read (loop end) — every TMEM band must be read before the slot is freed.
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
                                accum,
                                num=NOL,
                                row=0,
                                col=col,
                            )
                            k.tcgen05_wait_ld()
                            k.reg_cvt(
                                TensorSlice(tensor=out_tile, offsets=(sub * NOL,), shape=(NOL,)),
                                TensorSlice(tensor=band_frag, offsets=(0,), shape=(NOL,)),
                            )
                        store_iter = local_iter * r.wb_pipe_depth + ot
                        with k.if_(store_iter >= r.num_d_tiles):
                            k.cp_async_bulk_wait_group_read(r.num_d_tiles - 1)
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
                        # Warpgroup-sync FIRST (makes all threads' reg->smem writes CTA-visible),
                        # THEN a SINGLE-THREAD proxy fence (codegen guards it `if tid_in_wg==0`) —
                        # canon's order. The old all-128-thread fence-before-sync was the dominant
                        # epilogue stall (canon's own note); one thread's fence suffices post-sync.
                        k.wg_sync(barrier_id=wg_bar)
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
                    k.mbarrier_arrive(tmem_empty_leader, stage=slot)
            # Drain in-flight TMA stores once after the persistent loop (TIRx
            # drains before the tmem teardown). A per-task drain would leave the
            # NEXT task's pacing wait_group with no committed group to release on.
            k.cp_async_bulk_wait_group_read(0)
            k.wg_sync(barrier_id=wg_bar)

    # Cluster-wide barrier before freeing TMEM (canon emits `cluster_sync()` right before
    # relinquish + dealloc): every CTA's MMA / epilogue must finish using the accumulator
    # before warp 0 deallocs it. Without it, under multi-wave CLC the next wave's tcgen05
    # alloc races a still-live TMEM region from this wave — TMEM corruption that surfaces as
    # wrong values plus an accumulating launch failure (TMEM never cleanly released).
    if r.overlap:
        # canon's tmem_fin teardown: warp0/lane0 arrives at the PEER CTA's tmem_fin, then warp0
        # waits its own — a 2-CTA handshake that lets the other 7 warps exit without a full
        # cluster_sync. Both CTAs' epilogues (their TMEM reads) finish before either deallocs.
        with k.kernel_finalize(warp=0):
            k.mbarrier_arrive(k.mbar_ref(tmem_fin, remote_coord=1 - cta_rank), stage=0)
            k.mbarrier_wait(tmem_fin, stage=0, phase=0)
            k.tmem_dealloc(tmem_base, n_cols=N_COLS_TMEM, cta_group=CTA_GROUP)
    else:
        k.cluster_sync()
        with k.kernel_finalize(warp=0):
            k.tmem_dealloc(tmem_base, n_cols=N_COLS_TMEM, cta_group=CTA_GROUP)

    return k.build()


def _ring_index(local_iter, kt: int, k_tiles: int, pipe_depth: int):
    """SMEM-ring slot and occupancy parity for global k-step ``seq = local_iter*k_tiles +
    kt``: ``stage = seq % pipe_depth``, ``occ_parity = (seq // pipe_depth) % 2``.

    Algebraically identical to those two expressions, but with the build-time-constant
    parts folded in Python so the hot loop carries no runtime modular arithmetic over the
    runtime tile counter ``local_iter`` when it does not contribute. Writing
    ``seq % pipe_depth`` directly leaves ``local_iter`` live in a vector register (TVM does
    not simplify ``(local_iter*k_tiles) % pipe_depth`` to 0 even when ``pipe_depth | k_tiles``),
    forcing a vector→uniform (R2UR) conversion on every MMA operand descriptor. Folding the
    loop-invariant terms here — emitting ``local_iter*c`` only when its coefficient ``c`` is
    non-zero — keeps the ring index uniform (often a compile-time constant). ``kt`` is the
    Python-unrolled k-tile index, so ``kt`` and any pure-int result fold to literals.

    Decompose ``k_tiles = q*pipe_depth + rem``:
      seq // pipe_depth = local_iter*q + (local_iter*rem + kt)//pipe_depth
      seq %  pipe_depth = (local_iter*rem + kt) % pipe_depth
    so ``local_iter`` drops out entirely when ``rem == 0`` (stage) and additionally when
    ``q`` is even (parity). This is shape-agnostic: any shape evaluates the same formula."""
    q, rem = divmod(k_tiles, pipe_depth)
    stage_carry = (local_iter * rem) if rem else 0
    stage = (stage_carry + kt) % pipe_depth
    cross = ((local_iter * rem + kt) // pipe_depth) if rem else (kt // pipe_depth)
    qp = q % 2
    phase_carry = (local_iter * qp) if qp else 0
    occ_parity = (phase_carry + cross) % 2
    return stage, occ_parity


def _advance_ring(k: IRBuilder, stage_sc, phase_sc, pipe_depth: int) -> None:
    """Advance a persistent SMEM-ring (stage, phase) counter exactly like canon's
    PipelineState.advance: ``stage += 1``, wrapping to 0 at ``pipe_depth`` and flipping the
    phase parity on wrap. Used by the ROLLED k-loop so stage/phase is a clean loop-carried
    induction variable — the compiler then keeps it in UNIFORM registers and builds the MMA
    operand descriptors by uniform increment (UIADD3), instead of the per-unrolled-iteration
    vector address math (VIADD/LOP3 + R2UR) and register spill (LDL/STL) that the
    fully-unrolled form emits. (No else-branch in the builder: store the incremented value,
    then conditionally reset on wrap — the compiler folds it to a select.)"""
    # Snapshot stage+1 into its own scalar FIRST — `stage_sc + 1` is a lazy expression that
    # re-reads stage_sc, so storing into stage_sc before re-using it would alias (read the
    # just-written value). With nxt materialized, wrap it in-place then commit to stage_sc.
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
    """A CLC worker's consume loop, written out explicitly in nymph IR (canon's
    init-then-consume pattern, policy="custom"). The cluster's OWN launch tile is the
    cluster id (``cta_id // CTA_GROUP``) and is processed FIRST without a steal — the
    hardware `clc_query_cancel` returns *stolen* cluster ids, never the launcher's own.
    Each subsequent iteration consumes the handle: wait ``sched_arr`` (the scheduler's
    `clc_try_cancel` landed the 16B response), `clc_query_cancel` the next stolen tile,
    arrive ``sched_fin`` (cross-CTA to the leader, releasing the slot), and break on the
    drain sentinel (raw < 0, the broadcast 0xFFFFFFFF). ``task_id`` is the decoded tile
    (raw ctaid // CTA_GROUP); ``local_iter`` counts tiles (drives the data-ring phases)
    and equals the scheduler's fill counter, keeping the single-stage sched_arr lockstep."""
    task_id = k.scalar(initial=cluster_id, dtype=ScalarDType.I32)
    local_iter = k.scalar(initial=0, dtype=ScalarDType.I32)
    with k.loop():
        yield task_id, local_iter
        k.mbarrier_wait(sched_arr_full, stage=0, phase=local_iter % 2)
        raw = k.clc_query_cancel(sched, clc_handle)
        # Warpgroup consumers (epilogue): ALL 128 threads query the handle, so they must
        # all finish BEFORE the single elected thread arrives sched_fin and lets the
        # scheduler overwrite the handle. Canon's `consume_wg` does this warpgroup_sync;
        # without it the lagging threads read the next task's handle and desync (a
        # multi-tile-only deadlock — single-tile shapes never reuse the handle). The
        # single-elect loader/MMA loops run on one thread and pass wg_bar=None.
        if wg_bar is not None:
            k.wg_sync(barrier_id=wg_bar)
        elif warp_sync:
            # Full-warp consumers (e.g. the MMA warp): all 32 lanes query the handle, so
            # they must converge before the elected lane arrives sched_fin (else the
            # scheduler overwrites the handle while lagging lanes are still reading it).
            k.warp_sync()
        k.mbarrier_arrive(sched_fin_leader, stage=0)
        k.break_if(raw < 0)
        k.scalar_store(task_id, raw // CTA_GROUP)
        k.scalar_store(local_iter, local_iter + 1)


def _init_stages(k: IRBuilder, mbar: MBar, *, stages: int, count: int) -> None:
    # Emit `for i in T.unroll(stages): mbarrier.init(buf[i], count)` (canon's form) instead of
    # `stages` flattened init statements. T.unroll is compile-time-unrolled (same SASS) but the
    # source reads as a loop. Single-stage barriers stay a plain unguarded init (no loop).
    if stages == 1:
        k.mbarrier_init(mbar, count=count, stage=0)
        return
    with k.for_loop(stop=stages, unroll=True) as i:
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


def _validate_launch_shape(launch_shape: LaunchShape) -> None:
    if not isinstance(launch_shape, tuple) or len(launch_shape) != 1:
        raise ValueError("fp16_bf16_gemm requires a 1D launch_shape")
    count = launch_shape[0]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("fp16_bf16_gemm launch_shape[0] must be a positive integer")
    if count % CTA_GROUP != 0:
        raise ValueError("fp16_bf16_gemm launch_shape[0] must be divisible by cta_group")
