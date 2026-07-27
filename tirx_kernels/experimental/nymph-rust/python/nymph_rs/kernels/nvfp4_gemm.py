"""NVFP4 GEMM (Blackwell sm100 block-scaled fp4) expressed in Nymph IR."""

from __future__ import annotations

from dataclasses import dataclass

from ..builder import IRBuilder
from ..nymph_rs import (
    BlockScaleSpec,
    DType,
    FenceKind,
    FenceScope,
    Kernel,
    LaunchShape,
    MBarKind,
    MemorySpace,
    ScalarDType,
    SmemSwizzleLayout,
    Swizzle,
    TensorSlice,
)

CTA_M = 128  # per-CTA A tile rows (one M tile)
CTA_N = 128  # per-CTA B tile rows (this CTA's half of the shared N band) — canon's CTA_N
CTA_GROUP = 2
CLUSTER_M = 2
# MMA_N = CTA_N * CTA_GROUP = 256.
CTA_K = 256  # K per pipeline tile
MMA_N = CTA_N * CTA_GROUP  # 256, the shared N band the pair computes together
SF_BLOCK = 16  # one e4m3 scale per 16 K-elements
SF_CTA_K = CTA_K // SF_BLOCK  # 16 e4m3 scale bytes per row per k-tile (canon SF_CTA_K)
BLK_K_BYTES = CTA_K // 2  # packed fp4 bytes per row per k-tile (2 e2m1 per byte)
EPI_TILE = 64
TMEM_LD_SIZE = 8
# NOTE: a consumer-RAISE setmaxnreg.
ACC_DEPTH = 1  # accumulator TMEM stages (MMA_N=256 fills half of 512; one stage)
D_DEPTH = 2  # D_smem store ring depth (store pacing)
SMEM_DEPTH = 5  # SMEM pipeline depth (mirrors TIRx PIPE_DEPTH)
N_COLS_TMEM = 512
TILE_GROUPS_ROW_SIZE = 16
SM_NUMBER = 148
SF_BASE_COL = 448  # canon's fixed SF base column (the accumulator spans 0..MMA_N)


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _advance_ring(k: IRBuilder, stage_sc, phase_sc, pipe_depth: int) -> None:
    """Advance a persistent SMEM-ring (fp16_bf16_gemm's idiom)."""
    # Snapshot stage+1 into its own scalar FIRST.
    nxt = k.scalar(initial=stage_sc + 1, dtype=ScalarDType.I32)
    with k.if_(nxt >= pipe_depth):
        k.scalar_store(nxt, 0)
        k.scalar_store(phase_sc, (phase_sc + 1) % 2)
    k.scalar_store(stage_sc, nxt)



def _ab_smem_layout(blk_k_bytes):
    'Widest swizzle the A/B stage row (packed fp4 bytes) fits.'
    for swizzle, atom in ((Swizzle.B128, 128), (Swizzle.B64, 64), (Swizzle.B32, 32)):
        if blk_k_bytes >= atom and blk_k_bytes % atom == 0:
            return SmemSwizzleLayout(swizzle)
    return None


def _d_smem_layout(epi_tile: int) -> SmemSwizzleLayout | None:
    """Widest swizzle the epilogue store tile's row (epi_tile bf16 elems) fits."""
    row_bytes = epi_tile * 2
    for swizzle, atom in ((Swizzle.B128, 128), (Swizzle.B64, 64), (Swizzle.B32, 32)):
        if row_bytes >= atom and row_bytes % atom == 0:
            return SmemSwizzleLayout(swizzle)
    return None


@dataclass(frozen=True, slots=True)
class NvFp4GemmConfig:
    m: int = 1024
    n: int = 1024
    k: int = 1024
    # alpha = 1 / (A_global_sf * B_global_sf), applied as the epilogue rescale.
    alpha: float = 1.0
    launch_shape: LaunchShape | None = None
    # Per-CTA N tile (canon's CTA_N knob).
    cta_n: int | None = None
    # Epilogue store-tile column width.
    epi_tile: int | None = None
    # SMEM pipeline depth.
    smem_depth: int | None = None
    # D-store ring depth.
    d_depth: int | None = None
    # Persistent-scheduler L2 tile-group size (canon's L2_GROUP_SIZE).
    l2_group_size: int | None = None
    # L2 eviction policy on the A/B/SFA/SFB g2c loads (canon's `_tma_g2c_args` cache_hint).
    load_cache_hint: str | None = "evict_normal"
    # Epilogue schedule (canon's OVERLAP_EPI). Two TMEM-drain schedules over ONE accumulator.
    epilogue: str = "overlap"
    # K per pipeline tile (default 256; nvjet runs 64 = finer, deeper pipeline).
    cta_k: int | None = None
    # Per-warpgroup register budget (canon's INVARIANT-I1b per-role `setmaxnreg`).
    maxnreg_epilogue: int | None = None
    maxnreg_producer: int | None = None


def _cfg_cta_n(config: NvFp4GemmConfig) -> int:
    return config.cta_n if config.cta_n is not None else CTA_N


def _cfg_epi_tile(config: NvFp4GemmConfig) -> int:
    return config.epi_tile if config.epi_tile is not None else EPI_TILE


def _cfg_smem_depth(config: NvFp4GemmConfig) -> int:
    return config.smem_depth if config.smem_depth is not None else SMEM_DEPTH


def _cfg_d_depth(config: NvFp4GemmConfig) -> int:
    return config.d_depth if config.d_depth is not None else D_DEPTH


def _cfg_l2_group_size(config: NvFp4GemmConfig) -> int:
    return config.l2_group_size if config.l2_group_size is not None else TILE_GROUPS_ROW_SIZE


def nvfp4_task_config(tasks: int) -> NvFp4GemmConfig:
    """The canonical single-cluster examination setting, mirroring the fp16/bf16 and fp8 ones."""
    return NvFp4GemmConfig(
        m=CTA_M * CLUSTER_M, n=MMA_N * tasks, k=16384, alpha=1.0, launch_shape=(2,)
    )


CONFIGS = [
    {"m": s, "n": s, "k": s, "label": f"{s}x{s}x{s}"} for s in [1024, 2048, 4096, 8192, 16384]
]

# Per-shape tuning knobs (mirrors canon's TIRX_CONFIGS).
GEMM_CONFIGS = {
    # 1024: the coarse default MMA_N=256 tiling yields only 16 clusters (32 CTAs) on 148 SMs.
    (1024, 1024, 1024): {
        "cta_n": 64,
        "l2_group_size": 12,
        "load_cache_hint": None,
        # Canon's dynamic SMEM pool.
        "smem_depth": 5,
        # D-store ring 3 deep: only the last of 4 bands waits a store drain.
        "d_depth": 3,
        # Limit epilogue registers while using 32-column store tiles.
        "epi_tile": 32,
    },
    # 2048: l2_group_size=4 matches canon's TIRX_CONFIGS[2048] L2_GROUP_SIZE=4.
    (2048, 2048, 2048): {
        "l2_group_size": 4,
        "load_cache_hint": "evict_normal",
        # D-store ring 3 deep: cuts the per-band drain waits (0.906 -> 0.933).
        "d_depth": 3,
        "epi_tile": 32,
    },
    # 4096: the epilogue is the wall-clock residual.
    (4096, 4096, 4096): {
        "l2_group_size": 4,
        "load_cache_hint": None,
        "epilogue": "no_overlap",
        "epi_tile": 32,
    },
    # 8192: OVERLAP_EPI=False / L2_GROUP_SIZE=1 (canon); smem_depth=5 and
    # EPI_TILE=32 (canon has 4/16) both measured faster: depth 5 1.013 vs
    # 0.987, epi32's B64-swizzled 8-band store 0.995 vs B32 16-band 0.989.
    (8192, 8192, 8192): {
        "l2_group_size": 1,
        "load_cache_hint": None,
        "epilogue": "no_overlap",
        "smem_depth": 5,
        "epi_tile": 32,
    },
    # 16384: OVERLAP_EPI=False / EPI_TILE=16 (canon); smem_depth=5 (canon 4)
    # measured 1.046 vs 0.99; L2_GROUP 12 -> 16 (12 does not divide the
    # 64-row cluster-tile grid).
    (16384, 16384, 16384): {
        "l2_group_size": 16,
        "load_cache_hint": None,
        "epilogue": "no_overlap",
        "smem_depth": 5,
        "epi_tile": 16,
    },
}


def gemm_config_for(m: int, n: int, k: int) -> dict:
    """The per-shape knob overrides for (m, n, k); empty dict = all defaults."""
    return dict(GEMM_CONFIGS.get((m, n, k), {}))


# Cheap shapes for the protocol + value sweeps.
NVFP4_CONFIGS_SUPPORTED = [
    {"m": 256, "n": 256, "k": 256, "label": "256x256x256"},  # 1 k-tile, 1 tile
    {"m": 256, "n": 512, "k": 512, "label": "256x512x512"},  # 2 k-tiles, 2 N tiles
    {"m": 512, "n": 256, "k": 512, "label": "512x256x512"},  # 2 M tiles (2 pairs)
    {"m": 512, "n": 512, "k": 1024, "label": "512x512x1024"},  # 4 k-tiles, 2x2 tiles
    {"m": 1024, "n": 1024, "k": 1024, "label": "1024x1024x1024"},
]


def build_nvfp4_gemm(config: NvFp4GemmConfig = NvFp4GemmConfig()) -> Kernel:
    M, N, K = config.m, config.n, config.k
    cta_k = config.cta_k or CTA_K
    sf_cta_k = cta_k // SF_BLOCK
    blk_k_bytes = cta_k // 2
    _validate_config(config, cta_k)
    cta_group = CTA_GROUP
    # Per-shape tiling knobs (canon's CTA_N / EPI_TILE / PIPE_DEPTH).
    cta_n = _cfg_cta_n(config)
    l2_group_size = _cfg_l2_group_size(config)
    mma_n = cta_n * CTA_GROUP  # the pair's shared N band (256 default; 128 for cta_n=64)
    epi_tile = _cfg_epi_tile(config)
    smem_depth = _cfg_smem_depth(config)
    d_depth = _cfg_d_depth(config)  # output store-ring depth (canon WB_PIPE_DEPTH)
    load_cache_hint = config.load_cache_hint  # per-shape L2 hint on the g2c loads (see config)
    blk_m = CTA_M  # per-CTA A rows (its own M tile)
    blk_n = cta_n  # per-CTA B rows (its half of the shared N band)
    sched_rows = _ceil_div(M, CTA_M)  # M tiles
    sched_cols = _ceil_div(N, mma_n)  # N bands
    k_tiles = K // cta_k
    total_work = sched_rows * sched_cols
    store_tiles = mma_n // epi_tile

    launch_shape = config.launch_shape or (
        max(cta_group, min(SM_NUMBER, total_work) // cta_group * cta_group),
    )
    _validate_launch_shape(launch_shape, cta_group)
    pair_tasks = total_work // cta_group

    # packed fp4 operand tiles, e4m3 scale bytes (1B each, canon SFA_in layout), bf16 out.
    a_tile_bytes = blk_m * blk_k_bytes
    b_tile_bytes = blk_n * blk_k_bytes
    sfa_tile_bytes = blk_m * sf_cta_k  # per k-tile, this CTA's M rows x 16 e4m3 bytes
    # canon's SFB_N==MMA_N path.
    sfb_tile_bytes = mma_n * sf_cta_k  # per k-tile, the full N band's rows x 16 bytes
    d_tile_bytes = blk_m * epi_tile * 2

    a_off = 0
    b_off = a_off + smem_depth * a_tile_bytes
    sfa_off = b_off + smem_depth * b_tile_bytes
    sfb_off = sfa_off + smem_depth * sfa_tile_bytes
    d_off = sfb_off + smem_depth * sfb_tile_bytes
    data_end = d_off + d_depth * d_tile_bytes
    metadata_cursor = (data_end + 7) // 8 * 8
    smem_full_off = metadata_cursor
    metadata_cursor += smem_depth * 8
    sf_full_off = metadata_cursor
    metadata_cursor += smem_depth * 8
    smem_empty_off = metadata_cursor
    metadata_cursor += smem_depth * 8
    tmem_full_off = metadata_cursor
    metadata_cursor += ACC_DEPTH * 8
    tmem_empty_off = metadata_cursor
    metadata_cursor += ACC_DEPTH * 8
    tmem_fin_off = metadata_cursor
    metadata_cursor += 8
    inits_done_off = metadata_cursor
    metadata_cursor += 8
    tmem_addr_off = (metadata_cursor + 3) // 4 * 4
    smem_size_bytes = tmem_addr_off + 4

    k = IRBuilder(
        "nymph_nvfp4_gemm",
        num_warps=8,  # wg0 = tma/mma warps, wg1 = epilogue
        smem_size_bytes=smem_size_bytes,
        launch_shape=launch_shape,
        cluster_shape=(cta_group,),
    )
    # Operands are packed fp4.
    a_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.U8, shape=(M, K // 2))
    b_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.U8, shape=(N, K // 2))
    sfa_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.F8E4M3, shape=(M // 128, K // 64, 32, 16))
    sfb_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.F8E4M3, shape=(N // 128, K // 64, 32, 16))
    d_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.BF16, shape=(M, N))

    # Stage-major SMEM rings, indexed by a runtime pipeline stage.
    a_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.U8,
        shape=(smem_depth, blk_m, blk_k_bytes),
        layout=_ab_smem_layout(blk_k_bytes),
        byte_offset=a_off,
    )
    b_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.U8,
        shape=(smem_depth, blk_n, blk_k_bytes),
        layout=_ab_smem_layout(blk_k_bytes),
        byte_offset=b_off,
    )
    # SF SMEM: plain contiguous physical aliases, stage-prefixed.
    sfa_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.F8E4M3,
        shape=(smem_depth, blk_m // 128, sf_cta_k // 4, 32, 16),
        byte_offset=sfa_off,
    )
    sfb_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.F8E4M3,
        shape=(smem_depth, mma_n // 128, sf_cta_k // 4, 32, 16),
        byte_offset=sfb_off,
    )
    d_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.BF16,
        shape=(d_depth, blk_m, epi_tile),
        # Swizzle the epilogue staging tile as wide as the row allows: the
        # per-thread row stores (tid, c:c+chunk) otherwise land on the same
        # bank group, the measured SMEM store-bank-conflict driver.
        layout=_d_smem_layout(epi_tile),
        byte_offset=d_off,
    )

    # TMEM: accumulator (one MMA_N stage) at col 0.
    accum = k.tmem_tensor(0)
    sfa_tmem = k.tmem_tensor(SF_BASE_COL)
    # The FULL N band's B scales.
    sfa_cols = _ceil_div(blk_m, 128) * sf_cta_k
    sfb_tmem = k.tmem_tensor(SF_BASE_COL + sfa_cols)

    # Each epilogue tile drains one full epi_tile column band with one tcgen05.ld.
    frag_width = mma_n if config.epilogue == "no_overlap" else epi_tile
    stmatrix_epi = config.epilogue == "stmatrix"
    accum_frag = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(frag_width,))
    out_frag = k.tensor(space=MemorySpace.REG, dtype=DType.BF16, shape=(frag_width,))

    smem_full = k.mbar(kind=MBarKind.TMA, byte_offset=smem_full_off, stages=smem_depth)
    smem_full_leader = k.mbar_ref(smem_full, remote_coord=0)
    # Separate SF-load completion barrier.
    sf_full = k.mbar(kind=MBarKind.TMA, byte_offset=sf_full_off, stages=smem_depth)
    sf_full_leader = k.mbar_ref(sf_full, remote_coord=0)
    smem_empty = k.mbar(kind=MBarKind.TCGEN05, byte_offset=smem_empty_off, stages=smem_depth)
    tmem_full = k.mbar(kind=MBarKind.TCGEN05, byte_offset=tmem_full_off, stages=ACC_DEPTH)
    # tmem_empty: one elected thread per CTA's epilogue arrives.
    tmem_empty = k.mbar(kind=MBarKind.THREAD, byte_offset=tmem_empty_off, stages=ACC_DEPTH)
    tmem_empty_leader = k.mbar_ref(tmem_empty, remote_coord=0)
    # tmem_fin: canon's exact lightweight 2-CTA teardown handshake (its `tmem_finished`).
    tmem_fin = k.mbar(kind=MBarKind.THREAD, byte_offset=tmem_fin_off, stages=1)
    # inits_done: per-CTA "barriers initialized" flag (warp 1 -> all roles).
    # Replaces the pair-wide cluster_sync on the startup critical path: the
    # producers' first TMA flight then overlaps warp 0's tcgen05.alloc
    # (~0.9us, the fp16 prologue lesson).
    inits_done = k.mbar(kind=MBarKind.THREAD, byte_offset=inits_done_off, stages=1)

    cta_id = k.cta_id()
    cta_rank = k.ctaid_in_cluster()
    task_space = k.task_space(grid=(pair_tasks,), fields=("pair_idx",))
    task_scheduler = k.scheduler(task_space)
    task_start = cta_id // cta_group
    task_step = k.launch_cta_count // cta_group

    ab_bytes = a_tile_bytes + b_tile_bytes
    sf_bytes = sfa_tile_bytes + sfb_tile_bytes

    def work_coords(task_id, cta_rank):
        """Canon's ClusterPersistentScheduler2D group-major mapping, at PAIR."""
        pair_rows = sched_rows // CLUSTER_M  # cluster-tile rows (canon CLUSTER_M_TILES)
        if pair_rows <= l2_group_size:
            pair_row = task_id % pair_rows
            n_idx = task_id // pair_rows
        else:
            group_span = l2_group_size * sched_cols
            group_id = task_id // group_span
            within = task_id % group_span
            pair_row = group_id * l2_group_size + within % l2_group_size
            n_idx = within // l2_group_size
        m_idx = pair_row * CLUSTER_M + cta_rank
        return m_idx, n_idx

    # mbarrier inits live on warp 1 (free role), which then publishes
    # inits_done. There is NO pair-wide cluster_sync on the startup path:
    # each role gates on inits_done once (cheap, local), so the producers'
    # first TMA flight overlaps warp 0's tcgen05.alloc (~0.9us, the fp16
    # prologue lesson). Cross-CTA init ordering is carried by the mbarrier
    # phase protocol itself: the peer's complete-tx/expect-tx land well
    # after the leader-side inits (flight ~0.5us >> init ~0.04us), and every
    # tmem-side signal is downstream of the MMA warp's own alloc.
    with k.if_warp(1):
        k.cluster_barrier_arrive()
        with k.if_elected():
            for s in range(smem_depth):
                k.mbarrier_init(smem_full, count=1, stage=s)
                k.mbarrier_init(sf_full, count=1, stage=s)
                k.mbarrier_init(smem_empty, count=1, stage=s)
            for s in range(ACC_DEPTH):
                k.mbarrier_init(tmem_full, count=1, stage=s)
                k.mbarrier_init(tmem_empty, count=cta_group, stage=s)
            # canon's tmem_finished (init_full=1): one cross-CTA arrival releases the wait.
            k.mbarrier_init(tmem_fin, count=1, stage=0)
            k.mbarrier_init(inits_done, count=1, stage=0)

    # Seal the mbarrier-init epoch (canon's `T.ptx.fence.mbarrier_init` + the
    # fused cluster_sync). The inits live on warp 1 (free role) so the sync
    # covers a ~0.2us init chain instead of warp 0's; the tcgen05.alloc is
    # pre-sync (the tmem-lifecycle checker's requirement).
    k.fence(kind=FenceKind.MBARRIER_INIT)
    k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
    k.cluster_sync()

    # Per-role register rebalance (canon's INVARIANT-I1b).
    with k.if_warpgroup(0):
        if config.maxnreg_producer is not None:
            k.set_maxnreg(config.maxnreg_producer)

        # ---- A/B producer (warp 2). SF loads live on warp 3 (canon's two-warp
        # producer topology).
        with k.if_warp(2):
            k.cluster_barrier_arrive()
            # Loop-carried SMEM-ring induction state (fp16_bf16_gemm's idiom):
            # no per-k-tile `% smem_depth` magic-divide chain.
            ld_stage = k.scalar(initial=0, dtype=ScalarDType.I32)
            ld_phase = k.scalar(initial=0, dtype=ScalarDType.I32)
            with k.for_each_task(task_scheduler) as task:
                m_idx, n_idx = work_coords(task.task_id, cta_rank)
                a_m = m_idx * CTA_M  # this CTA's own M tile
                b_n = n_idx * mma_n + cta_rank * cta_n  # this CTA's half of the N band
                # Canon's producer shape: ONE elected region wrapping the whole
                # rolled k-loop (no per-iteration elect open/close).
                with k.if_elected():
                    with k.for_loop(stop=k_tiles, unroll=False) as t:
                        k.mbarrier_wait(smem_empty, stage=ld_stage, phase=(ld_phase + 1) % 2)
                        # canon's split arrive.
                        with k.if_(cta_rank.eq(0)):
                            k.mbarrier_arrive_expect_tx(
                                smem_full_leader, bytes=cta_group * ab_bytes, stage=ld_stage
                            )
                        kb = t * blk_k_bytes  # packed-fp4 byte column
                        # canon tags every g2c load with the L2 `evict_normal` policy.
                        k.tma_load(
                            TensorSlice(
                                tensor=a_smem,
                                offsets=(ld_stage, 0, 0),
                                shape=(1, blk_m, blk_k_bytes),
                            ),
                            a_gmem,
                            mbar=smem_full_leader,
                            coords=(a_m, kb),
                            shape=(1, blk_m, blk_k_bytes),
                            gmem_shape=(blk_m, blk_k_bytes),
                            mbar_stage=ld_stage,
                            cache_hint=load_cache_hint,
                            cta_group=cta_group,
                        )
                        k.tma_load(
                            TensorSlice(
                                tensor=b_smem,
                                offsets=(ld_stage, 0, 0),
                                shape=(1, blk_n, blk_k_bytes),
                            ),
                            b_gmem,
                            mbar=smem_full_leader,
                            coords=(b_n, kb),
                            shape=(1, blk_n, blk_k_bytes),
                            gmem_shape=(blk_n, blk_k_bytes),
                            mbar_stage=ld_stage,
                            cache_hint=load_cache_hint,
                            cta_group=cta_group,
                        )
                        _advance_ring(k, ld_stage, ld_phase, smem_depth)
        # ---- SF producer (canon's second producer warp): scale-factor loads on
        # warp 3, pacing its own arrive/loads per k-tile.
        with k.if_warp(3):
            k.cluster_barrier_arrive()
            sf_stage = k.scalar(initial=0, dtype=ScalarDType.I32)
            sf_phase = k.scalar(initial=0, dtype=ScalarDType.I32)
            with k.for_each_task(task_scheduler) as task:
                m_idx, n_idx = work_coords(task.task_id, cta_rank)
                a_m = m_idx * CTA_M
                sf_n = n_idx * mma_n  # the FULL N band's B scales (rank-independent)
                with k.if_elected():
                    with k.for_loop(stop=k_tiles, unroll=False) as t:
                        k.mbarrier_wait(smem_empty, stage=sf_stage, phase=(sf_phase + 1) % 2)
                        with k.if_(cta_rank.eq(0)):
                            k.mbarrier_arrive_expect_tx(
                                sf_full_leader, bytes=cta_group * sf_bytes, stage=sf_stage
                            )
                        # SFA: this CTA's M rows.
                        sf_k_outer = t * (sf_cta_k // 4)
                        k.tma_load(
                            TensorSlice(
                                tensor=sfa_smem,
                                offsets=(sf_stage, 0, 0, 0, 0),
                                shape=(1, blk_m // 128, sf_cta_k // 4, 32, 16),
                            ),
                            sfa_gmem,
                            mbar=sf_full_leader,
                            coords=(a_m // 128, sf_k_outer, 0, 0),
                            shape=(1, blk_m // 128, sf_cta_k // 4, 32, 16),
                            gmem_shape=(blk_m // 128, sf_cta_k // 4, 32, 16),
                            mbar_stage=sf_stage,
                            cache_hint=load_cache_hint,
                            cta_group=cta_group,
                        )
                        # SFB is shared by the CTA pair.
                        if mma_n == 128:
                            with k.if_(cta_rank.eq(0)):
                                k.tma_load(
                                    TensorSlice(
                                        tensor=sfb_smem,
                                        offsets=(sf_stage, 0, 0, 0, 0),
                                        shape=(1, 1, sf_cta_k // 4, 32, 16),
                                    ),
                                    sfb_gmem,
                                    mbar=sf_full_leader,
                                    coords=(sf_n // 128, sf_k_outer, 0, 0),
                                    shape=(1, 1, sf_cta_k // 4, 32, 16),
                                    gmem_shape=(1, sf_cta_k // 4, 32, 16),
                                    mbar_stage=sf_stage,
                                    multicast_cta_mask=0b11,
                                    cache_hint=load_cache_hint,
                                    cta_group=cta_group,
                                )
                        else:
                            k.tma_load(
                                TensorSlice(
                                    tensor=sfb_smem,
                                    offsets=(sf_stage, cta_rank, 0, 0, 0),
                                    shape=(1, 1, sf_cta_k // 4, 32, 16),
                                ),
                                sfb_gmem,
                                mbar=sf_full_leader,
                                coords=(sf_n // 128 + cta_rank, sf_k_outer, 0, 0),
                                shape=(1, 1, sf_cta_k // 4, 32, 16),
                                gmem_shape=(1, sf_cta_k // 4, 32, 16),
                                mbar_stage=sf_stage,
                                multicast_cta_mask=0b11,
                                cache_hint=load_cache_hint,
                                cta_group=cta_group,
                            )
                        _advance_ring(k, sf_stage, sf_phase, smem_depth)

        # ---- MMA (wg0/warp0, cluster leader only).
        with k.if_warp(0):
            # tmem_alloc (warp-collective, full warp 0), then the split
            # cluster barrier: the pair's alloc completions rendezvous and
            # the wait makes them visible (the tmem-lifecycle checker's
            # alloc-before-use sync edge) WITHOUT gating the producers —
            # they arrive non-blocking at role entry (nvjet's alloc-late).
            k.tmem_alloc(0, N_COLS_TMEM, addr_byte_offset=tmem_addr_off, cta_group=cta_group)
            # Relinquish the alloc permit NOW (this kernel allocs once) — it
            # leaves the teardown with just the dealloc.
            k.tmem_relinquish(cta_group)
            k.cluster_barrier_arrive()
            k.cluster_barrier_wait()
            with k.if_elected():
                # Loop-carried SMEM-ring induction state (persistent across tasks).
                mma_stage = k.scalar(initial=0, dtype=ScalarDType.I32)
                mma_phase = k.scalar(initial=0, dtype=ScalarDType.I32)
                with k.for_each_task(task_scheduler) as task:
                    # NO shuffle_sync here.
                    local_iter = k.let((task.task_id - task_start) // task_step)
                    with k.if_(cta_rank.eq(0)):
                        tmem_idx = local_iter % ACC_DEPTH
                        k.mbarrier_wait(
                            tmem_empty, stage=tmem_idx, phase=(local_iter // ACC_DEPTH + 1) % 2
                        )
                        acc_op = accum.at(0, tmem_idx * mma_n)

                        def mma_ktile(accum_flag):
                            # Wait loads, copy scales to TMEM, and issue one block-scaled MMA k-tile.
                            # smem_full starts EMPTY.
                            k.mbarrier_wait(sf_full, stage=mma_stage, phase=mma_phase)
                            k.mbarrier_wait(smem_full, stage=mma_stage, phase=mma_phase)
                            for m_super in range(blk_m // 128):
                                for k_outer in range(sf_cta_k // 4):
                                    k.tcgen05_cp(
                                        sfa_tmem.at(0, 4 * m_super + (blk_m // 32) * k_outer),
                                        k.smem_tile(
                                            sfa_smem,
                                            prefix_indices=(mma_stage, m_super, k_outer),
                                            row_offset=0,
                                            col_offset=0,
                                            rows=32,
                                            cols=16,
                                        ),
                                        shape="32x128b",
                                        multicast="warp4",
                                        cta_group=cta_group,
                                    )
                            for m_super in range(mma_n // 128):
                                for k_outer in range(sf_cta_k // 4):
                                    k.tcgen05_cp(
                                        sfb_tmem.at(0, 4 * m_super + (mma_n // 32) * k_outer),
                                        k.smem_tile(
                                            sfb_smem,
                                            prefix_indices=(mma_stage, m_super, k_outer),
                                            row_offset=0,
                                            col_offset=0,
                                            rows=32,
                                            cols=16,
                                        ),
                                        shape="32x128b",
                                        multicast="warp4",
                                        cta_group=cta_group,
                                    )
                            # canon's cluster gemm.
                            k.tcgen05_mma(
                                acc_op,
                                k.mma_a_smem(
                                    k.smem_tile(
                                        a_smem,
                                        prefix_indices=(mma_stage,),
                                        row_offset=0,
                                        col_offset=0,
                                        rows=blk_m,
                                        cols=blk_k_bytes,
                                    )
                                ),
                                k.smem_tile(
                                    b_smem,
                                    prefix_indices=(mma_stage,),
                                    row_offset=0,
                                    col_offset=0,
                                    rows=blk_n,
                                    cols=blk_k_bytes,
                                ),
                                mma_m=256,
                                mma_n=mma_n,
                                format="f4_e2m1",
                                block_scale=BlockScaleSpec(
                                    sfa=sfa_tmem.at(0, 0),
                                    sfb=sfb_tmem.at(0, 0),
                                    sfa_k_offset=0,
                                    sfb_k_offset=0,
                                    scale_format="e4m3_fn",
                                    sf_per_mma=4,
                                    sf_reuse=1,
                                ),
                                accum=accum_flag,
                                trans_a=False,
                                trans_b=False,
                                ws=False,
                                cta_group=cta_group,
                            )
                            k.tcgen05_commit(
                                smem_empty, stage=mma_stage, cta_group=cta_group, multicast_cta_mask=0b11
                            )
                            _advance_ring(k, mma_stage, mma_phase, smem_depth)

                        # The k-loop with a RUNTIME accum cell (canon's shape;
                        # no first-k-tile peel doubling the MMA/SF-cp body).
                        accum_flag = k.scalar(initial=0, dtype=ScalarDType.I32)
                        with k.for_loop(stop=k_tiles, unroll=False) as t:
                            mma_ktile(accum_flag)
                            k.scalar_store(accum_flag, 1)
                        k.tcgen05_commit(
                            tmem_full, stage=tmem_idx, cta_group=cta_group, multicast_cta_mask=0b11
                        )

    # ---- epilogue (wg1) ---- stmatrix runs the OVERLAP schedule.
    no_overlap = config.epilogue == "no_overlap"
    # reg->smem store-chunk width.
    store_chunk = 16 if stmatrix_epi else TMEM_LD_SIZE

    def store_band(local_iter, d_m, d_n, ot, frag_off):
        """Chunked reg->smem + S->G TMA store of one EPI_TILE band."""
        store_iter = local_iter * store_tiles + ot
        # The ring-guard wait+sync is needed ONLY when the d_smem ring
        # actually wraps (store_iter >= depth); an unconditional per-band
        # sync is pure latency when store_tiles <= d_depth.
        with k.if_(store_iter >= d_depth):
            k.cp_async_bulk_wait_group_read(d_depth - 1)
            k.wg_sync(barrier_id=10)
        d_stage = store_iter % d_depth
        # Store reg->smem in store_chunk-wide sub-slices.
        for ki in range(epi_tile // store_chunk):
            k.reg_store(
                TensorSlice(
                    tensor=d_smem,
                    offsets=(d_stage, k.tid_in_wg(), ki * store_chunk),
                    shape=(1, 1, store_chunk),
                ),
                TensorSlice(
                    tensor=out_frag, offsets=(frag_off + ki * store_chunk,), shape=(store_chunk,)
                ),
            )
        # canon's epilogue-store order (and the fp16/bf16 port's).
        k.wg_sync(barrier_id=10)
        with k.if_(k.tid_in_wg().eq(0)):
            k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
            k.tma_store(
                d_gmem,
                TensorSlice(tensor=d_smem, offsets=(d_stage, 0, 0), shape=(1, blk_m, epi_tile)),
                coords=(d_m, d_n + ot * epi_tile),
                shape=(1, blk_m, epi_tile),
                gmem_shape=(blk_m, epi_tile),
            )
        k.cp_async_bulk_commit_group()

    with k.if_warpgroup(1):
        k.cluster_barrier_arrive()
        # Consumer epilogue register cap.
        if config.maxnreg_epilogue is not None:
            k.set_maxnreg(config.maxnreg_epilogue)
        with k.for_each_task(task_scheduler) as task:
            # Same `let` SSA form as the TMA producer (was shuffle_sync).
            local_iter = k.let((task.task_id - task_start) // task_step)
            m_idx, n_idx = work_coords(task.task_id, cta_rank)
            d_m = m_idx * CTA_M
            d_n = n_idx * mma_n
            tmem_idx = local_iter % ACC_DEPTH
            k.mbarrier_wait(tmem_full, stage=tmem_idx, phase=(local_iter // ACC_DEPTH) % 2)
            acc_col0 = tmem_idx * mma_n
            if no_overlap:
                # canon's OVERLAP_EPI=False (reg_all = (128, MMA_N)).
                for ot in range(store_tiles):
                    k.tcgen05_ld(
                        TensorSlice(tensor=accum_frag, offsets=(ot * epi_tile,), shape=(epi_tile,)),
                        accum.at(0, acc_col0 + ot * epi_tile),
                        num=epi_tile,
                    )
                k.tcgen05_wait_ld()
                # Free accumulator TMEM before scale, cast, and store.
                k.wg_sync(barrier_id=10)
                with k.if_(k.tid_in_wg().eq(0)):
                    k.mbarrier_arrive(tmem_empty_leader, stage=tmem_idx)
                # Fire the teardown handshake's ARRIVE here on the last task
                # (an mbarrier arrive is loop-legal; only alloc/dealloc/
                # relinquish are restricted): the trailing wait is then free.
                with k.if_(local_iter.eq((pair_tasks - 1 - task_start) // task_step)):
                    with k.if_(k.tid_in_wg().eq(0)):
                        k.mbarrier_arrive(k.mbar_ref(tmem_fin, remote_coord=1 - cta_rank), stage=0)
                # alpha is baked at build time: skip the rescale entirely when 1.0
                # (a per-thread mul-by-1.0 over the whole fragment is pure waste).
                if config.alpha != 1.0:
                    k.reg_mul(accum_frag, accum_frag, config.alpha)
                k.reg_cvt(out_frag, accum_frag)
                for ot in range(store_tiles):
                    store_band(local_iter, d_m, d_n, ot, frag_off=ot * epi_tile)
            else:
                # OVERLAP: fuse {load EPI_TILE}.
                for ot in range(store_tiles):
                    k.tcgen05_ld(accum_frag, accum.at(0, acc_col0 + ot * epi_tile), num=epi_tile)
                    k.tcgen05_wait_ld()
                    # alpha baked at build time: skip the rescale when 1.0.
                    if config.alpha != 1.0:
                        k.reg_mul(accum_frag, accum_frag, config.alpha)
                    k.reg_cvt(out_frag, accum_frag)
                    if ot == store_tiles - 1:
                        # Free the accumulator after the LAST tile's reads.
                        k.wg_sync(barrier_id=10)
                        with k.if_(k.tid_in_wg().eq(0)):
                            k.mbarrier_arrive(tmem_empty_leader, stage=tmem_idx)
                        # The teardown handshake's ARRIVE fires here on the
                        # last task (loop-legal op): the trailing wait is free.
                        with k.if_(local_iter.eq((pair_tasks - 1 - task_start) // task_step)):
                            with k.if_(k.tid_in_wg().eq(0)):
                                k.mbarrier_arrive(k.mbar_ref(tmem_fin, remote_coord=1 - cta_rank), stage=0)
                    store_band(local_iter, d_m, d_n, ot, frag_off=0)
        # No final full drain: the last task's TMA stores drain on the HW
        # side at kernel exit (D visibility is completion-ordered), so the
        # tmem_fin handshake + dealloc below overlaps the last store flight
        # (nvjet's PREEXIT-style tail).

    # TMEM teardown via canon's tmem_finished 2-CTA handshake (NOT a bare cluster_sync).
    epilogue_warp = 4  # wg1's first warp (num_warps=8: wg0=0-3, wg1=4-7); canon's EPILOGUE
    with k.if_warp(epilogue_warp):
        k.mbarrier_wait(tmem_fin, stage=0, phase=0)
        k.tmem_dealloc(0, N_COLS_TMEM, cta_group)

    return k.build()


def _validate_config(config: NvFp4GemmConfig, cta_k: int = CTA_K) -> None:
    for name in ("m", "n", "k"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"nvfp4_gemm {name} must be a positive integer")
    if config.k % cta_k != 0:
        raise ValueError(f"nvfp4_gemm k must be a multiple of cta_k ({cta_k})")
    cta_n = _cfg_cta_n(config)
    if cta_n not in (64, 128):
        raise ValueError("nvfp4_gemm cta_n must be 64 (MMA_N=128) or 128 (MMA_N=256)")
    mma_n = cta_n * CTA_GROUP
    if config.n % mma_n != 0:
        raise ValueError(f"nvfp4_gemm n must be a multiple of MMA_N={mma_n} (cta_n={cta_n})")
    epi_tile = _cfg_epi_tile(config)
    if mma_n % epi_tile != 0 or epi_tile % TMEM_LD_SIZE != 0:
        raise ValueError("nvfp4_gemm epi_tile must divide MMA_N and be a multiple of TMEM_LD_SIZE")
    d_depth = _cfg_d_depth(config)
    if not isinstance(d_depth, int) or isinstance(d_depth, bool) or d_depth < 2:
        raise ValueError("nvfp4_gemm d_depth must be an integer >= 2")
    sched_rows = _ceil_div(config.m, CTA_M)
    if sched_rows % CTA_GROUP != 0:
        raise ValueError("nvfp4_gemm requires an even number of M tiles per cluster pair")
    # The group-major scheduler decodes a cluster's linear task index over the cluster-tile grid.
    l2_group_size = _cfg_l2_group_size(config)
    if not isinstance(l2_group_size, int) or isinstance(l2_group_size, bool) or l2_group_size < 1:
        raise ValueError("nvfp4_gemm l2_group_size must be a positive integer")
    pair_rows = sched_rows // CLUSTER_M
    if pair_rows > l2_group_size and pair_rows % l2_group_size != 0:
        raise ValueError("nvfp4_gemm l2_group_size must divide the cluster-tile row count")
    if config.epilogue not in ("overlap", "no_overlap", "stmatrix"):
        raise ValueError('nvfp4_gemm epilogue must be "overlap", "no_overlap", or "stmatrix"')
    if config.epilogue == "stmatrix" and epi_tile % 16 != 0:
        # This schedule uses 16-column reg->smem store chunks.
        raise ValueError("nvfp4_gemm stmatrix epilogue requires epi_tile a multiple of 16")


def _validate_launch_shape(launch_shape: LaunchShape, cta_group: int) -> None:
    if not isinstance(launch_shape, tuple) or len(launch_shape) != 1:
        raise ValueError("nvfp4_gemm requires a 1D launch_shape")
    count = launch_shape[0]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("nvfp4_gemm launch_shape[0] must be a positive integer")
    if count % cta_group != 0:
        raise ValueError("nvfp4_gemm launch_shape[0] must be divisible by cta_group")


# Bench-suite interface (see bench/nymph_bench_guide.md).

KERNEL_META = {"name": "nymph_nvfp4_gemm", "category": "experimental", "compute_capability": 10}
BENCH_CONFIGS = [
    {"M": s, "N": s, "K": s, "label": f"{s}x{s}x{s}"} for s in [1024, 2048, 4096, 8192, 16384]
]


def _compile_nymph(M, N, K, alpha):
    import importlib.util
    import os
    import tempfile

    import tvm

    from ..nymph_rs import kernel_to_tirx_source

    src = kernel_to_tirx_source(
        build_nvfp4_gemm(NvFp4GemmConfig(m=M, n=N, k=K, alpha=alpha, **gemm_config_for(M, N, K)))
    )
    p = os.path.join(tempfile.mkdtemp(prefix="nymph_nvfp4_"), "g.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nymph_nvfp4_emitted", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return tvm.compile(
        tvm.IRModule({"main": m.main}), tvm.target.Target("cuda"), tir_pipeline="tirx"
    )


def run_bench(M, N, K, *, warmup=None, repeat=None, timer=None, **kwargs):
    """Bench canon vs nymph with the bench-suite's exact methodology."""
    import torch

    import tvm
    from tirx_kernels.gemm.nvfp4_gemm import (
        _load_cublaslt_nvfp4_ext,
        prepare_data,
        tir_ws_kernel,
    )
    from tvm.tirx.bench import bench

    target = tvm.target.Target("cuda")
    with target:
        canon = tvm.compile(
            tvm.IRModule({"main": tir_ws_kernel(M, N, K)}), target, tir_pipeline="tirx"
        )
    A, B, Asf, Bsf, alpha, Cref = prepare_data(M, N, K)
    alpha_f = float(alpha)
    # Same math on both sides.
    nymph = _compile_nymph(M, N, K, alpha_f)
    at = torch.tensor([alpha_f], device="cuda", dtype=torch.float)
    # Reinterpret FlashInfer's contiguous 2-D scale bytes as the physical 4-D GMEM view.
    Ae = Asf.view(torch.float8_e4m3fn).view(M // 128, K // 64, 32, 16)
    Be = Bsf.view(torch.float8_e4m3fn).view(N // 128, K // 64, 32, 16)
    oc = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    on = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    obl = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    # cuBLASLt nvfp4 as a first-class impl (canon's inline ext, pure async launch).
    cublaslt_ext = _load_cublaslt_nvfp4_ext()
    funcs = {
        "tir": lambda: canon(A, B, Asf, Bsf, at, oc),
        "tirx": lambda: nymph(A, B, Ae, Be, on),
        "cublaslt": lambda: cublaslt_ext.nvfp4_cublaslt(A, B, Asf, Bsf, alpha_f, obl, M, N, K),
    }
    for fn in funcs.values():
        fn()
    torch.cuda.synchronize()
    for name, out in (("tir", oc), ("tirx", on), ("cublaslt", obl)):
        cos = torch.nn.functional.cosine_similarity(out.float().flatten(), Cref.flatten(), dim=0)
        if cos < 0.98:
            raise AssertionError(f"{name} output diverges from reference (cosine={cos:.4f})")
    return bench(funcs, warmup=warmup, repeat=repeat, timer=timer, **kwargs)


def register_bench_interface() -> None:
    """Self-register into the bench-suite kernel cache (see bench/nymph_bench_guide.md)."""
    import sys

    from tirx_kernels.registry import _KERNEL_CACHE

    _KERNEL_CACHE[KERNEL_META["name"]] = sys.modules[__name__]
