"""NVFP4 GEMM (Blackwell sm100 block-scaled fp4) expressed in Nymph IR.

Faithful port of ``tirx_kernels/gemm/nvfp4_gemm.py``. The operands are FP4
(``e2m1``, two values packed per ``uint8`` byte) and the GEMM is block-scaled at
block size 16: one ``e4m3`` scale factor per 16 contiguous K-elements, applied
to BOTH operands, with a final global ``alpha`` rescale in the epilogue. The
port keeps, at the same granularity as the fp16/bf16, FA4, and fp8-blockwise
ports:

- the cluster datapath: ``CTA_GROUP=2`` with ``CLUSTER_M=2`` — the cluster pair
  takes two adjacent M tiles (A split by M across the pair) and shares one
  ``MMA_N = CTA_N * CTA_GROUP = 256`` N band (B split by N across the pair). This
  is the verified ``(m=256, cta_group=2)`` block-scaled MMA;
- the role split: one TMA-load warp, one MMA warp (issuing from the cluster
  leader only), and one epilogue warpgroup. There is NO permute warp (canon has
  none): the SF SMEM rings carry canon's SF physical layout, so the TMA lands
  the e4m3 scales in the tcgen05.cp-ready order;
- the pipeline protocol: ``smem_full`` (leader-routed TMA arrive-expect-tx per
  k-tile covering A+B+SFA+SFB — both CTAs' completions signal the LEADER's
  barrier, the legal substitute for a peer ``try_wait``), ``smem_empty`` (a
  tcgen05_commit multicast to both CTAs), and the single-stage ``tmem_pipe``
  (full = tcgen05_commit multicast, empty = both CTAs' epilogues arrive at the
  leader, first wait passes via the +1 phase offset);
- the data path: per k-tile (CTA_K=256) the MMA issues ONE block-scaled
  instruction over the full K tile (canon's one-issue full-K ``gemm_async``; the
  IR's dense k is an ordered run of k/16 atomic MMAs). The 16 e4m3 scale blocks
  (``SF_CTA_K = CTA_K//16``) are TMA'd every k-tile, ``tcgen05.cp``'d into TMEM
  (the folded 128-lane super-block layout), and read by the issue. The
  accumulator (one TMEM stage of MMA_N cols) is rescaled by ``alpha`` and cast
  to bf16 in the epilogue.

Sub-value physical-layout details (SMEM swizzles, the e2m1 nibble packing, the
SF-cell TMEM broadcast) are modeled logically, exactly like the sibling
GEMM/attention ports. ``alpha`` is applied as the epilogue rescale; on silicon
it is a runtime ``(1,)`` buffer, here a power-of-two value-model constant (the
value test fixes the global scales).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..builder import IRBuilder, SfTmemBand, TmemBand
from ..nymph_rs import (
    DType,
    FenceKind,
    FenceScope,
    Kernel,
    LaunchShape,
    MBarKind,
    MemorySpace,
    TensorSlice,
)

CTA_M = 128  # per-CTA A tile rows (one M tile)
CTA_N = 128  # per-CTA B tile rows (this CTA's half of the shared N band)
CTA_GROUP = 2
CLUSTER_M = 2
MMA_N = CTA_N * CTA_GROUP  # 256, the shared N band the pair computes together
MMA_M = 256  # the pair's two M tiles
CTA_K = 256  # K per pipeline tile
SF_BLOCK = 16  # one e4m3 scale per 16 K-elements
SF_CTA_K = CTA_K // SF_BLOCK  # 16 e4m3 scale bytes per row per k-tile (canon SF_CTA_K)
BLK_K_BYTES = CTA_K // 2  # packed fp4 bytes per row per k-tile (2 e2m1 per byte)
EPI_TILE = 64
TMEM_LD_SIZE = 8
ACC_DEPTH = 1  # accumulator TMEM stages (MMA_N=256 fills half of 512; one stage)
D_DEPTH = 2  # D_smem store ring depth (store pacing)
SMEM_DEPTH = 5  # SMEM pipeline depth (mirrors TIRx PIPE_DEPTH)
N_COLS_TMEM = 512
TILE_GROUPS_ROW_SIZE = 16
SM_NUMBER = 148
SF_BASE_COL = 448  # canon's fixed SF base column (the accumulator spans 0..MMA_N)


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


@dataclass(frozen=True, slots=True)
class NvFp4GemmConfig:
    m: int = 1024
    n: int = 1024
    k: int = 1024
    # alpha = 1 / (A_global_sf * B_global_sf), applied as the epilogue rescale.
    # A power of two keeps the value model bit-exact; the default 1.0 matches a
    # unit global scale.
    alpha: float = 1.0
    launch_shape: LaunchShape | None = None


def nvfp4_task_config(tasks: int) -> NvFp4GemmConfig:
    """The canonical single-cluster examination setting, mirroring the fp16/bf16
    and fp8 ones: one m tile per CTA (M = CTA_M * CLUSTER_M = 256, two adjacent M
    tiles), k = 16384 (64 k-tiles per task), on ONE cluster pair
    (``launch_shape=(2,)``), with the task count varied through N = MMA_N * tasks
    (each pair-task covers one 256-wide N band)."""
    return NvFp4GemmConfig(
        m=CTA_M * CLUSTER_M, n=MMA_N * tasks, k=16384, alpha=1.0, launch_shape=(2,)
    )


CONFIGS = [
    {"m": s, "n": s, "k": s, "label": f"{s}x{s}x{s}"} for s in [1024, 2048, 4096, 8192, 16384]
]

# Cheap shapes for the protocol + value sweeps: different k-tile counts and tile
# counts, all on the single cluster datapath. Larger squares build/protocol only.
NVFP4_CONFIGS_SUPPORTED = [
    {"m": 256, "n": 256, "k": 256, "label": "256x256x256"},  # 1 k-tile, 1 tile
    {"m": 256, "n": 512, "k": 512, "label": "256x512x512"},  # 2 k-tiles, 2 N tiles
    {"m": 512, "n": 256, "k": 512, "label": "512x256x512"},  # 2 M tiles (2 pairs)
    {"m": 512, "n": 512, "k": 1024, "label": "512x512x1024"},  # 4 k-tiles, 2x2 tiles
    {"m": 1024, "n": 1024, "k": 1024, "label": "1024x1024x1024"},
]


def build_nvfp4_gemm(config: NvFp4GemmConfig = NvFp4GemmConfig()) -> Kernel:
    M, N, K = config.m, config.n, config.k
    _validate_config(config)
    cta_group = CTA_GROUP
    blk_m = CTA_M  # per-CTA A rows (its own M tile)
    blk_n = CTA_N  # per-CTA B rows (its half of the shared N band)
    sched_rows = _ceil_div(M, CTA_M)  # M tiles
    sched_cols = _ceil_div(N, MMA_N)  # N bands
    k_tiles = K // CTA_K
    total_work = sched_rows * sched_cols
    store_tiles = MMA_N // EPI_TILE

    launch_shape = config.launch_shape or (
        max(cta_group, min(SM_NUMBER, total_work) // cta_group * cta_group),
    )
    _validate_launch_shape(launch_shape, cta_group)
    pair_tasks = total_work // cta_group

    # packed fp4 operand tiles, e4m3 scale bytes, bf16 output tile
    a_tile_bytes = blk_m * BLK_K_BYTES
    b_tile_bytes = blk_n * BLK_K_BYTES
    sfa_tile_bytes = blk_m * SF_CTA_K  # per k-tile, this CTA's M rows x 16 e4m3 bytes
    # canon's SFB_N==MMA_N path: the FULL MMA_N-wide N band's scales live in
    # EVERY CTA, not split by N like the B operand. The band is MMA_N=256 rows,
    # so SFB_smem is (MMA_N, SF_CTA_K) and SFB_tmem holds all 256 rows
    # (physically folded into 128 lanes x 2 column super-blocks).
    sfb_tile_bytes = MMA_N * SF_CTA_K  # per k-tile, the full N band's rows x 16 bytes
    d_tile_bytes = blk_m * EPI_TILE * 2

    a_off = 0
    b_off = a_off + SMEM_DEPTH * a_tile_bytes
    sfa_off = b_off + SMEM_DEPTH * b_tile_bytes
    sfb_off = sfa_off + SMEM_DEPTH * sfa_tile_bytes
    d_off = sfb_off + SMEM_DEPTH * sfb_tile_bytes
    smem_size_bytes = d_off + D_DEPTH * d_tile_bytes

    k = IRBuilder(
        "nymph_nvfp4_gemm",
        num_warps=8,  # wg0 = tma/mma warps, wg1 = epilogue
        smem_size_bytes=smem_size_bytes,
        launch_shape=launch_shape,
        cluster_shape=(cta_group,),
    )
    # Operands are packed fp4: uint8[rows, K//2] (two e2m1 per byte), exactly the
    # TIRx A_packed/B_packed storage. Scales are e4m3 (one byte per 16-K block),
    # laid out (rows, K//16) exactly like canon's SFA_in/SFB_in.
    a_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.U8, shape=(M, K // 2))
    b_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.U8, shape=(N, K // 2))
    sfa_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.F8E4M3, shape=(M, K // SF_BLOCK))
    sfb_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.F8E4M3, shape=(N, K // SF_BLOCK))
    d_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.BF16, shape=(M, N))

    # Stage-major SMEM rings, indexed by a runtime pipeline stage (the continuous
    # PipelineState seq % depth, never reset per task).
    a_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.U8,
        shape=(SMEM_DEPTH, blk_m, BLK_K_BYTES),
        byte_offset=a_off,
    )
    b_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.U8,
        shape=(SMEM_DEPTH, blk_n, BLK_K_BYTES),
        byte_offset=b_off,
    )
    # SF SMEM: e4m3 (row, SF_CTA_K) per stage, the same (CTA_M, K//16) tile canon's
    # SFA_smem holds. The codegen gives any SF-usage SMEM buffer canon's
    # sf_smem_layout, so the TMA lands the bytes in the tcgen05.cp-ready order
    # (no permute warp).
    sfa_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.F8E4M3,
        shape=(SMEM_DEPTH, blk_m, SF_CTA_K),
        byte_offset=sfa_off,
    )
    sfb_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.F8E4M3,
        shape=(SMEM_DEPTH, MMA_N, SF_CTA_K),
        byte_offset=sfb_off,
    )
    d_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.BF16,
        shape=(D_DEPTH, blk_m, EPI_TILE),
        byte_offset=d_off,
    )

    # TMEM: accumulator (one MMA_N stage) at col 0; the e4m3 scale vectors at
    # canon's fixed SF base col (448), the SFB band derived from the SFA band's
    # folded span. The physical folded SFB SF-TMEM band is (128, SF_CTA_K *
    # SFB_N_CHUNKS) cells: the MMA_N-wide band's 256 logical rows fold into
    # SFB_N_CHUNKS=2 column super-blocks (the SfTmemBand rule); the cp dst and
    # the mma sfb operand address the band's folded physical base col0.
    accum = TmemBand(col0=0, dtype=DType.F32)
    sfa_tmem = SfTmemBand(col0=SF_BASE_COL, rows=blk_m, nblocks=SF_CTA_K)
    sfb_tmem = SfTmemBand(col0=sfa_tmem.col0 + sfa_tmem.n_cols, rows=MMA_N, nblocks=SF_CTA_K)

    accum_frag = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(TMEM_LD_SIZE,))
    out_frag = k.tensor(space=MemorySpace.REG, dtype=DType.BF16, shape=(TMEM_LD_SIZE,))

    smem_full = k.mbar(kind=MBarKind.TMA, stages=SMEM_DEPTH, leader_routed=True)
    smem_empty = k.mbar(kind=MBarKind.TCGEN05, stages=SMEM_DEPTH)
    tmem_full = k.mbar(kind=MBarKind.TCGEN05, stages=ACC_DEPTH)
    tmem_empty = k.mbar(kind=MBarKind.THREAD, stages=ACC_DEPTH)
    peer_smem_full = k.mbar_ref(smem_full, remote_coord=1)
    tmem_empty_leader = k.mbar_ref(tmem_empty, remote_coord=0)

    cta_id = k.cta_id()
    cta_rank = k.ctaid_in_cluster()
    task_space = k.task_space(grid=(pair_tasks,), fields=("pair_idx",))
    task_scheduler = k.scheduler(task_space)
    task_start = cta_id // cta_group
    task_step = k.launch_cta_count // cta_group

    ab_bytes = a_tile_bytes + b_tile_bytes
    sf_bytes = sfa_tile_bytes + sfb_tile_bytes

    def work_coords(work_idx):
        """ClusterPersistentScheduler2D group-major mapping: rows (M tiles) walk
        within a TILE_GROUPS_ROW_SIZE-row L2 group, groups row-major. Consecutive
        work indices (a cluster pair) land on adjacent M tiles of the same N band.
        Returns (m_idx, n_idx)."""
        if sched_rows <= TILE_GROUPS_ROW_SIZE:
            return work_idx % sched_rows, work_idx // sched_rows
        group_span = TILE_GROUPS_ROW_SIZE * sched_cols
        group_id = work_idx // group_span
        within = work_idx % group_span
        m_idx = group_id * TILE_GROUPS_ROW_SIZE + within % TILE_GROUPS_ROW_SIZE
        n_idx = within // TILE_GROUPS_ROW_SIZE
        return m_idx, n_idx

    with k.if_warp(0):
        # tmem_alloc is warp-collective (full warp 0); mbarrier.init is
        # per-thread, so one elected thread runs the inits.
        k.tmem_alloc(0, N_COLS_TMEM, cta_group)
        with k.if_elected():
            for s in range(SMEM_DEPTH):
                k.mbarrier_init(smem_full, count=1, stage=s)
                k.mbarrier_init(smem_empty, count=1, stage=s)
            for s in range(ACC_DEPTH):
                k.mbarrier_init(tmem_full, count=1, stage=s)
                k.mbarrier_init(tmem_empty, count=cta_group, stage=s)
    # Publish the prologue cluster-wide (the pair signals each other's
    # leader-routed mbars) before any wait/arrive touches the cells.
    k.fence(kind=FenceKind.MBARRIER_INIT)
    k.cluster_sync()

    # ---- TMA producer (wg0/warp0, one issuing thread) ----
    with k.if_warp(0), k.if_elected():
        with k.for_each_task(task_scheduler) as task:
            local_iter = (task.task_id - task_start) // task_step
            work_idx = task.task_id * cta_group + cta_rank
            m_idx, n_idx = work_coords(work_idx)
            a_m = m_idx * CTA_M  # this CTA's own M tile
            b_n = n_idx * MMA_N + cta_rank * CTA_N  # this CTA's half of the N band
            sf_n = n_idx * MMA_N  # the FULL N band's B scales (rank-independent)
            for t in range(k_tiles):
                seq = local_iter * k_tiles + t
                stage = seq % SMEM_DEPTH
                occ = seq // SMEM_DEPTH
                k.mbarrier_wait(smem_empty, stage=stage, phase=(occ + 1) % 2)
                k.mbarrier_arrive_expect_tx(smem_full, bytes=ab_bytes + sf_bytes, stage=stage)
                kb = t * BLK_K_BYTES  # packed-fp4 byte column
                k.tma_load(
                    TensorSlice(
                        tensor=a_smem, offsets=(stage, 0, 0), shape=(1, blk_m, BLK_K_BYTES)
                    ),
                    a_gmem,
                    mbar=smem_full,
                    coords=(a_m, kb),
                    shape=(1, blk_m, BLK_K_BYTES),
                    gmem_shape=(blk_m, BLK_K_BYTES),
                    mbar_stage=stage,
                    cta_group=cta_group,
                )
                k.tma_load(
                    TensorSlice(
                        tensor=b_smem, offsets=(stage, 0, 0), shape=(1, blk_n, BLK_K_BYTES)
                    ),
                    b_gmem,
                    mbar=smem_full,
                    coords=(b_n, kb),
                    shape=(1, blk_n, BLK_K_BYTES),
                    gmem_shape=(blk_n, BLK_K_BYTES),
                    mbar_stage=stage,
                    cta_group=cta_group,
                )
                # SFA: this CTA's M rows; SFB: the full N band. e4m3 (rows,
                # SF_CTA_K) tiles — the TMA lands cp-ready bytes (no permute).
                sf_k = t * SF_CTA_K
                k.tma_load(
                    TensorSlice(tensor=sfa_smem, offsets=(stage, 0, 0), shape=(1, blk_m, SF_CTA_K)),
                    sfa_gmem,
                    mbar=smem_full,
                    coords=(a_m, sf_k),
                    shape=(1, blk_m, SF_CTA_K),
                    gmem_shape=(blk_m, SF_CTA_K),
                    mbar_stage=stage,
                    cta_group=cta_group,
                )
                k.tma_load(
                    TensorSlice(tensor=sfb_smem, offsets=(stage, 0, 0), shape=(1, MMA_N, SF_CTA_K)),
                    sfb_gmem,
                    mbar=smem_full,
                    coords=(sf_n, sf_k),
                    shape=(1, MMA_N, SF_CTA_K),
                    gmem_shape=(MMA_N, SF_CTA_K),
                    mbar_stage=stage,
                    cta_group=cta_group,
                )

    # ---- MMA (wg0/warp1, cluster leader only, one issuing thread) ----
    with k.if_warp(1), k.if_elected():
        with k.for_each_task(task_scheduler) as task:
            local_iter = (task.task_id - task_start) // task_step
            with k.if_(cta_rank.eq(0)):
                tmem_idx = local_iter % ACC_DEPTH
                k.mbarrier_wait(tmem_empty, stage=tmem_idx, phase=(local_iter // ACC_DEPTH + 1) % 2)
                acc_op = accum.at(0, tmem_idx * MMA_N)
                for t in range(k_tiles):
                    seq = local_iter * k_tiles + t
                    stage = seq % SMEM_DEPTH
                    occ = seq // SMEM_DEPTH
                    # smem_full starts EMPTY (parity 0) and is flipped 0->1 only
                    # when the loader's TMA arrive + all four complete-tx land.
                    # The leader also waits the PEER CTA's barrier: the cluster
                    # MMA's cp/gemm read smem:cta1 too (in sim, per-CTA barriers;
                    # codegen routes both CTAs' TMA completions to the leader's
                    # barrier instead — the legal substitute for a peer try_wait).
                    k.mbarrier_wait(smem_full, stage=stage, phase=occ % 2)
                    k.mbarrier_wait(peer_smem_full, stage=stage, phase=occ % 2)
                    # Copy this k-tile's e4m3 scales SMEM -> TMEM (A: this CTA's
                    # M rows; B: the full N band's 256 rows, folded).
                    k.tcgen05_cp(
                        sfa_tmem.at(),
                        TensorSlice(
                            tensor=sfa_smem, offsets=(stage, 0, 0), shape=(1, blk_m, SF_CTA_K)
                        ),
                        cta_group=cta_group,
                    )
                    k.tcgen05_cp(
                        sfb_tmem.at(),
                        TensorSlice(
                            tensor=sfb_smem, offsets=(stage, 0, 0), shape=(1, MMA_N, SF_CTA_K)
                        ),
                        cta_group=cta_group,
                    )
                    # canon's cluster gemm: n = MMA_N (the full 256-wide N band the
                    # pair computes together), m = MMA_M; each CTA supplies its own
                    # blk_n=128-row B half and the 2-CTA MMA writes the per-CTA
                    # (128, MMA_N) accumulator. ONE block-scaled issue over the
                    # full CTA_K tile (the IR's dense k = an ordered run of k/16
                    # atomic MMAs); SFA (128, SF_CTA_K), SFB (256, SF_CTA_K).
                    a_op = TensorSlice(
                        tensor=a_smem, offsets=(stage, 0, 0), shape=(1, blk_m, BLK_K_BYTES)
                    )
                    b_op = TensorSlice(
                        tensor=b_smem, offsets=(stage, 0, 0), shape=(1, blk_n, BLK_K_BYTES)
                    )
                    k.tcgen05_mma(
                        acc_op,
                        a_op,
                        b_op,
                        m=MMA_M,
                        n=MMA_N,
                        k=CTA_K,
                        accum=t > 0,
                        cta_group=cta_group,
                        sfa=sfa_tmem.at(),
                        sfb=sfb_tmem.at(),
                        sf_e4m3=True,
                        sf_block=SF_BLOCK,
                        a_fp4=True,
                        b_fp4=True,
                    )
                    k.tcgen05_commit(
                        smem_empty, stage=stage, cta_group=cta_group, multicast_cta_mask=0b11
                    )
                k.tcgen05_commit(
                    tmem_full, stage=tmem_idx, cta_group=cta_group, multicast_cta_mask=0b11
                )

    # ---- epilogue (wg1) ----
    with k.if_warpgroup(1):
        with k.for_each_task(task_scheduler) as task:
            local_iter = (task.task_id - task_start) // task_step
            work_idx = task.task_id * cta_group + cta_rank
            m_idx, n_idx = work_coords(work_idx)
            d_m = m_idx * CTA_M
            d_n = n_idx * MMA_N
            tmem_idx = local_iter % ACC_DEPTH
            k.mbarrier_wait(tmem_full, stage=tmem_idx, phase=(local_iter // ACC_DEPTH) % 2)
            for ot in range(store_tiles):
                store_iter = local_iter * store_tiles + ot
                # The D ring's oldest bulk store still reads d_smem until its
                # group drains. The group lives on the leader thread's stream
                # (it issues the commits below); the wg_sync publishes the
                # drain to the other warps before anyone overwrites the stage.
                with k.if_(store_iter >= D_DEPTH):
                    with k.if_(k.tid_in_wg().eq(0)):
                        k.cp_async_bulk_wait_group_read(D_DEPTH - 1)
                    k.wg_sync(barrier_id=10)
                d_stage = store_iter % D_DEPTH
                for ki in range(EPI_TILE // TMEM_LD_SIZE):
                    col = tmem_idx * MMA_N + ot * EPI_TILE + ki * TMEM_LD_SIZE
                    k.tcgen05_ld(accum_frag, accum.at(0, col), num=TMEM_LD_SIZE)
                    k.tcgen05_wait_ld()
                    # alpha rescale (epilogue global scale), then f32 -> bf16
                    k.reg_mul(accum_frag, accum_frag, config.alpha)
                    k.reg_cvt(out_frag, accum_frag)
                    k.reg_store(
                        TensorSlice(
                            tensor=d_smem,
                            offsets=(d_stage, k.tid_in_wg(), ki * TMEM_LD_SIZE),
                            shape=(1, 1, TMEM_LD_SIZE),
                        ),
                        out_frag,
                    )
                # All four warps' accumulator loads and d_smem writes must land
                # before the single-thread tail (tmem release + bulk store).
                k.wg_sync(barrier_id=10)
                with k.if_(k.tid_in_wg().eq(0)):
                    if ot == store_tiles - 1:
                        # One arrive per CTA's epilogue (tmem_empty count = 2);
                        # the wg_sync above proves every warp's loads retired.
                        k.mbarrier_arrive(tmem_empty_leader, stage=tmem_idx)
                    k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                    k.tma_store(
                        d_gmem,
                        TensorSlice(
                            tensor=d_smem, offsets=(d_stage, 0, 0), shape=(1, blk_m, EPI_TILE)
                        ),
                        coords=(d_m, d_n + ot * EPI_TILE),
                        shape=(1, blk_m, EPI_TILE),
                        gmem_shape=(blk_m, EPI_TILE),
                    )
                    k.cp_async_bulk_commit_group()
        with k.if_(k.tid_in_wg().eq(0)):
            k.cp_async_bulk_wait_group_read(0)
        k.wg_sync(barrier_id=10)

    # Teardown: every warp's pipeline work happens-before the pair dealloc.
    k.cluster_sync()
    with k.if_warp(0):
        k.tmem_relinquish(cta_group)
        k.tmem_dealloc(0, N_COLS_TMEM, cta_group)

    return k.build()


def _validate_config(config: NvFp4GemmConfig) -> None:
    for name in ("m", "n", "k"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"nvfp4_gemm {name} must be a positive integer")
    if config.k % CTA_K != 0:
        raise ValueError(f"nvfp4_gemm k must be a multiple of {CTA_K}")
    sched_rows = _ceil_div(config.m, CTA_M)
    if sched_rows % CTA_GROUP != 0:
        raise ValueError("nvfp4_gemm requires an even number of M tiles per cluster pair")
    if sched_rows > TILE_GROUPS_ROW_SIZE and sched_rows % TILE_GROUPS_ROW_SIZE != 0:
        raise ValueError("nvfp4_gemm supports tail-only or full-group tilings")


def _validate_launch_shape(launch_shape: LaunchShape, cta_group: int) -> None:
    if not isinstance(launch_shape, tuple) or len(launch_shape) != 1:
        raise ValueError("nvfp4_gemm requires a 1D launch_shape")
    count = launch_shape[0]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("nvfp4_gemm launch_shape[0] must be a positive integer")
    if count % cta_group != 0:
        raise ValueError("nvfp4_gemm launch_shape[0] must be divisible by cta_group")
