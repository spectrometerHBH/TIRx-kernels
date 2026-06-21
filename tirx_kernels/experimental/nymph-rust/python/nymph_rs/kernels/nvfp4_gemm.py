"""NVFP4 GEMM (Blackwell sm100 block-scaled fp4) expressed in Nymph IR.

Faithful port of ``tirx_kernels/gemm/nvfp4_gemm.py``. The operands are FP4
(``e2m1``, two values packed per ``uint8`` byte) and the GEMM is block-scaled at
block size 16: one ``e4m3`` scale factor per 16 contiguous K-elements, applied
to BOTH operands, with a final global ``alpha`` rescale in the epilogue. The
port keeps, at the same granularity as the fp16/bf16, FA4, and fp8-blockwise
ports:

- the cluster datapath: ``CTA_GROUP=2`` with ``CLUSTER_M=2`` — the cluster pair
  takes two adjacent M tiles (A split by M across the pair) and shares one
  ``MMA_N = CTA_N * CTA_GROUP = 256`` N band (B split by N across the pair, its
  block scales held full-band in both CTAs) — canon's exact tiling. The
  block-scaled tcgen05 dispatch derives N from the accumulator's MMA_N columns
  and requires ``SFB_rows >= N``; ``SFB_tmem`` is ``128 * SFB_n_chunks`` rows
  (``SFB_n_chunks = MMA_N // 128 = 2``), i.e. a 256-row SF band that satisfies
  ``SFB_rows = 256 >= N = 256`` (physically folded into 128 lanes × 2 column
  super-blocks). The per-CTA accumulator is ``(128, MMA_N)``;
- the role split: one TMA-load warp (loads A/B AND the e4m3 scales — no permute
  warp; the TMA lands the scales in the tcgen05.cp-ready ``sf_smem_layout``), one
  MMA warp (issuing from the cluster leader only, ``elected`` so the cp/cp/gemm
  burst stays warp-converged), and one epilogue warpgroup;
- the pipeline protocol, mirroring canon's two TMA barriers: ``smem_full``
  (tile_full_bar — A+B arrive-expect-tx per k-tile) and ``sf_full``
  (scale_full_bar — SFA+SFB arrive-expect-tx); the MMA waits BOTH before the
  cp/gemm. ``smem_empty`` (a tcgen05_commit multicast to both CTAs) frees the
  SMEM ring stage; the single-stage ``tmem_pipe`` (``tmem_full`` =
  tcgen05_commit multicast, ``tmem_empty`` = both CTAs' epilogues arrive at the
  leader, first wait passes via the +1 phase offset);
- the data path: per k-tile (CTA_K=256) the MMA issues ``K_ITERS = CTA_K//MMA_K
  = 4`` block-scaled instructions of ``MMA_K=64``. Each issue covers 4 scale
  blocks (``SF_PER_MMA = MMA_K//16 = 4``) held as one packed-u32 e4m3 cell per
  operand row. SFA/SFB are TMA'd every k-tile, permuted in SMEM, ``tcgen05.cp``'d
  into TMEM, and read per issue. The accumulator (one TMEM stage of MMA_N cols)
  is rescaled by ``alpha`` and cast to bf16 in the epilogue.

Sub-value physical-layout details (SMEM swizzles, the scale permute's byte
shuffle, the e2m1 nibble packing, the SF-cell TMEM broadcast) are modeled
logically, exactly like the sibling GEMM/attention ports. ``alpha`` is applied
as the epilogue rescale; on silicon it is a runtime ``(1,)`` buffer, here a
power-of-two value-model constant (the value test fixes the global scales).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..builder import IRBuilder
from ..nymph_rs import (
    DType,
    FenceKind,
    FenceScope,
    Kernel,
    LaunchShape,
    MBarKind,
    MemorySpace,
    TensorSlice,
    TmemLayout,
    TmemLayoutKind,
)

CTA_M = 128  # per-CTA A tile rows (one M tile)
CTA_N = 128  # per-CTA B tile rows (this CTA's half of the shared N band) — canon's CTA_N
CTA_GROUP = 2
CLUSTER_M = 2
# MMA_N = CTA_N * CTA_GROUP = 256 — canon's exact tiling. The shared N band the pair
# computes together. The block-scaled tcgen05 dispatch derives N from the accumulator's
# MMA_N=256 column count and requires SFB_rows >= N=256; canon's SFB_tmem is therefore
# `128 * SFB_n_chunks` rows where SFB_n_chunks = SFB_N//128 = 2 → a 256-row SFB_tmem.
# The emitted gemm_async is `tmem[:, 0:256]` (N=256), B operand = MMA_N//CTA_GROUP = 128
# rows (CTA_N), B_N*cta_group = 128*2 = 256 = N, and SFB rows=256 >= 256 all hold.
MMA_N = CTA_N * CTA_GROUP  # 256, the shared N band the pair computes together
# SFB_tmem holds the full MMA_N-wide N band of B scales. TMEM is physically 128 lanes,
# so an MMA_N=256 band folds into SFB_N_CHUNKS=2 column super-blocks (canon's SFB_n_chunks
# = SFB_N // 128). The physical SF TMEM is (128, SF_CTA_K * SFB_N_CHUNKS).
SFB_N_CHUNKS = MMA_N // 128  # 2 — number of 128-row super-blocks in the SFB band
MMA_M = 256  # the pair's two M tiles
CTA_K = 256  # K per pipeline tile
MMA_K = 64  # block-scaled fp4 MMA instruction K
K_ITERS = CTA_K // MMA_K  # 4 MMA issues per k-tile
SF_BLOCK = 16  # one e4m3 scale per 16 K-elements
SF_PER_MMA = MMA_K // SF_BLOCK  # 4 scale blocks per MMA issue
SF_CTA_K = CTA_K // SF_BLOCK  # 16 e4m3 scale bytes per row per k-tile (canon SF_CTA_K)
SF_CELLS = CTA_K // SF_BLOCK // 4  # packed-u32 scale cells per row per k-tile == K_ITERS
BLK_K_BYTES = CTA_K // 2  # packed fp4 bytes per row per k-tile (2 e2m1 per byte)
EPI_TILE = 64
TMEM_LD_SIZE = 8
ACC_DEPTH = 1  # accumulator TMEM stages (MMA_N=256 fills half of 512; one stage)
D_DEPTH = 2  # D_smem store ring depth (store pacing)
SMEM_DEPTH = 5  # SMEM pipeline depth (mirrors TIRx PIPE_DEPTH)
N_COLS_TMEM = 512
TILE_GROUPS_ROW_SIZE = 16
SM_NUMBER = 148
U32_BYTES = 4


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

    # packed fp4 operand tiles, e4m3 scale bytes (1B each, canon SFA_in layout), bf16 out
    a_tile_bytes = blk_m * BLK_K_BYTES
    b_tile_bytes = blk_n * BLK_K_BYTES
    sfa_tile_bytes = blk_m * SF_CTA_K  # per k-tile, this CTA's M rows x 16 e4m3 bytes
    # canon's SFB_N==MMA_N path: the FULL MMA_N-wide N band's scales live in EVERY CTA
    # (multicast), not split by N like the B operand. The band is MMA_N=256 rows (canon's
    # SFB_N=CTA_N*CTA_GROUP=256), so SFB_smem is (MMA_N, SF_CTA_K) and SFB_tmem holds all
    # 256 rows (physically folded into 128 lanes × 2 column super-blocks; see SFB_N_CHUNKS).
    sfb_tile_bytes = MMA_N * SF_CTA_K  # per k-tile, the full N band's rows x 16 e4m3 bytes
    d_tile_bytes = blk_m * EPI_TILE * 2

    a_off = 0
    b_off = a_off + SMEM_DEPTH * a_tile_bytes
    sfa_off = b_off + SMEM_DEPTH * b_tile_bytes
    sfb_off = sfa_off + SMEM_DEPTH * sfa_tile_bytes
    d_off = sfb_off + SMEM_DEPTH * sfb_tile_bytes
    smem_size_bytes = d_off + D_DEPTH * d_tile_bytes

    k = IRBuilder(
        "nymph_nvfp4_gemm",
        num_warps=8,  # wg0 = tma/permute/mma warps, wg1 = epilogue
        smem_size_bytes=smem_size_bytes,
        launch_shape=launch_shape,
        cluster_shape=(cta_group,),
    )
    # Operands are packed fp4: uint8[rows, K//2] (two e2m1 per byte), exactly the
    # TIRx A_packed/B_packed storage. Scales are packed e4m3 cells (u32).
    a_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.U8, shape=(M, K // 2))
    b_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.U8, shape=(N, K // 2))
    # Scales are e4m3 (one byte per 16-K block), laid out (rows, K//16) exactly like
    # canon's SFA_in/SFB_in — the codegen synthesizes sf_smem_layout from the e4m3 dtype.
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
    # SFA_smem holds. The codegen gives any e4m3 SMEM buffer canon's sf_smem_layout,
    # so the TMA lands the bytes in the tcgen05.cp-ready order (no permute warp).
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

    # TMEM: accumulator (one MMA_N stage) at col 0; the e4m3 scale vectors at canon's
    # fixed SF cols (448 / 464). The codegen emits these as `alloc_sf(...,
    # "float8_e4m3fn", sf_per_mma=4)` (recognized by the e4m3 TMEM dtype).
    sfa_col0 = 448
    sfb_col0 = 464
    tmem_base = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F32,
        shape=(128, N_COLS_TMEM),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=0),
    )
    accum = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F32,
        shape=(128, ACC_DEPTH * MMA_N),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=0),
    )
    sfa_tmem = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F8E4M3,
        shape=(128, SF_CTA_K),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=sfa_col0),
    )
    # The FULL N band's B scales — canon's SFB_tmem = `128 * SFB_n_chunks` LOGICAL rows
    # with SFB_n_chunks = SFB_N//128 = MMA_N//128 = 2 (a 256-row SF band). The block-scaled
    # gemm dispatch requires SFB_rows >= N=MMA_N=256, so the EMITTED SFB_tmem is logically
    # (256, SF_CTA_K). But TMEM is physically 128 lanes: the 256 logical rows fold into
    # SFB_n_chunks=2 column super-blocks, so the IR/value-model tensor is the PHYSICAL
    # (128, SF_CTA_K * SFB_n_chunks) = (128, 32) — validate.rs/the cp/mma SF slices require
    # a 128-lane TMEM slice. The codegen un-folds this back to canon's logical (256, 16)
    # decl_buffer (sf_tmem_layout(rows=256) packs the 2nd super-block into cols 16..32).
    # Folded (128, 32) at col_start=464 spans cols 464..496, clear of SFA's 448..464.
    sfb_tmem = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F8E4M3,
        shape=(128, SF_CTA_K * SFB_N_CHUNKS),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=sfb_col0),
    )

    accum_frag = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(TMEM_LD_SIZE,))
    out_frag = k.tensor(space=MemorySpace.REG, dtype=DType.BF16, shape=(TMEM_LD_SIZE,))

    smem_full = k.mbar(kind=MBarKind.TMA, stages=SMEM_DEPTH)
    # Separate SF-load completion barrier (canon's `scale_full_bar`/buffer_6, distinct from
    # the A/B `tile_full_bar`/buffer_5). Canon waits BOTH in the MMA before the cp/gemm; the
    # scale and tile loads land on different barriers (and on different TMA warps). Mirror
    # canon exactly — one barrier for A/B, one for SFA/SFB — so the MMA orders against each.
    sf_full = k.mbar(kind=MBarKind.TMA, stages=SMEM_DEPTH)
    smem_empty = k.mbar(kind=MBarKind.TCGEN05, stages=SMEM_DEPTH)
    tmem_full = k.mbar(kind=MBarKind.TCGEN05, stages=ACC_DEPTH)
    tmem_empty = k.mbar(kind=MBarKind.THREAD, stages=ACC_DEPTH)
    tmem_empty_leader = k.mbar_ref(tmem_empty, remote_coord=0)
    # The cluster MMA reads the PEER CTA's A/B/SF tiles too (cta_group=2,
    # multicast_cta_mask=0b11): the leader's tcgen05_cp/tcgen05_mma read
    # smem:cta1 as well as its own. So the leader must wait the PEER's
    # smem_full before issuing — exactly like the fp16/bf16 port — or the
    # peer CTA's TMA load has no happens-before edge to the leader's
    # operand read (the tcgen05_operand_overwrite_before_drain race).
    peer_smem_full = k.mbar_ref(smem_full, remote_coord=1)
    # The SF cp also reads the PEER CTA's SF SMEM (cta_group=2), so the leader orders
    # against the peer's SF load via the peer's sf_full too (symmetric with peer_smem_full).
    peer_sf_full = k.mbar_ref(sf_full, remote_coord=1)
    # tmem_fin: canon's exact lightweight 2-CTA teardown handshake (its `tmem_finished`).
    # Issued by the epilogue warp (warp 4): each CTA's warp-4/lane-0 arrives at the PEER
    # CTA's barrier, then waits its OWN — so BOTH CTAs' epilogues finished their TMEM
    # accumulator reads before either CTA deallocs the shared TMEM region. Only this warp
    # waits; the rest exit. This is the canon-faithful teardown (a bare finalize
    # `cluster_sync` is needlessly heavy — it forces all 8 warps to converge at one PC,
    # and warp 0 is the MMA/loader, not the epilogue, so it wouldn't even order against
    # the epilogue's TMEM reads). Mirrors the proven fp16 overlap port's tmem_fin.
    tmem_fin = k.mbar(kind=MBarKind.THREAD, stages=1)

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

    with k.kernel_init(warp=0):
        k.tmem_alloc(tmem_base, n_cols=N_COLS_TMEM, cta_group=cta_group)
        for s in range(SMEM_DEPTH):
            k.mbarrier_init(smem_full, count=1, stage=s)
            k.mbarrier_init(sf_full, count=1, stage=s)
            k.mbarrier_init(smem_empty, count=1, stage=s)
        for s in range(ACC_DEPTH):
            k.mbarrier_init(tmem_full, count=1, stage=s)
            k.mbarrier_init(tmem_empty, count=cta_group, stage=s)
        # canon's tmem_finished (init_full=1): one cross-CTA arrival releases the wait.
        k.mbarrier_init(tmem_fin, count=1, stage=0)

    # Prologue cross-CTA sync (canon's `T.ptx.fence.mbarrier_init` +
    # `barrier.cluster.arrive/wait`): seal the mbarrier-init epoch so the inits and
    # the TMEM alloc are visible to the async engines, then converge every CTA
    # thread of the cluster before any role touches a peer mbarrier or the shared
    # TMEM. Written EXPLICITLY here (codegen never fabricates it) — the same
    # fused no-overlap form the fp16/bf16 port uses. Without it the MMA/epilogue
    # race the still-pending alloc/init on the peer CTA (illegal tcgen05).
    k.fence(kind=FenceKind.MBARRIER_INIT)
    k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
    k.cluster_sync()

    # ---- TMA producer (wg0/warp2 — canon's WarpRole.TMA) ----
    with k.role(warp=2):
        with k.for_each_task(task_scheduler) as task:
            local_iter = (task.task_id - task_start) // task_step
            work_idx = task.task_id * cta_group + cta_rank
            m_idx, n_idx = work_coords(work_idx)
            a_m = m_idx * CTA_M  # this CTA's own M tile
            b_n = n_idx * MMA_N + cta_rank * CTA_N  # this CTA's half of the N band
            sf_n = n_idx * MMA_N  # the FULL N band's B scales (rank-independent)
            # Rolled k-loop (canon's T.serial) — a Python range unrolls in the IR, which
            # ~doubles the emitted CUDA tcgen05 ops vs canon and breaks multi-k-tile.
            with k.for_loop(stop=k_tiles) as t:
                seq = local_iter * k_tiles + t
                stage = seq % SMEM_DEPTH
                occ = seq // SMEM_DEPTH
                k.mbarrier_wait(smem_empty, stage=stage, phase=(occ + 1) % 2)
                # canon's split arrive: A/B tx -> smem_full (tile_full_bar), SFA/SFB tx ->
                # sf_full (scale_full_bar). Two barriers, each with its own expect_tx.
                k.mbarrier_arrive_expect_tx(smem_full, bytes=ab_bytes, stage=stage)
                k.mbarrier_arrive_expect_tx(sf_full, bytes=sf_bytes, stage=stage)
                kb = t * BLK_K_BYTES  # packed-fp4 byte column
                k.tma_load(
                    TensorSlice(
                        tensor=a_smem, offsets=(stage, 0, 0), shape=(1, blk_m, BLK_K_BYTES)
                    ),
                    a_gmem,
                    mbar=smem_full,
                    bytes=a_tile_bytes,
                    coords=(a_m, kb),
                    shape=(1, blk_m, BLK_K_BYTES),
                    gmem_shape=(blk_m, BLK_K_BYTES),
                    mbar_stage=stage,
                )
                k.tma_load(
                    TensorSlice(
                        tensor=b_smem, offsets=(stage, 0, 0), shape=(1, blk_n, BLK_K_BYTES)
                    ),
                    b_gmem,
                    mbar=smem_full,
                    bytes=b_tile_bytes,
                    coords=(b_n, kb),
                    shape=(1, blk_n, BLK_K_BYTES),
                    gmem_shape=(blk_n, BLK_K_BYTES),
                    mbar_stage=stage,
                )
                # SFA: this CTA's M rows; SFB: the full N band. e4m3 (rows, SF_CTA_K)
                # straight from the (rows, K//16) GMEM at this k-tile's column band,
                # exactly canon's SFA_in/SFB_in slice.
                sf_k = t * SF_CTA_K
                k.tma_load(
                    TensorSlice(tensor=sfa_smem, offsets=(stage, 0, 0), shape=(1, blk_m, SF_CTA_K)),
                    sfa_gmem,
                    mbar=sf_full,
                    bytes=sfa_tile_bytes,
                    coords=(a_m, sf_k),
                    shape=(1, blk_m, SF_CTA_K),
                    gmem_shape=(blk_m, SF_CTA_K),
                    mbar_stage=stage,
                )
                # SFB: the FULL MMA_N=256-wide N band (rank-independent sf_n), so both
                # CTAs hold the same full-band scales (canon's SFB_N==MMA_N path). The B
                # operand is still N-split (b_n), but its scales are not.
                k.tma_load(
                    TensorSlice(tensor=sfb_smem, offsets=(stage, 0, 0), shape=(1, MMA_N, SF_CTA_K)),
                    sfb_gmem,
                    mbar=sf_full,
                    bytes=sfb_tile_bytes,
                    coords=(sf_n, sf_k),
                    shape=(1, MMA_N, SF_CTA_K),
                    gmem_shape=(MMA_N, SF_CTA_K),
                    mbar_stage=stage,
                )

    # ---- MMA (wg0/warp0, cluster leader only — canon's WarpRole.MMA) ----
    # MMA on warp 0 (the same warp that did the tcgen05.alloc), exactly like canon.
    # No permute warp (canon has none): the TMA lands the e4m3 scales in the
    # cp-ready layout, the MMA warp copies them SMEM->TMEM and issues ONE
    # block-scaled gemm over the full CTA_K tile, exactly like canon's execute_mma.
    # The physical folded SFB SF-TMEM is (128, SF_CTA_K * SFB_N_CHUNKS): the MMA_N-wide
    # band's 256 logical rows fold into SFB_N_CHUNKS column super-blocks. The cp dst and
    # the mma sfb slice address the full (128, sfb_cols) physical band (covers both
    # super-blocks → numel matches the (256, SF_CTA_K) src, and the MMA read region spans
    # the whole written footprint).
    sfb_cols = SF_CTA_K * SFB_N_CHUNKS  # the full N band's SF cells per lane (folded)
    # elected=True: the WHOLE MMA worker loop runs single-issue (canon's `if elect_sync():
    # while ...`). The B200 tensor pipe needs the cp/cp/gemm/commit burst issued by one
    # converged lane — per-op elect guards that reconverge the warp between tcgen05 issues
    # stall the async stream (a GPU deadlock). tcgen05.mma/cp are single-thread-ISSUE ops,
    # so the value model computes the MMA from SMEM/TMEM regardless of cohort size (the
    # checker's tcgen05 issuer rule accepts a single elected lane).
    with k.role(warp=0, elected=True):
        with k.for_each_task(task_scheduler) as task:
            local_iter = (task.task_id - task_start) // task_step
            with k.if_(cta_rank.eq(0)):
                tmem_idx = local_iter % ACC_DEPTH
                k.mbarrier_wait(tmem_empty, stage=tmem_idx, phase=(local_iter // ACC_DEPTH + 1) % 2)
                acc_slice = TensorSlice(
                    tensor=accum, offsets=(0, tmem_idx * MMA_N), shape=(128, MMA_N)
                )

                def mma_ktile(t, accum_flag):
                    # one k-tile: wait the staged loads, cp the e4m3 scales SMEM->TMEM,
                    # issue ONE block-scaled gemm over CTA_K, free the smem stage.
                    seq = local_iter * k_tiles + t
                    stage = seq % SMEM_DEPTH
                    occ = seq // SMEM_DEPTH
                    # smem_full starts EMPTY (parity 0) and is flipped 0->1 only when the
                    # loader's TMA arrive + all four complete-tx land. mbarrier_wait blocks
                    # while parity == phase and wakes on the flip, so the consumer must wait
                    # phase=occ%2 (block at the OLD parity until the load completes). The
                    # flipped (occ+1)%2 passes vacuously on the first occupancy — the cp/gemm
                    # then read the SF/operand SMEM before the load lands (the
                    # tcgen05_operand_overwrite_before_drain race the checker flags).
                    # canon waits BOTH the scale_full_bar (sf_full) and the tile_full_bar
                    # (smem_full) before the cp/gemm. Wait sf_full FIRST (canon's order:
                    # buffer_6 then buffer_5) so the SF cp's source is ready, then smem_full
                    # for the A/B operands.
                    k.mbarrier_wait(sf_full, stage=stage, phase=occ % 2)
                    k.mbarrier_wait(smem_full, stage=stage, phase=occ % 2)
                    # Also wait the PEER CTA's barriers: the cluster MMA's cp/gemm read
                    # smem:cta1 too, so the leader must order against the peer's TMA loads
                    # (both A/B via peer_smem_full and SF via peer_sf_full) before issuing.
                    k.mbarrier_wait(peer_sf_full, stage=stage, phase=occ % 2)
                    k.mbarrier_wait(peer_smem_full, stage=stage, phase=occ % 2)
                    k.tcgen05_cp(
                        TensorSlice(tensor=sfa_tmem, offsets=(0, 0), shape=(128, SF_CTA_K)),
                        TensorSlice(
                            tensor=sfa_smem, offsets=(stage, 0, 0), shape=(1, blk_m, SF_CTA_K)
                        ),
                        cta_group=cta_group,
                    )
                    k.tcgen05_cp(
                        TensorSlice(tensor=sfb_tmem, offsets=(0, 0), shape=(128, sfb_cols)),
                        TensorSlice(
                            tensor=sfb_smem, offsets=(stage, 0, 0), shape=(1, MMA_N, SF_CTA_K)
                        ),
                        cta_group=cta_group,
                    )
                    # canon's cluster gemm: n = MMA_N (the full 256-wide N band the pair
                    # computes together), m = MMA_M; each CTA supplies its own blk_n=128-row
                    # B half (b_smem holds it, CTA_N=MMA_N//CTA_GROUP rows) and the 2-CTA MMA
                    # writes the per-CTA (128, MMA_N) accumulator. The block-scaled tcgen05
                    # dispatch derives N from the accumulator's MMA_N=256 columns: B_N=128,
                    # B_N*cta_group=256=N, and SFB rows=256 >= N=256 (canon's 256-row
                    # SFB_tmem = 128*SFB_n_chunks). SFA (128, SF_CTA_K), SFB (256, SF_CTA_K).
                    a_op = TensorSlice(
                        tensor=a_smem, offsets=(stage, 0, 0), shape=(1, blk_m, BLK_K_BYTES)
                    )
                    b_op = TensorSlice(
                        tensor=b_smem, offsets=(stage, 0, 0), shape=(1, blk_n, BLK_K_BYTES)
                    )
                    k.tcgen05_mma(
                        acc_slice,
                        a_op,
                        b_op,
                        m=MMA_M,
                        n=MMA_N,
                        k=CTA_K,
                        accum=accum_flag,
                        cta_group=cta_group,
                        sfa=TensorSlice(tensor=sfa_tmem, offsets=(0, 0), shape=(128, SF_CTA_K)),
                        sfb=TensorSlice(tensor=sfb_tmem, offsets=(0, 0), shape=(128, sfb_cols)),
                        sf_e4m3=True,
                        sf_block=SF_BLOCK,
                        a_fp4=True,
                        b_fp4=True,
                    )
                    k.tcgen05_commit(
                        smem_empty, stage=stage, cta_group=cta_group, multicast_cta_mask=0b11
                    )

                # Peel the first k-tile (accum=False), roll the rest with accum=True
                # (canon's `accum=0` then `accum=1`). The rolled loop keeps the emitted
                # CUDA tcgen05 op count at canon's level (a Python range would unroll it).
                mma_ktile(0, False)
                with k.for_loop(stop=k_tiles - 1) as ti:
                    mma_ktile(ti + 1, True)
                k.tcgen05_commit(
                    tmem_full, stage=tmem_idx, cta_group=cta_group, multicast_cta_mask=0b11
                )

    # ---- epilogue (wg1) ----
    with k.role(warpgroup=1):
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
                with k.if_(store_iter >= D_DEPTH):
                    k.cp_async_bulk_wait_group_read(D_DEPTH - 1)
                    k.wg_sync(barrier_id=10)
                d_stage = store_iter % D_DEPTH
                for ki in range(EPI_TILE // TMEM_LD_SIZE):
                    col = tmem_idx * MMA_N + ot * EPI_TILE + ki * TMEM_LD_SIZE
                    k.tcgen05_ld(accum_frag, accum, num=TMEM_LD_SIZE, row=0, col=col)
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
                if ot == store_tiles - 1:
                    k.mbarrier_arrive(tmem_empty_leader, stage=tmem_idx)
                k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                k.wg_sync(barrier_id=10)
                k.tma_store(
                    d_gmem,
                    TensorSlice(tensor=d_smem, offsets=(d_stage, 0, 0), shape=(1, blk_m, EPI_TILE)),
                    coords=(d_m, d_n + ot * EPI_TILE),
                    shape=(1, blk_m, EPI_TILE),
                    gmem_shape=(blk_m, EPI_TILE),
                )
                k.cp_async_bulk_commit_group()
        k.cp_async_bulk_wait_group_read(0)
        k.wg_sync(barrier_id=10)

    # TMEM teardown via canon's tmem_finished 2-CTA handshake (NOT a bare cluster_sync).
    # Issued by the EPILOGUE warp (warp 4 = canon's WarpRole.EPILOGUE), which reaches the
    # finalize only after its whole warpgroup passed the post-loop `wg_sync(10)` — i.e.
    # all of wg1's accumulator (TMEM) reads are done. It arrives at the PEER CTA's tmem_fin,
    # then waits its OWN, so BOTH CTAs' epilogues finished reading the shared TMEM before
    # either deallocs. Only this one warp waits; the rest exit. This is canon's exact
    # `tmem_finished` teardown (also the proven fp16 overlap port's tmem_fin), and unlike a
    # bare `cluster_sync` it picks the EPILOGUE warp (warp 0 is the MMA/loader, whose dealloc
    # would not order against the epilogue's TMEM reads).
    epilogue_warp = 4  # wg1's first warp (num_warps=8: wg0=0-3, wg1=4-7); canon's EPILOGUE
    with k.kernel_finalize(warp=epilogue_warp):
        k.mbarrier_arrive(k.mbar_ref(tmem_fin, remote_coord=1 - cta_rank), stage=0)
        k.mbarrier_wait(tmem_fin, stage=0, phase=0)
        k.tmem_dealloc(tmem_base, n_cols=N_COLS_TMEM, cta_group=cta_group)

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
