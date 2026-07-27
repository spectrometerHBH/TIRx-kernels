"""FP8 blockwise GEMM (DeepGEMM SM100 shape) expressed in Nymph IR."""

from __future__ import annotations

import math
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
    SmemSwizzleLayout,
    Swizzle,
    TensorSlice,
)

BLK_K = 128
MMA_K = 32  # block-scaled f8 MMA instruction K
K_ITERS = BLK_K // MMA_K  # MMA issues per k-tile
SF_PACK = 4  # UE8M0 bytes packed per u32 = k-tiles per scale load
EPI_TILE = 32
TMEM_LD_SIZE = 8
TMEM_DEPTH = 2
N_COLS_TMEM = 512
TILE_GROUPS_ROW_SIZE = 16
SM_NUMBER = 148
U32_BYTES = 4


def _align(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _swizzle_mode(block_size: int, elem_size: int) -> int:
    for mode in (128, 64, 32, 16):
        if (block_size * elem_size) % mode == 0:
            return mode
    raise AssertionError("unreachable swizzle mode")


def _deepgemm_num_stages(
    *, swap_ab: bool, block_m: int, block_n: int, load_block_m: int, load_block_n: int
) -> int:
    """Match DeepGEMM SM100 shared-memory stage count for FP8 normal GEMM."""

    block_k = 128
    swizzle_cd = _swizzle_mode(block_n, 2)
    if swap_ab:
        smem_cd = 16 * block_n * 2 * 2
    else:
        smem_cd = min(block_m, 128) * swizzle_cd * 2
    smem_barriers = 32 * 8 * 3 + 2 * 8 * 2 + 8
    smem_tmem_ptr = 4
    smem_per_stage = (
        load_block_m * block_k
        + load_block_n * block_k
        + _align(block_m, 128) * 4
        + _align(block_n, 128) * 4
    )
    smem_capacity = 232448
    num_stages = (smem_capacity - smem_cd - smem_barriers - smem_tmem_ptr) // smem_per_stage
    return min(num_stages, 32)


def _choose_deepgemm_config(M: int, N: int, K: int) -> tuple[bool, int, int, int, int, int]:
    """Match DeepGEMM's SM100 FP8 normal-GEMM layout heuristic (verbatim)."""

    sm_count = SM_NUMBER
    candidates: list[tuple] = []
    for swap_ab in (False, True):
        if swap_ab:
            block_m_candidates: range | list[int] = range(16, 257, 16)
            block_n_candidates: list[int] = [128]
            cluster_candidates = [(1, 2)]
        else:
            block_m_candidates = [32] if M <= 32 else [64] if M <= 64 else [128]
            max_block_n = 128 if K <= 256 else 256
            block_n_candidates = [16, *range(32, max_block_n + 1, 32)]
            cluster_candidates = [(2, 1)]

        for cluster_m, cluster_n in cluster_candidates:
            if sm_count % (cluster_m * cluster_n) != 0:
                continue
            for block_m in block_m_candidates:
                load_block_m = block_m // cluster_n
                if load_block_m % 8 != 0:
                    continue
                if math.ceil(M / block_m) % cluster_m != 0:
                    continue
                for block_n in block_n_candidates:
                    load_block_n = block_n // cluster_m
                    if load_block_n % 8 != 0:
                        continue
                    if math.ceil(N / block_n) % cluster_n != 0:
                        continue
                    sf_block_m = _align(block_m, 128)
                    sf_block_n = _align(block_n, 128)
                    umma_n = block_m if swap_ab else block_n
                    if 2 * umma_n + sf_block_m // 32 + sf_block_n // 32 > 512:
                        continue
                    num_blocks = math.ceil(M / block_m) * math.ceil(N / block_n)
                    waves = math.ceil(num_blocks / sm_count)
                    last_wave_util = num_blocks % sm_count or sm_count
                    stages = _deepgemm_num_stages(
                        swap_ab=swap_ab,
                        block_m=block_m,
                        block_n=block_n,
                        load_block_m=load_block_m,
                        load_block_n=load_block_n,
                    )
                    candidates.append(
                        (
                            0 if waves == 1 else 1,
                            -cluster_m * cluster_n,
                            waves,
                            -last_wave_util,
                            block_m + block_n,
                            block_m * block_n,
                            swap_ab,
                            block_m,
                            block_n,
                            stages,
                            cluster_m,
                            cluster_n,
                        )
                    )

    if not candidates:
        raise RuntimeError(f"no DeepGEMM config candidate for M={M}, N={N}, K={K}")
    _, _, _, _, _, _, swap_ab, block_m, block_n, stages, cluster_m, cluster_n = min(candidates)
    return swap_ab, block_m, block_n, stages, cluster_m, cluster_n


@dataclass(frozen=True, slots=True)
class Fp8BlockwiseGemmConfig:
    m: int = 1024
    n: int = 1024
    k: int = 1024
    # Pin the DeepGEMM layout instead of resolving it from (m, n, k).
    swap_ab: bool | None = None
    block_m: int | None = None
    block_n: int | None = None
    launch_shape: LaunchShape | None = None


def _resolve_layout(config: Fp8BlockwiseGemmConfig) -> tuple[bool, int, int, int, int, int]:
    pins = (config.swap_ab, config.block_m, config.block_n)
    if all(p is None for p in pins):
        return _choose_deepgemm_config(config.m, config.n, config.k)
    if any(p is None for p in pins):
        raise ValueError("fp8_blockwise_gemm swap_ab/block_m/block_n must be pinned together")
    swap_ab, block_m, block_n = config.swap_ab, config.block_m, config.block_n
    cluster_m, cluster_n = (1, 2) if swap_ab else (2, 1)
    # The same candidate constraints the heuristic enforces.
    if block_m // cluster_n % 8 != 0 or block_n // cluster_m % 8 != 0:
        raise ValueError("fp8_blockwise_gemm pinned blocks violate the load alignment")
    umma_n = block_m if swap_ab else block_n
    if 2 * umma_n + _align(block_m, 128) // 32 + _align(block_n, 128) // 32 > 512:
        raise ValueError("fp8_blockwise_gemm pinned blocks exceed the TMEM column budget")
    if math.ceil(config.m / block_m) % cluster_m != 0 or math.ceil(config.n / block_n) % cluster_n:
        raise ValueError("fp8_blockwise_gemm pinned blocks do not tile the problem per cluster")
    stages = _deepgemm_num_stages(
        swap_ab=swap_ab,
        block_m=block_m,
        block_n=block_n,
        load_block_m=block_m // cluster_n,
        load_block_n=block_n // cluster_m,
    )
    return swap_ab, block_m, block_n, stages, cluster_m, cluster_n


def fp8_blockwise_task_config(tasks: int) -> Fp8BlockwiseGemmConfig:
    """The canonical single-cluster examination setting, mirroring the fp16/bf16 GEMM one."""
    return Fp8BlockwiseGemmConfig(
        m=240, n=256 * tasks, k=16384, swap_ab=True, block_m=240, block_n=128, launch_shape=(2,)
    )


CONFIGS = [
    {"m": s, "n": s, "k": s, "label": f"{s}x{s}x{s}"} for s in [1024, 2048, 4096, 8192, 16384]
]

# Cheap shapes covering both datapaths and different block sizes / pipeline depths / k-tile counts.
FP8_CONFIGS_SUPPORTED = [
    {"m": 1024, "n": 1024, "k": 1024, "label": "1024x1024x1024"},  # non-swap (128, 64), 9 stages
    {"m": 2048, "n": 1024, "k": 1024, "label": "2048x1024x1024"},  # non-swap (128, 128), 7 stages
    {"m": 1024, "n": 2048, "k": 512, "label": "1024x2048x512"},  # non-swap, 1 SF pack
    {"m": 1920, "n": 512, "k": 1024, "label": "1920x512x1024"},  # swap_ab (64, 128), 10 stages
    {"m": 2400, "n": 512, "k": 512, "label": "2400x512x512"},  # swap_ab (80, 128): odd EPI count
    {"m": 2048, "n": 2048, "k": 2048, "label": "2048x2048x2048"},  # swap_ab (240, 128): partial M
    {"m": 2048, "n": 1056, "k": 512, "label": "2048x1056x512"},  # non-swap (128, 128): partial N
]


def build_fp8_blockwise_gemm(config: Fp8BlockwiseGemmConfig = Fp8BlockwiseGemmConfig()) -> Kernel:
    M, N, K = config.m, config.n, config.k
    swap_ab, dg_block_m, dg_block_n, smem_depth, log_m, log_n = _resolve_layout(config)
    cta_group = log_m * log_n
    if swap_ab:
        # The MMA computes the transposed C tile.
        mma_n = dg_block_m
        blk_m = dg_block_m // cta_group  # per-CTA A rows (M split across the pair)
        blk_n = dg_block_n  # per-CTA B rows (each CTA's own n tile)
        sched_rows = _ceil_div(N, dg_block_n)
        sched_cols = _ceil_div(M, dg_block_m)
        epi_tile = 16  # 16-row transposed C stores
    else:
        mma_n = dg_block_n
        blk_m = dg_block_m  # per-CTA A tile rows (each CTA owns its own m tile)
        blk_n = dg_block_n // cta_group  # per-CTA B rows (N split across the pair)
        sched_rows = _ceil_div(M, dg_block_m)
        sched_cols = _ceil_div(N, dg_block_n)
        epi_tile = EPI_TILE
    mma_m = 2 * (blk_n if swap_ab else blk_m)  # 256 on both datapaths
    blk_sfa = _align(dg_block_m, 128)
    blk_sfb = _align(dg_block_n, 128)
    sfa_cols = blk_sfa // 32  # packed-u32 TMEM columns per stage
    sfb_cols = blk_sfb // 32
    k_tiles = K // BLK_K
    sf_k_packs = k_tiles // SF_PACK
    total_work = sched_rows * sched_cols
    store_tiles = mma_n // epi_tile
    _validate_config(
        config,
        swap_ab=swap_ab,
        cta_group=cta_group,
        dg_block_m=dg_block_m,
        dg_block_n=dg_block_n,
        sched_rows=sched_rows,
        store_tiles=store_tiles,
        epi_tile=epi_tile,
        mma_m=mma_m,
        mma_n=mma_n,
    )
    launch_shape = config.launch_shape or (
        max(cta_group, min(SM_NUMBER, total_work) // cta_group * cta_group),
    )
    _validate_launch_shape(launch_shape, cta_group)
    pair_tasks = total_work // cta_group

    a_tile_bytes = blk_m * BLK_K
    b_tile_bytes = blk_n * BLK_K
    d_tile_bytes = (epi_tile * dg_block_n if swap_ab else blk_m * epi_tile) * 2
    sfa_tile_bytes = blk_sfa * U32_BYTES
    sfb_tile_bytes = blk_sfb * U32_BYTES
    a_off = 0
    b_off = a_off + smem_depth * a_tile_bytes
    sfa_off = b_off + smem_depth * b_tile_bytes
    sfb_off = sfa_off + smem_depth * sfa_tile_bytes
    d_off = sfb_off + smem_depth * sfb_tile_bytes
    data_end = d_off + TMEM_DEPTH * d_tile_bytes
    metadata_cursor = (data_end + 7) // 8 * 8
    smem_full_off = metadata_cursor
    metadata_cursor += smem_depth * 8
    smem_empty_off = metadata_cursor
    metadata_cursor += smem_depth * 8
    trans_done_off = metadata_cursor
    metadata_cursor += smem_depth * 8
    tmem_full_off = metadata_cursor
    metadata_cursor += TMEM_DEPTH * 8
    tmem_empty_off = metadata_cursor
    metadata_cursor += TMEM_DEPTH * 8
    tmem_addr_off = (metadata_cursor + 3) // 4 * 4
    smem_size_bytes = tmem_addr_off + 4

    k = IRBuilder(
        "nymph_fp8_blockwise_gemm",
        num_warps=8,  # WG_NUMBER=2: wg0 = tma/permute/mma warps, wg1 = epilogue
        smem_size_bytes=smem_size_bytes,
        launch_shape=launch_shape,
        cluster_shape=(cta_group,),
    )
    a_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.F8E4M3, shape=(M, K))
    b_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.F8E4M3, shape=(N, K))
    sfa_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.U32, shape=(sf_k_packs, M))
    sfb_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.U32, shape=(sf_k_packs, N))
    d_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.BF16, shape=(M, N))

    # Stage-major SMEM rings, indexed by a RUNTIME pipeline stage.
    a_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.F8E4M3,
        shape=(smem_depth, blk_m, BLK_K),
        layout=SmemSwizzleLayout(Swizzle.B128),
        byte_offset=a_off,
    )
    b_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.F8E4M3,
        shape=(smem_depth, blk_n, BLK_K),
        layout=SmemSwizzleLayout(Swizzle.B128),
        byte_offset=b_off,
    )
    sfa_smem = k.tensor(
        space=MemorySpace.SMEM, dtype=DType.U32, shape=(smem_depth, blk_sfa), byte_offset=sfa_off
    )
    sfb_smem = k.tensor(
        space=MemorySpace.SMEM, dtype=DType.U32, shape=(smem_depth, blk_sfb), byte_offset=sfb_off
    )
    # The TMA writes the source-order packed U32 tensors above.
    sfa_cp_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.U32,
        shape=(smem_depth, blk_sfa // 128, 32, 4),
        byte_offset=sfa_off,
    )
    sfb_cp_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.U32,
        shape=(smem_depth, blk_sfb // 128, 32, 4),
        byte_offset=sfb_off,
    )
    # One staged rank-3 buffer for both datapaths.
    d_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.BF16,
        shape=(TMEM_DEPTH, epi_tile, dg_block_n) if swap_ab else (TMEM_DEPTH, blk_m, epi_tile),
        byte_offset=d_off,
    )

    # TMEM: one 512-column allocation.
    accum = k.tmem_tensor(0)
    # SF bands: each 128-row super-block occupies four physical TMEM columns.
    sfa_tmem = k.tmem_tensor(TMEM_DEPTH * mma_n)
    sfb_tmem = k.tmem_tensor(TMEM_DEPTH * mma_n + TMEM_DEPTH * sfa_cols)

    accum_frag = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(TMEM_LD_SIZE,))
    out_frag = k.tensor(space=MemorySpace.REG, dtype=DType.BF16, shape=(TMEM_LD_SIZE,))
    sfa_perm_frag = k.tensor(space=MemorySpace.REG, dtype=DType.U32, shape=(blk_sfa // 32,))
    sfb_perm_frag = k.tensor(space=MemorySpace.REG, dtype=DType.U32, shape=(blk_sfb // 32,))

    # smem_pipe: full = the TMA engine.
    smem_full = k.mbar(kind=MBarKind.TMA, byte_offset=smem_full_off, stages=smem_depth)
    smem_empty = k.mbar(kind=MBarKind.TCGEN05, byte_offset=smem_empty_off, stages=smem_depth)
    # trans_done: both CTAs' permute warps arrive at the LEADER's barrier.
    trans_done = k.mbar(kind=MBarKind.THREAD, byte_offset=trans_done_off, stages=smem_depth)
    tmem_full = k.mbar(kind=MBarKind.TCGEN05, byte_offset=tmem_full_off, stages=TMEM_DEPTH)
    tmem_empty = k.mbar(kind=MBarKind.THREAD, byte_offset=tmem_empty_off, stages=TMEM_DEPTH)
    trans_done_leader = k.mbar_ref(trans_done, remote_coord=0)
    tmem_empty_leader = k.mbar_ref(tmem_empty, remote_coord=0)

    cta_id = k.cta_id()
    cta_rank = k.ctaid_in_cluster()
    task_space = k.task_space(grid=(pair_tasks,), fields=("pair_idx",))
    task_scheduler = k.scheduler(task_space)
    task_start = cta_id // cta_group
    task_step = k.launch_cta_count // cta_group

    ab_bytes = a_tile_bytes + b_tile_bytes
    sf_bytes = (dg_block_m + dg_block_n) * U32_BYTES

    def work_coords(work_idx):
        """ClusterPersistentScheduler2D group-major mapping (cluster_m=cluster_n=1)."""
        if sched_rows <= TILE_GROUPS_ROW_SIZE:
            row, col = work_idx % sched_rows, work_idx // sched_rows
        else:
            # sched_rows % TILE_GROUPS_ROW_SIZE == 0 (validated): full groups.
            group_span = TILE_GROUPS_ROW_SIZE * sched_cols
            group_id = work_idx // group_span
            within = work_idx % group_span
            row = group_id * TILE_GROUPS_ROW_SIZE + within % TILE_GROUPS_ROW_SIZE
            col = within // TILE_GROUPS_ROW_SIZE
        return (col, row) if swap_ab else (row, col)

    with k.if_warp(0):
        # tmem_alloc is warp-collective (full warp 0).
        k.tmem_alloc(0, N_COLS_TMEM, addr_byte_offset=tmem_addr_off, cta_group=cta_group)
        with k.if_elected():
            for s in range(smem_depth):
                k.mbarrier_init(smem_full, count=1, stage=s)
                k.mbarrier_init(smem_empty, count=1, stage=s)
                # One arrival per CTA: each CTA's permute warp arrives from a single elected thread.
                k.mbarrier_init(trans_done, count=cta_group, stage=s)
            for s in range(TMEM_DEPTH):
                k.mbarrier_init(tmem_full, count=1, stage=s)
                # One arrival per CTA: each CTA's epilogue leader thread.
                k.mbarrier_init(tmem_empty, count=cta_group, stage=s)
    # This sync IS the prologue ordering.
    k.cluster_sync()

    # ---- TMA producer.
    with k.if_warp(0), k.if_elected():
        with k.for_each_task(task_scheduler) as task:
            local_iter = (task.task_id - task_start) // task_step
            work_idx = task.task_id * cta_group + cta_rank
            m_idx, n_idx = work_coords(work_idx)
            # swap_ab: A is split by M halves across the pair.
            a_m = m_idx * dg_block_m + (cta_rank * blk_m if swap_ab else 0)
            b_n = n_idx * dg_block_n + (0 if swap_ab else cta_rank * blk_n)
            sf_m = m_idx * dg_block_m  # the FULL m band's scales, rank-independent
            sf_n = n_idx * dg_block_n
            for t in range(k_tiles):
                seq = local_iter * k_tiles + t
                stage = seq % smem_depth
                occ = seq // smem_depth
                k.mbarrier_wait(smem_empty, stage=stage, phase=(occ + 1) % 2)
                tx = ab_bytes + sf_bytes if t % SF_PACK == 0 else ab_bytes
                k.mbarrier_arrive_expect_tx(smem_full, bytes=tx, stage=stage)
                kc = t * BLK_K
                k.tma_load(
                    TensorSlice(tensor=a_smem, offsets=(stage, 0, 0), shape=(1, blk_m, BLK_K)),
                    a_gmem,
                    mbar=smem_full,
                    coords=(a_m, kc),
                    shape=(1, blk_m, BLK_K),
                    gmem_shape=(blk_m, BLK_K),
                    mbar_stage=stage,
                    cta_group=cta_group,
                )
                k.tma_load(
                    TensorSlice(tensor=b_smem, offsets=(stage, 0, 0), shape=(1, blk_n, BLK_K)),
                    b_gmem,
                    mbar=smem_full,
                    coords=(b_n, kc),
                    shape=(1, blk_n, BLK_K),
                    gmem_shape=(blk_n, BLK_K),
                    mbar_stage=stage,
                    cta_group=cta_group,
                )
                if t % SF_PACK == 0:
                    k.tma_load(
                        TensorSlice(tensor=sfa_smem, offsets=(stage, 0), shape=(1, dg_block_m)),
                        sfa_gmem,
                        mbar=smem_full,
                        coords=(t // SF_PACK, sf_m),
                        shape=(1, dg_block_m),
                        mbar_stage=stage,
                        cta_group=cta_group,
                    )
                    k.tma_load(
                        TensorSlice(tensor=sfb_smem, offsets=(stage, 0), shape=(1, dg_block_n)),
                        sfb_gmem,
                        mbar=smem_full,
                        coords=(t // SF_PACK, sf_n),
                        shape=(1, dg_block_n),
                        mbar_stage=stage,
                        cta_group=cta_group,
                    )

    # ---- scale-factor permute (TIRx wg0/warp2) ---- Whole-warp scope.
    with k.if_warp(2):
        # The TMA writes only the first DG_BLOCK rows of each (128-aligned) scale buffer.
        with k.if_elected():
            for s in range(smem_depth):
                for i in range(dg_block_m, blk_sfa):
                    k.store_scalar(TensorSlice(tensor=sfa_smem, offsets=(s, i), shape=(1, 1)), 0)
                for i in range(dg_block_n, blk_sfb):
                    k.store_scalar(TensorSlice(tensor=sfb_smem, offsets=(s, i), shape=(1, 1)), 0)
        # Each lane later reads the padding byte written for its physical row.
        k.warp_sync()
        with k.for_each_task(task_scheduler) as task:
            local_iter = (task.task_id - task_start) // task_step
            for t in range(k_tiles):
                seq = local_iter * k_tiles + t
                stage = seq % smem_depth
                occ = seq // smem_depth
                k.mbarrier_wait(smem_full, stage=stage, phase=occ % 2)
                if t % SF_PACK == 0:
                    lane = k.lane_id()

                    def load_scale(raw, frag, rows):
                        for r in range(rows // 32):
                            j = r ^ ((lane // 8) % 4)
                            m_super = j // 4
                            m_inner = j % 4
                            k.reg_load(
                                TensorSlice(tensor=frag, offsets=(r,), shape=(1,)),
                                TensorSlice(
                                    tensor=raw,
                                    offsets=(stage, m_super * 128 + m_inner * 32 + lane),
                                    shape=(1, 1),
                                ),
                            )

                    def store_scale(post, frag, rows):
                        for r in range(rows // 32):
                            j = r ^ ((lane // 8) % 4)
                            m_super = j // 4
                            m_inner = j % 4
                            k.reg_store(
                                TensorSlice(
                                    tensor=post,
                                    offsets=(stage, m_super, lane, m_inner),
                                    shape=(1, 1, 1, 1),
                                ),
                                TensorSlice(tensor=frag, offsets=(r,), shape=(1,)),
                            )

                    load_scale(sfa_smem, sfa_perm_frag, blk_sfa)
                    load_scale(sfb_smem, sfb_perm_frag, blk_sfb)
                    k.warp_sync()
                    store_scale(sfa_cp_smem, sfa_perm_frag, blk_sfa)
                    store_scale(sfb_cp_smem, sfb_perm_frag, blk_sfb)
                    k.warp_sync()
                    k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                # mbarrier.arrive is per-thread.
                with k.if_elected():
                    k.mbarrier_arrive(trans_done_leader, stage=stage)

    # ---- MMA (TIRx wg0/warp1, cluster leader only) ---- Single thread.
    with k.if_warp(1), k.if_elected():
        with k.for_each_task(task_scheduler) as task:
            local_iter = (task.task_id - task_start) // task_step
            with k.if_(cta_rank.eq(0)):
                tmem_idx = local_iter % TMEM_DEPTH
                k.mbarrier_wait(
                    tmem_empty, stage=tmem_idx, phase=(local_iter // TMEM_DEPTH + 1) % 2
                )
                acc_op = accum.at(0, tmem_idx * mma_n)
                sfa_op = sfa_tmem.at(0, tmem_idx * sfa_cols)
                sfb_op = sfb_tmem.at(0, tmem_idx * sfb_cols)
                for t in range(k_tiles):
                    seq = local_iter * k_tiles + t
                    stage = seq % smem_depth
                    occ = seq // smem_depth
                    k.mbarrier_wait(trans_done, stage=stage, phase=occ % 2)
                    if t % SF_PACK == 0:
                        for m_super in range(blk_sfa // 128):
                            k.tcgen05_cp(
                                sfa_tmem.at(0, tmem_idx * sfa_cols + 4 * m_super),
                                k.smem_tile(
                                    sfa_cp_smem,
                                    prefix_indices=(stage, m_super),
                                    row_offset=0,
                                    col_offset=0,
                                    rows=32,
                                    cols=4,
                                ),
                                shape="32x128b",
                                multicast="warp4",
                                cta_group=cta_group,
                            )
                        for m_super in range(blk_sfb // 128):
                            k.tcgen05_cp(
                                sfb_tmem.at(0, tmem_idx * sfb_cols + 4 * m_super),
                                k.smem_tile(
                                    sfb_cp_smem,
                                    prefix_indices=(stage, m_super),
                                    row_offset=0,
                                    col_offset=0,
                                    rows=32,
                                    cols=4,
                                ),
                                shape="32x128b",
                                multicast="warp4",
                                cta_group=cta_group,
                            )
                    for ki in range(K_ITERS):
                        ko = ki * MMA_K
                        a_op = k.smem_tile(
                            a_smem,
                            prefix_indices=(stage,),
                            row_offset=0,
                            col_offset=ko,
                            rows=blk_m,
                            cols=MMA_K,
                        )
                        b_op = k.smem_tile(
                            b_smem,
                            prefix_indices=(stage,),
                            row_offset=0,
                            col_offset=ko,
                            rows=blk_n,
                            cols=MMA_K,
                        )
                        # swap_ab computes the transposed C tile.
                        k.tcgen05_mma(
                            acc_op,
                            k.mma_a_smem(b_op if swap_ab else a_op),
                            a_op if swap_ab else b_op,
                            mma_m=mma_m,
                            mma_n=mma_n,
                            format="f8_e4m3",
                            block_scale=BlockScaleSpec(
                                sfa=sfb_op if swap_ab else sfa_op,
                                sfb=sfa_op if swap_ab else sfb_op,
                                sfa_k_offset=t % SF_PACK,
                                sfb_k_offset=t % SF_PACK,
                                scale_format="e8m0_fnu",
                                sf_per_mma=1,
                                sf_reuse=SF_PACK,
                            ),
                            accum=(t > 0 or ki > 0),
                            trans_a=False,
                            trans_b=False,
                            ws=False,
                            cta_group=cta_group,
                        )
                    # Release the SMEM stage in both CTAs after the k-tile is fully consumed.
                    k.tcgen05_commit(
                        smem_empty, stage=stage, cta_group=cta_group, multicast_cta_mask=0b11
                    )
                # Publish the accumulator stage to both CTAs' epilogues.
                k.tcgen05_commit(
                    tmem_full, stage=tmem_idx, cta_group=cta_group, multicast_cta_mask=0b11
                )

    # ---- epilogue (TIRx wg1) ---- Full-warpgroup scope for the warp-collective accumulator drains.
    with k.if_warpgroup(1):
        with k.for_each_task(task_scheduler) as task:
            local_iter = (task.task_id - task_start) // task_step
            work_idx = task.task_id * cta_group + cta_rank
            m_idx, n_idx = work_coords(work_idx)
            tmem_idx = local_iter % TMEM_DEPTH
            k.mbarrier_wait(tmem_full, stage=tmem_idx, phase=(local_iter // TMEM_DEPTH) % 2)
            for ot in range(store_tiles):
                store_iter = local_iter * store_tiles + ot
                # Pace the D_smem stage ring against in-flight TMA stores.
                with k.if_(store_iter >= TMEM_DEPTH):
                    with k.if_(k.tid_in_wg().eq(0)):
                        k.cp_async_bulk_wait_group_read(TMEM_DEPTH - 1)
                k.wg_sync(barrier_id=10)
                d_stage = store_iter % TMEM_DEPTH  # runtime: the EPI count may be odd
                if swap_ab:
                    # The accumulator holds C^T (rows = this CTA's n tile, cols = the m direction).
                    lane = k.lane_id()
                    tid = k.tid_in_wg()
                    for atom_m in range(2):
                        col = tmem_idx * mma_n + ot * epi_tile + atom_m * TMEM_LD_SIZE
                        k.tcgen05_ld(
                            TensorSlice(tensor=accum_frag, offsets=(0,), shape=(4,)),
                            accum.at(0, col),
                            shape="16x256b",
                            num=1,
                        )
                        k.tcgen05_ld(
                            TensorSlice(tensor=accum_frag, offsets=(4,), shape=(4,)),
                            accum.at(16, col),
                            shape="16x256b",
                            num=1,
                        )
                        k.tcgen05_wait_ld()
                        k.reg_cvt(out_frag, accum_frag)
                        k.stmatrix(
                            TensorSlice(
                                tensor=d_smem,
                                offsets=(
                                    d_stage,
                                    atom_m * 8 + lane % 8,
                                    (tid // 32) * 32 + (lane // 8) * 8,
                                ),
                                shape=(1, 1, 8),
                            ),
                            out_frag,
                            num=4,
                            trans=True,
                        )
                else:
                    for ki in range(epi_tile // TMEM_LD_SIZE):
                        col = tmem_idx * mma_n + ot * epi_tile + ki * TMEM_LD_SIZE
                        k.tcgen05_ld(accum_frag, accum.at(0, col), num=TMEM_LD_SIZE)
                        k.tcgen05_wait_ld()
                        k.reg_cvt(out_frag, accum_frag)
                        k.reg_store(
                            TensorSlice(
                                tensor=d_smem,
                                offsets=(d_stage, k.tid_in_wg(), ki * TMEM_LD_SIZE),
                                shape=(1, 1, TMEM_LD_SIZE),
                            ),
                            out_frag,
                        )
                # Synchronize all accumulator drains and d_smem writes before publishing the store.
                k.wg_sync(barrier_id=10)
                with k.if_(k.tid_in_wg().eq(0)):
                    if ot == store_tiles - 1:
                        # The accumulator stage is fully drained.
                        k.mbarrier_arrive(tmem_empty_leader, stage=tmem_idx)
                    k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                    if swap_ab:
                        k.tma_store(
                            d_gmem,
                            TensorSlice(
                                tensor=d_smem,
                                offsets=(d_stage, 0, 0),
                                shape=(1, epi_tile, dg_block_n),
                            ),
                            coords=(m_idx * dg_block_m + ot * epi_tile, n_idx * dg_block_n),
                            shape=(1, epi_tile, dg_block_n),
                            gmem_shape=(epi_tile, dg_block_n),
                        )
                    else:
                        k.tma_store(
                            d_gmem,
                            TensorSlice(
                                tensor=d_smem, offsets=(d_stage, 0, 0), shape=(1, blk_m, epi_tile)
                            ),
                            coords=(m_idx * dg_block_m, n_idx * dg_block_n + ot * epi_tile),
                            shape=(1, blk_m, epi_tile),
                            gmem_shape=(blk_m, epi_tile),
                        )
                    k.cp_async_bulk_commit_group()
        # Tail drain: only the committing stream may wait its bulk groups.
        with k.if_(k.tid_in_wg().eq(0)):
            k.cp_async_bulk_wait_group_read(0)
        k.wg_sync(barrier_id=10)

    # Teardown: every stream's pipeline work happens-before the dealloc.
    k.cluster_sync()
    with k.if_warp(0):
        k.tmem_relinquish(cta_group)
        k.tmem_dealloc(0, N_COLS_TMEM, cta_group)

    return k.build()


def _validate_config(
    config: Fp8BlockwiseGemmConfig,
    *,
    swap_ab: bool,
    cta_group: int,
    dg_block_m: int,
    dg_block_n: int,
    sched_rows: int,
    store_tiles: int,
    epi_tile: int,
    mma_m: int,
    mma_n: int,
) -> None:
    for name in ("m", "n", "k"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"fp8_blockwise_gemm {name} must be a positive integer")
    if cta_group != 2:
        raise ValueError("fp8_blockwise_gemm nymph port models the cta_group=2 cluster datapath")
    if swap_ab and dg_block_n != 128:
        raise ValueError("fp8_blockwise_gemm swap_ab requires the DeepGEMM block_n=128 grid")
    # Partial M/N tiles are first-class.
    if config.k % (BLK_K * SF_PACK) != 0:
        raise ValueError("fp8_blockwise_gemm k must be a multiple of 512 (4 packed k-tiles)")
    if sched_rows % cta_group != 0:
        raise ValueError("fp8_blockwise_gemm requires an even number of tile rows per cluster pair")
    if sched_rows > TILE_GROUPS_ROW_SIZE and sched_rows % TILE_GROUPS_ROW_SIZE != 0:
        raise ValueError("fp8_blockwise_gemm nymph port supports tail-only or full-group tilings")
    if not swap_ab and mma_n % epi_tile != 0:
        raise ValueError("fp8_blockwise_gemm MMA_N must be a multiple of EPI_TILE")
    if mma_m != 256 or mma_n % (16 if swap_ab else 32) != 0 or mma_n > 256:
        raise ValueError("fp8_blockwise_gemm resolved an unsupported MMA shape")


def _validate_launch_shape(launch_shape: LaunchShape, cta_group: int) -> None:
    if not isinstance(launch_shape, tuple) or len(launch_shape) != 1:
        raise ValueError("fp8_blockwise_gemm requires a 1D launch_shape")
    count = launch_shape[0]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("fp8_blockwise_gemm launch_shape[0] must be a positive integer")
    if count % cta_group != 0:
        raise ValueError("fp8_blockwise_gemm launch_shape[0] must be divisible by cta_group")
