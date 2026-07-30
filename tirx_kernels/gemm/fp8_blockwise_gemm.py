from __future__ import annotations

import math
import os

import torch
from deep_gemm.utils.math import per_block_cast_to_fp8, per_token_cast_to_fp8

from tvm.backend.cuda.operator.tile_primitive.tma_utils import SwizzleMode
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.bench import bench
from tvm.tirx.lang.pipeline import MBarrier, PipelineState, TCGen05Bar, TMABar
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D


def _align(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _swizzle_mode(block_size: int, elem_size: int) -> int:
    for mode in (128, 64, 32, 16):
        if (block_size * elem_size) % mode == 0:
            return mode
    raise AssertionError("unreachable swizzle mode")


def _tma_swizzle_mode(swizzle_bytes: int) -> SwizzleMode:
    return {
        128: SwizzleMode.SWIZZLE_128B_ATOM,
        64: SwizzleMode.SWIZZLE_64B_ATOM,
        32: SwizzleMode.SWIZZLE_32B_ATOM,
        16: SwizzleMode.SWIZZLE_NONE,
    }[swizzle_bytes]


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
    """Match DeepGEMM's SM100 FP8 normal-GEMM layout heuristic."""

    sm_count = 148
    candidates: list[tuple[int, int, int, int, int, int, bool, int, int, int, int, int]] = []
    for swap_ab in (False, True):
        if swap_ab:
            block_m_candidates = range(16, 257, 16)
            block_n_candidates = [128]
            cluster_candidates = [(1, 2)]
        else:
            block_m_candidates = [32] if M <= 32 else [64] if M <= 64 else [128]
            max_block_n = 128 if K <= 256 else 256
            block_n_candidates = (
                [16, *range(32, max_block_n + 1, 32)]
                if K <= 512 or M == N
                else range(16, max_block_n + 1, 16)
            )
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


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _dg_scale_view(tirx_scale_pack: torch.Tensor, mn: int) -> torch.Tensor:
    packed_k, physical_mn = tirx_scale_pack.shape
    if physical_mn != mn:
        raise ValueError(
            f"packed scale shape mismatch: expected physical MN {mn}, got {physical_mn}"
        )
    scale_i32 = tirx_scale_pack.view(torch.int32)
    return torch.as_strided(scale_i32, size=(mn, packed_k), stride=(1, physical_mn))


def prepare_data(M: int, N: int, K: int):
    A_origin = torch.randn((M, K), dtype=torch.float32)
    B_origin = torch.randn((N, K), dtype=torch.float32)
    A_fp8, sfa = per_token_cast_to_fp8(A_origin, use_ue8m0=True)
    B_fp8, sfb = per_block_cast_to_fp8(B_origin, use_ue8m0=True)
    sfa_uint8 = (sfa.view(torch.int32) >> 23).to(torch.uint8).contiguous()
    sfb_uint8 = (sfb.view(torch.int32) >> 23).to(torch.uint8).contiguous().repeat(128, 1)[:N, :]
    sfa_pack = sfa_uint8.view(torch.uint32).T.contiguous()
    sfb_pack = sfb_uint8.view(torch.uint32).T.contiguous()
    A_fp8_de = A_fp8.to(torch.float32)
    B_fp8_de = B_fp8.to(torch.float32)
    A_de = (
        A_fp8_de.reshape(M, K // 128, 128) * 2.0 ** (sfa_uint8[:, :, None].to(torch.float32) - 127)
    ).reshape(M, K)
    B_de = (
        B_fp8_de.reshape(N, K // 128, 128) * 2.0 ** (sfb_uint8[:, :, None].to(torch.float32) - 127)
    ).reshape(N, K)
    C_ref = torch.matmul(A_de, B_de.T).to(torch.bfloat16)
    return (
        A_fp8.to("cuda"),
        B_fp8.to("cuda"),
        sfa.to("cuda"),
        sfb.to("cuda"),
        sfa_pack.to("cuda"),
        sfb_pack.to("cuda"),
        C_ref.to("cuda"),
        A_origin.to("cuda"),
        B_origin.to("cuda"),
    )


def _cluster_sync_relaxed():
    T.evaluate(T.ptx.barrier.cluster.arrive(sem="relaxed", aligned=True))
    T.evaluate(T.ptx.barrier.cluster.wait(aligned=True))


def _runtime_instr_desc_with_sf_id(desc, sfa_id, sfb_id):
    runtime_desc = T.bitwise_and(desc, T.uint32(0x9FFFFFCF))
    runtime_desc = T.bitwise_or(runtime_desc, T.shift_left(T.cast(sfa_id, "uint32"), T.uint32(29)))
    return T.bitwise_or(runtime_desc, T.shift_left(T.cast(sfb_id, "uint32"), T.uint32(4)))


def _advance_umma_desc_lo(desc, base_lo, k_offset):
    return T.bitwise_or(
        T.bitwise_and(desc, T.shift_left(T.uint64(0xFFFFFFFF), T.uint64(32))),
        T.cast(base_lo + T.cast(k_offset // 16, "uint32"), "uint64"),
    )


def _transpose_sf_chunk(buf, stage, chunk, lane):
    """Match DeepGEMM's 128-element UTCCP shared-memory transpose."""

    base = chunk * 128
    values = T.alloc_local((4,), "uint32")
    T.buffer_store(
        values, T.ptx.ld(buf.ptr_to([stage, base + lane]), "uint32", "u32", space="shared"), [0]
    )
    T.buffer_store(
        values,
        T.ptx.ld(buf.ptr_to([stage, base + 32 + lane]), "uint32", "u32", space="shared"),
        [1],
    )
    T.buffer_store(
        values,
        T.ptx.ld(buf.ptr_to([stage, base + 64 + lane]), "uint32", "u32", space="shared"),
        [2],
    )
    T.buffer_store(
        values,
        T.ptx.ld(buf.ptr_to([stage, base + 96 + lane]), "uint32", "u32", space="shared"),
        [3],
    )
    T.evaluate(T.cuda.warp_sync())
    T.evaluate(
        T.ptx.st(
            buf.ptr_to([stage, base + lane * 4]),
            values[0],
            values[1],
            values[2],
            values[3],
            vec="v4",
            ptx_type="u32",
            space="shared",
        )
    )


@T.jit
def _kernel(
    A: T.Buffer((M, K), "float8_e4m3fn"),
    B: T.Buffer((N, K), "float8_e4m3fn"),
    SFA: T.Buffer((math.ceil(K / 128) // 4, M), "uint32"),
    SFB: T.Buffer((math.ceil(K / 128) // 4, N), "uint32"),
    D: T.Buffer((M, N), "bfloat16"),
    *,
    # problem size
    M: T.constexpr,
    N: T.constexpr,
    K: T.constexpr,
    # block + cluster layout
    SWAP_AB: T.constexpr,
    DG_BLOCK_M: T.constexpr,
    DG_BLOCK_N: T.constexpr,
    LOGICAL_M_CLUSTER: T.constexpr,
    LOGICAL_N_CLUSTER: T.constexpr,
    # tile / MMA sizes
    BLK_K: T.constexpr = 128,
    MMA_K: T.constexpr = 32,
    TMEM_LD_SIZE: T.constexpr = 8,
    # pipeline depths
    SMEM_DEPTH: T.constexpr,
    TMEM_DEPTH: T.constexpr = 2,
    # warp / SM / scheduler
    WG_NUMBER: T.constexpr = 2,
    SM_NUMBER: T.constexpr = 148,
    TILE_GROUPS_ROW_SIZE: T.constexpr = 16,
):
    CTA_GROUP = T.meta_var(LOGICAL_M_CLUSTER * LOGICAL_N_CLUSTER)
    M_CLUSTER = T.meta_var(CTA_GROUP)
    N_CLUSTER = T.meta_var(1)
    MMA_N = T.meta_var(DG_BLOCK_M if SWAP_AB else DG_BLOCK_N)
    BLK_M = T.meta_var(DG_BLOCK_M // LOGICAL_N_CLUSTER if SWAP_AB else DG_BLOCK_M)
    BLK_N = T.meta_var(DG_BLOCK_N if SWAP_AB else DG_BLOCK_N // LOGICAL_M_CLUSTER)
    BLK_SFA = T.meta_var(_align(DG_BLOCK_M, 128))
    BLK_SFB = T.meta_var(_align(DG_BLOCK_N, 128))
    K_TILES = T.meta_var(K // BLK_K)
    SFA_post_layout = T.meta_var(
        T.TileLayout(T.S[(SMEM_DEPTH, BLK_SFA // 128, 4, 32) : (BLK_SFA, 128, 1, 4)])
    )
    SFB_post_layout = T.meta_var(
        T.TileLayout(T.S[(SMEM_DEPTH, BLK_SFB // 128, 4, 32) : (BLK_SFB, 128, 1, 4)])
    )
    K_ITERS = T.meta_var(BLK_K // MMA_K)
    SFA_smem_fp8_layout = T.meta_var(SFA_post_layout.unpack(4).broadcast(K_ITERS))
    SFB_smem_fp8_layout = T.meta_var(SFB_post_layout.unpack(4).broadcast(K_ITERS))
    AB_bytes = T.meta_var(BLK_M * BLK_K + BLK_N * BLK_K)  # fp8 A+B operands: 1 byte/elem
    SFAB_bytes = T.meta_var((DG_BLOCK_M + DG_BLOCK_N) * 4)  # SF packed as uint32: 4 B
    SCHED_M_NUM = T.meta_var(math.ceil(N / DG_BLOCK_N) if SWAP_AB else math.ceil(M / DG_BLOCK_M))
    SCHED_N_NUM = T.meta_var(math.ceil(M / DG_BLOCK_M) if SWAP_AB else math.ceil(N / DG_BLOCK_N))
    D_SWIZZLE_BYTES = T.meta_var(_swizzle_mode(DG_BLOCK_N, 2))
    D_SMEM_M = T.meta_var(16 if SWAP_AB else min(DG_BLOCK_M, 128))
    D_SMEM_N = T.meta_var(DG_BLOCK_N if SWAP_AB else D_SWIZZLE_BYTES // 2)
    D_SWIZZLE = T.meta_var(_tma_swizzle_mode(D_SWIZZLE_BYTES))
    SMEM_CD_BYTES = T.meta_var(TMEM_DEPTH * D_SMEM_M * D_SMEM_N * 2)
    SMEM_BARRIER_BYTES = T.meta_var(32 * 8 * 3 + 2 * 8 * 2 + 8)
    SMEM_STAGE_BYTES = T.meta_var(AB_bytes + (BLK_SFA + BLK_SFB) * 4)
    DG_SMEM_BYTES = T.meta_var(
        SMEM_CD_BYTES + SMEM_BARRIER_BYTES + 4 + SMEM_DEPTH * SMEM_STAGE_BYTES
    )
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    # DeepGEMM uses relaxed cluster arrival at all three two-CTA sync points.
    _cluster_sync_relaxed()
    cbx, cby = T.cta_id_in_cluster([M_CLUSTER, N_CLUSTER])
    cluster_rank = T.ptx.fetch_register(32, "cluster_ctarank")
    bx = T.cta_id([SM_NUMBER])
    wg_id = T.warpgroup_id([WG_NUMBER])
    warp_id = T.warp_id_in_wg([4])
    tid_in_wg = T.thread_id_in_wg([128])
    lane_id = T.lane_id([32])
    pool = T.SMEMPool()
    # Preserve DeepGEMM's dynamic-SMEM order: D, A, B, SFA, SFB, barriers,
    # then the shared TMEM pointer.
    D_smem = pool.alloc_tcgen05_mma_AB(
        (TMEM_DEPTH, D_SMEM_M, D_SMEM_N), "bfloat16", swizzle_mode=D_SWIZZLE
    )
    A_smem = pool.alloc_tcgen05_mma_AB((SMEM_DEPTH, BLK_M, BLK_K), "float8_e4m3fn")
    B_smem = pool.alloc_tcgen05_mma_AB((SMEM_DEPTH, BLK_N, BLK_K), "float8_e4m3fn")
    SFA_smem = pool.alloc((SMEM_DEPTH, BLK_SFA), "uint32")
    SFB_smem = pool.alloc((SMEM_DEPTH, BLK_SFB), "uint32")
    full_barriers = TMABar(pool, SMEM_DEPTH, leader=False)
    empty_barriers = TCGen05Bar(pool, SMEM_DEPTH, leader=False)
    with_sf_full_barriers = MBarrier(pool, SMEM_DEPTH, leader=False)
    tmem_full_barriers = TCGen05Bar(pool, TMEM_DEPTH, leader=False)
    tmem_empty_barriers = MBarrier(pool, TMEM_DEPTH, phase_offset=1, leader=False)
    tmem_pool = T.TMEMPool(
        pool,
        total_cols=512,
        cta_group=CTA_GROUP,
        alloc_warp=2,
        dealloc_warp=0,
        sync_after_alloc=False,
    )
    acc_buf = tmem_pool.alloc_tcgen05_mma_D(
        (128, TMEM_DEPTH * MMA_N), "float32", M=128 * CTA_GROUP, cta_group=CTA_GROUP
    )
    acc = T.meta_var(acc_buf.rearrange("m (s n) -> s m n", s=TMEM_DEPTH))
    SFA_tmem = tmem_pool.alloc_sf(
        (BLK_SFA, 4 * K_ITERS), "float8_e8m0fnu", sf_per_mma=1, sf_reuse=K_ITERS
    )
    SFB_tmem = tmem_pool.alloc_sf(
        (BLK_SFB, 4 * K_ITERS), "float8_e8m0fnu", sf_per_mma=1, sf_reuse=K_ITERS
    )
    pool.commit(size=DG_SMEM_BYTES)
    if wg_id == 0:
        if warp_id == 1:
            if T.ptx.elect_sync():
                for i in T.unroll(SMEM_DEPTH):
                    T.ptx.mbarrier.init(full_barriers.ptr_to([i]), 1)
                    T.ptx.mbarrier.init(empty_barriers.ptr_to([i]), 1)
                    T.ptx.mbarrier.init(with_sf_full_barriers.ptr_to([i]), CTA_GROUP * 32)
                for i in T.unroll(TMEM_DEPTH):
                    T.ptx.mbarrier.init(tmem_full_barriers.ptr_to([i]), 1)
                    T.ptx.mbarrier.init(tmem_empty_barriers.ptr_to([i]), CTA_GROUP * 128)
                T.ptx.fence.mbarrier_init()
        elif warp_id == 2:
            tmem_pool.commit()
    _cluster_sync_relaxed()
    T.evaluate(T.ptx.griddepcontrol.wait())

    stage: T.int32
    tile_scheduler = ClusterPersistentScheduler2D(
        "tile_scheduler",
        num_m_tiles=SCHED_M_NUM,
        num_n_tiles=SCHED_N_NUM,
        l2_group_size=TILE_GROUPS_ROW_SIZE,
        num_clusters=SM_NUMBER,
    )
    m_idx = T.meta_var(tile_scheduler.n_idx if SWAP_AB else tile_scheduler.m_idx)
    n_idx = T.meta_var(tile_scheduler.m_idx if SWAP_AB else tile_scheduler.n_idx)

    if wg_id == 0:
        if warp_id == 0:
            tma_cur = PipelineState(SMEM_DEPTH, 0)
            a_m = T.meta_var(
                m_idx * DG_BLOCK_M + cluster_rank * BLK_M if SWAP_AB else m_idx * DG_BLOCK_M
            )
            sf_m = T.meta_var(m_idx * DG_BLOCK_M)
            b_n = T.meta_var(
                n_idx * DG_BLOCK_N if SWAP_AB else n_idx * DG_BLOCK_N + cluster_rank * BLK_N
            )
            sf_n = T.meta_var(n_idx * DG_BLOCK_N)

            @T.inline
            def tma_load(k_tile):
                empty_barriers.wait(tma_cur.stage, tma_cur.phase ^ 1)
                stage = tma_cur.stage
                k = T.meta_var(k_tile * BLK_K)
                tma_copy = T.meta_var(
                    {
                        "dispatch": "tma_auto",
                        "mbar": full_barriers.ptr_to([stage]),
                        "cta_group": 1,
                        "cache_hint": "evict_normal",
                        "prefetch_tensormap": True,
                    }
                )
                Tx.copy_async(A_smem[stage], A[a_m : a_m + BLK_M, k : k + BLK_K], **tma_copy)
                Tx.copy_async(B_smem[stage], B[b_n : b_n + BLK_N, k : k + BLK_K], **tma_copy)
                if k_tile % 4 == 0:
                    Tx.copy_async(
                        SFA_smem[stage, 0:DG_BLOCK_M],
                        SFA[k_tile // 4, sf_m : sf_m + DG_BLOCK_M],
                        **tma_copy,
                    )
                    Tx.copy_async(
                        SFB_smem[stage, 0:DG_BLOCK_N],
                        SFB[k_tile // 4, sf_n : sf_n + DG_BLOCK_N],
                        **tma_copy,
                    )
                full_barriers.arrive(
                    tma_cur.stage,
                    tx_count=T.if_then_else(k_tile % 4 == 0, AB_bytes + SFAB_bytes, AB_bytes),
                )

            @T.inline
            def tma_iter():
                for k_tile in T.serial(K_TILES):
                    tma_load(k_tile)
                    tma_cur.advance()

            if T.ptx.elect_sync():
                tile_scheduler.linear_idx = bx
                tile_scheduler.tile_count = 0
                while tile_scheduler.valid():
                    tile_scheduler.update_current_m_n_idx(tile_scheduler.linear_idx)
                    tma_iter()
                    tile_scheduler.linear_idx = tile_scheduler.linear_idx + SM_NUMBER
                    tile_scheduler.tile_count = tile_scheduler.tile_count + 1
        elif warp_id == 1 and cluster_rank == 0:
            SFA_smem_fp8 = SFA_smem.view("float8_e8m0fnu").view(
                SMEM_DEPTH, BLK_SFA, 4 * K_ITERS, layout=SFA_smem_fp8_layout
            )
            SFB_smem_fp8 = SFB_smem.view("float8_e8m0fnu").view(
                SMEM_DEPTH, BLK_SFB, 4 * K_ITERS, layout=SFB_smem_fp8_layout
            )
            desc_a: T.uint64
            desc_b: T.uint64
            desc_i: T.uint32
            runtime_desc_i: T.uint32
            a_desc_lo: T.uint32
            b_desc_lo: T.uint32
            a_desc_base_lo: T.uint32
            b_desc_base_lo: T.uint32
            tmem_idx: T.int32
            tmem_phase: T.int32
            mma_state = PipelineState(SMEM_DEPTH, 0)

            T.ptx.tcgen05.encode_instr_descriptor_block_scaled(
                T.address_of(desc_i),
                d_dtype="float32",
                a_dtype="float8_e4m3fn",
                b_dtype="float8_e4m3fn",
                sfa_dtype="float8_e8m0fnu",
                sfb_dtype="float8_e8m0fnu",
                sfa_tmem_addr=SFA_tmem.allocated_addr[0],
                sfb_tmem_addr=SFB_tmem.allocated_addr[0],
                M=128 * CTA_GROUP,
                N=MMA_N,
                K=MMA_K,
                trans_a=False,
                trans_b=False,
                n_cta_groups=CTA_GROUP,
            )
            T.ptx.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_a), A_smem.ptr_to([0, 0, 0]), ldo=0, sdo=64, swizzle=3
            )
            T.ptx.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_b), B_smem.ptr_to([0, 0, 0]), ldo=0, sdo=64, swizzle=3
            )
            a_desc_lo = T.Select(
                lane_id < SMEM_DEPTH,
                T.cast(T.bitwise_and(desc_a, T.uint64(0xFFFFFFFF)), "uint32")
                + T.cast(lane_id * (BLK_M * BLK_K // 16), "uint32"),
                T.uint32(0),
            )
            b_desc_lo = T.Select(
                lane_id < SMEM_DEPTH,
                T.cast(T.bitwise_and(desc_b, T.uint64(0xFFFFFFFF)), "uint32")
                + T.cast(lane_id * (BLK_N * BLK_K // 16), "uint32"),
                T.uint32(0),
            )
            tile_scheduler.linear_idx = bx
            tile_scheduler.tile_count = 0
            while tile_scheduler.valid():
                tile_scheduler.update_current_m_n_idx(tile_scheduler.linear_idx)
                tmem_idx = tile_scheduler.tile_idx % TMEM_DEPTH
                tmem_phase = tile_scheduler.tile_idx // TMEM_DEPTH & 1
                tmem_empty_barriers.wait(tmem_idx, tmem_phase)
                T.ptx.tcgen05.fence.after_thread_sync()
                for k_tile in T.serial(K_TILES, unroll=4):
                    ks = mma_state.stage
                    with_sf_full_barriers.wait(ks, mma_state.phase)
                    T.ptx.tcgen05.fence.after_thread_sync()
                    a_desc_base_lo = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), a_desc_lo, ks, 32, 32)
                    b_desc_base_lo = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), b_desc_lo, ks, 32, 32)
                    if T.ptx.elect_sync():
                        if k_tile % 4 == 0:
                            Tx.copy_async(SFA_tmem, SFA_smem_fp8[ks], cta_group=CTA_GROUP)
                            Tx.copy_async(SFB_tmem, SFB_smem_fp8[ks], cta_group=CTA_GROUP)
                        for ki in T.unroll(K_ITERS):
                            runtime_desc_i = _runtime_instr_desc_with_sf_id(
                                desc_i, k_tile % 4, k_tile % 4
                            )
                            desc_a = _advance_umma_desc_lo(desc_a, a_desc_base_lo, ki * MMA_K)
                            desc_b = _advance_umma_desc_lo(desc_b, b_desc_base_lo, ki * MMA_K)
                            if SWAP_AB:
                                T.ptx.tcgen05.mma.block_scale(
                                    tmem_idx * MMA_N,
                                    desc_b,
                                    desc_a,
                                    SFB_tmem.allocated_addr[0],
                                    SFA_tmem.allocated_addr[0],
                                    runtime_desc_i,
                                    d_dtype="float32",
                                    a_dtype="float8_e4m3fn",
                                    b_dtype="float8_e4m3fn",
                                    sfa_dtype="float8_e8m0fnu",
                                    sfb_dtype="float8_e8m0fnu",
                                    use_a_tmem=False,
                                    cta_group=CTA_GROUP,
                                    enable_input_d=T.Or(k_tile > 0, ki > 0),
                                )
                            else:
                                T.ptx.tcgen05.mma.block_scale(
                                    tmem_idx * MMA_N,
                                    desc_a,
                                    desc_b,
                                    SFA_tmem.allocated_addr[0],
                                    SFB_tmem.allocated_addr[0],
                                    runtime_desc_i,
                                    d_dtype="float32",
                                    a_dtype="float8_e4m3fn",
                                    b_dtype="float8_e4m3fn",
                                    sfa_dtype="float8_e8m0fnu",
                                    sfb_dtype="float8_e8m0fnu",
                                    use_a_tmem=False,
                                    cta_group=CTA_GROUP,
                                    enable_input_d=T.Or(k_tile > 0, ki > 0),
                                )
                    T.cuda.warp_sync()
                    if T.ptx.elect_sync():
                        empty_barriers.arrive(ks, cta_group=CTA_GROUP, cta_mask=3)
                    if k_tile == K_TILES - 1:
                        if T.ptx.elect_sync():
                            tmem_full_barriers.arrive(tmem_idx, cta_group=CTA_GROUP, cta_mask=3)
                    T.cuda.warp_sync()
                    mma_state.advance()
                tile_scheduler.linear_idx = tile_scheduler.linear_idx + SM_NUMBER
                tile_scheduler.tile_count = tile_scheduler.tile_count + 1

            final_iter = tile_scheduler.tile_idx - 1
            if final_iter >= 0:
                final_phase = final_iter // TMEM_DEPTH & 1
                tmem_empty_barriers.wait(final_iter % TMEM_DEPTH, final_phase ^ 1)
        elif warp_id == 2:
            trans_state = PipelineState(SMEM_DEPTH, 0)

            @T.inline
            def transpose(ks, k_tile):
                full_barriers.wait(ks, trans_state.phase)
                if k_tile % 4 == 0:
                    for chunk in T.unroll(BLK_SFA // 128):
                        _transpose_sf_chunk(SFA_smem, ks, chunk, lane_id)
                    T.ptx.fence.proxy_async("shared::cta")
                if k_tile % 4 == 0:
                    for chunk in T.unroll(BLK_SFB // 128):
                        _transpose_sf_chunk(SFB_smem, ks, chunk, lane_id)
                    T.ptx.fence.proxy_async("shared::cta")
                with_sf_full_barriers.arrive(ks, remote=0)

            @T.inline
            def trans_iter():
                for k_tile in T.serial(K_TILES):
                    transpose(trans_state.stage, k_tile)
                    trans_state.advance()

            tile_scheduler.linear_idx = bx
            tile_scheduler.tile_count = 0
            while tile_scheduler.valid():
                tile_scheduler.update_current_m_n_idx(tile_scheduler.linear_idx)
                trans_iter()
                tile_scheduler.linear_idx = tile_scheduler.linear_idx + SM_NUMBER
                tile_scheduler.tile_count = tile_scheduler.tile_count + 1
    elif wg_id == 1:
        tmem_idx: T.int32
        tmem_phase: T.int32
        tma_stage_idx: T.int32
        store_iter: T.int32

        # Stream acc -> D_smem -> TMA in EPI-wide slices. SWAP_AB only changes the
        # acc -> D_smem step (stmatrix transpose vs straight copy) and the tiling.
        EPI = T.meta_var(16 if SWAP_AB else D_SMEM_N)
        STORE_TILES = T.meta_var(MMA_N // EPI)
        D_TILE_M = T.meta_var(16 if SWAP_AB else DG_BLOCK_M)
        D_TILE_N = T.meta_var(DG_BLOCK_N if SWAP_AB else D_SMEM_N)

        @T.inline
        def epilogue():
            swap_frag = T.alloc_tcgen05_ldst_frag("16x256b", (128, 8), "float32")
            swap_bf16 = T.alloc_cast_frag(swap_frag, "bfloat16")
            for ot in T.unroll(STORE_TILES):
                stage = tma_stage_idx
                if store_iter >= TMEM_DEPTH:
                    if warp_id == 0:
                        T.ptx.cp_async.bulk.wait_group(TMEM_DEPTH - 1)
                    T.cuda.warpgroup_sync(8)
                if SWAP_AB:
                    for atom_m in T.unroll(2):
                        col_st: T.let = ot * 16 + atom_m * 8
                        Tx.wg.copy_async(swap_frag[:, :], acc[tmem_idx, :, col_st : col_st + 8])
                        T.ptx.tcgen05.wait.ld()
                        Tx.wg.cast(swap_bf16, swap_frag)
                        rs = T.meta_var(atom_m * 8)
                        Tx.wg.copy(
                            D_smem[stage, rs : rs + 8, 0:128],
                            swap_bf16.permute(1, 0),
                            dispatch="ldstmatrix",
                        )
                else:
                    for ki in T.unroll(D_SMEM_N // TMEM_LD_SIZE):
                        Dreg = T.wg_reg_tile(TMEM_LD_SIZE)
                        acc_n = T.meta_var(ot * D_SMEM_N + ki * TMEM_LD_SIZE)
                        Tx.wg.copy_async(Dreg, acc[tmem_idx, :, acc_n : acc_n + TMEM_LD_SIZE])
                        T.ptx.tcgen05.wait.ld()
                        Dreg_bf16 = T.wg_reg_tile(TMEM_LD_SIZE, dtype="bfloat16")
                        Tx.wg.cast(Dreg_bf16, Dreg)
                        Tx.wg.copy(
                            D_smem[stage, :, ki * TMEM_LD_SIZE : (ki + 1) * TMEM_LD_SIZE], Dreg_bf16
                        )
                if ot == STORE_TILES - 1:
                    T.ptx.tcgen05.fence.before_thread_sync()
                    tmem_empty_barriers.arrive(tmem_idx, remote=0)
                T.ptx.fence.proxy_async("shared::cta")
                T.cuda.warpgroup_sync(8)
                d_m: T.let = m_idx * DG_BLOCK_M + (ot * 16 if SWAP_AB else 0)
                d_n: T.let = n_idx * DG_BLOCK_N + (0 if SWAP_AB else ot * D_SMEM_N)
                if warp_id == 0:
                    if T.ptx.elect_sync():
                        Tx.copy_async(
                            D[d_m : d_m + D_TILE_M, d_n : d_n + D_TILE_N],
                            D_smem[stage],
                            dispatch="tma_auto",
                            prefetch_tensormap=True,
                        )
                        T.ptx.cp_async.bulk.commit_group()
                T.cuda.warp_sync()
                tma_stage_idx = tma_stage_idx ^ 1
                store_iter = store_iter + 1

        T.cuda.trap_when_assert_failed(tmem_pool.addr == 0)
        tma_stage_idx = 0
        store_iter = 0
        tile_scheduler.linear_idx = bx
        tile_scheduler.tile_count = 0
        while tile_scheduler.valid():
            tile_scheduler.update_current_m_n_idx(tile_scheduler.linear_idx)
            tmem_idx = tile_scheduler.tile_idx % TMEM_DEPTH
            tmem_phase = tile_scheduler.tile_idx // TMEM_DEPTH & 1
            tmem_full_barriers.wait(tmem_idx, tmem_phase)
            T.ptx.tcgen05.fence.after_thread_sync()
            epilogue()
            tile_scheduler.linear_idx = tile_scheduler.linear_idx + SM_NUMBER
            tile_scheduler.tile_count = tile_scheduler.tile_count + 1
    # The epilogue warpgroup and peer CTA must finish all TMEM reads first.
    _cluster_sync_relaxed()
    if wg_id == 0 and warp_id == 0:
        T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=512, cta_group=CTA_GROUP)


def tir_kernel(M: int, N: int, K: int):
    swap_ab, dg_block_m, dg_block_n, smem_pipe_depth, log_m, log_n = _choose_deepgemm_config(
        M, N, K
    )
    return _kernel.specialize(
        M=M,
        N=N,
        K=K,
        SWAP_AB=swap_ab,
        DG_BLOCK_M=dg_block_m,
        DG_BLOCK_N=dg_block_n,
        SMEM_DEPTH=smem_pipe_depth,
        LOGICAL_M_CLUSTER=log_m,
        LOGICAL_N_CLUSTER=log_n,
    )


KERNEL_META = {"name": "fp8_blockwise_gemm", "category": "gemm", "compute_capability": 10}
CONFIGS = [
    {"M": 4096, "N": 2112, "K": 7168, "label": "deepgemm_m4096_n2112_k7168"},
    {"M": 4096, "N": 576, "K": 7168, "label": "deepgemm_m4096_n576_k7168"},
    {"M": 4096, "N": 24576, "K": 1536, "label": "deepgemm_m4096_n24576_k1536"},
    {"M": 4096, "N": 32768, "K": 512, "label": "deepgemm_m4096_n32768_k512"},
    {"M": 4096, "N": 7168, "K": 16384, "label": "deepgemm_m4096_n7168_k16384"},
    {"M": 4096, "N": 4096, "K": 7168, "label": "deepgemm_m4096_n4096_k7168"},
    {"M": 4096, "N": 7168, "K": 2048, "label": "deepgemm_m4096_n7168_k2048"},
]


def get_kernel(M, N, K):
    return tir_kernel(M, N, K)


def run_test(M=1024, N=1024, K=1024):
    """Compile, run, and verify kernel."""
    import torch
    import torch.nn.functional as F

    from tirx_kernels.runner import compile_kernel

    kernel = tir_kernel(M, N, K)
    A_fp8, B_fp8, sfa, sfb, sfa_pack, sfb_pack, C_ref, A_origin, B_origin = prepare_data(M, N, K)
    C_tvm = torch.zeros_like(C_ref).to(torch.bfloat16).to("cuda")
    ex = compile_kernel(kernel)
    ex(A_fp8, B_fp8, sfa_pack, sfb_pack, C_tvm)
    cosine_sim = F.cosine_similarity(C_tvm.reshape(-1).float(), C_ref.reshape(-1).float(), dim=0)
    assert cosine_sim > 0.97, f"fp8_blockwise_gemm cosine_sim {cosine_sim:.6f} <= 0.97"


def run_bench(
    M=1024,
    N=1024,
    K=1024,
    *,
    warmup=None,
    repeat=None,
    timer=None,
    kernel_fair: bool | None = None,
    **kwargs,
):
    """Benchmark DeepGEMM main kernel against the TIRx kernel."""
    import torch

    from tirx_kernels.runner import compile_kernel

    if kernel_fair is None:
        kernel_fair = _env_flag("TIRX_FP8_BLOCKWISE_GEMM_KERNEL_FAIR", default=True)

    kernel = tir_kernel(M, N, K)
    ex = compile_kernel(kernel)

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    A_fp8, B_fp8, sfa, sfb, sfa_pack, sfb_pack, C_ref, _, _ = prepare_data(M, N, K)
    C_tvm = torch.zeros_like(C_ref).to(torch.bfloat16).to("cuda")

    funcs = {"tir": lambda: ex(A_fp8, B_fp8, sfa_pack, sfb_pack, C_tvm)}

    def _deepgemm():
        import deep_gemm

        C_dg = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        if kernel_fair:
            sfa_dg = _dg_scale_view(sfa_pack, M)
            sfb_dg = _dg_scale_view(sfb_pack, N)
            return lambda: deep_gemm.fp8_gemm_nt(
                (A_fp8, sfa_dg), (B_fp8, sfb_dg), C_dg, disable_ue8m0_cast=False, recipe=(1, 1, 128)
            )
        return lambda: deep_gemm.fp8_gemm_nt(
            (A_fp8, sfa), (B_fp8, sfb), C_dg, disable_ue8m0_cast=False, recipe=None
        )

    result = bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"deepgemm": _deepgemm},
        **kwargs,
    )
    result["kernel_fair"] = kernel_fair
    return result
