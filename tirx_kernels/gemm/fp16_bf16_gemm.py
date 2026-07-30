from __future__ import annotations

import torch

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.bench import bench
from tvm.tirx.lang.pipeline import MBarrier, Pipeline, PipelineState
from tvm.tirx.lang.tile_scheduler import ClusterLaunchControlScheduler


def prepare_data(dtype, M, N, K):
    torch_dev = torch.device("cuda")
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    A = torch.randn(M, K).to(dtype).to(torch_dev)
    B = torch.randn(N, K).to(dtype).to(torch_dev)
    C = torch.zeros((M, N), dtype=dtype).to(torch_dev)
    return (A, B, C)


_DTYPE_MAP = {"fp16": tvm.DataType("float16"), "bf16": tvm.DataType("bfloat16")}


def _swizzle_for_row_bytes(row_bytes):
    """Pick the MMA-shared swizzle atom matching the tile row width (the 128/64/32B
    swizzle is selected from the row byte width)."""
    from tvm.backend.cuda.operator.tile_primitive.tma_utils import SwizzleMode

    if row_bytes % 128 == 0:
        return SwizzleMode.SWIZZLE_128B_ATOM
    if row_bytes % 64 == 0:
        return SwizzleMode.SWIZZLE_64B_ATOM
    return SwizzleMode.SWIZZLE_32B_ATOM


# Per-shape tuning knobs (CTA_M=256 always): cta_n/cta_k MMA tile, l2_group_size (L2
# supergroup), overlap_epilogue, pipe_depth (A/B smem), wb_pipe_depth (epilogue).
GEMM_CONFIGS = {
    1024: {
        "cta_n": 64,
        "cta_k": 64,
        "l2_group_size": 4,
        "overlap_epilogue": True,
        "pipe_depth": 10,
        "wb_pipe_depth": 2,
        "scheduler": "static",
    },
    2048: {
        "cta_n": 256,
        "cta_k": 64,
        "l2_group_size": 16,
        "overlap_epilogue": True,
        "pipe_depth": 6,
        "wb_pipe_depth": 8,
    },
    4096: {
        "cta_n": 256,
        "cta_k": 64,
        "l2_group_size": 16,
        "overlap_epilogue": True,
        "pipe_depth": 6,
        "wb_pipe_depth": 8,
    },
    8192: {
        "cta_n": 256,
        "cta_k": 64,
        "l2_group_size": 8,
        "overlap_epilogue": False,
        "pipe_depth": 4,
        "wb_pipe_depth": 8,
    },
    16384: {
        "cta_n": 256,
        "cta_k": 64,
        "l2_group_size": 4,
        "overlap_epilogue": False,
        "pipe_depth": 4,
        "wb_pipe_depth": 8,
    },
}
_DTYPE_GEMM_CONFIGS = {
    ("fp16", 8192): {
        "overlap_epilogue": True,
        "pipe_depth": 5,
        "d_pipe_depth": 8,
        "stmatrix_epilogue": True,
    },
    ("bf16", 8192): {"fused_no_overlap_epilogue": True, "stmatrix_epilogue": True},
    ("fp16", 16384): {
        "packed_no_overlap_epilogue": True,
        "prestage_packed_chunk": True,
        "stmatrix_epilogue": True,
    },
    ("bf16", 16384): {
        "fused_no_overlap_epilogue": True,
        "stmatrix_epilogue": True,
        "smem_desc_mode": "recompute",
    },
}
# Default config for shapes not in the table above.
_DEFAULT_CONFIG = {
    "cta_n": 256,
    "cta_k": 64,
    "l2_group_size": 8,
    "overlap_epilogue": False,
    "pipe_depth": 4,
    "wb_pipe_depth": 8,
    "scheduler": "clc",
}


@T.jit
def _kernel(
    A: T.Buffer((M, K), ab_type),
    B: T.Buffer((N, K), ab_type),
    D: T.Buffer((M, N), ab_type),
    *,
    M: T.constexpr,
    N: T.constexpr,
    K: T.constexpr,
    ab_type: T.constexpr,
    # Independent tuning knobs only (from GEMM_CONFIGS, selected by N). Everything
    # else is a constant or derived from these.
    MMA_N: T.constexpr,
    BLK_K: T.constexpr,
    PIPE_DEPTH: T.constexpr,
    WB_PIPE_DEPTH: T.constexpr,
    D_PIPE_DEPTH: T.constexpr,
    L2_GROUP_SIZE: T.constexpr,
    OVERLAP_EPILOGUE: T.constexpr,
    STATIC_SCHEDULER: T.constexpr,
    STMATRIX_EPILOGUE: T.constexpr,
    FUSED_NO_OVERLAP_EPILOGUE: T.constexpr,
    PACKED_NO_OVERLAP_EPILOGUE: T.constexpr,
    PRESTAGE_PACKED_CHUNK: T.constexpr,
    SMEM_DESC_MODE: T.constexpr,
):
    # Named locals: knob-branching or heavily-reused geometry; single-use values and
    # constants (CTA_GROUP=2, the cluster grid, ...) are inlined at their use-sites.
    NUM_CONSUMER = T.meta_var(1 if OVERLAP_EPILOGUE else 2)
    MMA_PIPE = T.meta_var(2 if OVERLAP_EPILOGUE else 1)
    TMEM_SLOTS = T.meta_var(MMA_PIPE if OVERLAP_EPILOGUE else NUM_CONSUMER)
    TMEM_PHASE_DEPTH = T.meta_var(MMA_PIPE if OVERLAP_EPILOGUE else 1)
    NUM_D_TILES = T.meta_var(D_PIPE_DEPTH)
    BLK_M = T.meta_var(128)  # CTA_M/2: per-CTA A rows (2-SM MMA combines the cluster)
    BLK_N = T.meta_var(MMA_N // 2)  # per-CTA B rows (cluster covers MMA_N)
    EPI_N = T.meta_var(MMA_N // WB_PIPE_DEPTH)  # epilogue N tile
    TMEM_COLS = T.meta_var(max(32, TMEM_SLOTS * MMA_N))
    NUM_M_TILES = T.meta_var(M // (256 * NUM_CONSUMER))
    NUM_N_TILES = T.meta_var(N // MMA_N)

    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    cbx, cby = T.cta_id_in_cluster([2, 1], preferred=[2, 1])
    bx = T.cta_id([(M // (256 * NUM_CONSUMER) * (N // MMA_N) * 2)])
    wg_id = T.warpgroup_id([(NUM_CONSUMER + 1)])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])

    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    tmem_pool = T.TMEMPool(
        pool, total_cols=TMEM_COLS, cta_group=2, tmem_addr=tmem_addr, sync_after_alloc=False
    )
    smem_mbar_leader = (wg_id == 0) & (warp_id == 1) & (lane_id == 0)
    tmem_mbar_leader = (wg_id == 0) & (warp_id == 2) & (lane_id == 0)
    alloc_mbar_leader = (wg_id == 0) & (warp_id == 0) & (lane_id == 0)
    # Input smem pipeline (full=tma expect_tx, empty=tcgen05 consumed).
    smem_pipe = Pipeline(
        pool,
        PIPE_DEPTH,
        full="tma",
        empty="tcgen05",
        init_empty=NUM_CONSUMER,
        leader=smem_mbar_leader,
    )
    # Accumulator tmem pipeline (full=tcgen05 commit, empty=mbar consumed).
    tmem_pipe = Pipeline(
        pool, TMEM_SLOTS, full="tcgen05", empty="mbar", init_empty=2 * 128, leader=tmem_mbar_leader
    )
    # CLC owns the work-stealing handshake. The 1024 static schedule launches
    # exactly one tile per cluster and avoids allocating or running it.
    if not STATIC_SCHEDULER:
        clc_sched = ClusterLaunchControlScheduler(
            pool,
            num_m_tiles=NUM_M_TILES,
            num_n_tiles=NUM_N_TILES,
            l2_group_size=L2_GROUP_SIZE,
            cta_group=2,
            finish_arrivals=((2 + NUM_CONSUMER) * 2 + NUM_CONSUMER),
        )
    # Teardown handshake: 1-arrival cross-CTA mbarrier (OVERLAP) vs the full cluster_sync.
    tmem_fin = Pipeline(pool, 1, full="mbar", empty="mbar", init_full=1, leader=tmem_mbar_leader)
    alloc_done = MBarrier(pool, 1, leader=alloc_mbar_leader)
    alloc_done.init(1)
    pool.move_base_to(1024)
    Asmem = pool.alloc_tcgen05_mma_AB((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ab_type)
    Bsmem = pool.alloc_tcgen05_mma_AB((PIPE_DEPTH, BLK_N, BLK_K), ab_type)
    Dsmem = pool.alloc_tcgen05_mma_AB(
        (NUM_CONSUMER, NUM_D_TILES, BLK_M, EPI_N),
        ab_type,
        swizzle_mode=_swizzle_for_row_bytes(EPI_N * (ab_type.bits // 8)),
    )
    pool.commit()
    smem_full_cta0 = smem_pipe.full.remote_view(0)
    tmem = tmem_pool.alloc((128, TMEM_COLS), "float32")
    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    # OVERLAP shapes split the prologue cluster barrier: arrive (relaxed) here, then each
    # active role waits(acquire) after its own setup so the latency overlaps it (idle warps
    # skip the wait). No-overlap shapes keep the cheaper fused cluster_sync.
    if OVERLAP_EPILOGUE:
        T.ptx.barrier.cluster.arrive(sem="relaxed", aligned=True)
    else:
        T.cuda.cluster_sync()

    # Delay the pair-collective allocation until after the prologue barrier:
    # loaders can begin their first TMA flight while warp 0 allocates TMEM.
    # MMA waits on the local alloc_done gate before its first tcgen05 access.
    tmem_pool.commit()
    if (wg_id == 0) & (warp_id == 0) & T.ptx.elect_sync():
        alloc_done.arrive(0)

    static_task = T.meta_var(bx // 2)
    if NUM_M_TILES <= L2_GROUP_SIZE:
        static_m = T.meta_var(static_task % NUM_M_TILES)
        static_n = T.meta_var(static_task // NUM_M_TILES)
    else:
        static_group_span = T.meta_var(L2_GROUP_SIZE * NUM_N_TILES)
        static_group = T.meta_var(static_task // static_group_span)
        static_within = T.meta_var(static_task % static_group_span)
        static_m = T.meta_var(static_group * L2_GROUP_SIZE + static_within % L2_GROUP_SIZE)
        static_n = T.meta_var(static_within // L2_GROUP_SIZE)

    if wg_id == NUM_CONSUMER:
        # ==================== PRODUCER warpgroup ====================
        T.ptx.setmaxnreg(False, 56)  # reduce the producer's per-warp register budget
        if warp_id == 3:
            # -------- LOADER (TMA) --------
            tma_cur = PipelineState(PIPE_DEPTH, 1)
            if OVERLAP_EPILOGUE:
                T.ptx.barrier.cluster.wait(
                    acquire=True, aligned=False
                )  # split cluster barrier (loader)

            @T.inline
            def tma_load_stage(k_tile, m_idx, n_idx):
                smem_pipe.empty.wait(tma_cur.stage, tma_cur.phase)
                stage = tma_cur.stage
                tma_config = T.meta_var(
                    {
                        "dispatch": "tma_auto",
                        "cta_group": 2,
                        "mbar": smem_full_cta0.ptr_to([stage]),
                        # Prefetch the A/B tensormaps at entry.
                        "prefetch_tensormap": True,
                    }
                )
                k = T.meta_var(k_tile * BLK_K)
                b_n = T.meta_var((n_idx * 2 + cbx) * BLK_N)
                # Each CTA loads its OWN A rows / B cols (depend on cbx); cta_group=2
                # routes the completion mbarrier to the cluster, not a data multicast.
                for c in T.unroll(NUM_CONSUMER):
                    a_m = T.meta_var(((m_idx * 2 + cbx) * NUM_CONSUMER + c) * BLK_M)
                    Tx.copy_async(
                        Asmem[stage, c], A[a_m : a_m + BLK_M, k : k + BLK_K], **tma_config
                    )
                Tx.copy_async(Bsmem[stage], B[b_n : b_n + BLK_N, k : k + BLK_K], **tma_config)
                # Loader-side expect_tx for the whole stage's bytes; cbx==0 owns the mbar.
                if cbx == 0:
                    smem_full_cta0.arrive(
                        stage,
                        2 * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * (ab_type.bits // 8),
                    )

            @T.inline
            def tma_load(m_idx, n_idx):
                for k_tile in T.serial(K // BLK_K):
                    tma_load_stage(k_tile, m_idx, n_idx)
                    tma_cur.advance()

            if STATIC_SCHEDULER:
                if T.ptx.elect_sync():
                    tma_load(static_m, static_n)
            else:
                # CLC loader: load the current tile, then consume the schedule for the next.
                ld = clc_sched.worker("ld_sched")
                ld.init(bx // 2)
                if T.ptx.elect_sync():
                    while ld.valid():
                        m_idx = T.meta_var(ld.m_idx)
                        n_idx = T.meta_var(ld.n_idx)
                        tma_load(m_idx, n_idx)
                        ld.consume()
                        ld.advance_coords()
                        ld.mark_done_if_drained()
        elif warp_id == 2:
            # -------- CLC SCHEDULER --------
            if not STATIC_SCHEDULER:
                if OVERLAP_EPILOGUE:
                    T.ptx.barrier.cluster.wait(
                        acquire=True, aligned=False
                    )  # split barrier (scheduler)
                clc_sched.run_scheduler(cbx)
                # CLC has no more tiles to hand out. Match nvjet's PREEXIT tail
                # so a following same-stream launch can enter while the final
                # MMA and epilogue work drains.
                if cbx == 0:
                    T.ptx.griddepcontrol.launch_dependents()
        elif (warp_id < NUM_CONSUMER) & (cbx == 0):
            # -------- MMA (tcgen05) --------
            mma_smem = PipelineState(PIPE_DEPTH, 0)
            # tmem wait state: double-buffer (overlap, depth=MMA_PIPE) or a single
            # slot toggled per tile (no-overlap, depth=1).
            tmem_buf = PipelineState(TMEM_PHASE_DEPTH, 1)
            accum: T.int32
            if OVERLAP_EPILOGUE:
                T.ptx.barrier.cluster.wait(
                    acquire=True, aligned=False
                )  # split cluster barrier (MMA)
            alloc_done.wait(0, 0)

            @T.inline
            def mma_stage(buf):
                smem_pipe.full.wait(mma_smem.stage, mma_smem.phase)
                stage = mma_smem.stage
                tmem_n = T.meta_var(buf * MMA_N)
                # 2-SM tcgen05 A@B^T (B stored (N,K), transB=False); accum=0 on the
                # first k-tile (overwrite), 1 thereafter.
                Tx.gemm_async(
                    tmem[:, tmem_n : tmem_n + MMA_N],
                    Asmem[stage, warp_id],
                    Bsmem[stage],
                    accum=accum,
                    dispatch="tcgen05",
                    cta_group=2,
                    smem_desc=SMEM_DESC_MODE,
                )
                accum = 1
                smem_pipe.empty.arrive(mma_smem.stage, cta_group=2, cta_mask=3)

            @T.inline
            def mma():
                slot = T.meta_var(tmem_buf.stage if OVERLAP_EPILOGUE else warp_id)
                tmem_pipe.empty.wait(slot, tmem_buf.phase)
                accum = 0
                for k_tile in T.serial(K // BLK_K):
                    mma_stage(slot)
                    mma_smem.advance()
                tmem_pipe.full.arrive(slot, cta_group=2, cta_mask=3)
                tmem_buf.advance()

            if STATIC_SCHEDULER:
                if T.ptx.elect_sync():
                    mma()
            else:
                # CLC MMA: consume the schedule, then accumulate. mma() ignores the tile
                # coords (it MMAs whatever the loader staged), so reset() not init().
                mm = clc_sched.worker("mma_sched")
                mm.reset()
                if T.ptx.elect_sync():
                    while mm.valid():
                        mm.consume()
                        mma()
                        mm.mark_done_if_drained()
    elif wg_id < NUM_CONSUMER:
        # ==================== CONSUMER / EPILOGUE warpgroup(s) ====================
        if not OVERLAP_EPILOGUE:
            T.ptx.setmaxnreg(True, 224)  # raise the consumer's per-warp register budget
        wb_buf = PipelineState(TMEM_PHASE_DEPTH, 0)
        store_iter: T.int32
        store_iter = 0
        if OVERLAP_EPILOGUE:
            T.ptx.barrier.cluster.wait(
                acquire=True, aligned=False
            )  # split cluster barrier (consumer)

        @T.inline
        def writeback(m_idx, n_idx):
            slot = T.meta_var(wb_buf.stage if OVERLAP_EPILOGUE else wg_id)
            tmem_pipe.full.wait(slot, wb_buf.phase)
            tmem_base = T.meta_var(slot * MMA_N)
            if OVERLAP_EPILOGUE or FUSED_NO_OVERLAP_EPILOGUE:
                # Fused per-chunk load+store, overlapping the next MMA. Keep Dreg_16b
                # exactly EPI_N wide: a wider frag drops STSM dispatch -> STS.128 and spills
                # regs 38->255 (measured).
                if STMATRIX_EPILOGUE:
                    Dreg = T.alloc_tcgen05_ldst_frag("16x256b", (128, EPI_N), "float32")
                    Dreg_16b = T.alloc_cast_frag(Dreg, ab_type)
                else:
                    Dreg = T.wg_reg_tile(EPI_N)
                    Dreg_16b = T.wg_reg_tile(EPI_N, dtype=ab_type)
                for i in T.unroll(WB_PIPE_DEPTH):
                    tn = T.meta_var(tmem_base + i * EPI_N)
                    Tx.wg.copy_async(Dreg, tmem[:, tn : tn + EPI_N])
                    T.ptx.tcgen05.wait.ld()
                    Tx.wg.cast(Dreg_16b, Dreg)
                    if i == WB_PIPE_DEPTH - 1:
                        tmem_pipe.empty.arrive(slot, remote=0, pred=True)
                    db = T.meta_var(store_iter % NUM_D_TILES)
                    if store_iter >= NUM_D_TILES:
                        T.ptx.cp_async.bulk.wait_group(NUM_D_TILES - 1, read=True)
                        T.cuda.warpgroup_sync(wg_id + 10)
                    if STMATRIX_EPILOGUE:
                        for jv in T.unroll(EPI_N // 16):
                            if OVERLAP_EPILOGUE:
                                Tx.wg.copy(
                                    Dsmem[0, db, :, jv * 16 : jv * 16 + 16],
                                    Dreg_16b[:, jv * 16 : jv * 16 + 16],
                                    dispatch="ldstmatrix",
                                )
                            else:
                                Tx.wg.copy(
                                    Dsmem[wg_id, db, :, jv * 16 : jv * 16 + 16],
                                    Dreg_16b[:, jv * 16 : jv * 16 + 16],
                                    dispatch="ldstmatrix",
                                )
                    else:
                        for jv in T.unroll(EPI_N // 8):
                            if OVERLAP_EPILOGUE:
                                Tx.wg.copy(
                                    Dsmem[0, db, :, jv * 8 : jv * 8 + 8],
                                    Dreg_16b[:, jv * 8 : jv * 8 + 8],
                                )
                            else:
                                Tx.wg.copy(
                                    Dsmem[wg_id, db, :, jv * 8 : jv * 8 + 8],
                                    Dreg_16b[:, jv * 8 : jv * 8 + 8],
                                )
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if (warp_id == 0) & (lane_id == 0):
                        # Proxy fence by the single TMA-issuing thread (warpgroup_sync above
                        # made writes CTA-visible; an all-128-thread fence was the dominant stall).
                        T.ptx.fence.proxy_async("shared::cta")
                        d_m = T.meta_var(((m_idx * 2 + cbx) * NUM_CONSUMER + wg_id) * BLK_M)
                        d_n = T.meta_var(n_idx * MMA_N + i * EPI_N)
                        if OVERLAP_EPILOGUE:
                            Tx.copy_async(
                                D[d_m : d_m + BLK_M, d_n : d_n + EPI_N],
                                Dsmem[0, db],
                                dispatch="tma_auto",
                                cache_hint="evict_first",
                                prefetch_tensormap=True,
                            )
                        else:
                            Tx.copy_async(
                                D[d_m : d_m + BLK_M, d_n : d_n + EPI_N],
                                Dsmem[wg_id, db],
                                dispatch="tma_auto",
                                cache_hint="evict_first",
                                prefetch_tensormap=True,
                            )
                    # commit_group collectively reconverges the warpgroup (no post-sync).
                    T.ptx.cp_async.bulk.commit_group()
                    store_iter = store_iter + 1
            else:
                # No-overlap: load+cast all chunks, free the accumulator, then store. Stage
                # the tmem->reg f32 load in 16-col sub-chunks so the f32 footprint stays 16
                # (not EPI_N), else the consumer spills (LDL/STL).
                NOL = T.meta_var(EPI_N if PACKED_NO_OVERLAP_EPILOGUE else 16)
                if PACKED_NO_OVERLAP_EPILOGUE:
                    Dreg = T.alloc_tcgen05_ldst_frag("16x256b", (128, NOL), "float32")
                    Dreg_16b = T.meta_var(
                        [
                            T.alloc_cast_frag(Dreg, ab_type)
                            for _ in range(
                                WB_PIPE_DEPTH - 1 if PRESTAGE_PACKED_CHUNK else WB_PIPE_DEPTH
                            )
                        ]
                    )
                else:
                    Dreg = T.wg_reg_tile(NOL)
                    Dreg_16b = T.wg_reg_tile(MMA_N, dtype=ab_type)

                @T.inline
                def stage_chunk(i, db, frag):
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if PACKED_NO_OVERLAP_EPILOGUE:
                        for jv in T.unroll(EPI_N // 16):
                            Tx.wg.copy(
                                Dsmem[wg_id, db, :, jv * 16 : jv * 16 + 16],
                                frag[:, jv * 16 : jv * 16 + 16],
                                dispatch="ldstmatrix",
                            )
                    else:
                        for jv in T.unroll(EPI_N // 8):
                            c0 = T.meta_var(i * EPI_N + jv * 8)
                            Tx.wg.copy(
                                Dsmem[wg_id, db, :, jv * 8 : jv * 8 + 8], Dreg_16b[:, c0 : c0 + 8]
                            )

                @T.inline
                def issue_staged_chunk(i, db):
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if (warp_id == 0) & (lane_id == 0):
                        # Single-thread proxy fence after the CTA sync (see overlap path).
                        T.ptx.fence.proxy_async("shared::cta")
                        d_m = T.meta_var(((m_idx * 2 + cbx) * NUM_CONSUMER + wg_id) * BLK_M)
                        d_n = T.meta_var(n_idx * MMA_N + i * EPI_N)
                        Tx.copy_async(
                            D[d_m : d_m + BLK_M, d_n : d_n + EPI_N],
                            Dsmem[wg_id, db],
                            dispatch="tma_auto",
                            cache_hint="evict_first",  # evict-first L2 policy
                            prefetch_tensormap=True,  # prefetch the D tensormap
                        )
                    # commit_group collectively reconverges the warpgroup (no post-sync).
                    T.ptx.cp_async.bulk.commit_group()
                    store_iter = store_iter + 1

                @T.inline
                def store_chunk(i, frag):
                    db = T.meta_var(store_iter % NUM_D_TILES)
                    if store_iter >= NUM_D_TILES:
                        T.ptx.cp_async.bulk.wait_group(NUM_D_TILES - 1, read=True)
                    stage_chunk(i, db, frag)
                    issue_staged_chunk(i, db)

                if PACKED_NO_OVERLAP_EPILOGUE:

                    @T.inline
                    def load_cast_chunk(i, frag_i):
                        tn = T.meta_var(tmem_base + i * NOL)
                        Tx.wg.copy_async(Dreg, tmem[:, tn : tn + NOL])
                        T.ptx.tcgen05.wait.ld()
                        Tx.wg.cast(frag_i, Dreg)

                    if PRESTAGE_PACKED_CHUNK:
                        pre_db = T.meta_var(store_iter % NUM_D_TILES)
                        if store_iter >= NUM_D_TILES:
                            T.ptx.cp_async.bulk.wait_group(NUM_D_TILES - 1, read=True)
                        load_cast_chunk(0, Dreg_16b[0])
                        stage_chunk(0, pre_db, Dreg_16b[0])

                        @T.inline
                        def load_remaining(i):
                            if i < WB_PIPE_DEPTH:
                                load_cast_chunk(i, Dreg_16b[i - 1])
                                load_remaining(i + 1)

                        load_remaining(1)
                        tmem_pipe.empty.arrive(wg_id, remote=0, pred=True)
                        issue_staged_chunk(0, pre_db)

                        @T.inline
                        def store_remaining(i):
                            if i < WB_PIPE_DEPTH:
                                store_chunk(i, Dreg_16b[i - 1])
                                store_remaining(i + 1)

                        store_remaining(1)
                    else:

                        @T.inline
                        def load_chunks(i):
                            if i < WB_PIPE_DEPTH:
                                load_cast_chunk(i, Dreg_16b[i])
                                load_chunks(i + 1)

                        load_chunks(0)
                        tmem_pipe.empty.arrive(wg_id, remote=0, pred=True)

                        @T.inline
                        def store_chunks(i):
                            if i < WB_PIPE_DEPTH:
                                store_chunk(i, Dreg_16b[i])
                                store_chunks(i + 1)

                        store_chunks(0)
                else:
                    for i in T.unroll(MMA_N // NOL):
                        tn = T.meta_var(tmem_base + i * NOL)
                        Tx.wg.copy_async(Dreg, tmem[:, tn : tn + NOL])
                        T.ptx.tcgen05.wait.ld()
                        Tx.wg.cast(Dreg_16b[:, i * NOL : (i + 1) * NOL], Dreg)
                    tmem_pipe.empty.arrive(wg_id, remote=0, pred=True)
                    for i in T.unroll(WB_PIPE_DEPTH):
                        store_chunk(i, Dreg_16b)

        if STATIC_SCHEDULER:
            writeback(static_m, static_n)
            wb_buf.advance()
        else:
            # CLC consumer: capture the current tile, consume the schedule for the next
            # (overlapping it with the MMA-output wait), then store the captured tile.
            wb = clc_sched.worker("wb_sched")
            wb.init(bx // 2)
            cur_m: T.int32
            cur_n: T.int32
            while wb.valid():
                cur_m = wb.m_idx
                cur_n = wb.n_idx
                wb.consume_wg(wg_id, warp_id, lane_id)
                wb.advance_coords()
                cm = T.meta_var(cur_m)
                cn = T.meta_var(cur_n)
                writeback(cm, cn)
                wb_buf.advance()
                wb.mark_done_if_drained()
        # The overlap teardown can run while the final TMA store drains.
        if not OVERLAP_EPILOGUE:
            T.ptx.cp_async.bulk.wait_group(0)
        if OVERLAP_EPILOGUE:
            # Teardown: warpgroup_sync (all tmem reads done), then warp0 does a 1-arrival
            # cross-CTA mbarrier handshake before dealloc -- lighter than a full cluster_sync.
            T.cuda.warpgroup_sync(wg_id + 10)
            if (warp_id == 0) & (lane_id == 0):
                tmem_fin.full.arrive(0, remote=1 - cbx, pred=True)
            if warp_id == 0:
                tmem_fin.full.wait(0, 0)
    if not OVERLAP_EPILOGUE:
        # No-overlap keeps the full cluster_sync teardown.
        T.cuda.cluster_sync()
    tmem_pool.dealloc()


def tir_kernel(dtype: str, M: int, N: int, K: int):
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}")
    ab_type = _DTYPE_MAP[dtype]
    cfg = dict(GEMM_CONFIGS.get(N, _DEFAULT_CONFIG))
    cfg.update(_DTYPE_GEMM_CONFIGS.get((dtype, N), {}))
    # Bind only the independent knobs; _kernel derives all geometry from these.
    return _kernel.specialize(
        M=M,
        N=N,
        K=K,
        ab_type=ab_type,
        MMA_N=cfg["cta_n"],
        BLK_K=cfg["cta_k"],
        PIPE_DEPTH=cfg["pipe_depth"],
        WB_PIPE_DEPTH=cfg["wb_pipe_depth"],
        D_PIPE_DEPTH=cfg.get("d_pipe_depth", 2 if cfg["wb_pipe_depth"] > 1 else 1),
        L2_GROUP_SIZE=cfg["l2_group_size"],
        OVERLAP_EPILOGUE=cfg["overlap_epilogue"],
        STATIC_SCHEDULER=cfg.get("scheduler", "clc") == "static",
        STMATRIX_EPILOGUE=cfg.get("stmatrix_epilogue", False),
        FUSED_NO_OVERLAP_EPILOGUE=cfg.get("fused_no_overlap_epilogue", False),
        PACKED_NO_OVERLAP_EPILOGUE=cfg.get("packed_no_overlap_epilogue", False),
        PRESTAGE_PACKED_CHUNK=cfg.get("prestage_packed_chunk", False),
        SMEM_DESC_MODE=cfg.get("smem_desc_mode", "hoist"),
    )


KERNEL_META = {"name": "fp16_bf16_gemm", "category": "gemm", "compute_capability": 10}
CONFIGS = [
    {"dtype": d, "M": s, "N": s, "K": s, "label": f"{d}_{s}x{s}x{s}"}
    for d in ["fp16", "bf16"]
    for s in [1024, 2048, 4096, 8192, 16384]
]


def get_kernel(dtype, M, N, K, **kwargs):
    return tir_kernel(dtype, M, N, K)


def run_test(dtype, M, N, K, **kwargs):
    """Compile, run, and verify fp16/bf16 GEMM kernel."""
    from tirx_kernels.runner import compile_kernel

    A, B, C = prepare_data(dtype, M, N, K)
    kernel = tir_kernel(dtype, M, N, K)
    C_tvm = torch.zeros_like(C, device="cuda")
    target = tvm.target.Target("cuda")
    with target:
        ex = compile_kernel(kernel)
        ex(A, B, C_tvm)
    C_ref = torch.matmul(A, B.T)
    torch.testing.assert_close(C_tvm.cpu(), C_ref.cpu(), rtol=0.001, atol=0.01)


def run_bench(dtype, M, N, K, warmup=None, repeat=None, timer=None, **kwargs):
    """Benchmark fp16/bf16 GEMM."""
    kernel = tir_kernel(dtype, M, N, K)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    A, B, C = prepare_data(dtype, M, N, K)
    C_tir = torch.zeros_like(C, device="cuda")

    funcs = {"tir": lambda: ex(A, B, C_tir)}

    def _torch_cublas():
        C_out = torch.zeros_like(C, device="cuda")
        return lambda: torch.matmul(A, B.T, out=C_out)

    references = {"torch-cublas": _torch_cublas}
    if dtype == "bf16":

        def _deepgemm_cublaslt():
            import deep_gemm

            C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            return lambda: deep_gemm.cublaslt_gemm_nt(A, B, C_out, None)

        def _deepgemm_bf16():
            import deep_gemm

            C_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            return lambda: deep_gemm.bf16_gemm_nt(A, B, C_out)

        references.update(
            {"deepgemm-cublaslt": _deepgemm_cublaslt, "deepgemm-bf16": _deepgemm_bf16}
        )

    return bench(funcs, warmup=warmup, repeat=repeat, timer=timer, references=references, **kwargs)
