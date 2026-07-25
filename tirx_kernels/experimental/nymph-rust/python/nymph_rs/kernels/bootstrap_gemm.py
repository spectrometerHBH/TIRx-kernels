"""Minimal single-tile/single-k cta_group=2 fp16 GEMM in nymph IR — the codegen bootstrap target.
M=256 (128/CTA), N=128, K=64. No scheduler warp, no ring, no persistent loop, direct reg->gmem epilogue."""

from nymph_rs.builder import IRBuilder, TmemBand
from nymph_rs.nymph_rs import (
    DType,
    FenceKind,
    MBarKind,
    MemorySpace,
    SmemSwizzleLayout,
    Swizzle,
    TensorSlice,
)


def build_bootstrap_gemm(M=256, N=128, K=64, dtype=DType.F16):
    CTA_GROUP = 2
    BLK_M = M // CTA_GROUP
    BLK_N = N // CTA_GROUP
    eb = 2
    a_tb = BLK_M * K * eb
    b_tb = BLK_N * K * eb
    k = IRBuilder(
        "nymph_bootstrap_gemm",
        num_warps=8,
        smem_size_bytes=a_tb + b_tb,
        launch_shape=(CTA_GROUP,),
        cluster_shape=(CTA_GROUP,),
    )
    a_g = k.arg(space=MemorySpace.GMEM, dtype=dtype, shape=(M, K))
    b_g = k.arg(space=MemorySpace.GMEM, dtype=dtype, shape=(N, K))
    c_g = k.arg(space=MemorySpace.GMEM, dtype=dtype, shape=(M, N))
    sw = SmemSwizzleLayout(Swizzle.B128)
    a_s = k.tensor(space=MemorySpace.SMEM, dtype=dtype, shape=(BLK_M, K), layout=sw, byte_offset=0)
    b_s = k.tensor(
        space=MemorySpace.SMEM, dtype=dtype, shape=(BLK_N, K), layout=sw, byte_offset=a_tb
    )
    # TMEM: the f32 accumulator band at physical col 0 (the alloc covers cols 0..N).
    accum = TmemBand(col0=0, dtype=DType.F32)
    accum_frag = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(8,))
    out_frag = k.tensor(space=MemorySpace.REG, dtype=dtype, shape=(8,))
    smem_full = k.mbar(kind=MBarKind.TMA, stages=1, leader_routed=True)
    mma_done = k.mbar(kind=MBarKind.TCGEN05, stages=1)
    peer_full = k.mbar_ref(smem_full, remote_coord=1)
    cr = k.ctaid_in_cluster()
    with k.if_warp(0):
        # tmem_alloc is warp-collective (full warp 0); mbarrier.init is
        # per-thread, so one elected thread runs the inits.
        k.tmem_alloc(0, N, CTA_GROUP)
        with k.if_elected():
            k.mbarrier_init(smem_full, count=1)
            k.mbarrier_init(mma_done, count=1)
    # Prologue seal (canon's `fence.mbarrier_init` + cluster barrier): the inits
    # and the TMEM alloc must be complete before any role touches a barrier or
    # the shared TMEM — the checker's lifecycle proof needs this happens-before
    # edge, not just the sampled epoch order.
    k.fence(kind=FenceKind.MBARRIER_INIT)
    k.cluster_sync()
    with k.if_warp(4):  # loader
        a_m = cr * BLK_M
        b_n = cr * BLK_N
        # Single-thread issue: one elected lane drives the whole TMA burst
        # (the arrive + both loads), exactly canon's elected loader.
        with k.if_elected():
            k.mbarrier_arrive_expect_tx(smem_full, bytes=a_tb + b_tb)
            k.tma_load(
                a_s, a_g, mbar=smem_full, coords=(a_m, 0), shape=(BLK_M, K), cta_group=CTA_GROUP
            )
            k.tma_load(
                b_s, b_g, mbar=smem_full, coords=(b_n, 0), shape=(BLK_N, K), cta_group=CTA_GROUP
            )
    with k.if_warp(5):  # MMA (leader)
        with k.if_(cr.eq(0)):
            k.mbarrier_wait(smem_full, phase=0)
            k.mbarrier_wait(peer_full, phase=0)
            # tcgen05.mma/commit are strictly single-thread issue: one elected
            # lane runs the MMA burst (canon's `if elect_sync(): gemm; commit`).
            with k.if_elected():
                # ONE full-K MMA over the whole K extent — the IR's dense k is an
                # ordered run of k/16 atomic MMAs (canon's one-issue full-K gemm_async).
                k.tcgen05_mma(
                    accum.at(0, 0),
                    a_s[:, :],
                    b_s[:, :],
                    m=M,
                    n=N,
                    k=K,
                    accum=False,
                    cta_group=CTA_GROUP,
                )
                k.tcgen05_commit(mma_done, cta_group=CTA_GROUP, multicast_cta_mask=0b11)
    with k.if_warpgroup(0):  # epilogue
        k.mbarrier_wait(mma_done, phase=0)
        for cb in range(N // 8):
            col = cb * 8
            k.tcgen05_ld(accum_frag, accum.at(0, col), num=8)
            k.tcgen05_wait_ld()
            k.reg_cvt(out_frag, accum_frag)
            k.reg_store(
                TensorSlice(tensor=c_g, offsets=(cr * BLK_M + k.tid_in_wg(), col), shape=(1, 8)),
                out_frag,
            )
    # Cluster-wide barrier before freeing TMEM (canon's `cluster_sync()` right
    # before relinquish + dealloc): both CTAs' MMA writes and epilogue TMEM
    # reads must be done before either CTA frees the shared band.
    k.cluster_sync()
    with k.if_warp(0):
        k.tmem_relinquish(CTA_GROUP)
        k.tmem_dealloc(0, N, CTA_GROUP)
    return k.build()
