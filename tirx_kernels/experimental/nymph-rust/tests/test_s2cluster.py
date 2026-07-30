"""cp_async_bulk_s2cluster — dS cross-CTA exchange (cp.async.bulk.shared::cluster).

Async bulk copy of this CTA's SMEM into the PEER CTA's SMEM, signalling the peer's
mbar on completion (the 2-CTA flash-bwd dS handoff). The protocol-relevant design:
the WRITE is attributed to the PEER CTA's SMEM pool (so the race checker matches it
against the peer's read) and the completion is a complete_tx on the PEER's mbar (so
the cross-CTA happens-before closes through the peer's wait). Test both:
  - WITH the peer's mbarrier_wait → the cross-CTA write→read is ordered (Passed) and
    the value lands in the peer (out[c] = g[peer]).
  - WITHOUT the wait → the peer reads s2cluster-written SMEM before waiting its barrier
    → the checker flags it (access-before-wait / race), proving the HB isn't faked.

The per-thread mbarrier.init runs under `if_warp(0) + if_elected()` and is
published to the peer CTA by a top-level `cluster_sync()`; the s2cluster issue
is single-thread (`tid_in_wg()==0`, its issue gate).
"""

import numpy as np
import nymph_rs as nr
from nymph_rs import DType, FenceKind, FenceScope, IRBuilder, MBarKind, MemorySpace, TensorSlice


def _sl(t, o, s):
    return TensorSlice(tensor=t, offsets=o, shape=s)


def _build(with_wait: bool):
    # 2-CTA cluster. Each CTA c loads g[c*4:(c+1)*4] -> sxchg, s2clusters the 16B
    # into the PEER's s, signalling the peer's `full`; then (with_wait) waits its
    # own `full` and reads s -> out[c*4:]. So out[c*4+i] should equal g[(c^1)*4+i].
    k = IRBuilder("s2c", num_warps=4, smem_size_bytes=256, launch_shape=(2,), cluster_shape=(2,))
    g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(8,))
    out = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(8,))
    sxchg = k.tensor(space=MemorySpace.SMEM, dtype=DType.F32, shape=(4,), byte_offset=0)
    s = k.tensor(space=MemorySpace.SMEM, dtype=DType.F32, shape=(4,), byte_offset=64)
    r = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(4,))
    full = k.mbar(kind=MBarKind.TMA, byte_offset=128, stages=1)
    # mbarrier.init is per-thread: one elected thread initializes; the peer CTA
    # arrives on this mbar remotely, so publish the init cluster-wide first.
    with k.if_warp(0), k.if_elected():
        k.mbarrier_init(full, count=1)
    k.cluster_sync()
    with k.if_warp(0):
        cta = k.ctaid_in_cluster()
        peer = 1 - cta
        with k.if_(k.tid_in_wg().eq(0)):
            # gmem coord = this CTA's global id (== ctaid_in_cluster for one cluster)
            k.reg_load(r, _sl(g, (cta * 4,), (4,)))
            k.reg_store(_sl(sxchg, (0,), (4,)), r)
        k.warp_sync()
        k.fence(
            kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA
        )  # generic SMEM write -> async copy
        with k.if_(k.tid_in_wg().eq(0)):
            k.mbarrier_arrive_expect_tx(k.mbar_ref(full, remote_coord=peer), bytes=16)
            k.cp_async_bulk_s2cluster(
                _sl(s, (0,), (4,)),
                _sl(sxchg, (0,), (4,)),
                mbar=k.mbar_ref(full, remote_coord=peer),
                bytes=16,
            )
        if with_wait:
            k.mbarrier_wait(full, phase=0)  # our `full`, filled by the peer's s2cluster
        with k.if_(k.tid_in_wg().eq(0)):
            k.reg_load(r, _sl(s, (0,), (4,)))
            k.reg_store(_sl(out, (cta * 4,), (4,)), r)
    return k.build(), g, out


def test_s2cluster_cross_cta_happens_before_and_value():
    kernel, g, out = _build(with_wait=True)
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"
    res = nr.interpret(
        kernel, {g: np.array([10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0], np.float32)}
    )
    o = np.asarray(res[out.id], np.float32).reshape(8)
    # out[c*4+i] = g[peer*4+i] = g[(c^1)*4+i]
    np.testing.assert_allclose(
        o, np.array([20.0, 21.0, 22.0, 23.0, 10.0, 11.0, 12.0, 13.0], np.float32)
    )


def test_s2cluster_without_wait_is_flagged():
    kernel, _, _ = _build(with_wait=False)
    result = nr.check_protocol(kernel)
    assert result["status"] != "Passed", (
        "peer read of s2cluster-written SMEM before the wait must be flagged"
    )
