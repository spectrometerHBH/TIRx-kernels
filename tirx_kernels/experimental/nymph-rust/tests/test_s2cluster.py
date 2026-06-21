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
"""

import numpy as np
import pytest

import nymph_rs as nr
from nymph_rs import IRBuilder, MemorySpace, DType, TensorSlice, FenceKind, FenceScope, MBarKind


def _sl(t, o, s):
    return TensorSlice(tensor=t, offsets=o, shape=s)


def _build(with_wait: bool):
    # 2-CTA cluster. Each CTA c loads g[c] -> sxchg, s2clusters it into the PEER's s,
    # signalling the peer's `full`; then (with_wait) waits its own `full` and reads s
    # -> out[c]. So out[c] should equal g[c^1].
    k = IRBuilder("s2c", num_warps=4, smem_size_bytes=256, launch_shape=(2,), cluster_shape=(2,))
    g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(2,))
    out = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(2,))
    sxchg = k.tensor(space=MemorySpace.SMEM, dtype=DType.F32, shape=(1,), byte_offset=0)
    s = k.tensor(space=MemorySpace.SMEM, dtype=DType.F32, shape=(1,), byte_offset=64)
    r = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(1,))
    full = k.mbar(kind=MBarKind.TMA, stages=1)
    with k.role(warp=0):
        k.mbarrier_init(full, count=1)
        cta = k.ctaid_in_cluster()
        peer = 1 - cta
        with k.if_(k.tid_in_wg().eq(0)):
            # gmem coord = this CTA's global id (== ctaid_in_cluster for one cluster)
            k.reg_load(r, _sl(g, (cta,), (1,)))
            k.reg_store(_sl(sxchg, (0,), (1,)), r)
        k.warp_sync()
        k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)  # generic SMEM write -> async copy
        with k.if_(k.tid_in_wg().eq(0)):
            k.mbarrier_arrive_expect_tx(k.mbar_ref(full, remote_coord=peer), bytes=4)
            k.cp_async_bulk_s2cluster(_sl(s, (0,), (1,)), _sl(sxchg, (0,), (1,)),
                                      mbar=k.mbar_ref(full, remote_coord=peer), bytes=4)
        if with_wait:
            k.mbarrier_wait(full, phase=0)   # our `full`, filled by the peer's s2cluster
        with k.if_(k.tid_in_wg().eq(0)):
            k.reg_load(r, _sl(s, (0,), (1,)))
            k.reg_store(_sl(out, (cta,), (1,)), r)
    return k.build(), g, out


def test_s2cluster_cross_cta_happens_before_and_value():
    kernel, g, out = _build(with_wait=True)
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"
    res = nr.interpret(kernel, {g: np.array([10.0, 20.0], np.float32)})
    o = np.asarray(res[out.id], np.float32).reshape(2)
    # out[c] = g[peer] = g[c^1]
    np.testing.assert_allclose(o, np.array([20.0, 10.0], np.float32))


def test_s2cluster_without_wait_is_flagged():
    kernel, _, _ = _build(with_wait=False)
    result = nr.check_protocol(kernel)
    assert result["status"] != "Passed", "peer read of s2cluster-written SMEM before the wait must be flagged"
