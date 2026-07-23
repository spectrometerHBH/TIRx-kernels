"""named_barrier — cross-warpgroup `bar.sync barrier_id, num_warps*32`.

Models flashattn's NamedBarrierBwdSm100 (a barrier spanning the 2 compute warpgroups).
Unlike wg_sync (per-warpgroup) / cta_sync (whole CTA), this rendezvouses threads from
DIFFERENT warpgroup branches on a shared barrier_id and establishes a cross-stream
happens-before edge in the protocol checker (bar.sync's release/acquire semantics).

Per-warp adaptation of the nymph-codegen original: `role(warpgroup=w)` becomes
`if_warpgroup(w)` (masked If dispatch); the two NamedBarrier STATEMENTS still
rendezvous on one (cta, barrier_id) — the identity change under test.
"""

import nymph_rs as nr
from nymph_rs import DType, IRBuilder, MemorySpace, TensorSlice


def _sl(t, o, s):
    return TensorSlice(tensor=t, offsets=o, shape=s)


def _build(with_barrier: bool):
    # wg0 writes sm[0] = g[3]; wg1 reads sm[0] -> g[1]. The cross-warpgroup
    # SMEM write->read is ordered ONLY by the named barrier between the branches.
    k = IRBuilder("nb", num_warps=16, smem_size_bytes=256, launch_shape=(1,))
    g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(8,))
    sm = k.tensor(space=MemorySpace.SMEM, dtype=DType.F32, shape=(8,), byte_offset=0)
    r = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(1,))
    with k.if_warpgroup(0):
        with k.if_(k.tid_in_wg().eq(0)):
            k.reg_load(r, _sl(g, (3,), (1,)))
            k.reg_store(_sl(sm, (0,), (1,)), r)
        if with_barrier:
            k.named_barrier(barrier_id=1, num_warps=8)
    with k.if_warpgroup(1):
        if with_barrier:
            k.named_barrier(barrier_id=1, num_warps=8)
        with k.if_(k.tid_in_wg().eq(0)):
            k.reg_load(r, _sl(sm, (0,), (1,)))
            k.reg_store(_sl(g, (1,), (1,)), r)
    return k.build(), g


def test_named_barrier_establishes_happens_before():
    kernel, _ = _build(with_barrier=True)
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"


def test_named_barrier_absent_is_a_race():
    # Negative control: without the barrier the cross-warpgroup SMEM write->read
    # is an unordered conflict — the checker must still flag it.
    kernel, _ = _build(with_barrier=False)
    result = nr.check_protocol(kernel)
    assert result["status"] == "Failed"
    assert any(d.get("code") == "memory_data_race" for d in result.get("diagnostics", []))
