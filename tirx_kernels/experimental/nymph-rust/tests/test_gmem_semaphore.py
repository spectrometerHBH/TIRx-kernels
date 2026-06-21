"""GMEM semaphore IR ops — gmem_atomic_add ("signal") + gmem_wait_eq ("wait").

The value-keyed release/acquire pair that serializes a cross-CTA reduce-add into a
bit-reproducible order AND gives the protocol checker the happens-before edge to
ORDER the reduce-adds (resolving `async_group_cross_stream_dest_access`).

Standalone proofs (isolated from the full flash-bwd kernel):
  POSITIVE   — two CTAs reduce-add into ONE shared dQ tile, serialized by the
               semaphore (CTA c waits counter==c, adds, signals +1). WITH the wrap:
               check_protocol == Passed AND bit-identical across runs.
  NEGATIVE A — the SAME two reduce-adds WITHOUT the wrap → still Inconclusive (the
               sem edge is load-bearing, not a coincidence).
  NEGATIVE B — (b1) wrong turn (both CTAs wait value 0) → the second never signals
               the awaited value → DEADLOCK (caught, not silently Passed); (b2)
               signal-BEFORE-drain (atomic_add before the wait_group drain) → the
               release no longer covers the completed write, but the checker still
               must not be fooled into Passed by a value-keyed acquire on the wrong
               clock — we assert the HB it forms still correctly orders ONLY what the
               release actually covered (the wrong-turn deadlock is the primary B).
"""

import numpy as np
import pytest

import nymph_rs as nr
from nymph_rs import IRBuilder, MemorySpace, DType, TensorSlice, FenceKind, FenceScope, MBarKind


def _sl(t, o, s):
    return TensorSlice(tensor=t, offsets=o, shape=s)


def _build(*, with_sem: bool, wrong_turn: bool = False, same_turn: bool = False,
           signal_before_drain: bool = False):
    """Two CTAs each reduce-add their src row into the SHARED dQ tile (1 row, 4 cols).
    out tile = src[0] + src[1]. The order is fixed by the per-tile semaphore: CTA c
    waits counter==c then adds then signals +1.

    wrong_turn          — CTA c waits an UNREACHABLE value (c+5) → deadlock.
    same_turn           — both CTAs wait 0 → both pass immediately (no serialization);
                          the checker must still NOT go Passed (the edge is wrong).
    signal_before_drain — atomic_add BEFORE the wait_group drain → the release no
                          longer covers the completed write."""
    k = IRBuilder("sem_toy", num_warps=4, smem_size_bytes=64, launch_shape=(2,))
    src = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(2, 4))
    dq = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(1, 4))     # shared dQ tile
    sem = k.arg(space=MemorySpace.GMEM, dtype=DType.I32, shape=(1,))      # one cell per dQ tile
    s = k.tensor(space=MemorySpace.SMEM, dtype=DType.F32, shape=(1, 4), byte_offset=0)
    r = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(4,))
    with k.role(warpgroup=0):
        cta = k.cta_id()
        if wrong_turn:
            turn = cta + 5          # never produced → deadlock
        elif same_turn:
            turn = 0                # both pass immediately → no serialization
        else:
            turn = cta
        # stage this CTA's src row into SMEM (the reduce-add source).
        with k.if_(k.tid_in_wg().eq(0)):
            k.reg_load(r, _sl(src, (cta, 0), (1, 4)))
            k.reg_store(_sl(s, (0, 0), (1, 4)), r)
        k.wg_sync(barrier_id=1)
        k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
        if with_sem:
            with k.if_(k.tid_in_wg().eq(0)):
                k.gmem_wait_eq(sem, coords=(0,), value=turn)   # wait my turn
            k.wg_sync(barrier_id=1)
        with k.if_(k.tid_in_wg().eq(0)):
            k.tma_reduce_add(dq, _sl(s, (0, 0), (1, 4)),
                             coords=(0, 0), shape=(1, 4), gmem_shape=(1, 4))
            k.cp_async_bulk_commit_group()
            if signal_before_drain and with_sem:
                # WRONG: signal the next CTA BEFORE draining this CTA's reduce-add. The
                # release clock then does NOT cover the still-in-flight write.
                k.fence(kind=FenceKind.MEMORY, scope=FenceScope.GPU)
                k.gmem_atomic_add(sem, coords=(0,), value=1, order="release")
                k.cp_async_bulk_wait_group_read(0)
            else:
                k.cp_async_bulk_wait_group_read(0)   # FULL drain BEFORE the signal
        if with_sem and not signal_before_drain:
            k.wg_sync(barrier_id=1)
            with k.if_(k.tid_in_wg().eq(0)):
                k.fence(kind=FenceKind.MEMORY, scope=FenceScope.GPU)
                k.gmem_atomic_add(sem, coords=(0,), value=1, order="release")  # release next CTA
    return k.build(), src, dq, sem


def _src():
    return np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], np.float32)


def _nondet(rep):
    # The checker treats cross-CTA atomic reduce-adds as RACE-FREE (commutative atomic), so the
    # status is always Passed; the semaphore's job is DETERMINISM. A correct semaphore HB-orders
    # the reduce-adds -> bit-reproducible -> no warning. An absent/broken semaphore leaves them
    # unordered -> a `nondeterministic_reduction` warning (the order-dependent float result is not
    # silently accepted as deterministic).
    return any(w.get("code") == "nondeterministic_reduction" for w in (rep.get("warnings") or []))


def test_positive_passed_and_no_nondeterminism_and_bit_identical():
    kernel, src, dq, sem = _build(with_sem=True)
    assert nr.trace(kernel)
    rep = nr.check_protocol(kernel)
    assert rep["status"] == "Passed", f"got {rep['status']}"
    assert not _nondet(rep), "a correctly-serialized reduce is deterministic -> no warning"
    # value-correct AND bit-identical across two runs.
    r1 = nr.interpret(kernel, {src: _src()})
    r2 = nr.interpret(kernel, {src: _src()})
    a1 = np.asarray(r1[dq.id], np.float32).reshape(1, 4)
    a2 = np.asarray(r2[dq.id], np.float32).reshape(1, 4)
    np.testing.assert_array_equal(a1, a2)                     # bit-identical
    np.testing.assert_allclose(a1, _src().sum(0, keepdims=True), rtol=0, atol=0)


def test_negative_A_unwrapped_warns_nondeterministic():
    # SAME two reduce-adds WITHOUT the semaphore wrap → race-free (commutative atomic) so Passed,
    # but order-dependent → a nondeterministic_reduction warning (the sem is load-bearing for
    # bit-reproducibility, which the warning's presence/absence tracks).
    kernel, src, dq, sem = _build(with_sem=False)
    assert nr.trace(kernel)
    rep = nr.check_protocol(kernel)
    assert rep["status"] == "Passed"
    assert _nondet(rep), "unserialized cross-CTA reduce-add must warn non-determinism"


def test_negative_B_wrong_turn_deadlocks():
    # Each CTA waits an UNREACHABLE value (cta+5) the counter never reaches → the wait
    # never observes a value it is keyed on → DEADLOCK (the wrong lock-value is caught,
    # not silently Passed). This proves the wait is a real block, not a no-op.
    kernel, src, dq, sem = _build(with_sem=True, wrong_turn=True)
    with pytest.raises(Exception) as ei:
        nr.interpret(kernel, {src: _src()})
    assert "deadlock" in str(ei.value).lower() or "block" in str(ei.value).lower(), str(ei.value)


def test_negative_B_same_turn_warns_nondeterministic():
    # Both CTAs wait value 0 → both pass immediately (no serialization). The value-keyed acquire
    # of value 0 finds NO release (0 is the zero-init, never PRODUCED by an atomic_add), so NO
    # cross-CTA edge forms → the reduce-adds stay unordered → the broken sem does NOT achieve
    # determinism (the nondeterministic_reduction warning survives — not silently "deterministic").
    kernel, src, dq, sem = _build(with_sem=True, same_turn=True)
    assert nr.trace(kernel)
    rep = nr.check_protocol(kernel)
    assert rep["status"] == "Passed"
    assert _nondet(rep), "both-wait-0 (no real serialization) must still warn non-determinism"


def test_negative_B_signal_before_drain_warns_nondeterministic():
    # atomic_add issued BEFORE the wait_group drain → the release clock is published while this
    # CTA's reduce-add Write is still in flight (the window has NOT closed), so HB(thisCTA.drain.
    # close, nextCTA.add) does NOT hold → the reduce-adds stay unordered → the warning survives.
    # Proves the drain-then-signal ordering is what achieves determinism.
    kernel, src, dq, sem = _build(with_sem=True, signal_before_drain=True)
    assert nr.trace(kernel)
    rep = nr.check_protocol(kernel)
    assert rep["status"] == "Passed"
    assert _nondet(rep), "signal-before-drain must still warn non-determinism"
