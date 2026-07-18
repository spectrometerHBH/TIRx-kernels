"""TMA reduce-add (`cp.reduce.async.bulk...add.f32`) — the IR node the flash_bwd_sm100
dQ accumulation needs (reduce warps atomically add the per-tile dQ into global dQaccum).
Value-mode accumulates `dst += src` (f32); trace/protocol treat it as a GMEM-output bulk
async RMW. Approved IR extension.

A FLOAT reduce-add is order-dependent (float add is not associative), so the IR makes
the non-determinism opt-in: validate rejects `tma_reduce_add` unless the builder passes
`allow_nondet_reduce=True`; with the flag the checker still warns
`nondeterministic_reduction` (covered by tests/test_gmem_semaphore.py)."""

import numpy as np
import nymph_rs as nr
import pytest
from nymph_rs import DType, FenceKind, FenceScope, IRBuilder, MemorySpace, TensorSlice


def _sl(t, o, s):
    return TensorSlice(tensor=t, offsets=o, shape=s)


def _build(n_accum, allow_nondet_reduce):
    M, N = 8, 16
    k = IRBuilder("tra", num_warps=4, smem_size_bytes=M * N * 4 + 256, launch_shape=(1,))
    src_g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(M, N))
    acc_g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(M, N))  # zero-init output
    sm = k.tensor(space=MemorySpace.SMEM, dtype=DType.F32, shape=(M, N), byte_offset=0)
    r = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(1,))
    with k.role(warp=0):
        tid = k.tid_in_wg()
        with k.if_(tid < M):
            for c in range(N):
                k.reg_load(r, _sl(src_g, (tid, c), (1, 1)))
                k.reg_store(_sl(sm, (tid, c), (1, 1)), r)
        k.warp_sync()
        k.fence(
            kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA
        )  # generic SMEM write -> async TMA read
        with k.if_(tid.eq(0)):
            for _ in range(n_accum):
                k.tma_reduce_add(
                    acc_g,
                    _sl(sm, (0, 0), (M, N)),
                    coords=(0, 0),
                    shape=(M, N),
                    allow_nondet_reduce=allow_nondet_reduce,
                )
                k.cp_async_bulk_commit_group()
                k.cp_async_bulk_wait_group_read(0)
    return k, src_g, acc_g


@pytest.mark.parametrize("n_accum", [1, 2, 3])
def test_tma_reduce_add_accumulates(n_accum):
    k, src_g, acc_g = _build(n_accum, allow_nondet_reduce=True)
    kernel = k.build()
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"
    M, N = 8, 16
    src = np.random.default_rng(0).normal(size=(M, N)).astype(np.float32)
    res = nr.interpret(kernel, {src_g: src})  # acc_g is output-only -> auto zero-init
    out = np.asarray(res[acc_g.id], dtype=np.float32).reshape(M, N)
    np.testing.assert_allclose(out, n_accum * src, atol=0, rtol=0)


def test_tma_reduce_add_float_requires_opt_in():
    # No flag: the f32 (non-integer) reduce-add is order-dependent, so validate
    # fails closed — the checker's `nondeterministic_reduction` is only a warning
    # and must not be the only thing standing between an author and a non-
    # bit-reproducible kernel.
    k, _, _ = _build(1, allow_nondet_reduce=False)
    with pytest.raises(ValueError, match="allow_nondet_reduce"):
        k.build()


def test_tma_reduce_add_opt_in_passes_protocol():
    # With the flag the SAME kernel validates and passes protocol (single-CTA, so
    # no cross-stream reduce -> no nondeterministic_reduction warning here; the
    # warn channel itself is covered by tests/test_gmem_semaphore.py).
    k, _, _ = _build(1, allow_nondet_reduce=True)
    report = nr.check_protocol(k.build())
    assert report["status"] == "Passed", report["diagnostics"][:2]
    assert not any(w["code"] == "nondeterministic_reduction" for w in report["warnings"])
