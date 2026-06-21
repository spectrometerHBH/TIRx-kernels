"""TMA reduce-add (`cp.reduce.async.bulk...add.f32`) — the IR node the flash_bwd_sm100
dQ accumulation needs (reduce warps atomically add the per-tile dQ into global dQaccum).
Value-mode accumulates `dst += src` (f32); trace/protocol treat it as a GMEM-output bulk
async RMW. Approved IR extension."""

import numpy as np
import pytest

import nymph_rs as nr
from nymph_rs import DType, FenceKind, FenceScope, IRBuilder, MemorySpace, TensorSlice


def _sl(t, o, s):
    return TensorSlice(tensor=t, offsets=o, shape=s)


@pytest.mark.parametrize("n_accum", [1, 2, 3])
def test_tma_reduce_add_accumulates(n_accum):
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
        k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)  # generic SMEM write -> async TMA read
        with k.if_(tid.eq(0)):
            for _ in range(n_accum):
                k.tma_reduce_add(acc_g, _sl(sm, (0, 0), (M, N)), coords=(0, 0), shape=(M, N))
                k.cp_async_bulk_commit_group()
                k.cp_async_bulk_wait_group_read(0)
    kernel = k.build()
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"
    src = np.random.default_rng(0).normal(size=(M, N)).astype(np.float32)
    res = nr.interpret(kernel, {src_g: src})  # acc_g is output-only -> auto zero-init
    out = np.asarray(res[acc_g.id], dtype=np.float32).reshape(M, N)
    np.testing.assert_allclose(out, n_accum * src, atol=0, rtol=0)
