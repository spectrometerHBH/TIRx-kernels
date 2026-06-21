"""tcgen05_mma with an operand sourced directly from TMEM (the f32 accumulator).

FlashInfer's GDN GEMM3/4/5/6/7 read operands from TMEM (e.g. the recurrent state
S). nymph's MMA value model reads every operand as logical f32, so an f32 TMEM
operand is exact: here A is a bf16 SMEM tile and B is the f32 state S held in
TMEM, read transposed (S^T) — D = A @ S must match numpy exactly.
"""

import numpy as np
import pytest

import nymph_rs as nr
from nymph_rs import (
    DType,
    FenceKind,
    FenceScope,
    IRBuilder,
    MBarKind,
    MemorySpace,
    TensorSlice,
    TmemLayout,
    TmemLayoutKind,
)

ml_dtypes = pytest.importorskip("ml_dtypes")
bf16 = ml_dtypes.bfloat16


def _sl(t, o, s):
    return TensorSlice(tensor=t, offsets=o, shape=s)


def test_tcgen05_mma_tmem_operand():
    M = N = K = 128
    rng = np.random.default_rng(7)
    A = (rng.normal(size=(M, K)) * 0.3).astype(bf16)       # SMEM bf16 operand
    S = (rng.normal(size=(K, N)) * 0.3).astype(np.float32)  # TMEM f32 operand (state S[dk,dv])
    d_ref = A.astype(np.float32) @ S

    k = IRBuilder("tmop", num_warps=4, smem_size_bytes=M * K * 2 + 256, launch_shape=(1,))
    ag = k.arg(space=MemorySpace.GMEM, dtype=DType.BF16, shape=(M, K))
    sg = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(K, N))
    og = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(M, N))
    asm = k.tensor(space=MemorySpace.SMEM, dtype=DType.BF16, shape=(M, K), byte_offset=0)
    base = k.tensor(space=MemorySpace.TMEM, dtype=DType.F32, shape=(128, 512),
                    layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=0))
    s_tmem = k.tensor(space=MemorySpace.TMEM, dtype=DType.F32, shape=(128, N),
                      layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=0))
    acc = k.tensor(space=MemorySpace.TMEM, dtype=DType.F32, shape=(128, N),
                   layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=256))
    row = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(N,))
    rb = k.tensor(space=MemorySpace.REG, dtype=DType.BF16, shape=(1,))
    done = k.mbar(kind=MBarKind.TCGEN05, stages=1)
    with k.kernel_init(warp=0):
        k.tmem_alloc(base, n_cols=512)
        k.mbarrier_init(done, count=1)
    with k.role(warpgroup=0):
        tid = k.tid_in_wg()
        with k.if_(tid < M):
            for c in range(K):
                k.reg_load(rb, _sl(ag, (tid, c), (1, 1)))
                k.reg_store(_sl(asm, (tid, c), (1, 1)), rb)
        with k.if_(tid < 128):
            for c in range(N):
                k.reg_load(_sl(row, (c,), (1,)), _sl(sg, (tid, c), (1, 1)))
            k.tcgen05_st(s_tmem, row, num=N, row=0, col=0)
        k.tcgen05_wait_st()
        k.wg_sync(barrier_id=10)
        k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
        with k.role(warp=0):
            for g in range(K // 16):  # B = s_tmem^T = S^T (trans_b slice is [k, n])
                k.tcgen05_mma(acc, _sl(asm, (0, g * 16), (M, 16)),
                              _sl(s_tmem, (g * 16, 0), (16, N)),
                              m=M, n=N, k=16, trans_b=True, accum=(g != 0))
            k.tcgen05_commit(done)
        k.mbarrier_wait(done, phase=0)
        with k.if_(tid < 128):
            k.tcgen05_ld(row, acc, shape="32x32b", num=N, row=0, col=0)
            k.tcgen05_wait_ld()
            for c in range(N):
                k.reg_store(_sl(og, (tid, c), (1, 1)), _sl(row, (c,), (1,)))
    with k.kernel_finalize(warp=0):
        k.tmem_dealloc(base, n_cols=512)
    kernel = k.build()
    res = nr.interpret(kernel, {ag: A, sg: S})
    d = np.asarray(res[og.id], dtype=np.float32).reshape(M, N)
    np.testing.assert_allclose(d, d_ref, atol=1e-3, rtol=1e-3)
