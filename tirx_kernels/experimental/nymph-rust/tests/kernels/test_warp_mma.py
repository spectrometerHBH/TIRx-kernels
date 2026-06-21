"""Warp-level SM80 MMA (`mma.sync.aligned.m16n8k{8,16}.row.col.f32.{abt}.{abt}.f32`).

End-to-end: lay out A/B in SMEM (bf16 OR f16), ldmatrix-load them into the mma
fragment layout, `mma_sync` (ab_dtype = the PTX operand type), scatter D via the
accumulator fragment layout, and check D = A·Bᵀ. Exercises the WarpMma IR node, its
coupling with ldmatrix, and first-class bf16/f16 operand decode.
"""

import numpy as np
import pytest

import nymph_rs as nr
from nymph_rs import DType, IRBuilder, MemorySpace, TensorSlice

ml_dtypes = pytest.importorskip("ml_dtypes")
bf16 = ml_dtypes.bfloat16

_DT = {DType.BF16: bf16, DType.F16: np.float16}


def _sl(t, o, s):
    return TensorSlice(tensor=t, offsets=o, shape=s)


@pytest.mark.parametrize("dt", [DType.BF16, DType.F16], ids=["bf16", "f16"])
@pytest.mark.parametrize("K", [16, 8], ids=["m16n8k16", "m16n8k8"])
def test_warp_mma_matches_reference(K, dt):
    M, N = 16, 8
    npdt = _DT[dt]
    rng = np.random.default_rng(1)
    A = (rng.normal(size=(M, K)) * 0.5).astype(npdt)
    B = (rng.normal(size=(N, K)) * 0.5).astype(npdt)
    d_ref = A.astype(np.float32) @ B.astype(np.float32).T

    n_kt = K // 8
    A_sm = np.zeros((32, 8), npdt)
    for kt in range(n_kt):
        for mt in range(2):
            A_sm[(mt + 2 * kt) * 8 : (mt + 2 * kt) * 8 + 8] = A[mt * 8 : mt * 8 + 8, kt * 8 : kt * 8 + 8]
    B_sm = np.zeros((32, 8), npdt)
    for kt in range(n_kt):
        for n8 in range(8):
            for k8 in range(8):
                B_sm[kt * 8 + n8, k8] = B[n8, kt * 8 + k8]

    n_a, n_b = M * K // 64, N * K // 64
    k = IRBuilder("warp_mma", num_warps=4, smem_size_bytes=32 * 8 * 2 * 2 + 64, launch_shape=(1,))
    ag = k.arg(space=MemorySpace.GMEM, dtype=dt, shape=(32, 8))
    bg = k.arg(space=MemorySpace.GMEM, dtype=dt, shape=(32, 8))
    og = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(M, N))
    asm = k.tensor(space=MemorySpace.SMEM, dtype=dt, shape=(32, 8), byte_offset=0)
    bsm = k.tensor(space=MemorySpace.SMEM, dtype=dt, shape=(32, 8), byte_offset=32 * 8 * 2)
    af = k.tensor(space=MemorySpace.REG, dtype=DType.U32, shape=(n_a,))
    bfr = k.tensor(space=MemorySpace.REG, dtype=DType.U32, shape=(n_b,))
    cf = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(4,))
    df = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(4,))
    rb = k.tensor(space=MemorySpace.REG, dtype=dt, shape=(1,))
    with k.role(warp=0):
        tid = k.tid_in_wg()
        lane = tid % 32
        with k.if_(tid < 32):
            for c in range(8):
                k.reg_load(rb, _sl(ag, (tid, c), (1, 1)))
                k.reg_store(_sl(asm, (tid, c), (1, 1)), rb)
                k.reg_load(rb, _sl(bg, (tid, c), (1, 1)))
                k.reg_store(_sl(bsm, (tid, c), (1, 1)), rb)
        k.warp_sync()
        k.ldmatrix(af, _sl(asm, (tid, 0), (1, 8)), num=n_a, trans=False)
        k.ldmatrix(bfr, _sl(bsm, (tid, 0), (1, 8)), num=n_b, trans=True)
        k.reg_fill(cf, 0.0)
        k.mma_sync(df, af, bfr, cf, m=M, n=N, k=K, ab_dtype=dt)
        for ri in range(4):
            k.reg_store(
                _sl(og, ((ri // 2) * 8 + lane // 4, 2 * (lane % 4) + ri % 2), (1, 1)),
                _sl(df, (ri,), (1,)),
            )
    kernel = k.build()
    res = nr.interpret(kernel, {ag: A_sm, bg: B_sm})
    d = np.asarray(res[og.id], dtype=np.float32).reshape(M, N)
    # 16-bit inputs, f32 accumulate → exact vs the numpy f32 matmul of the 16-bit values.
    np.testing.assert_allclose(d, d_ref, atol=1e-4, rtol=1e-3)
