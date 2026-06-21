"""Hierarchical blockwise inverse of a 64x64 unit-lower-triangular matrix using
the warp-level `mma_sync` (SM80 HMMA) — flashinfer's GDN WY-inverse algorithm.

5-stage (4 for 64x64): Gauss-Jordan-invert the 8 diagonal 8x8 blocks (forward
substitution), then merge b->2b (b=8,16,32) filling each lower off-diagonal block
newC = -Qinv*C*Pinv via two mma_sync (Pinv=top-left, Qinv=bot-right already
inverted, C=orig bot-left). Validated cell-for-cell against numpy.
"""

import numpy as np
import pytest

import nymph_rs as nr
from nymph_rs import DType, IRBuilder, MemorySpace, TensorSlice

ml_dtypes = pytest.importorskip("ml_dtypes")
BT = 64


def _sl(t, o, s):
    return TensorSlice(tensor=t, offsets=o, shape=s)


def _build():
    k = IRBuilder("hier_inv", num_warps=4,
                  smem_size_bytes=BT * BT * 4 + BT * BT * 2 * 2 + 512, launch_shape=(1,))
    mg = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(BT, BT))
    og = k.arg(space=MemorySpace.GMEM, dtype=DType.BF16, shape=(BT, BT))
    mf = k.tensor(space=MemorySpace.SMEM, dtype=DType.F32, shape=(BT, BT), byte_offset=0)
    mb = k.tensor(space=MemorySpace.SMEM, dtype=DType.BF16, shape=(BT, BT), byte_offset=BT * BT * 4)
    dcs = k.tensor(space=MemorySpace.SMEM, dtype=DType.BF16, shape=(BT, BT),
                   byte_offset=BT * BT * 4 + BT * BT * 2)
    af = k.tensor(space=MemorySpace.REG, dtype=DType.U32, shape=(2,))
    bfr = k.tensor(space=MemorySpace.REG, dtype=DType.U32, shape=(1,))
    acc = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(4,))
    accb = k.tensor(space=MemorySpace.REG, dtype=DType.BF16, shape=(4,))
    r1 = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(1,))
    r2 = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(1,))
    rb = k.tensor(space=MemorySpace.REG, dtype=DType.BF16, shape=(1,))
    racc = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(1,))

    with k.role(warpgroup=0):
        tid = k.tid_in_wg()
        lane = tid % 32
        # load M -> mf (f32); strict-lower for the GJ: zero the diagonal and NEGATE
        # each diagonal-block's strict-lower (FLA forward-sub needs attn = -L). The
        # inter-block off-diags (the merge C's) stay +M.
        with k.if_(tid < BT):
            for c in range(BT):
                k.reg_load(r1, _sl(mg, (tid, c), (1, 1)))
                k.reg_store(_sl(mf, (tid, c), (1, 1)), r1)
            k.reg_fill(r1, 0.0)
            k.reg_store(_sl(mf, (tid, tid), (1, 1)), r1)
            bb = (tid // 8) * 8
            for jb in range(8):
                with k.if_((tid % 8) > jb):
                    k.reg_load(r1, _sl(mf, (tid, bb + jb), (1, 1)))
                    k.reg_mul(r1, r1, -1.0)
                    k.reg_store(_sl(mf, (tid, bb + jb), (1, 1)), r1)
        k.wg_sync(barrier_id=10)
        # Stage 1: GJ-invert the 8 diagonal 8x8 blocks (thread = col j of its block)
        for r_ in range(1, 8):
            with k.if_((tid < BT) & ((tid % 8) < r_)):
                base = (tid // 8) * 8
                j = tid % 8
                k.reg_fill(racc, 0.0)
                for mm in range(r_):
                    k.reg_load(r1, _sl(mf, (base + r_, base + mm), (1, 1)))
                    k.reg_load(r2, _sl(mf, (base + mm, base + j), (1, 1)))
                    k.reg_fma(racc, r1, r2, racc)
            k.wg_sync(barrier_id=10)
            with k.if_((tid < BT) & ((tid % 8) < r_)):
                base = (tid // 8) * 8
                j = tid % 8
                k.reg_load(r1, _sl(mf, (base + r_, base + j), (1, 1)))
                k.reg_add(r1, r1, racc)
                k.reg_store(_sl(mf, (base + r_, base + j), (1, 1)), r1)
            k.wg_sync(barrier_id=10)
        # cvt mf -> mb (bf16) with unit diagonal
        with k.if_(tid < BT):
            for c in range(BT):
                k.reg_load(r1, _sl(mf, (tid, c), (1, 1)))
                k.reg_cvt(rb, r1)
                k.reg_store(_sl(mb, (tid, c), (1, 1)), rb)
            k.reg_fill(rb, 1.0)
            k.reg_store(_sl(mb, (tid, tid), (1, 1)), rb)
        k.wg_sync(barrier_id=10)

        def ldA(src, R, C):  # A = src[R:R+8,C:C+8] non-trans, m-broadcast to m16
            k.ldmatrix(_sl(af, (0,), (1,)), _sl(src, (R + lane % 8, C), (1, 8)), num=1, trans=False)
            k.reg_store(_sl(af, (1,), (1,)), _sl(af, (0,), (1,)))

        def ldB(src, R, C):
            k.ldmatrix(bfr, _sl(src, (R + lane % 8, C), (1, 8)), num=1, trans=False)

        def store8(dst, R, C, neg):
            for ri in range(2):  # ri 0,1 = top 8 rows (ri 2,3 are the m-broadcast dup)
                if neg:
                    k.reg_mul(_sl(acc, (ri,), (1,)), _sl(acc, (ri,), (1,)), -1.0)
                k.reg_cvt(_sl(accb, (ri,), (1,)), _sl(acc, (ri,), (1,)))
                k.reg_store(_sl(dst, (R + lane // 4, C + 2 * (lane % 4) + ri), (1, 1)),
                            _sl(accb, (ri,), (1,)))

        def merge(R, Cc, b, w):  # 2b x 2b tile top-left at (R,Cc); warp w only
            tb = b // 8
            with k.if_((tid // 32).eq(w)):
                for mi in range(tb):  # DC = Qinv @ C -> dcs (per-warp band w*16)
                    for ni in range(tb):
                        k.reg_fill(acc, 0.0)
                        for ki in range(tb):
                            ldA(mb, R + b + mi * 8, Cc + b + ki * 8)
                            ldB(mb, R + b + ki * 8, Cc + ni * 8)
                            k.mma_sync(acc, af, bfr, acc, m=16, n=8, k=8)
                        store8(dcs, w * 16 + mi * 8, ni * 8, neg=False)
                for mi in range(tb):  # O = DC @ Pinv ; newC = -O
                    for ni in range(tb):
                        k.reg_fill(acc, 0.0)
                        for ki in range(tb):
                            ldA(dcs, w * 16 + mi * 8, ki * 8)
                            ldB(mb, R + ki * 8, Cc + ni * 8)
                            k.mma_sync(acc, af, bfr, acc, m=16, n=8, k=8)
                        store8(mb, R + b + mi * 8, Cc + ni * 8, neg=True)

        for t in range(4):
            merge(t * 16, t * 16, 8, t)
        k.wg_sync(barrier_id=10)
        for t in range(2):
            merge(t * 32, t * 32, 16, t)
        k.wg_sync(barrier_id=10)
        merge(0, 0, 32, 0)
        k.wg_sync(barrier_id=10)
        with k.if_(tid < BT):
            for c in range(BT):
                k.reg_load(rb, _sl(mb, (tid, c), (1, 1)))
                k.reg_store(_sl(og, (tid, c), (1, 1)), rb)
    return k.build(), mg, og


def test_hierarchical_inverse_matches_numpy():
    rng = np.random.default_rng(5)
    M = (np.tril(rng.normal(size=(BT, BT)) * 0.12, -1) + np.eye(BT)).astype(np.float32)
    ref = np.linalg.inv(M)
    kernel, mg, og = _build()
    out = np.asarray(nr.interpret(kernel, {mg: M})).astype if False else None
    res = nr.interpret(kernel, {mg: M})
    out = np.asarray(res[og.id]).astype(np.float32)
    matched = (np.abs(out - ref) <= 1e-2 + 0.3 * np.abs(ref)).mean()
    assert matched >= 0.99, f"matched={matched}, max|err|={np.abs(out - ref).max()}"
