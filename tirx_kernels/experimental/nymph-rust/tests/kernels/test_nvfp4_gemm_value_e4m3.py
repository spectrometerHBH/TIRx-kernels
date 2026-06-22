"""Bit-exact value check for the nvfp4 GEMM kernel's e4m3-SF datapath.

The kernel (``python/nymph_rs/kernels/nvfp4_gemm.py``) takes the block-scale
factors as raw ``e4m3`` bytes laid out ``(M, K//16)`` / ``(N, K//16)`` — one
scale per 16 contiguous K-elements, exactly canon's ``SFA_in``/``SFB_in`` slice
(NOT the packed-u32 scale cells the fp8-blockwise port uses). This test feeds
that layout and checks the interpreter's block-scaled product bit-for-bit.

Self-consistent quantization (like the fp8 port's UE8M0 cast, not a vendor
library): small e2m1 operand codes, power-of-two e4m3 block scales, and a
power-of-two alpha. Every product and partial sum is then exact in f32 and both
sides round bf16 identically, so the interpreter must match the numpy reference
with ZERO mismatches.
"""

from __future__ import annotations

import numpy as np
import nymph_rs as nr
import pytest
from nymph_rs.kernels import build_nvfp4_gemm
from nymph_rs.kernels.nvfp4_gemm import SF_BLOCK, NvFp4GemmConfig

# The 16 e2m1 magnitudes: {0, .5, 1, 1.5, 2, 3, 4, 6} and their negatives.
_E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32
)


def _pack_fp4(codes: np.ndarray) -> np.ndarray:
    """(R, K) e2m1 nibble codes -> (R, K//2) u8: element 2i in the low nibble,
    2i+1 in the high nibble."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)


def _round_bf16(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def _prepare(m: int, n: int, k: int, alpha: float, seed: int):
    """Self-consistent nvfp4 inputs + the exact bf16 reference.

    Returns packed A/B (u8, (R, K//2)), the per-16-K-block scale factors as the
    kernel's ``f8e4m3`` (R, K//16) args — passed as the exact POWER-OF-TWO f32
    values (the interpreter coerces an f8e4m3 arg from float32 and rounds to
    e4m3, so a power of two round-trips bit-exactly) — and the bf16 reference."""
    rng = np.random.default_rng(seed)
    # e2m1 codes spanning small magnitudes and both signs (codes 0..4 and 8..12
    # = {0,.5,1,1.5,2} and negatives) keep every partial sum exact in f32.
    nblk = k // SF_BLOCK

    def codes(rows):
        pos = rng.integers(0, 5, size=(rows, k))
        sgn = rng.integers(0, 2, size=(rows, k)) << 3
        return (pos | sgn).astype(np.uint8)

    a_codes = codes(m)
    b_codes = codes(n)
    # Power-of-two exponents in [-1, 1]: 2^p is an exact e4m3 normal (zero
    # mantissa), so the f32->e4m3 coercion is lossless.
    a_p = rng.integers(-1, 2, size=(m, nblk))
    b_p = rng.integers(-1, 2, size=(n, nblk))
    a_scale = (2.0**a_p).astype(np.float32)  # (M, K//16) f32 powers of two
    b_scale = (2.0**b_p).astype(np.float32)  # (N, K//16)

    a = _E2M1[a_codes]
    b = _E2M1[b_codes]
    a_s = (a.reshape(m, nblk, SF_BLOCK) * a_scale[:, :, None]).reshape(m, k).astype(np.float32)
    b_s = (b.reshape(n, nblk, SF_BLOCK) * b_scale[:, :, None]).reshape(n, k).astype(np.float32)
    ref = _round_bf16(alpha * (a_s @ b_s.T))
    # SF args: the f32 scale values, viewed as f8e4m3 (M, K//16) / (N, K//16) —
    # exactly the kernel's sfa_gmem/sfb_gmem declaration.
    return _pack_fp4(a_codes), _pack_fp4(b_codes), a_scale, b_scale, ref


# Shapes × alphas: 1 / 2 / 4 k-tiles, 1 / 2 N bands, 1 / 2 M pairs, and a
# non-unit power-of-two alpha — all cell-exact.
_CASES = [
    (256, 256, 256, 1.0),  # 1 k-tile, 1 tile
    (256, 512, 512, 1.0),  # 2 k-tiles, 2 N bands
    (512, 256, 512, 0.5),  # 2 M pairs, non-unit alpha
    (512, 512, 1024, 2.0),  # 4 k-tiles, 2x2 tiles
    (1024, 1024, 1024, 0.25),  # larger square
]


@pytest.mark.parametrize("m,n,k,alpha", _CASES, ids=[f"{m}x{n}x{k}@{a}" for m, n, k, a in _CASES])
def test_nvfp4_gemm_value_e4m3_sf_cell_exact(m, n, k, alpha):
    kernel = build_nvfp4_gemm(NvFp4GemmConfig(m=m, n=n, k=k, alpha=alpha, launch_shape=(2,)))
    a_t, b_t, sfa_t, sfb_t, d_t = kernel.args
    a_q, b_q, sfa, sfb, ref = _prepare(m, n, k, alpha, seed=m + n + k)
    out = nr.interpret(kernel, {a_t: a_q, b_t: b_q, sfa_t: sfa, sfb_t: sfb})
    d = np.asarray(out[d_t.id], dtype=np.float32).reshape(m, n)
    assert np.isfinite(d).all()
    assert int((d != ref).sum()) == 0, f"{int((d != ref).sum())} mismatches"


# The no_overlap epilogue (canon's OVERLAP_EPI=False, used per-shape e.g. at 4096) is a distinct
# TMEM-drain schedule over the SAME accumulator; it must be cell-exact too (not a special path).
_NO_OVERLAP_CASES = [
    (512, 512, 1024, 1.0),  # 4 k-tiles
    (1024, 1024, 1024, 0.5),  # larger square, non-unit alpha
]


@pytest.mark.parametrize(
    "m,n,k,alpha",
    _NO_OVERLAP_CASES,
    ids=[f"{m}x{n}x{k}@{a}-noov" for m, n, k, a in _NO_OVERLAP_CASES],
)
def test_nvfp4_gemm_value_no_overlap_epilogue(m, n, k, alpha):
    cfg = NvFp4GemmConfig(
        m=m, n=n, k=k, alpha=alpha, launch_shape=(2,), epilogue="no_overlap", epi_tile=32
    )
    kernel = build_nvfp4_gemm(cfg)
    a_t, b_t, sfa_t, sfb_t, d_t = kernel.args
    a_q, b_q, sfa, sfb, ref = _prepare(m, n, k, alpha, seed=m + n + k)
    out = nr.interpret(kernel, {a_t: a_q, b_t: b_q, sfa_t: sfa, sfb_t: sfb})
    d = np.asarray(out[d_t.id], dtype=np.float32).reshape(m, n)
    assert np.isfinite(d).all()
    assert int((d != ref).sum()) == 0, f"{int((d != ref).sum())} mismatches (no_overlap)"
