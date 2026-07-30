"""Build + value tests for the nvfp4 block-scaled GEMM port.

Value bar: self-consistent quantization (like the fp8 port's UE8M0 cast, not a
vendor library) — small e2m1 operand codes, power-of-two e4m3 block scales, and
a power-of-two alpha. Every product and partial sum is then exact in f32 and
both sides round bf16 identically, so the interpreter must match the numpy
reference with ZERO mismatches.

"""

from __future__ import annotations

import numpy as np
import nymph_rs as nr
import pytest
from nymph_rs.kernels import build_nvfp4_gemm
from nymph_rs.kernels.nvfp4_gemm import NVFP4_CONFIGS_SUPPORTED, SF_BLOCK, NvFp4GemmConfig

# The 16 e2m1 magnitudes: {0, .5, 1, 1.5, 2, 3, 4, 6} and their negatives.
_E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32
)


def _pack_fp4(codes: np.ndarray) -> np.ndarray:
    """(R, K) e2m1 nibble codes -> (R, K//2) u8: element 2i in the high nibble,
    2i+1 in the low nibble."""
    return ((codes[:, 0::2] << 4) | codes[:, 1::2]).astype(np.uint8)


def _round_bf16(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def _pack_sf(scale_p: np.ndarray) -> np.ndarray:
    """Encode logical scales in the plain physical ``(R//128,K//64,32,16)`` view."""
    rows, scale_k = scale_p.shape
    values = (2.0**scale_p).astype(np.float32)
    return (
        values.reshape(rows // 128, 4, 32, scale_k // 4, 4)
        .transpose(0, 3, 2, 1, 4)
        .reshape(rows // 128, scale_k // 4, 32, 16)
    )


def _prepare(m: int, n: int, k: int, alpha: float, seed: int):
    """Self-consistent nvfp4 inputs + the exact bf16 reference."""
    rng = np.random.default_rng(seed)
    nblk = k // SF_BLOCK

    # e2m1 codes spanning small magnitudes and both signs (codes 0..4 and 8..12
    # = {0,.5,1,1.5,2} and negatives) keep every partial sum exact in f32.
    def codes(rows):
        pos = rng.integers(0, 5, size=(rows, k))
        sgn = rng.integers(0, 2, size=(rows, k)) << 3
        return (pos | sgn).astype(np.uint8)

    a_codes = codes(m)
    b_codes = codes(n)
    # Power-of-two exponents in [-1, 1]: 2^p is an exact e4m3 normal (zero
    # mantissa), so the byte packing is lossless.
    a_p = rng.integers(-1, 2, size=(m, nblk))
    b_p = rng.integers(-1, 2, size=(n, nblk))

    a = _E2M1[a_codes]
    b = _E2M1[b_codes]
    a_s = (a.reshape(m, nblk, SF_BLOCK) * (2.0**a_p)[:, :, None]).reshape(m, k)
    b_s = (b.reshape(n, nblk, SF_BLOCK) * (2.0**b_p)[:, :, None]).reshape(n, k)
    ref = _round_bf16(alpha * (a_s.astype(np.float32) @ b_s.astype(np.float32).T))
    return _pack_fp4(a_codes), _pack_fp4(b_codes), _pack_sf(a_p), _pack_sf(b_p), ref


@pytest.mark.parametrize(
    "cfg", NVFP4_CONFIGS_SUPPORTED, ids=[c["label"] for c in NVFP4_CONFIGS_SUPPORTED]
)
def test_nvfp4_gemm_builds_and_validates(cfg):
    kernel = build_nvfp4_gemm(
        NvFp4GemmConfig(m=cfg["m"], n=cfg["n"], k=cfg["k"], launch_shape=(2,))
    )
    kernel.validate()


@pytest.mark.parametrize(
    "cfg", NVFP4_CONFIGS_SUPPORTED, ids=[c["label"] for c in NVFP4_CONFIGS_SUPPORTED]
)
def test_nvfp4_gemm_protocol_all_supported_shapes(cfg):
    # Every supported shape protocol-checks, not just the default config.
    kernel = build_nvfp4_gemm(
        NvFp4GemmConfig(m=cfg["m"], n=cfg["n"], k=cfg["k"], launch_shape=(2,))
    )
    assert nr.check_protocol(kernel)["status"] == "Passed"


# Shapes x alphas: 1 / 2 / 4 k-tiles, 1 / 2 N bands, 1 / 2 M pairs, and a
# non-unit power-of-two alpha — all cell-exact.
_VALUE_CASES = [
    (256, 256, 256, 1.0),  # 1 k-tile, 1 tile
    (256, 512, 512, 1.0),  # 2 k-tiles, 2 N bands
    (512, 256, 512, 0.5),  # 2 M pairs, non-unit alpha
    (512, 512, 1024, 2.0),  # 4 k-tiles, 2x2 tiles
]


@pytest.mark.parametrize(
    "m,n,k,alpha", _VALUE_CASES, ids=[f"{m}x{n}x{k}@{a}" for m, n, k, a in _VALUE_CASES]
)
def test_nvfp4_gemm_value_e4m3_sf_cell_exact(m, n, k, alpha):
    kernel = build_nvfp4_gemm(NvFp4GemmConfig(m=m, n=n, k=k, alpha=alpha, launch_shape=(2,)))
    a_t, b_t, sfa_t, sfb_t, d_t = kernel.args
    a_q, b_q, sfa, sfb, ref = _prepare(m, n, k, alpha, seed=m + n + k)
    out = nr.interpret(kernel, {a_t: a_q, b_t: b_q, sfa_t: sfa, sfb_t: sfb})
    d = np.asarray(out[d_t.id], dtype=np.float32).reshape(m, n)
    assert np.isfinite(d).all()
    assert int((d != ref).sum()) == 0, f"{int((d != ref).sum())} mismatches"
