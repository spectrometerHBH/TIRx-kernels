"""NVFP4 GEMM nymph port: build, trace, protocol, and cell-exact value tests.

The value tests use a self-consistent quantization (like the fp8 port's UE8M0
cast, not a vendor library): small e2m1 operand codes, power-of-two e4m3 block
scales, and a power-of-two alpha. Every product and partial sum is then exact in
f32 and both sides round bf16 identically, so the interpreter must match the
numpy reference bit-for-bit (zero mismatches).
"""

from __future__ import annotations

import numpy as np
import nymph_rs as nr
import pytest
from nymph_rs.kernels import build_nvfp4_gemm, nvfp4_task_config
from nymph_rs.kernels.nvfp4_gemm import NVFP4_CONFIGS_SUPPORTED, NvFp4GemmConfig

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


def _prepare(m: int, n: int, k: int, alpha: float, seed: int = 0):
    """Self-consistent nvfp4 inputs + the exact bf16 reference.

    The scale factors are the kernel's ``f8e4m3`` (R, K//16) args, passed as the
    exact POWER-OF-TWO f32 values (the interpreter coerces an f8e4m3 arg from
    float32 and rounds to e4m3, so a power of two round-trips bit-exactly)."""
    rng = np.random.default_rng(seed)
    # e2m1 codes spanning small magnitudes and both signs (codes 0..4 and 8..12
    # = {0,.5,1,1.5,2} and negatives) keep every partial sum exact in f32.
    pos = rng.integers(0, 5, size=(m, k))
    sgn = rng.integers(0, 2, size=(m, k)) << 3
    a_codes = (pos | sgn).astype(np.uint8)
    pos = rng.integers(0, 5, size=(n, k))
    sgn = rng.integers(0, 2, size=(n, k)) << 3
    b_codes = (pos | sgn).astype(np.uint8)
    a_p = rng.integers(-1, 2, size=(m, k // 16))
    b_p = rng.integers(-1, 2, size=(n, k // 16))

    a = _E2M1[a_codes]
    b = _E2M1[b_codes]
    a_scale = (2.0**a_p).astype(np.float32)  # (M, K//16) f32 powers of two
    b_scale = (2.0**b_p).astype(np.float32)  # (N, K//16)
    a_s = (a.reshape(m, k // 16, 16) * a_scale[:, :, None]).reshape(m, k).astype(np.float32)
    b_s = (b.reshape(n, k // 16, 16) * b_scale[:, :, None]).reshape(n, k).astype(np.float32)
    ref = _round_bf16(alpha * (a_s @ b_s.T))
    return _pack_fp4(a_codes), _pack_fp4(b_codes), a_scale, b_scale, ref


def test_nvfp4_gemm_protocol_passes_canonical_task_shape():
    kernel = build_nvfp4_gemm(nvfp4_task_config(6))
    report = nr.check_protocol(kernel)
    assert report["status"] == "Passed", report["diagnostics"][:2]
    for entry in report["pass_summary"]:
        assert entry["status"] in ("Passed", "Skipped"), entry


def test_nvfp4_gemm_value_canonical_task_shape():
    cfg = nvfp4_task_config(2)
    m, n, k = cfg.m, cfg.n, cfg.k
    kernel = build_nvfp4_gemm(cfg)
    a_t, b_t, sfa_t, sfb_t, d_t = kernel.args
    a_q, b_q, sfa_pack, sfb_pack, ref = _prepare(m, n, k, cfg.alpha)
    out = nr.interpret(kernel, {a_t: a_q, b_t: b_q, sfa_t: sfa_pack, sfb_t: sfb_pack})
    d = np.asarray(out[d_t.id], dtype=np.float32).reshape(m, n)
    assert np.isfinite(d).all()
    assert int((d != ref).sum()) == 0


@pytest.mark.parametrize("cfg", NVFP4_CONFIGS_SUPPORTED, ids=lambda c: c["label"])
def test_nvfp4_gemm_protocol_passes(cfg):
    kernel = build_nvfp4_gemm(
        NvFp4GemmConfig(m=cfg["m"], n=cfg["n"], k=cfg["k"], launch_shape=(2,))
    )
    report = nr.check_protocol(kernel)
    assert report["status"] == "Passed", report["diagnostics"]


# Per-warpgroup register budgets (canon's INVARIANT-I1b setmaxnreg) — the 1024 lever.
# setmaxnreg is a pure register-allocation hint (no value / protocol / rendezvous effect),
# so the protocol must still pass with the consumer cap alone and with the producer.inc /
# consumer.dec pair.
@pytest.mark.parametrize(
    "mnr_epi,mnr_prod", [(96, None), (96, 152)], ids=["epi96", "epi96-prod152"]
)
def test_nvfp4_gemm_setmaxnreg_protocol_passes(mnr_epi, mnr_prod):
    kernel = build_nvfp4_gemm(
        NvFp4GemmConfig(
            m=1024,
            n=1024,
            k=1024,
            launch_shape=(2,),
            epi_tile=32,
            maxnreg_epilogue=mnr_epi,
            maxnreg_producer=mnr_prod,
        )
    )
    report = nr.check_protocol(kernel)
    assert report["status"] == "Passed", report["diagnostics"]


# A few shapes × alphas exercising multiple M pairs, N bands, k-tiles, and a
# non-unit power-of-two alpha — all cell-exact.
_VALUE_CASES = [
    (256, 256, 256, 1.0),
    (256, 512, 512, 1.0),
    (512, 256, 512, 0.5),
    (512, 512, 1024, 2.0),
    (1024, 1024, 1024, 0.25),
]


@pytest.mark.parametrize(
    "m,n,k,alpha", _VALUE_CASES, ids=[f"{m}x{n}x{k}@{a}" for m, n, k, a in _VALUE_CASES]
)
def test_nvfp4_gemm_value_cell_exact(m, n, k, alpha):
    kernel = build_nvfp4_gemm(NvFp4GemmConfig(m=m, n=n, k=k, alpha=alpha, launch_shape=(2,)))
    a_t, b_t, sfa_t, sfb_t, d_t = kernel.args
    a_q, b_q, sfa_pack, sfb_pack, ref = _prepare(m, n, k, alpha, seed=m + n + k)
    out = nr.interpret(kernel, {a_t: a_q, b_t: b_q, sfa_t: sfa_pack, sfb_t: sfb_pack})
    d = np.asarray(out[d_t.id], dtype=np.float32).reshape(m, n)
    assert np.isfinite(d).all()
    assert int((d != ref).sum()) == 0


# The dynamic-SMEM-pool variant (canon's T.SMEMPool() allocation form, gated to the
# small squares). It is purely the SMEM allocation FORM — the physical byte layout is
# unchanged (the buffers alias the pool at the same IR offsets) — so the protocol and the
# bit-exact value model must be identical to the default static-buffer form.
@pytest.mark.parametrize(
    "m,n,k,alpha", _VALUE_CASES, ids=[f"{m}x{n}x{k}@{a}" for m, n, k, a in _VALUE_CASES]
)
def test_nvfp4_gemm_smem_pool_value_cell_exact(m, n, k, alpha):
    kernel = build_nvfp4_gemm(
        NvFp4GemmConfig(m=m, n=n, k=k, alpha=alpha, launch_shape=(2,), smem_pool=True)
    )
    assert kernel.smem_pool
    a_t, b_t, sfa_t, sfb_t, d_t = kernel.args
    a_q, b_q, sfa_pack, sfb_pack, ref = _prepare(m, n, k, alpha, seed=m + n + k)
    out = nr.interpret(kernel, {a_t: a_q, b_t: b_q, sfa_t: sfa_pack, sfb_t: sfb_pack})
    d = np.asarray(out[d_t.id], dtype=np.float32).reshape(m, n)
    assert np.isfinite(d).all()
    assert int((d != ref).sum()) == 0


@pytest.mark.parametrize("cfg", NVFP4_CONFIGS_SUPPORTED, ids=lambda c: c["label"])
def test_nvfp4_gemm_smem_pool_protocol_passes(cfg):
    kernel = build_nvfp4_gemm(
        NvFp4GemmConfig(m=cfg["m"], n=cfg["n"], k=cfg["k"], launch_shape=(2,), smem_pool=True)
    )
    report = nr.check_protocol(kernel)
    assert report["status"] == "Passed", report["diagnostics"]


def test_nvfp4_gemm_smem_pool_emits_dynamic_pool():
    # The pool variant emits canon's `T.SMEMPool()` + `pool.alloc(... scope="shared.dyn")`,
    # while the default form emits the static `T.alloc_buffer(scope="shared")`. The big
    # data buffers must move to the dynamic pool; the small mbar buffers stay static.
    static = nr.kernel_to_tirx_source(
        build_nvfp4_gemm(NvFp4GemmConfig(m=1024, n=1024, k=1024, launch_shape=(2,)))
    )
    pool = nr.kernel_to_tirx_source(
        build_nvfp4_gemm(NvFp4GemmConfig(m=1024, n=1024, k=1024, launch_shape=(2,), smem_pool=True))
    )
    assert "T.SMEMPool()" not in static
    assert "T.alloc_shared" in static and 'scope="shared.dyn"' not in static
    assert "T.SMEMPool()" in pool
    assert "pool.commit()" in pool
    assert 'scope="shared.dyn"' in pool
    # The whole shared window (data buffers AND the mbar/tmem_addr buffers) goes through
    # the dynamic pool — there is no separate static T.alloc_shared region (mixing static
    # `shared` with the `shared.dyn` pool misaligns the pool base; everything in one pool
    # keeps the 1024-byte alignment, exactly like canon).
    assert "T.alloc_shared" not in pool
    assert 'pool.alloc([5], "uint64"' in pool  # mbars come from the pool
    assert "tmem_addr = pool.alloc(" in pool


@pytest.mark.parametrize("size", [1024, 2048, 4096, 8192, 16384])
def test_nvfp4_gemm_builds_every_config(size):
    # Every TIRx CONFIGS square builds. Protocol is checked up to 4096^3 here;
    # the k=16384 protocol path is covered by the canonical-task-shape test, so
    # the two largest squares are build-only to keep the suite fast.
    kernel = build_nvfp4_gemm(NvFp4GemmConfig(m=size, n=size, k=size, launch_shape=(2,)))
    assert kernel.args, "kernel built with no arguments"
    if size <= 4096:
        report = nr.check_protocol(kernel)
        assert report["status"] == "Passed", report["diagnostics"][:2]
