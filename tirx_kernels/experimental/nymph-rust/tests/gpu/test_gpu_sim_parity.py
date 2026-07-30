"""sim⇄GPU bit-exact parity — the fidelity loop, made measurable.

The nymph CPU value simulator is the semantic authority for the IR. These tests
close the loop "sim passes ⇒ silicon matches sim": the SAME kernel is run
through ``nr.interpret`` and through the tirx codegen + ``tvm.compile`` on a
real GPU with the SAME inputs, and the outputs are compared BITWISE (via the
integer bit pattern, so +0.0/-0.0 would even count as a divergence).

The input recipes keep every intermediate exact in f32, so any divergence is a
REAL semantics gap, not rounding noise:

- nvfp4: small e2m1 codes, power-of-two e4m3 block scales, a power-of-two
  alpha — ``tests/kernels/test_nvfp4_gemm.py``'s ``_prepare``. Every product
  and partial sum is exact in f32 and both sides round the identical f32
  accumulator to bf16 with RNE, so the GPU's bf16 widened back to f32 must
  equal sim's f32 output bit-for-bit.
- fp16: small-integer f16 inputs — ``tests/kernels/test_interpreter_e2e.py``'s
  convention. Products and partial sums are integers well inside 2^24 (exact
  in f32, order-independent), and both sides round the same f32 accumulator to
  f16 with RNE.

Skipped when no CUDA device is present.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tvm = pytest.importorskip("tvm")

import nymph_rs as nr  # noqa: E402
from nymph_rs.kernels import (  # noqa: E402
    build_fp8_blockwise_gemm,
    build_fp16_bf16_gemm,
    build_nvfp4_gemm,
)
from nymph_rs.kernels.fp8_blockwise_gemm import Fp8BlockwiseGemmConfig  # noqa: E402
from nymph_rs.kernels.fp16_bf16_gemm import Fp16Bf16GemmConfig  # noqa: E402
from nymph_rs.kernels.nvfp4_gemm import NvFp4GemmConfig  # noqa: E402

# Reuse the exact-input quantization recipe from the CPU value test (sibling
# dir; pytest's prepend import mode has not necessarily put it on sys.path when
# this module is collected first).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kernels")))
from test_fp8_blockwise_gemm import _prepare as _fp8_prepare  # noqa: E402
from test_nvfp4_gemm import _prepare as _nvfp4_prepare  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]


def _compile_tirx(kernel, prefix):
    """Emit the kernel's TIRx source and compile it for the local GPU — the
    same flow as bench/*._compile_nymph (temp .py + importlib + tvm.compile)."""
    src = nr.kernel_to_tirx_source(kernel)
    path = os.path.join(tempfile.mkdtemp(prefix=prefix), "g.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"nymph_emitted_{prefix}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return tvm.compile(
        tvm.IRModule({"main": mod.main}), tvm.target.Target("cuda"), tir_pipeline="tirx"
    )


def _assert_bit_equal(gpu: np.ndarray, sim: np.ndarray, what: str):
    gpu_bits = gpu.view(np.uint32 if gpu.dtype == np.float32 else np.uint16)
    sim_bits = sim.view(np.uint32 if sim.dtype == np.float32 else np.uint16)
    if np.array_equal(gpu_bits, sim_bits):
        return
    mism = np.argwhere(gpu_bits != sim_bits)
    shown = ", ".join(
        f"{tuple(idx)}: gpu={gpu[tuple(idx)]!r} sim={sim[tuple(idx)]!r}" for idx in mism[:8]
    )
    raise AssertionError(f"{what}: {len(mism)} bitwise mismatches (first: {shown})")


def test_nvfp4_gemm_gpu_matches_sim_bitwise():
    # The 1024 entry of NVFP4_CONFIGS_SUPPORTED with the same alpha and launch
    # the CPU cell-exact value test uses — sim-vs-numpy is already established
    # there; here sim-vs-silicon must be bitwise.
    m = n = k = 1024
    cfg = NvFp4GemmConfig(m=m, n=n, k=k, alpha=0.25, launch_shape=(2,))
    kernel = build_nvfp4_gemm(cfg)
    a_t, b_t, sfa_t, sfb_t, d_t = kernel.args
    a_q, b_q, sfa, sfb, _ref = _nvfp4_prepare(m, n, k, cfg.alpha, seed=m + n + k)
    out = nr.interpret(kernel, {a_t: a_q, b_t: b_q, sfa_t: sfa, sfb_t: sfb})
    sim = np.asarray(out[d_t.id], dtype=np.float32).reshape(m, n)

    fn = _compile_tirx(kernel, "nvfp4_sim_parity")
    d_gpu = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    fn(
        torch.from_numpy(a_q).cuda(),
        torch.from_numpy(b_q).cuda(),
        # _prepare already returns the explicit plain physical
        # (rows//128, K//64, 32, 16) GMEM view consumed by both paths.
        torch.from_numpy(sfa).to(torch.float8_e4m3fn).cuda(),
        torch.from_numpy(sfb).to(torch.float8_e4m3fn).cuda(),
        d_gpu,
    )
    torch.cuda.synchronize()
    gpu = d_gpu.to(torch.float32).cpu().numpy()  # bf16 -> f32 is exact
    _assert_bit_equal(gpu, sim, "nvfp4 1024^3")


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(1024, 1024, 1024), (2400, 512, 512)],
    ids=["non_swap_1024", "swap_odd_epi_2400x512x512"],
)
def test_fp8_blockwise_gemm_gpu_matches_sim_bitwise(m, n, k):
    cfg = Fp8BlockwiseGemmConfig(m=m, n=n, k=k, launch_shape=(2,))
    kernel = build_fp8_blockwise_gemm(cfg)
    a_t, b_t, sfa_t, sfb_t, d_t = kernel.args
    a_q, b_q, sfa, sfb, _ref = _fp8_prepare(m, n, k)
    out = nr.interpret(kernel, {a_t: a_q, b_t: b_q, sfa_t: sfa, sfb_t: sfb})
    sim = np.asarray(out[d_t.id], dtype=np.float32).reshape(m, n)

    fn = _compile_tirx(kernel, "fp8_sim_parity")
    d_gpu = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    fn(
        torch.from_numpy(a_q).to(torch.float8_e4m3fn).cuda(),
        torch.from_numpy(b_q).to(torch.float8_e4m3fn).cuda(),
        torch.from_numpy(sfa).cuda(),
        torch.from_numpy(sfb).cuda(),
        d_gpu,
    )
    torch.cuda.synchronize()
    gpu = d_gpu.to(torch.float32).cpu().numpy()
    _assert_bit_equal(gpu, sim, f"fp8 blockwise {m}x{n}x{k}")


@pytest.mark.parametrize(
    "dtype, torch_dtype, label",
    [(nr.DType.F16, torch.float16, "fp16"), (nr.DType.BF16, torch.bfloat16, "bf16")],
)
def test_fp16_bf16_gemm_gpu_matches_sim_bitwise(dtype, torch_dtype, label):
    # 1024 config (GEMM_CONFIGS[1024]: mma_n=64, overlap epilogue, CLC
    # scheduler). Integer inputs in [-2, 2]: products <= 4, K=1024 partial sums
    # stay integers |s| <= 4096 < 2^24 — exact in f32 regardless of accumulation
    # order; the single f32->f16/bf16 rounding at the store is RNE on both sides.
    m = n = k = 1024
    cfg = Fp16Bf16GemmConfig(m=m, n=n, k=k, dtype=dtype)
    kernel = build_fp16_bf16_gemm(cfg)
    a_t, b_t, c_t = kernel.args
    rng = np.random.default_rng(0)
    a = rng.integers(-2, 3, size=(m, k)).astype(np.float32)
    b = rng.integers(-2, 3, size=(n, k)).astype(np.float32)
    out = nr.interpret(kernel, {a_t: a, b_t: b})
    sim = np.asarray(out[c_t.id]).reshape(m, n)
    assert sim.dtype == (np.float16 if dtype == nr.DType.F16 else np.float32)

    fn = _compile_tirx(kernel, f"{label}_sim_parity")
    c_gpu = torch.empty((m, n), dtype=torch_dtype, device="cuda")
    fn(
        torch.from_numpy(a).to(torch_dtype).cuda(),
        torch.from_numpy(b).to(torch_dtype).cuda(),
        c_gpu,
    )
    torch.cuda.synchronize()
    gpu = c_gpu.cpu().numpy() if dtype == nr.DType.F16 else c_gpu.to(torch.float32).cpu().numpy()
    _assert_bit_equal(gpu, sim, f"{label} 1024^3")
