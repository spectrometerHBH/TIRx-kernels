"""End-to-end interpreter tests: the fp16/bf16 GEMM through the Rust value
simulator, checked against a numpy reference cell-for-cell.

The GEMM exercises the whole interpreter pipeline — TMA load/store, cta_group=2
MMA, the TMEM collective + scratchpad, mbarrier handshakes (event-driven wake),
cooperative sync, the register ALU/cvt, and the direct-mutation runner — so a
cell-exact match here is a strong correctness signal for the port as a whole.
"""

import numpy as np
import nymph_rs as nr
import pytest
from nymph_rs.kernels import build_fp16_bf16_gemm
from nymph_rs.kernels.fp16_bf16_gemm import Fp16Bf16GemmConfig

try:
    from nymph_rs import DType
except Exception:  # pragma: no cover
    DType = None


def _inputs(cfg, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(-2, 3, size=(cfg.m, cfg.k)).astype(np.float32)
    b = rng.integers(-2, 3, size=(cfg.n, cfg.k)).astype(np.float32)
    return a, b


def _np_dtype(cfg):
    name = str(cfg.dtype)
    if "BF16" in name or "bf16" in name:
        ml = pytest.importorskip("ml_dtypes")
        return ml.bfloat16
    return np.float16


def _reference(a, b, cfg):
    return (a.astype(np.float64) @ b.T).astype(_np_dtype(cfg)).astype(np.float64)


@pytest.mark.parametrize("dtype_name", ["F16", "BF16"])
def test_gemm_matches_numpy_reference(dtype_name):
    if DType is None:
        pytest.skip("DType binding unavailable")
    cfg = Fp16Bf16GemmConfig(dtype=getattr(DType, dtype_name))
    a, b = _inputs(cfg)
    kernel = build_fp16_bf16_gemm(cfg)
    a_t, b_t, c_t = kernel.args
    outputs = nr.interpret(kernel, {a_t: a, b_t: b})
    raw = outputs[c_t.id]
    assert raw.dtype == (np.float16 if dtype_name == "F16" else np.float32)
    c = np.asarray(raw, np.float64).reshape(cfg.m, cfg.n)
    ref = _reference(a, b, cfg)
    assert int(np.sum(c != ref)) == 0


def test_gemm_completes_and_is_deterministic():
    cfg = Fp16Bf16GemmConfig()
    a, b = _inputs(cfg, seed=3)
    kernel = build_fp16_bf16_gemm(cfg)
    a_t, b_t, c_t = kernel.args
    first = np.asarray(nr.interpret(kernel, {a_t: a, b_t: b})[c_t.id], np.float64)
    second = np.asarray(nr.interpret(kernel, {a_t: a, b_t: b})[c_t.id], np.float64)
    assert np.array_equal(first, second)


# (label, config) pairs covering both epilogue paths and the persistent
# multi-task-per-cluster schedule (launch_shape pins ONE cluster so the
# scheduler warp's broadcast + the SMEM/TMEM ring pipeline are exercised across
# tasks, not just a single tile).
_PATH_CONFIGS = {
    "overlap_single": dict(
        m=256, n=64, k=64, overlap_epilogue=True, mma_n=64, blk_k=32, pipe_depth=2, wb_pipe_depth=2
    ),
    "overlap_persistent": dict(
        m=1024,
        n=256,
        k=64,
        overlap_epilogue=True,
        mma_n=64,
        blk_k=32,
        pipe_depth=3,
        wb_pipe_depth=2,
        launch_shape=(2,),
    ),
    "no_overlap_single": dict(
        m=512,
        n=256,
        k=128,
        overlap_epilogue=False,
        mma_n=256,
        blk_k=64,
        pipe_depth=3,
        wb_pipe_depth=4,
    ),
    "no_overlap_persistent": dict(
        m=2048,
        n=512,
        k=128,
        overlap_epilogue=False,
        mma_n=256,
        blk_k=64,
        pipe_depth=2,
        wb_pipe_depth=2,
        l2_group_size=8,
        launch_shape=(2,),
    ),
}


@pytest.mark.parametrize("label", sorted(_PATH_CONFIGS))
def test_gemm_epilogue_paths_match_reference(label):
    cfg = Fp16Bf16GemmConfig(**_PATH_CONFIGS[label])
    a, b = _inputs(cfg, seed=5)
    kernel = build_fp16_bf16_gemm(cfg)
    a_t, b_t, c_t = kernel.args
    c = np.asarray(nr.interpret(kernel, {a_t: a, b_t: b})[c_t.id], np.float64).reshape(cfg.m, cfg.n)
    assert int(np.sum(c != _reference(a, b, cfg))) == 0


@pytest.mark.parametrize("label", sorted(_PATH_CONFIGS))
def test_gemm_epilogue_paths_pass_protocol(label):
    cfg = Fp16Bf16GemmConfig(**_PATH_CONFIGS[label])
    report = nr.check_protocol(build_fp16_bf16_gemm(cfg))
    assert report["status"] == "Passed"
    assert all(e["status"] in ("Passed", "Skipped") for e in report["pass_summary"])
