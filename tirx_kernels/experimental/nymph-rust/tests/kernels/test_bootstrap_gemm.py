"""Value + protocol tests for the bootstrap GEMM (the codegen's smallest target)."""

import numpy as np
import nymph_rs as nr
from nymph_rs.kernels import build_bootstrap_gemm


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(-2, 3, size=(256, 64)).astype(np.float32)
    b = rng.integers(-2, 3, size=(128, 64)).astype(np.float32)
    return a, b


def test_bootstrap_gemm_matches_numpy_reference():
    a, b = _inputs()
    kernel = build_bootstrap_gemm()
    a_t, b_t, c_t = kernel.args
    out = nr.interpret(kernel, {a_t: a, b_t: b})
    c = np.asarray(out[c_t.id], np.float64).reshape(256, 128)
    ref = (a.astype(np.float64) @ b.T).astype(np.float16).astype(np.float64)
    assert int((c != ref).sum()) == 0


def test_bootstrap_gemm_protocol():
    assert nr.check_protocol(build_bootstrap_gemm())["status"] == "Passed"
