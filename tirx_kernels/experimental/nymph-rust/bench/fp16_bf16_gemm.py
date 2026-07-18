"""Bench-suite interface for the nymph-codegen'd fp16/bf16 GEMM.

Mirrors canon ``tirx_kernels/gemm/fp16_bf16_gemm.py``'s bench-suite interface
(``KERNEL_META`` / ``CONFIGS`` / ``run_bench(dtype, M, N, K, ...)``). Driven via
``run_kernel_bench(registry={"nymph_fp16_bf16_gemm": <this module>})`` since the
registry skips ``experimental/``. Canon is imported read-only; both impls go
through the identical ``bench()`` call.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

_NYMPH_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
if _NYMPH_PY not in sys.path:
    sys.path.insert(0, _NYMPH_PY)

import nymph_rs as nr  # noqa: E402
import torch  # noqa: E402
from nymph_rs.kernels import build_fp16_bf16_gemm  # noqa: E402
from nymph_rs.kernels.fp16_bf16_gemm import Fp16Bf16GemmConfig  # noqa: E402

import tvm  # noqa: E402
from tirx_kernels.gemm.fp16_bf16_gemm import prepare_data, tir_kernel  # noqa: E402
from tirx_kernels.runner import compile_kernel  # noqa: E402
from tvm.tirx.bench import bench  # noqa: E402

KERNEL_META = {"name": "nymph_fp16_bf16_gemm", "category": "experimental", "compute_capability": 10}
CONFIGS = [
    {"dtype": d, "M": s, "N": s, "K": s, "label": f"{d}_{s}x{s}x{s}"}
    for d in ["fp16", "bf16"]
    for s in [1024, 2048, 4096, 8192, 16384]
]


def _compile_nymph(dtype, M, N, K):
    ndt = nr.DType.F16 if dtype == "fp16" else nr.DType.BF16
    src = nr.kernel_to_tirx_source(
        build_fp16_bf16_gemm(Fp16Bf16GemmConfig(m=M, n=N, k=K, dtype=ndt))
    )
    p = os.path.join(tempfile.mkdtemp(prefix="nymph_gemm_"), "g.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nymph_gemm_emitted", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return tvm.compile(
        tvm.IRModule({"main": m.main}), tvm.target.Target("cuda"), tir_pipeline="tirx"
    )


def run_bench(dtype, M, N, K, warmup=None, repeat=None, timer=None, **kwargs):
    target = tvm.target.Target("cuda")
    with target:
        canon = compile_kernel(tir_kernel(dtype, M, N, K))
    nymph = _compile_nymph(dtype, M, N, K)
    a, b, c = prepare_data(dtype, M, N, K)
    oc, on = torch.zeros_like(c, device="cuda"), torch.zeros_like(c, device="cuda")
    funcs = {"tir": lambda: canon(a, b, oc), "tirx": lambda: nymph(a, b, on)}
    # One-shot correctness gate BEFORE timing (a ratio of two kernels computing
    # different results is meaningless): both outputs vs the torch reference.
    for fn in funcs.values():
        fn()
    torch.cuda.synchronize()
    ref = torch.mm(a, b.T)
    for name, out in (("tir", oc), ("tirx", on)):
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.float().flatten(), dim=0
        )
        if cos < 0.99:
            raise AssertionError(f"{name} output diverges from reference (cosine={cos:.4f})")
    return bench(funcs, warmup=warmup, repeat=repeat, timer=timer, **kwargs)


# Register into the bench-suite kernel registry so the STANDARD lookup path
# (registry.load_kernel -> _KERNEL_CACHE, which run_kernel_bench uses by default)
# resolves this kernel by name. The registry's dir-scan discovery skips
# experimental/, so nymph registers itself here at import time instead — no
# canon/registry edits, and no ad-hoc registry={...} injection at the call site.
import sys as _sys  # noqa: E402

from tirx_kernels.registry import _KERNEL_CACHE as _REG  # noqa: E402

_REG[KERNEL_META["name"]] = _sys.modules[__name__]
