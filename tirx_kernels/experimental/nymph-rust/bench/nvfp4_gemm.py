"""Bench-suite interface for the nymph-codegen'd NVFP4 GEMM.

Mirrors canon ``tirx_kernels/gemm/nvfp4_gemm.py``'s bench-suite interface
(``KERNEL_META`` / ``CONFIGS`` / ``run_bench``). The bench-suite registry SKIPS
``experimental/`` where nymph lives, so this is driven via
``run_kernel_bench(registry={"nymph_nvfp4_gemm": <this module>})`` — the runner's
injection point — reusing the bench-suite's exact timer + rounds. Canon is
imported read-only; both impls go through the identical ``bench()`` call.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile

import nymph_rs as nr
import torch
from nymph_rs.kernels import NvFp4GemmConfig, build_nvfp4_gemm, gemm_config_for

import tvm
from tirx_kernels.gemm.nvfp4_gemm import prepare_data, tir_ws_kernel
from tvm.tirx.bench import bench

KERNEL_META = {"name": "nymph_nvfp4_gemm", "category": "experimental", "compute_capability": 10}
CONFIGS = [
    {"M": s, "N": s, "K": s, "label": f"{s}x{s}x{s}"} for s in [1024, 2048, 4096, 8192, 16384]
]


def _compile_nymph(M, N, K):
    src = nr.kernel_to_tirx_source(
        build_nvfp4_gemm(NvFp4GemmConfig(m=M, n=N, k=K, **gemm_config_for(M, N, K)))
    )
    p = os.path.join(tempfile.mkdtemp(prefix="nymph_nvfp4_"), "g.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nymph_nvfp4_emitted", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return tvm.compile(
        tvm.IRModule({"main": m.main}), tvm.target.Target("cuda"), tir_pipeline="tirx"
    )


def run_bench(M, N, K, *, warmup=None, repeat=None, timer=None, **kwargs):
    target = tvm.target.Target("cuda")
    with target:
        canon = tvm.compile(
            tvm.IRModule({"main": tir_ws_kernel(M, N, K)}), target, tir_pipeline="tirx"
        )
    nymph = _compile_nymph(M, N, K)
    A, B, Asf, Bsf, alpha, Cref = prepare_data(M, N, K)
    at = torch.tensor([float(alpha)], device="cuda", dtype=torch.float)
    Ae, Be = Asf.view(torch.float8_e4m3fn), Bsf.view(torch.float8_e4m3fn)
    oc = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    on = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    funcs = {
        "canon": lambda: canon(A, B, Asf, Bsf, at, oc),
        "nymph": lambda: nymph(A, B, Ae, Be, on),
    }
    return bench(funcs, warmup=warmup, repeat=repeat, timer=timer, **kwargs)


# Register into the bench-suite kernel registry so the STANDARD lookup path
# (registry.load_kernel -> _KERNEL_CACHE, which run_kernel_bench uses by default)
# resolves this kernel by name. The registry's dir-scan discovery skips
# experimental/, so nymph registers itself here at import time instead — no
# canon/registry edits, and no ad-hoc registry={...} injection at the call site.
import sys as _sys  # noqa: E402

from tirx_kernels.registry import _KERNEL_CACHE as _REG  # noqa: E402

_REG[KERNEL_META["name"]] = _sys.modules[__name__]
