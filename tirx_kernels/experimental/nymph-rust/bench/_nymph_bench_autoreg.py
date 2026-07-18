"""Env-gated auto-registration of the nymph bench-suite kernel interfaces.

Loaded at interpreter startup via a `nymph_bench.pth` import line in
site-packages. Does NOTHING unless NYMPH_BENCH_SUITE=1, so ordinary python
runs pay zero cost; with the gate on, every python in the tree — including the
bench-suite orchestrator's per-workload `python -m tirx_kernels.bench`
subprocesses — resolves the `nymph_*` kernels through the standard registry
path. No canon/registry edits.
"""

import importlib.util
import os
import sys

if os.environ.get("NYMPH_BENCH_SUITE") == "1":
    _HERE = os.path.dirname(os.path.abspath(__file__))

    def _register(mod_name, filename):
        spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, filename))
        m = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = m
        spec.loader.exec_module(m)  # self-registers via KERNEL_META

    try:
        _register("_nymph_iface_nvfp4_gemm", "nvfp4_gemm.py")
        _register("_nymph_iface_fp16_bf16_gemm", "fp16_bf16_gemm.py")
    except Exception as e:  # never break an unrelated python
        sys.stderr.write(f"[nymph autoreg] skipped: {e}\n")
