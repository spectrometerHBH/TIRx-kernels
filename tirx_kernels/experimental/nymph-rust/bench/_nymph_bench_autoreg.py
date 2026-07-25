"""Env-gated auto-registration of the nymph bench-suite kernel interfaces.

Loaded at interpreter startup via the REPO-LOCAL ``bench/sitecustomize.py``
(auto-imported by every python that has ``bench/`` on its PYTHONPATH —
``bench/run_suite.py`` arranges that for the orchestrator and its workers).
Does NOTHING unless NYMPH_BENCH_SUITE=1; with the gate on, every python in
the tree — including the bench-suite orchestrator's per-workload
subprocesses — resolves the ``nymph_*`` kernels through the standard
registry path.

The interfaces (KERNEL_META / CONFIGS / run_bench) live IN the kernel modules
(``python/nymph_rs/kernels/``); this hook just makes them importable and calls
their ``register_bench_interface()``. It also prepends the cu13 cutlass-dsl
overlay (``_cutlass_dsl_overlay.py``, the NVIDIA/cutlass#3259 repair) so the
flashinfer baseline the gdn bench imports later resolves to the cu13 tree —
this runs at interpreter startup, safely before any ``import cutlass``.
"""

import os
import sys

if os.environ.get("NYMPH_BENCH_SUITE") == "1":
    _PY = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
    if _PY not in sys.path:
        sys.path.insert(0, _PY)

    try:
        import _cutlass_dsl_overlay

        _cutlass_dsl_overlay.ensure_cutlass_dsl_cu13()

        from nymph_rs.kernels import fp16_bf16_gemm, gdn_prefill, nvfp4_gemm

        nvfp4_gemm.register_bench_interface()
        fp16_bf16_gemm.register_bench_interface()
        gdn_prefill.register_bench_interface()
    except Exception as e:  # never break an unrelated python
        sys.stderr.write(f"[nymph autoreg] skipped: {e}\n")
