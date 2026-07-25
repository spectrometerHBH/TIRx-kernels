"""Repo-local ``sitecustomize`` for the nymph bench-suite subprocesses.

Python's ``site`` module imports the FIRST ``sitecustomize`` module found on
``sys.path`` at every interpreter startup. ``bench/run_suite.py`` prepends
THIS directory to the bench subprocesses' ``PYTHONPATH``, so the orchestrator
and every per-workload worker get the nymph kernel auto-registration from
inside the repo — no machine-local hook (the registration used to ride a
sitecustomize outside the repo, which a clean checkout/CI does not have).

Everything here is gated on ``NYMPH_BENCH_SUITE=1`` (set by ``run_suite.py``);
any other python that happens to have this directory on its path is
untouched. Two steps:

1. Drop stale ``_apache_tvm_editable`` meta-path finders. A machine that once
   had an editable apache-tvm install keeps a ``.pth``-installed finder
   pointing at its (possibly deleted) tree; that finder shadows the
   PYTHONPATH-provided tvm and breaks ``import tvm`` outright. Removing it is
   a no-op on machines without the stale hook.
2. Import ``_nymph_bench_autoreg.py`` from this directory, which registers
   the ``nymph_*`` bench kernel interfaces into the standard registry so
   ``load_kernel("nymph_nvfp4_gemm")`` resolves in every worker.
"""

import os
import sys

if os.environ.get("NYMPH_BENCH_SUITE") == "1":
    # 1. stale editable-tvm finder repair (see module docstring).
    sys.meta_path[:] = [
        f
        for f in sys.meta_path
        if getattr(getattr(f, "__class__", None), "__module__", "") != "_apache_tvm_editable"
    ]

    # 2. repo-local kernel registration (itself guarded + exception-safe).
    try:
        import _nymph_bench_autoreg  # noqa: F401
    except Exception as _e:  # never break an unrelated python
        sys.stderr.write(f"[nymph autoreg] skipped: {_e}\n")
