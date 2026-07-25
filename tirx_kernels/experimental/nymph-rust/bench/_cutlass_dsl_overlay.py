"""Prepend the cu13 cutlass-dsl overlay that repairs NVIDIA/cutlass#3259.

flashinfer's SM100 CuTeDSL GDN kernel JITs through nvidia-cutlass-dsl. The
4.5.2 PyPI wheels let ``nvidia-cutlass-dsl-libs-base`` and ``-libs-cu13``
collide on the SAME ``nvidia_cutlass_dsl/`` tree, and the base variant — which
lacks the CUDA-13 MLIR bytecode behind ``tcgen05.make_tmem_copy`` — won on
this host, ICE-ing every gdn compile ("failed to legalize unresolved
materialization from '!cute_nvgpu.atom.tmem_load<f32, 32 DP, 32 bit, x32>'").
site-packages is read-only, so the repair lives in a full-tree cu13 OVERLAY:

    <workspace>/.txdev-shell/cutlass-dsl-cu13/nvidia_cutlass_dsl/
        python_packages/  lib/  include/

Prepending ``<overlay>/python_packages`` to ``sys.path`` makes
``import cutlass`` resolve to the cu13 tree. The prepend is idempotent and
harmless (a path that shadows with identical-or-repaired content), so we do it
unconditionally instead of sniffing which variant is currently importable —
but it MUST run before the first ``import cutlass`` (sys.modules beats
sys.path). The FULL tree is required: #3259's collision also covers the
``_cutlass_ir`` MLIR extension and the dialect files, not just
``lib/libcute_dsl_runtime.so`` (a single-.so CUTE_DSL_LIBS override was tried
and does NOT fix the ICE).

Resolution order:

1. ``$NYMPH_CUTLASS_DSL_OVERLAY`` — the ``nvidia_cutlass_dsl`` directory (or
   its ``python_packages`` child directly), else
2. ``<repo-root>/../.txdev-shell/cutlass-dsl-cu13/nvidia_cutlass_dsl`` derived
   from THIS file (repo root = ``<workspace>/tirx-kernels``) — no absolute
   paths hardcoded; a clean checkout on another machine simply skips.

A missing directory is skipped silently (the flashinfer baseline then fails
with the #3259 ICE signature, which ``bench_gdn_wave1.py`` reports).

Rebuild the overlay (site-packages is read-only, so pip cannot patch it):

    pip download nvidia-cutlass-dsl-libs-cu13==4.5.2 --no-deps -d /tmp/cu13
    # unzip the wheel; copy nvidia_cutlass_dsl/{python_packages,lib,include}
    # into <workspace>/.txdev-shell/cutlass-dsl-cu13/nvidia_cutlass_dsl/
"""

from __future__ import annotations

import os
import sys


def _overlay_python_packages() -> str | None:
    """Resolve the overlay's python_packages dir, or None if absent."""
    env = os.environ.get("NYMPH_CUTLASS_DSL_OVERLAY")
    if env:
        candidates = [env, os.path.join(env, "python_packages")]
    else:
        # bench/ -> nymph-rust/ -> experimental/ -> tirx_kernels/ -> <repo-root>
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 4)))
        base = os.path.join(
            os.path.dirname(repo_root), ".txdev-shell", "cutlass-dsl-cu13", "nvidia_cutlass_dsl"
        )
        candidates = [os.path.join(base, "python_packages")]
    for cand in candidates:
        pp = os.path.abspath(cand)
        if os.path.basename(pp) == "python_packages" and os.path.isdir(pp):
            return pp
    return None


def ensure_cutlass_dsl_cu13() -> bool:
    """Prepend the cu13 overlay's python_packages to sys.path (idempotent).

    Returns True when the overlay was found (and is now on sys.path).
    """
    pp = _overlay_python_packages()
    if pp is None:
        return False
    if pp not in sys.path:
        sys.path.insert(0, pp)
    return True
