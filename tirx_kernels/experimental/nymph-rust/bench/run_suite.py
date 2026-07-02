"""Bench the nymph GEMM kernels through the tirx-kernels bench-suite runner.

Loads the per-kernel interface modules under this dir (each self-registers into
the bench-suite kernel registry on import), then benches every config via the
STANDARD path — ``registry.load_kernel`` + ``runner.run_kernel_bench`` — the same
path ``python -m tirx_kernels.bench`` uses internally. No ``registry={...}``
injection at the call site. See ``nymph_bench_guide.md``.

    CUDA_VISIBLE_DEVICES=<idle gpu> python bench/run_suite.py --rounds 10
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# One entry per nymph bench-suite interface module in this dir.
_INTERFACES = [
    ("_nymph_iface_nvfp4_gemm", "nvfp4_gemm.py"),
    ("_nymph_iface_fp16_bf16_gemm", "fp16_bf16_gemm.py"),
]


def _load_and_register(mod_name: str, filename: str) -> str:
    """Import an interface module (it self-registers into the registry). Returns its kernel name."""
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, filename))
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m  # standard: in sys.modules before exec so it can self-register
    spec.loader.exec_module(m)
    return m.KERNEL_META["name"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rounds", type=int, default=10, help="bench-suite measurement rounds")
    ap.add_argument(
        "--timer", default="proton", help="event | proton | cudagraph_proton (all cold-cache)"
    )
    ap.add_argument("--max-shape", type=int, default=8192, help="skip configs with M above this")
    args = ap.parse_args()

    from tirx_kernels.registry import load_kernel
    from tirx_kernels.runner import run_kernel_bench

    names = [_load_and_register(mod_name, fn) for mod_name, fn in _INTERFACES]

    for kname in names:
        mod = load_kernel(kname)  # STANDARD lookup (resolves via the registry cache)
        for cfg in mod.CONFIGS:
            if cfg["M"] > args.max_shape:
                continue
            result = run_kernel_bench(kname, cfg, rounds=args.rounds, timer=args.timer)
            im = result["impls"]
            ratio = im["canon"] / im["nymph"]
            rs = result.get("round_samples", {})
            spread = ""
            if rs.get("canon") and rs.get("nymph"):
                pr = sorted(c / n for c, n in zip(rs["canon"], rs["nymph"]))
                spread = f"  per-round[{pr[0]:.3f}-{pr[-1]:.3f}]"
            print(
                f"{kname} {cfg['label']}: canon/nymph={ratio:.3f}{spread}  "
                f"canon={im['canon']:.1f}us nymph={im['nymph']:.1f}us",
                flush=True,
            )


if __name__ == "__main__":
    main()
