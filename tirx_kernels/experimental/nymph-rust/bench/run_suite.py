"""Bench the nymph GEMM kernels — THE way: the bench-suite orchestrator.

Thin wrapper over ``python -m tirx_kernels.bench_suite``: automatic GPU
selection + interference requeue, per-workload subprocess isolation,
json/report artifacts under ``<tirx-kernels>/.bench-suite/``. The nymph
kernels self-register in every worker via the ``NYMPH_BENCH_SUITE=1``
sitecustomize hook (see ``_nymph_bench_autoreg.py``).

    python bench/run_suite.py [--rounds 5] [--max-shape 8192] [--filter nvfp4] [--label L]

Requires the nymph_rs extension importable (built wheel in ``_pybuild/`` or
the source-tree ``.so``) and the usual tvm / tirx_kernels PYTHONPATH.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_NYMPH_RUST = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))


def _config_m(config: str) -> int:
    m = re.search(r"(\d+)x\d+x\d+", config)
    return int(m.group(1)) if m else 0


def _filtered_workloads(max_shape: int) -> str:
    src = os.path.join(_HERE, "nymph_workloads.yaml")
    out = []
    for ln in open(src).read().splitlines():
        m = re.search(r"config:\s*(\S+)}", ln)
        if m and _config_m(m.group(1)) > max_shape:
            continue
        out.append(ln)
    f = tempfile.NamedTemporaryFile(
        mode="w", prefix="nymph_workloads_", suffix=".yaml", delete=False
    )
    f.write("\n".join(out) + "\n")
    f.close()
    return f.name


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rounds", type=int, default=5, help="standard-timer rounds per workload")
    ap.add_argument("--max-shape", type=int, default=8192, help="skip configs with M above this")
    ap.add_argument("--filter", default=None, help="kernel-name substring filter")
    ap.add_argument("--label", default=None, help="run label (default: git short sha)")
    ap.add_argument("--cooldown", type=float, default=None, help="per-impl cooldown seconds")
    args = ap.parse_args()

    yaml_path = _filtered_workloads(args.max_shape)
    env = os.environ.copy()
    env["NYMPH_BENCH_SUITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [os.path.join(_NYMPH_RUST, "python"), _REPO, env.get("PYTHONPATH", "")] if p
    )
    cmd = [
        sys.executable,
        "-m",
        "tirx_kernels.bench_suite",
        "--workloads",
        yaml_path,
        "--no-report",
        "--rounds",
        str(args.rounds),
    ]
    if args.filter:
        cmd += ["--filter", args.filter]
    if args.label:
        cmd += ["--label", args.label]
    if args.cooldown is not None:
        cmd += ["--cooldown", str(args.cooldown)]

    out_dir = os.path.join(_REPO, ".bench-suite")
    before = set(glob.glob(os.path.join(out_dir, "runs", "*.json")))
    r = subprocess.run(cmd, cwd=_REPO, env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)

    new = sorted(
        set(glob.glob(os.path.join(out_dir, "runs", "*.json"))) - before, key=os.path.getmtime
    )
    if not new:
        sys.exit("run_suite: no run json produced")
    data = json.load(open(new[-1]))
    for res in data["results"]:
        if res.get("status") != "ok":
            print(f"{res['kernel']} {res['config']}: {res.get('status')}")
            continue
        im = res["impls"]
        ratio = im["tir"] / im["tirx"]
        rs = res.get("round_samples", {})
        spread = ""
        if rs.get("tir") and rs.get("tirx"):
            pr = sorted(c / n for c, n in zip(rs["tir"], rs["tirx"]))
            spread = f"  per-round[{pr[0]:.3f}-{pr[-1]:.3f}]"
        print(
            f"{res['kernel']} {res['config']}: tir/tirx={ratio:.3f}{spread}  "
            f"tir={im['tir']:.1f}us tirx={im['tirx']:.1f}us  (gpu {res.get('gpu')})"
        )


if __name__ == "__main__":
    main()
