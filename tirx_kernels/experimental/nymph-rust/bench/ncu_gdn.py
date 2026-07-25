"""ncu collection harness for the GDN port's Wave-3 instruction-level alignment.

Profiles exactly ONE launch of the target GDN kernel with the ncu CLI and
parses the report with the ``ncu_report`` python module into csv/json under
``bench/ncu_data/``. Self-spawning: the outer invocation re-runs itself with
``--worker`` under ``ncu --profile-from-start off``; the worker does
cute.compile/tvm.compile + warmup launches FIRST, then calls
``torch.cuda.profiler.start()`` around the single measured launch, so
input-gen/JIT/aux kernels never enter the profile. (An NVTX
``--nvtx-include`` filter was tried first: this ncu version only matches
start/end ranges, not torch's push/pop ranges, so profiler start/stop is the
robust mechanism.)

    python bench/ncu_gdn.py --impl flashinfer --shape ns1_t512 \
        [--sections InstructionStats,MemoryWorkloadAnalysis_Tables] \
        [--out-dir bench/ncu_data] [--launch-count 4] [--reuse-report]

Outputs (per impl+shape): ``.ncu-rep`` (raw), ``.inst.csv``
(sass__inst_executed_per_opcode breakdown), ``.mem.csv`` (MemoryWorkloadAnalysis
key metrics), ``.json`` (both + run metadata incl. discovered kernel names).

nymph side: identical flow via ``gdn_prefill._compile_nymph`` /
``_nymph_callable`` (same inputs, same marking). TODAY the codegen fails
closed (Wave 2 in flight) — the worker prints the exact error and exits 0
WITHOUT a profile instead of crashing, so the harness is already in place.

Methodology: docs/perf-methodology.md §2 (InstructionStats per-opcode diff,
largest gaps first; MemoryWorkloadAnalysis_Tables for the memory side).
FLASHINFER_REFERENCE.md holds the per-chunk steady-state expectations this
data cross-validates. Kernel launches reuse ``gdn_prefill._bench_inputs`` /
``_flashinfer_callable`` read-only — identical data to the bench.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_NYMPH_RUST = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_NCU = "/usr/local/cuda/bin/ncu"
_OPCODE_METRIC = "sass__inst_executed_per_opcode"
# MemoryWorkloadAnalysis(_Tables) key metrics; missing ones are skipped.
_MEM_METRICS = [
    "memory_l2_theoretical_sectors_global",
    "memory_l2_theoretical_sectors_global_excessive",
    "memory_l1_wavefronts_shared",
    "memory_l1_wavefronts_shared_excessive",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "lts__t_sectors.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "sm__sass_inst_executed.sum",
    "smsp__inst_executed.sum",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]


def _shape_to_kwargs(gdn, shape: str) -> dict:
    for cfg in gdn.BENCH_CONFIGS:
        if cfg["label"] == shape:
            return {k: v for k, v in cfg.items() if k != "label"}
    raise ValueError(f"unknown shape {shape!r}; have {[c['label'] for c in gdn.BENCH_CONFIGS]}")


def _worker(impl: str, shape: str) -> int:
    """Compile + warm, then launch the kernel ONCE inside the NVTX range."""
    # Stale editable-tvm finder repair INSIDE the worker (ncu strips the env
    # in a way that kills the txdev startup sitecustomize; the stale
    # `_apache_tvm_editable` meta-path finder then shadows the PYTHONPATH
    # worktree tvm — mirrors bench/sitecustomize.py, unconditional).
    sys.meta_path[:] = [
        f
        for f in sys.meta_path
        if getattr(getattr(f, "__class__", None), "__module__", "") != "_apache_tvm_editable"
    ]
    for _p in (os.path.join(_NYMPH_RUST, "python"), _REPO):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import torch
    from nymph_rs.kernels import gdn_prefill as gdn

    kw = _shape_to_kwargs(gdn, shape)
    seq_lens = gdn._bench_seqlens(kw.get("num_seqs"), kw.get("seqlen"), kw.get("seqlens"))

    if impl == "flashinfer":
        data = gdn._bench_inputs(seq_lens)
        run, _, _ = gdn._flashinfer_callable(data, len(seq_lens))  # cute.compile + 1st call
    else:  # nymph — expected to fail closed today (Wave 2)
        try:
            ex, config = gdn._compile_nymph(seq_lens)
        except Exception as e:
            print(f"NYMPH_SIDE_NOT_COMPILABLE: {type(e).__name__}: {e}", flush=True)
            print("nymph codegen is Wave-2 work; rerun once it lands.", flush=True)
            return 0
        data = gdn._bench_inputs(seq_lens, config.io_dtype)
        run, _, _ = gdn._nymph_callable(ex, config, data, seq_lens)
        run()
        torch.cuda.synchronize()

    for _ in range(2):  # warmup beyond the compile-triggering first call
        run()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()  # pairs with ncu --profile-from-start off
    run()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"WORKER_OK impl={impl} shape={shape}", flush=True)
    return 0


def _parse_report(rep_path: str, impl: str, shape: str, out_dir: str) -> dict:
    import ncu_report

    ctx = ncu_report.load_report(rep_path)
    kernels = []
    inst_rows = []
    mem_rows = []
    for r in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(r)
        for a in range(rng.num_actions()):
            action = rng.action_by_idx(a)
            kname = action.name()
            kernels.append(kname)
            m = action.metric_by_name(_OPCODE_METRIC)
            if m is not None and m.num_instances() > 0:
                names = m.correlation_ids()  # IMetric whose value(i) is the opcode string
                for i in range(m.num_instances()):
                    inst_rows.append(
                        {"kernel": kname, "opcode": str(names.value(i)), "count": m.value(i)}
                    )
            for name in _MEM_METRICS:
                mm = action.metric_by_name(name)
                if mm is not None and mm.has_value():
                    try:
                        val = mm.as_uint64()
                    except Exception:
                        val = mm.as_double()
                    mem_rows.append(
                        {"kernel": kname, "metric": name, "value": val, "unit": str(mm.unit())}
                    )
    base = os.path.join(out_dir, f"{impl}_{shape}")
    with open(base + ".inst.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel", "opcode", "count"])
        w.writeheader()
        w.writerows(sorted(inst_rows, key=lambda r: -r["count"]))
    with open(base + ".mem.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel", "metric", "value", "unit"])
        w.writeheader()
        w.writerows(mem_rows)
    meta = {
        "impl": impl,
        "shape": shape,
        "kernels": kernels,
        "ncu_report": os.path.basename(rep_path),
        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inst_total": sum(r["count"] for r in inst_rows),
        "inst_csv": os.path.basename(base + ".inst.csv"),
        "mem_csv": os.path.basename(base + ".mem.csv"),
        "mem": mem_rows,
    }
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--impl", choices=["flashinfer", "nymph"], required=True)
    ap.add_argument("--shape", required=True, help="BENCH_CONFIGS label, e.g. ns1_t512")
    ap.add_argument(
        "--sections",
        default="InstructionStats,MemoryWorkloadAnalysis_Tables",
        help="comma-separated ncu sections",
    )
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "ncu_data"))
    ap.add_argument(
        "--launch-count", type=int, default=4, help="max launches inside the NVTX range"
    )
    ap.add_argument("--ncu", default=_NCU, help="ncu CLI path")
    ap.add_argument(
        "--reuse-report",
        action="store_true",
        help="skip collection if the .ncu-rep already exists (re-parse only)",
    )
    args = ap.parse_args()

    if args.worker:
        sys.exit(_worker(args.impl, args.shape))

    os.makedirs(args.out_dir, exist_ok=True)
    rep_base = os.path.join(args.out_dir, f"{args.impl}_{args.shape}")
    rep_path = rep_base + ".ncu-rep"

    if not (args.reuse_report and os.path.exists(rep_path)):
        cmd = [args.ncu, "--target-processes", "all", "-f", "-o", rep_base]
        for sec in args.sections.split(","):
            cmd += ["--section", sec.strip()]
        cmd += [
            "--profile-from-start",
            "off",  # worker re-enables around the single target launch
            "--launch-count",
            str(args.launch_count),
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--impl",
            args.impl,
            "--shape",
            args.shape,
        ]
        print("+", " ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"ncu failed with exit code {r.returncode}")
        if not os.path.exists(rep_path):
            # nymph-side known gap: worker exits 0 without any kernel launch.
            sys.exit(0)

    meta = _parse_report(rep_path, args.impl, args.shape, args.out_dir)
    print(f"kernels profiled: {meta['kernels']}")
    print(f"total SASS inst_executed: {meta['inst_total']}")
    print(f"wrote {rep_base}.{{ncu-rep,inst.csv,mem.csv,json}}")


if __name__ == "__main__":
    main()
