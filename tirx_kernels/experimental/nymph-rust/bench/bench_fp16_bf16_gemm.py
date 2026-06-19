"""tir-bench comparison: nymph-codegen'd fp16/bf16 GEMM vs the canonical TIRx kernel.

The ONLY measure of kernel speed here is the wall-clock ratio that the tir repo's
``/tir-bench`` produces — i.e. ``tvm.tirx.bench.bench`` (proton/CUPTI timer, an L2
eviction between reps, warmup=100, repeat=30). No locked-clock / cycle counts: a
small kernel's real cost includes the clock/boost it actually runs at, and the bench
captures that. ``ratio = canon_time / nymph_time`` (== nymph's speed relative to canon):
``>= 0.99`` means nymph is at least 0.99x of canon; ``> 1.0`` means nymph is faster.

Both kernels are built for the SAME shape and compiled through the SAME
``tir_pipeline="tirx"`` path, then benched back-to-back in one process so they see the
same GPU clock state.

Usage (from the nymph-rust dir, with the built nymph_rs on PYTHONPATH):

    # one shape
    CUDA_VISIBLE_DEVICES=1 python bench/bench_fp16_bf16_gemm.py 4096 fp16

    # all bench shapes x both dtypes
    CUDA_VISIBLE_DEVICES=1 python bench/bench_fp16_bf16_gemm.py --all

    # more reps / custom shape
    python bench/bench_fp16_bf16_gemm.py 2048 bf16 --reps 8

Env:
    CUDA_VISIBLE_DEVICES   pick an idle GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import tempfile

import torch
import tvm

import nymph_rs as nr
from nymph_rs.kernels import build_fp16_bf16_gemm
from nymph_rs.kernels.fp16_bf16_gemm import Fp16Bf16GemmConfig
from tirx_kernels.gemm.fp16_bf16_gemm import prepare_data, tir_kernel
from tirx_kernels.runner import compile_kernel
from tvm.tirx.bench import bench, tensor_bytes

BENCH_SHAPES = (1024, 2048, 4096, 8192, 16384)
DTYPES = ("fp16", "bf16")


def _compile_nymph(s: int, dt: str):
    """Codegen the nymph IR -> TVMScript source -> exec -> compile (the production path).

    The TVMScript parser reads the prim_func's source via inspect, so the emitted source
    must live in a real .py file on disk (exec-from-string fails)."""
    ndt = nr.DType.F16 if dt == "fp16" else nr.DType.BF16
    src = nr.kernel_to_tirx_source(build_fp16_bf16_gemm(Fp16Bf16GemmConfig(m=s, n=s, k=s, dtype=ndt)))
    d = tempfile.mkdtemp(prefix="nymph_bench_")
    path = os.path.join(d, "g.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nymph_emitted_gemm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return tvm.compile(tvm.IRModule({"main": mod.main}), tvm.target.Target("cuda"), tir_pipeline="tirx")


def bench_shape(s: int, dt: str, reps: int) -> float:
    """Bench canon vs nymph for one (shape, dtype). Returns the median ratio."""
    target = tvm.target.Target("cuda")
    with target:
        canon = compile_kernel(tir_kernel(dt, s, s, s))
        nymph = _compile_nymph(s, dt)

    def make_input():
        a, b, c = prepare_data(dt, s, s, s)
        case = {
            "canon": (a, b, torch.zeros_like(c, device="cuda")),
            "nymph": (a, b, torch.zeros_like(c, device="cuda")),
        }
        return case, tensor_bytes(a, b, c)

    funcs = {
        "canon": lambda c: canon(*c["canon"]),
        "nymph": lambda c: nymph(*c["nymph"]),
    }
    tflop = 2 * s * s * s / 1e12
    ratios = []
    print(f"--- {s} {dt}: {reps} in-process reps ---")
    for i in range(reps):
        res = bench(funcs, make_input, warmup=100, repeat=30, timer="proton", proton_name="bench_gemm")
        im = res["impls"]
        ct, nt = im["canon"], im["nymph"]
        r = ct / nt  # canon/nymph == nymph's speed relative to canon (>=0.99 good, >1 nymph faster)
        ratios.append(r)
        print(f"rep{i}: ratio={r:.3f}  canon={tflop / (ct / 1e3):.0f}TF nymph={tflop / (nt / 1e3):.0f}TF")
    ratios.sort()
    med = ratios[len(ratios) // 2]
    print(f"median={med:.3f} min={ratios[0]:.3f} max={ratios[-1]:.3f}")
    return med


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("shape", nargs="?", type=int, help="square M=N=K (e.g. 4096)")
    p.add_argument("dtype", nargs="?", choices=DTYPES, default="fp16")
    p.add_argument("--reps", type=int, default=5, help="in-process bench reps (median is reported)")
    p.add_argument("--all", action="store_true", help="run every BENCH_SHAPE x both dtypes")
    args = p.parse_args()

    if args.all:
        summary = []
        for dt in DTYPES:
            for s in BENCH_SHAPES:
                summary.append((s, dt, bench_shape(s, dt, args.reps)))
        print("\n=== summary (median ratio = canon/nymph; >=0.99 means nymph is >=0.99x of canon) ===")
        for s, dt, r in summary:
            print(f"  {s:>5} {dt}: {r:.3f}  {'OK' if r >= 0.99 else 'UNDER'}")
    else:
        if args.shape is None:
            p.error("give a shape (e.g. `4096 fp16`) or pass --all")
        bench_shape(args.shape, args.dtype, args.reps)


if __name__ == "__main__":
    main()
