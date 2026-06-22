"""Bench the nymph-codegen'd nvfp4 GEMM vs canonical TIRx. ratio = canon/nymph (>=0.99 good).

nymph takes 5 args (alpha is a compile-time constant); canon takes 6 (runtime alpha). GEMM
timing is data-independent, so each kernel is fed correctly-shaped inputs of its own layout.
"""
import argparse
import importlib.util
import os
import tempfile

import torch
import tvm

import nymph_rs as nr
from nymph_rs.kernels import build_nvfp4_gemm, NvFp4GemmConfig
from nymph_rs.kernels import gemm_config_for
from tirx_kernels.gemm.nvfp4_gemm import prepare_data, tir_ws_kernel
from tvm.tirx.bench import bench

BENCH_SHAPES = (1024, 2048, 4096, 8192, 16384)
CTA_K = 256
SF_CELLS = 4  # CTA_K // SF_BLOCK(16) // 4


def _compile_nymph(M, N, K, **ov):
    # Per-shape knob overrides (cta_n / epi_tile / smem_depth), mirroring canon's
    # TIRX_CONFIGS. Explicit kwargs win over the table. NVFP4_OV (env JSON) is a
    # bench-time override sweep hook (merged last).
    import json as _json

    env_ov = _json.loads(os.environ.get("NVFP4_OV", "{}"))
    knobs = {**gemm_config_for(M, N, K), **ov, **env_ov}
    cfg = NvFp4GemmConfig(m=M, n=N, k=K, **knobs)
    src = nr.kernel_to_tirx_source(build_nvfp4_gemm(cfg))
    d = tempfile.mkdtemp(prefix="nvfp4_")
    path = os.path.join(d, "g.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nvfp4_emitted", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return tvm.compile(tvm.IRModule({"main": mod.main}), tvm.target.Target("cuda"), tir_pipeline="tirx")


def _nymph_inputs(M, N, K):
    # nvfp4: a/b packed-fp4 u8 (rows, K//2); sfa/sfb e4m3 scales (rows, K//16); d bf16.
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    sfa = torch.randint(0, 256, (M, K // 16), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    sfb = torch.randint(0, 256, (N, K // 16), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    d = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    return [a, b, sfa, sfb, d]


def bench_shape(s, reps, tflop_io=False):
    M = N = K = s
    target = tvm.target.Target("cuda")
    with target:
        canon = tvm.compile(tvm.IRModule({"main": tir_ws_kernel(M, N, K)}), target, tir_pipeline="tirx")
        nymph = _compile_nymph(M, N, K)

    A_fp4, B_fp4, A_sf, B_sf, alpha, C_ref = prepare_data(M, N, K)
    alpha_t = torch.tensor([alpha], device="cuda", dtype=torch.float)

    def make_input():
        out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        ni = _nymph_inputs(M, N, K)
        return {"canon": (A_fp4, B_fp4, A_sf, B_sf, alpha_t, out), "nymph": ni}, 0

    funcs = {"canon": lambda c: canon(*c["canon"]), "nymph": lambda c: nymph(*c["nymph"])}
    tflop = 2 * M * N * K / 1e12
    ratios = []
    print(f"--- {s}x{s}x{s}: {reps} reps ---")
    for i in range(reps):
        res = bench(funcs, make_input, warmup=100, repeat=30, timer="proton", proton_name="nvfp4")["impls"]
        ct, nt = res["canon"], res["nymph"]
        ratios.append(ct / nt)
        print(f"rep{i}: ratio={ct/nt:.3f}  canon={tflop/(ct/1e3):.0f}TF nymph={tflop/(nt/1e3):.0f}TF")
    ratios.sort()
    print(f"median={ratios[len(ratios)//2]:.3f} min={ratios[0]:.3f} max={ratios[-1]:.3f}")
    return ratios[len(ratios) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("shape", nargs="?", type=int)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    if args.all:
        for s in BENCH_SHAPES:
            bench_shape(s, args.reps)
    else:
        bench_shape(args.shape or 1024, args.reps)


if __name__ == "__main__":
    main()
