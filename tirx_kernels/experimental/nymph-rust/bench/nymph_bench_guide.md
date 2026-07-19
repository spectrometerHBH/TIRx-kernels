# Benching a nymph kernel against canon with the tirx-kernels bench-suite

nymph lives in `tirx_kernels/experimental/`, and the bench-suite registry's
directory-scan discovery **deliberately skips `experimental/`**
(`tirx_kernels/registry.py`: `_SKIP_CATEGORIES = {"bench", "test",
"bench_suite", "experimental"}`). So `discover_kernels()` never finds a nymph
kernel — and we must **not** make it discoverable by editing canon or the
registry (nymph is an independent repo; touch nothing outside
`experimental/nymph-rust/`).

The registry still resolves kernels the standard way: `load_kernel(name)` checks
its module cache (`registry._KERNEL_CACHE`) **first**, before any dir scan
(`registry.py:76`), and `runner.run_kernel_bench` uses `load_kernel` by default.
So a nymph kernel becomes first-class simply by **registering itself into that
cache at import time** — then the standard `load_kernel(name)` /
`run_kernel_bench(name, cfg)` path (the same one `python -m tirx_kernels.bench`
runs internally) resolves it, with **no `registry={...}` argument** at the call
site and **no changes to canon**.

## The interface: in the kernel file (canon's single-file structure)

Each nymph kernel carries its bench-suite interface **in its own kernel
module** (`python/nymph_rs/kernels/<kernel>.py`), mirroring canon's
single-file layout (see `tirx_kernels/gemm/nvfp4_gemm.py`). Do NOT put several
kernels behind one `kind=` switch — one module, one kernel, same shape as
canon.

The interface is three module-level names plus one registrar. **Pure data at
module level; bench-only deps (tvm / torch / canon) imported lazily inside the
functions** — the `nymph_rs` package must stay importable without them
(CPU-only value sim, wheel installs):

```python
# 1. KERNEL_META — identity. name = "nymph_<canon-kernel-name>"; category is
#    "experimental"; compute_capability matches the target (10 = sm_100 / B200).
KERNEL_META = {"name": "nymph_nvfp4_gemm", "category": "experimental", "compute_capability": 10}

# 2. CONFIGS — mirror canon's CONFIGS for this kernel: a list of param dicts, each
#    with a "label". run_kernel_bench feeds one dict (minus "label") to run_bench
#    as kwargs, so the dict keys must match run_bench's parameters.
CONFIGS = [{"M": s, "N": s, "K": s, "label": f"{s}x{s}x{s}"} for s in [1024, 2048, 4096, 8192, 16384]]

# 3. run_bench — build BOTH impls into ONE funcs dict and hand it to bench() ONCE.
#    canon is imported read-only (lazily, here); both impls go through the
#    identical bench() call, so they get the identical methodology (cold-cache
#    + rounds). Forward warmup/repeat/timer/**kwargs straight through — do NOT
#    hardcode them, and do NOT add your own cooldowns / warm-L2 / CUDA-event
#    timing (see "Don't" below).
def run_bench(M, N, K, *, warmup=None, repeat=None, timer=None, **kwargs):
    import torch, tvm
    from tirx_kernels.gemm.nvfp4_gemm import prepare_data, tir_ws_kernel
    from tvm.tirx.bench import bench
    ...  # compile canon + nymph; allocate inputs once (Triton pure-launch)
    # Impl names MUST be "tir" (canon) / "tirx" (nymph): the bench-suite's
    # OURS_IMPLS contract (`bench_suite/impls.py::is_our_impl`) recognizes only
    # tir/tirx(-prefixed) names — anything else is filtered out of reports.
    funcs = {"tir": lambda: canon(...), "tirx": lambda: nymph(...)}
    return bench(funcs, warmup=warmup, repeat=repeat, timer=timer, **kwargs)

# 4. register_bench_interface — self-register into the bench-suite kernel
#    cache so `load_kernel(name)` finds the kernel (the dir-scan discovery
#    skips experimental/). Called by bench/_nymph_bench_autoreg.py under
#    NYMPH_BENCH_SUITE=1 — NOT at import time, so plain `import nymph_rs` has
#    no registry side effects.
def register_bench_interface() -> None:
    import sys
    from tirx_kernels.registry import _KERNEL_CACHE
    _KERNEL_CACHE[KERNEL_META["name"]] = sys.modules[__name__]
```

Signature note: the config-dict keys ARE the run_bench parameters. nvfp4 uses
`{M,N,K}` → `run_bench(M, N, K, ...)`. fp16/bf16 adds `dtype` →
`run_bench(dtype, M, N, K, ...)` (exactly canon's signature).

## Running it (the orchestrator — the ONE default way)

`bench/run_suite.py` is a thin wrapper over the bench-suite **orchestrator**
(`python -m tirx_kernels.bench_suite`): automatic GPU selection + interference
requeue, per-workload subprocess isolation, json/report artifacts under
`<tirx-kernels>/.bench-suite/`. There is no per-GPU flag on purpose — the
orchestrator probes the visible cards and skips occupied ones.

```
python bench/run_suite.py [--rounds 5] [--max-shape 8192] [--filter nvfp4] [--label L]
```

Registration inside every orchestrator worker is env-gated: with
`NYMPH_BENCH_SUITE=1` (the wrapper sets it), the repo-local
`bench/sitecustomize.py` — auto-imported by every child python because the
wrapper prepends `bench/` to the subprocess `PYTHONPATH` — loads
`bench/_nymph_bench_autoreg.py`, which imports the interface modules (they
self-register). This is fully repo-contained: the registration no longer
depends on any machine-local sitecustomize outside the repo (a legacy hook
may still exist in a dev shell — it is idempotent and unused by the bench
path now). Direct invocation without the wrapper:

```
NYMPH_BENCH_SUITE=1 python -m tirx_kernels.bench_suite --workloads bench/nymph_workloads.yaml --no-report
```

`ratio = canon_us / nymph_us` (`>= 1.0` means nymph is at least as fast as
canon). Read the **round-aggregate** `impls` — that is what the bench-suite
reports; `round_samples` is the raw per-round series (fine for a spread
sanity-check, but the aggregate is the number). The historical numbers live in
`bench/RESULTS.md`.

For programmatic use (tests, one-off A/B), the runner path also works
in-process: import an interface module (self-registers), then
`tirx_kernels.runner.run_kernel_bench("nymph_...", cfg, ...)` — but note this
runs on the ambient `CUDA_VISIBLE_DEVICES` with no selection/requeue; it is
NOT the default bench method.

## Don't roll your own timing

The whole point is to use the bench-suite's methodology, not to invent one that
flatters the numbers. Concretely:

- **Don't** drop the L2 flush (measuring a warm/hot L2 makes big GEMMs look
  faster than reality — the bench-suite is cold-cache on purpose).
- **Don't** add `round_cooldown` / sleeps / `nvidia-smi` calls between launches:
  idling the GPU between measurements lets it go cold and produces slow outliers
  (a 200 us GEMM measuring 230 us). The bench-suite keeps the GPU busy.
- **Don't** hand-roll CUDA-event timing or "interleaving": events add launch
  overhead (noise on small kernels), and interleaving is a way to hide a
  measurement you set up wrong. Pass the closures to `bench()` and let it time.
- **Don't** shrink `repeat` below the bench-suite default — fewer iterations per
  round = a noisier average.

If a shape looks unstable, it is almost always the harness, not the kernel:
run the SAME shape through canon's own bench-suite entry
(`python -m tirx_kernels.bench --kernel nvfp4_gemm --config <label> --rounds 10`)
and compare — canon's `tir` there is stable, so match that setup.

## Files

- `python/nymph_rs/kernels/nvfp4_gemm.py`     — the kernel AND its bench interface
- `python/nymph_rs/kernels/fp16_bf16_gemm.py` — the kernel AND its bench interface
- `bench/run_suite.py`      — THE bench entry: thin wrapper over the bench-suite orchestrator
- `bench/sitecustomize.py`  — repo-local auto-imported hook: worker env repair + registration
- `bench/nymph_workloads.yaml` — the workload list the orchestrator consumes
- `bench/_nymph_bench_autoreg.py` — env-gated (NYMPH_BENCH_SUITE=1) registration hook
- `bench/RESULTS.md`        — dated bench records (ground truth for past numbers)
