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

## The interface: one module per nymph kernel

Each nymph kernel gets its **own** bench-suite interface module under `bench/`,
mirroring the corresponding canon kernel's interface (see canon's
`tirx_kernels/gemm/nvfp4_gemm.py` / `fp16_bf16_gemm.py`). Do NOT combine several
kernels behind one `kind=` switch — one module, one kernel, same shape as canon.

A module exposes the three bench-suite names, plus a one-line self-registration:

```python
# 1. KERNEL_META — identity. name = "nymph_<canon-kernel-name>"; category is
#    "experimental"; compute_capability matches the target (10 = sm_100 / B200).
KERNEL_META = {"name": "nymph_nvfp4_gemm", "category": "experimental", "compute_capability": 10}

# 2. CONFIGS — mirror canon's CONFIGS for this kernel: a list of param dicts, each
#    with a "label". run_kernel_bench feeds one dict (minus "label") to run_bench
#    as kwargs, so the dict keys must match run_bench's parameters.
CONFIGS = [{"M": s, "N": s, "K": s, "label": f"{s}x{s}x{s}"} for s in [1024, 2048, 4096, 8192, 16384]]

# 3. run_bench — build BOTH impls into ONE funcs dict and hand it to bench() ONCE.
#    canon is imported read-only; both impls go through the identical bench() call,
#    so they get the identical methodology (cold-cache + rounds). Forward
#    warmup/repeat/timer/**kwargs straight through — do NOT hardcode them, and do
#    NOT add your own cooldowns / warm-L2 / CUDA-event timing (see "Don't" below).
def run_bench(M, N, K, *, warmup=None, repeat=None, timer=None, **kwargs):
    ...  # compile canon + nymph; allocate inputs once (Triton pure-launch)
    # Impl names MUST be "tir" (canon) / "tirx" (nymph): the bench-suite's
    # OURS_IMPLS contract (`bench_suite/impls.py::is_our_impl`) recognizes only
    # tir/tirx(-prefixed) names — anything else is filtered out of reports.
    funcs = {"tir": lambda: canon(...), "tirx": lambda: nymph(...)}
    return bench(funcs, warmup=warmup, repeat=repeat, timer=timer, **kwargs)

# 4. Self-register into the bench-suite kernel cache so `load_kernel(name)` finds
#    it (the dir-scan discovery skips experimental/). This is the whole "plug into
#    the bench-suite" step — no canon/registry edits, no per-call injection.
import sys
from tirx_kernels.registry import _KERNEL_CACHE
_KERNEL_CACHE[KERNEL_META["name"]] = sys.modules[__name__]
```

Signature note: the config-dict keys ARE the run_bench parameters. nvfp4 uses
`{M,N,K}` → `run_bench(M, N, K, ...)`. fp16/bf16 adds `dtype` →
`run_bench(dtype, M, N, K, ...)` (exactly canon's signature).

## Running it (standard bench-suite path, no injection)

Import the interface modules (they self-register), then use the standard
`load_kernel` + `run_kernel_bench` — no `registry={...}`:

```python
import importlib.util, sys
from tirx_kernels.registry import load_kernel
from tirx_kernels.runner import run_kernel_bench   # the bench-suite runner

# import the interface module so it self-registers (in sys.modules before exec)
spec = importlib.util.spec_from_file_location("nvfp4_iface", "bench/nvfp4_gemm.py")
m = importlib.util.module_from_spec(spec); sys.modules["nvfp4_iface"] = m; spec.loader.exec_module(m)

mod = load_kernel("nymph_nvfp4_gemm")        # STANDARD lookup (via the registry cache)
for cfg in mod.CONFIGS:                      # each module iterates its OWN CONFIGS
    r = run_kernel_bench("nymph_nvfp4_gemm", cfg, rounds=10, timer="proton")  # no registry=
    im = r["impls"]                          # {"tir": us, "tirx": us} — round-aggregate
    print(f"{cfg['label']}: tir/tirx = {im['tir'] / im['tirx']:.3f}")
```

`bench/run_suite.py` in this dir does exactly this for all nymph GEMM kernels:

```
CUDA_VISIBLE_DEVICES=<idle gpu> python bench/run_suite.py --rounds 10
```

`ratio = canon_us / nymph_us` (`>= 1.0` means nymph is at least as fast as
canon). Read the **round-aggregate** `r["impls"]` — that is what the bench-suite
reports; `r["round_samples"]` is the raw per-round series (fine for a spread
sanity-check, but the aggregate is the number).

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

- `bench/nvfp4_gemm.py`     — nymph NVFP4 GEMM interface (`run_bench(M, N, K, ...)`)
- `bench/fp16_bf16_gemm.py` — nymph fp16/bf16 GEMM interface (`run_bench(dtype, M, N, K, ...)`)
- `bench/run_suite.py`      — driver: import interfaces → standard `run_kernel_bench`
