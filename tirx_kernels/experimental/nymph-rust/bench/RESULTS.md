# nymph GEMM bench baseline (canon = tir / nymph = tirx)

The current known-good baseline. Ratio = `tir / tirx` = canon_time / nymph_time;
**>1 means nymph is faster**. Target: ≥ 0.99 on every shape.

Method: `python bench/run_suite.py --max-shape 16384 --rounds 5` (bench-suite
orchestrator — auto GPU selection, proton timer, cold cache, correctness gate
on both impls). Full per-round data + artifacts live in the (gitignored,
machine-local) `.bench-suite/runs/*.json`; this file is the portable record.
History of this file is in git.

## 2026-07-21 @ 49139436 (post review-batch-3)

B200 ×7, orchestrator, rounds=5. Per-round spread in brackets.

| kernel | shape | ratio | spread | canon (us) | nymph (us) |
|---|---|---|---|---|---|
| nvfp4 | 1024 | 1.013 | 1.010-1.018 | 5.5 | 5.4 |
| nvfp4 | 2048 | **0.987** | 0.985-0.992 | 8.4 | 8.5 |
| nvfp4 | 4096 | 1.017 | 1.016-1.018 | 29.6 | 29.1 |
| nvfp4 | 8192 | 1.013 | 0.998-1.038 | 185.5 | 183.1 |
| nvfp4 | 16384 | 1.030 | 0.985-1.085 | 1507.6 | 1463.7 |
| fp16 | 1024 | 1.017 | 1.015-1.018 | 6.9 | 6.8 |
| fp16 | 2048 | 1.011 | 1.011-1.012 | 16.5 | 16.3 |
| fp16 | 4096 | 0.993 | 0.970-1.008 | 95.3 | 95.9 |
| fp16 | 8192 | 0.999 | 0.972-1.064 | 724.4 | 725.1 |
| fp16 | 16384 | 0.995 | 0.956-1.054 | 5755.7 | 5783.3 |
| bf16 | 1024 | 1.017 | 1.014-1.020 | 6.9 | 6.8 |
| bf16 | 2048 | 1.011 | 1.010-1.012 | 16.4 | 16.2 |
| bf16 | 4096 | 0.999 | 0.969-1.021 | 93.4 | 93.5 |
| bf16 | 8192 | 0.994 | 0.980-1.009 | 689.9 | 694.1 |
| bf16 | 16384 | 0.997 | 0.971-1.055 | 5583.6 | 5599.1 |

Watch item: nvfp4 2048 hovers at 0.987-0.996 across runs — borderline, not a
regression (same band as every prior measurement).

Refresh policy: re-run the command above after any perf-relevant change and
replace this table (one table, no diary). Diagnosis methodology:
`docs/perf-methodology.md`.
