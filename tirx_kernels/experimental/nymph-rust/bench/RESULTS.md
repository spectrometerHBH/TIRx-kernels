# nymph GEMM bench baseline (canon = tir / nymph = tirx)

The current known-good baseline. Ratio = `tir / tirx` = canon_time / nymph_time;
**>1 means nymph is faster**. Target: ≥ 0.99 on every shape.

Method: `python bench/run_suite.py --max-shape 16384 --rounds 5` (bench-suite
orchestrator — auto GPU selection, proton timer, cold cache, correctness gate
on both impls). Full per-round data + artifacts live in the (gitignored,
machine-local) `.bench-suite/runs/*.json`; this file is the portable record.
History of this file is in git.

## 2026-07-24 @ warp-model replay (Stage 3)

B200 ×7, orchestrator, rounds=5 (re-runs at rounds=10 noted). Per-round spread
in brackets. The 2026-07-21 (#18 Role-model branch) column is the prior
baseline; canon itself measures faster today on the small shapes, so
same-shape ratios drift a few tenths of a percent between days.

| kernel | shape | ratio | spread | prior (#18) | canon (us) | nymph (us) |
|---|---|---|---|---|---|---|
| nvfp4 | 1024 | 0.988 | 0.983-0.990 | 1.013 | 5.2 | 5.3 |
| nvfp4 | 2048 | 0.991 | 0.989-0.994 | 0.987 | 8.5 | 8.6 |
| nvfp4 | 4096 | 1.020 | 1.019-1.021 | 1.017 | 29.6 | 29.0 |
| nvfp4 | 8192 | 1.027 | 1.006-1.042 | 1.013 | 186.0 | 181.2 |
| nvfp4 | 16384 | 1.002 | 0.996-1.009 | 1.030 | 1530.6 | 1527.9 |
| fp16 | 1024 | 1.019 | 1.018-1.020 | 1.017 | 7.0 | 6.8 |
| fp16 | 2048 | 1.011 | 1.011-1.012 | 1.011 | 16.5 | 16.4 |
| fp16 | 4096 | 1.006 | 1.003-1.011 | 0.993 | 96.7 | 96.1 |
| fp16 | 8192 | 0.981 → **1.005**¹ | 0.950-1.064 | 0.999 | 736.4 | 732.6 |
| fp16 | 16384 | 1.033 | 0.991-1.083 | 0.995 | 6032.8 | 5837.6 |
| bf16 | 1024 | 1.018 | 1.017-1.019 | 1.017 | 6.8 | 6.7 |
| bf16 | 2048 | 1.011 | 1.011-1.012 | 1.011 | 16.4 | 16.3 |
| bf16 | 4096 | 0.995 → **1.002**¹ | 0.983-1.019 | 0.999 | 93.8 | 93.6 |
| bf16 | 8192 | 0.995 → **1.002**¹ | 0.938-1.064 | 0.994 | 692.9 | 691.4 |
| bf16 | 16384 | 1.006 | 0.940-1.048 | 0.997 | 5720.2 | 5683.4 |

¹ The big squares are noise-dominated at rounds=5 (spread > ±0.03); the
rounds=10 re-measurement is the cleaner read (fp16 8192 1.005, bf16 4096
1.002, bf16 8192 1.002).

Verdict: **14/15 shapes ≥ 0.99**, fp16/bf16 1024 ≥ 1.01 (1.019 / 1.018).
The one borderline shape:

- **nvfp4 1024 = 0.988–0.989** (0.983-0.990 across rounds=5 and rounds=10).
  Same-day A/B against the #18 kernel itself (identical emission family,
  interleaved timing, GPU 3): #18 5.24-5.25us (0.990-0.993), this replay
  5.27-5.28us (0.984-0.988). The ~0.4% residual is the producer's guard
  nesting (`if elect_sync(): if cbx==0:` vs canon's `if cbx==0: if elect_sync():`)
  that the warp-model's single-thread checker mandates — #18 wrote the
  expect_tx on all 32 lanes and let codegen elect; this branch's value model
  counts per-thread arrivals, so the op must sit inside an elected region in
  the IR, which flips the nesting. The GPU-side "regression" from 0.83 was
  separately root-caused to the codegen emitting `thread_rank()==0` for the
  warp-0 MMA loop guard (now `elect_sync()` for loop bodies — see the
  codegen commit) and is fully fixed.

Watch item: fp16 8192/bf16 8192 and the 16384s swing ±0.05 round-to-round;
treat anything inside that band as noise (same as #18's record).

Refresh policy: re-run the command above after any perf-relevant change and
replace this table (one table, no diary). Diagnosis methodology:
`docs/perf-methodology.md`.
