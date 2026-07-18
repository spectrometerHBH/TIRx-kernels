# nymph GEMM bench results (canon = tir / nymph = tirx)

Method: `CUDA_VISIBLE_DEVICES=<idle> python bench/run_suite.py --rounds 10 --max-shape 16384`
(bench-suite standard path, proton timer, cold cache, correctness cosine gate on both impls).
Ratio = `tir / tirx` = canon_time / nymph_time; **>1 means nymph is faster**.

## 2026-07-18 R2UR/uniform batch (ScalarLet) + shape retune — new baseline

B200, GPU2 (idle), rounds=10. After: (1) the `ScalarLet` single-assignment IR +
`k.let` + `T.let` codegen, with the fp16/nvfp4 per-task decode chains converted
from mutable scalars/shuffle_sync to lets; (2) nvfp4 4096 retune
(l2_group_size=4 + maxnreg_epilogue=240, see its GEMM_CONFIGS comment);
(3) fp16/bf16 1024 knob sweep (no change — plateau, see its comment).
ncu A/B on the let conversion: fp16 1024 R2UR 13504 -> 13504 (SHFL 2560 ->
1024), nvfp4 1024 R2UR 2016 -> 2016 — the residual R2UR is a whole-function
ptxas placement decision, evidence in docs/perf-methodology.md §3.

| kernel | shape | ratio | spread | canon (us) | nymph (us) |
|---|---|---|---|---|---|
| nvfp4 | 1024 | 1.040 | 0.907-1.418 | 5.7 | 5.5 |
| nvfp4 | 2048 | 0.996 | 0.991-1.002 | 8.7 | 8.7 |
| nvfp4 | 4096 | 1.012 | 1.010-1.014 | 29.5 | 29.1 |
| nvfp4 | 8192 | 1.019 | 0.987-1.043 | 188.3 | 184.9 |
| nvfp4 | 16384 | 1.024 | 0.988-1.092 | 1509.8 | 1474.7 |
| fp16 | 1024 | **0.981** | 0.979-0.984 | 6.9 | 7.1 |
| fp16 | 2048 | 0.995 | 0.994-0.997 | 16.5 | 16.6 |
| fp16 | 4096 | 0.998 | 0.989-1.007 | 96.8 | 97.0 |
| fp16 | 8192 | 0.995 | 0.940-1.046 | 734.4 | 738.4 |
| fp16 | 16384 | 1.037 | 0.949-1.101 | 6158.0 | 5939.8 |
| bf16 | 1024 | **0.981** | 0.977-0.983 | 6.9 | 7.0 |
| bf16 | 2048 | 0.994 | 0.993-0.996 | 16.3 | 16.4 |
| bf16 | 4096 | 1.005 | 0.979-1.025 | 94.1 | 93.6 |
| bf16 | 8192 | **0.977** | 0.952-1.016 | 692.6 | 708.7 |
| bf16 | 16384 | 0.999 | 0.950-1.052 | 5660.3 | 5666.8 |

Below the 0.99 target:

- fp16/bf16 1024: 0.981/0.981 — knob-swept to a plateau (l2_group_size,
  wb_pipe_depth, pipe_depth, mma_n, blk_k, epilogue all flat-or-worse; see
  the 1024 comment in fp16_bf16_gemm.py). Bottleneck: nymph executes +17%
  total instructions vs canon on this shape (SYNCS/BRA/IMAD/ISETP + the
  13.5K R2UR cluster), a WHOLE-FUNCTION ptxas scalar-placement issue the
  T.let conversion provably does not move (nvrtc/nvdisasm bisection in
  docs/perf-methodology.md §3). Not fixable by config knobs; needs a
  whole-kernel static-complexity reduction to flip ptxas's placement.
- bf16 8192: 0.977 but spread 0.952-1.016 — noise-dominated (fp16 8192 read
  0.995 in the same run; both recorded ~0.99-1.01 in prior baselines). A
  standalone 10-round rerun the same day read 0.992 [0.970-1.020], so it
  straddles the target within noise rather than sitting below it.

## 2026-07-18 baseline @ refactor/nymph-tmem-codegen-genericity branch point

B200, GPU1 (idle), rounds=10. Baseline for the TMEM refactor / codegen genericity /
R2UR work. Per-round spread in brackets.

| kernel | shape | ratio | spread | canon (us) | nymph (us) |
|---|---|---|---|---|---|
| nvfp4 | 1024 | 1.010 | 1.004-1.037 | 5.4 | 5.4 |
| nvfp4 | 2048 | 0.990 | 0.988-0.994 | 8.6 | 8.6 |
| nvfp4 | 4096 | **0.977** | 0.977-0.979 | 29.6 | 30.3 |
| nvfp4 | 8192 | 1.015 | 1.005-1.038 | 189.0 | 186.1 |
| nvfp4 | 16384 | 1.012 | 0.974-1.044 | 1521.8 | 1503.6 |
| fp16 | 1024 | 0.983 | 0.980-0.985 | 6.9 | 7.0 |
| fp16 | 2048 | 0.993 | 0.992-0.994 | 16.6 | 16.7 |
| fp16 | 4096 | 1.003 | 0.994-1.031 | 96.6 | 96.4 |
| fp16 | 8192 | **0.974** | 0.911-1.019 | 726.1 | 745.7 |
| fp16 | 16384 | **0.981** | 0.928-1.064 | 5959.6 | 6076.0 |
| bf16 | 1024 | 0.982 | 0.977-0.986 | 6.9 | 7.0 |
| bf16 | 2048 | 0.993 | 0.991-0.994 | 16.5 | 16.6 |
| bf16 | 4096 | 0.999 | 0.995-1.007 | 93.9 | 94.0 |
| bf16 | 8192 | 0.998 | 0.950-1.037 | 702.0 | 703.3 |
| bf16 | 16384 | 1.001 | 0.938-1.103 | 5777.3 | 5772.7 |

## 2026-07-18 codegen genericity refactor (items 1-6) — no regression

B200, GPU1 (idle), rounds=10. After the codegen-genericity refactor (per-variant
fail-closed Errs, cluster-derived ids, vendored mma_shared_layout, full-K dense
MMA without the codegen run-collapse, explicit MBar.leader_routed, tightened
is_nonneg, structured guard merge, exhaustiveness gate). Emission was
byte-identical for fp16/bootstrap and import-swap-only + the nvfp4 %5 form for
nvfp4, so no movement is expected; all cells within noise of the baseline.

| kernel | shape | ratio | spread | canon (us) | nymph (us) |
|---|---|---|---|---|---|
| nvfp4 | 1024 | 0.998 | 0.995-1.002 | 5.4 | 5.4 |
| nvfp4 | 2048 | 0.989 | 0.987-0.995 | 8.5 | 8.6 |
| nvfp4 | 4096 | 0.981 | 0.980-0.983 | 29.6 | 30.1 |
| nvfp4 | 8192 | 1.015 | 0.994-1.051 | 189.4 | 186.6 |
| nvfp4 | 16384 | 1.007 | 0.941-1.073 | 1513.2 | 1503.1 |
| fp16 | 1024 | 0.985 | 0.983-0.986 | 6.9 | 7.0 |
| fp16 | 2048 | 0.994 | 0.992-0.995 | 16.6 | 16.7 |
| fp16 | 4096 | 0.998 | 0.973-1.010 | 96.5 | 96.7 |
| fp16 | 8192 | 1.009 | 0.956-1.069 | 741.2 | 734.8 |
| fp16 | 16384 | 1.044 | 0.975-1.280 | 6281.8 | 6019.7 |
| bf16 | 1024 | 0.985 | 0.983-0.985 | 6.9 | 7.0 |
| bf16 | 2048 | 0.994 | 0.992-0.995 | 16.5 | 16.6 |
| bf16 | 4096 | 1.001 | 0.993-1.012 | 94.2 | 94.1 |
| bf16 | 8192 | 0.994 | 0.923-1.024 | 705.8 | 710.3 |
| bf16 | 16384 | 1.012 | 0.948-1.107 | 5756.4 | 5690.9 |

Shapes below the 0.99 target (perf work items, R2UR investigation see
`docs/perf-methodology.md` once it lands):

- nvfp4 4096: 0.977 (kernel comments recorded 0.999 under the old, non-same-math
  bench — treat historical comment numbers with care)
- nvfp4 2048: 0.990 (borderline)
- fp16 8192: 0.974 (comment recorded ~1.00)
- fp16 16384: 0.981 (comment recorded 1.000-1.007)
- fp16/bf16 1024: 0.983/0.982 (known producer-side R2UR: elected-region scalars
  can't be shuffle-uniformized, see fp16_bf16_gemm.py producer comment)

Note: numbers recorded in kernel-file comments predate the same-math bench
(commit 81c26acf) and the bench-suite integration; this file is the ground truth
going forward.
