# nymph GEMM bench results (canon = tir / nymph = tirx)

Method: `CUDA_VISIBLE_DEVICES=<idle> python bench/run_suite.py --rounds 10 --max-shape 16384`
(bench-suite standard path, proton timer, cold cache, correctness cosine gate on both impls).
Ratio = `tir / tirx` = canon_time / nymph_time; **>1 means nymph is faster**.

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
