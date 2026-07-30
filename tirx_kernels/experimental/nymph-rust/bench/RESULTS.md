# Nymph GEMM benchmark record

Current explicit-physical-IR baseline, measured on 2026-07-27 UTC with the
bench-suite orchestrator:

```bash
python bench/run_suite.py --max-shape 4096 --rounds 5 \
  --filter nymph_fp16_bf16_gemm --label nymph-physical-ir-final-fp16-bf16
python bench/run_suite.py --max-shape 4096 --rounds 5 \
  --filter nymph_nvfp4_gemm --label nymph-physical-ir-final-nvfp4
```

The portable run records are `.bench-suite/runs/23.json` and
`.bench-suite/runs/26.json`. Every row passed the bench-suite correctness gate
for both implementations. `ratio = tir / tirx`; a ratio above one means Nymph
is faster.

| kernel | shape | dynamic SMEM (bytes) | launch | cluster | warps | correctness | canon (us) | Nymph (us) | ratio | per-round ratio |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| BF16 | 1024³ | 221340 | (128,) | (2,) | 8 | pass | 6.8804 | 7.4808 | 0.9197 | 0.9175–0.9210 |
| BF16 | 2048³ | 180380 | (128,) | (2,) | 8 | pass | 16.3449 | 16.8448 | 0.9703 | 0.9701–0.9706 |
| BF16 | 4096³ | 229508 | (256,) | (2,) | 12 | pass | 93.1003 | 94.5566 | 0.9846 | 0.9820–0.9890 |
| FP16 | 1024³ | 221340 | (128,) | (2,) | 8 | pass | 6.7993 | 7.3926 | 0.9197 | 0.9190–0.9207 |
| FP16 | 2048³ | 180380 | (128,) | (2,) | 8 | pass | 16.5819 | 17.0798 | 0.9708 | 0.9701–0.9725 |
| FP16 | 4096³ | 229508 | (256,) | (2,) | 12 | pass | 95.3724 | 96.8302 | 0.9849 | 0.9710–0.9979 |
| NVFP4 | 1024³ | 131196 | (64,) | (2,) | 8 | pass | 5.4366 | 6.6354 | 0.8193 | 0.8151–0.8234 |
| NVFP4 | 2048³ | 211092 | (128,) | (2,) | 8 | pass | 8.4814 | 11.6321 | 0.7291 | 0.7281–0.7308 |
| NVFP4 | 4096³ | 211092 | (148,) | (2,) | 8 | pass | 29.5446 | 51.2410 | 0.5766 | 0.5763–0.5767 |

The explicit physical IR is slower than the canonical implementation in all
nine rows, most visibly for NVFP4. The physical representation and kernel
schedule remain unchanged based on these measurements, as required. Run 26
automatically discarded two interfered attempts for NVFP4 4096³ and reports
the clean third attempt.
