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

## cuBLAS alignment (2026-07-27, dev fc4c5f17)

`cublas/tirx = torch.matmul_us / nymph_us` from `python bench/run_suite.py
--rounds 5 --filter fp16_bf16 --max-shape 16384` (portable record:
`.bench-suite/runs/` latest `cublas_*` labels). After the nvjet-shape
alignment (one launch tile per cluster, nvjet stage counts and M-major
raster, static task source at 1024, single-consumer epilogue at 4096) and
the prologue restructure (tcgen05.alloc overlapped with the first TMA
flight via an `alloc_done` gate; mbarrier inits split across warps 1/2):

| shape | fp16 | bf16 |
|---|---:|---:|
| 1024 | 0.93–0.95 | 0.93–0.95 |
| 2048 | 0.99–1.01 | 0.98–1.00 |
| 4096 | 1.00–1.03 | 1.00 |
| 8192 | 1.01–1.05 | 1.00–1.04 |
| 16384 | 0.99–1.03 | 0.97–1.01 |

1024 stop-rule (measured, not converged at kernel-body level): the K-sweep
gap is a pure ~0.34us intercept (per-k-tile slopes identical); a micro
kernel measures `tcgen05.alloc+relinquish+dealloc` at ~0.87us per cluster,
of which the alloc part is now hidden under the first TMA flight
(+0.03 ratio); the remainder sits in the cluster prologue barrier, the
D-store drain wait, and the dealloc pair rendezvous. nvjet avoids this
class entirely: its SASS contains no tcgen05.alloc/dealloc (TMEM is taken
via `UTCATOMSWS.FIND_AND_SET` atomics) and it uses PREEXIT for the exit
overlap. Closing 1024 requires launch/framework-level work (atomic TMEM
management / PREEXIT / PDL launch attributes), which is out of the
kernel-body boundary.
