# ncu baseline dataset: flashinfer GDN chunked prefill (Wave 3)

Collected with `bench/ncu_gdn.py` (single launch, `--profile-from-start off` +
`torch.cuda.profiler.start()` around the measured launch; sections
InstructionStats + MemoryWorkloadAnalysis_Tables; CUDA_VISIBLE_DEVICES=1,
B200). Files per shape: `.ncu-rep` (raw), `.inst.csv` (per-opcode SASS
`inst_executed`), `.mem.csv` (memory metrics), `.json` (metadata).

Kernel (both shapes, same static config → same cute.compile cache):
`kernel_cutlass_kernel_flashinfergdn_kernelsblackwellgated_delta_net_chunkedGatedDeltaNetChunkedKernel_...`
Grid (8,1,1) × block (384,1,1) — 8 (batch=1 × HV=8) tiles, persistent.
Shapes: ns1_t512 = 8 chunks/tile, ns1_t2048 = 32 chunks/tile.

Total SASS inst_executed: **513,096** (t512) / **1,977,960** (t2048).
`launch__registers_per_thread` = 168, occupancy limit = 1 block (as designed).

## Top-15 opcodes

| # | ns1_t512 | count | ns1_t2048 | count |
|---|---|---|---|---|
| 1 | F2FP | 62,720 | F2FP | 260,096 |
| 2 | SYNCS | 58,010 | SYNCS | 222,506 |
| 3 | FMUL2 | 44,032 | FMUL2 | 191,488 |
| 4 | FMUL | 40,960 | FMUL | 163,840 |
| 5 | IMAD | 37,277 | IMAD | 145,714 |
| 6 | BRA | 34,416 | BRA | 128,681 |
| 7 | NANOSLEEP | 22,901 | NANOSLEEP | 87,293 |
| 8 | LDS | 21,760 | LDS | 87,232 |
| 9 | FADD | 16,704 | FADD | 66,816 |
| 10 | UIADD3 | 14,904 | UIADD3 | 54,216 |
| 11 | MUFU | 12,032 | MUFU | 48,128 |
| 12 | SHF | 10,624 | ISETP | 39,264 |
| 13 | ISETP | 10,080 | SHF | 38,608 |
| 14 | ULOP3 | 8,048 | HADD2 | 31,744 |
| 15 | UMOV | 8,016 | ULOP3 | 28,832 |

Alignment-relevant opcodes (below top-15): UTCHMMA 2,688 / 11,136; STSM
5,760 / 23,040; LDSM 5,248 / 21,760; HMMA 3,072 / 12,288; LDTM 3,456 /
14,208; STTM 3,712 / 16,000; UTMALDG 384 / 1,536; UTMASTG 128 / 512;
UTCBAR 936 / 3,816; LDGSTS 128 / 512; R2UR 5,168 / 20,528; SHFL 3,392 /
13,568; LDG 448; STS 416; STG 4,608; ELECT 296; BAR 1,576; USETMAXREG 407.

## Cross-validation vs FLASHINFER_REFERENCE.md (per-chunk derived × chunks × 8 CTAs)

### EXACT matches (measured = derived, both shapes)

| opcode | derived expectation | t512 | t2048 |
|---|---|---|---|
| UTCHMMA | 8 CTAs × (28 + (C−1)×44), C=chunks | 2,688 ✓ | 11,136 ✓ |
| STSM | 90/chunk (86 x4 + 4 x1) × C × 8 | 5,760 ✓ | 23,040 ✓ |
| LDSM | (54 first + 86 steady)/CTA × 8 (first chunk skips V-read 32) | 5,248 ✓ | 21,760 ✓ |
| HMMA (mma.sync) | 48/chunk (8× m16n8k8 + 40× m16n8k16) × C × 8 | 3,072 ✓ | 12,288 ✓ |
| LDGSTS | beta cp.async 2/chunk × C × 8 | 128 ✓ | 512 ✓ |

These five pin the port's shape: 44 UTCHMMA/steady chunk (28 first),
stmatrix/ldmatrix per-chunk volume AND the first-chunk delta, the mma.sync
inverse chain volume, and the beta LDGSTS path all reproduce exactly — the
doc's per-chunk numbers are validated on silicon.

### Consistent-but-scaled matches (constant factor vs naive count, both shapes)

| opcode | naive (copies) | t512 | t2048 | factor | likely cause |
|---|---|---|---|---|---|
| UTMALDG | 3 G2S/chunk | 384 = 6×64 | 1,536 = 6×256 | ×2 | TMA box split along the 128 B swizzle span (each 16 KB copy = 2 UTMALDG) |
| UTMASTG | 1 S2G/chunk | 128 = 2×64 | 512 = 2×256 | ×2 | same box-split behavior on the O store |
| LDTM | 16 warp-ops/chunk (readbacks) | 3,456 (54/CTA-chunk) | 14,208 (55.5) | ≈×3.4 | tcgen05.ld atoms decompose into multiple LDTM (the x32 state atom dominates); per-chunk rate is stable across shapes |
| STTM | 15 warp-ops/chunk (staging) | 3,712 (58/CTA-chunk) | 16,000 (62.5) | ≈×4 | same for tcgen05.st; steady rate ≈60/CTA-chunk both shapes |
| UTCBAR | 7 commits/chunk | 936 (≈117/CTA) | 3,816 (≈477/CTA) | ≈×2 | likely 2 UTCBAR per tcgen05.commit (arrive+track); confirm via nvdisasm |

For Wave 3: match nymph on the RATIO side per site (16x256b vs 32x32b atom
choice shows up in the LDTM/STTM totals — the reference doc's suspect
"M=128 → 32x32b" divergence would move these counts), not on the naive
per-copy count. Exact LDTM/STTM per-atom decomposition: defer to nvdisasm
attribution when the first nymph diff appears.

### Order-of-magnitude anchors (no exact doc estimate)

- F2FP 62,720/260,096 — f32↔bf16 conversion traffic (kk/qk/beta/state/vks/nv/
  decay/O staging), the #1 opcode. FMUL2/FMUL — the T/beta/decay/scale
  elementwise multiplies. HADD2/HFMA2 — bf16 ALU on halves.
- MUFU 12,032/48,128 — exp2/log2 (T-pairwise 32/thread·chunk + gate warp +
  decay_scale), ≈188/CTA-chunk.
- SYNCS 58,010/222,506 + NANOSLEEP 22,901/87,293 — mbarrier polling waits;
  dominant sync traffic as expected for 6-warp-role pipelines. No reference
  number (nymph's sync structure differs by construction) — this is a pure
  diff target for Wave 3, largest-first per perf-methodology §2.
- R2UR 5,168/20,528 — flashinfer HAS the uniform-datapath mover too (TMA
  coords, ≈80/CTA-chunk). Wave-3 target for nymph: same order, not zero.
- LDS 21,760/87,232 — SMEM scalar reads (cumsumlog/beta per-element reads in
  T-pairwise/epis + inverse stage-1 rows). SHFL 3,392/13,568 — gate
  prefix-sum (11/chunk) + inverse stage-1 (~49/block × 8 blocks).

### MemoryWorkloadAnalysis anchors (mem.csv)

- l1tex shared: wavefronts 107,028 (t512); bank conflicts ld 18,266 (≈17%),
  st 11,499 (≈11%) — the stmatrix/ldmatrix/inverse SMEM traffic is NOT
  conflict-free in flashinfer either; nymph should land in the same range.
- L2: lts__t_sectors 417,688 (13.4 MB) / 1,001,271 (32 MB).
- DRAM: read 69,040 sectors (2.2 MB) / 268,720 (8.6 MB); write 0 both
  (outputs/state stay L2-resident at these sizes).
- global store sectors (l1tex op_st) 131,589 — final-state STG + tensormap
  descriptor updates in GMEM workspace (128 B/descriptor/tile).

## Wave-3 usage (nymph side)

Same harness, nothing else to build:

```bash
CUDA_VISIBLE_DEVICES=1 python bench/ncu_gdn.py --impl nymph --shape ns1_t512
```

Today it prints `NYMPH_SIDE_NOT_COMPILABLE: ValueError: codegen: RegUnary
not yet supported` and exits 0 (Wave-2 codegen pending); once codegen lands
it produces `nymph_<shape>.{ncu-rep,inst.csv,mem.csv,json}` with identical
input data (same `_bench_inputs` seed) and the same single-launch isolation.
Diff convention (perf-methodology §2): sort |nymph − flashinfer| per opcode,
largest first; kill the top cluster, re-profile, repeat. The five EXACT
anchors above are the pass/fail gates; the ×2/×3.4/×4 families must show the
SAME factors in nymph (a different factor = different atom/box choice =
a finding). `--reuse-report` re-parses without re-profiling.
