# Wave-3 fix queue (SASS-attributed, 2026-07-25)

Attribution run over `bench/ncu_data/{flashinfer,nymph}_ns1_t2048.ncu-rep`
(`--page source` SASS with per-line inst_executed; nymph `main_kernel` vs
flashinfer `kernel_cutlass_...GDN`). Total 1,977,960 → 5,530,508 (+3.55M,
2.80x); bench fi/nymph = 0.225 (t512) / 0.181 (t2048); time tracks the
instruction count (UTCHMMA exact-equal at 11,136 both, IPC comparable), so
the queue is ordered by removable instructions. Anchors already exact:
UTCHMMA 1.00x, HMMA 3.50x (=168/48, D6), STSM 8.35x, LDSM 3.95x (D4),
STTM 0.13x, LDGSTS 512→0 (nymph TMA-loads gate/beta: 5 UTMALDG/chunk vs
fi 3 — nymph's choice is fine, NOT a gap).

## Queue (ordered by est. removable instructions)

| # | theme / sub-item | est. inst. excess (t2048) | root cause (SASS evidence) | fix direction |
|---|---|---|---|---|
| 1 | C1: per-element 64-bit SMEM address math in every staging/epi loop | IMAD +399K, LOP3 +381K, VIADD +120K, LEA +120K (~1.0M) | every element recomputes full swizzled 64-bit address: `IMAD.U32 Rx,Rx,0x10000` hi-word pairs + VIADD/LOP3/LEA per element (SASS 5650-5728, 9041-9092) — fi holds fragments in registers, zero per-element addressing | codegen/TIRx emission form: emit register-fragment tile ops (fragment-resident elementwise over wg_reg_tile) instead of per-element `_sl(...)` SMEM indexing; fold swizzle into the iterator, not per-element ALU |
| 2 | D2/D3 structural SMEM bounce (delta/NV/gated-NV/QS) | STSM +169K, LDS +347K, STS +187K, bank conflicts st 406K/ld 257K (fi 11K/18K), wavefronts 15x | §2d-supp D2/D3: acc→stmatrix→SMEM→per-element LDS→tcgen05_st, and QS via SMEM not TMEM in-place (SASS: LDS.U16 streams 9041+, STS.U16 per element) | kernel body: stage readback→TMEM directly like fi (needs D1's St16x128b + operand fragment layouts; orientation D5 decides whether A-side TMEM operands are required — B-side has no PTX form); kills the bounce AND the conflicts |
| 3 | D: sync/polling | SYNCS +411K, NANOSLEEP +213K, BAR +35K (36,992 vs 1,576), FENCE+MEMBAR ~+21K | spin form identical to fi (`SYNCS.PHASECHK.TRYWAIT`+NANOSLEEP retry, both sides) — iterations ∝ wait time: DOWNSTREAM of #1/#2 (fix first). Topology part: nymph does `wg_sync`+`fence_pub` (BAR.SYNC+FENCE.VIEW+MEMBAR) per staging handoff (SASS 7073-7079) where fi uses mbarrier pipelines + 2 fences/chunk | #1/#2 first, re-measure; then kernel body: replace per-step wg_sync/fence handoffs with mbarrier arrive/wait (fi topology, FLASHINFER_REFERENCE §1); codegen: elide redundant fence/membar pairs |
| 4 | C2: local-memory address tables | LDL +112,680 (fi=0), STL +2,368 | per-chunk staging loops spill computed SMEM-address arrays to local (SASS 6374-6430 STL of SHL'd addrs to descending local slots; 8607-8657 LDL reads) — dynamically-indexed arrays, NOT classic reg spill (regs=168 static both, no USETMAXREG in nymph) | dies with #1 (no address arrays once fragments are register-resident); secondary: per-role setmaxnreg (fi: 224/256/24; nymph: none) if CG1's fragment pressure still exceeds 168 after #1 |
| 5 | C3: unpacked scalar math + single-element converts | FMUL +383K (FMUL2 fi 191K → nymph 0), F2F +161,792 (fi=0), PRMT +91K, FADD +169K, F2FP +93K | per-element `FMUL.FTZ` + `F2F.BF16.F32` + `F2FP.PACK`(with RZ = half-packed) + `PRMT 0x7610` byte placement (SASS 5653-5728) vs fi's `FMUL2 .F32x2.HI_LO` + packed F2FP on fragments (fi SASS 1274-1287) | same as #1 (tile-op form packs automatically: FMUL2/F2FP packed pairs, no PRMT); no separate fix |
| 6 | D6-inverse chain shape | HMMA +30,720 (43,008 vs 12,288 = 3.50x), +SHFL-vs-SMEM stage-1 (part of LDS/FFMA excess) | §2d-supp D6: merge all `m16n8k8` (168/chunk) vs fi 8×m16n8k8+40×m16n8k16; stage-1 = SMEM-load FFMA chain vs fi shuffle GJ | kernel body: stages 3/4 → m16n8k16 (fi's atoms), stage-1 → shuffle broadcast; expected once #1-#3 land (HMMA is only 0.8% of total) |
| 7 | MUFU dedup (exp2 recompute per epi) | MUFU +115K (3.39x) | exp2(gcs) recomputed per element per helper: rss/delta/ointer/vnew each 32/thread·chunk (SASS 8488-8516 clusters) vs fi's ONE T-pairwise precompute reused | kernel body: precompute T/dexp/decay fragments once per chunk into registers, reuse across epis (fi CK:2355-2371 pattern) |
| 8 | BRA/guard-branch form | BRA +230K, BSSY/BSYNC ~+33K, NOP +50K | tiny per-element loop bodies (loop-back BRA per iteration) + per-op guard `if`s (perf-methodology's BSSY/BSYNC lesson applies) | #1 collapses the loops; codegen: elect_sync guards + same-guard folding (the GEMM-era fix, already in codegen vocabulary) |

## Notes

- #1/#5 and #2 overlap (the bounce loops ARE the per-element loops): treat
  #1 as the emission-form enabler, #2 as the structural target; expect their
  combined removal to erase ~1.8-2.2M of the 3.55M excess. #3's spin share
  is re-measured after — do NOT tune sync first (perf-methodology §3:
  symptoms move with the work).
- Small-shape ratios (ns1_t64 0.605, ns48_t64 0.618) vs long-seq (t2048
  0.181): the per-chunk serial cost IS the gap; multi-tile shapes hide it.
  Expect #1+#2 to move t2048 most, closing toward the small-shape band.
- A/B verify per step: (1) correctness gate FIRST — full pytest green +
  `run_bench` cosine ≥0.999 vs flashinfer on the same shape (correctness
  gate precedes timing, bench/ncu_gdn.py flow); (2) ncu same shape, diff
  vs the tables here (goal: the #1-#5 opcodes → fi counts, anchors
  UTCHMMA/HMMA/LDGSTS unchanged); (3) bench ratio fi/nymph on all 6 shapes,
  target ≥0.99. One theme per commit; re-profile between themes.
- fi-side SASS for reference dumps: `/tmp/fi_t2048_sass.csv`,
  `/tmp/nymph_t2048_sass.csv` (regenerate: `ncu --import <rep> --page source --csv`).
