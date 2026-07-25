# FlashInfer GDN chunked prefill — instruction-level alignment reference

Authoritative "what the flashinfer side looks like" checklist for Wave 3 ncu
alignment (InstructionStats per-opcode SASS counts + MemoryWorkloadAnalysis).
Source of truth: `~/flashinfer/flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py`
(3735 lines, repo 517cca9c) and `gated_delta_net_tile_scheduler.py` (263 lines).
Line refs below are `CK:<line>` (chunked kernel) / `TS:<line>` (tile scheduler).

Audience: whoever is diffing nymph vs flashinfer SASS. Each section ends with
**HARD** (nymph must match item-for-item; any diff is a finding to eliminate)
and **FREE** (allowed to differ; do not chase). Numbers labeled *derived* are
computed from the tilers/layouts, not literally printed in the source — confirm
against the ncu dump before acting on them.

Environment note: the flashinfer kernel compiles and runs on this host only
with the cu13 cutlass-dsl overlay (`bench/_cutlass_dsl_overlay.py`,
NVIDIA/cutlass#3259). All SMEM cosizes/swizzles quoted here were produced by
tracing the exact `sm100_utils.make_smem_layout_*` calls (CK:594-621) with the
installed cutlass-dsl 4.5.2, not estimated.

## 0. Algorithm and GEMM semantics map (per chunk, BT=64)

Kernel docstring CK:29-84. GDN math (FLA `chunk_gated_delta_rule` fwd), per
chunk of 64 tokens, state S held **transposed as S^T (DV,DK)** in TMEM fp32
(this is why flashinfer's gmem state layout is `[N,H,V,K]` — `_store_final_state`
writes the TMEM tile straight out, CK:3098-3108):

| GEMM | math | tiler (M,N,K) | tiled_mma | A operand | B operand | C (TMEM) | issue site |
|---|---|---|---|---|---|---|---|
| 1 kk | W_kk = K·Kᵀ (BT,BT) | (64,64,128) | `tiled_mma_qk` | sK (SMEM) | sK (SMEM) | shared_acc | CK:2097 |
| 2 qk | W_qk = Q·Kᵀ (BT,BT) | (64,64,128) | `tiled_mma_qk` | sQ (SMEM) | sK (SMEM) | shared_acc | CK:2115 |
| 3 k·state | KS = K·S ⇒ KSᵀ (DV,BT) | (128,64,128) | `tiled_mma_qs` | S^T bf16 (TMEM state_inp) | sK (SMEM) | shared_acc | CK:2136 |
| 4 q·state | QS = Q·S ⇒ QSᵀ (DV,BT) | (128,64,128) | `tiled_mma_qs` | S^T bf16 (TMEM state_inp) | sQ (SMEM) | q_state_acc | CK:2150 |
| 5 new_v | NV = A_inv·(V−KS) ⇒ NVᵀ (DV,BT) | (128,64,64) | `tiled_mma_qkv` (first chunk: `_ss`, A from SMEM) | (V−KS)ᵀ bf16 (TMEM shared_inp) | sAinv (SMEM) | shared_acc | CK:2173 |
| 6 qkv | O_intra = W_qkv·NV ⇒ O_intraᵀ (DV,BT), **accumulated onto scaled QSᵀ** | (128,64,64) | `tiled_mma_qkv` | NVᵀ bf16 (TMEM shared_inp) | sQk = W_qkv (SMEM) | q_state_acc | CK:2196 |
| 7 kv_update | dS = Kᵀ·(decay·NV) ⇒ dSᵀ (DV,DK), **accumulated onto Φ·Sᵀ** | (128,128,64) | `tiled_mma_kv` | (decay·NV)ᵀ bf16 (TMEM shared_inp) | sK_trans = Kᵀ MN-major (SMEM) | state | CK:2219 |

The MMA computes C = A·Bᵀ (B is K-major) in every case; the "ᵀ" results above
are the TMEM-resident orientations and are *hard semantics* — the A-operands of
GEMMs 3-7 are bf16 tensors staged into TMEM by CG1 (`tCtState_inp`,
`tCtShared_inp`), never SMEM, and the f16 kind is used end-to-end.

Variable-name trap (do not propagate into the port): the fragment comments at
CK:2064-2087 have A/B roles swapped (`tCrS_A` labeled "K as A" is actually the
TMEM state input; `tCrQkv_A` labeled "W_qkv as A" actually carries NV;
`tCrNv_B` labeled "NV as B" is actually sQk = W_qkv). The `cute.gemm(...)`
call arguments are authoritative.

Position in the mainloop (steady state, MMA warp CK:2089-2229): kk → qk →
(3+4 only when `valid_state`) → 5 → 6 → 7. First chunk of a tile
(`is_first_chunk`, no initial state): GEMMs 3/4 skipped, GEMM 5 uses
`tiled_mma_qkv_ss` (A = sV SMEM, MN-major), GEMMs 6/7 zero-init their
accumulators at kphase 0 (CK:2194, CK:2218), CG0/CG1 take the
`not valid_state` paths.

Scale/beta/decay application points (all ALU, f32, then cvt to io dtype):
- kk_epi (CG0): `M_kk[i,j] = W_kk[i,j]·T[i,j]·beta[i]` (CK:2416-2418).
- qk_epi (CG0): `W_qkv[i,j] = W_qk[i,j]·T[i,j]·scale` (CK:2456).
- inverse tail (CG0): `A_inv[i,j] *= beta[j]` (CK:2553-2555).
- kv_decay (CG1): `Sᵀ *= Φ` where Φ = cumprod[BT−1] = exp2(cumsumlog[BT−1])
  (CK:3339, 3392-3399).
- v−k·state (CG1): `delta = V − KS·cumprod[bt]` (CK:3456-3460).
- state*q_epi (CG1): `QSᵀ *= cumprod[bt]·scale` in TMEM, GEMM 6 accumulates on
  top (CK:3484-3491).
- kv_decay_v (CG1): `dv = NV·exp2(cumsumlog[BT−1]−cumsumlog[bt])` (CK:3531).
- T-pairwise (CG0): `T[i,j] = exp2(cumsumlog[i]−cumsumlog[j])` for i≥j else 0,
  precomputed per-thread into registers before the TMEM waits (CK:2363-2371).

## 1. Warp-specialize scheme

12 warps = 384 threads, occupancy 1 CTA/SM, cluster (1,1,1), no 2-CTA MMA
(CK:258-263). Register budgets via setmaxnreg: CG0 224, CG1 256, all others 24
(CK:241-243; 128·224+128·256+128·24 = 64512 ≤ 65536). Warp roles CK:231-239:

| warps | role | per-chunk steady-state work | method |
|---|---|---|---|
| 0-3 | compute group 0 (CG0) | T-pairwise (exp2), kk_epi, qk_epi, A_inv hierarchical inverse | `compute_group_0` CK:2245 |
| 4-7 | compute group 1 (CG1) | kv_decay (Φ·S + bf16 state staging), v−k·state, state*q_epi, new_v_epi, kv_decay_v, O epilogue staging | `compute_group_1` CK:3113 |
| 8 | MMA warp | issues all 7 GEMMs in dependency order | `mma_warp` CK:1952 |
| 9 | TMA qkv warp | 1×TMA K (double-buffered), 1×TMA Q, 1×TMA V per chunk; per-tile descriptor update of q/k/v in GMEM workspace | `tma_qkv_warp` CK:1657 |
| 10 | gate/beta warp | gate: ldg→log2→warp prefix-sum→exp2→sts (cumsumlog+cumprod); beta: cp.async; OOB predication on last chunk | `load_gate_beta_warp` CK:1802 |
| 11 | epilogue warp | 1×TMA S2G store of O per chunk; per-tile O descriptor update; bulk commit+wait before releasing sO | `epilogue_warp` CK:3583 |

Each warp role runs its OWN persistent scheduler loop
(tile = (batch, head); inner loop over chunks, CK:1174, 1231, 1379, 1490,
1563, 1614). Every warp re-derives `num_chunks_b = ceil_div(cu[b+1]−cu[b], 64)`
per tile — varlen is data-dependent control flow in every warp.

Named barriers (CK:271-293): id=1 `tmem_alloc` 288 thr (mma+CG0+CG1), id=2
`inverse` 128 thr (CG0), id=3 `inverse_inner` 64 thr (stage-4 warp pair),
id=4 `init_state_store` 128 thr (CG1). Everything else is mbarrier pipelines.

mbarrier topology (all created CK:1024-1149; type = pipeline class,
stages, producer group → consumer group, arrive counts):

| # | name (SMEM field) | type | stages | producer → consumer | tx/purpose |
|---|---|---|---|---|---|
| 1 | load_k | TmaUmma | 2 | TMA(1 thr) → MMA(1 thr) | 16384 B/stage |
| 2 | load_q | TmaUmma | 1 | TMA → MMA | 16384 B |
| 3 | load_v | TmaAsync | 1 | TMA → CG1 (4 thr: one/warp) | 16384 B |
| 4 | load_gate | Async | 1 | warp10(32) → CG0+CG1(256) | sw commit (cumsumlog+cumprod) |
| 5 | load_beta | CpAsync | 1 | warp10(32) → CG0(128) | cp.async commit |
| 6 | kv_acc | UmmaAsync | 1 | MMA(1) → CG1(128) | GEMM 7 done (state RMW serialization) |
| 7 | q_state_acc | UmmaAsync | 1 | MMA(1) → CG1(128) | GEMM 4 / GEMM 6 done |
| 8 | shared_acc | UmmaAsync | 2 | MMA(1) → CG0(128) | GEMMs 1/2/3/5 done (CG1 waits too; see below) |
| 9 | a_inv_ready | AsyncUmma | 1 | CG0(128) → MMA(1) | sAinv ready for GEMM 5 |
| 10 | qk_ready | AsyncUmma | 1 | CG0(128) → MMA(1) | sQk (W_qkv) ready for GEMM 6 |
| 11 | state_inp_ready | AsyncUmma | 1 | CG1(128) → MMA(1) | bf16 Sᵀ staged in TMEM for GEMMs 3/4 |
| 12 | shared_inp_ready | AsyncUmma | 2 | CG1(128) → MMA(1) | vks / nv / decay_v bf16 staged (cycled ×3 per chunk) |
| 13 | o_store | Async | 1 | CG1(128) → epi(32) | sO ready for TMA store |
| 14 | group_order | Async | 1 | CG0(128) → CG1(128) | kk/qk epi ordering rendezvous |

`shared_acc` nuance: consumer_group is cg_cg0 (128) so each stage's release
expects 128 arrivals; CG0 releases the kk/qk stages and CG1 releases the ks/nv
stages, each with its own 128 threads (CK:2428, 2465, 3458, 3520). CG0
additionally bare-`advance()`s past the ks (+nv) stages it never reads —
1 advance, 2 when valid_state (CK:2475-2478) — so its phase counter stays
aligned with the 2-stage cycle. `kv_acc` doubles as the state-TMEM
read-modify-write serializer: CG1 waits GEMM 7 of chunk c−1 before the
kv_decay RMW, releases so MMA can issue GEMM 7 of chunk c on top of Φ·S
(CK:3347-3407). Producer `commit()` on UmmaAsync pipelines is the
`tcgen05.commit` → mbarrier mechanism.

Critical path per chunk (steady state): GEMM1 → kk_epi → inverse ⟂ GEMM2 →
qk_epi (qk_ready) ∥ kv_decay → state_inp → GEMM3 → v−k·state → vks → GEMM 5
(needs ainv_ready) → new_v_epi → GEMM 6 (needs qk_ready + scaled QS) →
qkv_epilogue → o_store → TMA O; kv_decay_v → GEMM 7 → (next chunk's kv_decay).

**HARD**: warp count/roles; register budgets (occupancy 1 is a perf fact — any
nymph variant that spills or changes 384-thread/1-CTA shape is a finding);
mbarrier count 14 with these stage depths and arrive counts; the kv_acc RMW
serialization; 2-stage K double buffer and 2-stage shared_acc/shared_inp
cycles; the per-chunk dependency order of the 7 GEMMs.
**FREE**: named-barrier IDs; which warp id numbers map to roles (0-11 is
conventional); pipeline class *names* (nymph uses its own MBar vocabulary) —
the arrive/wait topology and counts are what must be isomorphic.

## 2. Key instruction list (per warp, per chunk, steady state)

### 2a. TMA (UTMALDG/UTMASTG), warp 9 + warp 11

| copy | tile | bytes | descriptor | issue site |
|---|---|---|---|---|
| K G2S | (BT=64, DK=128) bf16, stage = k_handle.index (2-stage) | 16384 | slot 1, updated per tile | CK:1731 |
| Q G2S | (64, 128) bf16, 1 stage | 16384 | slot 0, updated per tile | CK:1760 |
| V G2S | (DV=128, BT=64) bf16, 1 stage | 16384 | slot 2, updated per tile | CK:1789 |
| O S2G | (DV=128, BT=64) bf16 box | 16384 | slot 3, updated per tile | CK:3631 |

Varlen: descriptors rebuilt per TILE over "bounded" tensors whose token dim is
capped at `batch_end` (CK:1498-1518, 1621-1628) — TMA zero-fills OOB input
rows and clamps OOB output rows; `fence_tensormap_update` on chunk 0 of each
tile (CK:1729, 1759, 1788, 1639). Descriptor workspace: 4 slots × 128 B per
CTA in GMEM, `init_tensormap_from_atom` once at kernel start (CK:1479-1488,
1608-1612); `bytes_per_tensormap=128`, `num_tensormaps=4` (CK:193-194;
workspace size = 128·4·num_sm persistent, CK:3700-3707).
Epilogue does `cp_async_bulk_commit_group` + `wait_group 0` before releasing
sO (CK:3641-3642) — one outstanding O store per chunk, serialized.

**HARD**: exactly 3 G2S + 1 S2G TMA copies per chunk at 16 KB each; K issued
into a 2-stage pipeline; per-tile (not per-chunk) descriptor update cadence;
bulk-group commit/wait before SMEM slot release. **FREE**: descriptor
workspace layout/slot order; fence placement details.

### 2b. tcgen05.mma (UTCHMMA), warp 8

Atom: `tcgen05.MmaF16BF16Op(io, f32, (M,N,16), CtaGroup.ONE, OperandSourceA,
majors)` (CK:526-534) — PTX `tcgen05.mma.cta_group::1.kind::f16`, f32 acc,
K=16 per instruction. Per chunk (kphases = K/16, unroll_full):

| GEMM | tiler | kphases | UTCHMMA/chunk | A source / majors |
|---|---|---|---|---|
| 1 kk | m64 n64 k16 | 8 | 8 | SMEM, K/K (CK:537-544) |
| 2 qk | m64 n64 k16 | 8 | 8 | SMEM, K/K |
| 3 k·state | m128 n64 k16 | 8 | 8 | TMEM (A=TMEM!), K/K (CK:546-553) |
| 4 q·state | m128 n64 k16 | 8 | 8 | TMEM, K/K |
| 5 new_v | m128 n64 k16 | 4 | 4 | TMEM (first chunk: SMEM via `_ss`, MN/K), K/K (CK:555-571) |
| 6 qkv | m128 n64 k16 | 4 | 4 | TMEM, K/K |
| 7 kv_update | m128 n128 k16 | 4 | 4 | TMEM, K/MN (CK:573-580) |

Steady-state total **44 UTCHMMA/chunk**; first chunk (no state) 28.
`ACCUMULATE` flag: kphase 0 = false except GEMM 6/7 where `valid_state` forces
true from kphase 0 (accumulate onto QSᵀ / Φ·Sᵀ) (CK:2096, 2114, 2135, 2149,
2172, 2194, 2218). GEMM 3/4 read Sᵀ **directly from TMEM as the A operand**
(bf16, staged by CG1 into `tmem_state_inp`), no SMEM round-trip.

**HARD**: the 7 tilers exactly (pinned by `can_implement` CK:357-372), kind
f16, cta_group 1, K=16 atoms, per-GEMM kphase counts, A-operand sources
(TMEM vs SMEM) per GEMM, ACCUMULATE pattern. **FREE**: issue interleaving
inside the MMA warp beyond the dependency edges (the mbarrier waits order it).

### 2c. tcgen05.ld/st (TMEM ↔ register), CG0 + CG1

Atom shapes and loop bounds are quoted from source; "inst/thread/chunk" is
*derived* (per-thread elements ÷ atom payload: ld.16x256b.x8 = 32 f32,
ld/st.32x32b.x32 = 32 f32, st.32x32b.x16 = 32 bf16, st.16x128b.x8 = 64 bf16).

| site | tensor (per thread) | atom (source line) | copies/chunk (source loop) |
|---|---|---|---|
| CG0 kk readback | W_kk (64,64) f32 | `Ld16x256bOp(8)` CK:2304 | 2-sub loop, 32-elem subtiles (CK:2409-2426) |
| CG0 qk readback | W_qk (64,64) f32 | same atom CK:2304 | 2-sub loop (CK:2449-2462) |
| CG1 state read | Sᵀ (128,128) f32, 128/thread | `Ld32x32bOp(32)` CK:3173 | 4-sub loop (CK:3350-3355) → *derived* 4 ld |
| CG1 state writeback (Φ·S) | same, 128/thread | `St32x32bOp(32)` CK:3176 | 4-sub loop (CK:3392-3399) → 4 st |
| CG1 state_inp store (bf16 Sᵀ) | (128,128) bf16, 128 el/thread | `St32x32bOp(16)` CK:3204 | same 4-sub loop (CK:3383-3387) → 4 st |
| CG1 KS readback | (128,64) f32, 64/thread | `Ld16x256bOp(8)` CK:3234 | single copy (CK:3451-3455) → *derived* 2 ld |
| CG1 QS scale (ld+st) | (128,64) f32, 64/thread | `Ld16x256bOp(8)` / `St16x256bOp(8)` CK:3286-3290 | 2-sub loop (CK:3477-3491) → 2 ld + 2 st |
| CG1 NV readback | (128,64) f32, 64/thread | `Ld16x256bOp(8)` CK:3234 | 2-sub loop (CK:3505-3510) → 2 ld |
| CG1 O readback | (128,64) f32, 64/thread | `Ld16x256bOp(8)` CK:3321 | single copy (CK:3552-3556) → *derived* 2 ld |
| CG1 shared_inp stores (vks, nv, decay_v) | (128,64) bf16, 64 el/thread | `St16x128bOp(8)` CK:3259 | vks single (CK:3461-3465), nv 2-sub (CK:3514-3518), decay_v 2-sub (CK:3535-3539) → 5 copies |
| CG1 `_load_initial_state` (per TILE) | (128,128) f32, 128/thread | `St32x32bOp(32)` CK:2968 | 4-sub loop (CK:2985-3004) |
| CG1 `_store_final_state` (per TILE) | same | `Ld32x32bOp(32)` CK:3056 | 4-sub loop (CK:3069-3075) |

Every TMEM store is followed by `fence_view_async_tmem_store` before the
pipeline commit (CK:3388, 3403, 3466, 3492, 3519, 3540, 3005).

Alignment note (RESOLVED 2026-07-25, see §2d-supp): flashinfer uses
**16x256b.x8 for ALL (64,64) and (128,64) f32 accumulator readbacks** and
reserves 32x32b.x32 for the (128,128) f32 state only. The suspected
divergence (nymph docstring "M=64 → 16x256b, M=128 → 32x32b") does NOT
exist at the readback sites: nymph has no M=128 (128,64) accs at all — its
GEMM 3-6 accs are (BT=64, DV=128), i.e. flashinfer's accs TRANSPOSED, read
with the same 16x256b atom. The real divergences live elsewhere (staging
atom 32x32b-vs-16x128b, SMEM round-trips, x1-vs-x4 matrix ops, mma.sync
k-shape) — full audit in §2d-supp.

**HARD**: atom shape + repetition per site (these are the SASS
`tcgen05.ld/st` opcodes ncu will show), the bf16 TMEM staging stores, the
fence-before-commit discipline. **FREE**: exact sub-loop trip decomposition
(2 vs 4 subs) as long as the atom and total element volume match.

### 2d. ldmatrix/stmatrix (SMEM ↔ register)

Counts are warp-ops per CTA per chunk (ncu InstructionStats view);
stmatrix.x4 / ldmatrix.x4 move 4×(8×8) = 256 bf16 per warp-op, x1 moves 64.

| site | atom | warp-ops/chunk *derived* |
|---|---|---|
| CG0 kk_epi → sAinv ((64,64) bf16) | `StMatrix8x8x16bOp(4, no-trans)` CK:2321 | 4096/256 = 16 |
| CG0 qk_epi → sQk | `StMatrix8x8x16bOp(4, no-trans)` CK:2337 | 16 |
| CG0 inverse beta-pass: A_inv read + write (CK:2546, 2556) | `LdMatrix8x8x16bOp(4)` CK:2330 / stmatrix.x4 | 16 ld + 16 st |
| CG1 V read (sV (128,64)) | `LdMatrix8x8x16bOp(4, TRANS)` CK:3309 | 8192/256 = 32 ldmatrix.x4.trans |
| CG1 O store → sO ((128,64)) | `StMatrix8x8x16bOp(4, TRANS)` CK:3328 | 32 stmatrix.x4.trans |
| inverse stage 2, per tile (D (8,8), C (8,8), A (8,8); O (8,8)) ×4 tiles | ldmatrix.x1, x1.trans, x1.trans; stmatrix.x1 (CK:2655-2680) | 4×(3 ld.x1) = 12 ld.x1 + 4 st.x1 |
| inverse stage 3, per tile (D/C/A (16,16); O (16,16)) ×2 tiles | ldmatrix.x4, x4.trans, x4.trans; stmatrix.x4 (CK:2764-2788) | 2×(3 ld.x4) = 6 ld.x4 + 2 st.x4 |
| inverse stage 4, per warp (D (16,32), C/A (32,32); O (16,32)) ×2 warps | ldmatrix.x4, x4.trans, x4.trans; stmatrix.x4 (CK:2855-2879) | 2×(2+4+4 ld.x4) = 20 ld.x4 + 2×2 st.x4 = 4 st.x4 |

Per-chunk totals *derived*: **stmatrix 86 x4 + 4 x1** (no-trans x4 = kk 16 +
qk 16 + beta-write 16 + stage-3 2 + stage-4 4 = 54; trans x4 = O store 32;
x1 no-trans = stage-2 4). **ldmatrix 74 x4 + 12 x1** (no-trans x4 =
beta-read 16 + stage-3 D 2 + stage-4 D 4 = 22; trans x4 = V 32 + stage-3
C/A 4 + stage-4 C/A 16 = 52; x1 = stage-2 D 4 plain + C/A 8 trans).

**HARD**: x4 vs x1 matrix counts and transpose flags per site (STSM/LDSM
opcodes + `.trans` in SASS), the epilogue staging via stmatrix (not plain
STS). **FREE**: none significant — this is a datapath-shape requirement.

### 2d-supp. nymph-side atom audit (2026-07-25)

Audit of the nymph kernel's tcgen05.ld/st call sites against §2c, read-only.
nymph side at dev `e1a01f75` (+ uncommitted Wave-2 codegen work in `src/`,
which does not touch the kernel body); nymph file refs are
`python/nymph_rs/kernels/gdn_prefill.py:N`. The interpreter models each
atom's exact TMEM cell map per shape (`src/interpreter/semantics/tcgen05.rs`
`datapath_index_arrays_cached(shape, num)`, B200-verified alignment rules),
and the builder's `tcgen05_ld/st` DEFAULT shape is `32x32b`
(`builder.py:847-879`) — every nymph call below without an explicit `shape=`
is 32x32b. Reg formula (codegen.rs:2294): `.32x32b` = num b32/thread,
`.16x128b` = 2·num, `.16x256b` = 4·num.

#### Verdict table (per call site)

| nymph site | use | nymph atom | flashinfer atom (§2c) | verdict |
|---|---|---|---|---|
| `rss()` L524-527 | CG0 kk/qk epi readback, (64,64) f32 acc | `16x256b` num=8, 1 call/acc | `Ld16x256bOp(8)`, 2-sub loop | **MATCH** (atom + orientation); loop split is FREE |
| `_read128` L1080 | CG1 acc readbacks (delta/ointer/vnew/store_out), (64,128) f32 | `16x256b` num=8 ×2 col-blocks | `Ld16x256bOp(8)` | **MATCH atom** — but nymph acc is (BT,DV) M=64, flashinfer's is (DV,BT) M=128 (see D5) |
| L926, L1029 | state read `s_tmem` (128,128) f32 (chunk-top decay; tile-end final store) | `32x32b` num=64 ×2 halves | `Ld32x32bOp(32)` ×4 subs | **MATCH** (128 f32/thread both) |
| L933 | state decay writeback (Φ·S) | `32x32b` num=64 ×2 | `St32x32bOp(32)` ×4 | **MATCH** |
| L934-936 | `state_inp` staging, (128,128) 16-bit | `32x32b` (default) num=64 ×1 | `St32x32bOp(16)` ×4 subs | **MATCH** (same 32x32b family; 64 vs 4×16 reg grouping is FREE) |
| L910 | per-tile state zero-fill `s_tmem` | `32x32b` num=128 ×1 | — (none: flashinfer zero-inits via GEMM 7 ACCUMULATE=false) | EXTRA (1/tile; nymph GEMM 7 also uses accum0 — belt-and-braces, harmless) |
| L957-958, L996, L999-1000 | `shared_inp`/`shared_inp_b` staging, (128,64) bf16 | `32x32b` (default) num=32 | **`St16x128bOp(8)`** | **DIVERGE D1 — staging atom shape** (see below) |

#### Divergences found (recorded for Wave 3 / agent-31; NOT fixed here)

**D1 — shared_inp staging atom: `St.32x32b` vs `St.16x128b`.** nymph stages
all (128,64) bf16 TMEM operands (deltaᵀ, NV, gated-NV) with the default
32x32b datapath; flashinfer uses 16x128b.x8 (CK:3259). Same element volume
(64 bf16/thread), different lane/datapath pattern → the STTM per-copy factor
differs from the flashinfer dataset's ≈×4 coefficient; a nymph-side
LDTM/STTM diff here is expected until this is matched.

**D2 — NV/vks/decay_v go through an extra SMEM round-trip.** flashinfer:
acc → regs → `St16x128b` straight into TMEM. nymph: acc → regs →
`stmatrix.trans` → SMEM (`_read128_delta` L1125, `_read128_vnew` L1155-1159)
→ per-element `reg_load` (L956, L994-998) → `tcgen05_st` → TMEM
(L957, L996, L999). Cost per steady chunk: +3×32 warp-ops STSM.x1.trans
(delta, vnewt, vnew-gated), +~128 LDS/thread, plus the STTM store that
flashinfer also has. Root cause is D5's orientation: the transposed GEMM
operand cannot flow from the 16x256b readback fragment straight into the
TMEM operand layout, so it is transposed through SMEM.

**D3 — QS scaling bounces through SMEM instead of TMEM in-place.**
flashinfer: QS scaled in TMEM (`Ld16x256b` 2 + `St16x256b` 2, CK:3478-3490),
GEMM 6 accumulates on top, one final O readback. nymph: QS readback → scale
→ `stmatrix` to `out_s` SMEM (`_read128_ointer` L1128-1140), then O_intra
readback + SMEM add + `stmatrix` again (`_read128_store_out` L1098-1109).
nymph has NO `St16x256b` counterpart anywhere; +2×32 warp-ops STSM/chunk and
two acc readbacks where flashinfer scales in place.

**D4 — stmatrix/ldmatrix are `num=1` (x1) everywhere in nymph.** `_stm`
(L1088-1095) and the inverse `_store8`/`_ldA`/`_ldB` (L824-849) issue one
m8n8 tile per instruction; flashinfer packs 4 (`StMatrix8x8x16bOp(4)`,
§2d). Same bytes moved, ≈4× the STSM/LDSM instruction count for the same
volume — expect a large STSM/LDSM opcode diff that is count-level, not
layout-level (flashinfer's own x1 usage is confined to inverse stage 2).

**D5 — acc orientation (the root structural difference).** nymph computes
GEMMs 3-6 as (M=64, N=128) non-transposed accs (`issue(...)` L716-745:
`m=BT=64, n=V_DIM=128`); flashinfer computes the transposed (M=128, N=64)
accs. UTCHMMA count is unchanged (m64n128k16 vs m128n64k16, same kphases,
44/chunk steady both). Consequences: (a) the (64,128) readbacks use the
M=64 16x256b scatter layout (nymph kernel comment L316-317) — atom matches
flashinfer's, lane pattern does not; (b) D2/D3's SMEM bounces; (c) nymph's
GEMM 3/4/5/6 put the TMEM operand on the **B side** (`state_inp` /
`shared_inp` with `trans_b`, L716-745) where flashinfer puts it on the A
side (`OperandSourceA=TMEM`) — hardware note for the codegen: tcgen05.mma
takes A from TMEM or SMEM but B from SMEM only, so the TMEM-B form has no
direct PTX lowering (agent-31's call); (d) state rows are K in nymph vs V
in flashinfer (square tensor; gmem layouts already differ accordingly —
handled by the wave-1 bench transpose).

**D6 — mma.sync chain shape (adjacent finding, beyond tcgen05 scope).**
nymph's inverse merge (`_merge` L851-875) uses `mma_sync(m=16,n=8,k=8)` for
ALL stages: per chunk stage2 4 blocks×2 = 8, stage3 2×16 = 32, stage4
1×128 = 128 → **168× m16n8k8**, vs flashinfer's 8× m16n8k8 + 40× m16n8k16
(§2e). Same merge math (newC = −Qinv·C·Pinv), different instruction mix —
the ncu HMMA anchor will split differently (all HMMA.882). Also: nymph's
stage-1 GJ inversion is an SMEM-load + FFMA chain (L790-812), NOT
flashinfer's shuffle-broadcast chain (SHFL anchor will differ too). FLOP
equality per stage was spot-checked structurally, not proven — Wave 3
should keep an eye on the stage-4 block (nymph's per-block mma count grows
as (b/8)³).

#### Docstring-vs-implementation check

nymph kernel docstring "M=64 read-backs to 16x256b … M=128 to .32x32b" is
CONSISTENT with the implementation: nymph's accs are all M=64 (16x256b ✓),
only the (128,128) state is M=128 (32x32b ✓). The §2c suspicion of a
32x32b-on-(128,64)-acc divergence is hereby closed: no such site exists.
The docstring's "GEMM3/4 read S directly from TMEM as the B operand … f32
TMEM operand" is imprecise on one point: the GEMM operand is the 16-bit
`state_inp` copy, not the f32 `s_tmem` (L934 comment "fp16 copy" is right).

### 2e. mma.sync (SM80 HMMA) — the WY inverse, CG0 only

A_inv = (I + M_kk)⁻¹ hierarchical blockwise inverse, in-place on sAinv
reinterpreted row-major bf16, barriers `inverse`(128) between stages and
`inverse_inner`(64) inside stage 4 (CK:2494-2528, helpers CK:2580-2934).
Per chunk (BT=64; source comments citing 16/8/2 blocks are stale BT=128 text):

| stage | blocks (BT=64) | warps | MMA per block | ldmatrix/stmatrix | site |
|---|---|---|---|---|---|
| 1: GJ-invert diagonal 8×8 | 8 | 0-1 (tidx//8) | 0 — shuffle pivot chain, 7 steps × (1 SHFL + FFMA)/row elem | — | `_invert_diagonal_NxN` CK:2612 |
| 2: 8→16 off-diag | 4 | 0-3, 1 tile each | 2× m16n8k8 (`MmaF16BF16Op (16,8,8)` CK:2648) | ld.x1, ld.x1.trans ×2; st.x1 | `_blockwise_diagonal_8x8_to_16x16` CK:2641 |
| 3: 16→32 | 2 | 0-1 | 2 gemm calls × (16/16)·(16/8)·(16/16) = 4 m16n8k16 (atom (16,8,16), perm (16,16,16) CK:2755-2760) | ld.x4, ld.x4.trans ×2; st.x4 | `_blockwise_diagonal_16x16_to_32x32` CK:2748 |
| 4: 32→64 | 1 | 0-1, half slab each | 2 calls × (16/16)·(32/8)·(32/16) = 16 m16n8k16 (perm (16,32,32) CK:2846-2851) | ld.x4, ld.x4.trans ×2; st.x4 | `_blockwise_diagonal_32x32_to_64x64` CK:2839 |

Per-chunk totals *derived*: **8× HMMA m16n8k8 + 40× HMMA m16n8k16** (stage 2:
4 tiles×2 = 8; stage 3: 2×4 = 8; stage 4: 2 warps×16 = 32 → 40 m16n8k16),
f32 accumulators, both gemms of the `newC = −Qinv·C·Pinv` pair per tile, with
a negate + acc→A-operand repack (`_make_acc_tensor_into_a_view` CK:2590)
between them. Stage 1 is shuffle-only: per 8×8 block, 7 pivot steps, each
step up to 7 SHFL.IDX + conditional FFMA per row — mask `0b1100000011111`
(CK:2630) is part of the numerics (restricted broadcast).

**HARD**: mma.sync shapes and counts (m16n8k8 vs m16n8k16 in SASS: HMMA.882
vs HMMA.16816 families), f32 accumulate, the 8→16→32→64 stage structure with
exactly 2 MMAs per correction block, ldmatrix operands (with `.trans` on
C/A), stmatrix results, stage-1 shuffle GJ chain. **FREE**: register repack
details between the two gemms of a block.

### 2f. gate/beta warp (warp 10) and special math

Per chunk (CK:1846-1947), BT=64 → 2 elements/lane:
- gate: 2× LDG.32 (universal copy, last chunk predicated with OOB fill 1.0),
  `log2(x + 1e-10, fastmath=True)` ×2 (MUFU.LG2 + add, CK:1902),
  inclusive prefix sum: 5 Hillis-Steele steps over offsets 1/2/4/8/16
  (SHFL.UP + predicated FADD, ×2 element cols = 10 SHFL, CK:1903-1909),
  cross-column carry: 1 SHFL.IDX from lane 31 + FADD (CK:1912-1917),
  `exp2(fastmath=True)` ×2 (MUFU.EX2, CK:1920), 2× STS.32 ×2 (cumsumlog +
  cumprod), software `commit()` (CK:1930).
- beta: 2× cp.async (LDGSTS, cache mode ALWAYS, 32-bit) into sBeta, last
  chunk predicated + pre-zeroed SMEM (CK:1934-1946), commit.

T-pairwise in CG0 (overlapped with GEMMs): 32 exp2f per thread (MUFU.EX2)
covering its (2,16) fragment (CK:2361-2371).

**HARD**: log2-with-1e-10-guard and exp2 in fastmath (numerics gate:
bit-exactness vs the oracle rides on these), the 5-step shuffle prefix-sum
shape, beta via cp.async not TMA, LDG (not TMA) for gate. **FREE**: shuffle
mask constants, whether carry uses SHFL.IDX vs SMEM bounce.

### 2g. cp.async

Only one cp.async site in the whole kernel: beta load (§2f). Everything else
bulk is TMA. `cp_async_bulk_commit_group`/`wait_group 0` appears once per
chunk in the epilogue (CK:3641-3642). **HARD**: beta = LDGSTS; **FREE**:
cache-mode flag.

## 3. Hyperparameters

| param | value | source |
|---|---|---|
| BT (chunk) | 64 | CK:398 (`self.b_t`) |
| DK / DV | 128 / 128 | __call__ shapes; state (128,128) |
| mma_tiler_qk | (64, 64, 128) | pinned CK:357-360; adapter passes it `gdn_prefill.py:175` |
| mma_tiler_qs | (128, 64, 128) | pinned CK:361-364; `gdn_prefill.py:176` |
| mma_tiler_qkv | (128, 64, 64) | pinned CK:365-368; `gdn_prefill.py:177` |
| mma_tiler_kv | (128, 128, 64) | pinned CK:369-372; `gdn_prefill.py:178` |
| io dtype | Float16 or BFloat16 (bench: bf16) | CK:349-352 |
| acc dtype | Float32 everywhere (state too) | CK:353-356 |
| state gmem dtype | f32 default (sm100 also bf16/f16/fp8) | `gdn_prefill.py:76-91` |
| cta_group / cluster | ONE / (1,1,1), no multicast | CK:258-262 |
| occupancy | 1 CTA/SM (`min_blocks_per_mp=1`) | CK:838 |
| SMEM stages | q=1 k=2 v=1 ainv=1 qk=1 o=1 gate=1 beta=1 | CK:299-308 |
| TMEM stages | state=1 q_state=1 state_inp=1 shared_acc=2 shared_inp=2 | CK:313-317 |
| register budget | 224 / 256 / 24 | CK:241-243 |
| persistent | True (adapter); grid = (min(B·H_o, num_sm),1,1) | `gdn_prefill.py:185` |
| checkpoints | supported, off for the bench path | `gdn_prefill.py:184` |

The tilers are *not* free: `can_implement` rejects any other value, so the
adapter hardcodes them (`gdn_prefill.py:175-178`).

## 4. Tensor allocation

### 4a. SMEM (≈96.8 KiB + mbarriers; all 1024 B aligned, CK:333)

Traced cosizes (cutlass-dsl 4.5.2, exact `make_smem_layout_*` calls):

| buffer | tile (elems bf16) | stages | bytes | swizzle (inner) | role |
|---|---|---|---|---|---|
| sQ | (64,128) | 1 | 16384 | S<3,4,3> (SW128) | A of GEMM 2 (CK:594) |
| sK | (64,128) | 2 | 32768 | S<3,4,3> | B of GEMMs 1/2 (CK:597) |
| sK_trans | view of sK, MN-major (DK,BT) | 2 | (shared) | S<3,4,3> | B of GEMM 7 (CK:600-601, 959-961) |
| sV | (128,64) | 1 | 16384 | S<3,4,3> | A of GEMM 5 first chunk (CK:604) |
| sAinv | (64,64) | 1 | 8192 | S<3,4,3> | M_kk → A_inv; B of GEMM 5 (CK:608) |
| sQk | (64,64) | 1 | 8192 | S<3,4,3> | W_qkv; B of GEMM 6 (CK:612) |
| sO | (128,64) | 1 | 16384 | S<3,4,3> | O epilogue staging (CK:616) |
| cumsumlog / cumprod / beta | 64 f32 each | 1 | 768 total | none (flat) | CK:623-626 |
| mbarriers (14 pipelines) | 34× Int64 | — | 272 | — | CK:635-682 |
| tmem_holding_buf | Int32 | — | 4 | — | CK:684 |

Data buffers: 99,072 B ≈ 96.8 KiB; +272 B barriers +4 B TMEM token. The header
table in the kernel docstring (CK:57-67, "225.5 KB, 32768 B buffers") is
**stale BT=128 text** — trust the traced numbers, not the comment. sAinv is
first M_kk/A_inv then reused; sQk is W_qkv (naming: comments say "QK output /
O store" but the O staging buffer is sO — another stale comment).

### 4b. TMEM (512 columns total; f32 = 32 lanes·b/col... column = 128 lanes × 4 B)

Offsets (CK:319-331; region = stages × cols/stage):

| region | offset (col) | cols | contents |
|---|---|---|---|
| state | 0 | 128 | Sᵀ (128,128) f32, GEMM 7 acc + CG1 RMW |
| q_state_acc | 128 | 64 | (128,64) f32, GEMM 4 + GEMM 6 acc → O |
| state_inp | 192 | 64 | (128,128) bf16, A operand of GEMMs 3/4 (staged by CG1) |
| shared_acc | 256 | 2×64 | (64,64)/(128,64) f32 accs of GEMMs 1/2/3/5, cycled |
| shared_inp | 384 | 2×64 | (128,64) bf16 A operands of GEMMs 5/6/7 (vks/nv/decay_v cycled ×3) |

Total exactly 512 cols = full SM100 TMEM; allocated by CG1 warp 4 via
`TmemAllocator` with the 288-thread named barrier (CK:987-993, 1222-1225),
freed at tile-loop end after `relinquish_alloc_permit` (CK:1360-1361).
Allocation token in SMEM (`tmem_holding_buf`).

### 4c. Register fragments (the inverse chain)

- CG0 T-pairwise fragment: (2,16) f32 = 32 regs/thread covering the (64,64)
  T matrix row-slice; live across kk_epi+qk_epi (CK:2361; comment claims
  128 regs — stale).
- kk/qk readback fragments: 32 f32 in + 32 bf16 out per thread (CK:2405-2408).
- Inverse stage regs: per stage, A/B fragments via ldmatrix (§2d), two
  (16,8)-atom f32 accumulators per correction block, acc repacked as the
  next gemm's A operand (ratio-2 reshape for k16 atoms, CK:2590-2609).
- CG1 steady-state live set: state fragment 128 f32 during kv_decay (the
  biggest single live range — drives the 256-reg budget), KS/V/NV fragments
  64 f32 / 64 bf16.

**HARD (all of §4)**: SMEM per-buffer tile shape/stages/bytes and S<3,4,3>
swizzle (bank-conflict behavior feeds MemoryWorkloadAnalysis); TMEM column
plan 128/64/64/128/128 at offsets 0/128/192/256/384 (512 total); state held
as Sᵀ f32 in TMEM with bf16 shadow in state_inp; SMEM ≈96.8 KiB so it fits with
1 CTA/SM. **FREE**: SMEM base-address order/alignment padding; register
fragment repack mechanics; tensormap workspace layout.

## 5. Tile scheduler

`gated_delta_net_tile_scheduler.py`. Tile = one (batch, head) pair; the
assigned CTA loops over ALL chunks of the tile sequentially (state
propagates chunk-by-chunk inside the CTA's TMEM; TS:32-34).

- num_work = B · H_o, H_o = num_q_heads if is_GQA (HQ≥HV) else num_v_heads
  (TS:76-81) — i.e. B · max(HQ, HV). Head mapping for operands: flat
  `head_idx` ∈ [0, H_o) over the hierarchical `(h_r, h_qv)` layout; the
  broadcast side has stride 0 (CK:408-452, 1678-1680).
- Persistent (the adapter's mode): grid = (min(num_work, num_sm=148), 1, 1)
  (TS:132-139); linear tile idx starts at bidx and advances by
  num_persistent_ctas (grid-stride round-robin, TS:256-260); decode is
  head-major `linear = batch·H_o + head` via two FastDivmod divmods
  (TS:228-237). Non-persistent: 2D grid (B, H_o), one tile per CTA.
- varlen: per tile, every warp reads `cu_seqlens[b]`, derives
  `seqlen_b`/`num_chunks_b = ceil_div(seqlen_b, 64)` at runtime
  (CK:1176-1178 etc.). Last chunk OOB: TMA bounded descriptors zero-fill
  inputs and clamp output rows (§2a); gate/beta warp predicates to
  gate=1/beta=0 (§2f); chunk loops are per-warp data-dependent.
- Zero-length sequences work: num_chunks_b = 0 skips all loops (bench
  shapes avoid them anyway).

**HARD**: persistent grid-stride with head-major linear order and
num_work = B·max(HQ,HV) (CTA→tile assignment affects L2 locality and the
multi-tile/CTA bench shape ns48_t64); per-tile sequential chunk processing
(state never crosses CTAs). **FREE**: FastDivmod vs plain div; the 2D
non-persistent mode (unused).

## 6. Wave-3 alignment checklist (summary)

Hard alignment points, in the order ncu diffs should be eliminated:

1. 44 UTCHMMA/chunk (kind f16, cta_group 1): 8+8+8+8+4+4+4 with tilers
   (64,64,128)×2, (128,64,128)×2, (128,64,64)×2, (128,128,64)×1; GEMM 3/4 A
   from TMEM, GEMM 7 B = Kᵀ MN-major; ACCUMULATE pattern per §2b.
2. TMA: 3 G2S + 1 S2G per chunk, 16 KB each, K 2-stage, per-tile descriptor
   updates; UTMALDG/UTMASTG counts match.
3. TMEM column plan 512 cols at offsets 0/128/192/256/384 with the 5 regions
   of §4b; state = Sᵀ f32 (128 cols) + bf16 shadow (state_inp).
4. tcgen05.ld/st atoms per site: 16x256b.x8 for every (64,64)/(128,64) f32
   acc readback (incl. q_state writeback St16x256b.x8), 32x32b.x32 for the
   (128,128) f32 state, St32x32b.x16 for (128,128) bf16 state_inp,
   St16x128b.x8 for (128,64) bf16 shared_inp. (The nymph-side
   "M=128 → 32x32b" suspicion is RESOLVED — no such divergence at the
   readbacks; the real gaps are §2d-supp D1-D6, above all the orientation
   flip D5 and the shared_inp staging atom D1.)
5. mma.sync inverse chain: 8× m16n8k8 + 40× m16n8k16 per chunk, f32 acc,
   ldmatrix x1/x4 (+trans on C/A), stmatrix x1/x4, shuffle-only stage 1.
6. stmatrix epilogue staging: GEMM-operand SMEM writes go through stmatrix,
   never plain STS — CG0's kk/qk results and the A_inv beta-pass (x4,
   no-trans), CG1's O staging (x4.trans); NV/vks/decay_v instead go to TMEM
   via St16x128b (§2c), V is read via ldmatrix.x4.trans.
7. Gate warp: LDG gate + log2(x+1e-10) MUFU + 5-step SHFL prefix + MUFU.EX2
   cumprod + STS; beta = cp.async LDGSTS. Numerics (bit-exactness vs oracle)
   ride on the fastmath log2/exp2 and the +1e-10 guard.
8. 12-warp/384-thread/1-CTA shape with reg budgets 224/256/24 and the
   14-mbarrier topology/stage depths of §1 (incl. kv_acc RMW serialization
   and K double-buffer).
9. SMEM buffer plan of §4a (96.8 KiB, S<3,4,3> everywhere, k=2 stages);
   shared_acc/shared_inp 2-stage cycles.
10. Persistent scheduler: grid min(B·H_o, 148), head-major linear order,
    per-tile sequential chunks, TMA-bounded varlen OOB handling (gate=1,
    beta=0 neutral fill).

Free bits (do not chase): SMEM base offsets/padding, named-barrier IDs,
descriptor workspace layout, shuffle mask constants, FastDivmod,
position-independent-layout addressing trick, pipeline class names in the
nymph IR, and the exact sub-loop trip counts where only the atom shape and
element volume matter.

Stale comments in the flashinfer source (do not propagate): SMEM "225.5 KB"
table (CK:57-67) and "128 regs T-pairwise" (CK:2358) and inverse block counts
"16/8/2" (CK:2495-2522) and "Thread tidx owns 4 positions" (CK:1820) are all
BT=128-era text; "GEMMs 3/4 shared_acc advance" comment (CK:2470-2474) should
read "GEMM 3 (ks) and GEMM 5 (nv) stages"; A/B operand comments at
CK:2064-2087 are swapped (see §0 trap); sQk's "QK output / O store" comment
(CK:63, 706-709, 969) predates the dedicated sO buffer.
