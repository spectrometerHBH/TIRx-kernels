# IR op semantics ledger (nymph IR vs PTX ISA)

Per-op record of what the interpreter models, what it deliberately abstracts,
and which hardware configurations are modeled / rejected / silently accepted.
Audience: kernel authors (human or agent) deciding how far to trust the
value simulator + protocol checker.

Reference: PTX ISA 9.3 (https://docs.nvidia.com/cuda/parallel-thread-execution/),
tcgen05 in §9.7.17, mbarrier §9.7.14.16, async copy §9.7.9.26, cluster
§9.7.14.3, CLC §9.7.14.18-19. Section numbers per op below.

Coverage states: **modeled+checked** | **rejected** (validate / interpreter
fails closed) | **SILENT** (accepted but semantics undefined — the dangerous
class; being eliminated) | **inexpressible** (no IR field, unreachable).

"Fixed in audit batch" = fail-closed rule added on branch
`refactor/nymph-tmem-codegen-genericity` (see its commits); earlier batches:
commit 76600421 (codegen-side rejections + sf_block validation).

## Global modeling abstractions (apply to every op)

1. **Async made synchronous**: TMA / CLC / tcgen05 effects land at issue in the
   value model. Consequence: a read-after-issue-before-wait sees NEW data in
   sim (old data on HW) — that bug class is only caught by trace-mode
   `invalidate_block` + the protocol checker's async windows.
2. **expect-tx must precede complete-tx** (sim rejects negative tx); HW tx-count
   is signed and tolerates complete-before-expect — a real CLC race exists on
   GPU that sim cannot reproduce (codegen.rs comment documents it).
3. **CLC round-robin oracle** (trusted seam): `ClcQueryCancel` returns
   canonical round-robin tasks. Holds for ANY order: each task exactly once,
   termination. The oracle is parameterized (`check_protocol(...,
   clc_oracle_offset=k)` rotates each cluster's steal order within its own
   residue class — k=0 is the canonical order), and the protocol suite runs
   the fp16/bf16 GEMM under multiple offsets (test_protocol.py). Still
   canonical-order-only: value regressions and the mailbox stage/phase reuse
   rhythm. Full assignment-confluence is a documented non-goal (RFC §5).
4. **Value at issue for MMA/cp**; completion observed via commit/mbar. Same-
   stream order makes this sound; the checker drains windows at the commit.
5. **Accumulation order**: MMA via OpenBLAS sgemm (BLAS blocking order), not
   the HW's fixed tensor-core order. Exact for exact arithmetic; f32-rounding-
   level differences otherwise. Hardware fixtures use exact small integers.
6. **elect = fixed lane 0**; PTX elect.sync picks deterministically but does
   not promise lane 0 (low risk).
7. **sim⇄GPU bit-exactness is measured, not assumed**:
   `tests/gpu/test_gpu_sim_parity.py` runs one kernel + one input set through
   BOTH the value simulator and the tirx codegen + `tvm.compile` on a real GPU
   and compares the outputs bitwise. On B200, the nvfp4 1024³ (exact-recipe
   e2m1 codes / power-of-two e4m3 scales / power-of-two alpha) and fp16 1024³
   (small-integer inputs) GEMMs match sim bit-for-bit — for exact-arithmetic
   recipes, abstractions 1/4/5 introduce no observable divergence. (Skipped
   without a CUDA device.)

## tcgen05.mma (§9.7.17.10.9, Table 42 §9.7.17.2.1, Tables 45-47, 54-55, 59-60)

Modeled: kind::f16 (cg1 m∈{64,128}, cg2 m∈{128,256}, Layouts A/B/D/F — B200
fixtures), mxf8f6f4 (k=32, UE8M0), mxf4nvf4 (k=64, e4m3 block-16, cg2 m=256),
k=128/256 as explicit IR-level k-tile folding.

Envelope gaps → **fixed in audit batch** (validate fail-closed):
- k×dtype coupling: dense no-sf ⇒ k any positive multiple of 16 (one IR MMA
  with k=16q means the q atomic k=16 MMAs accumulated in issue order — canon's
  one-issue full-K gemm_async, lowered by TVM to the atom sequence); UE8M0 sf
  ⇒ k∈{32,128,256}; fp4 ⇒ k∈{64,128,256}
- cg1 m=64 ⇒ no scale modes (mxf* cg1 is M=128 only)
- fp4 ⇒ (cg1,m=128)|(cg2,m=256), and no trans (Table 54); trace mode aligned
- B operand must be SMEM (PTX residency table); A-in-TMEM kept (GDN state)
- mixed-dtype exception narrowed to f32×f32 (tf32 path)
- lane_align!=0 ⇒ only (cg1, m=64); sf_e4m3 ⇒ sf_byte==0

Known abstractions: SF held 1 byte/cell, 128-lane flat, no 4x-subpartition
duplication (HW packs 4 e4m3/u32 cell + duplicates; value-equal, TMEM column
footprint ~4x — column-budget reasoning does not match HW); tf32 operands
computed as full f32 (HW truncates RZ — exact only for tf32-representable
inputs); issue granularity counted once (PTX is single-thread issue).

SILENT (documented, NOT yet rejected — do not rely on):
- in-place accum=true on unwritten cells: in-place path treats as 0.0,
  fallback path errors `missing_tmem_value` (audit batch aligns to error)
- cg2 odd-CTA issue is silently dropped; PTX allows EITHER CTA of the pair to
  issue. Kernels must issue from the leader (cta_rank==0) — enforced by
  convention only.
- no peer-active gate on mma (commit has one)

Rejected today, HW-legal (completeness backlog, deliberate for now):
f8f6f4 dense (no-sf f8), i8, mxf4+UE8M0 (block32), D=f16, B=f32-SMEM tf32,
n-granularity finer than current (cg1 m=128 step 8; cg2 step 16), .ws, .sp,
disable_output_lane, K=96 (sm_103a).

## tcgen05.cp (§9.7.17.9.2) — SF staging copy

This op is a value-level abstraction of canon's `Tx.copy_async` SF datapath,
NOT a raw PTX tcgen05.cp: HW shapes (.128x256b/.4x256b/...), .warpx4,
decompression formats are all inexpressible. Modeled: u32 (UE8M0, lane-major
flat stream) and e4m3 (nvfp4, super-block folded) with numel-equal src/dst.

Envelope gaps → **fixed in audit batch**: dst/src dtype must be equal
(was: e4m3→u32 dst silently wrote bytes into word cells); src layout
constraints (u32 ⇒ 1-D; e4m3 ⇒ dst cols % src last-dim == 0); issuer check
(full warp or single elected lane, aligned with mma); peer-active gate for
cta_group=2.

## tcgen05.ld / tcgen05.st (§9.7.17.8, Tables 52-53)

Modeled: all five shapes × legal num (atom tables transcribed from CUTLASS
SM100 TMEM copy traits; b32 verified cell-exact on B200 via
`tests/tcgen05_ldst_hardware.rs`, all 35 (shape,num) pairs, both directions).
Illegal num rejected (validate + runtime). Row-alignment rejected (B200-
measured). Divergent taddr / non-full-warp rejected.

Envelope gaps → **fixed in audit batch**: row+atom_span must stay within the
issuing warp's 32-lane subpartition (was: silent cross-subpartition access =
HW UB); wait::ld/st get full-warp cohort checks.

Known limitations: 16x32bx2 fixes immHalfSplitoff=num (arbitrary splitoff
inexpressible); f16/bf16 packed path models mat-D packing (lo|hi<<16 per
word) — NOT verified against HW pack::16b wording, no f16 fixture (marked
suspect); REG slice declares reg_size f16 elems while 2*reg_size are moved
(validate accepts by declaration; OOB caught only at runtime — wart).

## tcgen05.commit (§9.7.17.12.1) / wait (§9.7.17.8.5)

Modeled: arrive-one per target; multicast_cta_mask retargets same-offset mbar
(mask 0 / out-of-cluster rejected); cta_group=2 peer-active gate modeled.
wait::ld/st are markers in value mode, drain points in trace/checker. Both
documented abstractions, no known divergence.

## tcgen05.alloc / dealloc / relinquish_alloc_permit (§9.7.17.7)

Modeled: physical column-band lifecycle (overlap, order, mismatch, missing —
all fail closed), cta_group=2 two-CTA collective rendezvous, and the alloc
PERMIT: `relinquish_alloc_permit` flips a per-CTA flag (idempotent — giving
the permit up twice is a no-op); a later `tcgen05.alloc` targeting a
relinquished CTA errors `tmem_alloc_after_relinquish` (PTX §9.7.17.7.1).

The IR deliberately admits only what the codegen lowers — the generated code
carries ONE base-0 TMEM view fed by a single alloc — so validate walk 4
additionally rejects (**fixed in audit batch**): `base_col != 0`, a second
concurrently-live alloc, a lifecycle op whose cta_group differs from the
kernel-level group (cluster-size derived, mirroring the codegen per-op
checks of commit 76600421), and any alloc after a relinquish. Codegen
re-checks base_col/cta_group per op (it is reachable without validate);
before this batch it dropped both fields via `..`, emitting one alloc for
any number of IR bands. Walk 4 also rejects lifecycle ops (alloc/dealloc/
relinquish) nested inside a `ForLoop`/`Loop`/`If`/`ForEachTask`/
`SchedulerImpl` body OR inside a nested `Role` body — a loop-carried or
subset-cohort lifecycle is only safe under a path-sensitive analysis the
IR does not have yet (a `for_loop(stop=2)` alloc passed the one-pass walk
but double-allocated on iteration two; a `Role(warp0){ Role(warp1){ … } }`
alloc never executed and only the protocol check reported it missing).

The protocol checker's `tmem_lifecycle_order` pass proves the lifetime
against ALL legal interleavings, not the sampled one, over explicit
per-band generation intervals (alloc_i, dealloc_i): a re-allocation must be
happens-before-after the previous generation's dealloc; every TMEM access
binds to the unique generation whose alloc is happens-before it without
that dealloc intervening (no binding →
`tmem_lifecycle_use_without_allocation`); and a bound access must be
happens-before its generation's dealloc (cross-stream edges must come from
real synchronization — mbar handshakes, fused cta/cluster syncs, a
fence-sealed split cluster barrier — recorded in `OrderingAnalysis`; the
sampled epoch interleaving alone proves nothing). One relaxation, scoped
to the kernel-TEARDOWN dealloc (`KernelFinalize`): an access drained on
its own stream (`tcgen05.wait::ld/st`, or a `tcgen05.commit` whose mbar
is waited) counts as complete without the barrier edge — canon's
GPU-verified teardown shape, see LIMITATIONS.md. Missing edges fail
`tmem_lifecycle_hb_missing`.

## tcgen05.ld / tcgen05.st codegen support set

The interpreter models all five shapes; the CODEGEN lowers only
`tcgen05.ld` with **shape=32x32b, row=static 0, dtype=f32** — the single
(128, cols) f32 base-0 view window `Tx.wg.copy_async` encodes (a `.16x*b`
atom reads col_factor×num columns from a 16-lane half-slab; the dropped
`..` used to emit the 32x32b text for it anyway). Anything else fails
closed at codegen (**fixed in audit batch**). `tcgen05.st` and
`tcgen05.wait::st` stay sim-only (codegen `Err`). The `.16x*b` atoms DO
still reach silicon — via a REG fragment declared `reg_frag`, whose TIRx
dispatch is driven by the fragment layout, not by this IR field.

## mbarrier (§9.7.14.16)

Core algebra (init/arrive/expect_tx/complete_tx/phase completion/reset/parity
flip) matches PTX item-for-item; UB cases (over-arrive, tx underflow,
uninitialized, double init) all fail closed.

Envelope gaps → **fixed in audit batch**: init count ≤ 2^20-1; wait phase
required (was: sim used current parity while codegen emitted constant 0 —
latent divergence, no current kernel triggered it).

SILENT / deferred (load-bearing, needs kernel rework):
- **peer (remote_coord) wait is HW-illegal** (mbarrier ops are .shared::cta
  only) yet sim models it and fp16_bf16_gemm.py:583 uses it; codegen silently
  DROPS it (correctness there rests on a leader-routing argument in comments).
  Do not add new uses. Tracked for kernel rework.
- MBar.arrive_count is a dead field (declared, never consumed).

## TMA / bulk copy (§9.7.9.26)

Modeled: routing rules (cg2 unicast mbar ∈ {dst,peer}; cg2 multicast parity
routing — matches PTX), OOB clamp/zero-fill/squash, reduce_add (f32) with
checker TmaReduce events, commit/wait groups as counters (never block;
.read-vs-not distinction unmodeled). A float reduce_add is order-dependent, so
it is IR-level OPT-IN: validate rejects `TmaStore.reduce_add` on a non-integer
dst unless `allow_nondet_reduce` is set (the checker's
`nondeterministic_reduction` warning still fires with the flag).

SILENT / known divergences:
- **cg2 multicast to a SHARED (leader) mbar**: sim dedups tx per unique cell
  (completes once); HW signals each destination (2x bytes on the shared
  barrier). Validate now REJECTS this combination (audit batch); kernels
  already avoid it (nvfp4 SFB reads the full band per CTA instead).
- CpAsyncBulkS2Cluster: bytes==numel×dtype_bytes, %16, dst≠self — all added
  in audit batch (op is sim-only; codegen rejects it).
- cache_hint: free-form string → enum-checked in audit batch.

## Sync / fences / misc (§9.7.14.1-4, §9.7.14.15, §9.7.9.6, §9.7.20.5)

Modeled: CtaSync / WgSync / NamedBarrier (rendezvous + reuse), ClusterSync
(peer liveness), fences trace-only, SetMaxNReg range checks (interpreter
no-op by design).

Fence codegen (**fixed in audit batch**): `AsyncProxy` lowers its IR `scope`
1:1 to the `fence.proxy.async` space qualifier — Cta → `shared::cta`,
Cluster → `shared::cluster`, Gpu → unqualified (orders every space, exactly
the checker's `Gpu => covers any access`; a `.global` narrowing would run a
weaker fence than sim validates). Was: the scope field was `..`-dropped and
every proxy fence emitted `shared::cta`. `Memory`/`View` fences are sim-only
ordering markers (trace `Generic` events) with no TIRx lowering — codegen
now fails closed on them instead of silently emitting nothing.

Envelope gaps → **fixed in audit batch**: ShuffleSync src_lane must be in
[0,32) (was: negatives silently clamped to lane 0) and is REJECTED inside
elected roles (single-lane __shfl_sync(0xffffffff) is HW UB — nvfp4's MMA
warp had exactly this, fixed in the Phase-0 commit); SetMaxNReg warpgroup
index bounds; NamedBarrier/WgSync cross-kind barrier_id dedup.

Known abstractions: ClusterBarrierArrive/Wait modeled one-shot (HW auto-
reinits; reentry rejected); ClusterBarrierWait in an elected context is
codegen-fail-loud (single-lane cluster wait deadlocks on HW — sim does not
expose this); WarpSync not emitted by codegen (warp lockstep assumption);
CLC handle's 16B async-proxy write not modeled (race checker can't see it).

## CLC (§9.7.14.18-19)

Modeled: try_cancel → per-cluster slot + 16B complete-tx to cta_group CTAs
(matches .multicast::cluster::all); query-before-try rejected; drained → -1.
The oracle's steal order is rotatable per cluster via
`check_protocol`'s `clc_oracle_offset` (see global abstraction 3) — a cheap
way to re-run protocol checks under ≥2 task orders without touching the
runner; the offset stays within each cluster's residue class, so "each task
exactly once" and termination are preserved by construction.
Envelope gaps → **fixed in audit batch**: mbar kind must be TMA; handle ≥16B.
Deferred: re-try after drained is idempotent in sim, UB on HW (reject later).

## sim-only ops (codegen rejects — validated/simulated but not compilable)

GmemAtomicAdd / GmemWaitEq (value-keyed release/acquire; checker models
ordering), WarpMma, RegUnary, CpAsyncBulkS2Cluster, TmaStore.reduce_add
(no TIRx lowering). If a kernel needs these on GPU, the IR must grow a
lowering first — sim-green means nothing for silicon here.
