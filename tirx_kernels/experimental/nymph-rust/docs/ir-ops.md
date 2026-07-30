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
2. **Signed mbarrier transaction balance**: `expect_tx` subtracts expected
   bytes; TMA/CLC completion adds actual bytes. Both simulator and checker
   allow either instruction to happen first, so the intermediate balance may
   be negative or positive. The phase completes only when that balance and
   pending arrivals are both exactly zero.
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
6. **Election sugar = explicit lane 0**: `if_elected` builds the ordinary IR
   predicate `lane_id == 0`. It is not a PTX `elect.sync` operation, and
   codegen prints the predicate unchanged.
7. **sim⇄GPU bit-exactness is measured, not assumed**:
   `tests/gpu/test_gpu_sim_parity.py` runs one kernel + one input set through
   BOTH the value simulator and the tirx codegen + `tvm.compile` on a real GPU
   and compares the outputs bitwise. On B200, the nvfp4 1024³ (exact-recipe
   e2m1 codes / power-of-two e4m3 scales / power-of-two alpha) and fp16 1024³
   (small-integer inputs) GEMMs match sim bit-for-bit — for exact-arithmetic
   recipes, abstractions 1/4/5 introduce no observable divergence. (Skipped
   without a CUDA device.)

## Codegen emission guards: the zero-inference rule

**Codegen never synthesizes or rewrites an emission guard. Every condition
and every nesting edge in the emitted source comes from the IR.**

Hardware single-issue ops — `tma_load` / `tma_store` /
`cp_async_bulk_s2cluster` (the TMA issue family), `tcgen05_mma` /
`tcgen05_cp` / `tcgen05_commit`, `mbarrier_init` (an unguarded init is a
double-init error in both models), `clc_try_cancel` (stream-level in the
interpreter) — are legal ONLY under an explicit single-lane `If` in the IR:
the `if_elected` sugar (`lane_id == 0`), or any predicate PROVABLY selecting
at most one lane per warp (`tid_in_wg == 0`, `thread_rank == 0`, or an
`And`/`Mul` chain with at least one such operand — the runtime operands only
narrow the set, e.g. flash_bwd's `(~kept) & (tid == 0)`; see
`thread_filter.rs::proves_single_lane_per_warp`). The single-lane upper
bound is sticky: a deeper runtime branch (e.g. `cta_rank == 0`) only selects
a subset, so it cannot reintroduce double-issue.

- **Validator (`single_issue_scope`)**: a single-issue op outside such a
  branch fails `Kernel::validate` at build; `emit_single_issue` in codegen
  hard-errors as the second line of defense. Codegen emits the operation at
  its exact IR location and never wraps it in a new branch.
- **Per-thread ops emit per-thread**: `mbarrier.arrive` /
  `expect_tx` / `arrive_expect_tx`, `store_scalar`, and async-proxy fences —
  the interpreter applies them once per executing lane, so an inferred guard
  undercounts arrivals/tx-bytes (the gdn `gate_ready` deadlock class) or drops
  lane-varying stores. Sites that need one arrival/lane write elect it in the
  IR; the emitted code then matches 1:1.
- The scope is still *classified* (`static_thread_filter`) for legality checks
  (CTA-wide `cta_sync` at function scope, warp-collective forms, `Elected`
  recognition for the single-issue arms) — analysis of explicit IR conditions,
  never invention of new predicates.
- The condition of every `Stmt::If` is printed literally. Thus
  `if_elected`'s `lane_id == 0` remains `if lane_id == 0:`; it is never
  replaced by `T.ptx.elect_sync()` or `T.cuda.thread_rank() == 0`.
- Structural identity is part of the contract: sibling IR `If`s remain
  siblings, nested `If`s remain nested, and statement order is preserved.
  Codegen does not chain top-level branches, synthesize a warpgroup parent,
  merge equal predicates, hoist bodies, or re-nest statements.
- Negative coverage: `tests/test_compile_gate.py::test_single_issue_scope_negative`
  (build rejected without an explicit single-lane branch and passes with one)
  and
  `src/ir/validate.rs::tests::single_issue_scope_rule` (function scope /
  full-warp rejected, `if_elected` and `tid_in_wg == 0` accepted).

Historical inferred-guard and top-level-`If` chaining passes have been deleted.
If a kernel needs a shared guard or a nested decision tree, the kernel author
must build that exact tree in IR.


## tcgen05.mma (§9.7.17.10.9, Table 42 §9.7.17.2.1, Tables 45-47, 54-55, 59-60)

The IR is physical and self-contained:

- `dst` is a `TmemAddr`: `(row, TmemTensor.start_col + col)`.
- A is either one explicit `SmemTile` or a TMEM address plus
  `Flat`/`BankBatched` form; B is an explicit `SmemTile`.
- `mma_m`, `mma_n`, element format (`f16`, `bf16`, `f8_e4m3`, or
  `f4_e2m1`), `accum`, transpose flags, `ws`, and `cta_group` are instruction
  fields.
- A block-scaled form additionally carries explicit SFA/SFB TMEM addresses,
  scale format, `sf_per_mma`, and `sf_reuse`.

`tcgen05_layout.rs` is the single physical-geometry resolver used by
validation, interpreter, checker regions, and codegen. It derives the
instruction K extent from the operand tiles and format, selects the permitted
datapath for the explicit `mma_m`/CTA-group/`ws` combination, and computes
the exact D, optional TMEM-A, and scale-factor footprints. Unsupported
format/shape/residency/transpose combinations fail closed in that resolver.
There is no descriptor cache and no tensor-id or use-order inference.

Codegen declares non-owning D/A/SFA/SFB buffers immediately beside this
statement, at the exact address carried by the IR, using the resolved canonical
layouts. One IR statement emits exactly one `Tx.gemm_async` with explicit
`mma_m`, `mma_n`, `cta_group`, transpose, accumulation, and
weight-stationary arguments. TVM owns the lower-level descriptor encoding and
atomic instruction decomposition; Nymph does not move, merge, hoist, or split
the IR statement.

The interpreter consumes the same resolution, including canonical packed
scale-factor cells and their lane replicas. Every CTA whose explicit IR
control flow reaches an MMA executes it. If issue must be restricted to one
CTA, that condition must be written around the statement in the kernel; there
is no `leader_routed` metadata or implicit odd-CTA no-op.

Rejected today, HW-legal (completeness backlog, deliberate for now):
i8, mxf4+UE8M0 (block32), D=f16, `.sp`,
disable_output_lane, and K=96 (sm_103a). Other unsupported combinations are
reported by the shared resolver rather than silently approximated.

## tcgen05.cp (§9.7.17.9.2) — SF staging copy

The IR carries the physical interface needed by TIRx: destination
`TmemAddr`, explicit two-dimensional source `SmemTile`, `cta_group`, shape
(`128x256b`, `4x256b`, `128x128b`, `64x128b`, or `32x128b`), and multicast
(`none`, `warp2_02_13`, `warp2_01_23`, or `warp4`). The shared resolver
validates the legal shape/multicast pairing, source atom divisibility, packing
into 32-bit TMEM cells, and the exact destination lane/column footprint.

One IR statement creates one statement-local, non-owning TMEM destination
view and emits exactly one `Tx.copy_async` with the same shape, multicast, and
CTA group. It does not infer shape from tensor dimensions, unroll a copy into
multiple statements, or cache/relocate a destination view. The optional PTX
`dst_fmt/src_fmt` decompression qualifiers are not represented yet and
therefore are not silently selected.

Scale-factor sources are plain row-major SMEM tensors whose physical shape and
absolute byte offset are explicit. The CP statement names the exact tail tile
it reads; codegen and the interpreter consume that tile directly without a
scale-specific owning layout or usage-derived alias.

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
suspect). Direct codegen requires two f16/bf16 REG elements per physical b32
register, an even static slice offset, and fails closed otherwise.

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

The IR deliberately admits only what codegen lowers, so validate walk 4
rejects `base_col != 0`, a second concurrently-live allocation, a lifecycle op
whose CTA group differs from the kernel-level group, and any allocation after
relinquish. It also rejects lifecycle operations inside a re-executing body or
a runtime-value conditional; a statically classified warp/lane dispatch
`If` remains legal. Codegen re-checks the fields it consumes.

TMEM data is not represented by one kernel-wide buffer. `TmemTensor` stores
only an absolute `start_col`; each `TmemAddr` adds its explicit row and column.
Every data operation resolves its own footprint and emits a statement-local,
non-owning `decl_buffer` or a direct PTX address. `TmemAlloc` owns only the
physical column-band lifecycle and the four-byte shared-memory address cell
used by alloc/dealloc.

The protocol checker's `tmem_lifecycle_order` pass proves the lifetime
against ALL legal interleavings, not the sampled one, over explicit
per-band generation intervals (alloc_i, dealloc_i): a re-allocation must be
happens-before-after the previous generation's dealloc; every TMEM access
binds to the unique generation whose alloc is happens-before it without
that dealloc intervening (no binding →
`tmem_lifecycle_use_without_allocation`); and a bound access must be
retired — drained by `tcgen05.wait::ld/st` for a load/store, or by the
waited mbar of ANY covering `tcgen05.commit` for an mma/cp (a commit
tracks every prior async op of the warp, so a later commit covers the
access again and any one of those cells' waits counts) — with ONE such
drain happens-before the generation's dealloc (cross-stream edges must
come from
real synchronization — mbar handshakes, fused cta/cluster syncs, a
fence-sealed split cluster barrier — recorded in `OrderingAnalysis`; the
sampled epoch interleaving alone proves nothing). The rule is uniform:
no kernel-teardown exception, and a generation's dealloc must also be
happens-before-after its own alloc. Missing edges fail
`tmem_lifecycle_hb_missing`; an access with an ordering edge but no
observed completion fails `tmem_lifecycle_use_not_drained`.

## Dynamic shared-memory ownership

All owning shared-memory objects are views into one unconditional
`T.SMEMPool()`. SMEM tensors carry absolute `byte_offset`; mbarriers carry an
absolute 8-byte-aligned offset and occupy `stages * 8` bytes; `TmemAlloc`
carries the absolute 4-byte-aligned offset of its `tmem_addr` cell. Codegen
moves the pool base to each exact IR offset, allocates a `shared.dyn` view, and
commits once with `kernel.smem_size_bytes`, the complete physical extent
including explicit padding and metadata. It does not derive a metadata tail
from tensor ids or emission order, and it does not emit an owning
`T.alloc_buffer(scope="shared")`/`T.alloc_shared` path.

Scale factors use ordinary layoutless tensor IR. Their dtype, physical shape,
absolute `byte_offset`, and any same-offset aliases describe the storage
completely: FP8 uses plain raw/post shapes over the same bytes, while NVFP4
uses the explicit `(M_super, K_outer, 32, 16)` physical shape. Only a
statement-local non-owning `T.decl_buffer(layout=...)` is created for a
`Tcgen05Cp`/`Tcgen05Mma` call; it never changes the owning tensor.

## Direct datapath and register codegen

These operations do not pass through a logical warpgroup-tile abstraction:

- Every validated `Tcgen05Ld` emits one
  `T.ptx.tcgen05.ld(T.uint32(taddr), ...)` carrying the IR shape, `num`, and
  register tuple; every `Tcgen05St` analogously emits one
  `T.ptx.tcgen05.st(...)`. Wait statements emit the corresponding
  `T.ptx.tcgen05.wait.ld/st()` call. The TMEM address is the exact
  `start_col + col` and row encoded by the IR; codegen does not build a global
  TMEM data buffer.
- `LdMatrix` and `StMatrix` emit one `T.ptx.ldmatrix` or
  `T.ptx.stmatrix` using the explicit SMEM row address and packed REG handles.
  Supported b16 fragments are either `num` u32/i32 words or `2*num`
  f16/bf16 elements viewed as those words.
- `WarpMma` m16n8k8/m16n8k16 emits the non-legacy
  `T.ptx.mma(...)` intrinsic with the explicit A/B/C/D register fragments.
- A REG `TensorDef` emits exactly `T.alloc_local(IR_shape, IR_dtype)`.
  `RegLoad`/`RegStore` lower to `Tx.copy` over the exact IR slices;
  `RegCvt` lowers to `Tx.thread.cast`; supported fill/add/sub/mul/fma/unary
  operations lower to the matching `Tx.thread.*` call. Unsupported dtype,
  shape, rounding, or operation combinations fail closed rather than changing
  the register representation.

The interpreter retains the matching per-lane datapath mappings. Direct PTX
emission is a codegen rule, not permission to skip its validator or hardware
fixtures.

## mbarrier (§9.7.14.16)

Core algebra (init/arrive/expect_tx/complete_tx/phase completion/reset/parity
flip) uses a signed transaction balance: `expect_tx` subtracts expected bytes
and engine completion adds actual bytes. Either may execute first, producing a
negative or positive intermediate value; returning to exactly zero completes
the transaction side of the phase. Parity flips only when pending arrivals are
also zero. Invalid byte counts, over-arrive, uninitialized access, and double
init fail closed.

Envelope gaps → **fixed in audit batch**: init count ≤ 2^20-1; wait phase
required (was: sim used current parity while codegen emitted constant 0 —
latent divergence, no current kernel triggered it).

`MBarRef` is the complete address authority for every operation that contains
it. `remote_coord=None` names the local CTA's cell; a coordinate names that
exact peer cell. Arrive/arrive-expect use the intrinsic's explicit remote
operand, while expect, wait, TMA completion, and commit use the corresponding
mapped shared pointer. Codegen and interpreter do not infer an address from
mbar id, tensor id, producer/consumer role, or statement order. There is no
`leader_routed` flag and a remote coordinate is never silently dropped.
Where an operation has an explicit multicast mask, only that mask may expand
the set of completion targets.

`MBar.arrive_count` remains declaration metadata; executable arrival counts
come from the operation and active lanes.

## TMA / bulk copy (§9.7.9.26)

Modeled: routing rules (cg2 unicast mbar ∈ {dst,peer}; cg2 multicast parity
routing — matches PTX), including one complete-tx contribution per multicast
destination even when multiple destinations resolve to the same explicit
remote mbar cell; OOB clamp/zero-fill/squash; reduce_add (f32) with checker
TmaReduce events; commit/wait groups as counters (never block; .read-vs-not
distinction unmodeled). A float reduce_add is order-dependent, so it is
IR-level OPT-IN: validate rejects `TmaStore.reduce_add` on a non-integer dst
unless `allow_nondet_reduce` is set (the checker's
`nondeterministic_reduction` warning still fires with the flag).

SILENT / known divergences:
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
expose this); `WarpSync` emits `T.cuda.warp_sync()`;
CLC handle's 16B async-proxy write not modeled (race checker can't see it).

## Happens-before join points (per-op strength)

The only constructs that produce cross-lane or cross-warp happens-before,
each with its modeled strength and PTX basis. Anything not listed orders
nothing beyond per-lane program order. A new op adds its entry when it
joins. The contract these serve is `docs/execution-model.md`.

**Warp CONVERGENCE points** — fold all 32 lanes into the warp-shared
prefix; the op's own effects become warp-visible once it completes
(visibility rule 1):

- `ldmatrix` / `stmatrix` — `.sync.aligned` (§9.7.13.4.15-16).
- `tcgen05.ld` / `tcgen05.st` — `.sync.aligned` (§9.7.17.8).
- warp MMA — `mma.sync.aligned` (§9.7.13).
- TMEM alloc / dealloc — `tcgen05.alloc/dealloc.sync.aligned` (§9.7.17.7),
  keyed on the trace event itself whatever statement emitted it.
- full-warp cooperative arrivals / passages — `bar{.arrive}` / `barrier`
  execute per-warp aligned (§9.7.14.15), so a passage whose mask covers the
  warp converges it. A PARTIAL-mask rendezvous converges nothing.

**Rendezvous edges** (release/acquire, no convergence): every cooperative
barrier orders each ARRIVING lane's published order into every passer —
`bar.sync` carries memory-barrier semantics (§9.7.14.15), so a 16-lane
member of a 32-thread named barrier still receives the other arrivers'
writes, while lanes absent from the rendezvous are published by nobody.

**Cross-proxy publication** (visibility rule 4): `fence.proxy.async`
publishes the fencing thread's view into the async-proxy engines at the
fence's address scope, and every engine access acquires the view published
for the address space it touches. A thread fencing its own prior stores
covers those; other lanes' stores ride in only behind a convergence point or
a barrier. Nothing else crosses the boundary — an ordering edge alone does
not.

**Releases project their ARRIVING lanes only** (visibility rule 3):

- `mbarrier.arrive` / complete-tx — per-thread (§9.7.14.16, no `.aligned`):
  a full-warp arrive publishes all 32 lanes (each lane arrived itself); an
  elected arrive publishes one lane's order. Phase completion freezes the
  accumulated join for the waiters.
- semaphore release (`gmem_atomic_add` order=release) — value-keyed;
  relaxed publishes nothing (control order only).

Acquires join into the ACQUIRING lanes only: a masked wait delivers the
release to its lanes alone until a convergence point spreads it.

**Completion observation** (visibility rule 5): an engine access is ordered
by its completion object, not by its issue. A barrier that orders two
instruction streams says nothing about whether the engine has drained, so
anything that depends on an async access having LANDED — most sharply, freeing
the TMEM band it touches — must be ordered after the observation point:
`tcgen05.wait::ld/st` for a load or store, and for an mma or cp the wait on a
barrier some `tcgen05.commit` handed the work to. A commit tracks every async
op the warp issued before it, so a later commit covers the same work again and
waiting any one of those barriers suffices.

**Modeled WEAK** (over-report direction, by the ledger discipline):

- lane-0 selection — no election IR op; `if_elected` is a plain
  `If(lane_id == 0)` and codegen preserves it literally. No hardware
  `elect.sync` is introduced and no convergence is credited (a kernel needing
  the ordering writes `warp_sync`).
- `tcgen05.wait::ld/st` — drains the EXECUTING thread's own loads/stores
  (§9.7.17.8.5, per-thread); orders nothing across lanes.
- `tcgen05.commit` / `tcgen05.mma` — single-thread issues; no convergence.

**Seams to keep in view**:

- `WarpSync` is an explicit IR convergence point and lowers to
  `T.cuda.warp_sync()`; codegen never inserts one from inferred scope.
- The value simulator lands async-engine effects at issue, which is why the
  engine's real timing envelope is owned entirely by the checker's
  async-window passes (`async_group_lifetime`, `tcgen05_async_hazard`)
  rather than by the values.
- **`fence.mbarrier_init` seals, but nothing requires it.** The op exists
  (`FenceKind::MbarrierInit`) and the checker treats it as a release-side
  fence: it seals its executing lanes so a later relaxed cluster-barrier
  arrive by those lanes publishes. What is NOT checked is the obligation
  itself — a kernel that initializes barrier cells and lets a peer use them
  with no such fence is accepted, because `MbarInit` is an ordinary event on
  its stream and no pass demands publication of the barrier OBJECT (as
  opposed to the data a barrier hands over).
- **A new trace event kind orders nothing by default.** The scan that builds
  the clock decides acquires and releases with `match` arms that end in a

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
ordering), CpAsyncBulkS2Cluster, TmaStore.reduce_add, Fence Memory/View,
RegMax/RegMin/RegBitwise/RegReduce, and the specialized register
reduction/rescale family have no TIRx lowering. If a kernel needs these on
GPU, the IR must grow a lowering first — sim-green means nothing for silicon
here.

WarpMma, RegUnary, Tcgen05Ld/Tcgen05St, LdMatrix/StMatrix, RegCvt, and
RegLoad/RegStore are not sim-only: the supported forms lower directly as
described above and unsupported forms fail closed.
