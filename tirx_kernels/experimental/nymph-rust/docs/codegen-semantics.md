# Codegen Semantics

This document records the IR → TVMScript lowering performed by `kernel_to_tirx_source`
(`src/ir/codegen.rs`): a Rust pass over a validated `ir::Kernel` that emits a TVMScript
(`@T.prim_func`) **source string**, which the harness `exec`s and feeds to
`tvm.compile(tir_pipeline="tirx")`. It is the codegen analog of
[Interpreter Semantics](interpreter-semantics.md): rows describe what each statement
emits, not hardware behavior.

The codegen consumes the **same** `ir::Kernel` the interpreter and protocol checker
validate, so any kernel that compiles to silicon was first proven deadlock/race-free and
value-exact in simulation.

## The genericity invariant

The codegen is a **generic 1:1 translator**. Every emission decision is a function of
exactly four inputs, none of which is a shape, an `N`, a `GEMM_CONFIG`, or any runtime
value:

1. **The `Stmt` variant** — one match arm per variant; the body of the kernel is walked
   in order and each node prints its fixed TVMScript form.
2. **The emit `scope`** (`Function` / `Warp` / `Warpgroup` / `Elected`) — a structural
   property of *where in the role tree* the node sits, set by the enclosing `Role` /
   `KernelInit`, NOT by which shape is being built. (See "Scope" below.)
3. **IR flags carried on the node itself** — `Role { elected }`, `ForLoop { unroll }`.
   These are written into the IR by the kernel builder; the codegen only honors them.
4. **The dtype / element width** — handled uniformly for every dtype via `dtype_str`
   and `element_bytes` (no dtype is special-cased beyond its name/size).

There is **no** `is_overlap` flag, no `if num_m_tiles == …`, no `match shape`, no branch
on the config knobs (`mma_n`, `pipe_depth`, `wb_pipe_depth`, …) anywhere in the codegen.
Per-shape behavior comes entirely from the *IR the builder produced* (different shapes
build different IR; the codegen prints whatever IR it is given). The "Genericity audit"
section below enumerates every branch in the file and classifies it against this
invariant.

## Global state (`Ctx`)

`Ctx` is built once by `build_ctx(&Kernel)` and is **read-only** during emission. It holds
only deterministic name maps and structural metadata derived from the kernel — never a
shape parameter or a per-config knob.

| Field | Contents | Derivation (genericity) |
| --- | --- | --- |
| `names` | tensor id → emitted name (`A`/`B`/`C` args by position; `ab_smem0`, `tmem`, `d_smem0`, … by role/order) | deterministic by id order; no shape input |
| `mbar_names` | mbar id → name (`smem_full`, `tmem_empty`, `task_full`, …) | deterministic by id order |
| `peer_names` | mbar id → peer-view name for `map_shared_rank` remote arrives | one per `MBarRef(remote_coord)` in the IR |
| `mbar_stages` | mbar id → stage count | read off each `MBarDef` |
| `var_names` / `scalar_names` | loop-var / scalar id → name (`v0`, `s4`, …); scalars emit as SSA `T.int32` register vars (not `alloc_local(1)` cells) | deterministic by id order |
| `cta_group` | cluster CTA-group (1 or 2) | the kernel's cluster shape |
| `tmem_view_cols`, `sf_views` | the single f32 `tmem` view's column count + the e4m3 SF view decls (`SFA_tmem`/`SFB_tmem`) | structural: `max(base_col + n_cols)` over the kernel's `TmemAlloc`s; each SF view's folded logical shape from its `TmemOperand` base column |
| `reg_widths` | REG-tensor id → collapsed band width | `walk_reg_widths` takes `max(offset+width)` over every slice of each REG fragment in the body — a property of the IR's accesses, not the shape |
| `nonneg_vars` | var ids provably non-negative (the `%`/`//` strength-reduction gate) | computed by `collect_nonneg_vars`: ForLoop induction vars with a non-negative-literal start and positive-literal step, plus scalar vars whose every definition is provably non-negative (a fixpoint over `ScalarDef`/`ScalarStore`/`ScalarLet`/`ShuffleSync`; mailbox loads and `ClcQueryCancel` results are never assumed) |
| `tma_leader_mbars` | the mbar ids whose TMA tx is routed to CTA-0 (`smem_full`'s/`sf_full`'s leader view) | **explicit IR**: `MBar::leader_routed` set by the builder; validate requires a peer reference and TMA-load/expect_tx-only use. No structural guessing |

## Scope

`emit_stmt` carries a `Scope` that reflects the enclosing role frame:

- **`Function`** — top level of the prim_func (prologue / teardown). All CTA threads
  reach it, so CTA-wide collectives (`cta_sync`, an all-thread `fence.proxy_async`) are
  legal here.
- **`Warp` / `Warpgroup`** — inside a single-warp / single-warpgroup `Role` branch
  (`if warp_id == w:` / `if wg_id == c:`). Single-issue ops get their per-op
  `if lane_id==0:` / `if tid_in_wg==0:` guard here; CTA-wide collectives are suppressed
  (not all CTA threads reach them).
- **`Elected`** — inside an `elected` role (`if warp_id==w: if elect_sync():`), where the
  whole body runs on one thread, so per-op single-issue guards are dropped.

Scope is the *only* thing that changes how a `Fence` / `CtaSync` / collective prints, and
it is determined by the role tree, never by the shape. This is the mechanism that lets
the same `Fence(AsyncProxy)` IR node print all-thread in the prologue and single-thread in
the epilogue — generically, by position.

## Statement Families

This table covers the `Stmt` variants the GEMM lowering implements. The codegen is
intentionally partial, and partiality always surfaces as an `Err`, on two axes:

- **Unlowered variants** (the flash-attention set: `WarpMma`, `RegUnary`, `RegFill`,
  `GmemAtomicAdd`/`GmemWaitEq`, `CpAsyncBulkS2Cluster`, `SchedNext`, `LdMatrix`/
  `StMatrix`, …) return `Err("codegen: … not yet supported")` — each with an
  explicit arm in `emit_stmt`. There is NO catch-all: match exhaustiveness makes
  a variant added without a lowering decision a **compile error**, and the
  `variant_coverage_tests` gate (see below) greps all four consumers
  (validate / codegen / interpreter dispatch / protocol checker) per variant.
- **Unrepresentable field values on lowered variants**: the emitted TVMScript form
  fixes some conventions, and any IR value outside them is rejected rather than
  silently coerced — `TmaStore.reduce_add` (no `Tx.copy_async` reduce dispatch),
  per-op `cta_group` differing from the kernel-level engine group (TmaLoad / MMA /
  commit), MMA `trans_a`/`trans_b`/`lane_align != 0`, `a_fp4 != b_fp4`, any
  block-scaled mode other than NVFP4 (`sf_e4m3=true, sf_block=16, sf_byte=0`),
  a `ScalarOp` with no TVMScript lowering (e.g. `Xor`), and a `ForLoop { unroll }`
  whose range is not literal start=0/step=1.

### Scale-factor classification (usage-derived)

A tensor is treated as an NVFP4 scale factor **iff the IR uses it as one**: the
`sfa`/`sfb` operand of a `Tcgen05Mma`, an endpoint of a `tcgen05.cp` staging copy, or
the GMEM source of a `TmaLoad` filling an SF SMEM ring (`collect_sf_ids`). dtype is
never consulted — a plain fp8 e4m3 *data* tensor keeps its normal MMA swizzle/layout.
SF-classified tensors get canon's `sf_smem_layout(rows, sf_k, sf_per_mma=4)` /
SF-TMEM decl forms; the constants there (`sf_per_mma=4`, per-super-block `SF_K=16`)
are NVFP4 *format* invariants (`CTA_K // SF_BLOCK` with `SF_BLOCK=16`), not shape
parameters. An SF-classified tensor with a non-e4m3 dtype is an `Err`.

| Statement family | Emitted TVMScript | Conditional emission (and why it is generic) |
| --- | --- | --- |
| `TensorDef` (SMEM) | `T.alloc_buffer(shape, dt, scope="shared", layout=mma_shared_layout(...))` or a swizzle/`decl_buffer` form by the tensor's declared layout | branches on `space`/`layout`/`is_int_dtype` only — a property of the tensor, uniform across shapes |
| `TensorDef` (REG) | `name = T.wg_reg_tile(width, dtype=…)`, emitted **inline at the TensorDef site** (loop-local) | width from `reg_widths` (the IR's own slice extents); inline so ptxas scopes liveness per task. No shape input. |
| `TensorDef` (TMEM) | nothing here (the TMEM view buffer is declared once after `tcgen05.alloc`) | — |
| `MBarDef` | nothing here (mbar inits are emitted in `KernelInit`) | — |
| `KernelInit` | `if warp_id==0:` guard around the warp-0 body (tmem alloc + mbar inits) | the `warp` field on the node; emits ONLY the body — the prologue sync (`fence.mbarrier_init`, the cross-CTA barrier) is explicit IR, not fabricated here |
| `KernelFinalize` | `if warp_id==w:` guard around the body (relinquish + dealloc) | the `warp` field on the node |
| `Role` | `if warp_id==w:` / `if wg_id==c:`; if `elected`, a single role-wide `if elect_sync():` then the body at `Scope::Elected` | the `warp`/`warpgroup`/`elected` fields. A **leading** `ClusterBarrierWait` is hoisted out of the elect to warp scope (a cluster wait is warp-collective; a single-thread wait deadlocks) — a structural rule on the node, not a shape |
| `If` | `if <cond>:` + body | 1:1 |
| `ForLoop` | `for v in T.serial(N):` — or `for v in T.unroll(N):` when `unroll` is set | the `unroll` IR flag only; `T.serial` keeps the induction var in a uniform register. `unroll` requires literal start=0/step=1 (the emitted `T.unroll(stop)` form) — anything else fails closed with an `Err` |
| `Loop` / `BreakIf` | `while True:` / `if <cond>: break` | 1:1 |
| `SchedulerImpl` | transparent — emits its body (the explicit CLC scheduler IR) | 1:1 |
| `ScalarDef` / `ScalarStore` | `name: T.int32 = <init>` / `name = <expr>` (a mutable `local_scalar` cell in the tirx parser) | 1:1; kept for genuinely loop-carried values (ring counters, task id) |
| `ScalarLet` | `name: T.let[T.int32] = <expr>` (an immutable `T.Bind` SSA value — canon's tile-decode form) | 1:1; single-assignment (validate rejects `ScalarStore` to it); used for per-iteration derived values so the def-use chain is pure SSA |
| `ShuffleSync` / `ClcQueryCancel` | `name: T.int32 = T.cuda.__shfl_sync(...)` / `= T.ptx.clc_query_cancel(...)` | 1:1 |
| `ClcTryCancel` | `T.ptx.clc_try_cancel(handle, mbar)` | 1:1 — the `cta_group` field has no emission site (TIRx's clc lowering implies the multicast width) but is NOT unchecked: validate requires it == the kernel-level cta_group and codegen re-checks (`Err`) |
| `StoreScalar` | `task_smem[stage, field] = <value>` (SMEM mailbox write) | 1:1 |
| `MBarrierInit` | `T.ptx.mbarrier.init(bar.ptr_to([stage]), count)` | 1:1 |
| `MBarrierArrive` | `T.ptx.mbarrier.arrive(...)` (local) or `arrive(..., cta_id=…, pred=…)` (remote) | branches on `MBarRef.remote_coord` presence — a property of the node |
| `MBarrierExpectTx` / `MBarrierArriveExpectTx` | `arrive.expect_tx(bar, bytes)` (a `leader_routed` mbar's expect_tx nests under `if cbx == 0:` and targets the CTA-0 `_cta0` view with the full cluster byte count) | the `MBar::leader_routed` IR flag — explicit routing metadata, not a shape |
| `MBarrierWait` | `T.ptx.mbarrier.try_wait(bar.ptr_to([stage]), phase)`; a *run* of consecutive waits is coalesced, and at **function scope only** a trailing `tcgen05.fence.after_thread_sync() + cta_sync()` is appended | scope-based (`scope.is_function()`): the prologue needs the init-visibility fence; role-scope hot loops do not (the mbar handshake orders the async engines) |
| `TmaLoad` | `Tx.copy_async(smem[slice], gmem[slice], dispatch="tma", cta_group=…, mbar=…, prefetch_tensormap=True)` | the per-CTA A-row / B-band split is in the IR's coords; codegen prints them. The transfer size is derived from the tile extents — the IR `bytes` is cross-checked, not emitted: validate rejects a statically-known mismatch with the tile (shape × dtype), and the interpreter re-checks per launch (`tma_bytes_mismatch`) |
| `TmaStore` | `Tx.copy_async(C[slice], d_smem[slice], dispatch="tma", …)` | 1:1 |
| `Tcgen05Mma` | `Tx.gemm_async(tmem[band], a_smem[slice], b_smem[slice], accum=<bool>, dispatch="tcgen05", cta_group=…)` | 1:1 — the IR carries the MMA at the FULL k-tile granularity: a dense f16/bf16 `k` is any positive multiple of the k=16 atom (validate reads it as an ordered run of atomic MMAs), exactly canon's one-issue full-K `gemm_async` (a 16-wide sub-slice of a 128B-swizzle atom would fault). There is NO run-collapse pass — the IR, not codegen, owns the MMA granularity |
| `Tcgen05Commit` | `T.ptx.tcgen05.commit(bar.ptr_to([stage]), cta_group=…, cta_mask=…)` | 1:1 |
| `Tcgen05Ld` | `Tx.wg.copy_async(reg[:, col:col+w], tmem[:, col:col+w])` | 1:1 |
| `Tcgen05WaitLd` | `T.ptx.tcgen05.wait.ld()` | 1:1 |
| `RegCvt` / `RegStore` | `Tx.wg.cast(out[slice], in[slice])` / `Tx.wg.copy(smem[slice], reg[slice])` | 1:1; the lane-axis (`tid_in_wg`) row offset is recognized structurally for the slice form |
| `Fence` | `MbarrierInit` → `fence.mbarrier_init()`; `AsyncProxy` → `fence.proxy_async(space)` with the IR `scope` lowered 1:1 — `Cta` → `"shared::cta"`, `Cluster` → `"shared::cluster"`, `Gpu` → the unqualified form (orders every address space); **all-thread at function scope, single-thread (`if tid_in_wg==0:`) in a role**. `Memory`/`View` are sim-only ordering markers — codegen `Err` | scope-based, generic; the kind and scope are read off the node |
| `CtaSync` | `T.cuda.cta_sync()` — **only at function scope** (a partial-CTA `__syncthreads` is illegal) | scope-based |
| `ClusterSync` | `T.cuda.cluster_sync()` | 1:1 |
| `ClusterBarrierArrive` | `T.ptx.barrier.cluster.arrive(sem="relaxed", aligned=True)` | 1:1 |
| `ClusterBarrierWait` | `T.ptx.barrier.cluster.wait(acquire=True, aligned=False)`; **errors** (`Err`) if reached under `Scope::Elected` | a guard, not a shape — a single-thread cluster wait deadlocks, so the codegen fails loudly instead of emitting it |
| `WarpSync` | nothing (the elect/issue structure already orders the warp) | — |
| `WgSync` | `T.cuda.warpgroup_sync(barrier_id)` | 1:1 |
| `CpAsyncBulkCommitGroup` / `CpAsyncBulkWaitGroupRead` | `T.ptx.cp_async.bulk.commit_group()` / `wait_group(n, read=True)` | 1:1 |
| `TmemAlloc` / `TmemDealloc` | `T.ptx.tcgen05.alloc/dealloc(addr, n_cols, cta_group)` (+ `relinquish_alloc_permit`) | 1:1 |

## Genericity audit

Every branch in `src/ir/codegen.rs` falls into one of five generic categories. None tests
a shape, an `N`, a config knob, or a runtime value. (Line numbers approximate.) The rows
below are representative; the remaining branches not listed (the `swizzle_for_row_bytes`
row-byte thresholds, the `emit_expr` div/mod strength-reduction, the
`MBarrierArrive` `count==1` vs explicit-count form, the `RegStore` SMEM-vs-GMEM dst split,
the `strip_tid_in_wg` lane-term strip) each fall into the same dtype / tensor-property /
emit-scope / structural / expression-printing categories — none introduces a shape or
config dependence. Decisively, the config knobs the doc disclaims (`mma_n`, `pipe_depth`,
`wb_pipe_depth`, `l2_group_size`, `overlap_epilogue`) are **not fields on `Kernel`** at all
— the codegen reads only `args`, `body`, `num_warps`, `launch_cta_count()`, and
`cluster_shape`, so there is nothing shape-specific for it to branch on.

| Branch | Category | Why generic |
| --- | --- | --- |
| `dtype_str`, `element_bytes`, `is_int_dtype` (~170, 200, 211) | **dtype** | one uniform mapping per dtype; every dtype handled the same way |
| `t.space != Smem` / `t.layout.is_some()` (261, 368, 811, 824) | **tensor property** | SMEM vs REG vs TMEM and the declared layout — read off the tensor, not the shape |
| `reg_widths` / `walk_reg_widths` (992, 1066) | **IR-derived width** | `max(offset+width)` over the IR's own slices; a different shape simply has different slices |
| `MBar::leader_routed` (build_ctx) | **IR flag** | explicit routing metadata set by the builder on the cluster TMA-completion barriers; codegen honors it, never infers it. Validate enforces consistency (peer reference present; TmaLoad/expect_tx-only use) |
| `cta_id_in_cluster([cx, cy])` from `k.cluster_shape` | **kernel field** | the cluster geometry is a Kernel field, not a shape knob; a rank>2 cluster fails closed |
| vendored `mma_shared_layout` header helper | **self-containment** | the emitted source re-implements the layout helper on the public `tvm.tirx.layout` algebra — no TVM-private import path; verified layout-identical to TVM's for every (dtype, mode, shape) the kernels use |
| `WARP_LANES`/`WG_WARPS`/`WG_THREADS` consts | **hardware constant** | warp width / warpgroup width / TMEM lane rows are silicon invariants, not kernel parameters; `num_warps` (validated a multiple of 4) only decides the warpgroup COUNT |
| `scope.is_function()` (1525, 2262, 2276), `CtaSync` gate (2273) | **emit scope** | prologue (all CTA threads converge) vs role (partial) — position in the role tree |
| `matches!(scope, Scope::Elected)` on `ClusterBarrierWait` (2300) | **emit scope (guard)** | a warp-collective op must not be single-thread; fails loudly |
| `if *elected` on `Role` (1745), leading-`ClusterBarrierWait` hoist (1760) | **IR flag + structural** | the node's `elected` flag; the hoist is a structural rule on the node's body |
| `if *unroll` on `ForLoop` (1794) | **IR flag** | the builder set it; `T.unroll` vs `T.serial`. `unroll` with a non-literal start=0/step=1 range fails closed |
| `MBarRef.remote_coord` present (in `MBarrierArrive`) | **node property** | local vs remote arrive — read off the ref |
| `matches!(off, TidInWg)` (1398, 2393) | **structural axis** | the per-thread row axis of a wg-collective slice |
| `needs_paren` (1313), operator/precedence tables (1117, 1177) | **expression printing** | standard precedence; value-independent |
| mbarrier-wait-run coalescing (1484, 1509) | **structural run-merge** | collapses a run of consecutive `MBarrierWait`; structure, not shape |
| `merge_guards` single-issue-guard merge | **structured run-merge** | folds adjacent same-guard single-issue blocks keyed on the emission-site guard ANNOTATION (a line attribute), never on the line text — an IR `If` with a guard-looking condition is untouched |
| `fill_empty_blocks` empty-block `pass` fill | **structural** | any block opener (`if`/`else`/`for`/`while`/`def` …) with no body line gets a `pass` — an empty KernelInit/Role/If body otherwise renders invalid Python; one generic line-structure rule, not a per-construct ban |
| `is_nonneg` / `nonneg_vars` (the `%`/`//` strength-reduction gate) | **provable expression property** | the pow2 bit-op rewrite is an unconditional two's-complement identity (no gate); the trunc rewrite for other positive divisors is gated on the computed provably-non-negative var set — a sentinel-negative scalar keeps its floordiv/floormod form |
| unsupported `ScalarOp` (e.g. `Xor`) | **fail closed** | a codegen `Err`, never a placeholder literal in the Python source |
| `arg_name` (A–Z then `arg{i}`) | **id-derived naming** | positional, cosmetic (TVM matches args positionally); no fixed table to overflow |

The only place a *number* literal influences output is a tensor's declared shape/layout
and the REG-band widths — all of which are themselves properties of the IR the
builder emitted, printed verbatim. Two different bench shapes reach the codegen as two
different `ir::Kernel`s; the codegen contains no code that asks which shape it is.

## Exhaustiveness gate

`variant_coverage_tests::every_stmt_variant_is_consumed_everywhere` (in `src/ir/mod.rs`)
parses the `Stmt` variant list out of `stmt.rs` and requires every variant to appear in
all four consumers: `validate.rs` (`validate_stmt`), `codegen.rs` (`emit_stmt`), the
interpreter dispatch (`registry.rs::stmt_kind`), and the protocol checker's metadata walk
(`checker.rs::walk_tensors`). None of the four matches has a wildcard arm, so adding a
variant without a decision is additionally a **compile error** (the codegen/checker arms
for unlowered variants are explicit `Err` / explicit no-op entries, so a text-level grep
also stays truthful).

## Boundary

The codegen emits TVMScript text; it does not run, schedule, or allocate registers. Register
allocation, async-engine latency, swizzle lowering, and the SASS instruction stream are the
TVMScript-pipeline + ptxas's responsibility below this layer. What the codegen guarantees is
that the emitted source is a faithful, shape-agnostic transcription of the validated IR —
the same IR the interpreter and protocol checker accepted.
