# Interpreter Architecture

The interpreter runs one bounded Nymph kernel by expanding every modeled GPU
thread, materializing one execution stream per `(cta, warp)`, and stepping
those streams in deterministic source order. Every stream runs the whole
kernel body top to bottom. A warp's 32 lanes share one instruction stream and
execute it under masked divergence; the lanes advance through it independently
(sm_70+ Independent Thread Scheduling), so a memory dependency between two
lanes of one warp is ordered by a warp-level sync — `warp_sync`, a
warp-collective instruction, or a cooperative barrier the warp passes. Above
the warp, everything is concurrency. All ordering is explicit: cross-lane,
cross-warp, and cross-CTA dependencies alike must be expressed by modeled
synchronization or explicit protocol handshakes, and the protocol checker
verifies each of them.

## Modules

- `runner.rs` - initialization, scheduler loop, dispatch, direct mutation, fatal
  failure handling, and precise mbarrier wake.
- `protocol.rs` - execution mode, protocol reports, normalized trace event
  types, and optional trace event retention.
- `scheduler.rs` - per-warp execution streams, frame stacks, thread expansion,
  and CTA activity tracking.
- `state.rs` - `InterpreterState`, `RunOptions`, and `RunResult`.
- `threads.rs` - `ThreadId`, `ThreadMask`, and canonical mask helpers.
- `ids.rs` - discovery-order identity helpers.
- `registry.rs` and `semantics/` - statement executor dispatch and built-in
  per-op executors.
- `cohort.rs`, `slice_indexing.rs`, `transfer.rs`, `elementwise.rs`, and
  `mbar_ops.rs` - vectorized execution surface and shared services used by
  statement executors.
- `values/` - runtime values for GMEM/SMEM tensors, registers, TMEM, mbarriers,
  scalars, cooperative sync, dtype coercion, and tcgen05 datapath helpers.
- top-level `tmem.rs` - internal TMEM allocation and collective bookkeeping.
- `blas.rs` - OpenBLAS-backed SGEMM path for supported tcgen05 MMA cases.

## Execution Surface

Built-in statements execute through `StmtExecutorRegistry` and
`CohortContext`. Each semantics module registers executors for its statement
kinds. `default_executor_registry()` builds the dispatch table by iterating
those registrars; adding a new op family should not require editing the runner.

Executors operate on a whole active cohort. Tensor and register value movement
is vectorized over the cohort; scalar and protocol metadata may still be
resolved per active thread when uniformity or scope values matter.

Rust uses a direct-mutation model. An executor receives `&mut CohortContext`,
mutates `ctx.state` directly after its local checks, and returns a light
`StepStatus`:

- `Advance { wakes }` - the statement completed and the frame should advance.
- `AdvanceContinue` - structural control pushed or advanced a frame without
  recording an executed leaf statement.
- `Block(WakeCondition)` - the stream is parked.
- `Fail` - the run aborts.

Failed value-mode runs do not expose partial value state. Completed value-mode
runs return `RunPayload::Value { outputs }`, where `outputs` contains the final
GMEM tensor instances. Trace-mode runs return `RunPayload::Trace { report,
events }` for Passed, Failed, and Inconclusive protocol outcomes. Full
`check_protocol` runs retain the event stream for offline passes and optional
Python `include_events=True` marshalling. Raw `nr.trace()` disables offline
checking and may return an empty Rust event vector; Python raw trace returns only
status and progress counters.

See [Protocol Trace](protocol-trace.md) for the trace data structures and the
per-statement event contract.

The fallback `leaf` executor fails closed with `unsupported_stmt`. The Rust port
does not expose the historical Python per-thread custom handler registry.

## Basic Model

For each CTA in `kernel.launch_shape`, the scheduler creates one `ThreadId` for
every `(warp_id, lane_id)`. A `ThreadMask` is an ordered set of threads from one
CTA, sorted by `(warp_id, lane_id)`.

An `ExecutionStream` is one warp's scheduled path through the kernel. The
scheduler materializes one stream per `(cta, warp)` eagerly at launch, each
with the warp's full 32-lane mask, and every stream executes the whole kernel
body from top to bottom. Each stream owns:

- a stream id;
- CTA and cluster coordinates;
- the initial active mask (the warp's 32 lanes);
- a frame stack with source bodies, program counters, and active masks;
- blocked state, tracked by the runner while the stream is parked.

The interpreter creates streams for every CTA. It never simulates only CTA 0.

## Scheduler Shape

The scheduler visits CTAs in deterministic order. Within a scheduler round, one
CTA visit can advance each of that CTA's live streams by at most one semantic
source statement.

```text
while some CTA schedule is not complete:
  snapshot CTA activity for this round
  for each live stream selected from CTA schedules:
    step one source statement
    refresh that CTA's activity before selecting more work
  if no stream progressed during the round:
    fail with deadlock
```

There are no epochs: no part of the kernel body is held back until another
part drains, and no implicit launch-wide, cluster-wide, CTA-wide, or
warp-pair barrier exists. If one warp must observe another warp's effects, the
kernel must order them explicitly — an mbarrier handshake, `cta_sync`,
`wg_sync`, `warp_sync`, or `cluster_sync`. `TensorDef` and `MBarDef` are
metadata declarations and have no dynamic stream effect.

## Thread Dispatch And Frames

All thread dispatch is `Stmt::If` over a free-form per-thread scalar predicate
built from thread-coordinate scope values: `warp_id()`, `lane_id()`,
`warpgroup_id()`, and `tid_in_wg()`. `If` evaluates its condition over the
active cohort, pushes a child frame containing only the true lanes, and
reconverges to the parent frame when the child drains. A divergent `If` never
splits a stream; masked execution inside one warp is exactly the hardware
model.

The builder provides canonical sugar over these predicates:

| Builder form | Predicate | Active threads |
| --- | --- | --- |
| `if_warp(n)` | `warp_id() == n` | all 32 lanes of warp `n` (one stream) |
| `if_warpgroup(n)` | `warpgroup_id() == n` | warps `4n..4n+4` (four streams) |
| `if_lane(n)` | `lane_id() == n` | lane `n` of every warp in context |
| `if_elected()` | `lane_id() == 0` | lane 0 of every warp in context |

`if_elected()` is per-warp election. One thread per warpgroup is
`if_(tid_in_wg().eq(0))`; one thread per CTA nests an election under
`if_warp(w)`.

A warpgroup selection is not one stream. The four selected warps are four
independent streams that each execute the guarded body concurrently; ordering
between them requires explicit sync (typically `wg_sync`).

`set_maxnreg(n)` emits `Stmt::SetMaxNReg`, a `setmaxnreg` directive for the
enclosing warpgroup(s). It is simulation metadata only — register pressure is
not modeled — carried in the IR so codegen can emit the PTX directive.

### Static thread filters

Validation rules that depend on which threads can reach a statement are
reachability statements over the enclosing dispatch predicates.
`static_thread_filter(cond, num_warps)` in `src/ir/thread_filter.rs` evaluates
a predicate at every `(warp, lane)` point of one CTA and returns
`Known(ThreadSet)` or `Unknown`. Predicates built purely from thread-coordinate
scope values and integer constants resolve to `Known`; anything touching a
runtime value or a CTA coordinate degrades to `Unknown`, and the static rule
skips — the runtime semantics (rendezvous accounting, issue gates) still own
the check. The rules that fire when the filter is `Known`:

- `cta_sync` and `cluster_sync` must be reachable by every thread of the CTA;
- `wg_sync` must cover exactly one full warpgroup;
- `warp_sync` must cover whole warps (a sub-warp barrier deadlocks);
- `tmem_alloc`/`tmem_dealloc` must be issued by exactly one full warp;
- `sched_next` must be confined to a single warp and, because a multi-warp
  `sched_next` silently hands each warp a different task, requires a
  statically-resolvable branch;
- `set_maxnreg` must cover whole warpgroup(s) and, having no runtime backstop,
  also requires a statically-resolvable branch.

### Frames

Nested `If`, `ForLoop`, `ForEachTask`, `SchedulerImpl`, and `Loop` statements
do not create new scheduler streams. They push or advance frames on the
current stream. Loop bounds and step must be uniform across the active mask;
divergent bounds fail with `divergent_loop_bounds`, and dynamic non-positive
steps fail with `invalid_loop_step`.

## Blocking And Wake

A stream that blocks is parked with a `WakeCondition`.

- `WakeCondition::Mbar { key, phase }` is precise. A later mutating statement
  returns the touched mbar cell keys in `Advance { wakes }`; the runner
  re-checks parked waiters and advances satisfied frames directly. The wait
  statement is not re-run.
- `WakeCondition::Polled` is retried each round. This is used for cooperative
  sync rendezvous, TMEM collective, and peer-active gates whose retry paths
  are idempotent. Cooperative syncs (`cta_sync`, `wg_sync`, `warp_sync`,
  `cluster_sync`) rendezvous in shared state with stream-granular arrival
  counts — a stream's cohort arrives atomically — so a blocked stream's
  re-poll is an O(1) hash lookup and count compare.

If an entire scheduler round makes no progress, the run fails with
`failure_reason="deadlock"` and returns `blocked_frontier` entries containing
the stream id, statement id, statement type, and block reason.

## Runtime Values

GMEM and SMEM values are dense tensors keyed by `(tensor, owner)`. GMEM has one
global owner; SMEM is owned by CTA. Missing shared destinations can be created
by value-mode stores when the statement semantics allow it.

Register values are vectorized register files keyed by `(reg tensor, cta_id)`.
Each instance stores rows for CTA-local threads and columns for tensor elements.
REG values are not generic tensor instances.

TMEM values are CTA-local physical scratchpads keyed by `(lane, col)`. TMEM
tensors are views over the scratchpad. `TmemAlloc` and `TmemDealloc` maintain
allocation metadata and ensure or clear scratchpad ranges.

Mbarriers are phase cells keyed by mbar identity, CTA, and stage. The model
tracks expected arrivals, pending arrivals, pending transaction bytes, and
parity. It models phase bookkeeping only, not hardware latency or memory-ordering
proofs.

## Testing Expectations

For interpreter changes, run at least:

```bash
cargo test --test interpreter_runner
cargo test --lib
```

For Python bridge changes, also run:

```bash
cargo test --features python --lib
cargo test --features python --test interpreter_runner
./run_python_tests.sh
```
