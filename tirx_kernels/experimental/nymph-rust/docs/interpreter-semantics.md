# Interpreter Semantics

This document records the reviewed semantics boundary for the Rust interpreter.
Rows describe deterministic value-simulator behavior, not full hardware
ordering, latency, cache, descriptor, or compiler-lowering semantics.

Structural control statements are interpreter core: `If`, `ForLoop`,
`ForEachTask`, `SchedulerImpl`, `Loop`, and `BreakIf`. All thread dispatch is
`If` over per-thread scalar predicates. The same structure is authoritative
for codegen: predicates are printed literally, sibling/nested structure is
preserved, and no `lane_id == 0` condition is replaced with `elect_sync()` or
`thread_rank() == 0` — the `if_elected` sugar carries its own `Elected`
predicate (the elect.sync intrinsic), evaluated as lane 0 of each
fully-active warp. Leaf-style operation statements execute through built-in
executors. Unsupported statements fail closed with `unsupported_stmt`.

Each execution stream is one warp, and a statement executes over the stream's
active lane mask — the lanes that survive the enclosing `If` masks. Statement
semantics follow the PTX execution-thread model:

- per-thread instructions (`mbarrier` arrive/expect_tx/arrive_expect_tx):
  every executing thread applies its operand once; the trace carries one event
  per statement with summed counts;
- single-thread instructions (`tma_load`, `tma_store`, `tcgen05_mma`,
  `tcgen05_cp`, `tcgen05_commit`): the executing mask must be exactly one
  thread; `mbarrier_init` is per-thread and therefore also demands a
  single-thread branch, because every executing thread would re-initialize
  the cell;
- warp-collective datapaths (`tcgen05_ld`/`tcgen05_st` and
  `ldmatrix`/`stmatrix`, full-warp issue) and the vectorized register ops:
  each executing lane drives its own slice of the datapath.

In `ExecutionMode::Value`, REG/SMEM/GMEM values are stored in typed Rust/ndarray
containers. F16 and BF16 values are rounded at the typed boundary and then held
as f32-backed arrays for hot-path arithmetic. Integer arrays use fixed-width
native integer storage. TMEM is modeled as a CTA-local physical scratchpad of
32-bit cells.

In `ExecutionMode::Trace`, scalar/control/protocol state still executes, but
payload numeric work is skipped. Trace runs return `RunPayload::Trace` for
Passed, Failed, and Inconclusive outcomes; failed value runs return no payload.
The normalized trace data structures and per-statement event emissions are
documented in [Protocol Trace](protocol-trace.md).

Every mbarrier operation resolves the `MBarRef` it directly contains. A
missing `remote_coord` means the current CTA; a present coordinate means that
exact peer CTA. Arrive, expect/arrive-expect, wait, TMA completion, and
tcgen05 commit do not infer a target from mbar id, producer/consumer role, or
statement ordering; an operation's explicit multicast mask may add only the
targets that mask describes.

## Statement Families

| Statement family | Status | Runtime behavior | Main fail-closed conditions |
| --- | --- | --- | --- |
| `TensorDef`, `MBarDef` | Core metadata | Discovery metadata only; no dynamic execution stream effect. Mbar definitions seed runtime mbar identity availability. | Invalid metadata is rejected by IR validation. |
| `ScalarDef`, `ScalarStore`, `StoreScalar` | Reviewed scalar | Per-thread scalar values across the active lanes. `ScalarDef(initial=TensorSlice)` can load one GMEM/SMEM scalar in value mode, and in trace mode when the scalar cell is concrete and valid. `StoreScalar` writes a scalar expression to an SMEM scalar slice in both modes. | Missing input in value mode, invalid skipped-payload scalar bridge in trace mode, unsupported scalar dtype, unresolved expression, undefined variable. |
| `SetMaxNReg` | Reviewed metadata | `setmaxnreg` directive for the enclosing warpgroup(s); simulation metadata only, register pressure is not modeled. | Validation requires a statically-resolvable branch covering whole warpgroup(s) and `nreg` a positive multiple of 8. |
| `ForLoop` | Core structural | Requires uniform bounds and positive step; writes the loop variable before body execution; repeated body statements are distinct dynamic occurrences. | `divergent_loop_bounds`, `invalid_loop_step`, undefined loop variable use. |
| `ForEachTask` | Reviewed scheduler | Functional scheduler consumer loop. `grid_stride` maps each cluster to the canonical task subsequence and writes a task variable before each body execution. | Invalid scheduler metadata, unsupported policy, undefined task variable use. |
| `SchedulerImpl`, `SchedNext`, `Loop`, `BreakIf` | Reviewed scheduler | Concurrent scheduler body executes as ordinary stream code. `SchedNext` returns the canonical flat task index and `-1` terminal sentinel; `Loop`/`BreakIf` model runtime scheduler/consumer loops. | Invalid scheduler policy, missing dynamic loop for `BreakIf`, divergent/unresolved break condition, invalid task space. |
| `If` | Core structural | Evaluates the condition per thread over the active lanes, pushes a true-lane child frame, and reconverges to the parent frame. All thread dispatch (warp/warpgroup/lane selection) is expressed this way; `static_thread_filter` resolves canonical predicates for placement validation. | Unresolved condition. |
| `TmemAlloc`, `TmemDealloc` | Reviewed lifecycle | Issued by exactly one full warp (statically validated when the dispatch predicate resolves). `cta_group=1` acts on the issuing CTA. `cta_group=2` is a CTA-pair collective that blocks until the peer reaches the same collective occurrence. Allocation ensures a scratchpad; deallocation clears the physical range. | Invalid mask, missing peer, duplicate allocation, non-identical overlap, allocation order violation, missing/mismatched deallocation, leaked allocation. |
| `MBarrierInit` | Reviewed phase | Initializes a mbar cell with expected arrivals, pending arrivals, zero pending tx bytes, and parity 0. Per-thread instruction: every executing thread would re-initialize the cell, so it must execute from a single-thread branch. | Multi-lane mask, duplicate init, invalid stage, remote CTA out of range, divergent target. |
| `MBarrierArrive` | Reviewed phase | Per-thread: every executing thread arrives once with its own count; parity flips when arrivals and pending tx bytes reach zero. A phase can complete and re-arm partway through the lanes. The trace carries one event with the summed count. | Uninitialized cell, non-positive count, arrival underflow, invalid stage, divergent operands. |
| `MBarrierExpectTx` | Reviewed phase | Per-thread: every executing thread subtracts its expected byte count from the current phase's signed transaction balance; the trace carries one event with the summed bytes. A later engine completion adds actual bytes, but completion is equally allowed to occur first and leave a positive balance. Either order may be the operation that returns the balance to zero. | Uninitialized cell, invalid stage, divergent target. |
| `MBarrierArriveExpectTx` | Reviewed phase | Per-thread: each executing thread subtracts its expected bytes, then applies one arrival. Phase completion requires both the signed byte balance and pending arrivals to be exactly zero. | Uninitialized cell, arrival overflow/underflow, invalid stage, divergent target. |
| `MBarrierWait` | Reviewed phase | Blocks while the requested phase is current. Phase-less waits park on the current parity through a precise `WakeCondition::Mbar` and advance without re-running when parity flips. | Uninitialized cell, invalid phase, invalid stage, divergent target. |
| `TmaLoad` | Reviewed tile copy | Single-thread issue. Copies a GMEM tile into SMEM in value mode and adds its actual completion bytes to the signed balance resolved from the exact `MBarRef` plus any explicit multicast mask. Multicast writes all selected destination CTAs. Trace mode emits region events, completes mbar tx, and invalidates destination scalar cells without moving payload bytes. | Multi-thread issue, missing input in value mode, unsupported rank, OOB slice, byte-count mismatch, invalid multicast mask, missing peer, divergent operands. |
| `TmaStore` | Reviewed tile store | Single-thread issue. Copies a current-CTA SMEM tile into GMEM in value mode. Existing GMEM values are preserved outside the tile. Trace mode emits region events and invalidates destination scalar cells without reading SMEM bytes. | Multi-thread issue, missing SMEM source in value mode, unsupported rank, OOB slice, metadata mismatch, divergent operands. |
| `CpAsyncBulkCommitGroup`, `CpAsyncBulkWaitGroupRead` | Reviewed markers | Per-stream group markers only. Commit increments a stream counter; wait with `n=0` clears it. No tensor movement and no blocking in v1. | Nonzero wait count if it bypasses IR validation; trace limit. |
| `Tcgen05Mma` | Reviewed value/trace | Single-thread issue. The statement explicitly carries D `(TmemTensor.start_col + row/col)`, A residency/form, B SMEM tile, `mma_m/mma_n`, element format, optional block-scale addresses/format/reuse, transpose, weight-stationary, accumulation, and CTA group. The shared physical resolver determines the exact D/A/SF footprints used by validation, value execution, trace, and codegen. Value mode computes supported dense or block-scaled combinations into those TMEM cells; trace mode emits the resolved regions without payload BLAS work. Every CTA that reaches the statement executes it—there is no implicit leader or odd-CTA no-op. | Multi-thread issue, unsupported format/shape/group/layout combination, missing operands in value mode, missing accumulator/allocation, TMEM out of range. |
| `Tcgen05Cp` | Reviewed value/trace | Single-thread issue. The statement explicitly carries destination TMEM address, source two-dimensional SMEM tile, physical CP shape, multicast, and CTA group. The shared resolver maps that combination to the exact lane/column cells and reads the owning tensor's explicit physical bytes: 128-bit forms require a plain 16-byte row, while 256-bit forms require an explicit B32 `mma_operandAB` owner. | Multi-thread issue, unsupported shape/multicast combination, invalid source tile or owning layout, missing allocation, TMEM out of range. |
| `Tcgen05Commit` | Reviewed mbar bookkeeping | Single-thread issue. Immediately applies one mbar arrival to selected targets. `cta_group=2` is a peer-active gate, not a matched-operation rendezvous. | Multi-thread issue, uninitialized mbar, arrival overflow, invalid stage, missing/exited peer, divergent operands. |
| `Tcgen05Ld` | Reviewed value | Reads each active thread's datapath-assigned TMEM cells into register slices. Datapath arrays cover all supported shape/num configurations. | Non-full-warp issue, non-uniform or non-32-aligned row, dtype mismatch, wrong register count, out-of-range or unwritten TMEM cell. |
| `Tcgen05St` | Reviewed value | Writes register slices into each active thread's datapath-assigned TMEM cells. Uses the same datapath as `Tcgen05Ld`, reversed. | Non-full-warp issue, non-uniform or non-32-aligned row, dtype mismatch, wrong register count, out-of-range cell, overlapping writes, missing scratchpad. |
| `Tcgen05WaitLd`, `Tcgen05WaitSt` | Reviewed markers | Trace/value markers only. The value model copies synchronously at ld/st. | None beyond trace limit. |
| `LdMatrix` | Reviewed value/trace | Models PTX `ldmatrix.sync.aligned.m8n8.x{1,2,4}{.trans}.shared.b16`. Each active warp uses lane groups 0..7, 8..15, 16..23, and 24..31 as row-address providers for matrices 0..3. Value mode packs two raw b16 SMEM elements into each lane's b32 REG fragment. Trace mode records the exact SMEM row-address footprint and register write without reading payload bytes. | Non-full-warp issue, unsupported shape/num/type, wrong row slice or register fragment size, invalid dtype, OOB or unwritten SMEM cell in value mode. |
| `StMatrix` | Reviewed value/trace | Models PTX `stmatrix.sync.aligned.m8n8.x{1,2,4}{.trans}.shared.b16` as the inverse raw-bit scatter from b32 REG fragments to SMEM row-address slices. Trace mode records the exact register read and SMEM row-address footprint, then invalidates destination SMEM payload cells. | Non-full-warp issue, unsupported shape/num/type, wrong row slice or register fragment size, invalid dtype, overlapping SMEM writes, missing REG source in value mode. |
| `RegLoad`, `RegStore` | Reviewed value | Vectorized movement between register rows and SMEM/GMEM dense values. Values are coerced at the destination dtype boundary. | Missing source, OOB slice, metadata mismatch, overlapping shared/global writes. |
| `RegAdd`, `RegSub`, `RegMul`, `RegMax`, `RegMin` | Reviewed value | Vectorized REG ALU for f16/bf16/f32/i32/u32. Float results round to destination dtype; integer ops wrap to 32 bits, with signed i32 and unsigned u32 comparisons. | Missing source, OOB slice, metadata mismatch, nonnumeric dtype. |
| `RegFma` | Reviewed value | Vectorized REG `a * b + c` for f16/bf16/f32, rounded to destination dtype. | Missing source, OOB slice, metadata mismatch, unsupported dtype. |
| `RegCvt` | Reviewed value | Converts f32 REG values to f16 or bf16 using round-to-nearest-even. | Missing source, OOB slice, metadata mismatch, unsupported conversion. |
| `Fence` | Reviewed marker | Records an ordering marker with kind, scope, and active mask; mutates no runtime values. | Trace limit. |
| `CtaSync` | Reviewed rendezvous | Blocks until the full current CTA reaches the same occurrence; arrivals are stream-granular. | Partial CTA arrival deadlock; validation requires reachability by every CTA thread when the dispatch predicate is static. |
| `WgSync` | Reviewed rendezvous | Blocks until the current warpgroup reaches the same occurrence; `barrier_id` is part of the rendezvous key. | Partial warpgroup arrival deadlock; validation requires exactly one full warpgroup when the dispatch predicate is static. |
| `WarpSync` | Reviewed rendezvous | Blocks until every represented warp has full-lane arrival. | Partial warp arrival deadlock; validation requires whole-warp coverage when the dispatch predicate is static. |
| `ClusterSync` | Reviewed rendezvous | Blocks until every CTA in the current cluster reaches the same occurrence. | Missing or exited peer CTA; partial cluster arrival deadlock. |

## Boundary

The interpreter models deterministic statement-level value effects and protocol
bookkeeping. It does not model async operation queues, hardware instruction
latency, tensor-core exact accumulation order, cache effects, PTX memory
ordering, descriptor encoding, or backend swizzle lowering. Checkers and
hardware tests must consume the executed statement stream and runtime evidence
to validate those properties separately.

The codegen representation deliberately does not turn REG or TMEM metadata into
an extra execution model. REG `TensorDef`s become exact-shape `T.alloc_local`
arrays; supported register operations become `Tx.thread.*`, and transfers
become `Tx.copy`. `tcgen05.ld/st`, `ldmatrix/stmatrix`, and warp MMA are direct
`T.ptx.*` calls. A `Tcgen05Mma` or `Tcgen05Cp` gets statement-local, non-owning
TMEM buffer declarations and exactly one `Tx.gemm_async` or `Tx.copy_async`
call, respectively. `TmemTensor` itself stores only `start_col`.
