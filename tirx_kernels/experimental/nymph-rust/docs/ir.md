# Nymph Rust IR

The Rust IR is the pure foundation layer consumed by the interpreter and built
by the Python bridge. `src/ir/` does not depend on the interpreter. Interpreter
modules import from `ir`, not the other way around.

## Structure

- `dtype.rs` - simple enums: `MemorySpace`, `DType`, `Swizzle`,
  `TmemLayoutKind`, `MBarKind`, `FenceKind`, `FenceScope`, `VarBinding`,
  `ScalarDType`, and `ScalarOp`.
- `scalar.rs` - `Var`, `ScalarExpr`, `ScalarValue`, and `ScalarInitial`.
- `tensor.rs` - `Tensor`, `TensorSlice`, `Layout`, `SmemSwizzleLayout`, and
  `TmemLayout`.
- `mbar.rs` - `MBar` and `MBarRef`.
- `scheduler.rs` - tile scheduler nodes: `TaskSpace`, `Scheduler`, and
  `SchedulerPolicy`.
- `stmt.rs` - the `Stmt` enum, including control, metadata, register, TMA,
  TMEM, tcgen05, mbarrier, fence, sync, and cp.async statements.
- `kernel.rs` - `Kernel`, launch/cluster helpers, and source-body traversal.
- `thread_filter.rs` - static per-thread evaluation of `If` dispatch
  predicates into `Known(ThreadSet)` or `Unknown`.
- `validate.rs` - kernel validation for per-node checks and context-sensitive
  placement/coverage checks.

The Rust port uses `Arc<Tensor>` and `Arc<MBar>` handles. A tensor or mbar can be
shared throughout a kernel while still exposing its metadata during validation
and interpretation.

## Conventions

- Statements are variants of one `Stmt` enum. There is no intermediate
  op-family base class.
- Body-bearing variants implement `Stmt::child_bodies()`. Structural walks use
  this hook, so new control nodes must expose their nested bodies there.
- Tensor layouts are metadata for lowering and for modeled TMEM physical
  mapping. The value interpreter honors `TmemLayout`; SMEM swizzle metadata is
  recorded but not simulated.
- `Kernel::validate()` owns the validation pass. Rust cannot run Python-style
  dataclass `__post_init__` hooks, so construction is cheap and validation is an
  explicit kernel-level check.
- Placement rules that need enclosing context live in `validate.rs`: sync
  placement, loop/scalar variable definedness, and statement-specific
  thread-coverage requirements. Coverage rules are reachability statements
  over the enclosing `If` dispatch predicates: `static_thread_filter(cond,
  num_warps)` in `thread_filter.rs` evaluates a predicate at every
  `(warp, lane)` point of one CTA, and a rule fires only when the result is
  `Known` — `cta_sync`/`cluster_sync` must reach every CTA thread, `wg_sync`
  exactly one full warpgroup, `warp_sync` whole warps, `tmem_alloc`/
  `tmem_dealloc` exactly one full warp, and `sched_next` and `set_maxnreg`
  (which have no runtime backstop) require statically-resolvable branches
  confined to a single warp and whole warpgroup(s) respectively.

## Builder Bridge

The Python package under `python/nymph_rs/` is the user-facing builder surface
and constructs Rust IR objects through PyO3. Thread dispatch is `if_` over
scope-value predicates, with canonical sugar `if_warp(n)`, `if_warpgroup(n)`,
`if_lane(n)`, and `if_elected()` (lane 0 of every warp in context), plus
`set_maxnreg(n)` for the `SetMaxNReg` directive. The exported
`interpret(kernel, inputs)` binding validates and runs the Rust kernel, marshals
typed numpy inputs into `ValueArray1`, and returns GMEM outputs as numpy arrays.

## Adding A Statement

1. Add the variant and fields to `src/ir/stmt.rs`.
2. Add per-node and context validation in `src/ir/validate.rs`.
3. Add Python bridge construction in `src/py.rs` and the Python builder wrapper
   if the statement is user-facing.
4. Register an executor in the matching `src/interpreter/semantics/*.rs` module
   and append that module's `register` call to `semantics/mod.rs` if it is a new
   op family.
5. Add focused Rust integration coverage in `tests/interpreter_runner.rs`, plus
   Python binding coverage when the bridge surface changes.

## Change Policy

IR changes affect the builder bridge, interpreter, validation, and tests. Treat
them as user-visible API changes and keep them narrowly scoped.
