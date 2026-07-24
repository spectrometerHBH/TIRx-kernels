# nymph-rust — interpreter port status

A Rust port of the nymph GPU-kernel **value simulator**. The Python IRBuilder
remains the user-facing construction surface and builds a Rust IR over PyO3; the
interpreter is fully reimplemented in Rust.

For the migrated architecture and semantics docs, start with `README.md`, then
see `docs/ir.md`, `docs/interpreter.md`, `docs/interpreter-semantics.md`, and
`docs/hardware-verification.md`.

## What's done

### IR + builder bridge
- `src/ir/` — the full IR (dtype, scalar, tensor, mbar, scheduler, stmt, kernel,
  thread_filter, validate).
- `src/py.rs` — PyO3 bindings: the Python builder constructs Rust IR; `Kernel`
  auto-validates on construction. Plus the `interpret(kernel, inputs)` entry point,
  which marshals typed numpy inputs in and kernel-produced GMEM tensors back out.

### Interpreter
File layout mirrors the Python package; the **modularity contract is preserved**:

- `interpreter/values/` — the simulated memory/value layer: `arrays` (the typed
  `ValueArray1/2` storage), `dtypes` (f16/bf16/f32 rounding incl. the bf16 bit-formula),
  `indexing`, `reg_numerics`, `tensors` (GMEM/SMEM + valid mask), `smem` (byte-pool
  SMEM scratchpad), `registers`
  (per-thread reg file), `tmem` (128×512 cell grid + addressing), `mbars`, `scalars`,
  `cooperative`, `runtime`, `tcgen05_datapath`.
- engine: `ids`, `scalar_eval` (floor-div/mod, scope values, uniform fast path),
  `outcomes` (the `StepStatus` + `WakeCondition` protocol), `state`, `slice_indexing`,
  `warp_context` (the executor context + lane-vectorized register/shared read-write), `mbar_ops`
  (pure phase-cell algebra + target resolution), `elementwise` (reg ALU), `transfer`,
  `tmem` (alloc lifecycle), `scheduler` (per-warp streams, frame stack, grid
  expansion, CTA activity), `runner` (the main loop, dispatch, direct mutation,
  precise wake).
- `interpreter/registry.rs` + `interpreter/semantics/` — per-op executors. A semantics
  module registers its executors; `default_executor_registry` iterates the registrars.
  **Adding an op = new module + one `register` line; the runner is never edited.**
  Modules: control, scalar, metadata, fence, cp_async, mbarrier, reg, tma, tcgen05,
  tmem, sync, leaf.

Design points:
- **Per-warp streams** — one execution stream per `(cta, warp)`, all materialized
  eagerly; every stream runs the whole kernel body top to bottom. A warp's lanes
  share one instruction stream under masked divergence and advance through it
  independently (sm_70+); above the warp everything is concurrency. All ordering
  is explicit and checked: cross-warp/cross-CTA ordering comes from sync ops, and
  cross-LANE ordering inside a warp comes from a warp-level sync — `warp_sync`, a
  warp-collective instruction (ldmatrix/stmatrix/tcgen05_ld/tcgen05_st/warp MMA),
  or a cooperative barrier the warp passes. Thread dispatch is `If` over
  per-thread predicates (`if_warp` / `if_warpgroup` / `if_lane` / `if_elected`
  sugar). Cross-warp access pairs inside one CTA are ordinary concurrency and get
  race-checked like any other pair (pinned by
  `tests/interpreter/test_warp_model.py`); cross-lane pairs inside one warp are
  verified by `memory_race_check` and reported as `intra_warp_cross_lane_race`
  (pinned by `tests/interpreter/test_cross_lane.py`).
- **Lane vectorization** — handlers operate on whole `ThreadMask`s, never loop
  threads in op logic.
- **Direct mutation** — every executor takes `&mut WarpContext` (holding `&mut state`),
  mutates state in place, and returns a light `StepStatus` (`Advance{wakes}` /
  `AdvanceContinue` / `Block(WakeCondition)` / `Fail`). There is no staged commit.
- **Precise wake** — a stream that blocks on `mbarrier_wait` parks on a
  `WakeCondition::Mbar{key,phase}`; when a later mutating step touches that cell, the
  runner re-checks the parked waiters and **advances their frame directly** — the wait
  never re-runs, so there is no phase latch and no idempotency discipline. The
  cooperative-sync rendezvous / tmem-collective / tcgen05 peer-active blocks use
  `WakeCondition::Polled` (re-run each round; their re-runs are naturally idempotent,
  and a cooperative re-poll is an O(1) count lookup).
- **Typed values** — `ValueArray1/2` stores each tensor/register in a container chosen
  for value-losslessness: f16/bf16/f32 are **f32-backed** (f16/bf16 rounded on write,
  then held exactly), integers/bool use their **native fixed-width** type. The MMA reads
  its SMEM operands as borrowed `&[f32]` (zero copy) and `sgemm`s straight into the
  column-major f32 TMEM grid; the integer-native storage keeps i64/u64 exact.
- **Fatal failure / protocol report** — failed value runs report diagnostics/frontier
  metadata and no payload, because direct mutation may have left state partial. Trace
  runs return a `RunPayload::Trace` report for Passed, Failed, and Inconclusive
  protocol outcomes.

## Correctness

The fp16/bf16 GEMM (m=512, n=256, k=64, cta_group=2) runs end-to-end and is **cell-exact**:
- vs a numpy reference `round(A @ Bᵀ, dtype)` — 0 mismatches, **fp16 and bf16**.
- vs the **original Python interpreter** — 0 mismatches, cell for cell.

This kernel exercises TMA load/store, the cta_group=2 MMA, the TMEM collective +
scratchpad, mbarrier handshakes (incl. the precise wake), cooperative sync, the
register ALU/cvt, and the direct-mutation runner.

Test coverage: `cargo test` (lib unit tests — IR, the typed value layer + dtype
coercion, engine loop — plus 11 Rust-internal integration tests in
`interpreter_runner.rs` for state the Python API can't observe: mbar-cell parity,
no-partial-values-on-failure, internal commit cells) + `tests/` (Python: `ir/`
binding/validation/structure, `interpreter/` per-op value behavior — incl.
`test_warp_model.py`, which pins the per-warp concurrency semantics: cross-warp
race reporting, warp-specialized intra-iteration handshakes — `kernels/`
e2e/parity/determinism).

## Performance

The standing benchmark fixes one GEMM task tile (m=512, n=256, **k=16384**, cta_group=2,
launch=(2,) — one persistent cluster) and scales the task count via
`ForEachTask(grid_stride)`.
Profile small task counts (1/2/4/8/16), linear-fit `total = a + b·tasks`, and use the
per-task slope `b`; setup (input marshalling) is the intercept `a`, kept separate from
per-task simulation. Measure clean (no `NYMPH_STATS`; the profiler adds ~3 ms/task).

Measured on the per-warp execution model (one stream per `(cta, warp)`):

| mode | per-task (k=16384) |
|---|---|
| value (`interpret`) | ~97 ms |
| raw trace (`trace`) | ~22 ms |
| full `check_protocol` | ~70 ms |

Raw trace stays well below value mode — it skips payload byte movement and BLAS — and
full protocol checking adds the offline passes on top of trace execution. The Python
interpreter this port replaced ran the same tile at ~850 ms/task.

In value mode the OpenBLAS `sgemm` for the MMA dominates the per-task time (the
irreducible compute), followed by the MMA operand read and the TMA gmem↔smem tile
copies; the rest is interpreter overhead (dispatch, scalar resolve, scheduler, mbarrier
handshakes). The hot path is copy-free: contiguous-layout MMA borrows the SMEM f32
operands and accumulates in place into the TMEM grid.

`NYMPH_STATS=1` emits a per-phase / per-executor profile to stderr (gated, ~zero cost
when off). `NYMPH_BLAS_THREADS=N` pins the OpenBLAS thread count (default 1).

### Remaining levers (not yet done)
- **Vectorize `scalar_eval::eval_scalar_vec`** — it still loops the lanes for non-uniform
  (lane-dependent) offsets where Python uses numpy; array-based eval would cut `eval_slice`
  and the per-stmt cost across the board.
- **Avoid the per-statement lane-mask clone** in `current_stmt` (return an index/slice).
- **Multithreading** — the Rust interpreter has no GIL; independent CTAs/tasks could run
  on a thread pool. The Python interpreter cannot.

## Notes
- Non-contiguous MMA layouts (Layout F: cta_group=1 m=64; Layout B: cta_group=2 m=128)
  use the general `matmul_f32` + per-cell scatter path; contiguous D/A layouts use
  the zero-copy in-place sgemm when operands are non-transposed rank-2 SMEM slices.
  Both are cell-exact; the scatter path is the universal fallback and is not on the
  measured GEMM's hot path.
- `[profile.release] debug = 1` is kept for line-table profiling (negligible runtime cost).
- Build/run: `cargo build --release --features python`, copy the `.so` into `_pybuild/`,
  then `PYTHONPATH=_pybuild`. OpenBLAS is linked via `build.rs` (override its directory
  with `BLAS_LIB_DIR`).
