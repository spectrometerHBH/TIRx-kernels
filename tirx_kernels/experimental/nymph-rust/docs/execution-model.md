# The nymph Execution Model

This is the language contract shared by the mental model, the value
simulator, and the protocol checker. It is one semantics with three
projections; nothing below is checker-internal detail.

## 1. Presentation layer

- **A warp is a sequential stream.** A kernel is a set of warps; each warp
  reads the whole program top to bottom. Control flow is warp-structured:
  branch and loop conditions are warp-uniform, an `If` re-joins at its end —
  there is no lane-level program counter.
- **An instruction is a warp-collective action with a lane mask.**
  `if lane >= 16: ...` means "this instruction's mask is lanes 16..31", not
  "half the warp went to another PC". A mask narrows the data plane, never
  the control plane.
- **The discipline, in one sentence: data leaving your hands — to another
  lane, another warp, another CTA, another proxy, an engine — must cross a
  nameable edge** (a convergence point, a synchronization primitive, a
  fence, a completion object). If you cannot name the edge, it is a bug.

The presentation layer lets a user simulate in "warp lockstep" terms, but
that fiction is *certified*, not axiomatic (see §3).

## 2. Visibility rules (semantic axioms — each weaker than or equal to hardware)

1. **Unmasked warp-collective effects** are visible to the whole warp after
   the instruction (the collective visibility of `.sync.aligned` ops is the
   hardware's own).
2. **A masked write is invisible to other lanes until a warp-level
   convergence point** (`warp_sync` or a warp-collective instruction).
   Volta (sm_70)+ Independent Thread Scheduling lets lane progress skew:
   after `if lane >= 16: st.shared` followed by `if lane == 0:
   mbarrier.arrive`, the arrive does **not** carry lanes 16..31's stores —
   without a convergence point, that is a race.
3. **Across warps / CTAs there is only explicit synchronization** (mbarrier,
   named barrier, cluster barrier, semaphore). Strength follows the `sem`
   qualifier: **release/acquire gives memory order, relaxed gives control
   order only** (usable by deadlock/wait witnesses, unusable by race
   checks). **A release publishes only the arrivers' own order** (projected
   by the arriving lane mask) — one lane's arrive does not vouch for its
   warp-mates' unconverged writes.
4. **Between proxies there is only the fence**: generic-proxy effects
   (ordinary ld/st) consumed by the async proxy (TMA / tensormap / tcgen05
   engines) must cross `fence.proxy.*` (`fence.mbarrier_init` likewise for
   mbarrier-object publication). Fence-synchronization is a per-THREAD
   relation: a fence releases the executing thread's own prior accesses.
5. **Async engines: issue ≠ landing; effects land only when a completion
   object is observed** (mbar expect-tx/complete-tx, commit_group /
   wait_group, `tcgen05.wait::ld/st`). Completion objects are themselves the
   synchronization primitives of rule 3, so asynchronous delivery reuses the
   synchronization vocabulary — there is no fourth mechanism. The dual rule
   (the async window): between issue and the observed drain, the source must
   not change.

## 3. The three-layer contract and the soundness direction

- **Mental model (presentation) = simulator (value projection) = checker
  (ordering projection) — one semantics.** The simulator evaluates one
  canonical schedule exactly (fast, deterministic; answers "are the values
  right"). The checker builds happens-before from the same engine's trace
  and answers the stronger question: *within the bounded configuration, do
  all legal interleavings agree on observable memory and outputs, and can
  every wait make progress?*
- **A passing check is a per-program lockstep license.** Hardware does not
  guarantee lockstep; lockstep thinking is safe because the checker proved
  the program's correctness depends on no implicit timing — every data
  handoff crosses an explicit edge. When the check fails, it points at the
  implicit timing you used. Lockstep is a controlled fiction of the
  presentation layer, never a promise of the model.
- **Direction promise: model guarantees ⊆ hardware guarantees** (only
  weaker is allowed; over-reporting is acceptable, under-reporting never).
  Any happens-before edge the model derives must hold on hardware; the set
  of races the model reports contains every race hardware could exhibit.
  The only way to break the direction is to invent an edge hardware does
  not give (treating a relaxed arrive as a release, sharing multicast tx
  counts on one mbar) — the per-op ledger (`docs/ir-ops.md`) and hardware
  fixtures guard that boundary.
- **Codegen discipline: faithful translation + fail closed.** IR that
  cannot be faithfully lowered is rejected in validate/codegen; silent
  degradation (dropping a field, changing a semantic) is this system's
  worst enemy.
- End-to-end proposition: `check passed + codegen faithful + model ⊆
  hardware ⇒ hardware output == simulator output`. Where this cannot yet be
  held, the gap is listed honestly in `LIMITATIONS.md`.

## 4. The checker's formalization

- **Baseline: per-lane program order.** Every lane has its own event slice
  (the instruction stream filtered by mask). The happens-before clock gives
  each `(stream, lane)` its own dimension; a masked write lives only in the
  writing lanes' dimensions until a convergence point folds the warp
  together.
- **Join points (the complete vocabulary producing happens-before)**:
  warp-collective instructions (convergence), `warp_sync`, mbarrier (by
  `sem`), cooperative barriers (cta / warpgroup / named / cluster
  rendezvous), semaphore (value-keyed release/acquire), `fence.proxy.*`.
  Section 5 is the per-op ledger. A new op must pass the exhaustiveness
  gate (every `Stmt` variant handled explicitly in validate / interpreter /
  checker / codegen); the vocabulary never drifts silently.
- The checker judges **every conflicting access pair** by happens-before
  and every wait by a witness — it checks no "scenario rules", so new
  patterns need no new patches; "a masked write is invisible" is a
  **theorem** of the clock algebra, not an axiom.
- **Complexity**: linear scan (frontier + vector clocks, standard race
  detection). A stream's 32 lanes share one number line — its event
  ordinal — so the clock carries one all-lane PREFIX entry per stream plus
  lane entries only where a masked release actually published one lane's
  progress. A full-warp publication is a single prefix entry; per-lane
  precision costs only where divergent publication happens, so this is not
  a ×32 blowup.

## 5. Join-point ledger (per-op strength)

The only constructs that produce cross-lane or cross-warp happens-before,
each with its modeled strength and PTX basis. Anything not listed orders
nothing beyond per-lane program order. A new op adds its entry when it
joins.

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

**Releases project their ARRIVING lanes only** (visibility rule 3):

- `mbarrier.arrive` / complete-tx — per-thread (§9.7.14.16, no `.aligned`):
  a full-warp arrive publishes all 32 lanes (each lane arrived itself); an
  elected arrive publishes one lane's order. Phase completion freezes the
  accumulated join for the waiters.
- semaphore release (`gmem_atomic_add` order=release) — value-keyed;
  relaxed publishes nothing (control order only).

Acquires join into the ACQUIRING lanes only: a masked wait delivers the
release to its lanes alone until a convergence point spreads it.

**Modeled WEAK** (over-report direction, by the ledger discipline):

- `elect` — no IR op; `if_elected` lowers to a plain `If`. Hardware
  `elect.sync` synchronizes its membermask, but no convergence is credited
  (a kernel needing the ordering writes `warp_sync`).
- `tcgen05.wait::ld/st` — drains the EXECUTING thread's own loads/stores
  (§9.7.17.8.5, per-thread); orders nothing across lanes.
- `tcgen05.commit` / `tcgen05.mma` — single-thread issues; no convergence.

**Seams to keep in view**: `WarpSync` is checker/simulator vocabulary that
codegen does not lower to `bar.warp.sync`, so a proof leaning on it compiles
to code that relies on the warp launching converged; and the value simulator
lands async-engine effects at issue, which is why the engine's real timing
envelope is owned entirely by the checker's async-window passes
(`async_group_lifetime`, `tcgen05_async_hazard`) rather than by the values.
