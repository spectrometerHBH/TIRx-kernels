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
   engines) must cross `fence.proxy.*`, and mbarrier-object publication asks
   the same of `fence.mbarrier_init` (modeled as a release-side fence; the
   obligation itself is unchecked — see section 5's seams).
   Fence-synchronization is a per-THREAD
   relation: a fence releases the executing thread's own view — its own
   prior accesses, plus whatever a convergence point or barrier had already
   carried into it. This is a SECOND obligation on top of rule 3, not a
   substitute: an mbarrier may order a store before an engine reads it and
   the bytes still be unpublished, and a fence may publish bytes that
   nothing orders. Both are required, and each is checked against the same
   clocks.
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
  counts on one mbar) — the per-op ledger (`docs/ir-ops.md`, summarized in
  section 5) and the hardware fixtures guard that boundary.
- **Codegen discipline: faithful translation + fail closed.** IR that
  cannot be faithfully lowered is rejected in validate/codegen; silent
  degradation (dropping a field, changing a semantic) is this system's
  worst enemy.
- End-to-end proposition: `check passed + codegen faithful + model ⊆
  hardware ⇒ hardware output == simulator output`. Where this cannot yet be
  held, the gap is listed in `LIMITATIONS.md` and in section 5's seams.

## 4. The checker's formalization

- **Baseline: per-lane program order.** Every lane has its own event slice
  (the instruction stream filtered by mask). The happens-before clock gives
  each `(stream, lane)` its own dimension; a masked write lives only in the
  writing lanes' dimensions until a convergence point folds the warp
  together.
- **Join points (the complete vocabulary producing happens-before)**:
  warp-collective instructions (convergence), `warp_sync`, mbarrier (by
  `sem`), cooperative barriers (cta / warpgroup / named / cluster
  rendezvous), semaphore (value-keyed release/acquire), and — for the
  cross-proxy obligation of rule 4 — `fence.proxy.*`, which releases the
  fencing thread's view into the async-proxy engines so that an engine
  access acquires it like any other release. Section 5 is the per-op ledger,
  and every op that joins belongs in it — a discipline, not something the
  code checks (see the seams).
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

## 5. Join-point ledger and known seams

The per-op record — which construct produces which happens-before edge, at
what strength, on which PTX basis — lives in `docs/ir-ops.md` under
"Happens-before join points", so that the op ledger stays in one file. Read
it as the normative list: anything not in it orders nothing beyond per-lane
program order, and a new op that joins belongs there.

The seams that keep the end-to-end proposition from holding today are listed
there too, alongside the ops they belong to. The two that bite hardest:
`fence.mbarrier_init` seals lanes but nothing requires a kernel to publish
its barrier objects with it, and a trace event kind the clock scan does not
name falls into a catch-all and orders nothing — the closed vocabulary of
section 4 is a discipline, not something the code enforces.
