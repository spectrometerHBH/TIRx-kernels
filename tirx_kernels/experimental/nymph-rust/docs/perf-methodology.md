# Perf methodology: closing the nymph ↔ canon gap on B200

How the nymph GEMM kernels are measured and how a perf gap is diagnosed.
The ground-truth numbers live in `bench/RESULTS.md` — record every
measurement there (or in a commit message referenced from there), never only
in chat.

Current structural rule: codegen is a literal IR emitter. It preserves every
`If` condition, sibling/nested relationship, and statement order; it never
substitutes `elect_sync()`/`thread_rank()` for `lane_id == 0`, folds adjacent
guards, or builds an if/else dispatch chain. Historical measurements below
that mention those transforms remain evidence about older code, not a tuning
technique available in the current codegen. A desired control-flow form must
be authored explicitly in the kernel IR and reviewed as an algorithm change.

## 1. Measuring

```bash
python bench/run_suite.py --rounds 10 --max-shape 16384
```

(run_suite.py is the thin wrapper over the bench-suite orchestrator — automatic
GPU selection/requeue; no CUDA_VISIBLE_DEVICES needed.)

- bench-suite standard path (`load_kernel` + `run_kernel_bench`), proton
  timer, cold cache; both impls (tir=canon, tirx=nymph) share the same data
  and the same `bench()` call, and each impl is correctness-gated (cosine)
  before timing — a ratio of two kernels computing different results fails
  loudly.
- Ratio convention: `tir/tirx` > 1 means nymph is faster. Target: ≥ 0.99 on
  every bench shape.
- Watch the per-round spread: a shape with spread > ±0.03 (e.g. fp16 16384
  0.928-1.103) is noise-dominated; do not tune against single runs.

## 2. Diagnosing: the ncu opcode diff

When a shape is below target, the first tool is a SASS opcode-execution-count
diff against canon (method established in commit 9ef5a61a / 034ae826):

1. Profile both kernels with ncu's InstructionStats:
   `ncu --set full --section InstructionStats --metrics sass__inst_executed_per_opcode`
   on a single-cluster launch of each, and export.
2. Diff `sass__inst_executed_per_opcode` between nymph and canon, largest
   absolute gaps first. Historical findings and what they meant:
   - **R2UR excess** (nymph 13.5K vs canon 1.8K on fp16 1024): scalar values
     living in vector registers feeding uniform-pipe instructions (UTMALDG
     TMA coords). Fix class: uniform-datapath placement (SSA `T.let` form,
     `shuffle_sync` outside elected regions) — see the producer comment in
     fp16_bf16_gemm.py. NOTE (2026-07): the `T.let` form landed and did NOT
     move the count — the placement failure is whole-function; see §3.
   - **BSSY/BSYNC pairs**: per-op guard `if` blocks reconverging. Historical
     experiments used inferred election guards and folding; those transforms
     are deleted. Current diagnosis must trace the explicit IR branches.
   - **UTMACMDFLUSH 4x**: historically attributed to epilogue
     `commit_group` guard placement. Current guard placement must be changed
     in the kernel IR, never inferred from codegen scope.
   - **IMAD/ISETP excess**: index math in vector registers; same fix class
     as R2UR.
   - **LDL/STL**: register spills — shrink live fragments (epilogue tile
     width, `wb_pipe_depth`, `maxnreg` rebalance).
3. Attribute with lineinfo: `nvdisasm -gi` on the cubin to map hot
   instruction ranges back to source lines (call-site attribution vs canon).
4. SM-side cross-check when instruction counts look fine but time is off:
   compare SM busy %, regs/thread, and launch-to-launch state (a gap that
   only appears in back-to-back launches is state/thermal, not the kernel).

## 3. What does NOT work (measured, do not retry blindly)

- **Line-level TIRx equivalence edits**: a full audit (workspace
  audit_4096_nymph_vs_canon.md) made the emitted TIRx line-equivalent and
  was bench-neutral — the residual gap on those shapes sits below emitted
  TIRx (ptxas register allocation/scheduling). Don't re-litigate line-level
  diffs; go straight to opcode/SASS level.
- **stmatrix epilogue** for the nvfp4 shapes: verified working via ncu but
  regresses on 1024/2048/4096 (0.968/0.980/0.940). Capability exists, off.
- **R2UR via the scalar-binding emission form** (fp16 1024, 2026-07): the
  per-task decode chain was converted from mutable `T.int32` local cells to
  single-assignment `T.let` SSA bindings (IR `ScalarLet`, canon's exact
  source form — verified in the lowered CUDA as `int s10 = (s8_ptr[0] & 3);`)
  and the three nvfp4/fp16 `shuffle_sync` spots were replaced by lets.
  Measured A/B (ncu InstructionStats, single launch): fp16 1024 R2UR
  13504 -> 13504 (SHFL 2560 -> 1024, total inst 322280 -> 322135);
  nvfp4 1024 R2UR 2016 -> 2016 (canon fp16: 1792, canon nvfp4: 4544 —
  nvfp4-nymph was already better than canon). So the decode-chain binding
  form is NOT what keeps the chain off the uniform datapath. The R2UR is a
  WHOLE-FUNCTION ptxas uniform-placement decision, evidenced by
  nvrtc+nvdisasm bisection on the dumped device source (R2UR SASS lines):
  strip the CLC scheduler region -> 3 (from 90), strip MMA -> 12, strip
  epilogue -> 74, producer-only -> 2; splicing CANON's scheduler verbatim
  into the nymph kernel stays 90 (content-irrelevant); injecting nymph's
  control-flow shapes into canon stays 4 (shape-irrelevant). Morphs that did
  NOT flip it: vacuous-break removal, canon ring-advance form, expect_tx
  order, init-elect batching (90->85), MMA peel+roll -> single loop w/ accum
  cell (112, worse), meta_var-equivalent inlining of the let exprs at use
  sites (90), scheduler done-flag form, ptxas --register-usage-level 6/8/10,
  --allow-expensive-optimizations. `-O1` drops it (90->62) but is not
  viable. The actionable direction (not yet done): shrink the whole-kernel
  static complexity until ptxas's placement flips — canon is ~29% smaller
  in SASS than nymph on this shape.
- **Full executed-opcode diff** (fp16 1024, single launch each, ncu
  `sass__inst_executed_per_opcode`, 2026-07-18; canon `_kernel_kernel` vs
  nymph `main_kernel`). Total 267,182 vs 310,063 = **+42,881 (+16.0%)**,
  in three clusters:
  - *uniform-placement* (the ptxas whole-function issue above): R2UR +11,712
    (1,792 -> 13,504); IMAD +9,414; ISETP +8,359; VIADD +5,952; MOV +4,992;
    LEA +3,584; ULEA +3,328 — scalar index/address math executed on the
    vector pipe that canon keeps on the uniform pipe.
  - *synchronization structure*: SYNCS +10,228 (30,013 -> 40,241, mbarrier
    family); NANOSLEEP +5,497 (try_wait backoff); ELECT +2,496; BSSY/BSYNC
    +1,280; YIELD +1,088; WARPSYNC +512 — nymph's guard/wait structure
    executes more sync work per task.
  - *control flow*: BRA +8,730; SEL +2,752 — more branches around guards
    and loop forms.
  Partial offsets (nymph executes FEWER): MEMBAR -1,024, UTMACMDFLUSH -768,
  S2R -704, LDCU -640.

## 5. fp16 1024: CUDA-level convergence and the R2UR flip (2026-07-19)

This section records a historical convergence experiment. In particular, the
codegen-created dispatch chain and election spelling used at that point are
not present in the current implementation. The numerical/SASS measurements
remain useful evidence; reproducing the structural variants now requires an
explicit IR kernel change and review.

The R2UR placement failure is a **whole-function static-complexity threshold**,
now proven directly (§3 had it as a hypothesis):

- *Padding test*: injecting a never-taken (`blockIdx.x >= 1000`, so ptxas cannot
  fold it) dead block of TMA-shaped code into **canon's** CUDA flips canon's own
  placement: 1155 SASS/4 R2UR -> 1398/121. Even +135 lines (1290) flips it to
  106. The same pad on the converged nymph kernel (1104) flips it back to 74.
  So there is no "bad construct" to fix line-by-line — ptxas's uniform-datapath
  selection degrades globally once the function crosses a size/complexity
  budget (~1100-1200 SASS lines for this kernel shape). All the §3 morphs that
  "did not flip it" were single-region edits that could never move the total.
- *lineinfo attribution* of the 90 R2UR lines (nvrtc `-lineinfo` +
  `nvdisasm -gi`): 56 in the producer k-tile loop (TMA coords + ring advance
  feeding UTMALDG), 12 in the MMA region, ~13 epilogue (commit_group / cvt /
  tail), ~9 prologue (mbarrier-init addresses). These are the places a
  vector-computed scalar feeds an instruction that needs a uniform operand —
  the whole class disappears once under budget.

### 5.1 Structural diff inventory (canon `_kernel_kernel` vs nymph `main_kernel`, fp16 1024)

Region-by-region classification of the pre-convergence dumps
(/tmp/canon_1024.cu 945 lines vs /tmp/nymph_1024.cu 875 lines):

- **(a) naming / dead code**: 8 dead `(((int)threadIdx.x) & 127);` exprs + dead
  `v_1..v_7` vars; duplicate `descA_1`/`descB_1` descriptor encodes (used only
  by the rolled MMA loop); dead `cse_vN` temporaries both sides carry. Zero
  SASS effect (ptxas DCEs); the descriptor dedup is real and came free with the
  MMA merge.
- **(b) equivalent logic, different form**: scheduler loop counter+`break_if`
  vs canon done-flag + phase cells; ring advance `nxt>=5` vs `stage==5`;
  phase `(s7+1)&1` vs cell `phase^0`; epilogue `if (1 <= s25)` warmup guards
  vs canon's unconditional wait_group+wg_sync; `commit_group` inside the
  single-thread guard vs canon's all-threads commit; mbarrier-init under
  per-barrier `elect.sync` vs canon's grouped `thread_rank()==0`; missing
  prologue `fence.proxy_async` + `warp_sync` after tcgen05.alloc; loop-carried
  cell coords vs `T.let` SSA (measured neutral in §3). Individually all
  flip-neutral — they matter only through total size.
- **(c) really different logic**: **MMA peel+roll** (16 static MMA call sites
  vs canon's 8 in one rolled loop with a runtime accum cell); **flat role
  dispatch** (5 top-level guards) vs canon's nested if/else decision tree;
  epilogue CLC query at loop bottom vs canon query-at-top with coord cells;
  scheduler `expect_tx` before `clc_try_cancel` vs canon's reverse; MMA's
  `cbx==0` guard inside the elect vs canon's at role level; teardown handshake
  placement. Of these, the flip needed only: the MMA merge (pinned rolled),
  the dispatch chain, and the size from (b) — the query positions, scheduler
  order, and elect nesting were each measured flip-NEUTRAL at the threshold
  (A/B on the converged kernel).

### 5.2 The convergence set and the flip point

Applied in order (R2UR lines / total SASS lines, nvrtc+nvdisasm A/B):
1. desc dedup + sched order + warmup-guard removal + commit unguard: 90/1482
   -> 90/1472 (no flip, as expected under threshold).
2. MMA peel+roll -> ONE rolled 8-tile loop + runtime accum cell: 112/1447
   (worse R2UR — ptxas re-unrolled the merged loop); + `#pragma unroll 1`:
   **77/1349** (the pin is what banks the merge's ~130 lines).
3. Prologue batching (grouped `thread_rank()==0` inits + canon order + the
   proxy fence): **74/1225**.
4. Scheduler canon form + MMA canon form + dispatch if/else chain:
   **4/1104 — FLIP**. (Epilogue/producer canon-form rewrites were tried too:
   +25/+16 lines, flip-neutral, reverted.) Flat dispatch instead of the chain:
   74/1214 (flips back); nested-without-else: 65/1142 (flips back) — the
   else-chain shape itself is required, not just guard sharing.
5. MMA query-at-top vs bottom: neutral (1088/4). Scheduler expect_tx/try_cancel
   order: neutral (1079/4).

Final real-pipeline state: **1100 SASS / 3 R2UR** vs canon 1155/4. Threshold
for this content measured by padding: flips back between 1104 (4) and 1240 (74).
Historical runtime confirmation (ncu, single launch each): executed R2UR 13,504 -> 768
(canon 1,792), total instructions 310,063 -> 279,092 (canon 267,352, +16.0% ->
+4.4%). Bench (rounds=10, same-day baseline comparison): **fp16 1024 0.981 ->
1.019, bf16 1024 0.981 -> 1.018** (target ≥0.99), fp16/bf16 2048 ~1.01,
4096 ~1.00, 8192 ~0.99-1.00, nvfp4 unchanged. These are retained as
historical wave data; the current explicit-physical-IR baseline supersedes
them and is recorded in `bench/RESULTS.md`.

### 5.3 What changed in the tree (and what it cost)

- IR: `ForLoop(unroll=False)` (->
  `T.serial(N, unroll=False)` -> `#pragma unroll 1`), `Tcgen05Mma.accum:
  ScalarValue` (runtime accum predicate, canon's accum cell), `ScalarLet`
  (the `T.let` SSA binding), CLC try/query-cancel, split cluster barrier,
  explicit dynamic-SMEM offsets, `cache_hint`/`prefetch_tensormap`. The old
  `reg_frag`/warpgroup-tile representation has since been removed: REG tensors
  now preserve their IR shape through `T.alloc_local`.
- codegen: control-flow structure is emitted 1:1 from IR; codegen does not
  synthesize warpgroup parents, merge equal predicates, or re-nest sibling
  role branches, and it prints each predicate literally. `commit_group` emits
  unguarded (canon's actual
  shape — the 2026-05 guarded form was a deliberate UTMACMDFLUSH trade, now
  reverted for size). Mbarrier init is legal only beneath an explicit
  single-lane IR branch; codegen neither inserts nor coalesces that branch.
  `if_elected` is builder sugar for `lane_id == 0`, which remains
  `lane_id == 0` in the emitted source.
- builder (fp16_bf16_gemm): scheduler done-flag + phase cells; MMA single
  merged k-loop + per-task accum cell + `unroll=False`; prologue inits grouped
  before the alloc + proxy fence on both paths; `_init_stages` flat (so the
  inits coalesce); loader section now precedes the scheduler (canon's order).
- **Residual divergences (deliberate)**: the epilogue keeps the
  `store_iter >= num_d_tiles` guard on the pacing wait_group — the protocol
  checker's `deadlock_freedom_missing_release_witness` rule requires a
  committed group before any wait_group on a stream, so canon's unconditional
  form is rejected (~8 SASS lines, under threshold regardless); the wg_sync
  that canon nests under the same runtime guard is hoisted OUT (the
  `unknown_filter_sync` lint — a sync whose scope depends on a runtime value
  warns); the tmem_empty arrive rides AFTER a warpgroup_sync (the cross-lane
  publication rule — canon's all-thread arrive is the alternative); nymph keeps
  `expect_tx` before `try_cancel` in the scheduler (flip-neutral, and the
  checker's ordering assumptions were validated on it); epilogue/producer keep
  the query-at-bottom loop rotation (flip-neutral, less churn than canon's
  query-at-top rotation); no `warp_sync` after the tcgen05.alloc (SMEM
  visibility is already forced by the later cluster barriers). The historical
  codegen-created dispatch chain described in §5.2 has been removed; current
  codegen cannot create an `else` edge absent from the IR.

## 4. Rules of engagement

- One knob at a time, and record the A/B pair (config, ratio, rounds) in the
  kernel's GEMM_CONFIGS comment AND bench/RESULTS.md.
- A config change that wins one shape but regresses another goes into
  per-shape config, never global.
- The simulator/checker stay the correctness gate: any kernel change keeps
  `cargo test` + value-sim + protocol checker green before benching.
