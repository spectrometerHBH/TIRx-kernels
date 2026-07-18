# Perf methodology: closing the nymph ↔ canon gap on B200

How the nymph GEMM kernels are measured and how a perf gap is diagnosed.
The ground-truth numbers live in `bench/RESULTS.md` — record every
measurement there (or in a commit message referenced from there), never only
in chat.

## 1. Measuring

```bash
CUDA_VISIBLE_DEVICES=<idle gpu> python bench/run_suite.py --rounds 10 --max-shape 16384
```

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
   - **BSSY/BSYNC pairs**: per-op guard `if` blocks reconverging; fixed by
     elect_sync guards + adjacent same-guard folding in codegen.
   - **UTMACMDFLUSH 4x**: epilogue commit_group not single-thread-guarded;
     fixed by scope-aware guards.
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

## 4. Rules of engagement

- One knob at a time, and record the A/B pair (config, ratio, rounds) in the
  kernel's GEMM_CONFIGS comment AND bench/RESULTS.md.
- A config change that wins one shape but regresses another goes into
  per-shape config, never global.
- The simulator/checker stay the correctness gate: any kernel change keeps
  `cargo test` + value-sim + protocol checker green before benching.
