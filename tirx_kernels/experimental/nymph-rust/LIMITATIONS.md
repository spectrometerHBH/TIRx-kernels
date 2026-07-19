# LIMITATIONS — known sim ↔ silicon divergences

Read this BEFORE trusting a green value-sim / protocol-check run. Each entry is
a place where "simulator says OK" does not imply "GPU behaves the same".
Per-op detail lives in `docs/ir-ops.md`; this file is the short trust boundary.

## Timing / asynchrony

- **err 719 launch fault under full-async OVERLAP** (nvfp4_gemm.py:367-371):
  with ~10+ tasks/cluster a timing-sensitive `cudaErrorLaunchFailure` appears;
  hidden by any serialization. The 8192 no-overlap config is believed to
  sidestep it (40+ stress reps clean) — believed, not proven.
- **expect-tx / complete-tx race**: sim requires expect-tx before complete-tx;
  HW tolerates the opposite order and the CLC multicast tx can outrun the
  expect_tx on GPU (codegen.rs comment). Sim cannot reproduce this race.
- **Value-at-issue**: TMA/CLC/tcgen05 writes land at issue in sim. A read
  between issue and wait sees NEW data in sim but OLD data on HW. Only the
  trace-mode checker catches that bug class — always run the protocol checker,
  not just the value sim.

## Modeled-differently-than-hardware

- **cg2 multicast into a SHARED (leader) mbar**: sim dedups tx-count per unique
  cell; HW signals each destination (2x bytes). Validate now rejects this
  combination — do not re-enable without fixing the sim model.
- **SF (scale factor) TMEM footprint**: sim stores 1 byte/cell flat over 128
  lanes; HW packs 4 e4m3 per u32 cell with 4x subpartition duplication. Values
  equal; TMEM column budget ~4x off. Column-planning math must use sim units.
- **tcgen05.cp** is a value-level abstraction of canon's SF copy_async path,
  not a raw PTX tcgen05.cp — no HW cp shape/warpx4/decompression is modeled.
- **MMA accumulation order**: OpenBLAS sgemm blocking order, not the tensor
  core's fixed order. Bit-exact only for exact arithmetic (the nvfp4 value
  tests choose exact inputs on purpose).
- **f16/bf16 tcgen05.ld/st packed path**: no B200 fixture (b32 shapes only);
  marked suspect in ir-ops.md.
- **CLC round-robin oracle**: query_cancel returns canonical round-robin
  tasks. Only "each task exactly once + termination" holds for arbitrary
  orders; value regressions are only checked under round-robin.

## HW-illegal but sim-accepts (do not use; rejection pending kernel rework)

- **peer (remote_coord) mbarrier wait** — used by fp16_bf16_gemm.py:583,
  silently dropped by codegen; correctness rests on a comment-level
  leader-routing argument.
- **odd-CTA tcgen05.mma issue** — silently dropped in sim; HW would execute.
  Always issue MMA from the cluster leader (cta_rank==0).

## sim-only ops (no codegen lowering — sim-green means nothing for silicon)

WarpMma, RegUnary, GmemAtomicAdd, GmemWaitEq, CpAsyncBulkS2Cluster,
TmaStore.reduce_add, Tcgen05St / Tcgen05WaitSt, Fence Memory/View, and any
Tcgen05Ld outside (shape=32x32b, row=0, f32). Compile gates reject them; do
not treat their sim validation as GPU evidence.

## IR restrictions that keep validate and codegen on one semantics

- **TMEM**: one base-0 column band per kernel, live at most once; every
  alloc/dealloc/relinquish carries the kernel cta_group; no alloc after a
  `relinquish_alloc_permit` (also enforced per CTA at runtime, PTX
  §9.7.17.7.1); lifecycle ops are top-level only (never inside a
  loop/conditional body — validate rejects them there). Multi-band/base-offset
  TMEM plans are IR-illegal on purpose (the generated code is a single base-0
  view). The checker's lifecycle pass requires happens-before edges
  alloc→access→dealloc (real sync, not the sampled interleaving).
- **codegen destructures no Stmt field silently**: `, .. }` is banned in
  codegen.rs (compile-gated text check); an IR field the emission cannot
  honor is an `Err`, never a dropped default.

## Environment / config hardcodes

- `SM_NUMBER=148` hardcoded in both GEMM kernels (B200 SM count; wrong on
  other parts).
- alpha is a build-time immediate in the nvfp4 nymph kernel (canon loads it
  at runtime) — same math, different capability.
- build.rs defaults BLAS to a machine-specific conda path; override with
  `BLAS_LIB_DIR`.
