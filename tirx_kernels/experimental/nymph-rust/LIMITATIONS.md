# LIMITATIONS — known sim ↔ silicon divergences

Read this BEFORE trusting a green value-sim / protocol-check run. Each entry is
a place where "simulator says OK" does not imply "GPU behaves the same".
Per-op detail lives in `docs/ir-ops.md`; this file is the short trust boundary.

## Timing / asynchrony

- **err 719 launch fault under full-async OVERLAP** (nvfp4_gemm.py, the
  default-overlap epilogue path):
  with ~10+ tasks/cluster a timing-sensitive `cudaErrorLaunchFailure` appears;
  hidden by any serialization. The 8192 no-overlap config is believed to
  sidestep it (40+ stress reps clean) — believed, not proven.
- **expect-tx / complete-tx race**: sim requires expect-tx before complete-tx;
  HW tolerates the opposite order and the CLC multicast tx can outrun the
  expect_tx on GPU (codegen.rs comment). Sim cannot reproduce this race.
- **Value-at-issue — and the division of labor it forces**: TMA/CLC/tcgen05
  writes land at issue in the value sim, so a read between issue and wait sees
  NEW data in sim but OLD data on HW. The engine's real timing envelope is
  therefore owned ENTIRELY by the checker's async-window passes
  (`async_group_lifetime`, `tcgen05_async_hazard`: source immutable and
  destination unread from issue to the observed drain), never by the values.
  Always run the protocol checker, not just the value sim — sim values are
  trustworthy only where those windows verify.

## Modeled-differently-than-hardware

- **TMEM lifetime proof is uniform — drain + ordering, no teardown
  exception**: for EVERY deallocated generation the lifecycle checker
  requires each access to be retired by the hardware completion mechanism
  (`tcgen05.wait::ld/st` for a load/store, the wait on the mbar a
  `tcgen05.commit` handed the work to for an mma/cp) AND the drain to be
  happens-before the free. There is no kernel-teardown relaxation: a band
  freed at kernel end needs the same edges as a mid-kernel one, so the nymph
  kernels emit a cluster barrier before the teardown dealloc. This is
  STRICTER than silicon: canon's teardown shape frees the band with only a
  drain on the consumer stream and no closing rendezvous — flash_attention4
  deletes the tail barrier outright
  (tirx_kernels/attention/flash_attention4.py:1015-1022, 50x reused-module
  GPU verify PASS) and fp8_blockwise_gemm deallocs BEFORE its closing
  cluster_sync (tirx_kernels/gemm/fp8_blockwise_gemm.py:505-506) — both
  GPU-validated upstream, and both shapes the checker now rejects. That is
  the conservative direction (sim rejects what GPU tolerates), and it costs
  the kernels one barrier they did not strictly need on hardware.
  Re-alloc similarly requires the previous generation's dealloc edge, and a
  bare (never-drained) access fails even with an ordering edge.
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
- **`elect` is not a convergence point in the model**: hardware `elect.sync`
  synchronizes its membermask, but `if_elected` lowers to a plain `If` and the
  checker credits no convergence — a handoff relying on the election's
  implicit sync must write `warp_sync`. Over-report direction; see the
  join-point ledger in `docs/ir-ops.md`.
- **`WarpSync` is sim/checker vocabulary**: codegen does not emit
  `bar.warp.sync` for it, so a proof that leans on `warp_sync` compiles to
  code relying on the warp launching converged.
- **`fence.mbarrier_init` seals but is not required**: the checker treats it
  as a release-side fence (it seals its executing lanes for a later relaxed
  cluster-barrier arrive), but no pass demands that a kernel publish its
  mbarrier OBJECTS with one — barrier cells initialized and handed to a peer
  without it are accepted.

## HW-illegal but sim-accepts (do not use; rejection pending kernel rework)

- **peer (remote_coord) mbarrier wait** — used by the fp16/bf16 + nvfp4 GEMMs
  for the leader's wait on the peer CTA's `smem_full`/`sf_full`,
  silently dropped by codegen; correctness rests on the leader-routing
  argument (both CTAs' TMA tx lands on the LEADER's physical barrier copy,
  so the leader's local wait observes everything).
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
