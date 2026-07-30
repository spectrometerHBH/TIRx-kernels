# LIMITATIONS — known sim ↔ silicon divergences

Read this BEFORE trusting a green value-sim / protocol-check run. Each entry is
a place where "simulator says OK" does not imply "GPU behaves the same".
Per-op detail lives in `docs/ir-ops.md`; this file is the short trust boundary.

The mbarrier transaction counter is not a divergence: both simulator and
checker use a signed balance. `expect_tx` subtracts expected bytes and an
engine completion adds actual bytes, so either may happen first. A phase flips
only after that balance and the pending-arrival count are both exactly zero.

## Timing / asynchrony

- **err 719 launch fault under full-async OVERLAP** (nvfp4_gemm.py, the
  default-overlap epilogue path):
  with ~10+ tasks/cluster a timing-sensitive `cudaErrorLaunchFailure` appears;
  hidden by any serialization. The 8192 no-overlap config is believed to
  sidestep it (40+ stress reps clean) — believed, not proven.
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
- **tcgen05.cp destination/source formats**: the IR explicitly carries
  destination TMEM address, source SMEM tile, CTA group, one of the supported
  physical shapes, and multicast. It does not yet carry the optional PTX
  `dst_fmt/src_fmt` decompression pair. The supported form is lowered as one
  statement-local `Tx.copy_async`.
- **MMA accumulation order**: OpenBLAS sgemm blocking order, not the tensor
  core's fixed order. Bit-exact only for exact arithmetic (the nvfp4 value
  tests choose exact inputs on purpose).
- **f16/bf16 tcgen05.ld/st packed path**: no B200 fixture (b32 shapes only);
  marked suspect in ir-ops.md.
- **CLC round-robin oracle**: query_cancel returns canonical round-robin
  tasks. Only "each task exactly once + termination" holds for arbitrary
  orders; value regressions are only checked under round-robin.
- **lane selection is not a convergence point**: `if_elected` is only builder
  sugar for the explicit IR predicate `lane_id == 0`. Codegen prints that
  predicate literally and does not introduce `elect.sync`; the checker
  therefore credits no implicit convergence. A handoff needing warp ordering
  must write `warp_sync`. See the join-point ledger in `docs/ir-ops.md`.
- **No inferred warp convergence**: an explicit `WarpSync` lowers to
  `T.cuda.warp_sync()`, but codegen never inserts one from thread-scope
  analysis. Any proof that needs convergence must keep the operation in IR.
- **`fence.mbarrier_init` seals but is not required**: the checker treats it
  as a release-side fence (it seals its executing lanes for a later relaxed
  cluster-barrier arrive), but no pass demands that a kernel publish its
  mbarrier OBJECTS with one — barrier cells initialized and handed to a peer
  without it are accepted.

## sim-only ops (no codegen lowering — sim-green means nothing for silicon)

GmemAtomicAdd, GmemWaitEq, CpAsyncBulkS2Cluster, TmaStore.reduce_add,
Fence Memory/View, RegMax/RegMin/RegBitwise/RegReduce and the remaining
specialized register reductions/rescales. Compile gates reject them; do not
treat their sim validation as GPU evidence. WarpMma, RegUnary,
Tcgen05Ld/Tcgen05St and the matrix load/store family do have explicit
lowerings; their individual format/shape restrictions still fail closed.

## IR restrictions that keep validate and codegen on one semantics

- **TMEM allocation**: one base-0 column band per kernel, live at most once; every
  alloc/dealloc/relinquish carries the kernel cta_group; no alloc after a
  `relinquish_alloc_permit` (also enforced per CTA at runtime, PTX
  §9.7.17.7.1); lifecycle ops are rejected inside re-executing bodies and
  runtime-value conditionals (static warp/lane dispatch `If`s remain legal).
  Multi-band/base-offset TMEM allocations are IR-illegal on purpose. A
  `TmemTensor` is not an owning buffer and stores only `start_col`; every
  instruction adds its explicit row/column and declares its own non-owning
  statement-local view. The checker's lifecycle pass requires happens-before edges
  alloc→access→dealloc (real sync, not the sampled interleaving).
- **Dynamic SMEM**: all owned tensors, mbarrier cells, and the optional
  four-byte TMEM-address cell live in one `T.SMEMPool`. Their IR byte offsets
  are absolute, and `smem_size_bytes` is the complete committed extent,
  including padding and metadata.
- **Scale factors**: scale-factor storage is a plain row-major physical view.
  Its dimensions and absolute byte offset are explicit; codegen does not infer
  a scale layout or cache aliases from dtype or `tcgen05` use.
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
