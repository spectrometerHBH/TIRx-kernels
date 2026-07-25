# GDN TIRx bring-up: bisection matrix + GPU evidence log

Status as of the lower-tirx lowering wave (commits b7738ffd..43916ed8).
Purpose: pin down the ns1_t64 GPU deadlock + WarpMma NaN before the
correctness gate. Filled in as evidence lands; do not edit conclusions
retroactively.

## Verification ladder state

- cargo test: 160 + 15 passed.
- gdn value sim (interpreter, bit-exact vs numpy oracle): **75 passed**.
- compile gate (`tvm.compile`, tirx pipeline): **5 passed** (bootstrap /
  fp16_bf16 / nvfp4 / empty-if / **gdn_prefill**).
- full pytest minus gdn file: **401 passed**.
- GPU (ns1_t64, fixed-length kernel): compiles, launches, **deadlocks**
  (`torch.cuda.synchronize()` never returns; 124 on a 240 s timeout).

## Bisection matrix (GPU probes, protocol-checked first)

| Probe (kernel content) | Protocol | GPU result |
| --- | --- | --- |
| `bisect_st`: tcgen05.st + wait.st + ld 32x32b + wait.ld roundtrip | n/a | **WORKS** — thread-0 row round-trips 1.0 |
| `bisect_mma64`: TMA → m=64 F-datapath MMA + commit → wait → 16x256b ld readback | Passed | **WORKS** — out = 8.0 (Σ16 1.0·0.5) |
| `bisect_mma64_tb`: same with trans_b | Passed | **WORKS** — out = 8.0 |
| `bisect_mma64_ta`: same with trans_a | Passed | **WORKS** — out = 8.0 |
| `bisect_warpmma`: ldmatrix + mma.sync(m16n8k8 bf16) + stmatrix | (probe-bug) | **no hang**, out = NaN (see §WarpMma NaN) |
| fp16_bf16_gemm `run_bench('fp16', 1024³)` | — | **RUNS** — cosine ≥0.99, tirx 6.69 µs |
| gdn stripped: no WY inverse, no CG1 readbacks | Passed | **HANGS** |
| gdn stripped + no CG0 rss, no gate compute | Passed | **HANGS** |
| gdn stripped + no EPI warp, no zrow st | Passed | **HANGS** |

Early probe kernels of mine deadlocked *by construction* (TMA producer warp
nested inside the consumer warpgroup's branch — the top-level If chaining
serializes the wg body before the nested producer in the producer's own
stream). fp16's proven shape puts producer warps in a *different* warpgroup
than their consumers; all later probes follow that.

## Established facts

- All *individual* HW paths used by gdn work on B200: tcgen05.st/wait.st,
  32x32b + 16x256b ld, m=64 datapath-F tcgen05.mma, transA/transB MMA,
  ldmatrix/mma.sync/stmatrix (no hang), TMA load/store, named barriers.
- The gdn barrier skeleton (7 GEMMs, d_*/f_*/ready barriers, gate/beta full
  -tile TMA, k_s stage ring, EPI TMA store) hangs even with ALL compute
  stripped — so the divergence is in the *stream/barrier skeleton or the
  remaining instruction forms* (full-tile f32 gate/beta TMA, k_s ring TMA,
  point stores, scalar reg elementwise, scheduler loop), not the warp-matrix
  or MMA datapaths.
- The WarpMma probe produces NaN — open whether it's my WarpMma lowering
  (mma.legacy fragment order / accumulator init) or the probe's own dump
  path (its write-back loop was admittedly broken).

## WarpMma NaN — interpreter contract

Interpreter (`src/interpreter/semantics/warpmma.rs`): D = A·Bᵀ + C over the
standard warp fragment layout (groupID g = lane/4, threadID t = lane%4):
- A (M×K bf16 packed u32): reg ru → tile (mt=ru%2, kt=ru/2); halves hold
  A[mt·8 + g][kt·8 + 2t + h].
- B (N×K bf16 packed u32): reg ru → kt=ru; halves hold B[2t + h][kt·8 + g].
- C/D (f32): reg ri holds [(ri/2)·8 + g][2t + ri%2].

The lowering emits `T.ptx.mma.legacy("m16n8k8", "row", "col", ab, ab,
"float32", A_flat_ab.data, 0, B_flat_ab.data, 0, D_flat.data, 0, False,
dtype="float32")` — the intrinsic derives the fragment register counts from
shape+dtype and reads the fragments in PTX register order; the u32 words
must hold (lo,hi) = (elem 2t, elem 2t+1) per the ldmatrix pack (little-
endian, matches the interpreter's pack_b16x2). Open: accumulator init in
the probe (reg_fill(acc, 0.0) under a narrowed `if_` may not have executed),
the probe's broken write-back loop.

## Next steps

1. Sentinel-write bisect on the stripped gdn skeleton → stall site.
2. Fix WarpMma probe dump → confirm/deny mma.sync values vs interpreter.
3. Then the main ladder: ns1_t64 cosine → 6 shapes → full pytest → bench.

## Resolution wave (commits 2d1448ac, 02b55833, c5da1718, db96f0fe, zero-inference wave)

All issues above are now resolved; final gate numbers at the bottom.

### Root causes found (each with GPU evidence)

1. **mbarrier.arrive undercount → pipeline deadlock** (`2d1448ac`). The
   codegen emitted `arrive` under an inferred single-issue guard (elect_sync);
   the interpreter arrives once per executing lane and gdn's barrier counts
   are sized for exactly that (32/128 arrivals). One arrival never completed
   a phase: TMA/GATE warps exited, MMA/CG0/CG1/EPI spun in try_wait forever
   (cuda-gdb backtrace). Fix: arrive emits per-thread; single-arrival sites
   are elected in the IR already.
2. **wg-view column-slice offset dropped** (`02b55833`). Sliced reg element
   ops (`frag[:, 1:2]`, `[r:r+1]`) lost the slice offset through the wg
   tile view (B200 probe: `fill(frag[:,1:2])` wrote element 0). Fix: sliced
   reg elementwise ops lower to the per-thread scalar flat form; full-extent
   ops keep `Tx.wg.*`.
3. **Aux-view reverse-construction invalid + WarpSync no-op** (`02b55833`).
   The `alloc_local(W)+view(...)` Apply mapped (tid,j) out of bounds; use
   `T.wg_reg_tile + .local()` (storage-layout view, thread axis peeled).
   `WarpSync` lowered to nothing; now emits `T.cuda.warp_sync()`.
4. **chain_top_level_ifs reorder → TMA-after-wait deadlock** (`c5da1718`).
   The top-level If-chaining pass ran each group's warpgroup-prefix body
   BEFORE its warp-level branches regardless of source order: a probe's
   warp-1 TMA issue (source-order before the consumer warpgroup's
   `mbarrier_wait`) was emitted AFTER the wait → spin forever. Only the
   canonical wg-prefix-then-warp-roles order chains now; mixed runs emit
   flat (source order, always sound). Regression test:
   `codegen.rs::tests::warp_before_warpgroup_run_stays_flat`.
5. **WarpMma B fragment transposed in the interpreter** (`db96f0fe`). The
   interpreter unpacked the mma.sync B operand as bmat[(2t+h)·kk + g]; real
   sm_100 hardware (pinned by tests/mma_sync_hardware.rs's direct-feed
   layout B[n=g][k=kt·8+2t+h]) wants bmat[g·kk + kt·8+2t+h]. A non-trans
   ldmatrix feed therefore computes D = A·tileᵀ on hardware but A·tile in
   the sim — every warp-mma consumer written against the sim (the whole
   hierarchical-inverse merge) was wrong on GPU by exactly a B transpose.
   GPU battery evidence: merge round-1 dcs matched `Q@Cᵀ` (max err 0.0019),
   not `Q@C` (1.14). Fixed as one atomic pair: interpreter unpack +
   gdn `_ldB` switches to a trans ldmatrix + `test_warp_mma` feeds B
   non-trans. The earlier `bisect_warpmma` probe missed it because
   ones×0.5 is transpose-invariant.
6. **RegLoad GMEM src silently dropped** (`c5da1718`). A point
   `reg_load` from a GMEM tensor fell through the codegen arm and emitted
   NOTHING (a probe's init loop stored uninitialized registers everywhere).
   Now lowered (raw element assignment) + the arm fails closed on any
   other unsupported src space.

### Merge-chain verification (probe_merge, standalone replica)

Post-GJ input (unit diag, diag 8×8 blocks inverted, strict lower), the
kernel's exact `_ldA/_ldB/_store8/_merge` on mode-3 swizzled ainv_s/dcs_s,
rounds b=8 ×4, b=16 ×2, b=32 ×1 with cross-warp wg_sync: **GPU vs numpy
replica bit-exact (max diff 0.000000)**, replica vs `np.linalg.inv` max
abs 0.0347 (bf16 store rounding only).

### Final gate numbers

- cargo test: 162 + 15 passed.
- pytest battery (warp_mma + gdn value sim + compile gate): **84 passed**.
- Oracle cosine gate (nymph GPU vs numpy oracle, no flashinfer JIT):

| shape | cos_out | cos_state |
| --- | --- | --- |
| ns1_t64 | 0.99999 | 1.00000 |
| ns1_t512 | 0.99999 | 1.00000 |
| ns1_t2048 | 0.99999 | 1.00000 |
| ns20_t192 | 0.99999 | 1.00000 |
| ns48_t64 | 0.99999 | 1.00000 |
| v_70_130 | 0.99999 | 1.00000 |

(v_70_130's out buffer is NaN-padded beyond the packed rows by design —
"padding content is irrelevant; the kernel masks OOB" — the comparison
crops to the 200 packed rows.)

### Zero-inference guard rule (user-mandated)

Codegen no longer synthesizes ANY emission guard from the thread scope:
single-issue ops (tma_load/store, cp_async_bulk_s2cluster, tcgen05
mma/cp/commit, mbarrier.init, clc_try_cancel) are legal only under an
explicit single-lane If (validator `single_issue_scope`, sticky one-lane
upper bound through runtime sub-branches); per-thread ops (arrive /
expect_tx / store_scalar / async-proxy fence) emit per-thread, matching
the interpreter. Rule + audit table: docs/ir-ops.md §"Codegen emission
guards". Negative tests: validate.rs `single_issue_scope_rule`,
tests/test_compile_gate.py `test_single_issue_scope_negative`.

### Zero-inference follow-up (8689f62a) + final gates

- `single_issue_scope` v1 rejected flash_bwd_sm100's s2cluster site (57
  tests): guard `(~kept) & (tid == 0)` hides the one-lane predicate behind
  a runtime And operand. `proves_single_lane_per_warp` proves the bound
  through And/Mul chains (intersection only narrows) — no kernel edit.
- Full pytest round 1 after the zero-inference wave: 418 passed / 61
  failed (57 flash_bwd + 4 stale fixtures) → all fixed; round 2: see below.
- bench/bench_gdn_wave1.py `--rounds 5` (GPU 1): all 6 shapes compile gate
  PASS, oracle-cos out=1.0000 state=1.0000. Timing (us, fi/nymph):
  ns1_t64 16.6/27.4 (0.61), ns1_t512 33.6/149.3 (0.23),
  ns1_t2048 92.7/512.0 (0.18), ns20_t192 42.5/118.1 (0.36),
  ns48_t64 50.0/80.9 (0.62), v_70_130 22.2/80.7 (0.28).
- v_70_130's earlier cos_out=nan was a probe artifact: `_nymph_callable`
  pads the varlen out buffer with NaN by contract ("padding content is
  irrelevant; the kernel masks OOB"); cropping to the 200 packed rows
  gives 0.99999/1.00000. No kernel bug.

### Wave-2 hardening landed: intermediate sim⇄GPU diff harness

`tests/tools/smidiff.py` + `tests/gpu/test_gpu_sim_diff.py` productize the
/tmp/gdn_debug.py + /tmp/probe_merge.py discovery flow as the standard
bring-up equipment:

- INJECTION is an optional IR pass over a BUILT kernel (no kernel-body
  edits): per DumpSpec it appends one GMEM dump arg and inserts per-thread
  point-store dump blocks at structural sites (after/before a predicate on
  statement kind + fields — e.g. `arrive_on(f_kk_id)`), with a chunk slot
  from the enclosing for_loop var and an optional task dimension from the
  enclosing for_each_task var (`task_mod` — REQUIRED for persistent
  launches; without it, all (seq, eh) tasks race onto the same dump slots —
  measured: bogus 0.89 "divergence" on m_s).
- The SAME instrumented kernel runs through `nr.interpret` (the reference)
  and the tirx GPU codegen; dumps are diffed point-wise, first divergence
  reported as (cell, sim, gpu). Default tolerance bit-exact; calibrated
  per-tensor tolerances in the shell.
- gdn shell (ns1_t64 + v_70_130): m_s / attn / A_inv(post-fold) dumps —
  sim⇄GPU max_abs 1.1e-06 / 1.2e-04 / 9.5e-07, tolerances pinned at
  1e-4 / 3e-3 / 1e-4. Sites: after f_kk / after f_qk arrives, and BEFORE
  the ainv_ready arrive (ainv_s aliases vnewt_s — after the release the
  region may already hold NV staging: a real race in BOTH backends, found
  by this harness on its first run).
- New IR-introspection surface for the pass (py.rs): PyStmt.kind /
  barrier_id / mbar_id / tensor / unroll, PyKernel.cluster_shape /
  smem_pool.

### Emission invariants — status after zero-inference

| Invariant | Status | Coverage |
| --- | --- | --- |
| mbarrier arrive count == arriving lanes | STRUCTURAL: arrive emits per-thread, no guard can be synthesized (zero-inference); count-vs-lanes consistency is a value property enforced by the interpreter's mbarrier accounting + the protocol checker's phase completion in every sim test | codegen test `mbarrier_arrive_emits_bare_per_thread` |
| per-thread ops (expect_tx / arrive_expect_tx / store_scalar / async-proxy fence) emit per-lane | STRUCTURAL (same reasoning) | codegen test `per_thread_ops_emit_bare_at_warpgroup_scope` |
| hardware single-issue ops only under explicit single-lane branch | validator `single_issue_scope` + codegen `emit_single_issue` hard error | validate.rs `single_issue_scope_rule`, compile-gate `test_single_issue_scope_negative`, interpreter mask tests now assert build-time rejection |
| single-lane proof holds through And-chains with runtime operands | `proves_single_lane_per_warp` (intersection only narrows) | thread_filter.rs `single_lane_proof_through_and_chains` |
