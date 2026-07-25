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
