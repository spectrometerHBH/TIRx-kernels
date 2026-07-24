# RFC: Protocol Checker Algorithm

**Status:** Draft.
**Scope:** `nymph-rust` offline protocol checker algorithms, trace schema
requirements, and report contract.

## 1. Problem

The trace interpreter executes one canonical protocol run. The offline checker
must prove stronger schedule-independent conditions: barriers close correctly,
blocking operations cannot deadlock, conflicting shared memory accesses are
happens-before ordered, async/proxy/TMEM windows are drained or ordered, and no
required proof fact is missing from the trace.

The checker does not re-run numeric value simulation or enumerate interleavings.
It consumes `Kernel IR + completed TraceEvent stream`.

Full checker passes run only after trace execution returns `Passed`.

- Trace execution `Failed`: return the trace failure and diagnostics.
- Trace execution `Inconclusive`: return the trace warning and stop.
- Trace execution `Passed`: run the offline checker pipeline.

## 2. Execution Model

- A stream is one warp (`(cta, warp)`) and executes its dynamic operations in
  order.
- Each modeled op is atomic at checker granularity.
- Different streams may interleave arbitrarily. Warps of one CTA are separate
  streams, so cross-warp pairs are ordered only by explicit synchronization.
- Within one stream, program order orders each LANE against itself. A pair of
  accesses by DIFFERENT lanes of the warp is ordered by a warp-level sync
  between them: a passed cooperative barrier (`warp_sync`, `wg_sync`,
  `cta_sync`, a named barrier, `cluster_sync`) or a warp-collective
  instruction (`ldmatrix`, `stmatrix`, `tcgen05.ld`, `tcgen05.st`, warp MMA),
  which every lane converges on. An access performed by an async engine
  (`proxy = async`) is ordered by its own drain rule, checked by the async
  passes.
- Barriers, waits, syncs, fences, commits, and async drains constrain legal
  interleavings.
- Event vector order is canonical trace order, not cross-stream
  happens-before.

## 3. Trace Schema Requirements

Every `TraceEvent` carries:

```rust
stmt_id: u32
stmt_kind: String
```

Memory access events use a single physical byte region:

```rust
Region {
    owner: PoolId,
    boxes: Vec<BoxN>,
    tensor_id: u32,
    // Per-lane attribution of `boxes`, before they are merged across the
    // warp's lanes. Carried for lane-divergent SMEM/TMEM accesses and for
    // single-lane masks; `None` means every executing lane touches the
    // whole region.
    lane_boxes: Option<Vec<(u8, BoxN)>>,
}

BoxN {
    ranges: Vec<(usize, usize)>, // half-open [start, end)
}

PoolId =
    Smem { cta_id }
  | Tmem { cta_id }
  | Gmem { tensor_id }
  | Reg  { cta_id, tensor_id }
```

The memory events are:

```rust
TraceEventKind::Read {
    region: Region,
    proxy: MemoryProxy,
    access_kind: MemoryAccessKind,
    scope: AccessScope,
}

TraceEventKind::Write {
    region: Region,
    proxy: MemoryProxy,
    access_kind: MemoryAccessKind,
    scope: AccessScope,
}
```

`TmemAlloc` and `TmemDealloc` also carry `Region`. `TmemWait` is retained as a
non-memory event.

`AccessScope` includes `stream_id`, `cluster_id`, global `cta_id`,
`ctaid_in_cluster`, `lane_count`, and `warp_id`. `MbarTargetEvent` includes
`mbar_id`, `cluster_id`, `ctaid_in_cluster`, and `stage`. Resource keys for
mbarrier, sync, deadlock, and cluster-scope fences must use cluster identity
where applicable.

`MemoryAccessKind` preserves instruction semantics:

```rust
MemoryAccessKind::Tensor(TensorAccessKind)
MemoryAccessKind::Tmem(TmemAsyncKind)
```

`tensor_id` is used for diagnostics and GMEM/REG owner consistency. Alias
identity is `PoolId + boxes`.

## 4. Region Contract

Trace emission must project memory accesses into exact physical byte regions.
The checker does not resolve logical tensor slices into physical footprints.

Validation rules:

- `boxes` is non-empty;
- all boxes in one region have the same rank;
- SMEM, GMEM, and REG regions are rank-1 byte ranges;
- TMEM regions are rank-2 boxes in `(lane, lane_byte)`;
- every dimension satisfies `start < end`;
- all ranges are within owner bounds;
- GMEM/REG owner tensor id matches `region.tensor_id`.

There are no separate alternate region types. A sparse access is `Vec<BoxN>`,
and an individual point is a unit box in every dimension.

Projection rules:

- SMEM tensor slices emit exact rank-1 physical byte boxes.
- GMEM and REG tensor slices emit rank-1 byte boxes for trace completeness.
- TMEM logical/cell accesses emit rank-2 byte boxes; IR column `col` maps to
  lane bytes `col * 4`.
- `tcgen05.mma` emits one box per closed-form layout block:
  `[lane, lane + rows) x [col * 4, (col + cols) * 4)`.
- A contiguous TMA or MMA rectangle must remain one box. Disconnected footprints
  are multiple boxes, not an enclosing box.
- If exact projection is impossible, trace emission returns `Inconclusive`
  rather than emitting an approximate region.

## 5. Result Semantics

The public status remains:

- `Passed`: all implemented required checks completed and found no violation.
- `Failed`: at least one check proved a protocol violation.
- `Inconclusive`: no violation was proved, but a required proof obligation
  could not be discharged from trace/IR facts.

Reports include `pass_summary`, `warnings`, and typed `diagnostics`. Every
failure names the statement/event, resource or region, expected condition, and
witness events when available.

## 6. Checker Architecture

The checker is a pass pipeline over `Kernel IR + TraceEvents`:

- `trace_schema_audit`: validates common event fields and non-region schema.
- `trace_region_audit`: validates `Region` owner/rank/bounds/empty-box rules.
- `barrier_cycle_audit`: audits mbarrier and sync counters/cycles.
- `ordering_analysis`: builds schedule-independent happens-before edges from
  per-stream program order, mbar phase-keyed release/acquire, and cooperative
  barriers. A cooperative-barrier generation is keyed by
  `(statement, rendezvous domain, cycle)`: all arrivals of one generation join
  into a release clock frozen by the completing arrival, and every passage of
  that generation acquires it.
- `deadlock_freedom`: proves modeled blocking operations have no wait cycle.
- `async_group_lifetime`: checks cp.async/TMA source windows.
- `tmem_async_hazard`: checks overlapping TMEM async windows.
- `tmem_lifecycle_order`: proves TMEM band lifetimes (alloc -> use -> free
  ordering, and the free waits for the access to be observed complete).
- `memory_race_check`: checks SMEM/TMEM data-race freedom.
- `cluster_peer_consistency`, `scheduler_handoff_consistency`,
  `trace_gap_audit`.

Implementations may share scans or helper indexes, but reports should use these
pass names or close equivalents.

## 7. Unified Memory Analysis

Memory helpers operate on `Region` directly:

```text
regions_overlap(a, b):
  a.owner == b.owner
  and any BoxN pair intersects in every dimension

region_covers(a, b):
  a.owner == b.owner
  and every box in b is covered by some box in a
```

`async_group_lifetime`, `tmem_async_hazard`, and
`tmem_lifecycle_order` all use these helpers. Barrier and deadlock passes do not
inspect regions.

`memory_race_check` proves that every conflicting shared memory pair is ordered.
For SMEM/TMEM accesses `A` and `B`, a race is:

```text
A.region.owner == B.region.owner
regions_overlap(A.region, B.region)
A.mode == Write || B.mode == Write
!happens_before(A.event_idx, B.event_idx)
!happens_before(B.event_idx, A.event_idx)
```

A conflicting pair on ONE stream is a race between two lanes of that warp
unless an intra-warp ordering fact covers it:

```text
A.stmt_kind is warp-collective || B.stmt_kind is warp-collective
|| A.proxy == async || B.proxy == async
|| a cooperative-barrier passage or warp-collective event of that stream
   lies strictly between A and B
|| every overlapping (A.lane_boxes, B.lane_boxes) pair shares one lane
```

The pass keeps per-owner read and write frontiers. Reads query only the write
frontier, may prune older covered reads ordered before the current read, and
then join the read frontier. Writes query both frontiers, then join the write
frontier and may prune older covered frontier entries that are ordered before
the new write. Partial cover is retained conservatively. Reads do not split
writes or drive global spatial partitioning.

The pass does not prove prior-write completeness, read-from identity, or write
consumption. A read without a prior write is not an error by itself, and an
unread write is not an error by itself. Failures use `memory_data_race`,
across streams and between lanes of one warp alike — the happens-before
clock is per (warp, lane), so both are the same missing edge (the diagnostic
names the lane pair when the conflict is lane-attributed).

## 8. Pass Notes

### `trace_region_audit`

Malformed regions indicate instrumentation or projection errors. Out-of-bounds
regions fail. Unknown tensor metadata needed for GMEM/REG bounds is
`Inconclusive`.

### `barrier_cycle_audit`

Mbarriers are keyed by `(mbar_id, cluster_id, ctaid_in_cluster, stage)`. Sync
barriers are keyed by their modeled hardware/resource identity. The pass checks
init before use, counter underflow/overflow, transaction byte balance, and
wait/completion phase consistency.

### `deadlock_freedom`

The pass builds a wait-for graph from blocking waits and their release witness
events. A cycle with no release outside the cycle is a failure. The completed
canonical trace alone is not a proof of deadlock freedom.

### `async_group_lifetime`

The pass tracks committed async source windows. Same-stream overlapping writes
before `wait_group` fail. Cross-stream overlaps without structural ordering are
recorded as trace gaps.

### `tmem_async_hazard`

The pass tracks TMEM async read/write windows by stream and async kind. Same
stream conflicting overlaps before the earlier window is closed fail.
Cross-stream conflicting overlaps without ordering are trace gaps.

### `tmem_lifecycle_order`

Two halves. The RESOURCE ALGEBRA: allocation and deallocation regions must
match exactly, and a band may not be allocated over a live one. The LIFETIME,
which is pure happens-before:

- an access belongs to the generation whose band covers it, that it is ordered
  after, and whose free it is not already past — a binding the clock decides,
  not the trace walk (which only enumerates generations, and cannot answer
  lifetime questions since it saw one schedule);
- a re-allocation of an overlapping band must be ordered after the previous
  generation's free;
- the free must be ordered after every access of that generation is observed to
  have COMPLETED. Ordering the issue is not enough: a barrier orders
  instruction streams, it does not drain an engine. The observation point is
  `tcgen05.wait::ld/st` for a load or store, and for an mma or cp the wait on a
  barrier some `tcgen05.commit` handed the work to (a commit tracks every async
  op the warp issued before it, so waiting any such barrier suffices).

### cross-proxy publication (inside `memory_race_check`)

An engine access that overlaps a generic-proxy SMEM write carries a SECOND
obligation beyond the ordering one: the write must have been published across
the proxy boundary. `fence.proxy.async` releases the fencing thread's view
into the async-proxy engines at the fence's address scope, and the engine
access acquires the view published for the address space it touches, so both
obligations are decided against the same clocks. A pair that is ordered but
unpublished reports `proxy_fence_missing`.

## 9. Test Matrix

Required coverage includes:

- `include_events=False` omits Python events by default;
- `include_events=True` returns unified `read`/`write` events with `stmt_id` and
  `stmt_kind`;
- same-owner overlap and different-owner no-alias region helpers;
- rank mismatch and out-of-bounds region audit failures;
- sparse/unit-box overlap and box coverage;
- SMEM physical aliasing where different tensor ids share bytes;
- TMEM lifecycle coverage over byte boxes;
- Layout F MMA emits multiple disjoint TMEM boxes and untouched lane gaps do
  not alias;
- TMA and contiguous MMA footprints remain single boxes when physically
  contiguous;
- `async_group_lifetime`, `tmem_async_hazard`, `tmem_lifecycle_order`,
  and `memory_race_check` use unified overlap helpers;
- value-mode behavior is unchanged.
