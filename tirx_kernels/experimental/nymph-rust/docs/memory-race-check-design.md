# Detailed Design: `memory_race_check`

`memory_race_check` proves shared memory data-race freedom for completed trace
executions. It consumes `Kernel IR + TraceEvent stream` after trace execution
has already returned `Passed`.

## Scope

The pass covers memory events expressed as unified physical byte regions:

```rust
Read  { region: Region, proxy, access_kind, scope }
Write { region: Region, proxy, access_kind, scope }
```

The checker target set is:

- SMEM rank-1 byte boxes in `PoolId::Smem { cta_id }`.
- TMEM rank-2 byte boxes in `PoolId::Tmem { cta_id }`, with coordinates
  `(lane, lane_byte)`.

GMEM and REG events remain in the trace for diagnostics and future analyses, but
they are not shared race targets for this pass. Alias identity is `PoolId +
boxes`; `tensor_id` is diagnostic metadata and does not separate aliases.

## Required Result

For every pair of SMEM/TMEM accesses `A` and `B`, the pass fails when all of
these are true:

```text
A.region.owner == B.region.owner
regions_overlap(A.region, B.region)
A.mode == Write || B.mode == Write
!happens_before(A.event_idx, B.event_idx)
!happens_before(B.event_idx, A.event_idx)
```

This pass does not prove prior-write completeness, read-from identity, or write
consumption. A read without a prior write is not an error by itself. A write that
is never read is not an error by itself.

The pass reports one race code, `memory_data_race`, whether the unordered
pair sits on two streams or on two lanes of one warp — both are the same
missing edge in the per-lane happens-before relation. The diagnostic carries
a lane-pair witness when the conflict is lane-attributed.

## Access Model

The pass normalizes memory events into:

```text
AccessRecord {
  event_idx,
  mode: Read | Write,
  region: Region,
}
```

Only records whose owner is SMEM or TMEM participate. Sparse accesses remain
`Vec<BoxN>` regions. A point is a unit box in every dimension.

## Frontier Algorithm

The checker maintains one `MemoryRaceFrontier` per `PoolId`:

```text
MemoryRaceFrontier {
  reads:  Vec<AccessRecord>,
  writes: Vec<AccessRecord>,
}
```

Events are processed in canonical trace order, but cross-stream trace order is
not an ordering proof.

- `Read`: query overlapping entries in the write frontier. A conflicting
  lane-slice pair with no happens-before order in either direction (see
  Ordering Inputs) is a `memory_data_race`. Then prune older read-frontier
  entries that are covered by this read and prunably ordered before it, and
  append the read to the read frontier.
- `Write`: query overlapping entries in both the write frontier and the read
  frontier with the same rule. Then prune older frontier entries that are
  fully covered by the new write and prunably ordered before it. Finally
  append the new write.

Reads never split existing writes or define global spatial partitions. A large write followed
by a partial read performs an overlap/HB query against the large write; it does
not create `[read)` and `[remaining)` frontier fragments. Partial cover is retained
conservatively.

## Ordering Inputs

Happens-before is per LANE. `OrderingAnalysis` gives every `(stream, lane)`
its own vector-clock dimension: an event ticks exactly its executing lanes
(the trace scope's `active_lanes` mask), a release publishes the join of the
ARRIVING lanes' clocks only, an acquire joins the release into the acquiring
lanes, and a warp-level convergence point — a warp-collective instruction
(`ldmatrix`, `stmatrix`, `tcgen05.ld/st`, warp MMA, the `.sync.aligned` TMEM
alloc/dealloc) or a full-warp cooperative passage — folds all 32 lanes into
one shared clock. On sm_70+ the lanes of a warp advance independently, so
this is the hardware relation: a masked write is invisible to every other
lane, and to every release those lanes perform, until a convergence point or
a real synchronization edge delivers it.

The race walk judges conflicting SLICE pairs through two queries:

```text
ordered_lane(a_event, a_lane, b_event, b_lane) -> bool   // one slice pair
happens_before(a_event, b_event) -> bool                  // every slice pair
```

A slice is `(lane, bytes)`: the region's `lane_boxes` entry when the builder
attributed bytes to lanes (lane-divergent SMEM/TMEM accesses and single-lane
masks), else every executing lane touching the whole footprint. For each
overlapping slice pair the pass requires `ordered_lane` in one direction
(either direction across streams; only forward within one stream, where a
later event's lane tick can never sit below an earlier one's). Same-lane
same-stream pairs are ordered by per-lane program order; everything else
must come out of the clock. When neither side is attributed, every pair
conflicts and the whole-event `happens_before` is the exact test.

Same-stream pairs where either member is performed by an async engine
(`proxy = async`) are not judged here: a copy/mma engine touches those bytes,
not a lane, and the engine-window passes (`async_group_lifetime`,
`tcgen05_async_hazard`) own those hazards against their own drains.

Frontier pruning needs an ordering that COMPOSES with any later conflict:
the whole-event relation (transitive), or same-lane coverage — both sides
attributed, no byte shared across different lanes, so each prior slice is
covered by the same lane's current slice under per-lane program order.

## Projection Requirements

Projection happens before trace emission. The checker does not resolve logical
tensor footprints.

- SMEM tensor slices must emit exact physical byte boxes.
- TMEM logical/cell accesses must emit exact `(lane, lane_byte)` boxes.
- TMA GMEM transfer endpoints should emit one byte-count box for the transfer
  span.
- `tcgen05.mma` must emit one box per closed-form MMA layout block:
  `[lane, lane + rows) x [col * 4, (col + cols) * 4)`.
- If an op cannot project an exact region, trace emission should return
  `Inconclusive`; it must not emit an approximate enclosing box.

## Diagnostics

`memory_data_race` diagnostics include:

- `left_event_idx` and `right_event_idx`;
- left/right `stmt_id` and `stmt_kind` when available;
- left/right access modes;
- owner summary;
- `lanes`, the unordered slice pair's lane numbers (`all` when neither side
  is lane-attributed);
- one overlapping witness box (the named lanes' overlap when attributed).

The pass works on SMEM and TMEM through the same `Region` overlap path. Tests
cover physical SMEM aliasing, write/write and write/read races, read/read
non-conflicts, owner separation, non-overlap, mbar HB ordering,
large-write/partial-read behavior, TMEM Layout F untouched lane gaps, and the
per-lane visibility theorems: lane-rotated dependency, `warp_sync` delivery,
same-lane reuse, warp-collective convergence, masked-write handoffs,
divergent-write publication through narrow and full-warp arrives, and
partial-mask rendezvous (arrived lanes ordered, absent lanes not).
