"""Intra-warp cross-lane memory visibility.

On sm_70+ (Independent Thread Scheduling) the lanes of a warp advance through
their shared instruction stream independently: the checker's happens-before
clock gives every (warp, lane) its own dimension, so a masked write is
invisible to every other lane — and to every release the other lanes perform —
until a warp-level convergence point (warp_sync / a warp-collective) or a real
synchronization edge delivers it. These pin that visibility rule at both ends:
an undelivered cross-lane dependency is a `memory_data_race`, and the checker
stays quiet when the kernel supplies the ordering (convergence, a
synchronization edge, or per-lane program order alone).
"""

import nymph_rs as nr
from helpers import builder, gmem_arg, reg_tensor, smem_tensor

LANES = 32
STMATRIX_ROWS = 16  # stmatrix .x2 uses address lanes 0..15 as row starts


def _codes(report):
    return {d["code"] for d in report["diagnostics"]}


def _cell(tensor, index):
    """The one-element slice `tensor[index]`, addressed per lane."""
    return nr.TensorSlice(tensor=tensor, offsets=(index,), shape=(1,))


def _rotate_kernel(*, rotate: bool, warp_sync: bool):
    """Each lane writes its own smem cell, then reads one lane's cell.

    `rotate` picks WHOSE cell is read: lane+1 (cross-lane) or its own
    (same-lane). `warp_sync` inserts the converging barrier between them.
    """
    b = builder("cross_lane_rotate", smem_size_bytes=LANES * 4)
    smem = smem_tensor(b, dtype=nr.DType.U32, shape=(LANES,), byte_offset=0)
    reg = reg_tensor(b, dtype=nr.DType.U32, shape=(1,))
    out = gmem_arg(b, dtype=nr.DType.U32, shape=(LANES,))
    with b.if_warp(0):
        b.reg_fill(reg, 7)
        b.reg_store(_cell(smem, b.lane_id()), reg)
        if warp_sync:
            b.warp_sync()
        source = (b.lane_id() + 1) % LANES if rotate else b.lane_id()
        b.reg_load(reg, _cell(smem, source))
        b.reg_store(_cell(out, b.lane_id()), reg)
    return b.build()


def test_lane_rotated_read_after_write_without_warp_sync_is_a_race():
    report = nr.check_protocol(_rotate_kernel(rotate=True, warp_sync=False))
    assert report["status"] == "Failed"
    assert "memory_data_race" in _codes(report), report["diagnostics"]
    detail = report["diagnostics"][0]["details"]
    assert detail["left_mode"] == "write"
    assert detail["right_mode"] == "read"
    # The witness names the two lanes whose footprints overlap.
    assert len(detail["lanes"].split(",")) == 2, detail["lanes"]


def test_warp_sync_orders_the_lane_rotated_read():
    report = nr.check_protocol(_rotate_kernel(rotate=True, warp_sync=True))
    assert report["status"] == "Passed", report["diagnostics"]


def test_same_lane_reuse_needs_no_sync():
    # Lane i writes and reads only cell i: per-lane program order is all the
    # ordering hardware needs, with or without a barrier.
    report = nr.check_protocol(_rotate_kernel(rotate=False, warp_sync=False))
    assert report["status"] == "Passed", report["diagnostics"]


def test_same_lane_reuse_across_loop_iterations_needs_no_sync():
    b = builder("cross_lane_same_lane_loop", smem_size_bytes=LANES * 4)
    smem = smem_tensor(b, dtype=nr.DType.U32, shape=(LANES,), byte_offset=0)
    reg = reg_tensor(b, dtype=nr.DType.U32, shape=(1,))
    out = gmem_arg(b, dtype=nr.DType.U32, shape=(LANES,))
    with b.if_warp(0):
        b.reg_fill(reg, 1)
        with b.for_loop(stop=4):
            b.reg_store(_cell(smem, b.lane_id()), reg)
            b.reg_load(reg, _cell(smem, b.lane_id()))
        b.reg_store(_cell(out, b.lane_id()), reg)
    report = nr.check_protocol(b.build())
    assert report["status"] == "Passed", report["diagnostics"]


def _stmatrix_kernel(*, collective: bool):
    """A cross-lane SMEM producer followed by a lane-rotated row read.

    `collective=True` produces the rows with `stmatrix` — a warp-collective
    instruction every lane converges on, which orders the pair by itself.
    `collective=False` writes the same footprint with per-lane `reg_store`s,
    which carries no cross-lane ordering.
    """
    b = builder("cross_lane_stmatrix", smem_size_bytes=STMATRIX_ROWS * 8 * 2)
    smem = smem_tensor(b, dtype=nr.DType.F16, shape=(STMATRIX_ROWS, 8), byte_offset=0)
    frag = reg_tensor(b, dtype=nr.DType.U32, shape=(2,))
    row = reg_tensor(b, dtype=nr.DType.F16, shape=(8,))
    with b.if_warp(0):
        b.reg_fill(frag, 3)
        if collective:
            b.stmatrix(smem[b.lane_id() % STMATRIX_ROWS, 0:8], frag, num=2, trans=False)
        else:
            b.reg_fill(row, 3)
            b.reg_store(smem[b.lane_id() % STMATRIX_ROWS, 0:8], row)
        b.reg_load(row, smem[(b.lane_id() + 1) % STMATRIX_ROWS, 0:8])
    return b.build()


def test_warp_collective_member_orders_the_pair():
    report = nr.check_protocol(_stmatrix_kernel(collective=True))
    assert report["status"] == "Passed", report["diagnostics"]


def test_collective_ordering_test_has_teeth():
    # Same footprint, same rotated read, produced by plain per-lane stores:
    # without the collective the dependency is an unordered cross-lane one.
    report = nr.check_protocol(_stmatrix_kernel(collective=False))
    assert report["status"] == "Failed"
    assert "memory_data_race" in _codes(report), report["diagnostics"]


def _publish_kernel(*, write: str, warp_sync: bool, arrive: str = "elected"):
    """Warp 0 writes smem then publishes with an mbarrier arrive; warp 1
    waits on the mbar and reads the bytes.

    `write` picks the producer:
      - "half":     lanes 16..31 each store their own cell (lane-divergent);
      - "full":     all 32 lanes each store their own cell (still
                    lane-divergent — 31 of the writers are not the arriver);
      - "elected":  lane 0 stores one cell (the arriving lane itself wrote
                    every published byte, per-lane program order suffices);
      - "stmatrix": a warp-collective produces the bytes (the warp converges
                    AT the write).
    `arrive` picks the publisher: "elected" (lane 0 only, count=1) or "full"
    (every lane arrives itself, count=32 — each lane's own store is
    release-ordered before its own arrive).
    `warp_sync` inserts the converging sync between the write and the arrive.
    """
    b = builder("cross_lane_publish", smem_size_bytes=STMATRIX_ROWS * 8 * 2)
    smem = smem_tensor(b, dtype=nr.DType.U32, shape=(LANES,), byte_offset=0)
    reg = reg_tensor(b, dtype=nr.DType.U32, shape=(1,))
    out = gmem_arg(b, dtype=nr.DType.U32, shape=(LANES,))
    full = b.mbar(kind=nr.MBarKind.THREAD)
    with b.if_warp(0), b.if_elected():
        b.mbarrier_init(full, count=1 if arrive == "elected" else LANES)
    b.cta_sync()
    with b.if_warp(0):
        if write == "half":
            b.reg_fill(reg, 7)
            with b.if_(b.lane_id() >= 16):
                b.reg_store(_cell(smem, b.lane_id()), reg)
        elif write == "full":
            b.reg_fill(reg, 7)
            b.reg_store(_cell(smem, b.lane_id()), reg)
        elif write == "elected":
            with b.if_elected():
                b.store_scalar(smem[0], 7)
        elif write == "stmatrix":
            smem_f16 = b.tensor(
                space=nr.MemorySpace.SMEM, dtype=nr.DType.F16, shape=(16, 8), byte_offset=0
            )
            frag = reg_tensor(b, dtype=nr.DType.U32, shape=(2,))
            b.reg_fill(frag, 3)
            b.stmatrix(smem_f16[b.lane_id() % 16, 0:8], frag, num=2, trans=False)
        if warp_sync:
            b.warp_sync()
        if arrive == "elected":
            with b.if_elected():
                b.mbarrier_arrive(full)
        else:
            b.mbarrier_arrive(full)
    with b.if_warp(1):
        b.mbarrier_wait(full, phase=0)
        b.reg_load(reg, _cell(smem, 16 + (b.lane_id() % 16)))
        b.reg_store(_cell(out, b.lane_id()), reg)
    return b.build()


def _race_details(report):
    return next(d for d in report["diagnostics"] if d["code"] == "memory_data_race")["details"]


def test_divergent_write_published_by_elected_arrive_fails():
    # Lanes 16..31 store, lane 0 arrives: the arrive's release carries only
    # lane 0's program order, so the waiter's read of the divergent bytes is
    # unordered against the writes — judged at the consuming access.
    report = nr.check_protocol(_publish_kernel(write="half", warp_sync=False))
    assert report["status"] == "Failed"
    assert "memory_data_race" in _codes(report), report["diagnostics"]
    detail = _race_details(report)
    assert detail["left_stmt_id"] != detail["right_stmt_id"]
    assert detail["owner"].startswith("smem"), detail


def test_warp_sync_before_the_elected_arrive_passes():
    # The warp_sync converges the lanes before lane 0 publishes.
    report = nr.check_protocol(_publish_kernel(write="half", warp_sync=True))
    assert report["status"] == "Passed", report["diagnostics"]


def test_full_warp_divergent_write_published_by_elected_arrive_fails():
    # All 32 lanes store their own cell, only lane 0 arrives: the other 31
    # writers' dimensions are absent from the published clock — the same
    # hazard as the half-warp variant (a full mask does not converge by
    # itself).
    report = nr.check_protocol(_publish_kernel(write="full", warp_sync=False))
    assert report["status"] == "Failed"
    assert "memory_data_race" in _codes(report), report["diagnostics"]


def test_full_warp_arrive_publishes_every_lanes_own_store():
    # Every lane arrives itself (mbar count = 32): lane i's store is
    # release-ordered before lane i's arrive, and the waiter's phase
    # completes only after ALL 32 arrivals — the publish covers every
    # writing lane with no warp_sync (fa4's softmax/correction handshakes).
    report = nr.check_protocol(_publish_kernel(write="full", warp_sync=False, arrive="full"))
    assert report["status"] == "Passed", report["diagnostics"]


def test_single_lane_write_published_by_elected_arrive_passes():
    # No divergent write: the one writing lane is the arriving lane, and its
    # own program order runs store -> arrive (the scheduler-bridge pattern).
    report = nr.check_protocol(_publish_kernel(write="elected", warp_sync=False))
    assert report["status"] == "Passed", report["diagnostics"]


def test_collective_write_published_by_elected_arrive_passes():
    # The producing write is itself warp-collective: the warp converges AT
    # the stmatrix, so the elected arrive publishes converged bytes.
    report = nr.check_protocol(_publish_kernel(write="stmatrix", warp_sync=False))
    assert report["status"] == "Passed", report["diagnostics"]


def test_unconsumed_divergent_write_with_elected_arrive_passes():
    # The mbar is only ever waited by the ARRIVING warp itself, and nobody
    # reads the divergent bytes: with no conflicting access pair there is
    # nothing to order — visibility is judged where data is consumed, never
    # at the release itself.
    b = builder("cross_lane_publish_within_warp", smem_size_bytes=LANES * 4)
    smem = smem_tensor(b, dtype=nr.DType.U32, shape=(LANES,), byte_offset=0)
    reg = reg_tensor(b, dtype=nr.DType.U32, shape=(1,))
    mbar = b.mbar(kind=nr.MBarKind.THREAD)
    with b.if_warp(0), b.if_elected():
        b.mbarrier_init(mbar, count=1)
    b.cta_sync()
    with b.if_warp(0):
        b.reg_fill(reg, 7)
        with b.if_(b.lane_id() >= 16):
            b.reg_store(_cell(smem, b.lane_id()), reg)
        with b.if_elected():
            b.mbarrier_arrive(mbar)
        b.mbarrier_wait(mbar, phase=0)
    report = nr.check_protocol(b.build())
    assert report["status"] == "Passed", report["diagnostics"]


def _partial_barrier_kernel(*, writers_join: bool):
    """A runtime-valued predicate (Unknown static filter, bypasses validate)
    masks each warp down to a 16-lane half; the two half-warps fill a
    32-thread named barrier together.

    `writers_join=True`: the joining lanes 0..15 write their own cells before
    the barrier, then read lane+1's cell after it — the barrier's
    release/acquire delivers each ARRIVING lane's writes to every other
    arriver (hardware bar.sync memory semantics; no warp convergence needed).
    `writers_join=False`: lanes 16..31 write and never join; the joining
    lanes read those cells after the barrier — an absent lane's writes are in
    nobody's release, so the dependency stays unordered.
    """
    b = builder("cross_lane_partial_sync", smem_size_bytes=LANES * 4)
    smem = smem_tensor(b, dtype=nr.DType.U32, shape=(LANES,), byte_offset=0)
    reg = reg_tensor(b, dtype=nr.DType.U32, shape=(1,))
    out = gmem_arg(b, dtype=nr.DType.U32, shape=(LANES,))
    with b.if_warp(0):
        half = b.scalar(initial=16, dtype=nr.ScalarDType.I32)
        if writers_join:
            with b.if_(b.lane_id() < half):
                b.reg_fill(reg, 7)
                b.reg_store(_cell(smem, b.lane_id()), reg)
                b.named_barrier(barrier_id=1, num_warps=1)
                b.reg_load(reg, _cell(smem, (b.lane_id() + 1) % 16))
                b.reg_store(_cell(out, b.lane_id()), reg)
        else:
            with b.if_(b.lane_id() >= half):
                b.reg_fill(reg, 7)
                b.reg_store(_cell(smem, b.lane_id()), reg)
            with b.if_(b.lane_id() < half):
                b.named_barrier(barrier_id=1, num_warps=1)
                b.reg_load(reg, _cell(smem, 16 + (b.lane_id() % 16)))
                b.reg_store(_cell(out, b.lane_id()), reg)
    with b.if_warp(1):
        half1 = b.scalar(initial=16, dtype=nr.ScalarDType.I32)
        with b.if_(b.lane_id() < half1):
            b.named_barrier(barrier_id=1, num_warps=1)
    return b.build()


def test_partial_mask_barrier_orders_the_arrived_lanes():
    # Each arriving lane's pre-barrier writes ride its arrival into the
    # barrier's release; the passage's acquire hands them to every other
    # arriver — a real rendezvous orders its members without converging the
    # warp.
    report = nr.check_protocol(_partial_barrier_kernel(writers_join=True), include_events=True)
    assert report["status"] == "Passed", report["diagnostics"]
    # The construction really produced a PARTIAL-mask passage.
    sync_lane_counts = {e["scope"]["lane_count"] for e in report["events"] if e["kind"] == "sync"}
    assert sync_lane_counts == {16}, sync_lane_counts


def test_partial_mask_barrier_publishes_nothing_for_absent_lanes():
    # The writers never joined the rendezvous: no arrival carries their
    # program order, so the post-barrier read of their cells is unordered —
    # and the passage must not be credited as a full-warp convergence.
    report = nr.check_protocol(_partial_barrier_kernel(writers_join=False), include_events=True)
    assert report["status"] == "Failed"
    assert "memory_data_race" in _codes(report), report["diagnostics"]
    sync_lane_counts = {e["scope"]["lane_count"] for e in report["events"] if e["kind"] == "sync"}
    assert sync_lane_counts == {16}, sync_lane_counts


def test_lane_boxes_are_exposed_on_lane_divergent_events():
    b = builder("cross_lane_events", smem_size_bytes=LANES * 4)
    smem = smem_tensor(b, dtype=nr.DType.U32, shape=(LANES,), byte_offset=0)
    reg = reg_tensor(b, dtype=nr.DType.U32, shape=(1,))
    with b.if_warp(0):
        b.reg_fill(reg, 7)
        b.reg_store(_cell(smem, b.lane_id()), reg)
    report = nr.check_protocol(b.build(), include_events=True)
    assert report["status"] == "Passed", report["diagnostics"]
    writes = [
        e
        for e in report["events"]
        if e["kind"] == "write" and e["region"]["owner"]["kind"] == "smem"
    ]
    lane_boxes = writes[0]["region"]["lane_boxes"]
    assert lane_boxes is not None
    assert len(lane_boxes) == LANES
    # Lane i owns cell i: bytes [4i, 4i+4).
    for entry in lane_boxes:
        lane = entry["lane"]
        assert entry["box"]["ranges"] == [(4 * lane, 4 * lane + 4)]
