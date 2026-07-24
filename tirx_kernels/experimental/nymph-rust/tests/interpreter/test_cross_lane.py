"""Intra-warp cross-lane memory dependency regressions.

On sm_70+ (Independent Thread Scheduling) the lanes of a warp advance through
their shared instruction stream independently, so a memory dependency between
DIFFERENT lanes of one warp is ordered by an explicit `warp_sync` or a
warp-collective instruction. These pin the checker's `intra_warp_cross_lane_race`
rule at both ends — it fires on a genuinely cross-lane dependency, and it stays
quiet when the kernel supplies the ordering (sync, collective) or when each lane
only depends on its own program order.
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
    assert "intra_warp_cross_lane_race" in _codes(report), report["diagnostics"]
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
    assert "intra_warp_cross_lane_race" in _codes(report), report["diagnostics"]


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
