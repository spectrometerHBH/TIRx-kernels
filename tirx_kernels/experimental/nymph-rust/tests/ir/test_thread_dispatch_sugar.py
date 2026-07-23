"""Builder sugar for thread dispatch: if_warp / if_warpgroup / if_lane /
if_elected / set_maxnreg emit canonical If predicates over scope values."""

import pytest

nr = pytest.importorskip("nymph_rs")


def builder(num_warps=8):
    return nr.IRBuilder(
        "sugar", num_warps=num_warps, smem_size_bytes=0, launch_shape=(1,), cluster_shape=(1,)
    )


def test_sugar_predicates_are_canonical_eq_shapes():
    b = builder()
    with b.if_warp(3):
        pass
    with b.if_warpgroup(1):
        b.set_maxnreg(232)
    with b.if_lane(5):
        pass
    with b.if_elected():
        pass
    kernel = b.build()
    expected = [("warp_id", 3), ("warpgroup_id", 1), ("lane_id", 5), ("lane_id", 0)]
    for stmt, (kind, value) in zip(kernel.body, expected, strict=True):
        cond = stmt.cond
        assert cond.op == nr.ScalarOp.EQ
        assert cond.args[0].kind == kind
        assert cond.args[1] == value


def test_setmaxnreg_nested_under_warpgroup_validates():
    b = builder()
    with b.if_warpgroup(1):
        b.set_maxnreg(232)
    b.build()


def test_nested_sugar_runs_in_simulator():
    b = nr.IRBuilder(
        "sugar_sim", num_warps=4, smem_size_bytes=64, launch_shape=(1,), cluster_shape=(1,)
    )
    scratch = b.tensor(space=nr.MemorySpace.SMEM, dtype=nr.DType.U32, shape=[1], byte_offset=0)
    with b.if_warp(1):
        with b.if_elected():
            b.store_scalar(scratch[0:1], 7)
    nr.interpret(b.build(), {})
