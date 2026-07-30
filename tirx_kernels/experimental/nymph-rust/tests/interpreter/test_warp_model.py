"""Per-warp execution model regressions — the load-bearing properties.

1. Cross-warp accesses inside one warpgroup are race-checked: staging writes
   by all four warps followed by a single-thread bulk store need the wg_sync
   between them, and the checker reports the race when it is missing.
2. PTX execution-thread rules hold: single-thread issue ops are rejected at
   BUILD by the validator's single_issue_scope rule when not under an
   explicit single-lane branch, and mbarrier arrives are per-thread (one
   full-warp arrive(1) fills a count=32 barrier).
3. Warps interleave freely, so a consumer that precedes its producer in
   source order still completes.
"""

import numpy as np
import nymph_rs as nr
import pytest
from helpers import builder, gmem_arg, run


def _epilogue_kernel(with_wg_sync: bool):
    b = builder("epilogue_race", num_warps=4, smem_size_bytes=4096)
    out = gmem_arg(b, shape=(128, 8))
    smem = b.tensor(space=nr.MemorySpace.SMEM, dtype=nr.DType.U32, shape=[128, 8], byte_offset=0)
    frag = b.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.U32, shape=[8])
    with b.if_warpgroup(0):
        # Staging: every thread writes its own smem row.
        b.reg_fill(frag, 7)
        b.reg_store(nr.TensorSlice(tensor=smem, offsets=(b.tid_in_wg(), 0), shape=(1, 8)), frag)
        if with_wg_sync:
            b.wg_sync(barrier_id=1)
        with b.if_(b.tid_in_wg().eq(0)):
            b.fence(kind=nr.FenceKind.ASYNC_PROXY, scope=nr.FenceScope.CTA)
            b.tma_store(out, smem, coords=(0, 0), shape=(128, 8))
    return b.build()


def test_missing_wg_sync_between_staging_and_store_is_a_race():
    # warp 1's reg_store and warp 0's tma_store read overlap in c_smem with
    # nothing ordering them.
    report = nr.check_protocol(_epilogue_kernel(with_wg_sync=False))
    assert report["status"] == "Failed"
    codes = {d["code"] for d in report["diagnostics"]}
    assert "memory_data_race" in codes, codes


def test_wg_sync_between_staging_and_store_passes():
    report = nr.check_protocol(_epilogue_kernel(with_wg_sync=True))
    assert report["status"] == "Passed", report["diagnostics"]


def test_single_thread_issue_ops_reject_multi_lane_masks():
    # tma_store from a full warp: rejected at BUILD by the validator's
    # single_issue_scope rule (hardware single-issue ops need an explicit
    # single-lane branch) — before the interpreter's mask gate could run.
    b = builder("gate_tma", num_warps=4, smem_size_bytes=64)
    out = gmem_arg(b, shape=(4,))
    smem = b.tensor(space=nr.MemorySpace.SMEM, dtype=nr.DType.U32, shape=[4], byte_offset=0)
    with b.if_warp(0):
        b.tma_store(out, smem[0:4], coords=(0,), shape=(4,))
    with pytest.raises(ValueError, match="single_issue_scope"):
        b.build()

    # tcgen05_mma from a full warp: same build-time rejection.
    b = builder("gate_mma", num_warps=4, smem_size_bytes=8192)
    acc = b.tmem_tensor(0).at(0, 0)
    a = b.tensor(space=nr.MemorySpace.SMEM, dtype=nr.DType.F16, shape=(128, 16), byte_offset=0)
    bb = b.tensor(space=nr.MemorySpace.SMEM, dtype=nr.DType.F16, shape=(16, 16), byte_offset=4096)
    a_tile = b.smem_tile(a, prefix_indices=(), row_offset=0, col_offset=0, rows=128, cols=16)
    b_tile = b.smem_tile(bb, prefix_indices=(), row_offset=0, col_offset=0, rows=16, cols=16)
    with b.if_warp(0):
        b.tmem_alloc(0, 32, addr_byte_offset=0)
        b.tcgen05_mma(
            acc,
            b.mma_a_smem(a_tile),
            b_tile,
            mma_m=128,
            mma_n=16,
            format="f16",
            block_scale=None,
            accum=False,
            trans_a=False,
            trans_b=False,
            ws=False,
            cta_group=1,
        )
    with pytest.raises(ValueError, match="single_issue_scope"):
        b.build()


def test_mbarrier_arrive_is_per_thread():
    # count=32; warp 1 issues ONE arrive(1) statement with all 32 threads.
    # Each thread applies the operand, so the 32 arrivals fill the barrier and
    # the waiter completes.
    b = builder("mbar_per_thread", num_warps=4)
    mbar = b.mbar(kind=nr.MBarKind.THREAD, byte_offset=0, stages=1)
    with b.if_warp(0):
        with b.if_elected():
            b.mbarrier_init(mbar, count=32)
    b.cta_sync()
    with b.if_warp(0):
        b.mbarrier_wait(mbar, phase=0)
    with b.if_warp(1):
        b.mbarrier_arrive(mbar, count=1)
    run(b.build())


def test_consumer_before_producer_in_source_order_completes():
    # warp 0 (earlier in source) WAITS; warp 1 (later in source) ARRIVES.
    # Source order does not constrain warps, so the two streams interleave and
    # the handshake completes.
    b = builder("consume_then_produce", num_warps=4, smem_size_bytes=64)
    src = gmem_arg(b, shape=(4,))
    out = gmem_arg(b, shape=(4,))
    reg = b.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.U32, shape=[4])
    smem = b.tensor(space=nr.MemorySpace.SMEM, dtype=nr.DType.U32, shape=[4], byte_offset=0)
    mbar = b.mbar(kind=nr.MBarKind.THREAD, byte_offset=0, stages=1)
    with b.if_warp(0):
        with b.if_elected():
            b.mbarrier_init(mbar, count=1)
    b.cta_sync()
    with b.if_warp(0):
        b.mbarrier_wait(mbar, phase=0)
        with b.if_elected():
            b.reg_load(reg, smem[0:4])
            b.reg_store(out, reg)
    with b.if_warp(1):
        with b.if_elected():
            b.reg_load(reg, src)
            b.reg_store(smem[0:4], reg)
            b.fence(kind=nr.FenceKind.MEMORY, scope=nr.FenceScope.CTA)
            b.mbarrier_arrive(mbar, count=1)
    outputs = run(b.build(), {src: np.asarray([5, 6, 7, 8], dtype=np.uint32)})
    np.testing.assert_array_equal(np.asarray(outputs[out.id], dtype=np.uint32), [5, 6, 7, 8])
