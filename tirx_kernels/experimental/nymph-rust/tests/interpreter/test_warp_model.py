"""Per-warp execution model regressions.

These pin the three behaviors the model switch was FOR — each was wrong or
unrepresentable under the fused role-stream model:

1. Intra-warpgroup cross-warp races are REAL races: staging writes by all
   warps followed by a single-thread bulk store must be rejected without the
   wg_sync (the fused model's same-stream shortcut silently ordered them).
2. PTX execution-thread rules are enforced: single-thread issue ops reject
   multi-thread cohorts, and mbarrier arrives really are per-thread (one
   full-warp arrive(1) fills a count=32 barrier — under once-per-execution
   semantics the waiter would deadlock).
3. Cross-warp producer/consumer where the CONSUMER precedes the producer in
   source order completes (the fused epoch schedule falsely deadlocked it).
"""

import numpy as np
import nymph_rs as nr
from helpers import builder, expect_runtime_error, gmem_arg, run


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
    # The exact hole the fused model had: warp 1's reg_store vs warp 0's
    # tma_store read were same-stream there and never checked.
    report = nr.check_protocol(_epilogue_kernel(with_wg_sync=False))
    assert report["status"] == "Failed"
    codes = {d["code"] for d in report["diagnostics"]}
    assert "memory_data_race" in codes, codes


def test_wg_sync_between_staging_and_store_passes():
    report = nr.check_protocol(_epilogue_kernel(with_wg_sync=True))
    assert report["status"] == "Passed", report["diagnostics"]


def test_single_thread_issue_ops_reject_multi_thread_cohorts():
    # tma_store from a full warp: PTX says single-thread issue.
    b = builder("gate_tma", num_warps=4, smem_size_bytes=64)
    out = gmem_arg(b, shape=(4,))
    smem = b.tensor(space=nr.MemorySpace.SMEM, dtype=nr.DType.U32, shape=[4], byte_offset=0)
    with b.if_warp(0):
        b.tma_store(out, smem[0:4], coords=(0,), shape=(4,))
    with expect_runtime_error("tma_store_mask"):
        run(b.build())

    # tcgen05_mma from a full warp (the OLD model REQUIRED the full warp).
    b = builder("gate_mma", num_warps=4, smem_size_bytes=8192)
    acc = b.tensor(
        space=nr.MemorySpace.TMEM,
        dtype=nr.DType.F32,
        shape=(128, 16),
        layout=nr.TmemLayout(nr.TmemLayoutKind.LANE_128, col_start=0),
    )
    a = b.tensor(space=nr.MemorySpace.SMEM, dtype=nr.DType.F16, shape=(128, 16), byte_offset=0)
    bb = b.tensor(space=nr.MemorySpace.SMEM, dtype=nr.DType.F16, shape=(16, 16), byte_offset=4096)
    with b.if_warp(0):
        b.tmem_alloc(acc, n_cols=32)
        b.tcgen05_mma(acc, a, bb, m=128, n=16, k=16)
    with expect_runtime_error("tcgen05_mma_mask"):
        run(b.build())


def test_mbarrier_arrive_is_per_thread():
    # count=32; warp 1 issues ONE arrive(1) statement with all 32 threads.
    # Per-thread semantics fills the barrier and the waiter completes; the
    # old once-per-execution semantics would leave 31 pending arrivals and
    # deadlock the waiter.
    b = builder("mbar_per_thread", num_warps=4)
    mbar = b.mbar(kind=nr.MBarKind.THREAD, stages=1)
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
    # The fused epoch model serialized source order and falsely deadlocked
    # this shape; per-warp streams interleave and complete it.
    b = builder("consume_then_produce", num_warps=4, smem_size_bytes=64)
    src = gmem_arg(b, shape=(4,))
    out = gmem_arg(b, shape=(4,))
    reg = b.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.U32, shape=[4])
    smem = b.tensor(space=nr.MemorySpace.SMEM, dtype=nr.DType.U32, shape=[4], byte_offset=0)
    mbar = b.mbar(kind=nr.MBarKind.THREAD, stages=1)
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
