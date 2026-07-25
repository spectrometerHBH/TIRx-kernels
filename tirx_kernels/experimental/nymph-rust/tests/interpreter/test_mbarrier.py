import nymph_rs as nr
from helpers import builder, expect_runtime_error, run


def test_mbarrier_wait_runtime_failures_are_closed():
    b = builder("mbarrier_expect_tx_deadlock")
    expect_tx = b.mbar(kind=nr.MBarKind.TMA)
    # mbarrier.init is per-thread now: issue it from a single elected thread.
    with b.if_warp(0), b.if_elected():
        b.mbarrier_init(expect_tx, count=1)
    with b.if_warp(0), b.if_elected():
        b.mbarrier_expect_tx(expect_tx, bytes=8)
        b.mbarrier_arrive(expect_tx)
        b.mbarrier_wait(expect_tx, phase=0)

    with expect_runtime_error("deadlock"):
        run(b.build())

    b = builder("mbar_duplicate")
    duplicate = b.mbar(kind=nr.MBarKind.TMA)
    # elected so the error pins DOUBLE init, not the multi-lane init.
    with b.if_warp(0), b.if_elected():
        b.mbarrier_init(duplicate, count=1)
        b.mbarrier_init(duplicate, count=1)

    with expect_runtime_error("mbarrier_already_initialized"):
        run(b.build())

    b = builder("mbar_remote_oob")
    remote = b.mbar(kind=nr.MBarKind.TMA)
    with b.if_warp(0), b.if_elected():
        b.mbarrier_arrive(b.mbar_ref(remote, remote_coord=2))

    with expect_runtime_error("mbarrier_remote_cta_oob"):
        run(b.build())
