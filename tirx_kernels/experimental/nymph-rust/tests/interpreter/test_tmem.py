import pytest
from helpers import builder, expect_runtime_error, nr, run, tmem_operand


def test_tmem_lifecycle_failures_are_fail_closed():
    # The de-tensored IR validates the column-band lifecycle at Kernel
    # construction (validate.rs check_tmem_alloc_bands): a duplicate or
    # overlapping band, or a dealloc that does not exactly match a live alloc,
    # is rejected before the interpreter ever runs.
    cases = []

    b = builder("tmem_duplicate")
    with b.kernel_init(warp=0):
        b.tmem_alloc(0, 128)
        b.tmem_alloc(0, 128)
    cases.append((b, "overlaps a live allocation"))

    b = builder("tmem_missing")
    with b.kernel_finalize(warp=0):
        b.tmem_dealloc(0, 128)
    cases.append((b, "does not match a live allocation"))

    b = builder("tmem_overlap")
    with b.kernel_init(warp=0):
        b.tmem_alloc(0, 64)
        b.tmem_alloc(32, 64)
    cases.append((b, "overlaps a live allocation"))

    b = builder("tmem_mismatch")
    with b.kernel_init(warp=0):
        b.tmem_alloc(0, 64)
    with b.kernel_finalize(warp=0):
        b.tmem_dealloc(32, 64)
    cases.append((b, "does not match a live allocation"))

    for case, msg in cases:
        with pytest.raises(ValueError, match=msg):
            case.build()


def test_tmem_operand_outside_live_band_fails_at_build():
    # Every TMEM operand's static column span must sit inside a live band at its
    # program point — an operand escaping every band is a build-time error now.
    b = builder("tmem_operand_outside_band")
    reg = b.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.F32, shape=(1,))
    with b.kernel_init(warp=0):
        b.tmem_alloc(0, 64)
    with b.role(warpgroup=0):
        b.tcgen05_ld(reg, tmem_operand(0, 64), shape="32x32b", num=1)
    with pytest.raises(ValueError, match="not inside a live tmem allocation band"):
        b.build()


def test_tmem_alloc_order_is_interpreter_fail_closed():
    # Non-overlapping bands pass the IR walk, but the interpreter still rejects
    # an alloc whose n_cols GROWS within a CTA (the hardware's alloc-order rule).
    b = builder("tmem_order")
    with b.kernel_init(warp=0):
        b.tmem_alloc(128, 64)
        b.tmem_alloc(0, 128)
    with expect_runtime_error("tmem_allocation_order"):
        run(b.build())


def test_tmem_cta_group2_missing_peer_fails_closed():
    b = builder("tmem_cta_group2_missing_peer")
    with b.kernel_init(warp=0):
        b.tmem_alloc(0, 128, cta_group=2)

    with expect_runtime_error("tmem_collective_peer"):
        run(b.build())
