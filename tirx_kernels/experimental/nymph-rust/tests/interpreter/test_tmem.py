import pytest
from helpers import builder, expect_runtime_error, run


def test_tmem_lifecycle_failures_are_fail_closed():
    # Most invalid lifecycle shapes now fail closed at kernel BUILD time: the
    # validator's single-live-base-0-band walk (base_col == 0, one live band,
    # dealloc must match) rejects them before the interpreter ever runs,
    # shadowing the old runtime codes (tmem_already_allocated,
    # missing_tmem_allocation, tmem_allocation_overlap,
    # tmem_allocation_mismatch).
    cases = []

    b = builder("tmem_duplicate")
    with b.if_warp(0):
        b.tmem_alloc(0, 128)
        b.tmem_alloc(0, 128)
    cases.append((b, "while another allocation is still live"))

    b = builder("tmem_missing")
    with b.if_warp(0):
        b.tmem_dealloc(0, 128)
    cases.append((b, "tmem_dealloc does not match a live allocation"))

    b = builder("tmem_overlap")
    with b.if_warp(0):
        b.tmem_alloc(0, 64)
        b.tmem_alloc(32, 64)
    cases.append((b, "tmem_alloc base_col must be 0"))

    b = builder("tmem_mismatch")
    with b.if_warp(0):
        b.tmem_alloc(0, 64)
    with b.if_warp(0):
        b.tmem_dealloc(32, 64)
    cases.append((b, "tmem_dealloc does not match a live allocation"))

    for case, pattern in cases:
        with pytest.raises(ValueError, match=pattern):
            case.build()

    # tmem_allocation_order stays a RUNTIME failure: a freed band may be
    # re-allocated (the single-live-band rule is satisfied), but the new
    # band's n_cols may not exceed the previous one's within a CTA.
    b = builder("tmem_order")
    with b.if_warp(0):
        b.tmem_alloc(0, 64)
        b.tmem_dealloc(0, 64)
        b.tmem_alloc(0, 128)
    with expect_runtime_error("tmem_allocation_order"):
        run(b.build())


def test_tmem_cta_group2_missing_peer_fails_closed():
    # A cta_group=2 alloc needs a cluster pair; with a 1-CTA cluster the
    # validator rejects the alloc's cta_group against the kernel-level group
    # at build time (shadowing the old runtime tmem_collective_peer failure —
    # with a real pair the peer always exists).
    b = builder("tmem_cta_group2_missing_peer")
    with b.if_warp(0):
        b.tmem_alloc(0, 128, cta_group=2)

    with pytest.raises(ValueError, match=r"tmem_alloc cta_group=2 != kernel cta_group=1"):
        b.build()
