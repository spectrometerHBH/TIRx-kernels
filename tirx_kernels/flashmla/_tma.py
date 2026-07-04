"""Shared TMA copy_async config for the sparse FlashMLA phase1 kernels."""

from __future__ import annotations

from typing import Any


def tma_config(
    *,
    mbar: Any = None,
    cta_group: int | None = None,
    cta_mask: Any = None,
    cache_hint: Any = None,
    mbarrier_addr: Any = None,
    gather_axis: int | None = None,
    indexer: list[Any] | None = None,
    prefetch_tensormap: bool = True,
    tensormap_l2_promotion: str = "L2::256B",
) -> dict[str, Any]:
    """Return the FlashMLA-standard ``Tx.copy_async`` TMA config dict.

    Every TMA copy in the phase1 kernels prefetches its tensormap and asks
    for ``L2::256B`` promotion; the remaining knobs are per-site:

    - ``mbar``: completion mbarrier pointer (loads; stores omit it).
    - ``cta_group`` / ``cta_mask``: tcgen05 CTA-pair scope of the copy.
    - ``cache_hint``: ``"evict_first"`` or a ``T.uint64`` encoded L2 hint.
    - ``mbarrier_addr``: mbar addressing switch for the gather path.
    - ``gather_axis`` / ``indexer``: tensor gather4 row gathering.

    ``None`` means "leave the knob out of the config" so dispatch defaults
    apply; a misspelled keyword fails loudly instead of passing through.
    """
    cfg: dict[str, Any] = {
        "dispatch": "tma",
        "prefetch_tensormap": prefetch_tensormap,
        "tensormap_l2_promotion": tensormap_l2_promotion,
    }
    for key, value in (
        ("mbar", mbar),
        ("cta_group", cta_group),
        ("cta_mask", cta_mask),
        ("cache_hint", cache_hint),
        ("mbarrier_addr", mbarrier_addr),
        ("gather_axis", gather_axis),
        ("indexer", indexer),
    ):
        if value is not None:
            cfg[key] = value
    return cfg
