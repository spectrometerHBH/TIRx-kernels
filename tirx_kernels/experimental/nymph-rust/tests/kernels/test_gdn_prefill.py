"""Build + validate tests for the chunked Gated Delta Net prefill kernel.

The kernel is the instruction-faithful nymph port of the FlashInfer Blackwell
CuteDSL chunked GDN prefill. This phase covers IR construction + static
validation (the mma.sync / log2 / TMEM surface); protocol and oracle value
tests land with the ordering-audit phase.
"""

import pytest
from nymph_rs.kernels.gdn_prefill import (
    CONFIGS_SUPPORTED,
    HEAD_CONFIGS,
    GdnPrefillConfig,
    build_gdn_prefill,
)


@pytest.mark.parametrize("cfg", CONFIGS_SUPPORTED, ids=[c["label"] for c in CONFIGS_SUPPORTED])
def test_gdn_prefill_builds_and_validates(cfg):
    kernel = build_gdn_prefill(GdnPrefillConfig(num_seqs=cfg["num_seqs"], seqlen=cfg["seqlen"]))
    kernel.validate()


@pytest.mark.parametrize("hc", HEAD_CONFIGS, ids=[h["label"] for h in HEAD_CONFIGS])
def test_gdn_prefill_head_configs_build(hc):
    cfg = GdnPrefillConfig(
        num_seqs=1, seqlen=128, num_q_heads=hc["num_q_heads"], num_v_heads=hc["num_v_heads"]
    )
    kernel = build_gdn_prefill(cfg)
    kernel.validate()


@pytest.mark.parametrize("io_dtype", ["bfloat16", "float16"])
def test_gdn_prefill_io_dtypes_build(io_dtype):
    kernel = build_gdn_prefill(GdnPrefillConfig(num_seqs=1, seqlen=128, io_dtype=io_dtype))
    kernel.validate()
