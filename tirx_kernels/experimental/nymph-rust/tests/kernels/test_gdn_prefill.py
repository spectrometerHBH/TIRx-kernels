"""Build + protocol + value tests for the chunked Gated Delta Net prefill kernel.

The kernel is the instruction-faithful nymph port of the FlashInfer Blackwell
CuteDSL chunked GDN prefill. Values are checked against the validated numpy
oracle (``_gdn_oracle.py``, the FLA ``chunk_gated_delta_rule`` forward);
protocol checks run under the per-warp execution model.
"""

import sys
from pathlib import Path

import numpy as np
import nymph_rs as nr
import pytest
from nymph_rs.kernels.gdn_prefill import (
    CONFIGS,
    CONFIGS_SUPPORTED,
    GVA_RATIO,
    HEAD_CONFIGS,
    HQK,
    HV,
    VARLEN_CONFIGS,
    GdnPrefillConfig,
    build_gdn_prefill,
)

sys.path.insert(0, str(Path(__file__).parent))
import _gdn_oracle as oracle

ml_dtypes = pytest.importorskip("ml_dtypes")
D = 128


def _inputs(T, seed=0):
    rng = np.random.default_rng(seed)
    q = (rng.normal(size=(T, HQK, D)) * 0.2).astype(np.float32)
    k = (rng.normal(size=(T, HQK, D)) * 0.2).astype(np.float32)
    v = (rng.normal(size=(T, HV, D)) * 0.2).astype(np.float32)
    g = (-np.log1p(np.exp(rng.normal(size=(T, HV)))) * 0.3).astype(np.float32)  # log-gate
    beta = (1.0 / (1.0 + np.exp(-rng.normal(size=(T, HV))))).astype(np.float32)
    return q, k, v, g, beta


def _tier(configs):
    """Mark the shapes whose simulation is measured in minutes: cost tracks
    TOTAL TOKENS, not the number of shapes. Fixed-length configs carry
    num_seqs/seqlen; varlen ones carry the per-sequence list."""

    def tokens(c):
        if "seqlens" in c:
            return sum(c["seqlens"])
        return c["num_seqs"] * c["seqlen"]

    return [
        pytest.param(c, marks=pytest.mark.slow) if tokens(c) >= 2048 else c for c in configs
    ]


@pytest.mark.parametrize(
    "cfg", _tier(CONFIGS_SUPPORTED), ids=[c["label"] for c in CONFIGS_SUPPORTED]
)
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


@pytest.mark.parametrize("hc", HEAD_CONFIGS, ids=[h["label"] for h in HEAD_CONFIGS])
def test_gdn_prefill_heads(hc):
    """Head-count sweep — all 8 flashinfer num_q/k/v configs (GVA num_v≥num_q AND GQA
    num_q>num_v): protocol + value. The kernel iterates neff=max(q,v) effective heads;
    gate/beta/out/state per eh; q/k/v loaded from the shared kv head (hr per group) or the
    unique head."""
    hq, hv = hc["num_q_heads"], hc["num_v_heads"]
    hk = min(hq, hv)  # num_k_heads (derived)
    neff = max(hq, hv)
    hr = neff // hk
    gqa = hq > hv
    seqlen = 128
    cfg = GdnPrefillConfig(num_seqs=1, seqlen=seqlen, num_q_heads=hq, num_v_heads=hv)
    kernel = build_gdn_prefill(cfg)
    assert nr.check_protocol(kernel)["status"] == "Passed"
    q_g, k_g, v_g, gate_g, beta_g, out_g, state_g = kernel.args
    rng = np.random.default_rng(2)
    q = (rng.normal(size=(seqlen, hq, D)) * 0.2).astype(np.float32)
    k = (rng.normal(size=(seqlen, hk, D)) * 0.2).astype(np.float32)
    v = (rng.normal(size=(seqlen, hv, D)) * 0.2).astype(np.float32)
    g = (-np.log1p(np.exp(rng.normal(size=(seqlen, neff)))) * 0.3).astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-rng.normal(size=(seqlen, neff))))).astype(np.float32)
    bf = lambda x: x.astype(ml_dtypes.bfloat16)  # noqa: E731
    res = nr.interpret(
        kernel,
        {q_g: bf(q), k_g: bf(k), v_g: bf(v), gate_g: np.exp(g).astype(np.float32), beta_g: beta},
    )
    out = np.asarray(res[out_g.id], dtype=np.float32)
    state = np.asarray(res[state_g.id], dtype=np.float32)
    for eh in range(neff):
        qh = eh if gqa else eh // hr
        kh = eh // hr
        vh = eh // hr if gqa else eh
        o_ref, s_ref = oracle.chunked(
            q[:, qh], k[:, kh], v[:, vh], g[:, eh], beta[:, eh], cfg.scale, None, BT=64
        )
        assert (np.abs(out[:, eh] - o_ref) <= 1e-2 + 0.3 * np.abs(o_ref)).mean() >= 0.9
        assert (np.abs(state[0, eh] - s_ref) <= 1e-2 + 0.3 * np.abs(s_ref)).mean() >= 0.9


@pytest.mark.parametrize("cfg", _tier(CONFIGS), ids=[c["label"] for c in CONFIGS])
def test_gdn_prefill_build_protocol(cfg):
    """Every fixed-length workload shape (1..32 chunks, batch 1..48) builds + protocol-checks."""
    kernel = build_gdn_prefill(GdnPrefillConfig(num_seqs=cfg["num_seqs"], seqlen=cfg["seqlen"]))
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"


@pytest.mark.parametrize(
    "cfg", _tier(VARLEN_CONFIGS), ids=[c["label"] for c in VARLEN_CONFIGS]
)
def test_gdn_prefill_varlen(cfg):
    """Varlen: per-tile runtime chunk count from cu_seqlens (incl. non-BT-multiple tails)."""
    seqlens = cfg["seqlens"]
    NS = len(seqlens)
    maxlen = max(seqlens)
    cu = np.array([0, *np.cumsum(seqlens)], np.int32)
    cfg = GdnPrefillConfig(num_seqs=NS, seqlen=maxlen, varlen=True)
    kernel = build_gdn_prefill(cfg)
    q_g, k_g, v_g, gate_g, beta_g, out_g, state_g, cu_g = kernel.args
    total_t = NS * maxlen
    q, k, v, g, beta = _inputs(total_t, seed=1)
    bf = lambda x: x.astype(ml_dtypes.bfloat16)  # noqa: E731
    inputs = {
        q_g: bf(q),
        k_g: bf(k),
        v_g: bf(v),
        gate_g: np.exp(g).astype(np.float32),
        beta_g: beta,
        cu_g: cu,
    }
    # protocol must be checked WITH the concrete cu_seqlens — varlen's per-tile chunk
    # count is data-dependent, so a no-input check is Inconclusive; with cu_seqlens the
    # checker resolves the schedule and verifies the barrier protocol (incl. multi-tile).
    pr = nr.check_protocol(kernel, inputs)
    assert pr["status"] == "Passed", pr["diagnostics"][:2]
    res = nr.interpret(kernel, inputs)
    out = np.asarray(res[out_g.id], dtype=np.float32)
    state = np.asarray(res[state_g.id], dtype=np.float32)
    for s in range(NS):
        a, b = int(cu[s]), int(cu[s + 1])
        for hv in range(HV):
            hqk = hv // GVA_RATIO
            o_ref, s_ref = oracle.chunked(
                q[a:b, hqk, :], k[a:b, hqk, :], v[a:b, hv, :],
                g[a:b, hv], beta[a:b, hv], cfg.scale, None, BT=64,
            )  # fmt: skip
            m_o = (np.abs(out[a:b, hv, :] - o_ref) <= 1e-2 + 0.3 * np.abs(o_ref)).mean()
            m_s = (np.abs(state[s, hv] - s_ref) <= 1e-2 + 0.3 * np.abs(s_ref)).mean()
            assert m_o >= 0.9, f"seq={s} hv={hv} output matched={m_o}"
            assert m_s >= 0.9, f"seq={s} hv={hv} state matched={m_s}"


@pytest.mark.parametrize("io_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize(
    "cfg", _tier(CONFIGS_SUPPORTED), ids=[c["label"] for c in CONFIGS_SUPPORTED]
)
def test_gdn_prefill_value(cfg, io_dtype):
    num_seqs, seqlen = cfg["num_seqs"], cfg["seqlen"]
    cfg = GdnPrefillConfig(num_seqs=num_seqs, seqlen=seqlen, io_dtype=io_dtype)
    kernel = build_gdn_prefill(cfg)
    q_g, k_g, v_g, gate_g, beta_g, out_g, state_g = kernel.args
    total_t = num_seqs * seqlen
    q, k, v, g, beta = _inputs(total_t)
    npdt = {"bfloat16": ml_dtypes.bfloat16, "float16": np.float16}[io_dtype]
    bf = lambda x: x.astype(npdt)  # noqa: E731 — io operand dtype (f16 exercises mma.sync f16)
    # The kernel's gate warp applies log2 to the RAW gate (flashinfer parity), so feed
    # a = exp(g) — log2(a)=g·log2(e), cumsum → exp2(Δ)=exp(Δ over g). Oracle uses g.
    raw_gate = np.exp(g).astype(np.float32)
    # state_g is output-only (S=0, no initial state read); omit it from inputs.
    res = nr.interpret(kernel, {q_g: bf(q), k_g: bf(k), v_g: bf(v), gate_g: raw_gate, beta_g: beta})
    out = np.asarray(res[out_g.id], dtype=np.float32)
    state = np.asarray(res[state_g.id], dtype=np.float32)  # (num_seqs, HV, D, D)
    for s in range(num_seqs):
        sl = slice(s * seqlen, (s + 1) * seqlen)
        for hv in range(HV):
            hqk = hv // GVA_RATIO
            o_ref, s_ref = oracle.chunked(
                q[sl, hqk, :], k[sl, hqk, :], v[sl, hv, :],
                g[sl, hv], beta[sl, hv], cfg.scale, None, BT=64,
            )  # fmt: skip
            m_o = (np.abs(out[sl, hv, :] - o_ref) <= 1e-2 + 0.3 * np.abs(o_ref)).mean()
            m_s = (np.abs(state[s, hv] - s_ref) <= 1e-2 + 0.3 * np.abs(s_ref)).mean()
            assert m_o >= 0.9, f"seq={s} hv={hv} output matched={m_o}"
            assert m_s >= 0.9, f"seq={s} hv={hv} state matched={m_s}"
