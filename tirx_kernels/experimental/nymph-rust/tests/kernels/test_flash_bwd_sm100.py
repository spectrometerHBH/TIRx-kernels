"""flash_bwd_sm100 (FA SM100 backward) — trace + check_protocol + value-sim vs the oracle.

Conventions (matching flashattn FlashAttentionBackwardSm100):
  - LSE input is log2-scaled (lse·log2e): the kernel computes exp2(S·scale_log2 − LSE).
  - dQ output (dq_g) is the UNSCALED dQaccum — flashattn applies softmax_scale in the
    postprocess kernel; the test multiplies by scale before comparing to the oracle.
  - dPsum input = rowsum(dO ∘ O) (the preprocess output).
"""

import numpy as np
import nymph_rs as nr
import pytest

ml_dtypes = pytest.importorskip("ml_dtypes")
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _fa_bwd_oracle as oracle  # noqa: E402
from nymph_rs.kernels.flash_bwd_sm100 import (  # noqa: E402
    LOG2_E,
    FlashBwdSm100Config,
    build_flash_bwd_sm100,
)

bf = lambda x: x.astype(ml_dtypes.bfloat16)  # noqa: E731

# milestone-1: single m-block / n-block (Sq=Sk=128), hd64, 1-CTA. Multi-block configs
# (the dV/dK cross-m-block accumulation) land as the kernel grows.
SINGLE_TILE = [
    {"batch": 1, "num_head": 1, "head_dim": 64, "label": "b1h1_hd64"},
    {"batch": 1, "num_head": 2, "head_dim": 64, "label": "b1h2_hd64"},
    {"batch": 2, "num_head": 2, "head_dim": 64, "label": "b2h2_hd64"},
    # hd96 1-CTA (tile_hdim=96): the t2r reads split into legal 32x32b .x{N} pieces
    # (48 -> .x32+.x16, 96 -> .x64+.x32) — _tc_ld in flash_bwd_sm100.py.
    {"batch": 1, "num_head": 1, "head_dim": 96, "label": "b1h1_hd96"},
    {"batch": 1, "num_head": 2, "head_dim": 96, "label": "b1h2_hd96"},
]


def _run(cfg_d, S=128):
    B, H, D = cfg_d["batch"], cfg_d["num_head"], cfg_d["head_dim"]
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0)
    Tq = B * S
    q = (rng.normal(size=(Tq, H, D)) * 0.3).astype(np.float32)
    k = (rng.normal(size=(Tq, H, D)) * 0.3).astype(np.float32)
    v = (rng.normal(size=(Tq, H, D)) * 0.3).astype(np.float32)
    do = (rng.normal(size=(Tq, H, D)) * 0.3).astype(np.float32)
    lse = np.zeros((Tq, H), np.float32)
    dpsum = np.zeros((Tq, H), np.float32)
    dq_ref = np.zeros((Tq, H, D), np.float32)
    dk_ref = np.zeros((Tq, H, D), np.float32)
    dv_ref = np.zeros((Tq, H, D), np.float32)
    for b in range(B):
        for h in range(H):
            sl = slice(b * S, (b + 1) * S)
            dq, dk, dv, o, l = oracle.fa_bwd_from_qkv(
                q[sl, h], k[sl, h], v[sl, h], do[sl, h], scale
            )
            lse[sl, h] = l * LOG2_E
            dpsum[sl, h] = (do[sl, h] * o).sum(-1)
            dq_ref[sl, h] = dq
            dk_ref[sl, h] = dk
            dv_ref[sl, h] = dv
    cfg = FlashBwdSm100Config(batch=B, num_head=H, seqlen_q=S, seqlen_k=S, head_dim=D)
    kernel = build_flash_bwd_sm100(cfg)
    qg, kg, vg, dog, lseg, dpsg, dqg, dkg, dvg = kernel.args
    return (
        kernel,
        scale,
        dict(
            inputs={qg: bf(q), kg: bf(k), vg: bf(v), dog: bf(do), lseg: lse, dpsg: dpsum},
            refs=(dq_ref, dk_ref, dv_ref),
            ids=(dqg.id, dkg.id, dvg.id),
            shape=(Tq, H, D),
        ),
    )


@pytest.mark.parametrize("cfg", SINGLE_TILE, ids=[c["label"] for c in SINGLE_TILE])
def test_flash_bwd_sm100_trace_protocol(cfg):
    kernel, _, _ = _run(cfg)
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"


# hd128 2-CTA: full cluster datapath — S/dP/dV/dK cluster GEMMs (no cross-CTA dep) PLUS
# the dS cross-CTA exchange + relay + dQ GEMM + dQ reduce. All of dQ/dK/dV are now
# VALUE-CORRECT. Each CTA loads its OWN half of every operand (A=its kv-tile, B=its N/2)
# into the same SMEM offset, so the interpreter's read_operand_ctas concat over the pair
# reconstructs the full operand. CTA c owns kv-tile c (rows [c*128,(c+1)*128) of Sk=256
# for dK/dV). For dQ the kv (K) is cluster-wide: each CTA s2clusters its dS half into the
# peer's sdS so both CTAs hold the full 256-kv dS, then the cta_group::2 m=128 (Layout B)
# dQ MMA splits q across the pair (CTA c -> q-rows [c*64,(c+1)*64)). No interpreter change.
CONFIGS_2CTA = [
    {
        "batch": 1,
        "num_head": 1,
        "seqlen_q": 128,
        "seqlen_k": 256,
        "head_dim": 128,
        "label": "2cta_b1h1_sq128sk256_hd128",
    }
]


def _run_2cta(cfg_d):
    """Build inputs over the full Sk via the oracle; the 2 CTAs of a pair produce the
    256-kv dK/dV (CTA c = kv-tile c) and the full dQ for the Sq=128 Q-tile (sum over all
    256 kv). Returns (kernel, info) for the dQ/dK/dV value assertion."""
    B, H, D = cfg_d["batch"], cfg_d["num_head"], cfg_d["head_dim"]
    Sq, Sk = cfg_d["seqlen_q"], cfg_d["seqlen_k"]
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0)
    q = (rng.normal(size=(B * Sq, H, D)) * 0.3).astype(np.float32)
    k = (rng.normal(size=(B * Sk, H, D)) * 0.3).astype(np.float32)
    v = (rng.normal(size=(B * Sk, H, D)) * 0.3).astype(np.float32)
    do = (rng.normal(size=(B * Sq, H, D)) * 0.3).astype(np.float32)
    lse = np.zeros((B * Sq, H), np.float32)
    dpsum = np.zeros((B * Sq, H), np.float32)
    dq_ref = np.zeros((B * Sq, H, D), np.float32)
    dk_ref = np.zeros((B * Sk, H, D), np.float32)
    dv_ref = np.zeros((B * Sk, H, D), np.float32)
    for b in range(B):
        for h in range(H):
            qs, ks = slice(b * Sq, (b + 1) * Sq), slice(b * Sk, (b + 1) * Sk)
            dq, dk, dv, o, l = oracle.fa_bwd_from_qkv(
                q[qs, h], k[ks, h], v[ks, h], do[qs, h], scale
            )
            lse[qs, h] = l * LOG2_E
            dpsum[qs, h] = (do[qs, h] * o).sum(-1)
            dq_ref[qs, h] = dq
            dk_ref[ks, h] = dk
            dv_ref[ks, h] = dv
    config = FlashBwdSm100Config(
        batch=B, num_head=H, seqlen_q=Sq, seqlen_k=Sk, head_dim=D, scale=scale
    )
    kernel = build_flash_bwd_sm100(config)
    qg, kg, vg, dog, lseg, dpsg, dqg, dkg, dvg = kernel.args
    return (
        kernel,
        scale,
        dict(
            inputs={qg: bf(q), kg: bf(k), vg: bf(v), dog: bf(do), lseg: lse, dpsg: dpsum},
            dq_ref=dq_ref,
            dq_id=dqg.id,
            dq_shape=(B * Sq, H, D),
            refs=(dk_ref, dv_ref),
            ids=(dkg.id, dvg.id),
            shape=(B * Sk, H, D),
        ),
    )


@pytest.mark.parametrize("cfg", CONFIGS_2CTA, ids=[c["label"] for c in CONFIGS_2CTA])
def test_flash_bwd_sm100_2cta_trace_protocol(cfg):
    kernel, _, _ = _run_2cta(cfg)
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"


@pytest.mark.parametrize("cfg", CONFIGS_2CTA, ids=[c["label"] for c in CONFIGS_2CTA])
def test_flash_bwd_sm100_2cta_value_dq_dk_dv(cfg):
    # dQ, dK, dV are ALL value-correct for the hd128 2-CTA cluster. dQ exercises the dS
    # cross-CTA exchange (s2cluster) + relay + cta_group::2 m=128 (Layout B) dQ GEMM +
    # per-CTA dQ reduce; each CTA owns q-rows [c*64,(c+1)*64) of the Sq=128 Q-tile.
    kernel, scale, info = _run_2cta(cfg)
    res = nr.interpret(kernel, info["inputs"])
    dk_id, dv_id = info["ids"]
    dk_ref, dv_ref = info["refs"]
    sh = info["shape"]
    dq = (
        np.asarray(res[info["dq_id"]], np.float32).reshape(info["dq_shape"]) * scale
    )  # dQaccum -> dQ
    dk = np.asarray(res[dk_id], np.float32).reshape(sh)
    dv = np.asarray(res[dv_id], np.float32).reshape(sh)
    for name, g, gr in (("dQ", dq, info["dq_ref"]), ("dK", dk, dk_ref), ("dV", dv, dv_ref)):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.03, f"{name} rel-err {rel}"


# 2-CTA MULTI-Q-TILE (n_mb>1, seqlen_q>128). hd128 cluster with Sq spanning 2/3/4 Q-tiles.
# The 2-CTA operand buffers are SINGLE-stage (flashattn Q_stage=1/dO_stage=1), so each of the
# four SEPARATE B-operands (sQ@gS, sdO@gdV, sQt@gdK, sdOt@gdP) needs its own leader-multicast
# empty/free barrier (q_free/do_free/qt_free/dot_free) and sKt (dQ B = K) is loaded ONCE per
# task (hoisted). The dQ-overlaps-S TMEM WAR is gated by gS's dq_free[mb-2] + gdQ's dq_free[mb-1]
# + s_cons (compute-consumed-S); the single-buffer sdS_full / sdS_xchg dS-exchange WARs are gated
# by dS_free (leader consumer_release) + the s2cluster source drain. All leader-multicast so both
# CTAs' producers see the release. (hd192 multi-Q-tile is WIP -> asserted n_mb=1.)
CONFIGS_2CTA_NMB = [
    {
        "batch": 1,
        "num_head": 2,
        "seqlen_q": 256,
        "seqlen_k": 256,
        "head_dim": 128,
        "label": "2cta_nmb2_sq256sk256",
    },
    {
        "batch": 1,
        "num_head": 2,
        "seqlen_q": 256,
        "seqlen_k": 512,
        "head_dim": 128,
        "label": "2cta_nmb2_sq256sk512",
    },
    {
        "batch": 1,
        "num_head": 2,
        "seqlen_q": 384,
        "seqlen_k": 256,
        "head_dim": 128,
        "label": "2cta_nmb3_sq384sk256",
    },
    {
        "batch": 1,
        "num_head": 2,
        "seqlen_q": 512,
        "seqlen_k": 512,
        "head_dim": 128,
        "label": "2cta_nmb4_sq512sk512",
    },
]


@pytest.mark.parametrize("cfg", CONFIGS_2CTA_NMB, ids=[c["label"] for c in CONFIGS_2CTA_NMB])
def test_flash_bwd_sm100_2cta_nmb_protocol_value(cfg):
    # 2-CTA multi-Q-tile: trace + check_protocol Passed + dQ/dK/dV value rel<0.02.
    kernel, scale, info = _run_2cta(cfg)
    assert nr.trace(kernel)["status"] == "Passed"
    assert nr.check_protocol(kernel)["status"] == "Passed"
    res = nr.interpret(kernel, info["inputs"])
    dk_id, dv_id = info["ids"]
    dk_ref, dv_ref = info["refs"]
    sh = info["shape"]
    dq = np.asarray(res[info["dq_id"]], np.float32).reshape(info["dq_shape"]) * scale
    dk = np.asarray(res[dk_id], np.float32).reshape(sh)
    dv = np.asarray(res[dv_id], np.float32).reshape(sh)
    for name, g, gr in (("dQ", dq, info["dq_ref"]), ("dK", dk, dk_ref), ("dV", dv, dv_ref)):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.02, f"{name} rel-err {rel}"


# 2-CTA DENSE ODD #kv-tiles (Sk a multiple of TILE_N=128 but NOT of cg*TILE_N=256). Previously
# `assert n_nb % cg == 0` rejected these tile-aligned-but-not-cluster-aligned shapes; now the grid
# rounds n_nb UP to a cg multiple and the last cluster's PADDING tile (kv past Sk) runs the cluster
# cta_group::2 MMA/compute/exchange in lockstep with its P forced to 0 (tile mask nb>=real_n_nb),
# so it contributes 0 to dS/dV/dK/dQ; the per-CTA dK/dV store + the per-row dQ reduce write only
# valid rows. The batch=2 entries are the K-OVERRUN regression: the padding tile's K load reads
# coords (batch+1)*Sk = the NEXT batch's K (not zero), so a missing dS=0 mask corrupts dQ (rel~0.48
# without the mask). All produce dQ/dK/dV rel<0.02. (Sk=896 is 7 tiles = 3 full + 1 odd cluster.)
CONFIGS_2CTA_ODD_NNB = [
    # (B, Hq, Hkv, Sq, Sk, Dq, Dv, label)
    (1, 1, 1, 128, 384, 128, 128, "odd_mha_hd128_sq128sk384"),
    (1, 1, 1, 256, 384, 128, 128, "odd_mha_hd128_sq256sk384"),
    (1, 1, 1, 128, 128, 128, 128, "odd_mha_hd128_sq128sk128"),  # 1 tile -> 1 cluster w/ pad
    (1, 1, 1, 384, 640, 128, 128, "odd_mha_hd128_sq384sk640"),  # Sk=640 = 5 tiles
    (1, 1, 1, 256, 896, 128, 128, "odd_mha_hd128_sq256sk896"),  # Sk=896 = 7 tiles
    (1, 4, 2, 128, 384, 128, 128, "odd_gqa_h4kv2_hd128_sq128sk384"),
    (1, 4, 2, 256, 384, 128, 128, "odd_gqa_h4kv2_hd128_sq256sk384"),
    (1, 1, 1, 256, 384, 192, 128, "odd_mha_hd192_sq256sk384"),
    (2, 1, 1, 128, 384, 128, 128, "odd_b2_mha_hd128_sq128sk384"),  # K-overrun (dQ corruption probe)
    (2, 4, 2, 128, 384, 128, 128, "odd_b2_gqa_h4kv2_hd128_sq128sk384"),  # K-overrun GQA
]


def _run_odd_nnb(B, Hq, Hkv, Sq, Sk, Dq, Dv):
    """Packed (B*Sq, B*Sk) buffers; per-(batch,head) oracle slice (kv-head = h//G). MHA dk_g is
    pre-scaled in-epilogue (no ×scale); GQA dk_g is the unscaled f32 reduce accumulator (×scale);
    dv_g direct; dq_g is the unscaled dQaccum (×scale)."""
    G = Hq // Hkv
    scale = 1.0 / np.sqrt(Dq)
    rng = np.random.default_rng(0)
    q = (rng.normal(size=(B * Sq, Hq, Dq)) * 0.3).astype(np.float32)
    kk = (rng.normal(size=(B * Sk, Hkv, Dq)) * 0.3).astype(np.float32)
    vv = (rng.normal(size=(B * Sk, Hkv, Dv)) * 0.3).astype(np.float32)
    do = (rng.normal(size=(B * Sq, Hq, Dv)) * 0.3).astype(np.float32)
    lse = np.zeros((B * Sq, Hq), np.float32)
    dpsum = np.zeros((B * Sq, Hq), np.float32)
    dq_ref = np.zeros((B * Sq, Hq, Dq), np.float32)
    dk_ref = np.zeros((B * Sk, Hkv, Dq), np.float32)
    dv_ref = np.zeros((B * Sk, Hkv, Dv), np.float32)
    for b in range(B):
        for h in range(Hq):
            kv = h // G
            qs, ks = slice(b * Sq, (b + 1) * Sq), slice(b * Sk, (b + 1) * Sk)
            dq, dk, dv, o, l = oracle.fa_bwd_from_qkv(
                q[qs, h], kk[ks, kv], vv[ks, kv], do[qs, h], scale
            )
            lse[qs, h] = l * LOG2_E
            dpsum[qs, h] = (do[qs, h] * o).sum(-1)
            dq_ref[qs, h] = dq
            if G > 1:  # GQA: the G q-heads of a group sum into the shared kv-head's rows
                dk_ref[ks, kv] += dk
                dv_ref[ks, kv] += dv
            else:
                dk_ref[ks, kv] = dk
                dv_ref[ks, kv] = dv
    cfg = FlashBwdSm100Config(
        batch=B,
        num_head=Hq,
        num_head_kv=Hkv,
        seqlen_q=Sq,
        seqlen_k=Sk,
        head_dim=Dq,
        head_dim_v=Dv,
        scale=scale,
    )
    kernel = build_flash_bwd_sm100(cfg)
    qg, kg, vg, dog, lseg, dpsg, dqg, dkg, dvg = kernel.args
    dks = scale if G > 1 else 1.0  # GQA dKaccum unscaled (×scale); MHA dK pre-scaled in epilogue
    return (
        kernel,
        scale,
        dict(
            inputs={qg: bf(q), kg: bf(kk), vg: bf(vv), dog: bf(do), lseg: lse, dpsg: dpsum},
            dq_ref=dq_ref,
            dq_id=dqg.id,
            dq_shape=(B * Sq, Hq, Dq),
            dk_ref=dk_ref,
            dk_id=dkg.id,
            dk_shape=(B * Sk, Hkv, Dq),
            dks=dks,
            dv_ref=dv_ref,
            dv_id=dvg.id,
            dv_shape=(B * Sk, Hkv, Dv),
        ),
    )


@pytest.mark.parametrize("cfg", CONFIGS_2CTA_ODD_NNB, ids=[c[-1] for c in CONFIGS_2CTA_ODD_NNB])
def test_flash_bwd_sm100_2cta_odd_nnb(cfg):
    # Dense 2-CTA, odd #kv-tiles (the removed `n_nb % cg == 0` restriction). trace Passed + value.
    # check_protocol is asserted Passed only for single-cluster-orderable shapes (n_cluster==1);
    # multi-cluster odd Sk inherits the SAME pre-existing cross-CTA reduce-add Inconclusive class as
    # any n_nb>1 dense shape (unrelated to the padding tile) — so value is the gate there.
    B, Hq, Hkv, Sq, Sk, Dq, Dv, _ = cfg
    kernel, scale, info = _run_odd_nnb(B, Hq, Hkv, Sq, Sk, Dq, Dv)
    assert nr.trace(kernel)["status"] == "Passed"
    # Protocol Passed for single-cluster Sk (n_cluster==1): no cross-CTA dQ reduce-add overlap.
    # Multi-cluster (n_cluster>1) inherits the pre-existing Inconclusive class of any dense n_nb>1
    # shape (the n_nb KV-tile CTAs reduce-add into the same dQ rows — unrelated to the padding tile).
    real_n_nb = (Sk + 127) // 128
    n_cluster = (real_n_nb + 1) // 2  # rounded-up cluster count
    if n_cluster == 1:
        assert nr.check_protocol(kernel)["status"] == "Passed"
    res = nr.interpret(kernel, info["inputs"])
    dq = np.asarray(res[info["dq_id"]], np.float32).reshape(info["dq_shape"]) * scale
    dk = np.asarray(res[info["dk_id"]], np.float32).reshape(info["dk_shape"]) * info["dks"]
    dv = np.asarray(res[info["dv_id"]], np.float32).reshape(info["dv_shape"])
    for name, g, gr in (
        ("dQ", dq, info["dq_ref"]),
        ("dK", dk, info["dk_ref"]),
        ("dV", dv, info["dv_ref"]),
    ):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.02, f"{name} rel-err {rel}"


# GQA × 2-CTA multi-Q-tile (n_mb=2). dK/dV f32 reduce-add over the q-heads of a group; the
# multi-Q-tile operand single-staging + dS-exchange WARs interact with the GQA epilogue.
def test_flash_bwd_sm100_gqa_2cta_nmb():
    kernel, scale, info = _run_gqa(
        {
            "num_head": 4,
            "num_head_kv": 2,
            "head_dim": 128,
            "seqlen_q": 256,
            "seqlen_k": 256,
            "label": "gqa2cta_nmb2",
        }
    )
    assert nr.trace(kernel)["status"] == "Passed"
    assert nr.check_protocol(kernel)["status"] == "Passed"
    res = nr.interpret(kernel, info["inputs"])
    dq = np.asarray(res[info["dq_id"]], np.float32).reshape(info["q_shape"]) * scale
    dk = (
        np.asarray(res[info["dk_id"]], np.float32).reshape(info["kv_shape"]) * scale
    )  # GQA dK unscaled
    dv = np.asarray(res[info["dv_id"]], np.float32).reshape(info["dv_shape"])
    for name, g, gr in (
        ("dQ", dq, info["dq_ref"]),
        ("dK", dk, info["dk_ref"]),
        ("dV", dv, info["dv_ref"]),
    ):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.02, f"{name} rel-err {rel}"


# hd192 DeepSeek (head_dim=192, head_dim_v=128) — 2-CTA, the dedicated overlapping-TMEM
# layout (dV=0 dK=128 S/P=320 dP/dS=384 dQ=416) with the straight S->dP->dK->dV->dQ MMA
# order + s_free (S consumed) / dq_free (dQ consumed) overlap barriers. Asymmetric:
# q/k/dq/dk are hd192, v/do/dv are hd128.
CONFIGS_HD192 = [
    {"batch": 1, "num_head": 1, "seqlen_q": 128, "seqlen_k": 256, "label": "hd192_b1h1_sq128sk256"},
    {"batch": 1, "num_head": 2, "seqlen_q": 128, "seqlen_k": 256, "label": "hd192_b1h2_sq128sk256"},
]


def _run_hd192(cfg_d, Dq=192, Dv=128):
    B, H = cfg_d["batch"], cfg_d["num_head"]
    Sq, Sk = cfg_d["seqlen_q"], cfg_d["seqlen_k"]
    scale = 1.0 / np.sqrt(Dq)
    rng = np.random.default_rng(0)
    q = (rng.normal(size=(B * Sq, H, Dq)) * 0.3).astype(np.float32)
    k = (rng.normal(size=(B * Sk, H, Dq)) * 0.3).astype(np.float32)
    v = (rng.normal(size=(B * Sk, H, Dv)) * 0.3).astype(np.float32)
    do = (rng.normal(size=(B * Sq, H, Dv)) * 0.3).astype(np.float32)
    lse = np.zeros((B * Sq, H), np.float32)
    dpsum = np.zeros((B * Sq, H), np.float32)
    dq_ref = np.zeros((B * Sq, H, Dq), np.float32)
    dk_ref = np.zeros((B * Sk, H, Dq), np.float32)
    dv_ref = np.zeros((B * Sk, H, Dv), np.float32)
    for b in range(B):
        for h in range(H):
            qs, ks = slice(b * Sq, (b + 1) * Sq), slice(b * Sk, (b + 1) * Sk)
            dq, dk, dv, o, l = oracle.fa_bwd_from_qkv(
                q[qs, h], k[ks, h], v[ks, h], do[qs, h], scale
            )
            lse[qs, h] = l * LOG2_E
            dpsum[qs, h] = (do[qs, h] * o).sum(-1)
            dq_ref[qs, h] = dq
            dk_ref[ks, h] = dk
            dv_ref[ks, h] = dv
    config = FlashBwdSm100Config(
        batch=B, num_head=H, seqlen_q=Sq, seqlen_k=Sk, head_dim=Dq, head_dim_v=Dv, scale=scale
    )
    kernel = build_flash_bwd_sm100(config)
    qg, kg, vg, dog, lseg, dpsg, dqg, dkg, dvg = kernel.args
    return (
        kernel,
        scale,
        dict(
            inputs={qg: bf(q), kg: bf(k), vg: bf(v), dog: bf(do), lseg: lse, dpsg: dpsum},
            dq_ref=dq_ref,
            dq_id=dqg.id,
            dq_shape=(B * Sq, H, Dq),
            dk_ref=dk_ref,
            dk_id=dkg.id,
            dk_shape=(B * Sk, H, Dq),
            dv_ref=dv_ref,
            dv_id=dvg.id,
            dv_shape=(B * Sk, H, Dv),
        ),
    )


@pytest.mark.parametrize("cfg", CONFIGS_HD192, ids=[c["label"] for c in CONFIGS_HD192])
def test_flash_bwd_sm100_hd192_trace_protocol(cfg):
    kernel, _, _ = _run_hd192(cfg)
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"


@pytest.mark.parametrize("cfg", CONFIGS_HD192, ids=[c["label"] for c in CONFIGS_HD192])
def test_flash_bwd_sm100_hd192_value_dq_dk_dv(cfg):
    # The hd192 overlapping-TMEM choreography is value-correct for dQ (hd192), dK (hd192),
    # dV (hd128). dQ exercises the per-CTA DQ_COL=96 reduce; dK reads TMEM dS before P
    # overwrites it (dK-before-dV order); dV reads P.
    kernel, scale, info = _run_hd192(cfg)
    res = nr.interpret(kernel, info["inputs"])
    dq = np.asarray(res[info["dq_id"]], np.float32).reshape(info["dq_shape"]) * scale
    dk = np.asarray(res[info["dk_id"]], np.float32).reshape(info["dk_shape"])
    dv = np.asarray(res[info["dv_id"]], np.float32).reshape(info["dv_shape"])
    for name, g, gr in (
        ("dQ", dq, info["dq_ref"]),
        ("dK", dk, info["dk_ref"]),
        ("dV", dv, info["dv_ref"]),
    ):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.03, f"{name} rel-err {rel}"


# hd192 MULTI-Q-TILE (n_mb>1, seqlen_q>128). The DeepSeek dedicated overlapping-TMEM layout with
# the STRAIGHT per-mb MMA order (S->dP->dK->dV->dQ, no skew). The single-stage operands are reused
# across mb, serialized by the time-mux + overlap barriers:
#   - sQ is TIME-MUXED Q->dOt->Qt within one mb (sdOt/sQt alias sQ). The next mb's Q reload waits
#     q_free, which hd192 arrives at gdK (sQ's LAST user = sQt reader), NOT at gS.
#   - dQ[416,512) overlaps S[320,448): gS(mb+1) clobbers tS over dQ(mb)'s overlap -> gS waits the
#     reduce's dq_free[mb] (local + peer). The reverse (gdQ(mb) over S(mb)) is ordered by gdQ's
#     dS_cluster_leader wait (compute produced dS only after reading S), so hd192 needs no s_cons.
#   - sdS_full (dQ A) single-buffer -> compute(mb) waits dS_free (gdQ(mb-1) consumer_release);
#     sdS_xchg aliases sdQ -> compute(mb) waits dQaccum_empty (reduce drained sdQ(mb-1)).
# Sk multiple of 256 (full clusters); n_mb in {2,3,4}.
CONFIGS_HD192_NMB = [
    {"batch": 1, "num_head": 1, "seqlen_q": 256, "seqlen_k": 256, "label": "hd192_nmb2_sq256sk256"},
    {"batch": 1, "num_head": 1, "seqlen_q": 384, "seqlen_k": 256, "label": "hd192_nmb3_sq384sk256"},
    {"batch": 1, "num_head": 1, "seqlen_q": 512, "seqlen_k": 256, "label": "hd192_nmb4_sq512sk256"},
    {"batch": 1, "num_head": 1, "seqlen_q": 256, "seqlen_k": 512, "label": "hd192_nmb2_sq256sk512"},
    {
        "batch": 1,
        "num_head": 2,
        "seqlen_q": 384,
        "seqlen_k": 512,
        "label": "hd192_nmb3_b1h2_sq384sk512",
    },
]


@pytest.mark.parametrize("cfg", CONFIGS_HD192_NMB, ids=[c["label"] for c in CONFIGS_HD192_NMB])
def test_flash_bwd_sm100_hd192_nmb_protocol_value(cfg):
    # hd192 multi-Q-tile: trace + check_protocol Passed (single-cluster Sk -> orderable) + value.
    kernel, scale, info = _run_hd192(cfg)
    assert nr.trace(kernel)["status"] == "Passed"
    assert nr.check_protocol(kernel)["status"] == "Passed"
    res = nr.interpret(kernel, info["inputs"])
    dq = np.asarray(res[info["dq_id"]], np.float32).reshape(info["dq_shape"]) * scale
    dk = np.asarray(res[info["dk_id"]], np.float32).reshape(info["dk_shape"])
    dv = np.asarray(res[info["dv_id"]], np.float32).reshape(info["dv_shape"])
    for name, g, gr in (
        ("dQ", dq, info["dq_ref"]),
        ("dK", dk, info["dk_ref"]),
        ("dV", dv, info["dv_ref"]),
    ):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.02, f"{name} rel-err {rel}"


# GQA × hd192 multi-Q-tile (n_mb>1): the dK/dV f32 group reduce-add interacts with the single-stage
# time-mux + dedicated-overlap WARs across mb. dQ/dK ×scale (dQaccum / GQA-unscaled dK); dV direct.
GQA_HD192_NMB = [
    {
        "num_head": 4,
        "num_head_kv": 2,
        "head_dim": 192,
        "head_dim_v": 128,
        "seqlen_q": 256,
        "seqlen_k": 256,
        "label": "gqa_hd192_nmb2_h4kv2_sq256sk256",
    },
    {
        "num_head": 4,
        "num_head_kv": 2,
        "head_dim": 192,
        "head_dim_v": 128,
        "seqlen_q": 384,
        "seqlen_k": 256,
        "label": "gqa_hd192_nmb3_h4kv2_sq384sk256",
    },
]


@pytest.mark.parametrize("cfg", GQA_HD192_NMB, ids=[c["label"] for c in GQA_HD192_NMB])
def test_flash_bwd_sm100_gqa_hd192_nmb(cfg):
    # GQA × hd192 multi-Q-tile: trace + protocol Passed (commutative-reduce) + dQ/dK/dV value.
    kernel, scale, info = _run_gqa(cfg)
    assert nr.trace(kernel)["status"] == "Passed"
    assert nr.check_protocol(kernel)["status"] == "Passed"
    res = nr.interpret(kernel, info["inputs"])
    dq = np.asarray(res[info["dq_id"]], np.float32).reshape(info["q_shape"]) * scale
    dk = (
        np.asarray(res[info["dk_id"]], np.float32).reshape(info["kv_shape"]) * scale
    )  # GQA dK unscaled
    dv = np.asarray(res[info["dv_id"]], np.float32).reshape(info["dv_shape"])
    for name, g, gr in (
        ("dQ", dq, info["dq_ref"]),
        ("dK", dk, info["dk_ref"]),
        ("dV", dv, info["dv_ref"]),
    ):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.02, f"{name} rel-err {rel}"


# ============================ LARGE-SHAPE coverage ============================
# Exercises many tiles / tasks at scale. The cross-CTA dQ reduce-add over n_nb>1 KV-tiles is the
# commutative-add the checker can't order → protocol is Inconclusive there UNLESS deterministic;
# value is correct in every case. So large-shape protocol asserts split by orderability:
#   - LARGE_PROTO (n_nb==1, or single-cluster 2-CTA, or batch×heads): protocol Passed + value.
#   - LARGE_VALUE (square n_nb>1, non-deterministic): value only (protocol Inconclusive, expected).
#   - deterministic large Sk: protocol Passed (the semaphore serializes the n_nb reduce-adds).
def _run_big(B, H, Dq, Dv, Sq, Sk, causal=False, deterministic=False):
    scale = 1.0 / np.sqrt(Dq)
    rng = np.random.default_rng(0)
    q = (rng.normal(size=(B * Sq, H, Dq)) * 0.3).astype(np.float32)
    k = (rng.normal(size=(B * Sk, H, Dq)) * 0.3).astype(np.float32)
    v = (rng.normal(size=(B * Sk, H, Dv)) * 0.3).astype(np.float32)
    do = (rng.normal(size=(B * Sq, H, Dv)) * 0.3).astype(np.float32)
    lse = np.zeros((B * Sq, H), np.float32)
    dpsum = np.zeros((B * Sq, H), np.float32)
    dq_ref = np.zeros((B * Sq, H, Dq), np.float32)
    dk_ref = np.zeros((B * Sk, H, Dq), np.float32)
    dv_ref = np.zeros((B * Sk, H, Dv), np.float32)
    for b in range(B):
        for h in range(H):
            qs, ks = slice(b * Sq, (b + 1) * Sq), slice(b * Sk, (b + 1) * Sk)
            dq, dk, dv, o, l = oracle.fa_bwd_from_qkv(
                q[qs, h], k[ks, h], v[ks, h], do[qs, h], scale, causal=causal
            )
            lse[qs, h] = l * LOG2_E
            dpsum[qs, h] = (do[qs, h] * o).sum(-1)
            dq_ref[qs, h] = dq
            dk_ref[ks, h] = dk
            dv_ref[ks, h] = dv
    cfg = FlashBwdSm100Config(
        batch=B,
        num_head=H,
        seqlen_q=Sq,
        seqlen_k=Sk,
        head_dim=Dq,
        head_dim_v=Dv,
        is_causal=causal,
        deterministic=deterministic,
        scale=scale,
    )
    kernel = build_flash_bwd_sm100(cfg)
    qg, kg, vg, dog, lseg, dpsg, dqg, dkg, dvg = kernel.args[:9]  # det adds trailing sem args
    return (
        kernel,
        scale,
        dict(
            inputs={qg: bf(q), kg: bf(k), vg: bf(v), dog: bf(do), lseg: lse, dpsg: dpsum},
            refs=(
                (dqg.id, dq_ref, (B * Sq, H, Dq)),
                (dkg.id, dk_ref, (B * Sk, H, Dq)),
                (dvg.id, dv_ref, (B * Sk, H, Dv)),
            ),
        ),
    )


def _assert_big_value(kernel, scale, info):
    res = nr.interpret(kernel, info["inputs"])
    # dQ output is the unscaled dQaccum (×scale to compare); dK/dV are direct.
    for name, (gid, gr, sh), sc in (
        ("dQ", info["refs"][0], scale),
        ("dK", info["refs"][1], 1.0),
        ("dV", info["refs"][2], 1.0),
    ):
        g = np.asarray(res[gid], np.float32).reshape(sh) * sc
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.03, f"{name} rel-err {rel}"


# ===================== CROSS-PRODUCT: causal × head_dim (>hd64) =====================
# causal is head-dim-orthogonal (a transposed mask + tile-skip), but it was only regression-
# tested at hd64. Exercising it at hd96 (1-CTA), hd128/hd192 (2-CTA) caught a real bug: hd192
# causal uses sdQaccum_stage=1, whose tail-drain wait_group had no committed group -> protocol
# Failed. n_nb==1 (Sk==128 for 1-CTA, single cluster for 2-CTA) keeps it protocol-orderable.
CAUSAL_HEADDIM = [
    ("causal_hd96_1cta", 1, 1, 96, 96, 128, 128),  # 1-CTA, n_nb=1
    ("causal_hd128_2cta", 1, 1, 128, 128, 128, 256),  # 2-CTA single cluster
    ("causal_hd192_2cta", 1, 1, 192, 128, 128, 256),  # 2-CTA DeepSeek (sdQaccum_stage=1)
]


@pytest.mark.parametrize("cfg", CAUSAL_HEADDIM, ids=[c[0] for c in CAUSAL_HEADDIM])
def test_flash_bwd_sm100_causal_headdim(cfg):
    kernel, scale, info = _run_big(*cfg[1:], causal=True)
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"
    _assert_big_value(kernel, scale, info)


# The 1-CTA feature configs (varlen / GQA / deterministic) at hd96 — they were regression-tested
# only at hd64; hd96 adds the non-power-of-2 tcgen05.ld split on top. (2-CTA varlen/GQA/det are
# guarded out by the kernel — a scope boundary, not a gap.)
def test_flash_bwd_sm100_gqa_hd96():
    # GQA's dK/dV cross-CTA reduce-add is now race-free (checker commutative-reduce rule) -> Passed
    # (with a nondeterministic_reduction warning, like every non-deterministic reduce). hd96 adds
    # the ld-split. dK is the unscaled fp32 reduce accumulator -> ×scale; dV unscaled; dQ ×scale.
    kernel, scale, info = _run_gqa(
        {"num_head": 4, "num_head_kv": 2, "head_dim": 96, "seqlen": 128, "label": "gqa_hd96"}
    )
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"
    res = nr.interpret(kernel, info["inputs"])
    dq = np.asarray(res[info["dq_id"]], np.float32).reshape(info["q_shape"]) * scale
    dk = np.asarray(res[info["dk_id"]], np.float32).reshape(info["kv_shape"]) * scale
    dv = np.asarray(res[info["dv_id"]], np.float32).reshape(info["kv_shape"])
    for name, g, gr in (
        ("dQ", dq, info["dq_ref"]),
        ("dK", dk, info["dk_ref"]),
        ("dV", dv, info["dv_ref"]),
    ):
        assert np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6) < 0.03, name


def test_flash_bwd_sm100_deterministic_hd96():
    kernel, scale, info = _run_det(
        {
            "batch": 1,
            "num_head": 1,
            "seqlen_q": 256,
            "seqlen_k": 256,
            "head_dim": 96,
            "label": "det_hd96",
        },
        deterministic=True,
    )
    assert nr.check_protocol(kernel)["status"] == "Passed"
    res = nr.interpret(kernel, info["inputs"])
    dq_id, dk_id, dv_id = info["ids"]
    dq_ref, dk_ref, dv_ref = info["refs"]
    qsh, ksh = info["shapes"]
    for name, g, gr in (
        ("dQ", np.asarray(res[dq_id], np.float32).reshape(qsh) * scale, dq_ref),
        ("dK", np.asarray(res[dk_id], np.float32).reshape(ksh), dk_ref),
        ("dV", np.asarray(res[dv_id], np.float32).reshape(ksh), dv_ref),
    ):
        assert np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6) < 0.03, name


def test_flash_bwd_sm100_varlen_hd96_value():
    # varlen's protocol check is Inconclusive without inputs (runtime tile count) — value is the
    # gate. hd96 adds the ld-split on top of the hd64 varlen datapath.
    kernel, scale, info = _run_varlen(
        {"seqlens": [64, 192, 128], "num_head": 1, "label": "vl_hd96"}, D=96
    )
    _assert_varlen_value(kernel, scale, info)


# deterministic × 2-CTA (hd128/hd192): the dQ GMEM semaphore serializes the cross-CLUSTER
# reduce-adds (cluster c spins until counter==c, adds, increments). multi-cluster (sk512+) is the
# real test — single cluster is already deterministic (each CTA owns disjoint q-rows).
DET_2CTA = [
    ("det_hd128_sk256_1cl", 1, 1, 128, 128, 128, 256),
    ("det_hd128_sk512_2cl", 1, 1, 128, 128, 128, 512),
    ("det_hd192_sk256_1cl", 1, 1, 192, 128, 128, 256),
    ("det_hd192_sk512_2cl", 1, 1, 192, 128, 128, 512),
]


@pytest.mark.parametrize("cfg", DET_2CTA, ids=[c[0] for c in DET_2CTA])
def test_flash_bwd_sm100_deterministic_2cta(cfg):
    kernel, scale, info = _run_big(*cfg[1:], deterministic=True)
    rep = nr.check_protocol(kernel)
    assert rep["status"] == "Passed"
    # serialized -> deterministic -> NO non-determinism warning (the multi-cluster gate).
    assert not any(
        w.get("code") == "nondeterministic_reduction" for w in (rep.get("warnings") or [])
    )
    r1 = nr.interpret(kernel, info["inputs"])
    r2 = nr.interpret(kernel, info["inputs"])
    for i, (gid, gr, sh) in enumerate(info["refs"]):
        assert np.array_equal(np.asarray(r1[gid]), np.asarray(r2[gid]))  # bit-identical
        g = np.asarray(r1[gid], np.float32).reshape(sh) * (scale if i == 0 else 1.0)
        assert np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6) < 0.03


# Orderable large shapes → protocol Passed + value. (B,H,Dq,Dv,Sq,Sk)
LARGE_PROTO = [
    ("1cta_sq1024_sk128_nmb8", 1, 1, 64, 64, 1024, 128),  # n_mb=8, n_nb=1
    ("1cta_sq2048_sk128_nmb16", 1, 1, 64, 64, 2048, 128),  # n_mb=16, n_nb=1
    ("1cta_b2h4_sq512_sk128", 2, 4, 64, 64, 512, 128),  # 8 tasks
    ("2cta_hd128_b2h4_sq128sk256", 2, 4, 128, 128, 128, 256),  # single cluster × 8 head-batch
    ("2cta_hd192_b2h4_sq128sk256", 2, 4, 192, 128, 128, 256),
]
# Non-orderable (square, n_nb>1, non-deterministic) → value only; protocol Inconclusive (expected).
LARGE_VALUE = [
    ("1cta_sq512_sk512_nnb4", 1, 1, 64, 64, 512, 512),
    ("1cta_hd96_sq384_sk384", 1, 1, 96, 96, 384, 384),
]


@pytest.mark.parametrize("cfg", LARGE_PROTO, ids=[c[0] for c in LARGE_PROTO])
def test_flash_bwd_sm100_large_protocol(cfg):
    kernel, scale, info = _run_big(*cfg[1:])
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"
    _assert_big_value(kernel, scale, info)


@pytest.mark.parametrize("cfg", LARGE_VALUE, ids=[c[0] for c in LARGE_VALUE])
def test_flash_bwd_sm100_large_value(cfg):
    # square n_nb>1 non-deterministic: the cross-CTA dQ reduce-adds are atomic + commutative, so
    # the checker passes them (race-free) with a nondeterministic_reduction warning (float reduce
    # is order-dependent). Value must still be correct.
    kernel, scale, info = _run_big(*cfg[1:])
    rep = nr.check_protocol(kernel)
    assert rep["status"] == "Passed"
    assert any(w.get("code") == "nondeterministic_reduction" for w in (rep.get("warnings") or []))
    _assert_big_value(kernel, scale, info)


# Large Sk deterministic: the GMEM semaphore serializes the n_nb KV-tile reduce-adds, so the
# cross-CTA Inconclusive flips to Passed even at n_nb=8.
LARGE_DET = [
    {
        "batch": 1,
        "num_head": 1,
        "seqlen_q": 256,
        "seqlen_k": 512,
        "head_dim": 64,
        "label": "det_sq256_sk512_nnb4",
    },
    {
        "batch": 1,
        "num_head": 1,
        "seqlen_q": 256,
        "seqlen_k": 1024,
        "head_dim": 64,
        "label": "det_sq256_sk1024_nnb8",
    },
]


@pytest.mark.parametrize("cfg", LARGE_DET, ids=[c["label"] for c in LARGE_DET])
def test_flash_bwd_sm100_large_deterministic(cfg):
    kernel, scale, info = _run_det(cfg, deterministic=True)
    rep = nr.check_protocol(kernel)
    assert rep["status"] == "Passed"
    # deterministic = HB-ordered reduce-adds = bit-reproducible -> NO non-determinism warning
    # (the distinguishing fact vs the non-deterministic path).
    assert not any(
        w.get("code") == "nondeterministic_reduction" for w in (rep.get("warnings") or [])
    )
    res = nr.interpret(kernel, info["inputs"])
    dq_id, dk_id, dv_id = info["ids"]
    dq_ref, dk_ref, dv_ref = info["refs"]
    qsh, ksh = info["shapes"]
    dq = np.asarray(res[dq_id], np.float32).reshape(qsh) * scale
    dk = np.asarray(res[dk_id], np.float32).reshape(ksh)
    dv = np.asarray(res[dv_id], np.float32).reshape(ksh)
    for name, g, gr in (("dQ", dq, dq_ref), ("dK", dk, dk_ref), ("dV", dv, dv_ref)):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.03, f"{name} rel-err {rel}"


# Multi-m-block: Sq=256 -> n_mb=2 Q-tiles attend one Sk=128 kv-tile (n_nb=1). Exercises
# the dV/dK cross-m-block accumulation + double-buffered operands + per-mb barrier phases.
MULTI_BLOCK = [
    {
        "batch": 1,
        "num_head": 1,
        "head_dim": 64,
        "seqlen_q": 256,
        "seqlen_k": 128,
        "label": "nmb2_b1h1_hd64",
    },
    {
        "batch": 1,
        "num_head": 2,
        "head_dim": 64,
        "seqlen_q": 256,
        "seqlen_k": 128,
        "label": "nmb2_b1h2_hd64",
    },
    # n_mb=3 (Sq=384) exercises operand stage REUSE (stage mb%2 used by mb and mb+2): the
    # NEW-3/F5 operand empty barriers (q_free/do_free/lse_free/dps_free) guard the load->consume
    # WAR that double-buffering alone cannot. Proves n_mb>2 correct.
    {
        "batch": 1,
        "num_head": 1,
        "head_dim": 64,
        "seqlen_q": 384,
        "seqlen_k": 128,
        "label": "nmb3_b1h1_hd64",
    },
]


def _run_mqk(cfg_d):
    B, H, D = cfg_d["batch"], cfg_d["num_head"], cfg_d["head_dim"]
    Sq, Sk = cfg_d["seqlen_q"], cfg_d["seqlen_k"]
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0)
    q = (rng.normal(size=(B * Sq, H, D)) * 0.3).astype(np.float32)
    k = (rng.normal(size=(B * Sk, H, D)) * 0.3).astype(np.float32)
    v = (rng.normal(size=(B * Sk, H, D)) * 0.3).astype(np.float32)
    do = (rng.normal(size=(B * Sq, H, D)) * 0.3).astype(np.float32)
    lse = np.zeros((B * Sq, H), np.float32)
    dpsum = np.zeros((B * Sq, H), np.float32)
    dq_ref = np.zeros((B * Sq, H, D), np.float32)
    dk_ref = np.zeros((B * Sk, H, D), np.float32)
    dv_ref = np.zeros((B * Sk, H, D), np.float32)
    for b in range(B):
        for h in range(H):
            qs, ks = slice(b * Sq, (b + 1) * Sq), slice(b * Sk, (b + 1) * Sk)
            dq, dk, dv, o, l = oracle.fa_bwd_from_qkv(
                q[qs, h], k[ks, h], v[ks, h], do[qs, h], scale
            )
            lse[qs, h] = l * LOG2_E
            dpsum[qs, h] = (do[qs, h] * o).sum(-1)
            dq_ref[qs, h] = dq
            dk_ref[ks, h] = dk
            dv_ref[ks, h] = dv
    cfg = FlashBwdSm100Config(batch=B, num_head=H, seqlen_q=Sq, seqlen_k=Sk, head_dim=D)
    kernel = build_flash_bwd_sm100(cfg)
    qg, kg, vg, dog, lseg, dpsg, dqg, dkg, dvg = kernel.args
    return (
        kernel,
        scale,
        dict(
            inputs={qg: bf(q), kg: bf(k), vg: bf(v), dog: bf(do), lseg: lse, dpsg: dpsum},
            refs=(dq_ref, dk_ref, dv_ref),
            ids=(dqg.id, dkg.id, dvg.id),
            shapes=((B * Sq, H, D), (B * Sk, H, D)),
        ),
    )


@pytest.mark.parametrize("cfg", MULTI_BLOCK, ids=[c["label"] for c in MULTI_BLOCK])
def test_flash_bwd_sm100_multiblock(cfg):
    kernel, scale, info = _run_mqk(cfg)
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"
    res = nr.interpret(kernel, info["inputs"])
    dq_id, dk_id, dv_id = info["ids"]
    dq_ref, dk_ref, dv_ref = info["refs"]
    qsh, ksh = info["shapes"]
    dq = np.asarray(res[dq_id], np.float32).reshape(qsh) * scale
    dk = np.asarray(res[dk_id], np.float32).reshape(ksh)
    dv = np.asarray(res[dv_id], np.float32).reshape(ksh)
    for name, g, gr in (("dQ", dq, dq_ref), ("dK", dk, dk_ref), ("dV", dv, dv_ref)):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.03, f"{name} rel-err {rel}"


# ============================== CAUSAL ==============================
# The bwd recomputes S=K·Qᵀ per (kv-tile, Q-tile) and masks the transposed fragment fragS
# [kv-row=tid, q-col=j] to −inf where kv_local > q_local + (Sk−Sq) BEFORE P=exp2 -> P=0 on
# masked pairs (then dS=P∘(dP−dPsum)=0 inherits the zero -> dV/dK/dQ get 0 contribution from
# masked (q,kv)). Implementation is MASK-ONLY (no m_block_min loop-skip): value-correct because
# fully-above-diagonal tiles mask to all-zero P (CAUSAL_MAP §C2/§3). The oracle masks
# identically (jj <= ii + (sk−sq) kept) with the causal LSE/dPsum from the masked forward.
#
# Two causal shapes:
#  (A) CAUSAL_PROTO  Sq=Sk=128, hd64, n_nb=1, n_mb=1 — single diagonal tile (lower-triangular
#      mask). n_nb=1 => one persistent task/CTA, so the dQ reduce-add has no cross-CTA dq_g
#      overlap and check_protocol PASSES. This is the trace+protocol+value gate.
#  (B) CAUSAL_VALUE  Sq=Sk=256, hd64, n_nb=2, n_mb=2 — the CAUSAL_MAP §2 worked example that
#      exercises ALL THREE tile classes: kv0×q0 diagonal (partial), kv0×q1 fully-below (no
#      mask), kv1×q0 fully-above (masked to all-zero P), kv1×q1 diagonal. VALUE-ONLY: with
#      n_nb=2 the two kv-tiles run as two persistent CTAs that BOTH dQ-reduce-add into the SAME
#      dq_g rows; check_protocol returns Inconclusive on that cross-CTA overlapping async
#      reduce-add (value-correct atomic accumulation, but the checker can't prove an ordering).
#      That Inconclusive is PRE-EXISTING for any n_nb>=2 shape (reproduces identically with
#      is_causal=False) and is unrelated to the mask, so this config asserts value only.
CAUSAL_PROTO = [
    {
        "batch": 1,
        "num_head": 1,
        "seqlen_q": 128,
        "seqlen_k": 128,
        "head_dim": 64,
        "label": "causal_b1h1s128_hd64",
    },
    {
        "batch": 1,
        "num_head": 2,
        "seqlen_q": 128,
        "seqlen_k": 128,
        "head_dim": 64,
        "label": "causal_b1h2s128_hd64",
    },
    {
        "batch": 2,
        "num_head": 2,
        "seqlen_q": 128,
        "seqlen_k": 128,
        "head_dim": 64,
        "label": "causal_b2h2s128_hd64",
    },
]
CAUSAL_VALUE = [
    {
        "batch": 1,
        "num_head": 1,
        "seqlen_q": 256,
        "seqlen_k": 256,
        "head_dim": 64,
        "label": "causal_b1h1s256_hd64",
    }
]


def _run_causal(cfg_d):
    B, H, D = cfg_d["batch"], cfg_d["num_head"], cfg_d["head_dim"]
    Sq, Sk = cfg_d["seqlen_q"], cfg_d["seqlen_k"]
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0)
    q = (rng.normal(size=(B * Sq, H, D)) * 0.3).astype(np.float32)
    k = (rng.normal(size=(B * Sk, H, D)) * 0.3).astype(np.float32)
    v = (rng.normal(size=(B * Sk, H, D)) * 0.3).astype(np.float32)
    do = (rng.normal(size=(B * Sq, H, D)) * 0.3).astype(np.float32)
    lse = np.zeros((B * Sq, H), np.float32)
    dpsum = np.zeros((B * Sq, H), np.float32)
    dq_ref = np.zeros((B * Sq, H, D), np.float32)
    dk_ref = np.zeros((B * Sk, H, D), np.float32)
    dv_ref = np.zeros((B * Sk, H, D), np.float32)
    for b in range(B):
        for h in range(H):
            qs, ks = slice(b * Sq, (b + 1) * Sq), slice(b * Sk, (b + 1) * Sk)
            dq, dk, dv, o, l = oracle.fa_bwd_from_qkv(
                q[qs, h], k[ks, h], v[ks, h], do[qs, h], scale, causal=True
            )
            lse[qs, h] = l * LOG2_E
            dpsum[qs, h] = (do[qs, h] * o).sum(-1)
            dq_ref[qs, h] = dq
            dk_ref[ks, h] = dk
            dv_ref[ks, h] = dv
    cfg = FlashBwdSm100Config(
        batch=B, num_head=H, seqlen_q=Sq, seqlen_k=Sk, head_dim=D, is_causal=True
    )
    kernel = build_flash_bwd_sm100(cfg)
    qg, kg, vg, dog, lseg, dpsg, dqg, dkg, dvg = kernel.args
    return (
        kernel,
        scale,
        dict(
            inputs={qg: bf(q), kg: bf(k), vg: bf(v), dog: bf(do), lseg: lse, dpsg: dpsum},
            refs=(dq_ref, dk_ref, dv_ref),
            ids=(dqg.id, dkg.id, dvg.id),
            shapes=((B * Sq, H, D), (B * Sk, H, D)),
        ),
    )


def _assert_causal_value(kernel, scale, info):
    res = nr.interpret(kernel, info["inputs"])
    dq_id, dk_id, dv_id = info["ids"]
    dq_ref, dk_ref, dv_ref = info["refs"]
    qsh, ksh = info["shapes"]
    dq = np.asarray(res[dq_id], np.float32).reshape(qsh) * scale  # dQaccum -> dQ
    dk = np.asarray(res[dk_id], np.float32).reshape(ksh)
    dv = np.asarray(res[dv_id], np.float32).reshape(ksh)
    for name, g, gr in (("dQ", dq, dq_ref), ("dK", dk, dk_ref), ("dV", dv, dv_ref)):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.03, f"{name} rel-err {rel}"


@pytest.mark.parametrize("cfg", CAUSAL_PROTO, ids=[c["label"] for c in CAUSAL_PROTO])
def test_flash_bwd_sm100_causal(cfg):
    # trace + protocol + value (single diagonal tile, n_nb=1 so dQ reduce has no cross-CTA dep).
    kernel, scale, info = _run_causal(cfg)
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"
    _assert_causal_value(kernel, scale, info)


@pytest.mark.parametrize("cfg", CAUSAL_VALUE, ids=[c["label"] for c in CAUSAL_VALUE])
def test_flash_bwd_sm100_causal_value_three_classes(cfg):
    # Value-only: Sq=Sk=256 (n_nb=2, n_mb=2) exercises all three causal tile classes
    # (diagonal / fully-below / fully-above-masked). Protocol is pre-existing-Inconclusive for
    # n_nb>=2 (cross-CTA dQ reduce-add overlap), unrelated to the mask — so we still trace
    # (structurally valid) but assert value, not protocol-Passed.
    kernel, scale, info = _run_causal(cfg)
    assert nr.trace(kernel)
    _assert_causal_value(kernel, scale, info)


@pytest.mark.parametrize("cfg", CAUSAL_VALUE, ids=[c["label"] for c in CAUSAL_VALUE])
def test_flash_bwd_sm100_causal_m_block_min_skip(cfg):
    # The m_block_min tile-skip has TEETH: for Sq=Sk=256 causal (n_nb=2) the (mb=0, nb=1) tile
    # is fully above the diagonal and must issue NO work. Compare the executed-statement count
    # of the causal build (which skips that tile in lockstep across load/mma/compute/reduce)
    # against the SAME-shape non-causal build (which issues every tile). Causal must execute
    # strictly fewer statements — proving the skip drops the above-diagonal tile, not just masks
    # it. Both builds are still value-correct (asserted by the value tests).
    B, H, D = cfg["batch"], cfg["num_head"], cfg["head_dim"]
    Sq, Sk = cfg["seqlen_q"], cfg["seqlen_k"]
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0)
    q = (rng.normal(size=(B * Sq, H, D)) * 0.3).astype(np.float32)
    k = (rng.normal(size=(B * Sk, H, D)) * 0.3).astype(np.float32)
    v = (rng.normal(size=(B * Sk, H, D)) * 0.3).astype(np.float32)
    do = (rng.normal(size=(B * Sq, H, D)) * 0.3).astype(np.float32)
    lse = np.zeros((B * Sq, H), np.float32)
    dpsum = np.zeros((B * Sq, H), np.float32)

    def executed_stmts(is_causal):
        c = FlashBwdSm100Config(
            batch=B, num_head=H, seqlen_q=Sq, seqlen_k=Sk, head_dim=D, is_causal=is_causal
        )
        ker = build_flash_bwd_sm100(c)
        a = ker.args
        rep = nr.trace(
            ker, {a[0]: bf(q), a[1]: bf(k), a[2]: bf(v), a[3]: bf(do), a[4]: lse, a[5]: dpsum}
        )
        assert rep["status"] == "Passed" and rep["completed"]
        return rep["executed_stmts"]

    nc = executed_stmts(False)
    ca = executed_stmts(True)
    assert ca < nc, f"causal skip has no teeth: causal {ca} >= non-causal {nc}"


@pytest.mark.parametrize("cfg", SINGLE_TILE, ids=[c["label"] for c in SINGLE_TILE])
def test_flash_bwd_sm100_value(cfg):
    kernel, scale, info = _run(cfg)
    res = nr.interpret(kernel, info["inputs"])
    dqg_id, dkg_id, dvg_id = info["ids"]
    dq_ref, dk_ref, dv_ref = info["refs"]
    sh = info["shape"]
    dq = np.asarray(res[dqg_id], np.float32).reshape(sh) * scale  # dQaccum -> dQ
    dk = np.asarray(res[dkg_id], np.float32).reshape(sh)
    dv = np.asarray(res[dvg_id], np.float32).reshape(sh)
    for name, g, gr in (("dQ", dq, dq_ref), ("dK", dk, dk_ref), ("dV", dv, dv_ref)):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.03, f"{name} rel-err {rel}"


# ============================== VARLEN (cu_seqlens) ==============================
# Packed variable-length sequences (kernel-only, no new IR — gdn_prefill does varlen with the
# same primitives). Self-attention (Sq==Sk), non-causal, 1-CTA hd64. The packed [T,H,D] buffers
# are over-allocated to batch*maxlen rows (maxlen = max seqlen, the sizing upper bound), and each
# sequence's real token range + per-seq tile counts come from a runtime cu_seqlens lookup. The
# partial last tile of a sub-TILE / non-TILE-multiple sequence is the critical correctness case:
# the compute role masks OOB rows/cols (kv-row>=slen_k or q-col>=slen_q -> S=-inf -> P=0, so dV/dK/
# dQ get a 0 contribution), and the dK/dV store is PREDICATED to valid rows (a scalar store, not a
# fixed-tile TMA) so a partial tail does not overrun into the next packed sequence's output rows.
# The dQ reduce-add is value-safe because the masked OOB q-rows accumulate bit-exact 0.
#
# Two tiers (mirroring CAUSAL_PROTO / CAUSAL_VALUE):
#  VARLEN_PROTO  — trace + protocol(cu_seqlens) + value. Single-task partials (1 seq, sub-TILE) and
#      multi-task all-exact batches keep n_nb=1 with non-overlapping dq_g rows, so check_protocol
#      resolves the runtime schedule (passed the concrete cu_seqlens) and PASSES.
#  VARLEN_VALUE  — value-only. Mixed batches with a sub-TILE tail AND a multi-tile sequence pack
#      multiple partial-tile tasks; protocol is Inconclusive (the same cross-CTA dQ reduce-add
#      overlap / partial-tile data-dependent branching that is Inconclusive for any n_nb>=2 or
#      multi-task-partial shape — pre-existing, reproduces non-varlen, unrelated to the tail mask).
#      Still trace (structurally valid) + per-sequence-oracle value.
VARLEN_PROTO = [
    {"seqlens": [64], "num_head": 1, "label": "vl_proto_64"},  # single sub-TILE partial
    {"seqlens": [128], "num_head": 1, "label": "vl_proto_128"},  # single exact tile
    {"seqlens": [128, 128], "num_head": 1, "label": "vl_proto_128_128"},  # multi-task exact
    {"seqlens": [128, 128, 128], "num_head": 2, "label": "vl_proto_3x128_h2"},
]
VARLEN_VALUE = [
    {
        "seqlens": [64, 192, 128],
        "num_head": 1,
        "label": "vl_64_192_128",
    },  # sub-TILE tail + multi-tile
    {"seqlens": [70, 130], "num_head": 1, "label": "vl_70_130"},  # both partial; 130=128+2
    {"seqlens": [128, 64, 192], "num_head": 1, "label": "vl_128_64_192"},
    {"seqlens": [200], "num_head": 1, "label": "vl_200"},  # single multi-tile partial
    {"seqlens": [64, 192, 128], "num_head": 2, "label": "vl_64_192_128_h2"},
]


def _run_varlen(cfg_d, D=64, causal=False):
    """Build a packed varlen batch + cu_seqlens; per-sequence oracle over the packed rows.
    Mirrors the fixed-length per-(b,h) loop (test_flash_bwd_sm100.py:46-52) but slices the
    packed buffer by cu_seqlens instead of b*S. Self-attention: cu_q == cu_k. When causal,
    each sequence's oracle slice is masked per-sequence (off_b = slen_k−slen_q = 0 for
    self-attention, but the per-seq causal mask + per-seq tile-skip still apply) — this
    validates D2's runtime off_b wiring (m_block_min + key_start use vg.off_b under varlen)."""
    seqlens = cfg_d["seqlens"]
    H = cfg_d["num_head"]
    NS = len(seqlens)
    maxlen = max(seqlens)
    cu = np.array([0] + list(np.cumsum(seqlens)), np.int32)  # cu[b]=start, cu[b+1]-cu[b]=len
    total = NS * maxlen  # over-allocated packed buffer
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0)
    q = (rng.normal(size=(total, H, D)) * 0.3).astype(np.float32)
    k = (rng.normal(size=(total, H, D)) * 0.3).astype(np.float32)
    v = (rng.normal(size=(total, H, D)) * 0.3).astype(np.float32)
    do = (rng.normal(size=(total, H, D)) * 0.3).astype(np.float32)
    lse = np.zeros((total, H), np.float32)
    dpsum = np.zeros((total, H), np.float32)
    dq_ref = np.zeros((total, H, D), np.float32)
    dk_ref = np.zeros((total, H, D), np.float32)
    dv_ref = np.zeros((total, H, D), np.float32)
    valid = np.zeros(total, bool)
    for s in range(NS):
        sl = slice(int(cu[s]), int(cu[s + 1]))
        valid[sl] = True
        for h in range(H):
            dq, dk, dv, o, l = oracle.fa_bwd_from_qkv(
                q[sl, h], k[sl, h], v[sl, h], do[sl, h], scale, causal=causal
            )
            lse[sl, h] = l * LOG2_E
            dpsum[sl, h] = (do[sl, h] * o).sum(-1)
            dq_ref[sl, h] = dq
            dk_ref[sl, h] = dk
            dv_ref[sl, h] = dv
    cfg = FlashBwdSm100Config(
        batch=NS,
        num_head=H,
        seqlen_q=maxlen,
        seqlen_k=maxlen,
        head_dim=D,
        varlen=True,
        is_causal=causal,
    )
    kernel = build_flash_bwd_sm100(cfg)
    qg, kg, vg, dog, lseg, dpsg, dqg, dkg, dvg, cuqg, cukg = kernel.args
    inputs = {
        qg: bf(q),
        kg: bf(k),
        vg: bf(v),
        dog: bf(do),
        lseg: lse,
        dpsg: dpsum,
        cuqg: cu,
        cukg: cu,
    }
    return (
        kernel,
        scale,
        dict(
            inputs=inputs,
            refs=(dq_ref, dk_ref, dv_ref),
            ids=(dqg.id, dkg.id, dvg.id),
            shape=(total, H, D),
            valid=valid,
        ),
    )


def _assert_varlen_value(kernel, scale, info):
    res = nr.interpret(kernel, info["inputs"])
    dq_id, dk_id, dv_id = info["ids"]
    dq_ref, dk_ref, dv_ref = info["refs"]
    sh = info["shape"]
    valid = info["valid"]  # only the real (non-padding) rows have a reference
    dq = np.asarray(res[dq_id], np.float32).reshape(sh) * scale  # dQaccum -> dQ
    dk = np.asarray(res[dk_id], np.float32).reshape(sh)
    dv = np.asarray(res[dv_id], np.float32).reshape(sh)
    for name, g, gr in (("dQ", dq, dq_ref), ("dK", dk, dk_ref), ("dV", dv, dv_ref)):
        rel = np.abs(g[valid] - gr[valid]).max() / (np.abs(gr[valid]).max() + 1e-6)
        assert rel < 0.03, f"{name} rel-err {rel}"
    # OOB-tail guard: the over-allocated padding rows (between each sequence's end and its
    # tile span) must be left EXACTLY zero — a partial tail must not overrun the predicated
    # dK/dV store / dQ reduce-add into the next sequence's (or padding) output rows.
    pad = ~valid
    if pad.any():
        assert np.abs(dk[pad]).max() == 0.0 and np.abs(dv[pad]).max() == 0.0, "dK/dV tail overrun"
        assert np.abs(dq[pad]).max() == 0.0, "dQ tail overrun"


@pytest.mark.parametrize("cfg", VARLEN_PROTO, ids=[c["label"] for c in VARLEN_PROTO])
def test_flash_bwd_sm100_varlen_proto(cfg):
    # trace + protocol (resolved by the concrete cu_seqlens) + per-sequence value.
    kernel, scale, info = _run_varlen(cfg)
    assert nr.trace(kernel, info["inputs"])["status"] == "Passed"
    assert nr.check_protocol(kernel, info["inputs"])["status"] == "Passed"
    _assert_varlen_value(kernel, scale, info)


@pytest.mark.parametrize("cfg", VARLEN_VALUE, ids=[c["label"] for c in VARLEN_VALUE])
def test_flash_bwd_sm100_varlen_value(cfg):
    # Value-only: mixed batches with a sub-TILE tail + multi-tile sequence. Protocol is
    # Inconclusive for these (cross-CTA dQ reduce-add overlap / multi-task partial-tile branching,
    # pre-existing for n_nb>=2, unrelated to the OOB-tail mask) — so trace (structurally valid)
    # + per-sequence-oracle value (incl. the exact-zero padding-row guard).
    kernel, scale, info = _run_varlen(cfg)
    assert nr.trace(kernel, info["inputs"])["status"] == "Passed"
    _assert_varlen_value(kernel, scale, info)


# ===================== VARLEN × 2-CTA (head_dim=128 -> cluster_size=2) =====================
# Packed mixed-seqlen batch at D=128 (the CONFIG_MAP head_dim>=128 -> 2-CTA threshold). A 2-CTA
# cluster covers cg=2 adjacent kv-tiles (nb = c_nb*cg + cta_in_cluster). The critical case is an
# ODD per-sequence kv-tile count (slen_k not a multiple of cg*TILE_N = 256): the cluster's leader
# tile is valid but the PEER tile is past-sequence. The kernel keeps the WHOLE cluster in lockstep
# — both CTAs run the cluster cta_group::2 MMA + compute + cross-CTA dS exchange under the CLUSTER
# predicate (nb-cta_in_cluster < n_nb_b), so the cross-CTA handshakes stay balanced (no deadlock);
# the past tile's K/V is overscan and its P is masked to 0 (kv-row>=slen_k), so it contributes 0.
# Only the per-CTA OUTPUT (dK/dV store) is gated by the per-CTA tile_valid (nb < n_nb_b) so the
# past-sequence CTA writes nothing. The dQ q-rows are cluster-shared (valid), so both CTAs reduce.
# The over-allocated grid rounds n_nb up to a multiple of cg, so an odd maxlen-kv-tile count is
# handled by the same masked-cluster mechanism. These seqlens exercise odd kv-tile counts, partial
# Q-tiles, and multi-sequence packing.
VARLEN_2CTA = [
    {"seqlens": [256, 128], "num_head": 1, "label": "vl2cta_256_128"},  # odd peer tile (seq1)
    {"seqlens": [128, 256], "num_head": 1, "label": "vl2cta_128_256"},  # odd leader-seq (seq0)
    {"seqlens": [256, 256], "num_head": 1, "label": "vl2cta_256_256"},  # both full clusters
    {"seqlens": [128, 128], "num_head": 1, "label": "vl2cta_128_128"},  # maxlen=128 -> n_nb rounded
    {"seqlens": [384, 128], "num_head": 1, "label": "vl2cta_384_128"},  # partial Q-tile + odd kv
    {
        "seqlens": [192, 320],
        "num_head": 1,
        "label": "vl2cta_192_320",
    },  # both partial, odd maxlen-nnb
    {"seqlens": [128, 64, 256], "num_head": 1, "label": "vl2cta_128_64_256"},  # multi-seq, sub-tile
]


@pytest.mark.parametrize("cfg", VARLEN_2CTA, ids=[c["label"] for c in VARLEN_2CTA])
def test_flash_bwd_sm100_varlen_2cta(cfg):
    # trace + protocol (resolved by the concrete cu_seqlens) + per-sequence value + zero padding
    # tail. The partial-cluster deadlock (odd kv-tile count) is the regression this guards.
    kernel, scale, info = _run_varlen(cfg, D=128)
    assert nr.trace(kernel, info["inputs"])["status"] == "Passed"
    assert nr.check_protocol(kernel, info["inputs"])["status"] == "Passed"
    _assert_varlen_value(kernel, scale, info)


# ===================== VARLEN + CAUSAL (D2: per-sequence causal offset) =====================
# Packed mixed-seqlen batch with causal masking. Self-attention (cu_q == cu_k, slen_q == slen_k
# per sequence), so the per-sequence causal offset off_b = slen_k_b − slen_q_b == 0 for every
# sequence — but the kernel must STILL be correct per sequence: the causal mask key_start and the
# m_block_min tile-skip are wired to the RUNTIME vg.off_b (D2), not the compile-time (Sk−Sq)
# (=maxlen−maxlen=0). For self-attention both evaluate to 0, but routing the runtime off_b is what
# makes the wiring correct (the field is no longer dead) and is the prerequisite for cross-
# attention varlen+causal. Each sequence's oracle slice is masked with causal=True. Value-only:
# the mixed seqlens pack >=2 tasks with a sub-TILE tail (same Inconclusive-protocol class as the
# non-causal VARLEN_VALUE), so trace (structurally valid) + per-sequence-oracle value.
VARLEN_CAUSAL_VALUE = [
    {
        "seqlens": [64, 192, 128],
        "num_head": 1,
        "label": "vlc_64_192_128",
    },  # sub-TILE tail + multi-tile
    {"seqlens": [128, 128], "num_head": 1, "label": "vlc_128_128"},  # multi-task exact diagonals
    {"seqlens": [64, 192, 128], "num_head": 2, "label": "vlc_64_192_128_h2"},
]


# ============================== GQA (grouped-query attention) ==============================
# num_q_heads > num_kv_heads: G = num_q_heads // num_kv_heads query-heads SHARE each kv-head.
# Two GQA-defining changes vs MHA (gated on G>1; G==1 is byte-identical to the MHA IR):
#   (1) head remap — K/V load (and the dK/dV epilogue target) use head_kv = head // G; Q/dO/
#       LSE/dPsum/dQ stay per-q-head (`head`). The scheduler still enumerates per q-head.
#   (2) dK/dV GROUP ACCUMULATION — the dK/dV epilogue switches from a plain bf16 tma_store to
#       a tma_reduce_add into the shared kv-head's FP32 accumulator (dk_g/dv_g become f32 args,
#       like dq_g). The G q-heads of a group each atomically add their dK/dV contribution into
#       the SAME kv-head rows, so dK[h_kv] = Σ_{h//G==h_kv} dK_contrib(h). The in-epilogue
#       dK·scale is DROPPED for GQA (dKaccum is unscaled, like dQaccum) — the test applies the
#       softmax_scale to dK in the postprocess (mirroring flashattn's GQA postprocess).
#
# The oracle (single-head fa_bwd_from_qkv) is UNCHANGED; the test driver does the GQA broadcast
# (each q-head runs against its shared kv-head) + group-sum (the G contributions sum per kv-head).
GQA = [
    {"num_head": 4, "num_head_kv": 2, "head_dim": 64, "seqlen": 128, "label": "gqa_h4kv2_s128_hd64"}
]


def _run_gqa(cfg_d):
    """GQA reference driver. Q/dO are q-headed [S,Hq,D]; K/V are kv-headed [S,Hkv,D].
    For each q-head h, kv-head = h//G; run the single-head oracle with the SHARED K/V, then
    dQ[h] = dq[h] and dK[kv]/dV[kv] = SUM over h in the group {h//G==kv} of dk_contrib/dv_contrib."""
    Hq, Hkv, D = cfg_d["num_head"], cfg_d["num_head_kv"], cfg_d["head_dim"]
    # symmetric `seqlen` OR asymmetric seqlen_q/seqlen_k (2-CTA needs Sk>=2 tiles for a cluster).
    Sq = cfg_d.get("seqlen_q", cfg_d.get("seqlen"))
    Sk = cfg_d.get("seqlen_k", cfg_d.get("seqlen"))
    Dv = cfg_d.get("head_dim_v", D)
    G = Hq // Hkv
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0)
    q = (rng.normal(size=(Sq, Hq, D)) * 0.3).astype(np.float32)
    do = (rng.normal(size=(Sq, Hq, Dv)) * 0.3).astype(np.float32)
    kk = (rng.normal(size=(Sk, Hkv, D)) * 0.3).astype(np.float32)
    vv = (rng.normal(size=(Sk, Hkv, Dv)) * 0.3).astype(np.float32)
    lse = np.zeros((Sq, Hq), np.float32)
    dpsum = np.zeros((Sq, Hq), np.float32)
    dq_ref = np.zeros((Sq, Hq, D), np.float32)
    dk_ref = np.zeros((Sk, Hkv, D), np.float32)
    dv_ref = np.zeros((Sk, Hkv, Dv), np.float32)
    for h in range(Hq):
        kv = h // G
        dq, dk, dv, o, l = oracle.fa_bwd_from_qkv(q[:, h], kk[:, kv], vv[:, kv], do[:, h], scale)
        lse[:, h] = l * LOG2_E
        dpsum[:, h] = (do[:, h] * o).sum(-1)
        dq_ref[:, h] = dq
        dk_ref[:, kv] += dk  # group-sum: G q-heads accumulate into the shared kv-head
        dv_ref[:, kv] += dv
    cfg = FlashBwdSm100Config(
        batch=1, num_head=Hq, num_head_kv=Hkv, seqlen_q=Sq, seqlen_k=Sk, head_dim=D, head_dim_v=Dv
    )
    kernel = build_flash_bwd_sm100(cfg)
    qg, kg, vg, dog, lseg, dpsg, dqg, dkg, dvg = kernel.args
    return (
        kernel,
        scale,
        dict(
            inputs={qg: bf(q), kg: bf(kk), vg: bf(vv), dog: bf(do), lseg: lse, dpsg: dpsum},
            dq_ref=dq_ref,
            dk_ref=dk_ref,
            dv_ref=dv_ref,
            dq_id=dqg.id,
            dk_id=dkg.id,
            dv_id=dvg.id,
            q_shape=(Sq, Hq, D),
            kv_shape=(Sk, Hkv, D),
            dv_shape=(Sk, Hkv, Dv),
        ),
    )


@pytest.mark.parametrize("cfg", GQA, ids=[c["label"] for c in GQA])
def test_flash_bwd_sm100_gqa(cfg):
    # trace + dQ/dK/dV value. dk_g/dv_g are FP32 reduce-add accumulators (unscaled); the test
    # applies softmax_scale to dK (dV is never scaled) and to dQaccum.
    #
    # PROTOCOL is Inconclusive (not asserted here) for the SAME pre-existing cross-CTA reduce-add
    # reason the dQ epilogue already hits at n_nb>=2 (see CAUSAL_VALUE / VARLEN_VALUE above): the G
    # q-heads of a group run as G separate persistent CTAs that all tma_reduce_add into the SHARED
    # kv-head's dk_g/dv_g rows. The reduce-add is a read-modify-write, so the checker sees one CTA's
    # accumulate reading rows another CTA's in-flight accumulate is writing and cannot prove an
    # ordering — it reports `async_group_cross_stream_dest_access` (Inconclusive). This is the
    # value-correct atomic-accumulation default (map §2; the deterministic semaphore-serialized
    # variant is out of milestone scope), NOT a GQA bug: the SAME shape with G=1 (num_head_kv=4,
    # each head writes its own rows, no overlap) passes check_protocol. So we trace (structurally
    # valid) + assert dQ/dK/dV value, mirroring every other overlapping-reduce-add config in this file.
    kernel, scale, info = _run_gqa(cfg)
    assert nr.trace(kernel)
    res = nr.interpret(kernel, info["inputs"])
    dq = (
        np.asarray(res[info["dq_id"]], np.float32).reshape(info["q_shape"]) * scale
    )  # dQaccum -> dQ
    dk = (
        np.asarray(res[info["dk_id"]], np.float32).reshape(info["kv_shape"]) * scale
    )  # dKaccum (unscaled) -> dK
    dv = np.asarray(res[info["dv_id"]], np.float32).reshape(info["dv_shape"])  # dV unscaled
    for name, g, gr in (
        ("dQ", dq, info["dq_ref"]),
        ("dK", dk, info["dk_ref"]),
        ("dV", dv, info["dv_ref"]),
    ):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.03, f"{name} rel-err {rel}"


# GQA × 2-CTA (head_dim>=128 -> cluster_size=2). The dK/dV epilogue stages f32 into sdK/sdV
# (aliased onto sK/sV, grown to the f32 size) and tma_reduce_add's the G q-heads of a group into
# the shared kv-head. The cross-CTA/cross-group reduce-add is race-free (checker commutative-reduce
# rule) -> protocol Passed. Asymmetric Sq=128/Sk>=256 keeps n_mb=1 (single Q-tile) while giving the
# cluster its cg=2 kv-tiles. hd192 (DeepSeek 192/128) exercises the dedicated overlapping TMEM.
GQA_2CTA = [
    {
        "num_head": 4,
        "num_head_kv": 2,
        "head_dim": 128,
        "seqlen_q": 128,
        "seqlen_k": 256,
        "label": "gqa2cta_h4kv2_hd128_sk256",
    },
    {
        "num_head": 8,
        "num_head_kv": 2,
        "head_dim": 128,
        "seqlen_q": 128,
        "seqlen_k": 256,
        "label": "gqa2cta_h8kv2_hd128_sk256",
    },
    {
        "num_head": 4,
        "num_head_kv": 2,
        "head_dim": 128,
        "seqlen_q": 128,
        "seqlen_k": 512,
        "label": "gqa2cta_h4kv2_hd128_sk512",
    },
    {
        "num_head": 2,
        "num_head_kv": 1,
        "head_dim": 128,
        "seqlen_q": 128,
        "seqlen_k": 256,
        "label": "gqa2cta_h2kv1_hd128_sk256",
    },
    {
        "num_head": 4,
        "num_head_kv": 2,
        "head_dim": 192,
        "head_dim_v": 128,
        "seqlen_q": 128,
        "seqlen_k": 256,
        "label": "gqa2cta_h4kv2_hd192_sk256",
    },
    {
        "num_head": 4,
        "num_head_kv": 2,
        "head_dim": 192,
        "head_dim_v": 128,
        "seqlen_q": 128,
        "seqlen_k": 512,
        "label": "gqa2cta_h4kv2_hd192_sk512",
    },
]


@pytest.mark.parametrize("cfg", GQA_2CTA, ids=[c["label"] for c in GQA_2CTA])
def test_flash_bwd_sm100_gqa_2cta(cfg):
    # GQA × 2-CTA: trace + check_protocol Passed (commutative-reduce) + dQ/dK/dV value.
    # Regression for M00018: the 2-CTA SMEM sizes dict had hardcoded *2 (bf16) for sdK/sdV while
    # the GQA tensors are f32 -> sdV overflowed its alloc and sdK's stage corrupted sdV[0:64] -> dV
    # garbage (rel ~1.9). The fixed f32 sizing makes dV correct.
    kernel, scale, info = _run_gqa(cfg)
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"
    res = nr.interpret(kernel, info["inputs"])
    dq = np.asarray(res[info["dq_id"]], np.float32).reshape(info["q_shape"]) * scale
    dk = np.asarray(res[info["dk_id"]], np.float32).reshape(info["kv_shape"]) * scale
    dv = np.asarray(res[info["dv_id"]], np.float32).reshape(info["dv_shape"])
    for name, g, gr in (
        ("dQ", dq, info["dq_ref"]),
        ("dK", dk, info["dk_ref"]),
        ("dV", dv, info["dv_ref"]),
    ):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.03, f"{name} rel-err {rel}"


@pytest.mark.parametrize("cfg", VARLEN_CAUSAL_VALUE, ids=[c["label"] for c in VARLEN_CAUSAL_VALUE])
def test_flash_bwd_sm100_varlen_causal_value(cfg):
    # D2 validation: per-sequence causal mask + per-seq tile-skip via the RUNTIME off_b. The
    # per-sequence oracle (causal=True) masks each sequence's slice independently; the kernel's
    # mask key_start = nb*TILE_N − vg.off_b and m_block_min = max(0,(nb*TILE_N − vg.off_b)//TILE_M)
    # must reproduce it per sequence (off_b == 0 for self-attention). dQ/dK/dV rel < 0.03 + the
    # exact-zero padding-row guard (D1: the dQ reduce-add tail explicitly zeroes OOB rows).
    kernel, scale, info = _run_varlen(cfg, causal=True)
    assert nr.trace(kernel, info["inputs"])["status"] == "Passed"
    _assert_varlen_value(kernel, scale, info)


# ============================== DETERMINISTIC ==============================
# Bit-reproducible dQ / GQA-dKV accumulation: each cross-CTA reduce-add into a shared
# output tile is SERIALIZED into a fixed order by a per-tile GMEM i32 semaphore (the
# new gmem_wait_eq / gmem_atomic_add IR ops). The dQ contenders (the n_nb KV-tile CTAs)
# add into dq_g[(b,h,mb)] in ascending n_block; the GQA-dKV contenders (the G q-heads of
# a group) add into dk_g/dv_g[(b,head_kv,nb,wg)] in ascending head%G. The add STAYS a
# reduce-add — only the ORDER is fixed. Because the order now EXISTS, the checker flips
# the cross-CTA dest-access Inconclusive -> Passed AND the result is bit-identical.
#
# POSITIVE  : Sq=Sk=256 hd64 n_nb=2 (two KV-tile CTAs reduce-add into the SAME dQ tile —
#             Inconclusive non-deterministically) and GQA h4/kv2 (G=2 q-heads share a
#             kv-head). With deterministic: Passed + value rel<0.03 + np.array_equal
#             across 2 interpret runs.
# NEGATIVE A: the SAME shapes WITHOUT deterministic stay Inconclusive (proves the sem is
#             load-bearing — see CAUSAL_VALUE/GQA which document the pre-existing gap).
# NEGATIVE B: a wrong lock-value (off by +1) makes a writer wait a counter value that is
#             never produced -> the interpreter DEADLOCKS (the wrong serialization is
#             caught, not silently Passed — the HB is real). Driven via a monkeypatched
#             lock-value in a tiny standalone kernel (tests/test_gmem_semaphore.py covers
#             the same proof + signal-before-drain at the IR level).
DETERMINISTIC = [
    {
        "batch": 1,
        "num_head": 1,
        "seqlen_q": 256,
        "seqlen_k": 256,
        "head_dim": 64,
        "label": "det_b1h1s256_hd64_nnb2",
    },
    {
        "batch": 1,
        "num_head": 2,
        "seqlen_q": 256,
        "seqlen_k": 256,
        "head_dim": 64,
        "label": "det_b1h2s256_hd64_nnb2",
    },
]


def _run_det(cfg_d, deterministic):
    B, H, D = cfg_d["batch"], cfg_d["num_head"], cfg_d["head_dim"]
    Sq, Sk = cfg_d["seqlen_q"], cfg_d["seqlen_k"]
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0)
    q = (rng.normal(size=(B * Sq, H, D)) * 0.3).astype(np.float32)
    k = (rng.normal(size=(B * Sk, H, D)) * 0.3).astype(np.float32)
    v = (rng.normal(size=(B * Sk, H, D)) * 0.3).astype(np.float32)
    do = (rng.normal(size=(B * Sq, H, D)) * 0.3).astype(np.float32)
    lse = np.zeros((B * Sq, H), np.float32)
    dpsum = np.zeros((B * Sq, H), np.float32)
    dq_ref = np.zeros((B * Sq, H, D), np.float32)
    dk_ref = np.zeros((B * Sk, H, D), np.float32)
    dv_ref = np.zeros((B * Sk, H, D), np.float32)
    for b in range(B):
        for h in range(H):
            qs, ks = slice(b * Sq, (b + 1) * Sq), slice(b * Sk, (b + 1) * Sk)
            dq, dk, dv, o, l = oracle.fa_bwd_from_qkv(
                q[qs, h], k[ks, h], v[ks, h], do[qs, h], scale
            )
            lse[qs, h] = l * LOG2_E
            dpsum[qs, h] = (do[qs, h] * o).sum(-1)
            dq_ref[qs, h] = dq
            dk_ref[ks, h] = dk
            dv_ref[ks, h] = dv
    cfg = FlashBwdSm100Config(
        batch=B, num_head=H, seqlen_q=Sq, seqlen_k=Sk, head_dim=D, deterministic=deterministic
    )
    kernel = build_flash_bwd_sm100(cfg)
    qg, kg, vg, dog, lseg, dpsg, dqg, dkg, dvg = kernel.args[:9]
    return (
        kernel,
        scale,
        dict(
            inputs={qg: bf(q), kg: bf(k), vg: bf(v), dog: bf(do), lseg: lse, dpsg: dpsum},
            refs=(dq_ref, dk_ref, dv_ref),
            ids=(dqg.id, dkg.id, dvg.id),
            shapes=((B * Sq, H, D), (B * Sk, H, D)),
        ),
    )


@pytest.mark.parametrize("cfg", DETERMINISTIC, ids=[c["label"] for c in DETERMINISTIC])
def test_flash_bwd_sm100_deterministic_dq(cfg):
    # POSITIVE: deterministic flips the cross-CTA dQ reduce-add Inconclusive -> Passed,
    # the value is correct, AND the dQ output is bit-identical across two interpret runs.
    kernel, scale, info = _run_det(cfg, deterministic=True)
    assert nr.trace(kernel)
    rep = nr.check_protocol(kernel)
    assert rep["status"] == "Passed", (
        f"deterministic must resolve Inconclusive, got {rep['status']}"
    )
    res1 = nr.interpret(kernel, info["inputs"])
    res2 = nr.interpret(kernel, info["inputs"])
    dq_id, dk_id, dv_id = info["ids"]
    dq_ref, dk_ref, dv_ref = info["refs"]
    qsh, ksh = info["shapes"]
    # bit-identical across runs (only holds in deterministic mode).
    for gid in (dq_id, dk_id, dv_id):
        assert np.array_equal(np.asarray(res1[gid]), np.asarray(res2[gid])), "not bit-identical"
    dq = np.asarray(res1[dq_id], np.float32).reshape(qsh) * scale
    dk = np.asarray(res1[dk_id], np.float32).reshape(ksh)
    dv = np.asarray(res1[dv_id], np.float32).reshape(ksh)
    for name, g, gr in (("dQ", dq, dq_ref), ("dK", dk, dk_ref), ("dV", dv, dv_ref)):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.03, f"{name} rel-err {rel}"


def _has_nondet_warning(rep):
    return any(w.get("code") == "nondeterministic_reduction" for w in (rep.get("warnings") or []))


@pytest.mark.parametrize("cfg", DETERMINISTIC, ids=[c["label"] for c in DETERMINISTIC])
def test_flash_bwd_sm100_nondeterministic_warns_but_passes(cfg):
    # The SAME shape WITHOUT deterministic: the cross-CTA dQ reduce-adds are atomic + commutative,
    # so the checker's commutative-reduce rule passes them (race-free) but emits a
    # `nondeterministic_reduction` warning (float reduce is order-dependent -> not bit-reproducible).
    # The deterministic path removes the warning (HB-ordered) — that's the distinguishing fact.
    kernel, _, _ = _run_det(cfg, deterministic=False)
    assert nr.trace(kernel)
    rep = nr.check_protocol(kernel)
    assert rep["status"] == "Passed", f"commutative reduce is race-free, got {rep['status']}"
    assert _has_nondet_warning(rep), "non-deterministic float reduce must warn"


GQA_DET = [
    {
        "num_head": 4,
        "num_head_kv": 2,
        "head_dim": 64,
        "seqlen": 128,
        "label": "det_gqa_h4kv2_s128_hd64",
    }
]


@pytest.mark.parametrize("cfg", GQA_DET, ids=[c["label"] for c in GQA_DET])
def test_flash_bwd_sm100_deterministic_gqa(cfg):
    # POSITIVE (GQA dKV): the G q-heads reduce-add into the shared kv-head's dK/dV; with
    # deterministic, the dK/dV semaphore path is Passed + bit-reproducible + value-correct.
    Hq, Hkv, D, S = cfg["num_head"], cfg["num_head_kv"], cfg["head_dim"], cfg["seqlen"]
    G = Hq // Hkv
    scale = 1.0 / np.sqrt(D)
    rng = np.random.default_rng(0)
    q = (rng.normal(size=(S, Hq, D)) * 0.3).astype(np.float32)
    do = (rng.normal(size=(S, Hq, D)) * 0.3).astype(np.float32)
    kk = (rng.normal(size=(S, Hkv, D)) * 0.3).astype(np.float32)
    vv = (rng.normal(size=(S, Hkv, D)) * 0.3).astype(np.float32)
    lse = np.zeros((S, Hq), np.float32)
    dpsum = np.zeros((S, Hq), np.float32)
    dq_ref = np.zeros((S, Hq, D), np.float32)
    dk_ref = np.zeros((S, Hkv, D), np.float32)
    dv_ref = np.zeros((S, Hkv, D), np.float32)
    for h in range(Hq):
        kv = h // G
        dq, dk, dv, o, l = oracle.fa_bwd_from_qkv(q[:, h], kk[:, kv], vv[:, kv], do[:, h], scale)
        lse[:, h] = l * LOG2_E
        dpsum[:, h] = (do[:, h] * o).sum(-1)
        dq_ref[:, h] = dq
        dk_ref[:, kv] += dk
        dv_ref[:, kv] += dv
    cfg_obj = FlashBwdSm100Config(
        batch=1,
        num_head=Hq,
        num_head_kv=Hkv,
        seqlen_q=S,
        seqlen_k=S,
        head_dim=D,
        deterministic=True,
    )
    kernel = build_flash_bwd_sm100(cfg_obj)
    qg, kg, vg, dog, lseg, dpsg, dqg, dkg, dvg = kernel.args[:9]
    inputs = {qg: bf(q), kg: bf(kk), vg: bf(vv), dog: bf(do), lseg: lse, dpsg: dpsum}
    assert nr.trace(kernel)
    rep = nr.check_protocol(kernel)
    assert rep["status"] == "Passed", f"deterministic GQA must be Passed, got {rep['status']}"
    res1 = nr.interpret(kernel, inputs)
    res2 = nr.interpret(kernel, inputs)
    for gid in (dqg.id, dkg.id, dvg.id):
        assert np.array_equal(np.asarray(res1[gid]), np.asarray(res2[gid])), "GQA not bit-identical"
    dq = np.asarray(res1[dqg.id], np.float32).reshape(S, Hq, D) * scale
    dk = np.asarray(res1[dkg.id], np.float32).reshape(S, Hkv, D) * scale
    dv = np.asarray(res1[dvg.id], np.float32).reshape(S, Hkv, D)
    for name, g, gr in (("dQ", dq, dq_ref), ("dK", dk, dk_ref), ("dV", dv, dv_ref)):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.03, f"{name} rel-err {rel}"


# ============== Cross-config combination matrix (every gated combo is now cleared) ==============
# Each tuple: (Hq, Hkv, Sq, Sk, Dq, Dv, causal, deterministic). Covers the COMBINATIONS that the
# per-feature tests above exercise in isolation — causal×2cta×multi-Q-tile, GQA×causal, det×GQA,
# det×2cta, hd192×{causal,gqa,det}×multi-Q-tile, varlen handled separately. Locks in the matrix.
CROSS_CONFIG = [
    (2, 2, 384, 384, 64, 64, True, False),  # 1-CTA hd64 causal n_mb=3
    (4, 2, 256, 256, 64, 64, True, False),  # 1-CTA hd64 GQA causal
    (2, 2, 256, 256, 64, 64, False, True),  # 1-CTA hd64 deterministic n_mb=2
    (2, 2, 512, 512, 128, 128, True, False),  # 2-CTA hd128 causal n_mb=4
    (4, 2, 256, 256, 128, 128, True, False),  # 2-CTA hd128 GQA causal
    (4, 2, 128, 256, 128, 128, False, True),  # 2-CTA hd128 deterministic GQA
    (2, 2, 256, 256, 128, 128, False, True),  # 2-CTA hd128 deterministic n_mb=2
    (2, 2, 256, 256, 192, 128, True, False),  # 2-CTA hd192 causal n_mb=2
    (4, 2, 384, 256, 192, 128, False, False),  # 2-CTA hd192 GQA n_mb=3
    (2, 2, 128, 256, 192, 128, False, True),  # 2-CTA hd192 deterministic
    (2, 2, 1024, 512, 128, 128, False, False),  # 2-CTA hd128 large n_mb=8
]


@pytest.mark.parametrize(
    "cfg",
    CROSS_CONFIG,
    ids=[
        f"h{c[0]}kv{c[1]}_sq{c[2]}sk{c[3]}_hd{c[4]}_{'c' if c[6] else ''}{'d' if c[7] else ''}"
        for c in CROSS_CONFIG
    ],
)
def test_flash_bwd_sm100_cross_config(cfg):
    Hq, Hkv, Sq, Sk, Dq, Dv, causal, det = cfg
    G = Hq // Hkv
    scale = 1.0 / np.sqrt(Dq)
    rng = np.random.default_rng(7)
    q = (rng.normal(size=(Sq, Hq, Dq)) * 0.3).astype(np.float32)
    do = (rng.normal(size=(Sq, Hq, Dv)) * 0.3).astype(np.float32)
    kk = (rng.normal(size=(Sk, Hkv, Dq)) * 0.3).astype(np.float32)
    vv = (rng.normal(size=(Sk, Hkv, Dv)) * 0.3).astype(np.float32)
    lse = np.zeros((Sq, Hq), np.float32)
    dps = np.zeros((Sq, Hq), np.float32)
    dqr = np.zeros((Sq, Hq, Dq), np.float32)
    dkr = np.zeros((Sk, Hkv, Dq), np.float32)
    dvr = np.zeros((Sk, Hkv, Dv), np.float32)
    for h in range(Hq):
        kv = h // G
        dq, dk, dv, o, l = oracle.fa_bwd_from_qkv(
            q[:, h], kk[:, kv], vv[:, kv], do[:, h], scale, causal=causal
        )
        lse[:, h] = l * LOG2_E
        dps[:, h] = (do[:, h] * o).sum(-1)
        dqr[:, h] = dq
        dkr[:, kv] += dk
        dvr[:, kv] += dv
    cfgo = FlashBwdSm100Config(
        batch=1,
        num_head=Hq,
        num_head_kv=Hkv,
        seqlen_q=Sq,
        seqlen_k=Sk,
        head_dim=Dq,
        head_dim_v=Dv,
        is_causal=causal,
        deterministic=det,
    )
    kernel = build_flash_bwd_sm100(cfgo)
    a = kernel.args
    assert nr.trace(kernel)
    assert nr.check_protocol(kernel)["status"] == "Passed"
    ins = {a[0]: bf(q), a[1]: bf(kk), a[2]: bf(vv), a[3]: bf(do), a[4]: lse, a[5]: dps}
    for x in a[9:]:
        ins[x] = np.zeros(tuple(x.shape), np.int32)  # cu_seqlens / deterministic semaphores
    res = nr.interpret(kernel, ins)
    dks = scale if Hq != Hkv else 1.0  # GQA dKaccum unscaled; MHA dK pre-scaled in epilogue
    dq = np.asarray(res[a[6].id], np.float32).reshape(dqr.shape) * scale
    dk = np.asarray(res[a[7].id], np.float32).reshape(dkr.shape) * dks
    dv = np.asarray(res[a[8].id], np.float32).reshape(dvr.shape)
    for name, g, gr in (("dQ", dq, dqr), ("dK", dk, dkr), ("dV", dv, dvr)):
        rel = np.abs(g - gr).max() / (np.abs(gr).max() + 1e-6)
        assert rel < 0.02, f"{name} rel-err {rel}"
