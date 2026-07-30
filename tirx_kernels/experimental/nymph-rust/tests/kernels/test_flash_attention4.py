import numpy as np
import nymph_rs as nr
import pytest
from nymph_rs.kernels.flash_attention4 import CONFIGS, FlashAttention4Config, build_flash_attention4


def _cfg(entry, **kwargs):
    values = dict(
        batch_size=entry["batch_size"],
        seq_len=entry["seq_len"],
        num_qo_heads=entry["num_qo_heads"],
        num_kv_heads=entry["num_kv_heads"],
        head_dim=entry["head_dim"],
        is_causal=entry["is_causal"],
    )
    values.update(kwargs)
    return FlashAttention4Config(**values)


@pytest.mark.parametrize("entry", CONFIGS, ids=[entry["label"] for entry in CONFIGS])
def test_flash_attention4_builds_all_bench_configs(entry):
    kernel = build_flash_attention4(_cfg(entry, launch_shape=(1,)))
    assert len(kernel.args) == 4


def test_flash_attention4_only_mma_operands_use_smem_layout():
    kernel = build_flash_attention4(FlashAttention4Config(seq_len=1024, launch_shape=(1,)))

    def walk(stmts):
        for stmt in stmts:
            yield stmt
            if stmt.kind in {"if", "for_loop", "loop", "for_each_task", "scheduler_impl"}:
                yield from walk(stmt.body)

    swizzled = {
        stmt.tensor.id
        for stmt in kernel.body
        if stmt.kind == "tensor_def" and stmt.tensor.layout is not None
    }
    mma_operands = set()
    for stmt in walk(kernel.body):
        if stmt.kind != "tcgen05_mma":
            continue
        if stmt.a.kind == "smem":
            mma_operands.add(stmt.a.tile.tensor.id)
        mma_operands.add(stmt.b.tensor.id)

    assert swizzled
    assert swizzled <= mma_operands


# Resident protocol tier: every kv-head config at seq <= 2048 plus one s4096
# representative (~2 min total). Per-shape cost scales x4 per seq doubling
# (value/trace/check alike), so the s4096/s8192 tiers are NOT resident — the
# full 16-config sweep was verified one-off on the per-warp model (all
# Passed; s1024 ~5 s, s2048 ~17 s, s4096 ~69 s, s8192 ~5 min each).
_PROTOCOL_TIER = [c for c in CONFIGS if c["seq_len"] <= 2048] + [
    next(c for c in CONFIGS if c["seq_len"] == 4096)
]


# The s4096 representative is minutes on its own; the rest of the tier is
# seconds, so only it carries `slow`.
_TIER_PARAMS = [
    pytest.param(c, marks=pytest.mark.slow) if c["seq_len"] >= 4096 else c for c in _PROTOCOL_TIER
]


@pytest.mark.parametrize("entry", _TIER_PARAMS, ids=[c["label"] for c in _PROTOCOL_TIER])
def test_flash_attention4_protocol_resident_tier(entry):
    kernel = build_flash_attention4(_cfg(entry, launch_shape=(1,)))
    report = nr.check_protocol(kernel)
    assert report["status"] == "Passed"


def test_flash_attention4_protocol_smoke_min_causal():
    kernel = build_flash_attention4(
        FlashAttention4Config(seq_len=1024, num_kv_heads=4, is_causal=True, launch_shape=(1,))
    )
    report = nr.check_protocol(kernel)
    assert report["status"] == "Passed"


def _reference_attention(q, k, v, cfg: FlashAttention4Config):
    gqa_ratio = cfg.num_qo_heads // cfg.num_kv_heads
    q_ref = q.astype(np.float32).reshape(1, cfg.seq_len, cfg.num_kv_heads, gqa_ratio, cfg.head_dim)
    q_ref = q_ref.transpose(0, 2, 3, 1, 4)
    k_ref = k.astype(np.float32).transpose(0, 2, 1, 3)
    v_ref = v.astype(np.float32).transpose(0, 2, 1, 3)
    scores = np.einsum("bhgmd,bhnd->bhgmn", q_ref, k_ref) / np.sqrt(cfg.head_dim, dtype=np.float32)
    if cfg.is_causal:
        rows = np.arange(cfg.seq_len)[:, None]
        cols = np.arange(cfg.seq_len)[None, :]
        scores = np.where(cols > rows, -np.inf, scores)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    ref = np.einsum("bhgmn,bhnd->bhgmd", probs, v_ref)
    ref = ref.transpose(0, 3, 1, 2, 4).reshape(1, cfg.seq_len, cfg.num_qo_heads, cfg.head_dim)
    return ref.astype(np.float16).astype(np.float32)


@pytest.mark.parametrize("is_causal", [False, True], ids=["noncausal", "causal"])
def test_flash_attention4_value_matches_numpy_reference_small(is_causal):
    cfg = FlashAttention4Config(seq_len=128, num_kv_heads=4, is_causal=is_causal, launch_shape=(1,))
    kernel = build_flash_attention4(cfg)
    q_t, k_t, v_t, o_t = kernel.args
    rng = np.random.default_rng(2)
    q = (rng.normal(size=(1, cfg.seq_len, cfg.num_qo_heads, cfg.head_dim)) * 0.25).astype(
        np.float16
    )
    k = (rng.normal(size=(1, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim)) * 0.25).astype(
        np.float16
    )
    v = (rng.normal(size=(1, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim)) * 0.25).astype(
        np.float16
    )

    out = np.asarray(nr.interpret(kernel, {q_t: q, k_t: k, v_t: v})[o_t.id], dtype=np.float32)
    ref = _reference_attention(q, k, v, cfg)

    np.testing.assert_allclose(out, ref, atol=7e-4, rtol=3e-2)


def test_flash_attention4_value_matches_numpy_reference_causal_multiblock_gqa1():
    cfg = FlashAttention4Config(
        seq_len=256, num_qo_heads=32, num_kv_heads=32, is_causal=True, launch_shape=(1,)
    )
    kernel = build_flash_attention4(cfg)
    q_t, k_t, v_t, o_t = kernel.args
    rng = np.random.default_rng(3)
    q = (rng.normal(size=(1, cfg.seq_len, cfg.num_qo_heads, cfg.head_dim)) * 0.25).astype(
        np.float16
    )
    k = (rng.normal(size=(1, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim)) * 0.25).astype(
        np.float16
    )
    v = (rng.normal(size=(1, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim)) * 0.25).astype(
        np.float16
    )

    out = np.asarray(nr.interpret(kernel, {q_t: q, k_t: k, v_t: v})[o_t.id], dtype=np.float32)
    ref = _reference_attention(q, k, v, cfg)

    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, atol=7e-4, rtol=3e-2)
