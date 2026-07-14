from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest import SkipTest

import torch

PAGE_SIZE = 64
D_QK = 576
D_V = 512
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
SUPPORTED_TOPKS = frozenset({512, 1024, 2048})
SUPPORTED_H_Q = frozenset({64, 128})
TRTLLM_GEN_WORKSPACE_BYTES = 128 * 1024 * 1024

_TACTIC_UNAVAILABLE_MARKERS = (
    "no valid tactic",
    "no available tactic",
    "tactic unavailable",
    "failed to find a valid tactic",
    "kernel not found",
)


def _cfg_field(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def is_sm100_device(device: torch.device | str | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)
    return major == 10 and minor == 0


def flashinfer_trtllm_decode_available() -> bool:
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla  # noqa: F401

        return True
    except ImportError:
        return False


def trtllm_gen_incompatible_reason(cfg: Any) -> str | None:
    d_qk = _cfg_field(cfg, "d_qk")
    d_v = _cfg_field(cfg, "d_v", D_V)
    topk = _cfg_field(cfg, "topk")
    h_q = _cfg_field(cfg, "h_q")
    h_kv = _cfg_field(cfg, "h_kv", 1)
    if d_qk != D_QK:
        return f"d_qk={d_qk} (TRTLLM-Gen sparse MLA requires {D_QK})"
    if d_v != D_V:
        return f"d_v={d_v} (TRTLLM-Gen sparse MLA requires {D_V})"
    if topk not in SUPPORTED_TOPKS:
        return f"topk={topk} (supported topk: {sorted(SUPPORTED_TOPKS)})"
    if h_q not in SUPPORTED_H_Q:
        return f"h_q={h_q} (supported h_q: {sorted(SUPPORTED_H_Q)})"
    if h_kv != 1:
        return f"h_kv={h_kv} (TRTLLM-Gen sparse MLA requires h_kv == 1)"
    if _cfg_field(cfg, "have_topk_length", False):
        return "have_topk_length=True is outside the TRTLLM-Gen sparse contract"
    if _cfg_field(cfg, "inject_invalid_indices", False):
        return "inject_invalid_indices=True is outside the TRTLLM-Gen sparse contract"
    return None


def trtllm_gen_config_compatible(cfg: Any) -> bool:
    return trtllm_gen_incompatible_reason(cfg) is None


def identity_paged_kv_shapes(
    s_kv: int, *, page_size: int = PAGE_SIZE, d_qk: int = D_QK
) -> tuple[int, int]:
    num_pages = (s_kv + page_size - 1) // page_size
    padded_tokens = num_pages * page_size
    return num_pages, padded_tokens


def make_identity_paged_kv_buffer(
    s_kv: int,
    h_kv: int,
    d_qk: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    page_size: int = PAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_pages, _ = identity_paged_kv_shapes(s_kv, page_size=page_size, d_qk=d_qk)
    kv_paged = torch.zeros((num_pages, 1, page_size, d_qk), device=device, dtype=dtype)
    kv_flat = kv_paged.reshape(-1, h_kv, d_qk)[:s_kv]
    return kv_paged, kv_flat


def identity_slot_to_paged_coords(slot: int, *, page_size: int = PAGE_SIZE) -> tuple[int, int]:
    return divmod(slot, page_size)


def validate_trtllm_sparse_indices(indices: torch.Tensor, *, s_kv: int) -> None:
    """Reject inputs TRTLLM-Gen cannot safely mask with -1."""
    flat = indices.reshape(-1)
    invalid = (flat < 0) | (flat >= s_kv)
    if torch.any(invalid):
        bad = flat[invalid][:8].tolist()
        raise ValueError(
            "TRTLLM-Gen sparse MLA requires all block_tables entries in "
            f"[0, s_kv={s_kv}); got invalid values including {bad}"
        )


def prepare_trtllm_block_tables(indices: torch.Tensor, *, s_kv: int, topk: int) -> torch.Tensor:
    block_tables = indices[:, :, :topk].contiguous()
    validate_trtllm_sparse_indices(block_tables, s_kv=s_kv)
    return block_tables


def _require_trtllm_gen_runtime(device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for TRTLLM-Gen sparse MLA baseline")
    if not is_sm100_device(device):
        major, minor = torch.cuda.get_device_capability(device)
        raise SkipTest(f"TRTLLM-Gen baseline requires SM100 (10.0), got SM{major}{minor}")
    if not flashinfer_trtllm_decode_available():
        raise SkipTest("FlashInfer decode.trtllm_batch_decode_with_kv_cache_mla is unavailable")


def _import_trtllm_decode():
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    except ImportError as exc:
        raise SkipTest(f"FlashInfer is unavailable: {exc}") from exc
    return trtllm_batch_decode_with_kv_cache_mla


def _reraise_trtllm_gen_probe_error(exc: BaseException) -> None:
    if isinstance(exc, SkipTest):
        raise exc
    if isinstance(exc, ValueError | TypeError | AssertionError):
        raise exc
    msg = str(exc).lower()
    if isinstance(exc, RuntimeError) and any(
        marker in msg for marker in _TACTIC_UNAVAILABLE_MARKERS
    ):
        raise SkipTest(f"TRTLLM-Gen sparse MLA tactic unavailable: {exc}") from exc
    raise exc


def probe_trtllm_gen_launch(launch: Callable[[], None]) -> None:
    try:
        launch()
        torch.cuda.synchronize()
    except Exception as exc:
        _reraise_trtllm_gen_probe_error(exc)


def prepare_trtllm_gen_launch(case: dict[str, Any], *, probe: bool = True) -> dict[str, Any]:
    cfg = case["config"]
    reason = trtllm_gen_incompatible_reason(cfg)
    if reason is not None:
        raise ValueError(reason)

    device = case["q"].device
    _require_trtllm_gen_runtime(device)
    decode = _import_trtllm_decode()

    s_q = _cfg_field(cfg, "s_q")
    s_kv = _cfg_field(cfg, "s_kv")
    topk = _cfg_field(cfg, "topk")
    h_q = _cfg_field(cfg, "h_q")
    h_kv = _cfg_field(cfg, "h_kv", 1)
    d_qk = _cfg_field(cfg, "d_qk", D_QK)
    have_attn_sink = _cfg_field(cfg, "have_attn_sink", False)

    kv_paged, kv_flat = make_identity_paged_kv_buffer(
        s_kv, h_kv, d_qk, device=device, dtype=case["kv"].dtype
    )
    kv_flat.copy_(case["kv"])

    q_trtllm = case["q"].view(s_q, 1, h_q, d_qk)
    block_tables = prepare_trtllm_block_tables(case["indices"], s_kv=s_kv, topk=topk)
    seq_lens = torch.full((s_q,), s_kv, dtype=torch.int32, device=device)
    workspace = torch.zeros(TRTLLM_GEN_WORKSPACE_BYTES, dtype=torch.int8, device=device)
    out = torch.empty((s_q, 1, h_q, D_V), dtype=torch.bfloat16, device=device)
    sinks = case["attn_sink"] if have_attn_sink else None

    common_kwargs = dict(
        query=q_trtllm,
        kv_cache=kv_paged,
        workspace_buffer=workspace,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=s_kv,
        sparse_mla_top_k=topk,
        out=out,
        bmm1_scale=case["sm_scale"],
        bmm2_scale=1.0,
        sinks=sinks,
        backend="trtllm-gen",
    )

    def launch() -> None:
        decode(**common_kwargs)

    if probe:
        probe_trtllm_gen_launch(launch)

    return {
        "launch": launch,
        "out": out,
        "workspace": workspace,
        "kv_paged": kv_paged,
        "kv_flat": kv_flat,
        "block_tables": block_tables,
        "q_trtllm": q_trtllm,
    }


def run_trtllm_gen_sparse_prefill(case: dict[str, Any]) -> torch.Tensor:
    prep = prepare_trtllm_gen_launch(case)
    prep["launch"]()
    torch.cuda.synchronize()
    return prep["out"].squeeze(1)


def trtllm_gen_reference_builder(case: dict[str, Any]) -> Callable[[], None]:
    prep = prepare_trtllm_gen_launch(case)
    return prep["launch"]


def flashmla_bench_references(
    case: dict[str, Any],
) -> dict[str, Callable[[], Callable[[], None] | None]]:
    from tirx_kernels.flashmla._flashmla_bench import flashmla_reference_builder

    def _flashmla_ref() -> Callable[[], None]:
        run = flashmla_reference_builder()
        return lambda: run(case)

    references: dict[str, Callable[[], Callable[[], None] | None]] = {"flashmla": _flashmla_ref}
    if trtllm_gen_config_compatible(case["config"]):
        references["trtllm_gen"] = lambda: trtllm_gen_reference_builder(case)
    return references


def max_abs_rel_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    abs_err = (actual_f - expected_f).abs()
    max_abs = float(abs_err.max().item())
    denom = expected_f.abs().clamp_min(1e-6)
    max_rel = float((abs_err / denom).max().item())
    return max_abs, max_rel


def bf16_allclose(
    actual: torch.Tensor, expected: torch.Tensor, *, rtol: float = 0.02, atol: float = 0.01
) -> bool:
    return torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol)


__all__ = [
    "D_QK",
    "D_V",
    "PAGE_SIZE",
    "SUPPORTED_TOPKS",
    "bf16_allclose",
    "flashinfer_trtllm_decode_available",
    "flashmla_bench_references",
    "identity_paged_kv_shapes",
    "identity_slot_to_paged_coords",
    "is_sm100_device",
    "make_identity_paged_kv_buffer",
    "max_abs_rel_error",
    "prepare_trtllm_block_tables",
    "prepare_trtllm_gen_launch",
    "probe_trtllm_gen_launch",
    "run_trtllm_gen_sparse_prefill",
    "trtllm_gen_config_compatible",
    "trtllm_gen_incompatible_reason",
    "trtllm_gen_reference_builder",
    "validate_trtllm_sparse_indices",
]
