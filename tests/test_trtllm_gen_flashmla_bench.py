from __future__ import annotations

from unittest import SkipTest, mock

import pytest
import torch

from tirx_kernels.flashmla._trtllm_gen_bench import (
    PAGE_SIZE,
    _reraise_trtllm_gen_probe_error,
    identity_paged_kv_shapes,
    identity_slot_to_paged_coords,
    is_sm100_device,
    make_identity_paged_kv_buffer,
    max_abs_rel_error,
    prepare_trtllm_block_tables,
    probe_trtllm_gen_launch,
    trtllm_gen_config_compatible,
    trtllm_gen_incompatible_reason,
    validate_trtllm_sparse_indices,
)


def test_trtllm_gen_config_compatibility_matrix() -> None:
    assert trtllm_gen_config_compatible(
        {"d_qk": 576, "d_v": 512, "topk": 512, "h_q": 64, "h_kv": 1}
    )
    assert trtllm_gen_config_compatible(
        {"d_qk": 576, "d_v": 512, "topk": 2048, "h_q": 128, "h_kv": 1}
    )
    assert "d_qk=512" in trtllm_gen_incompatible_reason(
        {"d_qk": 512, "d_v": 512, "topk": 512, "h_q": 64, "h_kv": 1}
    )
    assert "topk=1280" in trtllm_gen_incompatible_reason(
        {"d_qk": 576, "d_v": 512, "topk": 1280, "h_q": 128, "h_kv": 1}
    )
    assert "inject_invalid_indices=True" in trtllm_gen_incompatible_reason(
        {"d_qk": 576, "d_v": 512, "topk": 512, "h_q": 64, "h_kv": 1, "inject_invalid_indices": True}
    )
    assert "have_topk_length=True" in trtllm_gen_incompatible_reason(
        {"d_qk": 576, "d_v": 512, "topk": 512, "h_q": 64, "h_kv": 1, "have_topk_length": True}
    )


def test_trtllm_gen_probe_error_classification() -> None:
    with pytest.raises(SkipTest, match="tactic unavailable"):
        _reraise_trtllm_gen_probe_error(RuntimeError("no valid tactic for this shape"))

    with pytest.raises(ValueError, match="bad indices"):
        _reraise_trtllm_gen_probe_error(ValueError("bad indices"))

    with pytest.raises(RuntimeError, match="cuda illegal memory access"):
        _reraise_trtllm_gen_probe_error(RuntimeError("cuda illegal memory access"))


def test_probe_trtllm_gen_launch_wraps_tactic_runtime_error() -> None:
    def _launch() -> None:
        raise RuntimeError("failed to find a valid tactic")

    with pytest.raises(SkipTest, match="tactic unavailable"):
        probe_trtllm_gen_launch(_launch)


def test_probe_trtllm_gen_launch_propagates_value_error() -> None:
    def _launch() -> None:
        raise ValueError("wiring error")

    with pytest.raises(ValueError, match="wiring error"):
        probe_trtllm_gen_launch(_launch)


def test_identity_paged_kv_shapes_and_flat_view() -> None:
    num_pages, padded_tokens = identity_paged_kv_shapes(100, page_size=PAGE_SIZE, d_qk=576)
    assert num_pages == 2
    assert padded_tokens == 128

    kv_paged, kv_flat = make_identity_paged_kv_buffer(
        100, 1, 576, device="cpu", dtype=torch.bfloat16
    )
    assert kv_paged.shape == (2, 1, PAGE_SIZE, 576)
    assert kv_flat.shape == (100, 1, 576)

    kv_flat[0, 0, 3] = torch.tensor(1.25, dtype=torch.bfloat16)
    kv_flat[99, 0, 7] = torch.tensor(-2.5, dtype=torch.bfloat16)
    page, off = identity_slot_to_paged_coords(99)
    assert page == 1 and off == 35
    assert kv_paged[0, 0, 0, 3].item() == pytest.approx(1.25, rel=0, abs=1e-3)
    assert kv_paged[page, 0, off, 7].item() == pytest.approx(-2.5, rel=0, abs=1e-3)


def test_identity_slot_mapping_is_linear() -> None:
    s_kv = 250
    for slot in (0, 63, 64, 127, 249):
        page, off = identity_slot_to_paged_coords(slot)
        assert page * PAGE_SIZE + off == slot
        assert page < identity_paged_kv_shapes(s_kv)[0]


def test_validate_trtllm_sparse_indices_rejects_invalid_entries() -> None:
    valid = torch.tensor([[[10, 20, 30, 40]]], dtype=torch.int32)
    prepare_trtllm_block_tables(valid, s_kv=256, topk=4)

    invalid = torch.tensor([[[10, 20, -1, 300]]], dtype=torch.int32)
    with pytest.raises(ValueError, match="invalid values"):
        validate_trtllm_sparse_indices(invalid, s_kv=256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trtllm_gen_backend_does_not_fall_back_to_other_backends() -> None:
    from tirx_kernels.flashmla._trtllm_gen_bench import (
        flashinfer_trtllm_decode_available,
        prepare_trtllm_gen_launch,
    )
    from tirx_kernels.flashmla.sparse_prefill_head64_phase1 import prepare_data

    if not is_sm100_device():
        pytest.skip("SM100 required")
    if not flashinfer_trtllm_decode_available():
        pytest.skip("FlashInfer unavailable")

    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    case = prepare_data(
        label="backend_probe",
        s_q=2,
        s_kv=512,
        topk=512,
        d_qk=576,
        h_q=64,
        have_attn_sink=False,
        seed=0,
    )
    prep = prepare_trtllm_gen_launch(case, probe=False)
    launch_kwargs = dict(
        query=prep["q_trtllm"],
        kv_cache=prep["kv_paged"],
        workspace_buffer=prep["workspace"],
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=prep["block_tables"],
        seq_lens=torch.full((2,), 512, dtype=torch.int32, device=case["q"].device),
        max_seq_len=512,
        sparse_mla_top_k=512,
        out=prep["out"],
        bmm1_scale=case["sm_scale"],
        bmm2_scale=1.0,
        backend="trtllm-gen",
    )

    with (
        mock.patch(
            "flashinfer.mla._core.xqa_batch_decode_with_kv_cache_mla",
            side_effect=AssertionError("xqa fallback"),
        ),
        mock.patch(
            "flashinfer.cute_dsl.attention.cute_dsl_mla_decode",
            side_effect=AssertionError("cute-dsl fallback"),
        ),
    ):
        trtllm_batch_decode_with_kv_cache_mla(**launch_kwargs)
    torch.cuda.synchronize()


def _assert_sparse_prefill_refs_match_tirx(
    *, tirx_out: torch.Tensor, flashmla_out: torch.Tensor, trtllm_out: torch.Tensor, label: str
) -> None:
    rtol, atol = 0.02, 0.01
    pairs = (("flashmla", flashmla_out), ("trtllm_gen", trtllm_out))
    for name, out in pairs:
        max_abs, max_rel = max_abs_rel_error(out, tirx_out)
        print(f"{label} {name} vs tirx: max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
        torch.testing.assert_close(out, tirx_out, rtol=rtol, atol=atol)

    flash_trtllm_abs, flash_trtllm_rel = max_abs_rel_error(trtllm_out, flashmla_out)
    print(
        f"{label} trtllm_gen vs flashmla: "
        f"max_abs={flash_trtllm_abs:.6g} max_rel={flash_trtllm_rel:.6g}"
    )
    torch.testing.assert_close(trtllm_out, flashmla_out, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not is_sm100_device(), reason="SM100 required")
def test_trtllm_gen_matches_tirx_and_flashmla_head64() -> None:
    from tirx_kernels.flashmla._flashmla_bench import run_flashmla_sparse_prefill
    from tirx_kernels.flashmla._trtllm_gen_bench import run_trtllm_gen_sparse_prefill
    from tirx_kernels.flashmla.sparse_prefill_head64_phase1 import (
        _tirx_args,
        get_kernel,
        prepare_data,
    )
    from tirx_kernels.runner import compile_kernel

    kwargs = dict(
        label="correctness_hq64",
        s_q=32,
        s_kv=8192,
        topk=512,
        d_qk=576,
        h_q=64,
        have_attn_sink=True,
        seed=3,
    )
    case = prepare_data(**kwargs)
    ex = compile_kernel(get_kernel(**kwargs))
    ex(*_tirx_args(case))
    torch.cuda.synchronize()
    tirx_out = case["out"].clone()

    flashmla_out = run_flashmla_sparse_prefill(case)
    trtllm_out = run_trtllm_gen_sparse_prefill(case)
    _assert_sparse_prefill_refs_match_tirx(
        tirx_out=tirx_out, flashmla_out=flashmla_out, trtllm_out=trtllm_out, label="hq64/topk512"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not is_sm100_device(), reason="SM100 required")
def test_trtllm_gen_matches_tirx_and_flashmla_head128() -> None:
    from tirx_kernels.flashmla._flashmla_bench import run_flashmla_sparse_prefill
    from tirx_kernels.flashmla._trtllm_gen_bench import run_trtllm_gen_sparse_prefill
    from tirx_kernels.flashmla.sparse_prefill_head128_phase1 import (
        _tirx_args,
        get_kernel,
        prepare_data,
    )
    from tirx_kernels.runner import compile_kernel

    kwargs = dict(
        label="correctness_hq128",
        s_q=32,
        s_kv=8192,
        topk=2048,
        d_qk=576,
        h_q=128,
        have_attn_sink=True,
        seed=4,
    )
    case = prepare_data(**kwargs)
    ex = compile_kernel(get_kernel(**kwargs))
    ex(*_tirx_args(case))
    torch.cuda.synchronize()
    tirx_out = case["out"].clone()

    flashmla_out = run_flashmla_sparse_prefill(case)
    trtllm_out = run_trtllm_gen_sparse_prefill(case)
    _assert_sparse_prefill_refs_match_tirx(
        tirx_out=tirx_out, flashmla_out=flashmla_out, trtllm_out=trtllm_out, label="hq128/topk2048"
    )
