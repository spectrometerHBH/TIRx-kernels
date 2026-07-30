"""Compile-gate: kernels emitted by the Rust codegen must compile via
``tvm.compile(tir_pipeline="tirx")``. Compile-only (no GPU run).

The TVMScript parser reads the prim_func's source via ``inspect.getsourcelines``,
so the emitted source must live in a real .py file on disk (exec-from-string
fails) — hence the tmp_path round-trip.
"""

import importlib.util

import nymph_rs as nr
import pytest
from nymph_rs.kernels import (
    Fp8BlockwiseGemmConfig,
    Fp16Bf16GemmConfig,
    build_bootstrap_gemm,
    build_fp8_blockwise_gemm,
    build_fp16_bf16_gemm,
    build_nvfp4_gemm,
)
from nymph_rs.kernels.gdn_prefill import GdnPrefillConfig, build_gdn_prefill

tvm = pytest.importorskip("tvm", reason="tvm not importable in this environment")

pytestmark = pytest.mark.codegen

BUILDERS = {
    "bootstrap_gemm": build_bootstrap_gemm,
    "fp8_blockwise_gemm": lambda: build_fp8_blockwise_gemm(Fp8BlockwiseGemmConfig()),
    "fp8_blockwise_gemm_swap_odd_epi": lambda: build_fp8_blockwise_gemm(
        Fp8BlockwiseGemmConfig(m=2400, n=512, k=512, launch_shape=(2,))
    ),
    "nvfp4_gemm": build_nvfp4_gemm,
    "gdn_prefill": lambda: build_gdn_prefill(GdnPrefillConfig(num_seqs=1, seqlen=128)),
    **{
        f"{dtype_name}_gemm_{size}": (
            lambda dtype=dtype, size=size: build_fp16_bf16_gemm(
                Fp16Bf16GemmConfig(m=size, n=size, k=size, dtype=dtype)
            )
        )
        for dtype_name, dtype in (("fp16", nr.DType.F16), ("bf16", nr.DType.BF16))
        for size in (1024, 4096)
    },
}


def test_empty_if_body_emits_valid_python(tmp_path):
    # An empty `If` body used to render `if warp_id == 0:` with no indented
    # statement — invalid Python. The structured emitter fills empty blocks
    # with `pass` generically. (2-CTA cluster geometry: the tirx pipeline
    # requires a real cluster launch — every shipped kernel has one.)
    b = nr.IRBuilder(
        "empty_if", num_warps=4, smem_size_bytes=0, launch_shape=(2,), cluster_shape=(2,)
    )
    b.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.F32, shape=(1,))  # codegen requires an arg
    with b.if_warp(0):
        pass
    src = nr.kernel_to_tirx_source(b.build())
    assert "pass" in src
    mod_path = tmp_path / "emitted_empty_if.py"
    mod_path.write_text(src)
    spec = importlib.util.spec_from_file_location("emitted_empty_if", mod_path)
    emitted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(emitted)  # parse gate — a bare `if ...:` header would raise
    mod = tvm.IRModule({"main": emitted.main})
    tvm.compile(mod, tvm.target.Target("cuda"), tir_pipeline="tirx")


def test_two_explicit_u32_smem_layouts_remain_distinct_and_parse(tmp_path):
    b = nr.IRBuilder(
        "u32_smem_layouts", num_warps=4, smem_size_bytes=8192, launch_shape=(1,), cluster_shape=(1,)
    )
    b.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.U32, shape=(1,))
    layout = nr.SmemSwizzleLayout(nr.Swizzle.B32)
    b.tensor(
        space=nr.MemorySpace.SMEM, dtype=nr.DType.U32, shape=(128, 8), layout=layout, byte_offset=0
    )
    b.tensor(
        space=nr.MemorySpace.SMEM,
        dtype=nr.DType.U32,
        shape=(128, 8),
        layout=layout,
        byte_offset=4096,
    )

    src = nr.kernel_to_tirx_source(b.build())
    assert "ab_smem0_layout =" not in src
    assert "ab_smem1_layout =" not in src
    assert (
        'ab_smem0 = pool.alloc_tcgen05_mma_AB((128, 8), "uint32", '
        "swizzle_mode=SwizzleMode.SWIZZLE_32B_ATOM, align=1024)"
    ) in src
    assert (
        'ab_smem1 = pool.alloc_tcgen05_mma_AB((128, 8), "uint32", '
        "swizzle_mode=SwizzleMode.SWIZZLE_32B_ATOM, align=1024)"
    ) in src
    assert "task_smem" not in src
    assert "mma_shared_layout" not in src

    mod_path = tmp_path / "emitted_u32_smem_layouts.py"
    mod_path.write_text(src)
    spec = importlib.util.spec_from_file_location("emitted_u32_smem_layouts", mod_path)
    emitted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(emitted)
    assert emitted.main is not None


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_codegen_compiles(name, tmp_path):
    src = nr.kernel_to_tirx_source(BUILDERS[name]())
    mod_path = tmp_path / f"emitted_{name}.py"
    mod_path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"emitted_{name}", mod_path)
    emitted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(emitted)  # parse gate
    mod = tvm.IRModule({"main": emitted.main})
    tvm.compile(mod, tvm.target.Target("cuda"), tir_pipeline="tirx")


def test_single_issue_scope_negative():
    # Zero-inference guard rule: a hardware single-issue op (here: tma_load)
    # outside an explicit single-lane branch must be rejected at BUILD —
    # codegen never synthesizes the guard (it once wrapped these in an
    # inferred elect_sync). The elected form must pass.
    def build(with_elected: bool):
        b = nr.IRBuilder(
            "si_neg", num_warps=4, smem_size_bytes=8200, launch_shape=(1,), cluster_shape=(1,)
        )
        src_g = b.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.BF16, shape=(64, 64))
        dst_s = b.tensor(
            space=nr.MemorySpace.SMEM, dtype=nr.DType.BF16, shape=(64, 64), byte_offset=0
        )
        mbar = b.mbar(kind=nr.MBarKind.TMA, byte_offset=8192, stages=1)
        with b.if_warp(0):
            if with_elected:
                with b.if_elected():
                    b.mbarrier_arrive_expect_tx(mbar, bytes=64 * 64 * 2)
                    b.tma_load(
                        dst_s, src_g, mbar=mbar, coords=(0, 0), shape=(64, 64), gmem_shape=(64, 64)
                    )
            else:
                b.mbarrier_arrive_expect_tx(mbar, bytes=64 * 64 * 2)
                b.tma_load(
                    dst_s, src_g, mbar=mbar, coords=(0, 0), shape=(64, 64), gmem_shape=(64, 64)
                )
        return b.build()

    with pytest.raises(Exception, match="single_issue_scope"):
        build(with_elected=False)
    k = build(with_elected=True)
    src = nr.kernel_to_tirx_source(k)
    # The IR's own nested predicate is printed literally. Codegen must not
    # substitute either hardware spelling or move it out of the warp branch.
    assert src.count("if lane_id == 0:") == 1, src
    assert "if warp_id == 0:\n        if lane_id == 0:" in src
    assert "if T.ptx.elect_sync():" not in src
    assert "if T.cuda.thread_rank() == 0:" not in src
