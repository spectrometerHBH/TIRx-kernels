"""Compile-gate: kernels emitted by the Rust codegen must compile via
``tvm.compile(tir_pipeline="tirx")``. Compile-only (no GPU run).

The TVMScript parser reads the prim_func's source via ``inspect.getsourcelines``,
so the emitted source must live in a real .py file on disk (exec-from-string
fails) — hence the tmp_path round-trip.
"""

import importlib.util

import nymph_rs as nr
import pytest
from nymph_rs.kernels import build_bootstrap_gemm, build_fp16_bf16_gemm, build_nvfp4_gemm
from nymph_rs.kernels.gdn_prefill import GdnPrefillConfig, build_gdn_prefill

tvm = pytest.importorskip("tvm", reason="tvm not importable in this environment")

pytestmark = pytest.mark.codegen

BUILDERS = {
    "bootstrap_gemm": build_bootstrap_gemm,
    "fp16_bf16_gemm": build_fp16_bf16_gemm,
    "nvfp4_gemm": build_nvfp4_gemm,
    "gdn_prefill": lambda: build_gdn_prefill(GdnPrefillConfig(num_seqs=1, seqlen=128)),
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
