"""Compile-gate: kernels emitted by the Rust codegen must compile via
``tvm.compile(tir_pipeline="tirx")``. Compile-only (no GPU run).

The TVMScript parser reads the prim_func's source via ``inspect.getsourcelines``,
so the emitted source must live in a real .py file on disk (exec-from-string
fails) — hence the tmp_path round-trip.
"""

import importlib.util

import pytest

tvm = pytest.importorskip("tvm", reason="tvm not importable in this environment")

import nymph_rs as nr
from nymph_rs.kernels import (
    build_bootstrap_gemm,
    build_fp16_bf16_gemm,
    build_nvfp4_gemm,
)

BUILDERS = {
    "bootstrap_gemm": build_bootstrap_gemm,
    "fp16_bf16_gemm": build_fp16_bf16_gemm,
    "nvfp4_gemm": build_nvfp4_gemm,
}


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
