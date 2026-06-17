"""Compile-gate: emit the bootstrap GEMM TVMScript from the Rust codegen and
verify it compiles via tvm.compile(tir_pipeline="tirx"). Compile-only (no GPU run).

The TVMScript parser reads the prim_func's source via inspect.getsourcelines, so
the emitted source must live in a real .py file on disk (exec-from-string fails)."""

import importlib.util
import os
import sys
import tempfile

import nymph_rs as nr
from nymph_rs.kernels import build_bootstrap_gemm

import tvm

src = nr.kernel_to_tirx_source(build_bootstrap_gemm())
print("=" * 70)
print(src)
print("=" * 70)

# Write to a real file and import it so inspect can find the source.
tmpdir = tempfile.mkdtemp(prefix="nymph_codegen_")
mod_path = os.path.join(tmpdir, "emitted_gemm.py")
with open(mod_path, "w") as f:
    f.write(src)

spec = importlib.util.spec_from_file_location("emitted_gemm", mod_path)
emitted = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(emitted)
except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"PARSE_FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

main = emitted.main

try:
    mod = tvm.IRModule({"main": main})
    mod = tvm.compile(mod, tvm.target.Target("cuda"), tir_pipeline="tirx")
    print("COMPILE_OK")
except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"COMPILE_FAILED: {type(e).__name__}: {e}")
    sys.exit(2)
