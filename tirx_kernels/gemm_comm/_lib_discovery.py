# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Discover the communication libraries GemmComm benchmarks lock and preload.

The installed pip packages are the single source of truth: there are no
environment variables or CLI flags to point elsewhere. Pin library versions
with pip (for example ``pip install nvidia-nccl-cu13==2.30.7``), not with
per-run configuration. Requiring exactly one resolved file per library keeps
the benchmark comparison from silently drifting between installations.
"""

from __future__ import annotations

import hashlib
import site
import sys
import tempfile
from pathlib import Path

# library name -> (candidate lib dirs relative to a site-packages root, glob)
_LIBRARY_PATTERNS = {
    "nccl": (("nvidia/nccl/lib",), "libnccl.so.*"),
    "cublas": (("nvidia/cu13/lib", "nvidia/cu12/lib", "nvidia/cublas/lib"), "libcublas.so.*"),
    "cublasmp": (
        ("nvidia/cublasmp/cu13/lib", "nvidia/cublasmp/cu12/lib", "nvidia/cublasmp/lib"),
        "libcublasmp.so.*",
    ),
    "nvshmem": (("nvidia/nvshmem/lib",), "libnvshmem_host.so.*"),
}

# Used in error messages; CUDA 12 hosts use the cu12 variants.
_PIP_PACKAGES = {
    "nccl": "nvidia-nccl-cu13",
    "cublas": "nvidia-cublas",
    "cublasmp": "nvidia-cublasmp-cu13",
    "nvshmem": "nvidia-nvshmem-cu13",
}


def _site_roots() -> list[Path]:
    """Candidate site-packages roots, in deterministic order."""
    candidates: list[str] = []
    try:
        candidates.extend(site.getsitepackages())
        candidates.append(site.getusersitepackages())
    except AttributeError:
        pass
    candidates.extend(p for p in sys.path if p.endswith(("site-packages", "dist-packages")))
    roots: list[Path] = []
    for raw in candidates:
        path = Path(raw)
        if path.is_dir() and path not in roots:
            roots.append(path)
    return roots


def _discover_one(name: str) -> Path:
    dirs, pattern = _LIBRARY_PATTERNS[name]
    resolved: set[Path] = set()
    for root in _site_roots():
        for rel in dirs:
            lib_dir = root / rel
            if not lib_dir.is_dir():
                continue
            for candidate in lib_dir.glob(pattern):
                if candidate.is_file():
                    resolved.add(candidate.resolve())
    if len(resolved) == 1:
        return resolved.pop()
    package = _PIP_PACKAGES[name]
    if not resolved:
        raise RuntimeError(
            f"cannot find {pattern} under any site-packages root; "
            f"install it with `pip install {package}`"
        )
    listing = "\n".join(f"  - {path}" for path in sorted(resolved))
    raise RuntimeError(
        f"found multiple distinct {pattern} installations; GemmComm benchmarks "
        f"require exactly one so the comparison cannot drift:\n{listing}"
    )


def discover_libraries() -> dict[str, Path]:
    """Resolve the four locked communication libraries from pip packages."""
    return {name: _discover_one(name) for name in _LIBRARY_PATTERNS}


def ensure_nvshmem_home() -> Path:
    """Return an NVSHMEM_HOME directory compatible with TVM's compile-time finder.

    TVM requires ``include/nvshmem.h`` plus ``lib/libnvshmem.so`` (or
    ``lib/libnvshmem.a``), while the pip package ships ``libnvshmem_host.so.*``
    instead. When the package layout already satisfies the finder it is
    returned as-is; otherwise a shim directory of symlinks is created under
    the system temp dir and returned.
    """
    host_lib = _discover_one("nvshmem")
    package_dir = host_lib.parent.parent  # <site-packages>/nvidia/nvshmem
    include_dir = package_dir / "include"
    lib_dir = package_dir / "lib"
    if not (include_dir / "nvshmem.h").is_file():
        raise RuntimeError(f"{include_dir} does not contain nvshmem.h")
    if (lib_dir / "libnvshmem.so").exists() or (lib_dir / "libnvshmem.a").exists():
        return package_dir

    digest = hashlib.sha1(str(package_dir).encode()).hexdigest()[:12]
    shim = Path(tempfile.gettempdir()) / f"tirx-nvshmem-home-{digest}"
    shim_lib = shim / "lib"
    shim_lib.mkdir(parents=True, exist_ok=True)
    for entry in lib_dir.iterdir():
        link = shim_lib / entry.name
        if link.is_symlink() and link.resolve() == entry.resolve():
            continue
        link.unlink(missing_ok=True)
        link.symlink_to(entry)
    alias = shim_lib / "libnvshmem.so"
    if not alias.is_symlink() or alias.resolve() != host_lib.resolve():
        alias.unlink(missing_ok=True)
        alias.symlink_to(host_lib)
    shim_include = shim / "include"
    if not shim_include.is_symlink() or shim_include.resolve() != include_dir.resolve():
        shim_include.unlink(missing_ok=True)
        shim_include.symlink_to(include_dir, target_is_directory=True)
    return shim


__all__ = ["discover_libraries", "ensure_nvshmem_home"]
