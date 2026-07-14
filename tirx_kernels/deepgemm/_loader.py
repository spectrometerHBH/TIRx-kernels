"""Helpers for loading the DeepGEMM distribution without package shadowing."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

_DEEP_GEMM_MODULE_NAME = "deep_gemm"


def _editable_distribution_root(distribution: Any) -> Path | None:
    """Return an editable distribution's project root from direct_url.json."""
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        return None
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return None
    if not direct_url.get("dir_info", {}).get("editable"):
        return None
    parsed = urlparse(str(direct_url.get("url", "")))
    if parsed.scheme != "file":
        return None
    root = Path(unquote(parsed.path)).resolve()
    package = root / _DEEP_GEMM_MODULE_NAME / "__init__.py"
    return root if package.is_file() else None


def _import_deep_gemm_distribution(
    *, required_extension_symbols: tuple[str, ...] = ()
) -> tuple[Any, str]:
    """Import ``deep_gemm`` even when another distribution shadows its package."""
    try:
        distribution = importlib.metadata.distribution(_DEEP_GEMM_MODULE_NAME)
    except importlib.metadata.PackageNotFoundError:
        distribution = None

    editable_root = _editable_distribution_root(distribution) if distribution is not None else None
    expected_package = (
        (editable_root / _DEEP_GEMM_MODULE_NAME).resolve() if editable_root is not None else None
    )
    inserted_path = None
    if expected_package is not None:
        existing = sys.modules.get(_DEEP_GEMM_MODULE_NAME)
        if existing is not None:
            existing_file = Path(str(getattr(existing, "__file__", ""))).resolve()
            if not existing_file.is_relative_to(expected_package):
                raise RuntimeError(
                    "DeepGEMM package collision: deep_gemm was already imported from "
                    f"{existing_file}, but distribution metadata points to {expected_package}"
                )
        elif str(editable_root) not in sys.path:
            inserted_path = str(editable_root)
            sys.path.insert(0, inserted_path)

    try:
        module = importlib.import_module(_DEEP_GEMM_MODULE_NAME)
    finally:
        if inserted_path is not None:
            sys.path.remove(inserted_path)

    module_file = Path(str(getattr(module, "__file__", ""))).resolve()
    if expected_package is not None and not module_file.is_relative_to(expected_package):
        raise RuntimeError(
            "DeepGEMM package collision: imported deep_gemm from "
            f"{module_file}, but distribution metadata points to {expected_package}"
        )

    distribution_version = (
        str(distribution.version)
        if distribution is not None
        else str(getattr(module, "__version__", "unknown"))
    )
    module_version = str(getattr(module, "__version__", "unknown"))
    if module_version != "unknown" and module_version != distribution_version.split("+", 1)[0]:
        raise RuntimeError(
            "DeepGEMM package collision: imported module version "
            f"{module_version} from {module_file}, but distribution metadata reports "
            f"{distribution_version}"
        )

    if required_extension_symbols:
        extension = getattr(module, "_C", None)
        missing = [
            symbol
            for symbol in required_extension_symbols
            if extension is None or not hasattr(extension, symbol)
        ]
        if missing:
            raise RuntimeError(
                f"DeepGEMM runtime lacks required extension symbols {missing}; loaded {module_file}"
            )
    return module, distribution_version
