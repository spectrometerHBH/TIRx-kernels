# Copyright (c) 2026 The TIRX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Automated source and pre-dispatch IR gate for plain-TIRx kernels."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any

import tvm
from tvm.ir import Op


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _qualified_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _assigned_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Tuple | ast.List):
        return [name for item in node.elts for name in _assigned_names(item)]
    return []


def _source_violations(module: ModuleType) -> list[str]:
    path = Path(inspect.getsourcefile(module) or "")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    tainted: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "tile_primitive" in alias.name or alias.name.endswith(".tile"):
                    tainted.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            for alias in node.names:
                if (
                    "tile_primitive" in module_name
                    or module_name.endswith(".tile")
                    or (module_name.endswith("tirx") and alias.name == "tile")
                ):
                    tainted.add(alias.asname or alias.name)

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            value = node.value
            if value is None:
                continue
            qualified = _qualified_name(value)
            root = qualified.split(".")[0] if qualified else None
            is_tainted = bool(
                qualified
                and (
                    root in tainted
                    or qualified.startswith("tirx.tile.")
                    or ".tirx.tile." in qualified
                )
            )
            if not is_tainted:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in _assigned_names(target):
                    if name not in tainted:
                        tainted.add(name)
                        changed = True

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        qualified = _qualified_name(node.func)
        root = qualified.split(".")[0] if qualified else None
        tile_call = bool(
            qualified
            and (
                root in tainted
                or qualified.startswith("tirx.tile.")
                or ".tirx.tile." in qualified
                or qualified.endswith("TilePrimitiveCall")
            )
        )
        custom_tile = False
        if qualified and (
            "register" in qualified or qualified.endswith("Op.get") or qualified.endswith("get_op")
        ):
            for arg in [*node.args, *[keyword.value for keyword in node.keywords]]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith("tirx.tile.") or arg.value == "tile_primitive":
                        custom_tile = True
        if tile_call or custom_tile:
            violations.append(
                f"{path}:{node.lineno}:{node.col_offset + 1}: "
                f"tile-primitive source call {qualified or ast.dump(node.func)}"
            )
    return violations


def _ir_violations(func: Any, label: str) -> list[str]:
    violations: list[str] = []
    index = 0

    def visit(node):
        nonlocal index
        node_index = index
        index += 1
        op = None
        if isinstance(node, tvm.tirx.TilePrimitiveCall):
            op = node.op
            violations.append(
                f"specialization[{label}].body/post_order[{node_index}]: "
                f"TilePrimitiveCall op={getattr(op, 'name', op)}"
            )
            return
        if isinstance(node, tvm.ir.Call) and isinstance(node.op, Op):
            op = node.op
        if op is None:
            return
        name = op.name
        category = op.get_attr("TIRxOpCategory")
        if name.startswith("tirx.tile.") or category == "tile_primitive":
            violations.append(
                f"specialization[{label}].body/post_order[{node_index}]: "
                f"operator={name} category={category}"
            )

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return violations


def check_module(module: ModuleType) -> list[str]:
    violations = _source_violations(module)
    configs = getattr(module, "CONFIGS", None)
    get_kernel = getattr(module, "get_kernel", None)
    if not isinstance(configs, list) or not callable(get_kernel):
        raise TypeError(f"{module.__name__} must expose CONFIGS and get_kernel")
    for ordinal, config in enumerate(configs):
        params = {key: value for key, value in config.items() if key != "label"}
        label = str(config.get("label", ordinal))
        func = get_kernel(**params)
        funcs = func if isinstance(func, tuple | list) else [func]
        for func_ordinal, prim_func in enumerate(funcs):
            suffix = f"/{func_ordinal}" if len(funcs) > 1 else ""
            violations.extend(_ir_violations(prim_func, label + suffix))
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", help="fully-qualified target kernel module")
    args = parser.parse_args()
    module = importlib.import_module(args.module)
    violations = check_module(module)
    if violations:
        print("plain-TIRx gate: FAIL")
        for violation in violations:
            print(violation)
        raise SystemExit(1)
    print(f"plain-TIRx gate: PASS ({len(module.CONFIGS)} configurations)")


if __name__ == "__main__":
    main()
