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

"""Build and inspect the production six-stage MoE with the TVM megakernel DSL.

Run from the tirx-kernels checkout with the paired TVM checkout on
``PYTHONPATH``::

    python -m tirx_kernels.megakernel.examples.moe_dsl \
        --batch-size 128 --scheduler dynamic

The complete authoring source is ``build_moe_graph`` in
``tirx_kernels.megakernel.dsl.moe_dsl``.  This executable deliberately calls
that function instead of copying the graph, so the example and production
lowering always consume the same ``KernelSpec``.
"""

from __future__ import annotations

import argparse

from tirx_kernels.megakernel.dsl import (
    KernelSpec,
    MoeLowerer,
    VarSpec,
    build_moe_graph,
    policy_for_scheduler,
)
from tirx_kernels.megakernel.utils.config import MEGAKERNEL_MOE_BENCH_CONFIG


def build_example(batch_size: int = 128) -> KernelSpec:
    """Return the validated production MoE logical graph."""

    return build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, batch_size).validate()


def _extent_text(extent: int | VarSpec) -> str:
    return extent.name if isinstance(extent, VarSpec) else str(extent)


def _tensor_names(tensors) -> str:
    return ", ".join(tensor.name for tensor in tensors) or "-"


def _event_names(dependencies) -> str:
    return ", ".join(dependency.event.name for dependency in dependencies) or "-"


def describe_graph(spec: KernelSpec, batch_size: int) -> str:
    """Render the logical DSL graph without exposing scheduler-private fields."""

    lines = [
        f"kernel: {spec.name}",
        f"batch_size: {batch_size}",
        f"logical tensors: {len(spec.tensors)}",
        f"logical events ({len(spec.events)}): {', '.join(spec.events)}",
        f"tiles ({len(spec.tiles)}):",
    ]
    for tile in spec.tiles:
        tile_num = ", ".join(_extent_text(extent) for extent in tile.tile_num)
        lines.extend(
            [
                f"  - {tile.name}: {type(tile.impl).__name__} tile_num=({tile_num})",
                f"    reads: {_tensor_names(tile.reads)}",
                f"    writes: {_tensor_names(tile.writes)}",
                f"    waits: {_event_names(tile.waits)}",
                f"    notifies: {_event_names(tile.notifies)}",
            ]
        )
    return "\n".join(lines)


def describe_plan(spec: KernelSpec, scheduler: str) -> str:
    """Lower the example through one policy and render its physical-plan boundary."""

    plan = MoeLowerer(policy_for_scheduler(scheduler)).lower(spec)
    return "\n".join(
        [
            f"scheduler: {plan.policy_name}",
            f"physical events ({len(plan.events)}): "
            + ", ".join(event.name for event in plan.events),
            f"dispatch rules: {len(plan.dispatch_plans)}",
            f"queue upper bound: {plan.queue_upper_bound}",
            f"down coalescing: {plan.down_coalescing}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--scheduler", choices=("static", "unfused", "dynamic"))
    args = parser.parse_args(argv)

    spec = build_example(args.batch_size)
    print(describe_graph(spec, args.batch_size))
    if args.scheduler is not None:
        print()
        print(describe_plan(spec, args.scheduler))


if __name__ == "__main__":
    main()
