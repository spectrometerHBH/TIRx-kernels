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

"""Build and inspect the distributed GEMM graphs with the TVM megakernel DSL.

Run from the tirx-kernels checkout with the paired TVM checkout on
``PYTHONPATH``::

    python -m tirx_kernels.gemm_comm.examples.gemm_comm_dsl \
        --workload allgather_gemm --scheduler dynamic
"""

from __future__ import annotations

import argparse

from tirx_kernels.gemm_comm.dsl import (
    GemmCommLowerer,
    build_allgather_gemm_graph,
    build_gemm_reduce_scatter_graph,
    policy_for_scheduler,
)
from tvm.megakernel.dsl import KernelSpec


def build_example(workload: str) -> KernelSpec:
    if workload == "allgather_gemm":
        return build_allgather_gemm_graph().validate()
    if workload == "gemm_reduce_scatter":
        return build_gemm_reduce_scatter_graph().validate()
    raise ValueError(f"unsupported workload: {workload!r}")


def describe_graph(spec: KernelSpec) -> str:
    lines = [f"kernel: {spec.name}", f"logical events: {', '.join(spec.events)}", "tiles:"]
    for tile in spec.tiles:
        waits = ", ".join(dependency.event.name for dependency in tile.waits) or "-"
        notifies = ", ".join(dependency.event.name for dependency in tile.notifies) or "-"
        lines.append(
            f"  - {tile.name}: {type(tile.impl).__name__} "
            f"tile_num={tuple(tile.tile_num)} waits={waits} notifies={notifies}"
        )
    return "\n".join(lines)


def describe_plan(spec: KernelSpec, scheduler: str) -> str:
    lowered = GemmCommLowerer(policy_for_scheduler(scheduler)).lower(spec, plan_only=True)
    plan = lowered.plan
    lines = [
        f"scheduler: {plan.policy_name}",
        f"physical scheduler: {plan.physical_scheduler}",
        f"persistent clusters: {plan.persistent_clusters}",
        f"tasks per rank: {plan.task_count_per_rank}",
        f"lowerable: {str(plan.lowerable).lower()}",
    ]
    if plan.unsupported_reason is not None:
        lines.append(f"reason: {plan.unsupported_reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload", choices=("allgather_gemm", "gemm_reduce_scatter"), required=True
    )
    parser.add_argument("--scheduler", choices=("static", "dynamic"))
    args = parser.parse_args(argv)

    spec = build_example(args.workload)
    print(describe_graph(spec))
    if args.scheduler is not None:
        print()
        print(describe_plan(spec, args.scheduler))


if __name__ == "__main__":
    main()
