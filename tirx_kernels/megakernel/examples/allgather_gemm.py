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

"""Standalone TP4 AllGather+GEMM example written with the TVM megakernel DSL.

Run from the tirx-kernels checkout with the paired TVM checkout on
``PYTHONPATH``::

    python -m tirx_kernels.megakernel.examples.allgather_gemm \
        --scheduler dynamic

The complete logical ``KernelSpec`` construction is visible in this file.  It
does not call the production ``build_allgather_gemm_graph`` helper.  The
attached concrete ``TileImpl`` classes live beside the complete kernel in
``tirx_kernels.gemm_comm.allgather_gemm``.
"""

from __future__ import annotations

import argparse

from tirx_kernels.gemm_comm import allgather_gemm as impl
from tirx_kernels.gemm_comm.dsl import GemmCommLowerer, policy_for_scheduler
from tvm.megakernel.dsl import KernelSpec


def build_example() -> KernelSpec:
    """Construct and validate the complete AllGather+GEMM logical graph."""

    kernel = KernelSpec(
        "allgather_gemm", attrs={"source": "SM100 TP4 AllGather and GEMM overlap pipeline"}
    )
    local_a = kernel.input("local_a", (impl.LOCAL_M, impl.K), impl.a_type)
    local_weight = kernel.input("local_weight", (impl.LOCAL_N, impl.K), impl.b_type)
    gathered_a = kernel.intermediate("gathered_a", (impl.M, impl.K), impl.a_type)
    output = kernel.output("output", (impl.M, impl.LOCAL_N), impl.d_type)
    shard_ready = kernel.event(
        "shard_ready",
        (impl.WORLD_SIZE,),
        1,
        attrs={"meaning": "one source activation shard is visible on this rank"},
    )

    (
        kernel.tile(
            "allgather",
            impl=impl.AllGatherTileImpl(),
            tile_num=(impl.WORLD_SIZE, 1, 1),
            attrs={"purpose": "publish every source activation shard to all ranks"},
        )
        .read(local_a)
        .write(gathered_a)
        .notify(shard_ready, lambda m, n, k: (m,))
    )
    (
        kernel.tile(
            "gemm",
            impl=impl.AllGatherGemmTileImpl(),
            tile_num=(impl.GEMM_M_CLUSTERS, impl.GEMM_N_CLUSTERS, 1),
            attrs={"purpose": "multiply one gathered activation cluster by local weights"},
        )
        .read(local_a, gathered_a, local_weight)
        .write(output)
        .wait(shard_ready, lambda m, n, k: (m // impl.LOCAL_GEMM_M_CLUSTERS,))
    )
    return kernel.validate()


def describe_graph(spec: KernelSpec) -> str:
    """Render the scheduler-independent logical graph."""

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
    """Normalize the graph through one physical scheduling policy."""

    plan = GemmCommLowerer(policy_for_scheduler(scheduler)).lower(spec, plan_only=True).plan
    return "\n".join(
        [
            f"scheduler: {plan.policy_name}",
            f"physical scheduler: {plan.physical_scheduler}",
            f"persistent clusters: {plan.persistent_clusters}",
            f"tasks per rank: {plan.task_count_per_rank}",
            f"lowerable: {str(plan.lowerable).lower()}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", choices=("static", "dynamic"))
    args = parser.parse_args(argv)

    spec = build_example()
    print(describe_graph(spec))
    if args.scheduler is not None:
        print()
        print(describe_plan(spec, args.scheduler))


if __name__ == "__main__":
    main()
