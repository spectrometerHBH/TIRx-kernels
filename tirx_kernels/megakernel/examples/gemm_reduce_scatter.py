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

"""Standalone TP4 GEMM+ReduceScatter example using the TVM megakernel DSL.

Run from the tirx-kernels checkout with the paired TVM checkout on
``PYTHONPATH``::

    python -m tirx_kernels.megakernel.examples.gemm_reduce_scatter \
        --scheduler static

The complete ``KernelSpec`` construction is visible in this file.  It does not
call the production ``build_gemm_reduce_scatter_graph`` helper.
"""

from __future__ import annotations

import argparse

from tirx_kernels.gemm_comm import _gemm_reduce_scatter_impl as impl
from tirx_kernels.gemm_comm.dsl import GemmCommLowerer, policy_for_scheduler
from tirx_kernels.gemm_comm.dsl.tile_impl import (
    PartialGemmTileImpl,
    ReduceScatterTileImpl,
    ReduceSumTileImpl,
)
from tvm.megakernel.dsl import KernelSpec


def build_example() -> KernelSpec:
    """Construct and validate the complete GEMM+ReduceScatter logical graph."""

    m_clusters = impl.M // (impl.BLK_M * impl.CLUSTER_M * impl.NUM_CONSUMER)
    n_clusters = impl.N // impl.BLK_N
    local_m_clusters = m_clusters // impl.WORLD_SIZE
    kernel = KernelSpec(
        "gemm_reduce_scatter", attrs={"source": "SM100 TP4 GEMM and ReduceScatter overlap pipeline"}
    )
    local_a = kernel.input("local_a", (impl.M, impl.K), impl.a_type)
    local_weight = kernel.input("local_weight", (impl.N, impl.K), impl.b_type)
    partial = kernel.intermediate("partial", (impl.M, impl.N), impl.d_type)
    staging = kernel.intermediate("staging", (impl.WORLD_SIZE, impl.LOCAL_M, impl.N), impl.d_type)
    output = kernel.output("output", (impl.LOCAL_M, impl.N), impl.d_type)
    partial_ready = kernel.event(
        "partial_shard_ready",
        (impl.WORLD_SIZE,),
        local_m_clusters * n_clusters,
        attrs={"meaning": "all logical GEMM clusters for one output row shard are complete"},
    )
    staging_ready = kernel.event(
        "staging_ready",
        (impl.WORLD_SIZE,),
        impl.WORLD_SIZE,
        attrs={"meaning": "all source-rank partial shards reached one destination"},
    )

    (
        kernel.tile(
            "partial_gemm",
            impl=PartialGemmTileImpl(),
            tile_num=(m_clusters, n_clusters, 1),
            attrs={"purpose": "compute one cluster of the rank-local partial product"},
        )
        .read(local_a, local_weight)
        .write(partial)
        .notify(partial_ready, lambda m, n, k: (m // local_m_clusters,))
    )
    (
        kernel.tile(
            "transfer",
            impl=ReduceScatterTileImpl(),
            tile_num=(impl.WORLD_SIZE, impl.WORLD_SIZE, 1),
            attrs={"purpose": "move one source partial shard to one destination rank"},
        )
        .read(partial)
        .write(staging)
        .wait(partial_ready, lambda source, destination, k: (destination,))
        .notify(staging_ready, lambda source, destination, k: (destination,))
    )
    (
        kernel.tile(
            "reduce",
            impl=ReduceSumTileImpl(),
            tile_num=(impl.WORLD_SIZE, impl.LOCAL_M // impl.BLK_M_RS, impl.N // impl.BLK_N_RS),
            attrs={"purpose": "sum source-rank partials for one destination output tile"},
        )
        .read(staging)
        .write(output)
        .wait(staging_ready, lambda destination, m, n: (destination,))
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
    parser.add_argument("--scheduler", choices=("static", "dynamic"))
    args = parser.parse_args(argv)

    spec = build_example()
    print(describe_graph(spec))
    if args.scheduler is not None:
        print()
        print(describe_plan(spec, args.scheduler))


if __name__ == "__main__":
    main()
