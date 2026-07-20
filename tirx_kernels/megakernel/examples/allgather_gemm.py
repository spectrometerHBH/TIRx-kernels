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

"""Standalone dynamic AllGather+GEMM graph using the TVM megakernel DSL."""

from __future__ import annotations

import argparse

from tirx_kernels.gemm_comm import allgather_gemm as impl
from tirx_kernels.gemm_comm.dsl import (
    AllGatherGemmTileImpl,
    AllGatherTileImpl,
    GemmCommLowerer,
    policy_for_scheduler,
)
from tvm.megakernel.dsl import KernelSpec


def build_example() -> KernelSpec:
    """Construct the complete logical graph without calling the production helper."""

    config = impl.derive_config()
    kernel = KernelSpec(
        "allgather_gemm",
        attrs={
            "source": "SM100 AllGather and GEMM overlap pipeline",
            "world_size": config.world_size,
        },
    )
    local_a = kernel.tensor("local_a", (config.local_m, config.K), config.dtype)
    local_weight = kernel.tensor("local_weight", (config.local_n, config.K), config.dtype)
    gathered_a = kernel.tensor("gathered_a", (config.M, config.K), config.dtype)
    output = kernel.tensor("output", (config.M, config.local_n), config.dtype)
    shard_ready = kernel.event("shard_ready", (config.world_size,), 1)

    (
        kernel.tile(
            "allgather",
            impl=AllGatherTileImpl({"local_a": local_a, "gathered_a": gathered_a}, config),
            tile_num=(config.world_size, 1, 1),
            reads=[local_a],
            writes=[gathered_a],
        ).notify(shard_ready, lambda source, _n, _k: (source,))
    )
    (
        kernel.tile(
            "gemm",
            impl=AllGatherGemmTileImpl(
                {
                    "local_a": local_a,
                    "local_weight": local_weight,
                    "gathered_a": gathered_a,
                    "output": output,
                },
                config,
                None,
            ),
            tile_num=(config.gemm_m_clusters, config.gemm_n_clusters, 1),
            reads=[local_a, gathered_a, local_weight],
            writes=[output],
        ).wait(shard_ready, lambda m_idx, _n, _k: (m_idx // config.local_gemm_m_clusters,))
    )
    return kernel.validate()


def describe(spec: KernelSpec, scheduler: str) -> str:
    plan = GemmCommLowerer(policy_for_scheduler(scheduler)).lower(spec, plan_only=True).plan
    return "\n".join(
        [
            f"kernel: {spec.name}",
            f"physical scheduler: {plan.physical_scheduler}",
            f"device regions: {len(plan.execution.device_regions)}",
            f"host regions: {len(plan.execution.host_regions)}",
            f"tasks per rank: {plan.task_count_per_rank}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", choices=("dynamic",), default="dynamic")
    args = parser.parse_args(argv)
    print(describe(build_example(), args.scheduler))


if __name__ == "__main__":
    main()
