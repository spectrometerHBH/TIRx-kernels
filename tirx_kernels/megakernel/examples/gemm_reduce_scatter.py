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

"""Standalone fused dynamic-multimem GemmRS graph using the TVM megakernel DSL."""

from __future__ import annotations

import argparse

from tirx_kernels.gemm_comm import gemm_reduce_scatter as impl
from tirx_kernels.gemm_comm.dsl import (
    GemmCommLowerer,
    MultimemReduceScatterTileImpl,
    PartialGemmTileImpl,
    policy_for_scheduler,
)
from tvm.megakernel.dsl import KernelSpec


def build_example() -> KernelSpec:
    """Construct the complete logical graph without calling the production helper."""

    config = impl.derive_config()
    kernel = KernelSpec(
        "gemm_reduce_scatter",
        attrs={"source": "SM100 fused dynamic-multimem GemmRS", "world_size": config.world_size},
    )
    local_a = kernel.tensor("local_a", (config.M, config.k_local), config.dtype)
    local_weight = kernel.tensor("local_weight", (config.N, config.k_local), config.dtype)
    partial = kernel.tensor("partial", (config.M, config.N), config.dtype)
    output = kernel.tensor("output", (config.local_m, config.N), config.dtype)
    rs_ready = kernel.event(
        "reduce_scatter_ready", (config.rs_m_clusters, config.rs_n_clusters), config.world_size
    )

    (
        kernel.tile(
            "partial_gemm",
            impl=PartialGemmTileImpl(
                {"local_a": local_a, "local_weight": local_weight, "partial": partial}, config, None
            ),
            tile_num=(config.gemm_m_clusters * impl.NUM_CONSUMER, config.gemm_n_clusters, 1),
            reads=[local_a, local_weight],
            writes=[partial],
        ).notify(rs_ready, lambda m_idx, n_idx, _k: (m_idx % config.rs_m_clusters, n_idx))
    )
    (
        kernel.tile(
            "reduce_scatter",
            impl=MultimemReduceScatterTileImpl({"partial": partial, "output": output}, config),
            tile_num=(config.rs_m_clusters, config.rs_n_clusters, 1),
            reads=[partial],
            writes=[output],
        ).wait(rs_ready, lambda m_idx, n_idx, _k: (m_idx, n_idx))
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
            f"initial tasks per rank: {plan.task_count_per_rank}",
            f"pushed tasks per rank: {plan.pushed_task_count_per_rank}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", choices=("dynamic",), default="dynamic")
    args = parser.parse_args(argv)
    print(describe(build_example(), args.scheduler))


if __name__ == "__main__":
    main()
