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

"""Logical GemmComm graphs parameterized by the direct-kernel specialization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tvm.megakernel.dsl import KernelSpec

from .. import allgather_gemm as ag_kernel
from .. import gemm_reduce_scatter as rs_kernel
from .tile_impl import (
    AllGatherGemmTileImpl,
    AllGatherTileImpl,
    MultimemReduceScatterTileImpl,
    PartialGemmTileImpl,
)


def build_allgather_gemm_graph(
    config: ag_kernel.AllGatherGemmConfig | None = None,
    *,
    module_builder: Callable[[Any], Any] | None = None,
) -> KernelSpec:
    """Build the logical AllGather publication and persistent GEMM graph."""

    config = config or ag_kernel.derive_config()
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
    shard_ready = kernel.event(
        "shard_ready",
        (config.world_size,),
        1,
        attrs={"meaning": "one source activation shard is visible on this rank"},
    )

    (
        kernel.tile(
            "allgather",
            impl=AllGatherTileImpl({"local_a": local_a, "gathered_a": gathered_a}, config),
            tile_num=(config.world_size, 1, 1),
            reads=[local_a],
            writes=[gathered_a],
            attrs={"purpose": "publish one source activation shard to every rank"},
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
                module_builder,
            ),
            tile_num=(config.gemm_m_clusters, config.gemm_n_clusters, 1),
            reads=[local_a, gathered_a, local_weight],
            writes=[output],
            attrs={"purpose": "compute one gathered activation GEMM cluster"},
        ).wait(shard_ready, lambda m_idx, _n, _k: (m_idx // config.local_gemm_m_clusters,))
    )
    return kernel.validate()


def build_gemm_reduce_scatter_graph(
    config: rs_kernel.GemmRSConfig | None = None,
    *,
    module_builder: Callable[[Any], Any] | None = None,
) -> KernelSpec:
    """Build the two-program fused dynamic-multimem GemmRS graph."""

    config = config or rs_kernel.derive_config()
    logical_gemm_m_tiles = config.gemm_m_clusters * rs_kernel.NUM_CONSUMER
    kernel = KernelSpec(
        "gemm_reduce_scatter",
        attrs={
            "source": "SM100 fused persistent dynamic-multimem GemmRS",
            "world_size": config.world_size,
        },
    )
    local_a = kernel.tensor("local_a", (config.M, config.k_local), config.dtype)
    local_weight = kernel.tensor("local_weight", (config.N, config.k_local), config.dtype)
    partial = kernel.tensor("partial", (config.M, config.N), config.dtype)
    output = kernel.tensor("output", (config.local_m, config.N), config.dtype)
    rs_ready = kernel.event(
        "reduce_scatter_ready",
        (config.rs_m_clusters, config.rs_n_clusters),
        config.completion_count,
        attrs={"meaning": "all rank-local partials for one output tile are visible"},
    )

    (
        kernel.tile(
            "partial_gemm",
            impl=PartialGemmTileImpl(
                {"local_a": local_a, "local_weight": local_weight, "partial": partial},
                config,
                module_builder,
            ),
            tile_num=(logical_gemm_m_tiles, config.gemm_n_clusters, 1),
            reads=[local_a, local_weight],
            writes=[partial],
            attrs={"purpose": "compute one logical partial-GEMM tile"},
        ).notify(rs_ready, lambda m_idx, n_idx, _k: (m_idx % config.rs_m_clusters, n_idx))
    )
    (
        kernel.tile(
            "reduce_scatter",
            impl=MultimemReduceScatterTileImpl({"partial": partial, "output": output}, config),
            tile_num=(config.rs_m_clusters, config.rs_n_clusters, 1),
            reads=[partial],
            writes=[output],
            attrs={"purpose": "multimem-reduce one local output tile"},
        ).wait(rs_ready, lambda m_idx, n_idx, _k: (m_idx, n_idx))
    )
    return kernel.validate()


__all__ = ["build_allgather_gemm_graph", "build_gemm_reduce_scatter_graph"]
