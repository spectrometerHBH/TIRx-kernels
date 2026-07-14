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

"""Scheduler-independent logical graphs for the distributed GEMM kernels."""

from __future__ import annotations

from tvm.megakernel.dsl import KernelSpec

from .. import _allgather_gemm_impl as ag_impl
from .. import _gemm_reduce_scatter_impl as rs_impl
from .tile_impl import (
    AllGatherGemmTileImpl,
    AllGatherTileImpl,
    PartialGemmTileImpl,
    ReduceScatterTileImpl,
    ReduceSumTileImpl,
)


def build_allgather_gemm_graph() -> KernelSpec:
    """Describe TP4 AllGather followed by rank-local GEMM output columns."""

    kernel = KernelSpec(
        "allgather_gemm", attrs={"source": "SM100 TP4 AllGather and GEMM overlap pipeline"}
    )
    local_a = kernel.input("local_a", (ag_impl.LOCAL_M, ag_impl.K), ag_impl.a_type)
    local_weight = kernel.input("local_weight", (ag_impl.LOCAL_N, ag_impl.K), ag_impl.b_type)
    gathered_a = kernel.intermediate("gathered_a", (ag_impl.M, ag_impl.K), ag_impl.a_type)
    output = kernel.output("output", (ag_impl.M, ag_impl.LOCAL_N), ag_impl.d_type)
    shard_ready = kernel.event(
        "shard_ready",
        (ag_impl.WORLD_SIZE,),
        1,
        attrs={"meaning": "one source activation shard is visible on this rank"},
    )

    (
        kernel.tile(
            "allgather",
            impl=AllGatherTileImpl(),
            tile_num=(ag_impl.WORLD_SIZE, 1, 1),
            attrs={"purpose": "publish every source activation shard to all ranks"},
        )
        .read(local_a)
        .write(gathered_a)
        .notify(shard_ready, lambda m, n, k: (m,))
    )
    (
        kernel.tile(
            "gemm",
            impl=AllGatherGemmTileImpl(),
            tile_num=(ag_impl.GEMM_M_CLUSTERS, ag_impl.GEMM_N_CLUSTERS, 1),
            attrs={"purpose": "multiply one gathered activation cluster by local weights"},
        )
        .read(local_a, gathered_a, local_weight)
        .write(output)
        .wait(shard_ready, lambda m, n, k: (m // ag_impl.LOCAL_GEMM_M_CLUSTERS,))
    )
    return kernel


def build_gemm_reduce_scatter_graph() -> KernelSpec:
    """Describe rank-local partial GEMM, peer transfer, and local reduction."""

    m_clusters = rs_impl.M // (rs_impl.BLK_M * rs_impl.CLUSTER_M * rs_impl.NUM_CONSUMER)
    n_clusters = rs_impl.N // rs_impl.BLK_N
    local_m_clusters = m_clusters // rs_impl.WORLD_SIZE
    kernel = KernelSpec(
        "gemm_reduce_scatter", attrs={"source": "SM100 TP4 GEMM and ReduceScatter overlap pipeline"}
    )
    local_a = kernel.input("local_a", (rs_impl.M, rs_impl.K), rs_impl.a_type)
    local_weight = kernel.input("local_weight", (rs_impl.N, rs_impl.K), rs_impl.b_type)
    partial = kernel.intermediate("partial", (rs_impl.M, rs_impl.N), rs_impl.d_type)
    staging = kernel.intermediate(
        "staging", (rs_impl.WORLD_SIZE, rs_impl.LOCAL_M, rs_impl.N), rs_impl.d_type
    )
    output = kernel.output("output", (rs_impl.LOCAL_M, rs_impl.N), rs_impl.d_type)
    partial_ready = kernel.event(
        "partial_shard_ready",
        (rs_impl.WORLD_SIZE,),
        local_m_clusters * n_clusters,
        attrs={"meaning": "all logical GEMM clusters for one output row shard are complete"},
    )
    staging_ready = kernel.event(
        "staging_ready",
        (rs_impl.WORLD_SIZE,),
        rs_impl.WORLD_SIZE,
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
            tile_num=(rs_impl.WORLD_SIZE, rs_impl.WORLD_SIZE, 1),
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
            tile_num=(
                rs_impl.WORLD_SIZE,
                rs_impl.LOCAL_M // rs_impl.BLK_M_RS,
                rs_impl.N // rs_impl.BLK_N_RS,
            ),
            attrs={"purpose": "sum source-rank partials for one destination output tile"},
        )
        .read(staging)
        .write(output)
        .wait(staging_ready, lambda destination, m, n: (destination,))
    )
    return kernel


__all__ = ["build_allgather_gemm_graph", "build_gemm_reduce_scatter_graph"]
