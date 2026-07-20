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

"""Dynamic physical policy for the direct GemmComm kernels."""

from __future__ import annotations

from tvm.megakernel.dsl import KernelSpec
from tvm.megakernel.transform import (
    DeviceRegionPlan,
    ExecutionPlan,
    FetchGuardStep,
    HostCallStep,
    HostRegionPlan,
    QueuePushStep,
    RegionDependencyPlan,
    RunStep,
    TileProgram,
    logical_edges,
)

from .. import allgather_gemm as ag_kernel
from .. import gemm_reduce_scatter as rs_kernel
from .model import GemmCommPlan, PhysicalTask, RankSchedule
from .specs import build_allgather_gemm_graph, build_gemm_reduce_scatter_graph


def _allgather_tasks(spec: KernelSpec, rank: int) -> tuple[PhysicalTask, ...]:
    config = spec.tiles[1].impl.config
    tasks: list[PhysicalTask] = []
    offset = rank * config.local_gemm_m_clusters
    group_count = config.gemm_m_clusters // config.group_size
    for group in range(group_count):
        begin = group * config.group_size
        end = begin + config.group_size
        for n_idx in range(config.gemm_n_clusters):
            for m_idx in range(begin, end):
                tasks.append(PhysicalTask("gemm", (offset + m_idx) % config.gemm_m_clusters, n_idx))
    return tuple(tasks)


def _gemm_rs_tasks(spec: KernelSpec) -> tuple[PhysicalTask, ...]:
    config = spec.tiles[0].impl.config
    return tuple(
        PhysicalTask("partial_gemm", m_idx, n_idx)
        for group_begin in range(0, config.gemm_n_clusters, rs_kernel.GROUP_SIZE)
        for m_idx in range(config.gemm_m_clusters)
        for n_idx in range(
            group_begin, min(group_begin + rs_kernel.GROUP_SIZE, config.gemm_n_clusters)
        )
    )


def _gemm_rs_pushed_tasks(spec: KernelSpec) -> tuple[PhysicalTask, ...]:
    config = spec.tiles[1].impl.config
    return tuple(
        PhysicalTask("reduce_scatter", m_idx, n_idx)
        for m_idx in range(config.rs_m_clusters)
        for n_idx in range(config.rs_n_clusters)
    )


def _allgather_execution(spec: KernelSpec) -> ExecutionPlan:
    edges = logical_edges(spec)
    if len(edges) != 1:
        raise ValueError("AllGather+GEMM requires exactly one logical edge")
    edge = edges[0]
    tiles = {tile.name: tile for tile in spec.tiles}
    return ExecutionPlan(
        kernel=spec,
        device_regions=(
            DeviceRegionPlan(
                "gemm_device",
                fetch_steps=(FetchGuardStep(predicate="remote_rank != rank", edges=(edge,)),),
                tile_programs=(TileProgram(tiles["gemm"], (RunStep(),)),),
                attrs={"scheduler": "mpmc_queue", "entrypoint": ag_kernel.GEMM_DEVICE_ENTRYPOINT},
            ),
        ),
        host_regions=(
            HostRegionPlan(
                "allgather_host",
                (HostCallStep("collective"),),
                attrs={"tile": "allgather", "entrypoint": ag_kernel.ALLGATHER_HOST_ENTRYPOINT},
            ),
        ),
        region_dependencies=(
            RegionDependencyPlan("allgather_host", "gemm_device", "launch_order"),
        ),
        attrs={"policy": "dynamic"},
    ).validate()


def _gemm_rs_execution(spec: KernelSpec) -> ExecutionPlan:
    edges = logical_edges(spec)
    if len(edges) != 1:
        raise ValueError("GemmRS requires exactly one logical queue edge")
    edge = edges[0]
    tiles = {tile.name: tile for tile in spec.tiles}
    return ExecutionPlan(
        kernel=spec,
        device_regions=(
            DeviceRegionPlan(
                "fused_gemm_rs_device",
                tile_programs=(
                    TileProgram(
                        tiles["partial_gemm"],
                        (
                            RunStep(
                                repeat=rs_kernel.NUM_CONSUMER,
                                index_map=lambda m, n, k, repeat: (
                                    m * rs_kernel.NUM_CONSUMER + repeat,
                                    n,
                                    k,
                                ),
                            ),
                            QueuePushStep(edges=(edge,)),
                        ),
                    ),
                    TileProgram(tiles["reduce_scatter"], (RunStep(),)),
                ),
                attrs={"scheduler": "mpmc_queue", "entrypoint": rs_kernel.FUSED_DEVICE_ENTRYPOINT},
            ),
        ),
        attrs={"policy": "dynamic"},
    ).validate()


class GemmCommPolicy:
    name = "base"

    def normalize(self, spec: KernelSpec) -> GemmCommPlan:
        raise NotImplementedError


class DynamicPolicy(GemmCommPolicy):
    name = "dynamic"

    def normalize(self, spec: KernelSpec) -> GemmCommPlan:
        if spec.name == "allgather_gemm":
            config = spec.tiles[1].impl.config
            schedules = tuple(
                RankSchedule(rank=rank, shared_queue=_allgather_tasks(spec, rank))
                for rank in range(config.world_size)
            )
            plan = GemmCommPlan(
                execution=_allgather_execution(spec),
                persistent_clusters=ag_kernel.SM_NUMBER // ag_kernel.M_CLUSTER,
                rank_schedules=schedules,
            )
        elif spec.name == "gemm_reduce_scatter":
            config = spec.tiles[0].impl.config
            initial = _gemm_rs_tasks(spec)
            pushed = _gemm_rs_pushed_tasks(spec)
            schedules = tuple(
                RankSchedule(rank=rank, shared_queue=initial, pushed_tasks=pushed)
                for rank in range(config.world_size)
            )
            plan = GemmCommPlan(
                execution=_gemm_rs_execution(spec),
                persistent_clusters=rs_kernel.SM_NUMBER // rs_kernel.M_CLUSTER,
                rank_schedules=schedules,
            )
        else:
            raise ValueError(f"unsupported distributed GEMM graph: {spec.name!r}")
        return plan.validate()


class StaticPolicy(GemmCommPolicy):
    name = "static"

    def normalize(self, spec: KernelSpec) -> GemmCommPlan:
        del spec
        raise ValueError("the direct GemmComm kernels support only scheduler='dynamic'")


def policy_for_scheduler(scheduler: str) -> GemmCommPolicy:
    if scheduler == "dynamic":
        return DynamicPolicy()
    if scheduler == "static":
        return StaticPolicy()
    raise ValueError(f"unsupported distributed GEMM scheduler: {scheduler!r}")


def make_plan(workload: str, scheduler: str = "dynamic") -> GemmCommPlan:
    if workload == "allgather_gemm":
        spec = build_allgather_gemm_graph()
    elif workload == "gemm_reduce_scatter":
        spec = build_gemm_reduce_scatter_graph()
    else:
        raise ValueError(f"unsupported distributed GEMM workload: {workload!r}")
    return policy_for_scheduler(scheduler).normalize(spec)


__all__ = ["DynamicPolicy", "GemmCommPolicy", "StaticPolicy", "make_plan", "policy_for_scheduler"]
