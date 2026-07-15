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

"""Static and dynamic schedule policies for distributed GEMM graphs."""

from __future__ import annotations

from tvm.megakernel.dsl import KernelSpec
from tvm.megakernel.transform import (
    DeviceRegionPlan,
    ExecutionPlan,
    FetchGuardStep,
    HostCallStep,
    HostRegionPlan,
    HostSyncStep,
    MidBodyPortStep,
    RegionDependencyPlan,
    RunStep,
    TileProgram,
    logical_edges,
)

from .. import allgather_gemm as ag_kernel
from .. import gemm_reduce_scatter as rs_kernel
from .model import GemmCommPlan, PhysicalTask, RankSchedule
from .specs import build_allgather_gemm_graph, build_gemm_reduce_scatter_graph


def _allgather_tasks(rank: int) -> tuple[PhysicalTask, ...]:
    tasks: list[PhysicalTask] = []
    for shard_offset in range(ag_kernel.WORLD_SIZE):
        shard = (rank + shard_offset) % ag_kernel.WORLD_SIZE
        begin = shard * ag_kernel.LOCAL_GEMM_M_CLUSTERS
        end = begin + ag_kernel.LOCAL_GEMM_M_CLUSTERS
        for n_idx in range(ag_kernel.GEMM_N_CLUSTERS):
            for m_idx in range(begin, end):
                tasks.append(PhysicalTask("gemm", m_idx, n_idx))
    return tuple(tasks)


def _reduce_scatter_tasks(rank: int) -> tuple[PhysicalTask, ...]:
    m_clusters = rs_kernel.M // (rs_kernel.BLK_M * rs_kernel.CLUSTER_M * rs_kernel.NUM_CONSUMER)
    n_clusters = rs_kernel.N // rs_kernel.BLK_N
    local_m_clusters = m_clusters // rs_kernel.WORLD_SIZE
    tasks: list[PhysicalTask] = []
    for shard_offset in (*range(1, rs_kernel.WORLD_SIZE), 0):
        shard = (rank + shard_offset) % rs_kernel.WORLD_SIZE
        begin = shard * local_m_clusters
        end = begin + local_m_clusters
        for n_idx in range(n_clusters):
            for m_idx in range(begin, end):
                tasks.append(PhysicalTask("partial_gemm", m_idx, n_idx))
    return tuple(tasks)


def _distribute_static(
    rank: int, tasks: tuple[PhysicalTask, ...], persistent_clusters: int
) -> RankSchedule:
    queues = tuple(
        tuple(tasks[worker::persistent_clusters]) for worker in range(persistent_clusters)
    )
    return RankSchedule(rank=rank, worker_queues=queues)


def _allgather_execution(spec: KernelSpec, policy: str, scheduler: str) -> ExecutionPlan:
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
                attrs={"scheduler": scheduler, "entrypoint": ag_kernel.GEMM_DEVICE_ENTRYPOINT},
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
        attrs={"policy": policy},
    ).validate()


def _reduce_scatter_execution(spec: KernelSpec, policy: str, scheduler: str) -> ExecutionPlan:
    by_event = {edge.event.name: edge for edge in logical_edges(spec)}
    partial_edge = by_event["partial_shard_ready"]
    staging_edge = by_event["staging_ready"]
    tiles = {tile.name: tile for tile in spec.tiles}
    return ExecutionPlan(
        kernel=spec,
        device_regions=(
            DeviceRegionPlan(
                "partial_gemm_device",
                tile_programs=(
                    TileProgram(
                        tiles["partial_gemm"],
                        (
                            RunStep(),
                            MidBodyPortStep(
                                "after_store_before_pipeline_advance", edges=(partial_edge,)
                            ),
                        ),
                    ),
                ),
                attrs={
                    "scheduler": scheduler,
                    "entrypoint": rs_kernel.PARTIAL_GEMM_DEVICE_ENTRYPOINT,
                },
            ),
            DeviceRegionPlan(
                "reduce_device",
                tile_programs=(TileProgram(tiles["reduce"], (RunStep(),)),),
                attrs={"entrypoint": rs_kernel.REDUCE_SUM_DEVICE_ENTRYPOINT},
            ),
        ),
        host_regions=(
            HostRegionPlan(
                "reduce_scatter_host",
                (
                    HostCallStep("collective"),
                    HostSyncStep("communication_completion", edges=(staging_edge,)),
                ),
                attrs={"tile": "transfer", "entrypoint": rs_kernel.REDUCE_SCATTER_HOST_ENTRYPOINT},
            ),
        ),
        region_dependencies=(
            RegionDependencyPlan("partial_gemm_device", "reduce_scatter_host", "launch_order"),
            RegionDependencyPlan("reduce_scatter_host", "reduce_device", "completion"),
        ),
        attrs={"policy": policy},
    ).validate()


class GemmCommPolicy:
    name = "base"

    def normalize(self, spec: KernelSpec) -> GemmCommPlan:
        raise NotImplementedError


class StaticPolicy(GemmCommPolicy):
    name = "static"

    def normalize(self, spec: KernelSpec) -> GemmCommPlan:
        if spec.name == "allgather_gemm":
            clusters = ag_kernel.SM_NUMBER // ag_kernel.M_CLUSTER
            schedules = tuple(
                _distribute_static(rank, _allgather_tasks(rank), clusters)
                for rank in range(ag_kernel.WORLD_SIZE)
            )
            plan = GemmCommPlan(
                execution=_allgather_execution(spec, self.name, "rank_aware_grid_stride"),
                persistent_clusters=clusters,
                rank_schedules=schedules,
            )
        elif spec.name == "gemm_reduce_scatter":
            clusters = rs_kernel.GEMM_SMS // rs_kernel.CLUSTER_M
            schedules = tuple(
                _distribute_static(rank, _reduce_scatter_tasks(rank), clusters)
                for rank in range(rs_kernel.WORLD_SIZE)
            )
            plan = GemmCommPlan(
                execution=_reduce_scatter_execution(spec, self.name, "rank_aware_group_major"),
                persistent_clusters=clusters,
                rank_schedules=schedules,
            )
        else:
            raise ValueError(f"unsupported distributed GEMM graph: {spec.name!r}")
        return plan.validate()


class DynamicPolicy(GemmCommPolicy):
    name = "dynamic"

    def normalize(self, spec: KernelSpec) -> GemmCommPlan:
        if spec.name == "allgather_gemm":
            schedules = tuple(
                RankSchedule(rank=rank, shared_queue=_allgather_tasks(rank))
                for rank in range(ag_kernel.WORLD_SIZE)
            )
            plan = GemmCommPlan(
                execution=_allgather_execution(spec, self.name, "mpmc_queue"),
                persistent_clusters=ag_kernel.SM_NUMBER // ag_kernel.M_CLUSTER,
                rank_schedules=schedules,
            )
        elif spec.name == "gemm_reduce_scatter":
            schedules = tuple(
                RankSchedule(rank=rank, shared_queue=_reduce_scatter_tasks(rank))
                for rank in range(rs_kernel.WORLD_SIZE)
            )
            reason = (
                "the current GEMM pipeline advances TMA, MMA, and epilogue roles independently; "
                "a CTA-wide dynamic dequeue would serialize the inter-tile pipeline"
            )
            plan = GemmCommPlan(
                execution=_reduce_scatter_execution(spec, self.name, "planned_mpmc_queue"),
                persistent_clusters=rs_kernel.GEMM_SMS // rs_kernel.CLUSTER_M,
                rank_schedules=schedules,
                lowerable=False,
                unsupported_reason=reason,
            )
        else:
            raise ValueError(f"unsupported distributed GEMM graph: {spec.name!r}")
        return plan.validate()


def policy_for_scheduler(scheduler: str) -> GemmCommPolicy:
    if scheduler == "static":
        return StaticPolicy()
    if scheduler == "dynamic":
        return DynamicPolicy()
    raise ValueError(f"unsupported distributed GEMM scheduler: {scheduler!r}")


def make_plan(workload: str, scheduler: str) -> GemmCommPlan:
    if workload == "allgather_gemm":
        spec = build_allgather_gemm_graph()
    elif workload == "gemm_reduce_scatter":
        spec = build_gemm_reduce_scatter_graph()
    else:
        raise ValueError(f"unsupported distributed GEMM workload: {workload!r}")
    return policy_for_scheduler(scheduler).normalize(spec)


__all__ = ["DynamicPolicy", "GemmCommPolicy", "StaticPolicy", "make_plan", "policy_for_scheduler"]
