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

from .. import _allgather_gemm_impl as ag_impl
from .. import _gemm_reduce_scatter_impl as rs_impl
from .model import GemmCommPlan, PhysicalTask, RankSchedule
from .specs import build_allgather_gemm_graph, build_gemm_reduce_scatter_graph


def _allgather_tasks(rank: int) -> tuple[PhysicalTask, ...]:
    tasks: list[PhysicalTask] = []
    for shard_offset in range(ag_impl.WORLD_SIZE):
        shard = (rank + shard_offset) % ag_impl.WORLD_SIZE
        begin = shard * ag_impl.LOCAL_GEMM_M_CLUSTERS
        end = begin + ag_impl.LOCAL_GEMM_M_CLUSTERS
        for n_idx in range(ag_impl.GEMM_N_CLUSTERS):
            for m_idx in range(begin, end):
                tasks.append(PhysicalTask("gemm", m_idx, n_idx))
    return tuple(tasks)


def _reduce_scatter_tasks(rank: int) -> tuple[PhysicalTask, ...]:
    m_clusters = rs_impl.M // (rs_impl.BLK_M * rs_impl.CLUSTER_M * rs_impl.NUM_CONSUMER)
    n_clusters = rs_impl.N // rs_impl.BLK_N
    local_m_clusters = m_clusters // rs_impl.WORLD_SIZE
    tasks: list[PhysicalTask] = []
    for shard_offset in (*range(1, rs_impl.WORLD_SIZE), 0):
        shard = (rank + shard_offset) % rs_impl.WORLD_SIZE
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


class GemmCommPolicy:
    name = "base"

    def normalize(self, spec: KernelSpec) -> GemmCommPlan:
        raise NotImplementedError


class StaticPolicy(GemmCommPolicy):
    name = "static"

    def normalize(self, spec: KernelSpec) -> GemmCommPlan:
        if spec.name == "allgather_gemm":
            clusters = ag_impl.SM_NUMBER // ag_impl.M_CLUSTER
            schedules = tuple(
                _distribute_static(rank, _allgather_tasks(rank), clusters)
                for rank in range(ag_impl.WORLD_SIZE)
            )
            plan = GemmCommPlan(
                spec=spec,
                workload=spec.name,
                policy_name=self.name,
                scheduled_tile="gemm",
                persistent_clusters=clusters,
                rank_schedules=schedules,
                physical_scheduler="rank_aware_grid_stride",
                launch_steps=(("allgather", "gemm"),),
            )
        elif spec.name == "gemm_reduce_scatter":
            clusters = rs_impl.GEMM_SMS // rs_impl.CLUSTER_M
            schedules = tuple(
                _distribute_static(rank, _reduce_scatter_tasks(rank), clusters)
                for rank in range(rs_impl.WORLD_SIZE)
            )
            plan = GemmCommPlan(
                spec=spec,
                workload=spec.name,
                policy_name=self.name,
                scheduled_tile="partial_gemm",
                persistent_clusters=clusters,
                rank_schedules=schedules,
                physical_scheduler="rank_aware_group_major",
                launch_steps=(("partial_gemm", "transfer"), ("reduce",)),
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
                for rank in range(ag_impl.WORLD_SIZE)
            )
            plan = GemmCommPlan(
                spec=spec,
                workload=spec.name,
                policy_name=self.name,
                scheduled_tile="gemm",
                persistent_clusters=ag_impl.SM_NUMBER // ag_impl.M_CLUSTER,
                rank_schedules=schedules,
                physical_scheduler="mpmc_queue",
                launch_steps=(("allgather", "gemm"),),
            )
        elif spec.name == "gemm_reduce_scatter":
            schedules = tuple(
                RankSchedule(rank=rank, shared_queue=_reduce_scatter_tasks(rank))
                for rank in range(rs_impl.WORLD_SIZE)
            )
            reason = (
                "the current GEMM pipeline advances TMA, MMA, and epilogue roles independently; "
                "a CTA-wide dynamic dequeue would serialize the inter-tile pipeline"
            )
            plan = GemmCommPlan(
                spec=spec,
                workload=spec.name,
                policy_name=self.name,
                scheduled_tile="partial_gemm",
                persistent_clusters=rs_impl.GEMM_SMS // rs_impl.CLUSTER_M,
                rank_schedules=schedules,
                physical_scheduler="planned_mpmc_queue",
                launch_steps=(("partial_gemm", "transfer"), ("reduce",)),
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
