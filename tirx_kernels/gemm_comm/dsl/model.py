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

"""Private normalized schedule model for distributed GEMM lowering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tvm.megakernel.dsl import KernelSpec, TileSpec
from tvm.megakernel.transform import (
    DeviceRegionPlan,
    EdgeBindingPlan,
    ExecutionPlan,
    FetchGuardAction,
    HostCallAction,
    HostEdgeAction,
    HostRegionPlan,
    MidBodyPortAction,
    RegionDependencyPlan,
    RunAction,
    SchedulerFetchProgram,
    TileActionProgram,
    logical_edges,
)

_FORBIDDEN_LOGICAL_FIELDS = {
    "dispatch",
    "job_type",
    "level",
    "mask",
    "queue",
    "rank",
    "release",
    "runtime_init",
    "scheduler",
    "scope",
    "scope_id",
}


def _validate_attrs(attrs: Mapping[str, Any], *, owner: str) -> None:
    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in _FORBIDDEN_LOGICAL_FIELDS:
                    raise ValueError(f"{owner} contains scheduler field {path + key!r}")
                visit(item, f"{path}{key}.")
        elif isinstance(value, tuple | list):
            for index, item in enumerate(value):
                visit(item, f"{path}{index}.")

    visit(attrs, "")


@dataclass(frozen=True, order=True)
class PhysicalTask:
    """One physical scheduler item referring to a logical tile instance."""

    tile: str
    m: int
    n: int
    k: int = 0

    @property
    def indices(self) -> tuple[int, int, int]:
        return (self.m, self.n, self.k)


@dataclass(frozen=True)
class RankSchedule:
    """Either one shared dynamic queue or deterministic per-cluster queues."""

    rank: int
    shared_queue: tuple[PhysicalTask, ...] = ()
    worker_queues: tuple[tuple[PhysicalTask, ...], ...] = ()

    @property
    def tasks(self) -> tuple[PhysicalTask, ...]:
        if self.shared_queue:
            return self.shared_queue
        return tuple(task for queue in self.worker_queues for task in queue)


@dataclass(frozen=True)
class GemmCommPlan:
    """Validated policy output consumed by the workload-specific lowerer."""

    spec: KernelSpec
    workload: str
    policy_name: str
    scheduled_tile: str
    persistent_clusters: int
    rank_schedules: tuple[RankSchedule, ...]
    physical_scheduler: str
    launch_steps: tuple[tuple[str, ...], ...]
    lowerable: bool = True
    unsupported_reason: str | None = None

    @property
    def world_size(self) -> int:
        return len(self.rank_schedules)

    @property
    def is_dynamic(self) -> bool:
        return self.policy_name == "dynamic"

    @property
    def tile(self) -> TileSpec:
        return next(tile for tile in self.spec.tiles if tile.name == self.scheduled_tile)

    @property
    def task_count_per_rank(self) -> int:
        tile_num = tuple(self.tile.tile_num)
        if not all(isinstance(extent, int) for extent in tile_num):
            raise TypeError("distributed GEMM plans require compile-time tile extents")
        return tile_num[0] * tile_num[1] * tile_num[2]

    def execution_plan(self) -> ExecutionPlan:
        """Build the shared region, fetch, and tile-action contract."""

        edges = logical_edges(self.spec)
        tiles = {tile.name: tile for tile in self.spec.tiles}
        if self.workload == "allgather_gemm":
            if len(edges) != 1:
                raise ValueError("AllGather+GEMM requires exactly one logical edge")
            edge = edges[0]
            host_name = "allgather_host"
            device_name = "gemm_device"
            plan = ExecutionPlan(
                kernel=self.spec,
                device_regions=(
                    DeviceRegionPlan(
                        device_name,
                        fetch_program=SchedulerFetchProgram(
                            (
                                FetchGuardAction(
                                    (edge,),
                                    predicate="remote_rank != rank",
                                    payload={"event": edge.event},
                                ),
                            )
                        ),
                        tile_programs=(
                            TileActionProgram(tiles["gemm"], (RunAction(tiles["gemm"]),)),
                        ),
                        attrs={"scheduler": self.physical_scheduler},
                    ),
                ),
                host_regions=(
                    HostRegionPlan(
                        host_name, (HostCallAction(tiles["allgather"].impl.entrypoint),)
                    ),
                ),
                region_dependencies=(RegionDependencyPlan(host_name, device_name, "launch_order"),),
                edge_bindings=(EdgeBindingPlan(edge, "fetch_guard", device_name),),
                attrs={"policy": self.policy_name},
            )
        elif self.workload == "gemm_reduce_scatter":
            by_event = {edge.event.name: edge for edge in edges}
            partial_edge = by_event["partial_shard_ready"]
            staging_edge = by_event["staging_ready"]
            partial_name = "partial_gemm_device"
            transfer_name = "reduce_scatter_host"
            reduce_name = "reduce_device"
            plan = ExecutionPlan(
                kernel=self.spec,
                device_regions=(
                    DeviceRegionPlan(
                        partial_name,
                        tile_programs=(
                            TileActionProgram(
                                tiles["partial_gemm"],
                                (
                                    RunAction(tiles["partial_gemm"]),
                                    MidBodyPortAction(
                                        (partial_edge,), "after_store_before_pipeline_advance"
                                    ),
                                ),
                            ),
                        ),
                        attrs={"scheduler": self.physical_scheduler},
                    ),
                    DeviceRegionPlan(
                        reduce_name,
                        tile_programs=(
                            TileActionProgram(tiles["reduce"], (RunAction(tiles["reduce"]),)),
                        ),
                    ),
                ),
                host_regions=(
                    HostRegionPlan(
                        transfer_name,
                        (
                            HostCallAction(tiles["transfer"].impl.entrypoint),
                            HostEdgeAction((staging_edge,), "completion"),
                        ),
                    ),
                ),
                region_dependencies=(
                    RegionDependencyPlan(partial_name, transfer_name, "launch_order"),
                    RegionDependencyPlan(transfer_name, reduce_name, "completion"),
                ),
                edge_bindings=(
                    EdgeBindingPlan(
                        partial_edge,
                        "mid_body_port",
                        partial_name,
                        port="after_store_before_pipeline_advance",
                    ),
                    EdgeBindingPlan(staging_edge, "host_runtime", transfer_name),
                ),
                attrs={"policy": self.policy_name},
            )
        else:
            raise ValueError(f"unsupported distributed GEMM graph: {self.workload!r}")
        return plan.validate()

    def validate(self) -> GemmCommPlan:
        self.spec.validate()
        _validate_attrs(self.spec.attrs, owner="kernel attrs")
        for event in self.spec.events.values():
            _validate_attrs(event.attrs, owner=f"event {event.name!r} attrs")
        for tile in self.spec.tiles:
            _validate_attrs(tile.attrs, owner=f"tile {tile.name!r} attrs")
            for attribute in ("execution_space", "entrypoint", "tensor_specs", "run"):
                if not hasattr(tile.impl, attribute):
                    raise TypeError(
                        f"tile {tile.name!r} has an incompatible distributed GEMM TileImpl"
                    )

        if self.policy_name not in {"static", "dynamic"}:
            raise ValueError(f"unsupported policy {self.policy_name!r}")
        if self.persistent_clusters <= 0:
            raise ValueError("persistent cluster count must be positive")
        if tuple(schedule.rank for schedule in self.rank_schedules) != tuple(
            range(self.world_size)
        ):
            raise ValueError("rank schedules must be contiguous and rank ordered")
        if self.lowerable != (self.unsupported_reason is None):
            raise ValueError("lowerable state and unsupported reason disagree")

        tile_num = tuple(self.tile.tile_num)
        expected = {
            PhysicalTask(self.scheduled_tile, m, n, k)
            for m in range(tile_num[0])
            for n in range(tile_num[1])
            for k in range(tile_num[2])
        }
        for schedule in self.rank_schedules:
            if self.is_dynamic:
                if not schedule.shared_queue or schedule.worker_queues:
                    raise ValueError("dynamic policy must produce exactly one shared rank queue")
            elif schedule.shared_queue or len(schedule.worker_queues) != self.persistent_clusters:
                raise ValueError("static policy must produce one queue per persistent cluster")
            tasks = schedule.tasks
            if len(tasks) != len(expected) or set(tasks) != expected:
                raise ValueError(
                    f"rank {schedule.rank} schedule does not cover every logical tile exactly once"
                )

        if not self.launch_steps or any(not group for group in self.launch_steps):
            raise ValueError("launch plan must contain non-empty execution groups")
        stage_names = {tile.name for tile in self.spec.tiles}
        if set(name for group in self.launch_steps for name in group) != stage_names:
            raise ValueError("launch plan must cover every logical stage")
        self.execution_plan()
        return self

    def normalized_data(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "policy": self.policy_name,
            "scheduled_tile": self.scheduled_tile,
            "persistent_clusters": self.persistent_clusters,
            "physical_scheduler": self.physical_scheduler,
            "task_count_per_rank": self.task_count_per_rank,
            "launch_steps": self.launch_steps,
            "lowerable": self.lowerable,
            "unsupported_reason": self.unsupported_reason,
            "rank_schedules": [
                {
                    "rank": schedule.rank,
                    "shared_queue": [task.indices for task in schedule.shared_queue],
                    "worker_task_counts": [len(queue) for queue in schedule.worker_queues],
                }
                for schedule in self.rank_schedules
            ],
        }


__all__ = ["GemmCommPlan", "PhysicalTask", "RankSchedule"]
