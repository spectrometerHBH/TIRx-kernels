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

"""Validated physical-plan model for the direct GemmComm kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tirx_kernels._attrs import validate_no_nested_attr_keys
from tvm.megakernel.dsl import KernelSpec
from tvm.megakernel.transform import (
    DeviceRegionPlan,
    ExecutionPlan,
    FetchGuardStep,
    HostCallStep,
    HostRegionPlan,
    QueuePushStep,
    RunStep,
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


@dataclass(frozen=True, order=True)
class PhysicalTask:
    """One queue item referring to a logical tile kind and physical coordinates."""

    tile: str
    m: int
    n: int
    k: int = 0

    @property
    def indices(self) -> tuple[int, int, int]:
        return (self.m, self.n, self.k)


@dataclass(frozen=True)
class RankSchedule:
    """Initial dynamic work plus tasks published remotely by producer completion."""

    rank: int
    shared_queue: tuple[PhysicalTask, ...] = ()
    pushed_tasks: tuple[PhysicalTask, ...] = ()
    worker_queues: tuple[tuple[PhysicalTask, ...], ...] = ()

    @property
    def tasks(self) -> tuple[PhysicalTask, ...]:
        if self.shared_queue:
            return self.shared_queue
        return tuple(task for queue in self.worker_queues for task in queue)


@dataclass(frozen=True)
class GemmCommPlan:
    """One authoritative dynamic execution plan and its rank-local queue coverage."""

    execution: ExecutionPlan
    persistent_clusters: int
    rank_schedules: tuple[RankSchedule, ...]

    @property
    def spec(self) -> KernelSpec:
        return self.execution.kernel

    @property
    def workload(self) -> str:
        return self.spec.name

    @property
    def policy_name(self) -> str:
        policy = self.execution.attrs.get("policy")
        if not isinstance(policy, str) or not policy:
            raise ValueError("distributed GEMM execution has no policy")
        return policy

    @property
    def world_size(self) -> int:
        return len(self.rank_schedules)

    @property
    def is_dynamic(self) -> bool:
        return self.policy_name == "dynamic"

    @property
    def lowerable(self) -> bool:
        return True

    @property
    def unsupported_reason(self) -> None:
        return None

    @property
    def scheduled_region(self) -> DeviceRegionPlan:
        regions = [
            region for region in self.execution.device_regions if "scheduler" in region.attrs
        ]
        if len(regions) != 1:
            raise ValueError("distributed GEMM requires exactly one scheduled device region")
        return regions[0]

    @property
    def physical_scheduler(self) -> str:
        scheduler = self.scheduled_region.attrs.get("scheduler")
        if not isinstance(scheduler, str) or not scheduler:
            raise ValueError("scheduled device region has no physical scheduler")
        return scheduler

    @property
    def scheduled_tile(self) -> str:
        return self.scheduled_region.tile_programs[0].tile.name

    @property
    def task_count_per_rank(self) -> int:
        return len(self.rank_schedules[0].tasks)

    @property
    def pushed_task_count_per_rank(self) -> int:
        return len(self.rank_schedules[0].pushed_tasks)

    def region(self, name: str) -> DeviceRegionPlan | HostRegionPlan:
        matching = [
            region
            for region in (*self.execution.device_regions, *self.execution.host_regions)
            if region.name == name
        ]
        if len(matching) != 1:
            raise ValueError(f"execution requires exactly one region named {name!r}")
        return matching[0]

    def entrypoint_for(self, region_name: str) -> str:
        entrypoint = self.region(region_name).attrs.get("entrypoint")
        if not isinstance(entrypoint, str) or not entrypoint:
            raise ValueError(f"region {region_name!r} has no backend entrypoint")
        return entrypoint

    def _validate_rank_schedules(self) -> None:
        if not self.rank_schedules:
            raise ValueError("distributed GEMM requires at least one rank schedule")
        if tuple(schedule.rank for schedule in self.rank_schedules) != tuple(
            range(self.world_size)
        ):
            raise ValueError("rank schedules must be contiguous and rank ordered")
        if self.policy_name != "dynamic":
            raise ValueError("the direct GemmComm kernels require the dynamic policy")
        if any(
            not schedule.shared_queue or schedule.worker_queues for schedule in self.rank_schedules
        ):
            raise ValueError("dynamic policy must produce exactly one initial queue per rank")

        if self.workload == "allgather_gemm":
            config = self.spec.tiles[1].impl.config
            expected = {
                PhysicalTask("gemm", m_idx, n_idx)
                for m_idx in range(config.gemm_m_clusters)
                for n_idx in range(config.gemm_n_clusters)
            }
            pushed = set()
        else:
            config = self.spec.tiles[0].impl.config
            expected = {
                PhysicalTask("partial_gemm", m_idx, n_idx)
                for m_idx in range(config.gemm_m_clusters)
                for n_idx in range(config.gemm_n_clusters)
            }
            pushed = {
                PhysicalTask("reduce_scatter", m_idx, n_idx)
                for m_idx in range(config.rs_m_clusters)
                for n_idx in range(config.rs_n_clusters)
            }

        for schedule in self.rank_schedules:
            if len(schedule.tasks) != len(expected) or set(schedule.tasks) != expected:
                raise ValueError(
                    f"rank {schedule.rank} initial queue does not cover every physical task"
                )
            if len(schedule.pushed_tasks) != len(pushed) or set(schedule.pushed_tasks) != pushed:
                raise ValueError(
                    f"rank {schedule.rank} published queue does not cover every destination task"
                )

    def _validate_regions(self) -> None:
        physical_tiles = [
            program.tile.name
            for region in self.execution.device_regions
            for program in region.tile_programs
        ]
        physical_tiles.extend(
            tile.name for region in self.execution.host_regions for tile in region.owned_tiles
        )
        logical_tiles = [tile.name for tile in self.spec.tiles]
        if len(set(physical_tiles)) != len(physical_tiles) or set(physical_tiles) != set(
            logical_tiles
        ):
            raise ValueError("execution regions must cover every logical tile exactly once")

        placements = self.execution.edge_placements()
        if self.workload == "allgather_gemm":
            if len(self.execution.device_regions) != 1 or len(self.execution.host_regions) != 1:
                raise ValueError("AllGather+GEMM requires one host and one device region")
            device = self.execution.device_regions[0]
            host = self.execution.host_regions[0]
            if [program.tile.name for program in device.tile_programs] != ["gemm"]:
                raise ValueError("AllGather device region must contain only the GEMM tile")
            if (
                len(device.fetch_steps) != 1
                or not isinstance(device.fetch_steps[0], FetchGuardStep)
                or device.fetch_steps[0].predicate != "remote_rank != rank"
            ):
                raise ValueError("AllGather fetch program has an invalid guard")
            if (
                len(host.steps) != 1
                or not isinstance(host.steps[0], HostCallStep)
                or host.steps[0].name != "collective"
            ):
                raise ValueError("AllGather host region has an invalid collective")
            placement = placements[0]
            if (placement.location, placement.region) != ("fetch", "gemm_device"):
                raise ValueError("AllGather readiness must be placed at GEMM fetch")
        elif self.workload == "gemm_reduce_scatter":
            if len(self.execution.device_regions) != 1 or self.execution.host_regions:
                raise ValueError("fused GemmRS must contain one device region and no host region")
            region = self.execution.device_regions[0]
            if [program.tile.name for program in region.tile_programs] != [
                "partial_gemm",
                "reduce_scatter",
            ]:
                raise ValueError("fused GemmRS requires GEMM then ReduceScatter tile programs")
            gemm_steps = region.tile_programs[0].steps
            rs_steps = region.tile_programs[1].steps
            if tuple(type(step) for step in gemm_steps) != (RunStep, QueuePushStep):
                raise ValueError("GemmRS GEMM program must run before publishing RS work")
            if gemm_steps[0].repeat != 2:
                raise ValueError("one physical GemmRS task must cover two logical GEMM tiles")
            if tuple(type(step) for step in rs_steps) != (RunStep,):
                raise ValueError("GemmRS reduce program must contain one run step")
            placement = placements[0]
            if (placement.location, placement.region) != ("tile", "fused_gemm_rs_device"):
                raise ValueError(
                    "GemmRS readiness must be published inside the fused device region"
                )
        else:
            raise ValueError(f"unsupported distributed GEMM graph: {self.workload!r}")

    def validate(self) -> GemmCommPlan:
        self.execution.validate()
        validate_no_nested_attr_keys(
            self.spec.attrs, _FORBIDDEN_LOGICAL_FIELDS, owner="kernel attrs"
        )
        for event in self.spec.events.values():
            validate_no_nested_attr_keys(
                event.attrs, _FORBIDDEN_LOGICAL_FIELDS, owner=f"event {event.name!r} attrs"
            )
        for tile in self.spec.tiles:
            validate_no_nested_attr_keys(
                tile.attrs, _FORBIDDEN_LOGICAL_FIELDS, owner=f"tile {tile.name!r} attrs"
            )
            if not hasattr(tile.impl, "tensor_specs") or not callable(
                getattr(tile.impl, "run", None)
            ):
                raise TypeError(f"tile {tile.name!r} has an incompatible TileImpl")
        if self.persistent_clusters <= 0:
            raise ValueError("persistent cluster count must be positive")
        for region in (*self.execution.device_regions, *self.execution.host_regions):
            self.entrypoint_for(region.name)
        self._validate_rank_schedules()
        self._validate_regions()
        return self

    def normalized_data(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "policy": self.policy_name,
            "persistent_clusters": self.persistent_clusters,
            "physical_scheduler": self.physical_scheduler,
            "task_count_per_rank": self.task_count_per_rank,
            "pushed_task_count_per_rank": self.pushed_task_count_per_rank,
            "regions": [
                {
                    "name": region.name,
                    "kind": "device" if isinstance(region, DeviceRegionPlan) else "host",
                    "entrypoint": region.attrs["entrypoint"],
                    "tiles": (
                        [program.tile.name for program in region.tile_programs]
                        if isinstance(region, DeviceRegionPlan)
                        else [tile.name for tile in region.owned_tiles]
                    ),
                }
                for region in self.execution.regions_in_dependency_order()
            ],
            "rank_schedules": [
                {
                    "rank": schedule.rank,
                    "initial_tasks": [task.indices for task in schedule.tasks],
                    "pushed_tasks": [task.indices for task in schedule.pushed_tasks],
                }
                for schedule in self.rank_schedules
            ],
        }


__all__ = ["GemmCommPlan", "PhysicalTask", "RankSchedule"]
