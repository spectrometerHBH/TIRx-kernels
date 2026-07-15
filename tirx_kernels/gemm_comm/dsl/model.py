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
    ExecutionPlan,
    FetchGuardStep,
    HostCallStep,
    HostRegionPlan,
    HostSyncStep,
    MidBodyPortStep,
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
    """Validated policy output with one authoritative physical execution."""

    execution: ExecutionPlan
    persistent_clusters: int
    rank_schedules: tuple[RankSchedule, ...]
    lowerable: bool = True
    unsupported_reason: str | None = None

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
    def scheduled_region(self) -> DeviceRegionPlan:
        regions = [
            region for region in self.execution.device_regions if "scheduler" in region.attrs
        ]
        if len(regions) != 1:
            raise ValueError("distributed GEMM requires exactly one scheduled device region")
        return regions[0]

    @property
    def physical_scheduler(self) -> str:
        scheduler = self.scheduled_region.attrs["scheduler"]
        if not isinstance(scheduler, str) or not scheduler:
            raise ValueError("scheduled device region has no physical scheduler")
        return scheduler

    @property
    def scheduled_tile(self) -> str:
        programs = self.scheduled_region.tile_programs
        if len(programs) != 1:
            raise ValueError("scheduled device region requires exactly one tile program")
        return programs[0].tile.name

    @property
    def tile(self) -> TileSpec:
        return next(tile for tile in self.spec.tiles if tile.name == self.scheduled_tile)

    @property
    def task_count_per_rank(self) -> int:
        tile_num = tuple(self.tile.tile_num)
        if not all(isinstance(extent, int) for extent in tile_num):
            raise TypeError("distributed GEMM plans require compile-time tile extents")
        return tile_num[0] * tile_num[1] * tile_num[2]

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

    def validate(self) -> GemmCommPlan:
        self.execution.validate()
        _validate_attrs(self.spec.attrs, owner="kernel attrs")
        for event in self.spec.events.values():
            _validate_attrs(event.attrs, owner=f"event {event.name!r} attrs")
        for tile in self.spec.tiles:
            _validate_attrs(tile.attrs, owner=f"tile {tile.name!r} attrs")
            for attribute in ("tensor_specs", "run"):
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

        regions = (*self.execution.device_regions, *self.execution.host_regions)
        for region in regions:
            self.entrypoint_for(region.name)

        physical_tiles = [
            program.tile.name
            for region in self.execution.device_regions
            for program in region.tile_programs
        ]
        for region in self.execution.host_regions:
            tile_name = region.attrs.get("tile")
            if not isinstance(tile_name, str) or not tile_name:
                raise ValueError(f"host region {region.name!r} has no logical tile")
            physical_tiles.append(tile_name)
        logical_tiles = [tile.name for tile in self.spec.tiles]
        if len(set(physical_tiles)) != len(physical_tiles) or set(physical_tiles) != set(
            logical_tiles
        ):
            raise ValueError("execution regions must cover every logical tile exactly once")

        expected_tasks = {
            PhysicalTask(self.scheduled_tile, m, n, k)
            for m in range(self.tile.tile_num[0])
            for n in range(self.tile.tile_num[1])
            for k in range(self.tile.tile_num[2])
        }
        for schedule in self.rank_schedules:
            if self.is_dynamic:
                if not schedule.shared_queue or schedule.worker_queues:
                    raise ValueError("dynamic policy must produce exactly one shared rank queue")
            elif schedule.shared_queue or len(schedule.worker_queues) != self.persistent_clusters:
                raise ValueError("static policy must produce one queue per persistent cluster")
            tasks = schedule.tasks
            if len(tasks) != len(expected_tasks) or set(tasks) != expected_tasks:
                raise ValueError(
                    f"rank {schedule.rank} schedule does not cover every logical tile exactly once"
                )

        placements = {
            placement.edge.event.name: placement for placement in self.execution.edge_placements()
        }
        if self.workload == "allgather_gemm":
            expected_regions = {"allgather_host", "gemm_device"}
            placement = placements.get("shard_ready")
            if placement is None or (placement.location, placement.region, placement.port) != (
                "fetch",
                "gemm_device",
                None,
            ):
                raise ValueError("AllGather shard-ready edge must be placed at GEMM fetch")
            device = self.region("gemm_device")
            host = self.region("allgather_host")
            if not isinstance(device, DeviceRegionPlan) or not isinstance(host, HostRegionPlan):
                raise ValueError("AllGather execution region kinds are invalid")
            if (
                len(device.fetch_steps) != 1
                or not isinstance(device.fetch_steps[0], FetchGuardStep)
                or device.fetch_steps[0].predicate != "remote_rank != rank"
            ):
                raise ValueError("AllGather fetch program has an invalid guard step")
            if tuple(type(step) for step in device.tile_programs[0].steps) != (RunStep,):
                raise ValueError("AllGather GEMM program must contain exactly one run step")
            if (
                len(host.steps) != 1
                or not isinstance(host.steps[0], HostCallStep)
                or host.steps[0].name != "collective"
            ):
                raise ValueError("AllGather host region has an invalid collective step")
        elif self.workload == "gemm_reduce_scatter":
            expected_regions = {"partial_gemm_device", "reduce_scatter_host", "reduce_device"}
            partial = placements.get("partial_shard_ready")
            staging = placements.get("staging_ready")
            if partial is None or (partial.location, partial.region, partial.port) != (
                "tile",
                "partial_gemm_device",
                "after_store_before_pipeline_advance",
            ):
                raise ValueError("partial-ready edge must use the approved GEMM epilogue port")
            if staging is None or (staging.location, staging.region, staging.port) != (
                "host",
                "reduce_scatter_host",
                None,
            ):
                raise ValueError("staging-ready edge must be completed by the host collective")
            partial_region = self.region("partial_gemm_device")
            reduce_region = self.region("reduce_device")
            host = self.region("reduce_scatter_host")
            if (
                not isinstance(partial_region, DeviceRegionPlan)
                or not isinstance(reduce_region, DeviceRegionPlan)
                or not isinstance(host, HostRegionPlan)
            ):
                raise ValueError("GEMM+ReduceScatter execution region kinds are invalid")
            partial_steps = partial_region.tile_programs[0].steps
            reduce_steps = reduce_region.tile_programs[0].steps
            if tuple(type(step) for step in partial_steps) != (RunStep, MidBodyPortStep):
                raise ValueError("partial GEMM program has an invalid physical step order")
            if tuple(type(step) for step in reduce_steps) != (RunStep,):
                raise ValueError("reduce program must contain exactly one run step")
            if (
                len(host.steps) != 2
                or not isinstance(host.steps[0], HostCallStep)
                or host.steps[0].name != "collective"
                or not isinstance(host.steps[1], HostSyncStep)
                or host.steps[1].kind != "communication_completion"
            ):
                kind = getattr(host.steps[-1], "kind", None) if host.steps else None
                raise ValueError(f"reduce-scatter host region has invalid sync step {kind!r}")
        else:
            raise ValueError(f"unsupported distributed GEMM graph: {self.workload!r}")
        if {region.name for region in regions} != expected_regions:
            raise ValueError("execution regions do not match the distributed GEMM workload")
        return self

    def normalized_data(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "policy": self.policy_name,
            "scheduled_tile": self.scheduled_tile,
            "persistent_clusters": self.persistent_clusters,
            "physical_scheduler": self.physical_scheduler,
            "task_count_per_rank": self.task_count_per_rank,
            "regions": [
                {
                    "name": region.name,
                    "kind": "device" if isinstance(region, DeviceRegionPlan) else "host",
                    "entrypoint": region.attrs["entrypoint"],
                    "scheduler": region.attrs.get("scheduler"),
                    "tiles": (
                        [program.tile.name for program in region.tile_programs]
                        if isinstance(region, DeviceRegionPlan)
                        else [region.attrs["tile"]]
                    ),
                    "steps": (
                        [
                            type(step).__name__
                            for program in region.tile_programs
                            for step in program.steps
                        ]
                        if isinstance(region, DeviceRegionPlan)
                        else [type(step).__name__ for step in region.steps]
                    ),
                }
                for region in self.execution.regions_in_dependency_order()
            ],
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
