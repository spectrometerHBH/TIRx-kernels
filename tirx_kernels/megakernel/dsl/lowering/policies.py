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

"""Static, unfused, and dynamic policies for normalized MoE plans."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tirx_kernels.megakernel.utils.base import SemaphoreBase
from tirx_kernels.megakernel.utils.config import JobType, KernelConfig
from tirx_kernels.megakernel.utils.dynamic_scheduler import DynamicTileScheduler
from tirx_kernels.megakernel.utils.static_scheduler import StaticTileScheduler
from tvm.megakernel.transform import DeviceRegionPlan, ExecutionPlan

from ..moe_spec import build_moe_graph
from ..spec import KernelSpec
from .model import DynamicProtocolPlan, HostTask, MoeLoweringEnv, NormalizedPlan
from .normalize import (
    _attach_dispatch_steps,
    _dynamic_dispatch_rules,
    _enumerate_tile,
    _event_plans,
    _normalize_dispatch,
    _normalize_tiles,
    _validate_packed_tiles,
    _validate_policy_edges,
)


class MoePolicy:
    name = "base"
    is_dynamic = False
    unfused = False

    def __init__(self, *, queue_capacity: int | None = None):
        self.queue_capacity = queue_capacity

    def _lowering_env(self, spec: KernelSpec) -> MoeLoweringEnv:
        env = MoeLoweringEnv(spec)
        _validate_persistent_event_dependencies_acyclic(spec, self.name)
        return env

    def normalize(self, spec: KernelSpec) -> NormalizedPlan:
        raise NotImplementedError


def _validate_persistent_event_dependencies_acyclic(spec: KernelSpec, policy_name: str) -> None:
    """Reject tile-level cycles that can occupy every persistent worker before progress.

    This is intentionally stricter than the coordinate-projected check in TVM's
    reference static backend.  The current MoE policies schedule complete physical
    tile kinds and cannot represent a coordinate-disjoint cyclic phase.  A future
    coordinate-aware policy should consume that richer schedule explicitly instead
    of weakening this persistent-queue safety gate.
    """

    tile_ids = [id(tile) for tile in spec.tiles]
    adjacency = {tile_id: set() for tile_id in tile_ids}
    producers: dict[int, list[int]] = {}
    for tile in spec.tiles:
        for event, _ in tile.notifies:
            producers.setdefault(id(event), []).append(id(tile))
    for consumer in spec.tiles:
        for event, _ in consumer.waits:
            for producer_id in producers.get(id(event), ()):
                adjacency[producer_id].add(id(consumer))

    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(tile_id: int) -> None:
        if tile_id in visiting:
            raise ValueError(
                f"MoE {policy_name!r} persistent queue requires logical event "
                "dependencies to be acyclic"
            )
        if tile_id in visited:
            return
        visiting.add(tile_id)
        for consumer_id in adjacency[tile_id]:
            visit(consumer_id)
        visiting.remove(tile_id)
        visited.add(tile_id)

    for tile_id in tile_ids:
        visit(tile_id)


class StaticPolicy(MoePolicy):
    name = "static"

    def normalize(self, spec: KernelSpec) -> NormalizedPlan:
        env = self._lowering_env(spec)
        events = _event_plans(env, is_dynamic=False, unfused=self.unfused)
        tiles = _normalize_tiles(
            env, is_dynamic=False, unfused=self.unfused, down_coalescing=1, events=events
        )
        _validate_packed_tiles(tiles)
        by_name = {tile.tile.name: tile for tile in tiles}
        central = [
            HostTask(JobType.INIT_ETENSOR.value, event_idx, 0, 0)
            for event_idx in range(len(events))
        ]
        central.extend(_enumerate_tile(by_name["gating"]))
        central.extend(
            HostTask(JobType.WAIT_ETENSOR_INIT.value, cta, 0, 0)
            for cta in range(KernelConfig.SM_NUMBER)
        )
        for tile in tiles:
            if tile.tile.name != "gating":
                central.extend(_enumerate_tile(tile))
        queue_columns = (len(central) + KernelConfig.SM_NUMBER - 1) // KernelConfig.SM_NUMBER + 1
        capacity = (
            StaticTileScheduler.MAX_TASKS if self.queue_capacity is None else self.queue_capacity
        )
        if capacity != StaticTileScheduler.MAX_TASKS or queue_columns > capacity:
            raise ValueError(
                f"static host queue requires {queue_columns} columns, capacity is {capacity}"
            )
        execution = ExecutionPlan(
            kernel=spec,
            device_regions=(
                DeviceRegionPlan("moe_device", tile_programs=tiles, attrs={"schedule": self.name}),
            ),
        ).validate()
        return NormalizedPlan(
            execution=execution,
            env=env,
            policy_name=self.name,
            is_dynamic=False,
            unfused=self.unfused,
            events=events,
            central_tasks=tuple(central),
            seed_tasks=(),
            down_coalescing=1,
            down_dispatch_groups=16,
            queue_capacity=capacity,
            queue_upper_bound=queue_columns,
            persistent_ctas=KernelConfig.SM_NUMBER,
            protocol=None,
        ).validate()


class UnfusedPolicy(StaticPolicy):
    name = "unfused"
    unfused = True


class DynamicPolicy(MoePolicy):
    name = "dynamic"
    is_dynamic = True

    def __init__(
        self,
        *,
        down_coalescing: int | None = None,
        queue_capacity: int = DynamicTileScheduler.MAX_TASKS,
    ):
        super().__init__(queue_capacity=queue_capacity)
        self.down_coalescing = down_coalescing

    def normalize(self, spec: KernelSpec) -> NormalizedPlan:
        env = self._lowering_env(spec)
        batch_size = env.batch_size
        expected_coalescing = 1 if batch_size < 4 else 4
        coalescing = expected_coalescing if self.down_coalescing is None else self.down_coalescing
        if coalescing != expected_coalescing or coalescing <= 0 or 16 % coalescing:
            raise ValueError(
                f"illegal dynamic down coalescing q={coalescing} for batch {batch_size}"
            )
        capacity = self.queue_capacity
        if capacity is None or capacity <= 0 or capacity & (capacity - 1):
            raise ValueError("dynamic queue capacity must be a positive power of two")
        down_dispatch_groups = 16 // coalescing
        events = _event_plans(env, is_dynamic=True, unfused=False)
        tiles = _normalize_tiles(
            env, is_dynamic=True, unfused=False, down_coalescing=coalescing, events=events
        )
        _validate_packed_tiles(tiles)
        dispatch_rules = _dynamic_dispatch_rules(env, down_dispatch_groups)
        _validate_policy_edges(tiles, dispatch_rules)
        dispatch_steps = _normalize_dispatch(env, tiles, dispatch_rules, events)
        tiles = _attach_dispatch_steps(tiles, dispatch_steps)
        by_name = {tile.tile.name: tile for tile in tiles}
        seed = [
            HostTask(JobType.INIT_ETENSOR.value, event_idx, 0, 0)
            for event_idx in range(len(events))
        ]
        seed.extend(_enumerate_tile(by_name["gating"]))
        queue_upper_bound = len(seed) + sum(
            dispatch.enqueue_upper_bound for dispatch in dispatch_steps
        )
        if queue_upper_bound > capacity:
            raise ValueError(
                f"dynamic queue upper bound {queue_upper_bound} exceeds capacity {capacity}"
            )
        if capacity != DynamicTileScheduler.MAX_TASKS:
            raise ValueError(
                f"dynamic queue capacity must remain {DynamicTileScheduler.MAX_TASKS}; got {capacity}"
            )
        execution = ExecutionPlan(
            kernel=spec,
            device_regions=(
                DeviceRegionPlan("moe_device", tile_programs=tiles, attrs={"schedule": self.name}),
            ),
        ).validate()
        return NormalizedPlan(
            execution=execution,
            env=env,
            policy_name=self.name,
            is_dynamic=True,
            unfused=False,
            events=events,
            central_tasks=(),
            seed_tasks=tuple(seed),
            down_coalescing=coalescing,
            down_dispatch_groups=down_dispatch_groups,
            queue_capacity=capacity,
            queue_upper_bound=queue_upper_bound,
            persistent_ctas=KernelConfig.SM_NUMBER,
            protocol=DynamicProtocolPlan(
                pre_decrement=1,
                post_decrement=SemaphoreBase.base,
                scheduler_warp=DynamicTileScheduler.scheduler_warp,
                queue_discipline="fifo",
            ),
        ).validate()


def policy_for_scheduler(scheduler: str) -> MoePolicy:
    if scheduler == "static":
        return StaticPolicy()
    if scheduler == "unfused":
        return UnfusedPolicy()
    if scheduler == "dynamic":
        return DynamicPolicy()
    raise ValueError(f"unsupported MoE scheduler: {scheduler!r}")


def make_moe_plan(config: Mapping[str, Any], batch_size: int, scheduler: str) -> NormalizedPlan:
    # Local import avoids a policies <-> emission module cycle.
    from .lowerer import MoeLowerer

    graph = build_moe_graph(config, batch_size)
    return MoeLowerer(policy_for_scheduler(scheduler)).lower(graph)
