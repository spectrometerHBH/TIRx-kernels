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

"""Private normalized-plan data model for MoE lowering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Any

import numpy as np

from tirx_kernels.megakernel.utils.base import SemaphoreBase
from tirx_kernels.megakernel.utils.config import JobType, KernelConfig
from tirx_kernels.megakernel.utils.dynamic_scheduler import DynamicTileScheduler, MPMCQueueHost
from tirx_kernels.megakernel.utils.static_scheduler import StaticTileScheduler
from tirx_kernels.megakernel.utils.utils import pack_into_32bit
from tvm.megakernel.transform import (
    BarrierStep,
    ExecutionPlan,
    NotifyStep,
    QueuePushStep,
    RunStep,
    RuntimeEventInitStep,
    TileProgram,
    WaitStep,
)

from .._expr import ConstExpr, Expr, ScalarLoadExpr, TileIndexExpr, as_expr, walk_expr
from ..examples.moe import _max_rows
from ..spec import DependencyType, EventSpec, KernelSpec, TileSpec, VarSpec

_FORBIDDEN_SPEC_FIELDS = {
    "dispatch",
    "job_type",
    "level",
    "mask",
    "queue",
    "rank",
    "release",
    "runtime_init",
    "scope",
    "scope_id",
}


def _shape_tuple(shape) -> tuple[int | VarSpec, ...]:
    if isinstance(shape, int | VarSpec):
        return (shape,)
    return tuple(shape)


def _validate_logical_attrs(attrs: Mapping[str, Any], *, owner: str) -> None:
    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in _FORBIDDEN_SPEC_FIELDS:
                    raise ValueError(f"{owner} contains scheduler field {path + key!r}")
                visit(item, f"{path}{key}.")
        elif isinstance(value, tuple | list):
            for index, item in enumerate(value):
                visit(item, f"{path}{index}.")

    visit(attrs, "")


class MoeLoweringEnv:
    """Resolve native logical symbols and producer facts for MoE lowering."""

    def __init__(self, spec: KernelSpec):
        spec.validate()
        self.spec = spec
        self.tensor_map = dict(spec.tensors)
        self.event_map = dict(spec.events)
        self.tile_map = {tile.name: tile for tile in spec.tiles}
        if tuple(self.tile_map) != (
            "gating",
            "topk",
            "align",
            "count_sort",
            "gate_up_silu",
            "down",
        ):
            raise ValueError("MoE logical graph must preserve the six canonical stages")
        if tuple(self.event_map) != (
            "gating_done",
            "topk_done",
            "align_done",
            "count_sort_done",
            "gate_up_done",
        ):
            raise ValueError("MoE logical graph must contain exactly five logical events")

        hidden_shape = _shape_tuple(self.tensor_map["hidden_state"].shape)
        if len(hidden_shape) != 2 or not isinstance(hidden_shape[0], int):
            raise ValueError("hidden_state must provide a compile-time batch extent")
        self.batch_size = hidden_shape[0]
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.compile_env = {"B": self.batch_size}
        self.rmax = _max_rows(self.batch_size)
        scalar_shape = tuple(ConstExpr(value) for value in _shape_tuple((1,)))
        self.routed_rows = (
            ScalarLoadExpr("num_tokens_post_pad", ConstExpr(0), "int32", scalar_shape) // 128
        )

        _validate_logical_attrs(spec.attrs, owner="kernel attrs")
        for event in spec.events.values():
            _validate_logical_attrs(event.attrs, owner=f"event {event.name!r} attrs")
        for tile in spec.tiles:
            _validate_logical_attrs(tile.attrs, owner=f"tile {tile.name!r} attrs")
            for name in ("implementation", "job_type", "profile_event_type", "tensor_bindings"):
                if not hasattr(tile.impl, name):
                    raise TypeError(f"tile {tile.name!r} has an incompatible MoE TileImpl")

        self.tensor_producers: dict[str, str] = {}
        for tile in spec.tiles:
            for tensor in tile.writes:
                previous = self.tensor_producers.setdefault(tensor.name, tile.name)
                if previous != tile.name:
                    raise ValueError(f"tensor {tensor.name!r} has multiple tile producers")
        if self.tensor_producers.get("num_tokens_post_pad") != "align":
            raise ValueError("routed_rows must be produced by the align tile")
        runtime_tensor = self.tensor_map["num_tokens_post_pad"]
        if runtime_tensor.dtype not in {"int8", "int16", "int32", "int64"}:
            raise ValueError("routed_rows must be loaded from an integer tensor")
        if _shape_tuple(runtime_tensor.shape) != (1,):
            raise ValueError("num_tokens_post_pad must be a one-element tensor")

        for tile_name in ("gate_up_silu", "down"):
            tile = self.tile_map[tile_name]
            tile_num = tuple(tile.tile_num)
            if tile_num[0] is not spec.vars["routed_rows"]:
                raise ValueError(f"tile {tile_name!r} must use VarSpec('routed_rows') on axis 0")
            if runtime_tensor not in tile.reads:
                raise ValueError(f"tile {tile_name!r} must read num_tokens_post_pad")
        self._validate_runtime_producer_order()

    def _validate_runtime_producer_order(self):
        edges = {tile.name: set() for tile in self.spec.tiles}
        notifiers: dict[int, list[str]] = {}
        for tile in self.spec.tiles:
            for event, _ in tile.notifies:
                notifiers.setdefault(id(event), []).append(tile.name)
        for tile in self.spec.tiles:
            for event, _ in tile.waits:
                for producer in notifiers.get(id(event), ()):
                    edges[producer].add(tile.name)

        def reachable(source: str, target: str) -> bool:
            pending = [source]
            visited = set()
            while pending:
                current = pending.pop()
                if current == target:
                    return True
                if current in visited:
                    continue
                visited.add(current)
                pending.extend(edges[current])
            return False

        for consumer in ("gate_up_silu", "down"):
            if not reachable("align", consumer):
                raise ValueError(f"routed_rows is used before align completes for {consumer!r}")

    def extent(self, value: int | VarSpec) -> Expr:
        if isinstance(value, int) and not isinstance(value, bool):
            return ConstExpr(value)
        if value is self.spec.vars["routed_rows"]:
            return self.routed_rows
        raise ValueError(f"unsupported MoE runtime extent {value!r}")

    def upper_bound(self, value: int | VarSpec) -> int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if value is self.spec.vars["routed_rows"]:
            return self.rmax
        raise ValueError(f"runtime extent {value!r} has no static upper bound")

    def event_shape(self, event: EventSpec) -> tuple[int, ...]:
        return tuple(self.upper_bound(extent) for extent in _shape_tuple(event.shape))

    def event_init_count(self, event: EventSpec) -> int:
        if isinstance(event.init_count, int) and not isinstance(event.init_count, bool):
            return event.init_count
        if not callable(event.init_count):
            raise TypeError(f"event {event.name!r} has an invalid init_count")
        shape = self.event_shape(event)
        samples = [(0,) * len(shape), tuple(extent - 1 for extent in shape)]
        counts = [event.init_count(coord) for coord in samples]
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in counts
        ):
            raise ValueError(f"event {event.name!r} init_count must return a positive integer")
        if len(set(counts)) != 1:
            raise ValueError(f"event {event.name!r} must have a uniform physical init_count")
        return counts[0]

    def coord(self, tile: TileSpec, dependency: DependencyType) -> tuple[Expr, ...]:
        event, coord_map = dependency
        indices = tuple(TileIndexExpr(tile.name, axis) for axis in range(3))
        if callable(coord_map):
            try:
                first = coord_map(*indices)
                second = coord_map(*indices)
            except Exception as err:  # pylint: disable=broad-exception-caught
                raise ValueError(
                    f"tile {tile.name!r} coordinate map cannot be expanded symbolically"
                ) from err
            if type(first) is not type(second) or first != second:
                raise ValueError(f"tile {tile.name!r} has an impure coordinate map")
            values = first
        else:
            values = coord_map
        if not isinstance(values, tuple | list):
            raise ValueError(f"tile {tile.name!r} coordinate map must return tuple or list")
        if len(values) != len(_shape_tuple(event.shape)):
            raise ValueError(f"tile {tile.name!r} coordinate rank does not match its event")
        result = tuple(as_expr(value) for value in values)
        for expr in result:
            for node in walk_expr(expr):
                if isinstance(node, TileIndexExpr) and node.task != tile.name:
                    raise ValueError(f"tile {tile.name!r} coordinate uses a foreign tile index")
        return result


@dataclass(frozen=True)
class EventPlan:
    name: str
    shape: tuple[int, ...]
    init_count: int | None
    workspace_offset: int
    logical_spec: EventSpec | None = None

    @property
    def size(self) -> int:
        return reduce(mul, self.shape, 1)

    @property
    def is_logical(self) -> bool:
        return self.logical_spec is not None


@dataclass(frozen=True, kw_only=True)
class MoeTileProgram(TileProgram):
    """One MoE tile's complete physical program and proven task bounds."""

    runtime_extents: tuple[Expr, Expr, Expr]
    upper_bounds: tuple[int, int, int]
    scheduled_extents: tuple[Expr, Expr, Expr]
    scheduled_upper_bounds: tuple[int, int, int]

    @property
    def implementation(self) -> str:
        return self.tile.impl.implementation

    @property
    def job_type(self) -> int:
        return self.tile.impl.job_type

    @property
    def waits(self) -> tuple[WaitStep, ...]:
        return tuple(step for step in self.steps if isinstance(step, WaitStep))

    @property
    def notifies(self) -> tuple[NotifyStep, ...]:
        return tuple(step for step in self.steps if isinstance(step, NotifyStep))


@dataclass(frozen=True)
class _DynamicDispatchRule:
    source_tile: str
    event: str
    event_coord: tuple[Expr, ...]
    target_tile: str | None
    count: Expr
    tile_indices: tuple[Expr, Expr, Expr]
    push_level: str
    pre_scope: str
    pre_scope_id: int = 0
    pre_count: Expr = field(default_factory=lambda: ConstExpr(1))
    rank: int = -1


@dataclass(frozen=True, kw_only=True)
class DynamicDispatchStep(QueuePushStep):
    """A physical dispatch step with statically proven trigger and index ranges."""

    rule: _DynamicDispatchRule
    trigger_upper_bound: int
    count_lower_bound: int
    count_upper_bound: int
    enqueue_upper_bound: int
    event_coord_bounds: tuple[tuple[int, int], ...]
    tile_index_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class DynamicProtocolPlan:
    """Scheduler constants required by the dynamic two-phase semaphore protocol."""

    pre_decrement: int
    post_decrement: int
    scheduler_warp: int
    queue_discipline: str


@dataclass(frozen=True)
class HostTask:
    job_type: int
    m_idx: int
    n_idx: int
    k_idx: int

    def packed(self) -> int:
        return pack_into_32bit(self.m_idx, self.n_idx, self.k_idx, self.job_type)

    def as_manual_tuple(self) -> tuple[int, int, int, int]:
        return (self.m_idx, self.n_idx, self.k_idx, self.job_type)


@dataclass(frozen=True)
class NormalizedPlan:
    execution: ExecutionPlan
    env: MoeLoweringEnv
    policy_name: str
    is_dynamic: bool
    unfused: bool
    events: tuple[EventPlan, ...]
    central_tasks: tuple[HostTask, ...]
    seed_tasks: tuple[HostTask, ...]
    down_coalescing: int
    down_dispatch_groups: int
    queue_capacity: int
    queue_upper_bound: int
    persistent_ctas: int
    protocol: DynamicProtocolPlan | None

    @property
    def spec(self) -> KernelSpec:
        return self.execution.kernel

    @property
    def programs(self) -> tuple[MoeTileProgram, ...]:
        if len(self.execution.device_regions) != 1:
            raise ValueError("MoE execution must contain exactly one device region")
        programs = self.execution.device_regions[0].tile_programs
        if not all(isinstance(program, MoeTileProgram) for program in programs):
            raise TypeError("MoE execution contains a non-MoE tile program")
        return programs

    @property
    def tiles(self) -> tuple[MoeTileProgram, ...]:
        return self.programs

    def program(self, tile_name: str) -> MoeTileProgram:
        return next(program for program in self.programs if program.tile.name == tile_name)

    @property
    def user_events(self) -> tuple[EventPlan, ...]:
        return tuple(event for event in self.events if event.name != "event_init_complete")

    @property
    def workspace_size(self) -> int:
        return sum(event.size for event in self.events)

    @property
    def dispatch_steps(self) -> tuple[DynamicDispatchStep, ...]:
        return tuple(
            step
            for program in self.programs
            for step in program.steps
            if isinstance(step, DynamicDispatchStep)
        )

    @property
    def dispatch_rules(self) -> tuple[_DynamicDispatchRule, ...]:
        return tuple(step.rule for step in self.dispatch_steps)

    def runtime_init_step(self, event_name: str) -> RuntimeEventInitStep | None:
        matching = tuple(
            step
            for program in self.programs
            for step in program.steps
            if isinstance(step, RuntimeEventInitStep)
            and isinstance(step.event, EventPlan)
            and step.event.name == event_name
        )
        if len(matching) > 1:
            raise ValueError(f"event {event_name!r} has multiple runtime initialization steps")
        return None if not matching else matching[0]

    def runtime_init_tile(self, event_name: str) -> str | None:
        for program in self.programs:
            if any(
                isinstance(step, RuntimeEventInitStep)
                and isinstance(step.event, EventPlan)
                and step.event.name == event_name
                for step in program.steps
            ):
                return program.tile.name
        return None

    @property
    def pre_before_wait(self) -> bool:
        if not self.is_dynamic:
            return True
        for program in self.programs:
            steps = program.steps
            pre = [
                index for index, step in enumerate(steps) if isinstance(step, DynamicDispatchStep)
            ]
            runs = [index for index, step in enumerate(steps) if isinstance(step, RunStep)]
            waits = [index for index, step in enumerate(steps) if isinstance(step, WaitStep)]
            if len(pre) != 1 or len(runs) != 1:
                return False
            if pre[0] > runs[0]:
                return False
            if waits and pre[0] > min(waits):
                return False
        return True

    @property
    def post_after_run(self) -> bool:
        for program in self.programs:
            steps = program.steps
            posts = [index for index, step in enumerate(steps) if isinstance(step, NotifyStep)]
            runs = [index for index, step in enumerate(steps) if isinstance(step, RunStep)]
            if bool(posts) != bool(program.notifies):
                return False
            if posts and min(posts) < runs[0]:
                return False
        return True

    @property
    def fifo_drain(self) -> bool:
        if not self.is_dynamic or self.protocol is None:
            return True
        terminal = [step for step in self.dispatch_steps if step.rule.target_tile is None]
        if len(terminal) != 1:
            return False
        dispatch = terminal[0]
        source = self.program(dispatch.rule.source_tile)
        return (
            self.protocol.queue_discipline == "fifo"
            and dispatch.trigger_upper_bound == 1
            and dispatch.count_lower_bound == self.persistent_ctas
            and dispatch.count_upper_bound == self.persistent_ctas
            and not source.notifies
        )

    def event(self, name: str) -> EventPlan:
        return next(event for event in self.events if event.name == name)

    def tile(self, name: str) -> MoeTileProgram:
        return self.program(name)

    def dispatch(self, source_tile: str) -> _DynamicDispatchRule:
        return next(rule for rule in self.dispatch_rules if rule.source_tile == source_tile)

    def dispatch_step(self, source_tile: str) -> DynamicDispatchStep:
        return next(step for step in self.dispatch_steps if step.rule.source_tile == source_tile)

    def validate(self) -> NormalizedPlan:
        # Imported lazily to keep the immutable plan model independent from
        # normalization implementation details.
        from .normalize import (
            _enumerate_tile,
            _normalize_dispatch,
            _validate_dispatch_coverage,
            _validate_dynamic_protocol_links,
            _validate_event_notification_counts,
            _validate_tile_event_accesses,
        )

        self.execution.validate()
        if self.execution.kernel is not self.env.spec:
            raise ValueError("normalized execution belongs to a different KernelSpec")
        programs = self.programs
        offset = 0
        for event in self.events:
            if event.workspace_offset != offset:
                raise ValueError(f"event {event.name!r} has a non-contiguous workspace offset")
            offset += event.size

        event_ids = {id(event) for event in self.events}
        runtime_inits: dict[str, list[tuple[EventPlan, RuntimeEventInitStep, int]]] = {}
        for program in programs:
            for index, step in enumerate(program.steps):
                if not isinstance(step, RuntimeEventInitStep) or step.event is None:
                    continue
                if not isinstance(step.event, EventPlan) or id(step.event) not in event_ids:
                    raise ValueError("runtime event initialization references an unknown event")
                runtime_inits.setdefault(program.tile.name, []).append((step.event, step, index))

        tile_names = {program.tile.name for program in programs}
        if tile_names != {tile.name for tile in self.spec.tiles}:
            raise ValueError("MoE execution programs do not match the logical tiles")
        if any(len(entries) > 1 for entries in runtime_inits.values()):
            raise ValueError("a task may initialize at most one runtime event in the MoE MVP")
        for program in programs:
            steps = program.steps
            runs = [index for index, step in enumerate(steps) if isinstance(step, RunStep)]
            waits = [index for index, step in enumerate(steps) if isinstance(step, WaitStep)]
            notifies = [index for index, step in enumerate(steps) if isinstance(step, NotifyStep)]
            barriers = [index for index, step in enumerate(steps) if isinstance(step, BarrierStep)]
            runtime_steps = [
                index for index, step in enumerate(steps) if isinstance(step, RuntimeEventInitStep)
            ]
            tile_name = program.tile.name
            if program.smem_scope != "run_to_end":
                raise ValueError(f"tile {tile_name!r} must use run_to_end shared memory")
            if len(runs) != 1:
                raise ValueError(f"tile {tile_name!r} must execute exactly once")
            if len(waits) != len(program.tile.waits):
                raise ValueError(f"tile {tile_name!r} wait steps do not match its logical waits")
            if waits and max(waits) > runs[0]:
                raise ValueError(f"tile {tile_name!r} must wait before execution")
            if len(notifies) != len(program.tile.notifies):
                raise ValueError(
                    f"tile {tile_name!r} notify steps do not match its logical notifies"
                )
            if program.implementation == "align":
                if len(barriers) != 1 or barriers[0] < runs[0]:
                    raise ValueError("align must synchronize the CTA after tile execution")
                if runtime_inits and (len(runtime_steps) != 1 or runtime_steps[0] < barriers[0]):
                    raise ValueError("align event initialization must follow CTA synchronization")
                if runtime_steps and notifies and min(notifies) < runtime_steps[0]:
                    raise ValueError("align must initialize runtime events before completion")
            elif barriers or runtime_steps:
                raise ValueError(f"tile {tile_name!r} has an unsupported CTA synchronization")
            for event, runtime_init, runtime_index in runtime_inits.get(tile_name, ()):
                if runtime_index <= barriers[0]:
                    raise ValueError(f"event {event.name!r} must be initialized after cta barrier")
                if notifies and runtime_index > min(notifies):
                    raise ValueError(
                        f"event {event.name!r} must be initialized before task completion"
                    )
                if runtime_init.scope != "thread" or runtime_init.scope_id != 0:
                    raise ValueError(f"event {event.name!r} has an invalid runtime initialization")

        if not self.post_after_run:
            raise ValueError("post notification must execute after tile execution")
        _validate_tile_event_accesses(self.env, programs, self.events)
        _validate_event_notification_counts(self.env, programs, self.events)

        if self.is_dynamic:
            if self.central_tasks or not self.seed_tasks:
                raise ValueError("dynamic plan must use seed tasks instead of a central queue")
            expected_seed = tuple(
                [
                    HostTask(JobType.INIT_ETENSOR.value, event_idx, 0, 0)
                    for event_idx in range(len(self.events))
                ]
                + _enumerate_tile(self.tile("gating"))
            )
            if self.seed_tasks != expected_seed:
                raise ValueError("dynamic seed must contain only event-init and gating tasks")
            if self.protocol is None:
                raise ValueError("dynamic plan is missing its semaphore protocol")
            if (
                self.protocol.pre_decrement != 1
                or self.protocol.post_decrement != SemaphoreBase.base
                or self.protocol.scheduler_warp != DynamicTileScheduler.scheduler_warp
            ):
                raise ValueError("dynamic plan does not match the two-phase scheduler protocol")
            if (
                len(self.dispatch_steps) != len(tile_names)
                or {step.rule.source_tile for step in self.dispatch_steps} != tile_names
            ):
                raise ValueError("dynamic plan does not have one dispatch rule per task")
            expected_dispatch = _normalize_dispatch(
                self.env, programs, self.dispatch_rules, self.events
            )
            if self.dispatch_steps != expected_dispatch:
                raise ValueError("dynamic dispatch bounds do not match their expressions")
            _validate_dynamic_protocol_links(programs, self.dispatch_steps)
            _validate_dispatch_coverage(self.env, programs, self.dispatch_steps)
            expected_queue_bound = len(self.seed_tasks) + sum(
                step.enqueue_upper_bound for step in self.dispatch_steps
            )
            if self.queue_upper_bound != expected_queue_bound:
                raise ValueError("dynamic queue upper bound is not derived from dispatch rules")
            if self.queue_upper_bound > self.queue_capacity:
                raise ValueError(
                    f"dynamic queue upper bound {self.queue_upper_bound} exceeds capacity "
                    f"{self.queue_capacity}"
                )
            if self.persistent_ctas != KernelConfig.SM_NUMBER:
                raise ValueError("dynamic plan violates persistent CTA saturation")
            if (
                self.dispatch_step("gating").count_lower_bound != self.persistent_ctas
                or self.dispatch_step("gating").count_upper_bound != self.persistent_ctas
                or self.dispatch_step("align").count_lower_bound != self.persistent_ctas
                or self.dispatch_step("align").count_upper_bound != self.persistent_ctas
            ):
                raise ValueError("dynamic plan violates topk/count-sort saturation fanout")
            if not self.pre_before_wait:
                raise ValueError("dynamic pre-notification must execute before wait and run")
            if not self.fifo_drain:
                raise ValueError("dynamic plan does not provide a FIFO terminal drain")

            down_event = self.event("down_dispatch_done")
            runtime_init = self.runtime_init_step(down_event.name)
            expected_value = (
                ConstExpr(self.protocol.pre_decrement + self.protocol.post_decrement)
                * self.tile("down").scheduled_extents[0]
                * self.down_dispatch_groups
            )
            if (
                runtime_init is None
                or self.runtime_init_tile(down_event.name) != "align"
                or runtime_init.scope != "thread"
                or runtime_init.scope_id != 0
                or runtime_init.value != expected_value
            ):
                raise ValueError("dynamic down event has an invalid runtime initialization")
        else:
            if self.dispatch_steps or self.seed_tasks or self.protocol is not None:
                raise ValueError("static plan contains dynamic scheduling state")
            if runtime_inits:
                raise ValueError("static plan cannot contain runtime event initialization")
            expected_central = [
                HostTask(JobType.INIT_ETENSOR.value, event_idx, 0, 0)
                for event_idx in range(len(self.events))
            ]
            expected_central.extend(_enumerate_tile(self.tile("gating")))
            expected_central.extend(
                HostTask(JobType.WAIT_ETENSOR_INIT.value, cta, 0, 0)
                for cta in range(self.persistent_ctas)
            )
            for program in programs:
                if program.tile.name != "gating":
                    expected_central.extend(_enumerate_tile(program))
            if self.central_tasks != tuple(expected_central):
                raise ValueError("static central queue does not match the normalized tasks")
            queue_columns = (
                len(self.central_tasks) + self.persistent_ctas - 1
            ) // self.persistent_ctas + 1
            if self.queue_upper_bound != queue_columns or queue_columns > self.queue_capacity:
                raise ValueError("static queue bound does not match its central task plan")
        return self

    def make_static_queue(self) -> np.ndarray:
        if self.is_dynamic:
            raise ValueError("dynamic plans do not have a static queue")
        queue = np.zeros((KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS), dtype=np.int32)
        cursor = 0
        tile_idx = 0
        while cursor < len(self.central_tasks):
            for cta in range(KernelConfig.SM_NUMBER):
                if cursor < len(self.central_tasks):
                    queue[cta, tile_idx] = self.central_tasks[cursor].packed()
                    cursor += 1
                else:
                    queue[cta, tile_idx] = pack_into_32bit(-1, -1, -1, JobType.END.value)
            tile_idx += 1
        for cta in range(KernelConfig.SM_NUMBER):
            queue[cta, tile_idx] = pack_into_32bit(-1, -1, -1, JobType.END.value)
        return queue

    def make_dynamic_queue(self) -> MPMCQueueHost:
        if not self.is_dynamic:
            raise ValueError("static plans do not have a dynamic queue")
        queue = MPMCQueueHost(self.queue_capacity)
        for task in self.seed_tasks:
            queue.enqueue(task.job_type, task.m_idx, task.n_idx, task.k_idx)
        return queue

    def normalized_data(self) -> dict[str, object]:
        return {
            "policy": self.policy_name,
            "events": [
                {
                    "name": event.name,
                    "shape": event.shape,
                    "init_count": event.init_count,
                    "offset": event.workspace_offset,
                    "logical": event.is_logical,
                    "runtime_init_tile": self.runtime_init_tile(event.name),
                    "runtime_init": (
                        None
                        if self.runtime_init_step(event.name) is None
                        else {
                            "tile": self.runtime_init_tile(event.name),
                            "value": self.runtime_init_step(event.name).value.to_data(),
                            "scope": self.runtime_init_step(event.name).scope,
                            "scope_id": self.runtime_init_step(event.name).scope_id,
                        }
                    ),
                }
                for event in self.events
            ],
            "tiles": [
                {
                    "name": program.tile.name,
                    "job_type": program.job_type,
                    "implementation": program.implementation,
                    "upper_bounds": program.upper_bounds,
                    "runtime_extents": [extent.to_data() for extent in program.runtime_extents],
                    "scheduled_extents": [extent.to_data() for extent in program.scheduled_extents],
                    "scheduled_upper_bounds": program.scheduled_upper_bounds,
                    "smem_scope": program.smem_scope,
                    "steps": [type(step).__name__ for step in program.steps],
                    "reads": [tensor.name for tensor in program.tile.reads],
                    "writes": [tensor.name for tensor in program.tile.writes],
                    "waits": [
                        {
                            "event": wait.event.name,
                            "coord": [coord.to_data() for coord in wait.coord_map],
                            "level": wait.level,
                            "mask": wait.mask,
                        }
                        for wait in program.waits
                    ],
                    "notifies": [
                        {
                            "event": notify.event.name,
                            "coord": [coord.to_data() for coord in notify.coord_map],
                            "scope": notify.scope,
                            "scope_id": notify.scope_id,
                            "count": notify.count.to_data(),
                            "rank": notify.rank,
                            "release": notify.release,
                        }
                        for notify in program.notifies
                    ],
                }
                for program in self.programs
            ],
            "central_task_count": len(self.central_tasks),
            "central_tasks": [task.as_manual_tuple() for task in self.central_tasks],
            "seed_tasks": [task.as_manual_tuple() for task in self.seed_tasks],
            "dispatch": [
                {
                    "source_tile": dispatch.rule.source_tile,
                    "event": dispatch.rule.event,
                    "event_coord": [coord.to_data() for coord in dispatch.rule.event_coord],
                    "target_tile": dispatch.rule.target_tile,
                    "count": dispatch.rule.count.to_data(),
                    "tile_indices": [index.to_data() for index in dispatch.rule.tile_indices],
                    "push_level": dispatch.rule.push_level,
                    "pre_scope": dispatch.rule.pre_scope,
                    "pre_scope_id": dispatch.rule.pre_scope_id,
                    "pre_count": dispatch.rule.pre_count.to_data(),
                    "rank": dispatch.rule.rank,
                    "trigger_upper_bound": dispatch.trigger_upper_bound,
                    "count_lower_bound": dispatch.count_lower_bound,
                    "count_upper_bound": dispatch.count_upper_bound,
                    "enqueue_upper_bound": dispatch.enqueue_upper_bound,
                    "event_coord_bounds": dispatch.event_coord_bounds,
                    "tile_index_bounds": dispatch.tile_index_bounds,
                }
                for dispatch in self.dispatch_steps
            ],
            "down_coalescing": self.down_coalescing,
            "down_dispatch_groups": self.down_dispatch_groups,
            "queue_capacity": self.queue_capacity,
            "queue_upper_bound": self.queue_upper_bound,
            "persistent_ctas": self.persistent_ctas,
            "pre_before_wait": self.pre_before_wait,
            "post_after_run": self.post_after_run,
            "fifo_drain": self.fifo_drain,
            "protocol": (
                None
                if self.protocol is None
                else {
                    "pre_decrement": self.protocol.pre_decrement,
                    "post_decrement": self.protocol.post_decrement,
                    "scheduler_warp": self.protocol.scheduler_warp,
                    "queue_discipline": self.protocol.queue_discipline,
                }
            ),
        }
