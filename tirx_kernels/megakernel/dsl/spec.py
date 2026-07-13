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

"""Declarative task and event graph for the megakernel DSL."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from .expr import (
    ConstExpr,
    Expr,
    ExprLike,
    ScalarLoadExpr,
    TileIndexExpr,
    as_expr,
    evaluate_static,
    walk_expr,
)

_INTEGER_DTYPES = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
_TENSOR_ROLES = {"input", "output", "intermediate", "workspace"}
_WAIT_LEVELS = {"cta", "warp"}
_NOTIFY_SCOPES = {"thread", "warp", "warpgroup", "cta"}


def _expr_tuple(values: tuple[ExprLike, ...] | list[ExprLike]) -> tuple[Expr, ...]:
    return tuple(as_expr(value) for value in values)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[Expr, ...]
    dtype: str
    role: str

    def __init__(self, name: str, shape: tuple[ExprLike, ...], dtype: str, role: str):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "shape", _expr_tuple(shape))
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "role", role)

    def scalar(self, index: ExprLike = 0) -> ScalarLoadExpr:
        return ScalarLoadExpr(self.name, as_expr(index), self.dtype, self.shape)


@dataclass(frozen=True)
class TaskDomain:
    extents: tuple[Expr, ...]
    upper_bounds: tuple[Expr, ...]

    def __init__(self, extents: tuple[ExprLike, ...], upper_bounds: tuple[ExprLike, ...]):
        object.__setattr__(self, "extents", _expr_tuple(extents))
        object.__setattr__(self, "upper_bounds", _expr_tuple(upper_bounds))


@dataclass(frozen=True)
class EventSpec:
    name: str
    shape: tuple[Expr, ...]
    init_count: Expr | None

    def __init__(self, name: str, shape: tuple[ExprLike, ...], init_count: ExprLike | None):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "shape", _expr_tuple(shape))
        object.__setattr__(self, "init_count", None if init_count is None else as_expr(init_count))


@dataclass(frozen=True)
class WaitSpec:
    event: str
    coord: tuple[Expr, ...]
    level: str
    mask: int = 0xFFFFFFFF

    def __init__(self, event: str, coord: tuple[ExprLike, ...], level: str, mask: int = 0xFFFFFFFF):
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "coord", _expr_tuple(coord))
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class NotifySpec:
    event: str
    coord: tuple[Expr, ...]
    scope: str
    scope_id: int
    count: Expr
    rank: int = -1
    release: bool = False

    def __init__(
        self,
        event: str,
        coord: tuple[ExprLike, ...],
        scope: str,
        scope_id: int,
        count: ExprLike = 1,
        rank: int = -1,
        release: bool = False,
    ):
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "coord", _expr_tuple(coord))
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "scope_id", scope_id)
        object.__setattr__(self, "count", as_expr(count))
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "release", release)


@dataclass(frozen=True)
class TileBinding:
    job_type: int
    implementation: str


@dataclass(frozen=True)
class TaskSpec:
    name: str
    domain: TaskDomain
    tile_binding: TileBinding
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    waits: tuple[WaitSpec, ...]
    notifies: tuple[NotifySpec, ...]

    def __init__(
        self,
        name: str,
        domain: TaskDomain,
        tile_binding: TileBinding,
        reads: tuple[str, ...] = (),
        writes: tuple[str, ...] = (),
        waits: tuple[WaitSpec, ...] = (),
        notifies: tuple[NotifySpec, ...] = (),
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "tile_binding", tile_binding)
        object.__setattr__(self, "reads", tuple(reads))
        object.__setattr__(self, "writes", tuple(writes))
        object.__setattr__(self, "waits", tuple(waits))
        object.__setattr__(self, "notifies", tuple(notifies))


@dataclass(frozen=True)
class DispatchSpec:
    """Dynamic rule fired by a source task's pre-notification."""

    source_task: str
    event: str
    event_coord: tuple[Expr, ...]
    target_task: str | None
    count: Expr
    tile_indices: tuple[Expr, ...]
    push_level: str
    pre_scope: str
    pre_scope_id: int = 0
    pre_count: Expr = field(default_factory=lambda: ConstExpr(1))
    rank: int = -1

    def __init__(
        self,
        source_task: str,
        event: str,
        event_coord: tuple[ExprLike, ...],
        target_task: str | None,
        count: ExprLike,
        tile_indices: tuple[ExprLike, ...],
        push_level: str,
        pre_scope: str,
        pre_scope_id: int = 0,
        pre_count: ExprLike = 1,
        rank: int = -1,
    ):
        object.__setattr__(self, "source_task", source_task)
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "event_coord", _expr_tuple(event_coord))
        object.__setattr__(self, "target_task", target_task)
        object.__setattr__(self, "count", as_expr(count))
        object.__setattr__(self, "tile_indices", _expr_tuple(tile_indices))
        object.__setattr__(self, "push_level", push_level)
        object.__setattr__(self, "pre_scope", pre_scope)
        object.__setattr__(self, "pre_scope_id", pre_scope_id)
        object.__setattr__(self, "pre_count", as_expr(pre_count))
        object.__setattr__(self, "rank", rank)


def _contains_callable(value: Any) -> bool:
    if callable(value):
        return True
    if isinstance(value, tuple | list | set):
        return any(_contains_callable(item) for item in value)
    if isinstance(value, Mapping):
        return any(_contains_callable(item) for item in value.values())
    if hasattr(value, "__dataclass_fields__"):
        return any(_contains_callable(getattr(value, field.name)) for field in fields(value))
    return False


def _exprs_in_task(task: TaskSpec):
    yield from task.domain.extents
    yield from task.domain.upper_bounds
    for wait in task.waits:
        yield from wait.coord
    for notify in task.notifies:
        yield from notify.coord
        yield notify.count


def _path_exists(edges: Mapping[str, set[str]], source: str, target: str) -> bool:
    pending = [source]
    visited = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(edges.get(current, ()))
    return False


@dataclass(frozen=True)
class KernelSpec:
    name: str
    tensors: tuple[TensorSpec, ...]
    events: tuple[EventSpec, ...]
    tasks: tuple[TaskSpec, ...]
    dynamic_dispatch: tuple[DispatchSpec, ...]
    compile_env: Mapping[str, int]

    def __init__(
        self,
        name: str,
        tensors: tuple[TensorSpec, ...],
        events: tuple[EventSpec, ...],
        tasks: tuple[TaskSpec, ...],
        dynamic_dispatch: tuple[DispatchSpec, ...] = (),
        compile_env: Mapping[str, int] | None = None,
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tensors", tuple(tensors))
        object.__setattr__(self, "events", tuple(events))
        object.__setattr__(self, "tasks", tuple(tasks))
        object.__setattr__(self, "dynamic_dispatch", tuple(dynamic_dispatch))
        object.__setattr__(self, "compile_env", dict(compile_env or {}))

    def _validate_names(self):
        for kind, specs in (("tensor", self.tensors), ("event", self.events), ("task", self.tasks)):
            names = [spec.name for spec in specs]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {kind} name in kernel {self.name!r}")

    def _validate_tensors(self):
        for tensor in self.tensors:
            if tensor.role not in _TENSOR_ROLES:
                raise ValueError(f"tensor {tensor.name!r} has unsupported role {tensor.role!r}")
            for axis, extent in enumerate(tensor.shape):
                value = evaluate_static(
                    extent, self.compile_env, label=f"tensor {tensor.name!r} shape axis {axis}"
                )
                if value <= 0:
                    raise ValueError(f"tensor {tensor.name!r} has non-positive shape")

    def _validate_events(self, event_map: Mapping[str, EventSpec]):
        del event_map
        for event in self.events:
            if not event.shape:
                raise ValueError(f"event {event.name!r} must have at least one coordinate axis")
            for axis, extent in enumerate(event.shape):
                if any(
                    isinstance(node, ScalarLoadExpr | TileIndexExpr) for node in walk_expr(extent)
                ):
                    raise ValueError(f"event {event.name!r} has a runtime shape")
                value = evaluate_static(
                    extent, self.compile_env, label=f"event {event.name!r} shape axis {axis}"
                )
                if value <= 0:
                    raise ValueError(f"event {event.name!r} has non-positive shape")
            if event.init_count is not None:
                if any(
                    isinstance(node, ScalarLoadExpr | TileIndexExpr)
                    for node in walk_expr(event.init_count)
                ):
                    raise ValueError(
                        f"event {event.name!r} runtime initialization must be owned by a task policy"
                    )
                count = evaluate_static(
                    event.init_count, self.compile_env, label=f"event {event.name!r} init_count"
                )
                if count <= 0:
                    raise ValueError(f"event {event.name!r} has non-positive init_count")

    def _validate_domains(self):
        for task in self.tasks:
            if len(task.domain.extents) != 3 or len(task.domain.upper_bounds) != 3:
                raise ValueError(f"task {task.name!r} must have a three-axis domain")
            for axis, upper_bound in enumerate(task.domain.upper_bounds):
                if any(
                    isinstance(node, ScalarLoadExpr | TileIndexExpr)
                    for node in walk_expr(upper_bound)
                ):
                    raise ValueError(
                        f"task {task.name!r} axis {axis} does not have a static upper bound"
                    )
                upper = evaluate_static(
                    upper_bound,
                    self.compile_env,
                    label=f"task {task.name!r} upper bound axis {axis}",
                )
                if upper <= 0:
                    raise ValueError(f"task {task.name!r} has a non-positive upper bound")
                try:
                    extent = task.domain.extents[axis].evaluate(self.compile_env)
                except (KeyError, TypeError, ValueError):
                    continue
                if not isinstance(extent, int) or extent <= 0 or extent > upper:
                    raise ValueError(
                        f"task {task.name!r} runtime extent axis {axis} exceeds its upper bound"
                    )

    def _validate_graph(self, tensor_map, event_map, task_map):
        producers: dict[str, set[str]] = {name: set() for name in event_map}
        tensor_writers: dict[str, set[str]] = {name: set() for name in tensor_map}
        for task in self.tasks:
            if callable(task.tile_binding) or not isinstance(task.tile_binding, TileBinding):
                raise ValueError(f"task {task.name!r} has an invalid tile binding")
            for name in (*task.reads, *task.writes):
                if name not in tensor_map:
                    raise ValueError(f"task {task.name!r} references unknown tensor {name!r}")
            for name in task.writes:
                tensor_writers[name].add(task.name)
            waited_events = {wait.event for wait in task.waits}
            if len(waited_events) > 1:
                raise ValueError(f"task {task.name!r} uses unsupported DAG fan-in")
            for wait in task.waits:
                self._validate_event_coord(task.name, wait.event, wait.coord, event_map)
                if wait.level not in _WAIT_LEVELS:
                    raise ValueError(f"task {task.name!r} has invalid wait level {wait.level!r}")
                self._validate_tile_refs(task.name, wait.coord)
            for notify in task.notifies:
                self._validate_event_coord(task.name, notify.event, notify.coord, event_map)
                if notify.scope not in _NOTIFY_SCOPES:
                    raise ValueError(
                        f"task {task.name!r} has invalid notify scope {notify.scope!r}"
                    )
                self._validate_tile_refs(task.name, (*notify.coord, notify.count))
                producers[notify.event].add(task.name)

        dispatch_by_source: dict[str, list[DispatchSpec]] = {}
        for rule in self.dynamic_dispatch:
            if rule.source_task not in task_map:
                raise ValueError(f"dynamic rule has unknown source task {rule.source_task!r}")
            if rule.target_task is not None and rule.target_task not in task_map:
                raise ValueError(f"dynamic rule has unknown target task {rule.target_task!r}")
            self._validate_event_coord(rule.source_task, rule.event, rule.event_coord, event_map)
            if len(rule.tile_indices) != 3:
                raise ValueError(f"dynamic rule for {rule.source_task!r} must map three tile axes")
            if rule.push_level not in _NOTIFY_SCOPES or rule.pre_scope not in _NOTIFY_SCOPES:
                raise ValueError(f"dynamic rule for {rule.source_task!r} has an invalid scope")
            self._validate_tile_refs(
                rule.source_task,
                (*rule.event_coord, *rule.tile_indices, rule.count, rule.pre_count),
            )
            dispatch_by_source.setdefault(rule.source_task, []).append(rule)
            producers[rule.event].add(rule.source_task)

        for task in self.tasks:
            rules = dispatch_by_source.get(task.name, [])
            if len(rules) != 1:
                raise ValueError(f"task {task.name!r} must have exactly one dynamic dispatch rule")

        for event, event_producers in producers.items():
            if len(event_producers) > 1:
                raise ValueError(f"event {event!r} has multiple producers")

        edges: dict[str, set[str]] = {task.name: set() for task in self.tasks}
        for task in self.tasks:
            for wait in task.waits:
                event_producers = producers[wait.event]
                if len(event_producers) != 1:
                    raise ValueError(f"wait event {wait.event!r} does not have a unique producer")
                producer = next(iter(event_producers))
                edges[producer].add(task.name)
        self._validate_acyclic(edges)
        self._validate_scalar_loads(tensor_map, tensor_writers, edges)

    def _validate_event_coord(self, owner, event, coord, event_map, *, rank_only=False):
        if event not in event_map:
            raise ValueError(f"{owner!r} references unknown event {event!r}")
        if not rank_only and len(coord) != len(event_map[event].shape):
            raise ValueError(f"{owner!r} coordinate rank does not match event {event!r}")

    @staticmethod
    def _validate_tile_refs(owner: str, exprs: tuple[Expr, ...]):
        for expr in exprs:
            for node in walk_expr(expr):
                if isinstance(node, TileIndexExpr) and node.task != owner:
                    raise ValueError(f"{owner!r} references tile index owned by task {node.task!r}")

    @staticmethod
    def _validate_acyclic(edges: Mapping[str, set[str]]):
        state: dict[str, int] = {}

        def visit(task: str):
            if state.get(task) == 1:
                raise ValueError("task dependency graph contains a cycle")
            if state.get(task) == 2:
                return
            state[task] = 1
            for successor in edges[task]:
                visit(successor)
            state[task] = 2

        for task in edges:
            visit(task)

    def _validate_scalar_loads(self, tensor_map, tensor_writers, edges):
        dispatch_exprs: dict[str, list[Expr]] = {}
        for rule in self.dynamic_dispatch:
            dispatch_exprs.setdefault(rule.source_task, []).extend(
                [rule.count, rule.pre_count, *rule.event_coord, *rule.tile_indices]
            )

        for task in self.tasks:
            ordinary_exprs = tuple(_exprs_in_task(task))
            for expr, allow_same_task in (
                *((expr, False) for expr in ordinary_exprs),
                *((expr, True) for expr in dispatch_exprs.get(task.name, ())),
            ):
                for node in walk_expr(expr):
                    if not isinstance(node, ScalarLoadExpr):
                        continue
                    tensor = tensor_map.get(node.tensor)
                    if tensor is None or tensor.dtype not in _INTEGER_DTYPES:
                        raise ValueError(
                            f"runtime scalar {node.tensor!r} must come from an integer tensor"
                        )
                    if node.tensor not in task.reads:
                        raise ValueError(
                            f"task {task.name!r} does not declare runtime scalar {node.tensor!r} as read"
                        )
                    index = evaluate_static(
                        node.index, self.compile_env, label=f"runtime scalar {node.tensor!r} index"
                    )
                    shape0 = evaluate_static(
                        tensor.shape[0],
                        self.compile_env,
                        label=f"runtime scalar {node.tensor!r} shape",
                    )
                    if len(tensor.shape) != 1 or index < 0 or index >= shape0:
                        raise ValueError(f"runtime scalar {node.tensor!r} index is out of bounds")
                    writers = tensor_writers[node.tensor]
                    if len(writers) != 1:
                        raise ValueError(
                            f"runtime scalar {node.tensor!r} must have exactly one producer"
                        )
                    producer = next(iter(writers))
                    if producer == task.name and allow_same_task:
                        continue
                    if producer == task.name or not _path_exists(edges, producer, task.name):
                        raise ValueError(
                            f"runtime scalar {node.tensor!r} is used before producer completion"
                        )

    def validate(self) -> KernelSpec:
        if _contains_callable(self):
            raise ValueError("callables are outside the megakernel DSL MVP")
        self._validate_names()
        tensor_map = {tensor.name: tensor for tensor in self.tensors}
        event_map = {event.name: event for event in self.events}
        task_map = {task.name: task for task in self.tasks}
        self._validate_tensors()
        self._validate_events(event_map)
        self._validate_domains()
        self._validate_graph(tensor_map, event_map, task_map)
        return self

    def lower(self, lowerer):
        self.validate()
        return lowerer.lower(self)

    def normalized_data(self) -> dict[str, object]:
        """Return a deterministic, JSON-compatible representation for tests and reviews."""

        return {
            "name": self.name,
            "tensors": [
                {
                    "name": tensor.name,
                    "shape": [extent.to_data() for extent in tensor.shape],
                    "dtype": tensor.dtype,
                    "role": tensor.role,
                }
                for tensor in self.tensors
            ],
            "events": [
                {
                    "name": event.name,
                    "shape": [extent.to_data() for extent in event.shape],
                    "init_count": None if event.init_count is None else event.init_count.to_data(),
                }
                for event in self.events
            ],
            "tasks": [
                {
                    "name": task.name,
                    "job_type": task.tile_binding.job_type,
                    "implementation": task.tile_binding.implementation,
                    "extents": [extent.to_data() for extent in task.domain.extents],
                    "upper_bounds": [bound.to_data() for bound in task.domain.upper_bounds],
                    "reads": list(task.reads),
                    "writes": list(task.writes),
                    "waits": [
                        {
                            "event": wait.event,
                            "coord": [coord.to_data() for coord in wait.coord],
                            "level": wait.level,
                            "mask": wait.mask,
                        }
                        for wait in task.waits
                    ],
                    "notifies": [
                        {
                            "event": notify.event,
                            "coord": [coord.to_data() for coord in notify.coord],
                            "scope": notify.scope,
                            "scope_id": notify.scope_id,
                            "count": notify.count.to_data(),
                            "rank": notify.rank,
                            "release": notify.release,
                        }
                        for notify in task.notifies
                    ],
                }
                for task in self.tasks
            ],
            "dynamic_dispatch": [
                {
                    "source_task": rule.source_task,
                    "event": rule.event,
                    "event_coord": [coord.to_data() for coord in rule.event_coord],
                    "target_task": rule.target_task,
                    "count": rule.count.to_data(),
                    "tile_indices": [index.to_data() for index in rule.tile_indices],
                    "push_level": rule.push_level,
                    "pre_scope": rule.pre_scope,
                    "pre_scope_id": rule.pre_scope_id,
                    "pre_count": rule.pre_count.to_data(),
                    "rank": rule.rank,
                }
                for rule in self.dynamic_dispatch
            ],
        }
