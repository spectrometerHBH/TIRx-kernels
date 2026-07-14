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

"""Single-graph MoE DSL policies and lowering adapter.

The graph in this module contains no scheduler branches.  Static, unfused, and
dynamic behavior is selected by normalizing it through one of the three policy
classes.  Tile implementations remain opaque and are invoked by ``MoeLowerer``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import reduce
from itertools import product
from operator import mul
from typing import Any

import numpy as np

from tirx_kernels.megakernel.utils.base import SemaphoreBase
from tirx_kernels.megakernel.utils.config import JobType, KernelConfig
from tirx_kernels.megakernel.utils.dynamic_scheduler import DynamicTileScheduler, MPMCQueueHost
from tirx_kernels.megakernel.utils.static_scheduler import StaticTileScheduler
from tirx_kernels.megakernel.utils.utils import (
    MAX_K_IDX,
    MAX_M_IDX,
    MAX_N_IDX,
    MAX_TASK_TYPE,
    f_init_const,
    pack_into_32bit,
)
from tvm.script import tirx as T

from ._expr import (
    BinaryExpr,
    CeilDivExpr,
    ConstExpr,
    Expr,
    ScalarLoadExpr,
    TileIndexExpr,
    VarExpr,
    as_expr,
    walk_expr,
)
from .spec import DependencySpec, EventSpec, KernelSpec, TileSpec, VarSpec
from .tile_impl import (
    AlignTileImpl,
    CountSortTileImpl,
    DownTileImpl,
    GateUpSiluTileImpl,
    GatingTileImpl,
    TopkTileImpl,
)

_EXPECTED_CONFIG = {
    "HIDDEN_SIZE": 2048,
    "INTERMEDIATE_SIZE": 768,
    "NUM_EXPERTS": 128,
    "NUM_EXPERTS_PER_TOK": 8,
    "GATING_SPLIT_K_FACTOR": 4,
}
_EVENT_ATTRS = {
    "gating_done": "evt_gating",
    "topk_done": "evt_topk_softmax",
    "align_done": "evt_moe_align",
    "count_sort_done": "evt_count_and_sort",
    "gate_up_done": "evt_group_gemm_gate_up",
    "down_dispatch_done": "evt_group_gemm_down",
}
_STEP_PRE_NOTIFY = "pre_notify"
_STEP_WAIT = "wait"
_STEP_RUN = "run"
_STEP_CTA_SYNC = "cta_sync"
_STEP_RUNTIME_EVENT_INIT = "runtime_event_init"
_STEP_POST_NOTIFY = "post_notify"
_EXECUTION_STEPS = {
    _STEP_PRE_NOTIFY,
    _STEP_WAIT,
    _STEP_RUN,
    _STEP_CTA_SYNC,
    _STEP_RUNTIME_EVENT_INIT,
    _STEP_POST_NOTIFY,
}
_PACKED_INDEX_LIMITS = (MAX_M_IDX, MAX_N_IDX, MAX_K_IDX)
_SCOPE_WIDTHS = {
    "thread": 1,
    "warp": 32,
    "warpgroup": KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER,
    "cta": KernelConfig.NUM_THREADS,
}
_SCOPE_INSTANCES = {
    "thread": KernelConfig.NUM_THREADS,
    "warp": KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER,
    "warpgroup": KernelConfig.WG_NUMBER,
    "cta": 1,
}
_SCOPE_ORDER = {"thread": 0, "warp": 1, "warpgroup": 2, "cta": 3}


def _max_rows(batch_size: int) -> int:
    routed = batch_size * _EXPECTED_CONFIG["NUM_EXPERTS_PER_TOK"]
    experts = _EXPECTED_CONFIG["NUM_EXPERTS"]
    if routed < experts:
        return routed
    return experts + (routed - experts + 127) // 128


def _validate_mvp_config(config: Mapping[str, Any]):
    mismatch = {
        name: (config.get(name), expected)
        for name, expected in _EXPECTED_CONFIG.items()
        if config.get(name) != expected
    }
    if mismatch:
        raise ValueError(f"MoE megakernel DSL MVP only supports Qwen3-30B-A3B: {mismatch}")


def build_moe_graph(config: Mapping[str, Any], batch_size: int) -> KernelSpec:
    """Build the scheduler-independent six-stage MoE graph."""

    _validate_mvp_config(config)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive compile-time integer")

    max_rows = _max_rows(batch_size)
    max_tokens = max_rows * 128
    relaxed_rows = batch_size * 8 // 128 + 129
    routed_rows = VarSpec("routed_rows")
    kernel = KernelSpec(
        "qwen3_30b_a3b_moe", attrs={"source": "Qwen3-30B-A3B six-stage MoE pipeline"}
    )

    hidden_state = kernel.input("hidden_state", (batch_size, 2048), "float16")
    kernel.input("residual", (batch_size, 2048), "float16")
    kernel.output("output", (batch_size, 2048), "float16")
    gate_weight = kernel.input("gate_weight", (128, 2048), "float16")
    gate_up_weight = kernel.input("gate_up_weight", (128, 1536, 2048), "float16")
    down_weight = kernel.input("down_weight", (128, 2048, 768), "float16")
    gating_output = kernel.intermediate("gating_output", (batch_size, 128), "float32")
    topk_weights = kernel.intermediate("topk_weights", (batch_size, 8), "float32")
    topk_indices = kernel.intermediate("topk_indices", (batch_size, 8), "int32")
    sorted_token_ids = kernel.intermediate("sorted_token_ids", (max_tokens,), "int32")
    expert_ids = kernel.intermediate("expert_ids", (max_rows,), "int32")
    num_valid_tokens = kernel.intermediate("num_valid_tokens", (max_rows,), "int32")
    num_tokens_post_pad = kernel.intermediate("num_tokens_post_pad", (1,), "int32")
    cumsum_buffer = kernel.intermediate("cumsum_buffer", (129,), "int32")
    reordered_hidden_state = kernel.intermediate(
        "reordered_hidden_state", (max_tokens, 2048), "float16"
    )
    silu_mul_output = kernel.intermediate("silu_mul_output", (max_tokens, 768), "float16")
    topk_reduce_output = kernel.output("topk_reduce_output", (batch_size, 2048), "float16")

    gating_done = kernel.event(
        "gating_done",
        (1,),
        4 * ((batch_size + 127) // 128),
        attrs={"meaning": "all split-K gating tiles are complete"},
    )
    topk_done = kernel.event(
        "topk_done", (1,), 148, attrs={"meaning": "all persistent top-k tiles are complete"}
    )
    align_done = kernel.event(
        "align_done", (1,), 1, attrs={"meaning": "token-to-expert alignment metadata is ready"}
    )
    count_sort_done = kernel.event(
        "count_sort_done", (1,), 148, attrs={"meaning": "all count-and-sort tiles are complete"}
    )
    gate_up_done = kernel.event(
        "gate_up_done",
        (relaxed_rows,),
        12,
        attrs={"meaning": "all gate-up projections for one routed row are complete"},
    )

    (
        kernel.tile(
            "gating",
            impl=GatingTileImpl(config),
            tile_num=((batch_size + 127) // 128, 1, 4),
            attrs={
                "source_stage": "gating_output = hidden_state @ gate_weight.T",
                "purpose": "compute split-K expert logits",
            },
        )
        .read(hidden_state, gate_weight)
        .write(gating_output)
        .notify(gating_done, lambda m, n, k: (0,))
    )
    (
        kernel.tile(
            "topk",
            impl=TopkTileImpl(config, batch_size),
            tile_num=(148, 1, 1),
            attrs={
                "source_stage": "topk_weights, topk_indices = topk(gating_output)",
                "purpose": "select experts and routing weights",
            },
        )
        .read(gating_output)
        .write(topk_weights, topk_indices)
        .wait(gating_done, lambda m, n, k: (0,))
        .notify(topk_done, lambda m, n, k: (0,))
    )
    (
        kernel.tile(
            "align",
            impl=AlignTileImpl(config, batch_size),
            tile_num=(1, 1, 1),
            attrs={
                "source_stage": "align tokens by expert",
                "purpose": "produce padded routing metadata",
            },
        )
        .read(topk_indices)
        .write(sorted_token_ids, expert_ids, num_valid_tokens, num_tokens_post_pad, cumsum_buffer)
        .wait(topk_done, lambda m, n, k: (0,))
        .notify(align_done, lambda m, n, k: (0,))
    )
    (
        kernel.tile(
            "count_sort",
            impl=CountSortTileImpl(config, batch_size),
            tile_num=(148, 1, 1),
            attrs={
                "source_stage": "reorder hidden states by expert",
                "purpose": "count and scatter routed tokens",
            },
        )
        .read(topk_indices, sorted_token_ids, cumsum_buffer, hidden_state, num_tokens_post_pad)
        .write(reordered_hidden_state)
        .wait(align_done, lambda m, n, k: (0,))
        .notify(count_sort_done, lambda m, n, k: (0,))
    )
    (
        kernel.tile(
            "gate_up_silu",
            impl=GateUpSiluTileImpl(config, batch_size),
            tile_num=(routed_rows, 12, 1),
            attrs={
                "source_stage": "silu(gate) * up",
                "purpose": "compute routed gate-up projections and SiLU",
            },
        )
        .read(
            reordered_hidden_state,
            gate_up_weight,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_valid_tokens,
            num_tokens_post_pad,
        )
        .write(silu_mul_output)
        .wait(count_sort_done, lambda m, n, k: (0,))
        .notify(gate_up_done, lambda m, n, k: (m,))
    )
    (
        kernel.tile(
            "down",
            impl=DownTileImpl(config, batch_size),
            tile_num=(routed_rows, 16, 1),
            attrs={
                "source_stage": "topk_reduce_output = down(silu_mul_output)",
                "purpose": "compute and accumulate routed down projections",
            },
        )
        .read(
            silu_mul_output,
            down_weight,
            expert_ids,
            topk_weights,
            sorted_token_ids,
            num_valid_tokens,
            num_tokens_post_pad,
        )
        .write(topk_reduce_output)
        .wait(gate_up_done, lambda m, n, k: (m,))
    )
    return kernel


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


def _validate_logical_attrs(attrs: Mapping[str, Any], *, owner: str):
    def visit(value, path: str):
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
            for name in ("implementation", "job_type", "profile_event_type", "register"):
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
            if tile_num[0] != VarSpec("routed_rows"):
                raise ValueError(f"tile {tile_name!r} must use VarSpec('routed_rows') on axis 0")
            if runtime_tensor not in tile.reads:
                raise ValueError(f"tile {tile_name!r} must read num_tokens_post_pad")
        self._validate_runtime_producer_order()

    def _validate_runtime_producer_order(self):
        edges = {tile.name: set() for tile in self.spec.tiles}
        notifiers: dict[int, list[str]] = {}
        for tile in self.spec.tiles:
            for notify in tile.notifies:
                notifiers.setdefault(id(notify.event), []).append(tile.name)
        for tile in self.spec.tiles:
            for wait in tile.waits:
                for producer in notifiers.get(id(wait.event), ()):
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
        if value == VarSpec("routed_rows"):
            return self.routed_rows
        raise ValueError(f"unsupported MoE runtime extent {value!r}")

    def upper_bound(self, value: int | VarSpec) -> int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if value == VarSpec("routed_rows"):
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

    def coord(self, tile: TileSpec, dependency: DependencySpec) -> tuple[Expr, ...]:
        coord_map = dependency.coord_map
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
        if len(values) != len(_shape_tuple(dependency.event.shape)):
            raise ValueError(f"tile {tile.name!r} coordinate rank does not match its event")
        result = tuple(as_expr(value) for value in values)
        for expr in result:
            for node in walk_expr(expr):
                if isinstance(node, TileIndexExpr) and node.task != tile.name:
                    raise ValueError(f"tile {tile.name!r} coordinate uses a foreign tile index")
        return result


@dataclass(frozen=True)
class RuntimeEventInitPlan:
    """A post-tile store of a raw dynamic semaphore value."""

    tile: str
    value: Expr
    scope: str = "thread"
    scope_id: int = 0
    after_step: str = _STEP_CTA_SYNC


@dataclass(frozen=True)
class EventPlan:
    name: str
    shape: tuple[int, ...]
    init_count: int | None
    workspace_offset: int
    runtime_init: RuntimeEventInitPlan | None = None
    logical_spec: EventSpec | None = None

    @property
    def size(self) -> int:
        return reduce(mul, self.shape, 1)

    @property
    def runtime_init_tile(self) -> str | None:
        return None if self.runtime_init is None else self.runtime_init.tile

    @property
    def is_logical(self) -> bool:
        return self.logical_spec is not None


@dataclass(frozen=True)
class WaitPlan:
    logical_spec: DependencySpec
    event: str
    coord: tuple[Expr, ...]
    level: str
    mask: int = 0xFFFFFFFF


@dataclass(frozen=True)
class NotifyPlan:
    logical_spec: DependencySpec
    event: str
    coord: tuple[Expr, ...]
    scope: str
    scope_id: int
    count: Expr = field(default_factory=lambda: ConstExpr(1))
    rank: int = -1
    release: bool = False


@dataclass(frozen=True)
class TilePlan:
    spec: TileSpec
    runtime_extents: tuple[Expr, Expr, Expr]
    upper_bounds: tuple[int, int, int]
    scheduled_extents: tuple[Expr, Expr, Expr]
    scheduled_upper_bounds: tuple[int, int, int]
    execution_steps: tuple[str, ...]
    waits: tuple[WaitPlan, ...]
    notifies: tuple[NotifyPlan, ...]

    @property
    def implementation(self) -> str:
        return self.spec.impl.implementation

    @property
    def job_type(self) -> int:
        return self.spec.impl.job_type


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


@dataclass(frozen=True)
class DynamicDispatchPlan:
    """A dispatch rule plus statically proven trigger, count, and index ranges."""

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
    spec: KernelSpec
    env: MoeLoweringEnv
    policy_name: str
    is_dynamic: bool
    unfused: bool
    events: tuple[EventPlan, ...]
    tiles: tuple[TilePlan, ...]
    dispatch_plans: tuple[DynamicDispatchPlan, ...]
    central_tasks: tuple[HostTask, ...]
    seed_tasks: tuple[HostTask, ...]
    down_coalescing: int
    down_dispatch_groups: int
    queue_capacity: int
    queue_upper_bound: int
    persistent_ctas: int
    protocol: DynamicProtocolPlan | None

    @property
    def user_events(self) -> tuple[EventPlan, ...]:
        return tuple(event for event in self.events if event.name != "event_init_complete")

    @property
    def workspace_size(self) -> int:
        return sum(event.size for event in self.events)

    @property
    def dispatch_rules(self) -> tuple[_DynamicDispatchRule, ...]:
        return tuple(dispatch.rule for dispatch in self.dispatch_plans)

    @property
    def pre_before_wait(self) -> bool:
        if not self.is_dynamic:
            return True
        for tile in self.tiles:
            steps = tile.execution_steps
            if steps.count(_STEP_PRE_NOTIFY) != 1 or steps.count(_STEP_RUN) != 1:
                return False
            pre_index = steps.index(_STEP_PRE_NOTIFY)
            if pre_index > steps.index(_STEP_RUN):
                return False
            if _STEP_WAIT in steps and pre_index > steps.index(_STEP_WAIT):
                return False
        return True

    @property
    def post_after_run(self) -> bool:
        for tile in self.tiles:
            steps = tile.execution_steps
            has_post = _STEP_POST_NOTIFY in steps
            if has_post != bool(tile.notifies):
                return False
            if has_post and steps.index(_STEP_POST_NOTIFY) < steps.index(_STEP_RUN):
                return False
        return True

    @property
    def fifo_drain(self) -> bool:
        if not self.is_dynamic or self.protocol is None:
            return True
        terminal = [
            dispatch for dispatch in self.dispatch_plans if dispatch.rule.target_tile is None
        ]
        if len(terminal) != 1:
            return False
        dispatch = terminal[0]
        source = self.tile(dispatch.rule.source_tile)
        return (
            self.protocol.queue_discipline == "fifo"
            and dispatch.trigger_upper_bound == 1
            and dispatch.count_lower_bound == self.persistent_ctas
            and dispatch.count_upper_bound == self.persistent_ctas
            and not source.notifies
            and _STEP_POST_NOTIFY not in source.execution_steps
        )

    def event(self, name: str) -> EventPlan:
        return next(event for event in self.events if event.name == name)

    def tile(self, name: str) -> TilePlan:
        return next(tile for tile in self.tiles if tile.spec.name == name)

    def dispatch(self, source_tile: str) -> _DynamicDispatchRule:
        return next(rule for rule in self.dispatch_rules if rule.source_tile == source_tile)

    def dispatch_plan(self, source_tile: str) -> DynamicDispatchPlan:
        return next(
            dispatch for dispatch in self.dispatch_plans if dispatch.rule.source_tile == source_tile
        )

    def validate(self) -> NormalizedPlan:
        offset = 0
        runtime_inits: dict[str, list[EventPlan]] = {}
        for event in self.events:
            if event.workspace_offset != offset:
                raise ValueError(f"event {event.name!r} has a non-contiguous workspace offset")
            offset += event.size
            if event.runtime_init is not None:
                runtime_inits.setdefault(event.runtime_init.tile, []).append(event)

        tile_names = {tile.spec.name for tile in self.tiles}
        if set(runtime_inits) - tile_names:
            raise ValueError("runtime event initialization references an unknown tile")
        if any(len(events) > 1 for events in runtime_inits.values()):
            raise ValueError("a task may initialize at most one runtime event in the MoE MVP")
        for tile in self.tiles:
            steps = tile.execution_steps
            if any(step not in _EXECUTION_STEPS for step in steps) or len(steps) != len(set(steps)):
                raise ValueError(f"tile {tile.spec.name!r} has an invalid execution plan")
            if steps.count(_STEP_RUN) != 1:
                raise ValueError(f"tile {tile.spec.name!r} must execute exactly once")
            if (_STEP_WAIT in steps) != bool(tile.waits):
                raise ValueError(f"tile {tile.spec.name!r} wait step does not match its waits")
            if _STEP_WAIT in steps and steps.index(_STEP_WAIT) > steps.index(_STEP_RUN):
                raise ValueError(f"tile {tile.spec.name!r} must wait before execution")
            if (_STEP_POST_NOTIFY in steps) != bool(tile.notifies):
                raise ValueError(f"tile {tile.spec.name!r} post step does not match its notifies")
            if (_STEP_RUNTIME_EVENT_INIT in steps) != (tile.implementation == "align"):
                raise ValueError(
                    f"tile {tile.spec.name!r} has an invalid runtime initialization slot"
                )
            if tile.implementation == "align":
                if _STEP_CTA_SYNC not in steps or steps.index(_STEP_CTA_SYNC) < steps.index(
                    _STEP_RUN
                ):
                    raise ValueError("align must synchronize the CTA after tile execution")
                if steps.index(_STEP_RUNTIME_EVENT_INIT) < steps.index(_STEP_CTA_SYNC):
                    raise ValueError("align event initialization must follow CTA synchronization")
                if _STEP_POST_NOTIFY in steps and steps.index(_STEP_POST_NOTIFY) < steps.index(
                    _STEP_RUNTIME_EVENT_INIT
                ):
                    raise ValueError("align must initialize runtime events before completion")
            elif _STEP_CTA_SYNC in steps:
                raise ValueError(f"tile {tile.spec.name!r} has an unsupported CTA synchronization")
            for event in runtime_inits.get(tile.spec.name, ()):
                runtime_init = event.runtime_init
                if runtime_init is None or runtime_init.after_step not in steps:
                    raise ValueError(f"event {event.name!r} has an invalid runtime initialization")
                if steps.index(_STEP_RUNTIME_EVENT_INIT) <= steps.index(runtime_init.after_step):
                    raise ValueError(
                        f"event {event.name!r} must be initialized after {runtime_init.after_step}"
                    )
                if _STEP_POST_NOTIFY in steps and steps.index(
                    _STEP_RUNTIME_EVENT_INIT
                ) > steps.index(_STEP_POST_NOTIFY):
                    raise ValueError(
                        f"event {event.name!r} must be initialized before task completion"
                    )

        if not self.post_after_run:
            raise ValueError("post notification must execute after tile execution")
        _validate_tile_event_accesses(self.env, self.tiles, self.events)
        _validate_event_notification_counts(self.env, self.tiles, self.events)

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
                or self.protocol.scheduler_warp != 7
            ):
                raise ValueError("dynamic plan does not match the two-phase scheduler protocol")
            if (
                len(self.dispatch_plans) != len(tile_names)
                or {dispatch.rule.source_tile for dispatch in self.dispatch_plans} != tile_names
            ):
                raise ValueError("dynamic plan does not have one dispatch rule per task")
            expected_dispatch = _normalize_dispatch(
                self.env, self.tiles, self.dispatch_rules, self.events
            )
            if self.dispatch_plans != expected_dispatch:
                raise ValueError("dynamic dispatch bounds do not match their expressions")
            _validate_dynamic_protocol_links(self.tiles, self.dispatch_plans)
            _validate_dispatch_coverage(self.env, self.tiles, self.dispatch_plans)
            expected_queue_bound = len(self.seed_tasks) + sum(
                dispatch.enqueue_upper_bound for dispatch in self.dispatch_plans
            )
            if self.queue_upper_bound != expected_queue_bound:
                raise ValueError("dynamic queue upper bound is not derived from dispatch rules")
            if self.queue_upper_bound > self.queue_capacity:
                raise ValueError(
                    f"dynamic queue upper bound {self.queue_upper_bound} exceeds capacity "
                    f"{self.queue_capacity}"
                )
            if self.persistent_ctas != KernelConfig.SM_NUMBER or self.persistent_ctas != 148:
                raise ValueError("dynamic plan violates persistent CTA saturation")
            if (
                self.dispatch_plan("gating").count_lower_bound != self.persistent_ctas
                or self.dispatch_plan("gating").count_upper_bound != self.persistent_ctas
                or self.dispatch_plan("align").count_lower_bound != self.persistent_ctas
                or self.dispatch_plan("align").count_upper_bound != self.persistent_ctas
            ):
                raise ValueError("dynamic plan violates topk/count-sort saturation fanout")
            if not self.pre_before_wait:
                raise ValueError("dynamic pre-notification must execute before wait and run")
            if not self.fifo_drain:
                raise ValueError("dynamic plan does not provide a FIFO terminal drain")

            down_event = self.event("down_dispatch_done")
            runtime_init = down_event.runtime_init
            expected_value = (
                ConstExpr(self.protocol.pre_decrement + self.protocol.post_decrement)
                * self.tile("down").scheduled_extents[0]
                * self.down_dispatch_groups
            )
            if (
                runtime_init is None
                or runtime_init.tile != "align"
                or runtime_init.scope != "thread"
                or runtime_init.scope_id != 0
                or runtime_init.value != expected_value
            ):
                raise ValueError("dynamic down event has an invalid runtime initialization")
        else:
            if self.dispatch_plans or self.seed_tasks or self.protocol is not None:
                raise ValueError("static plan contains dynamic scheduling state")
            if any(_STEP_PRE_NOTIFY in tile.execution_steps for tile in self.tiles):
                raise ValueError("static task cannot contain a pre-notification step")
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
            for tile in self.tiles:
                if tile.spec.name != "gating":
                    expected_central.extend(_enumerate_tile(tile))
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
                    "runtime_init_tile": event.runtime_init_tile,
                    "runtime_init": (
                        None
                        if event.runtime_init is None
                        else {
                            "tile": event.runtime_init.tile,
                            "value": event.runtime_init.value.to_data(),
                            "scope": event.runtime_init.scope,
                            "scope_id": event.runtime_init.scope_id,
                            "after_step": event.runtime_init.after_step,
                        }
                    ),
                }
                for event in self.events
            ],
            "tiles": [
                {
                    "name": tile.spec.name,
                    "job_type": tile.job_type,
                    "implementation": tile.implementation,
                    "upper_bounds": tile.upper_bounds,
                    "runtime_extents": [extent.to_data() for extent in tile.runtime_extents],
                    "scheduled_extents": [extent.to_data() for extent in tile.scheduled_extents],
                    "scheduled_upper_bounds": tile.scheduled_upper_bounds,
                    "execution_steps": tile.execution_steps,
                    "reads": [tensor.name for tensor in tile.spec.reads],
                    "writes": [tensor.name for tensor in tile.spec.writes],
                    "waits": [
                        {
                            "event": wait.event,
                            "coord": [coord.to_data() for coord in wait.coord],
                            "level": wait.level,
                            "mask": wait.mask,
                        }
                        for wait in tile.waits
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
                        for notify in tile.notifies
                    ],
                }
                for tile in self.tiles
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
                for dispatch in self.dispatch_plans
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


def _evaluate(expr: Expr, env: Mapping[str, int], label: str) -> int:
    try:
        value = expr.evaluate(env)
    except ValueError as err:
        raise ValueError(f"{label} is not statically evaluable") from err
    if not isinstance(value, int):
        raise ValueError(f"{label} must evaluate to an integer")
    return value


_WAIT_LEVEL_BY_TILE = {
    "topk": "cta",
    "align": "cta",
    "count_sort": "cta",
    "gate_up_silu": "warp",
    "down": "warp",
}
_NOTIFY_SCOPE_BY_TILE = {
    "gating": ("warpgroup", 0),
    "topk": ("cta", 0),
    "align": ("thread", 0),
    "count_sort": ("cta", 0),
    "gate_up_silu": ("warpgroup", 0),
}


def _logical_dependency_plans(
    env: MoeLoweringEnv, tile: TileSpec
) -> tuple[tuple[WaitPlan, ...], tuple[NotifyPlan, ...]]:
    waits = tuple(
        WaitPlan(
            logical_spec=dependency,
            event=dependency.event.name,
            coord=env.coord(tile, dependency),
            level=_WAIT_LEVEL_BY_TILE[tile.name],
        )
        for dependency in tile.waits
    )
    notifies = tuple(
        NotifyPlan(
            logical_spec=dependency,
            event=dependency.event.name,
            coord=env.coord(tile, dependency),
            scope=_NOTIFY_SCOPE_BY_TILE[tile.name][0],
            scope_id=_NOTIFY_SCOPE_BY_TILE[tile.name][1],
        )
        for dependency in tile.notifies
    )
    return waits, notifies


def _execution_steps(
    tile: TileSpec,
    waits: tuple[WaitPlan, ...],
    notifies: tuple[NotifyPlan, ...],
    *,
    is_dynamic: bool,
    runtime_init: bool,
) -> tuple[str, ...]:
    steps = []
    if is_dynamic:
        steps.append(_STEP_PRE_NOTIFY)
    if waits:
        steps.append(_STEP_WAIT)
    steps.append(_STEP_RUN)
    if tile.impl.implementation == "align":
        steps.append(_STEP_CTA_SYNC)
        steps.append(_STEP_RUNTIME_EVENT_INIT)
    elif runtime_init:
        raise ValueError("only the align tile may initialize a runtime event in the MoE MVP")
    if notifies:
        steps.append(_STEP_POST_NOTIFY)
    return tuple(steps)


def _normalize_tiles(
    env: MoeLoweringEnv,
    *,
    is_dynamic: bool,
    unfused: bool,
    down_coalescing: int,
    runtime_init_tiles: set[str],
) -> tuple[TilePlan, ...]:
    plans = []
    for tile in env.spec.tiles:
        waits, notifies = _logical_dependency_plans(env, tile)
        if unfused and tile.name == "gate_up_silu":
            notifies = tuple(
                replace(notify, coord=(ConstExpr(0),)) if notify.event == "gate_up_done" else notify
                for notify in notifies
            )
        elif unfused and tile.name == "down":
            waits = tuple(
                replace(wait, coord=(ConstExpr(0),)) if wait.event == "gate_up_done" else wait
                for wait in waits
            )
        runtime_extents = tuple(env.extent(extent) for extent in tile.tile_num)
        upper = tuple(env.upper_bound(extent) for extent in tile.tile_num)
        scheduled_extents = runtime_extents
        scheduled = upper
        if tile.name == "down" and down_coalescing != 1:
            scheduled_extents = (
                runtime_extents[0],
                runtime_extents[1] // down_coalescing,
                runtime_extents[2],
            )
            scheduled = (upper[0], upper[1] // down_coalescing, upper[2])
        plans.append(
            TilePlan(
                spec=tile,
                runtime_extents=runtime_extents,
                upper_bounds=upper,
                scheduled_extents=scheduled_extents,
                scheduled_upper_bounds=scheduled,
                execution_steps=_execution_steps(
                    tile,
                    waits,
                    notifies,
                    is_dynamic=is_dynamic,
                    runtime_init=tile.name in runtime_init_tiles,
                ),
                waits=waits,
                notifies=notifies,
            )
        )
    return tuple(plans)


def _event_plans(
    env: MoeLoweringEnv, *, is_dynamic: bool, unfused: bool, down_dispatch_groups: int
) -> tuple[EventPlan, ...]:
    offset = 0
    plans = []
    for event in env.spec.events.values():
        shape = env.event_shape(event)
        count = env.event_init_count(event)
        if event.name == "gate_up_done" and unfused:
            shape = (1,)
            count = env.rmax * 12
        plan = EventPlan(
            name=event.name,
            shape=shape,
            init_count=count,
            workspace_offset=offset,
            logical_spec=event,
        )
        plans.append(plan)
        offset += plan.size

    runtime_init = None
    down_count = env.rmax * 16
    if is_dynamic:
        down_count = None
        runtime_init = RuntimeEventInitPlan(
            "align", ConstExpr(SemaphoreBase.base + 1) * env.routed_rows * down_dispatch_groups
        )
    down_event = EventPlan(
        name="down_dispatch_done",
        shape=(1,),
        init_count=down_count,
        workspace_offset=offset,
        runtime_init=runtime_init,
    )
    plans.append(down_event)
    offset += down_event.size
    if not is_dynamic:
        complete = EventPlan(
            "event_init_complete", (1,), len(plans) + 1 + KernelConfig.SM_NUMBER, offset
        )
        plans.append(complete)
    return tuple(plans)


def _enumerate_tile(tile: TilePlan) -> list[HostTask]:
    result = []
    for m_idx in range(tile.scheduled_upper_bounds[0]):
        for n_idx in range(tile.scheduled_upper_bounds[1]):
            for k_idx in range(tile.scheduled_upper_bounds[2]):
                result.append(HostTask(tile.job_type, m_idx, n_idx, k_idx))
    return result


def _validate_packed_tiles(tiles: tuple[TilePlan, ...]):
    for tile in tiles:
        job_type = tile.job_type
        m_extent, n_extent, k_extent = tile.scheduled_upper_bounds
        if not 0 <= job_type < MAX_TASK_TYPE:
            raise ValueError(f"tile {tile.spec.name!r} overflows packed task type")
        if m_extent > MAX_M_IDX or n_extent > MAX_N_IDX or k_extent > MAX_K_IDX:
            raise ValueError(f"tile {tile.spec.name!r} overflows packed tile indices")


def _known_extent_bounds(tiles: tuple[TilePlan, ...]) -> dict[Expr, int]:
    bounds: dict[Expr, int] = {}
    for tile in tiles:
        for extent, upper_bound in zip(
            tile.scheduled_extents, tile.scheduled_upper_bounds, strict=True
        ):
            if isinstance(extent, ConstExpr):
                continue
            bounds[extent] = min(bounds.get(extent, upper_bound), upper_bound)
    return bounds


def _expr_interval(
    expr: Expr,
    *,
    compile_env: Mapping[str, int],
    known_bounds: Mapping[Expr, int],
    tile_bounds: Mapping[str, tuple[int, int, int]],
    var_bounds: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """Conservatively bound a narrow DSL expression for host-side validation."""

    if expr in known_bounds:
        return (0, known_bounds[expr])
    if isinstance(expr, ConstExpr):
        return (expr.value, expr.value)
    if isinstance(expr, VarExpr):
        if var_bounds is not None and expr.name in var_bounds:
            return var_bounds[expr.name]
        if expr.name in compile_env:
            value = compile_env[expr.name]
            return (value, value)
        raise ValueError(f"expression variable {expr.name!r} does not have a validated range")
    if isinstance(expr, TileIndexExpr):
        if expr.task not in tile_bounds:
            raise ValueError(f"tile index for {expr.task!r} does not have a validated range")
        return (0, tile_bounds[expr.task][expr.axis] - 1)
    if isinstance(expr, ScalarLoadExpr):
        raise ValueError(f"runtime scalar {expr.tensor!r} does not have a validated range")

    if not isinstance(expr, BinaryExpr | CeilDivExpr):
        raise TypeError(f"unsupported expression node {type(expr).__name__}")
    lhs_lo, lhs_hi = _expr_interval(
        expr.lhs,
        compile_env=compile_env,
        known_bounds=known_bounds,
        tile_bounds=tile_bounds,
        var_bounds=var_bounds,
    )
    rhs_lo, rhs_hi = _expr_interval(
        expr.rhs,
        compile_env=compile_env,
        known_bounds=known_bounds,
        tile_bounds=tile_bounds,
        var_bounds=var_bounds,
    )
    if isinstance(expr, CeilDivExpr):
        if rhs_lo <= 0:
            raise ValueError("ceildiv divisor does not have a positive validated range")
        quotients = tuple(
            (lhs + rhs - 1) // rhs for lhs in (lhs_lo, lhs_hi) for rhs in (rhs_lo, rhs_hi)
        )
        return (min(quotients), max(quotients))
    if expr.op == "+":
        return (lhs_lo + rhs_lo, lhs_hi + rhs_hi)
    if expr.op == "-":
        return (lhs_lo - rhs_hi, lhs_hi - rhs_lo)
    if expr.op == "*":
        products = (lhs_lo * rhs_lo, lhs_lo * rhs_hi, lhs_hi * rhs_lo, lhs_hi * rhs_hi)
        return (min(products), max(products))
    if rhs_lo <= 0:
        raise ValueError(f"{expr.op} divisor does not have a positive validated range")
    if expr.op == "//":
        quotients = (lhs_lo // rhs_lo, lhs_lo // rhs_hi, lhs_hi // rhs_lo, lhs_hi // rhs_hi)
        return (min(quotients), max(quotients))
    return (0, max(abs(rhs_lo), abs(rhs_hi)) - 1)


def _validate_scope(
    owner: str, scope: str, scope_id: int, count_bounds: tuple[int, int], rank: int
):
    if rank != -1:
        raise ValueError(f"{owner} uses a cross-rank notification outside the MoE DSL MVP")
    if (
        isinstance(scope_id, bool)
        or not isinstance(scope_id, int)
        or scope_id < -1
        or scope_id >= _SCOPE_INSTANCES[scope]
    ):
        raise ValueError(f"{owner} has an invalid {scope} scope id {scope_id!r}")
    count_lo, count_hi = count_bounds
    if count_lo < 0 or count_hi <= 0 or count_hi > _SCOPE_WIDTHS[scope]:
        raise ValueError(
            f"{owner} notification count range {count_bounds} exceeds its {scope} scope"
        )


def _event_coord_bounds(
    owner: str,
    event: EventPlan,
    coord: tuple[Expr, ...],
    *,
    compile_env: Mapping[str, int],
    known_bounds: Mapping[Expr, int],
    tile_bounds: Mapping[str, tuple[int, int, int]],
) -> tuple[tuple[int, int], ...]:
    bounds = tuple(
        _expr_interval(
            index, compile_env=compile_env, known_bounds=known_bounds, tile_bounds=tile_bounds
        )
        for index in coord
    )
    for axis, ((lower, upper), extent) in enumerate(zip(bounds, event.shape, strict=True)):
        if lower < 0 or upper >= extent:
            raise ValueError(
                f"{owner} event coordinate axis {axis} is outside event {event.name!r}"
            )
    return bounds


def _validate_tile_event_accesses(
    env: MoeLoweringEnv, tiles: tuple[TilePlan, ...], events: tuple[EventPlan, ...]
):
    event_map = {event.name: event for event in events}
    tile_bounds = {tile.spec.name: tile.scheduled_upper_bounds for tile in tiles}
    known_bounds = _known_extent_bounds(tiles)
    for tile in tiles:
        for wait in tile.waits:
            if (
                isinstance(wait.mask, bool)
                or not isinstance(wait.mask, int)
                or not 0 <= wait.mask <= 0xFFFFFFFF
            ):
                raise ValueError(f"tile {tile.spec.name!r} has an invalid wait mask")
            _event_coord_bounds(
                f"tile {tile.spec.name!r}",
                event_map[wait.event],
                wait.coord,
                compile_env=env.compile_env,
                known_bounds=known_bounds,
                tile_bounds=tile_bounds,
            )
        for notify in tile.notifies:
            if not isinstance(notify.release, bool):
                raise ValueError(f"tile {tile.spec.name!r} has a non-boolean release flag")
            _event_coord_bounds(
                f"tile {tile.spec.name!r}",
                event_map[notify.event],
                notify.coord,
                compile_env=env.compile_env,
                known_bounds=known_bounds,
                tile_bounds=tile_bounds,
            )
            count_bounds = _expr_interval(
                notify.count,
                compile_env=env.compile_env,
                known_bounds=known_bounds,
                tile_bounds=tile_bounds,
            )
            _validate_scope(
                f"tile {tile.spec.name!r}", notify.scope, notify.scope_id, count_bounds, notify.rank
            )


def _validate_event_notification_counts(
    env: MoeLoweringEnv, tiles: tuple[TilePlan, ...], events: tuple[EventPlan, ...]
):
    event_map = {event.name: event for event in events}
    tile_bounds = {tile.spec.name: tile.scheduled_upper_bounds for tile in tiles}
    known_bounds = _known_extent_bounds(tiles)
    for tile in tiles:
        tile_volume = reduce(mul, tile.scheduled_upper_bounds, 1)
        for notify in tile.notifies:
            event = event_map[notify.event]
            if event.init_count is None:
                raise ValueError(
                    f"event {event.name!r} is notified without an initialization count"
                )
            coord_bounds = _event_coord_bounds(
                f"tile {tile.spec.name!r}",
                event,
                notify.coord,
                compile_env=env.compile_env,
                known_bounds=known_bounds,
                tile_bounds=tile_bounds,
            )
            coord_count = reduce(mul, (upper - lower + 1 for lower, upper in coord_bounds), 1)
            count_bounds = _expr_interval(
                notify.count,
                compile_env=env.compile_env,
                known_bounds=known_bounds,
                tile_bounds=tile_bounds,
            )
            if count_bounds[0] != count_bounds[1] or tile_volume % coord_count:
                raise ValueError(
                    f"tile {tile.spec.name!r} notification coverage is not statically uniform"
                )
            scope_multiplier = _SCOPE_INSTANCES[notify.scope] if notify.scope_id == -1 else 1
            expected_count = tile_volume // coord_count * count_bounds[0] * scope_multiplier
            if expected_count != event.init_count:
                raise ValueError(
                    f"event {event.name!r} expects {event.init_count} notifications per "
                    f"coordinate, but tile {tile.spec.name!r} provides {expected_count}"
                )


def _normalize_dispatch(
    env: MoeLoweringEnv,
    tiles: tuple[TilePlan, ...],
    rules: tuple[_DynamicDispatchRule, ...],
    events: tuple[EventPlan, ...],
) -> tuple[DynamicDispatchPlan, ...]:
    tile_map = {tile.spec.name: tile for tile in tiles}
    event_map = {event.name: event for event in events}
    tile_bounds = {tile.spec.name: tile.scheduled_upper_bounds for tile in tiles}
    known_bounds = _known_extent_bounds(tiles)
    plans = []
    for rule in rules:
        count_lo, count_hi = _expr_interval(
            rule.count,
            compile_env=env.compile_env,
            known_bounds=known_bounds,
            tile_bounds=tile_bounds,
        )
        if count_lo < 0 or count_hi <= 0:
            raise ValueError(f"dynamic rule for {rule.source_tile!r} has an invalid count range")
        pre_count_bounds = _expr_interval(
            rule.pre_count,
            compile_env=env.compile_env,
            known_bounds=known_bounds,
            tile_bounds=tile_bounds,
        )
        _validate_scope(
            f"dynamic rule for {rule.source_tile!r}",
            rule.pre_scope,
            rule.pre_scope_id,
            pre_count_bounds,
            rule.rank,
        )
        if _SCOPE_ORDER[rule.push_level] > _SCOPE_ORDER[rule.pre_scope]:
            raise ValueError(
                f"dynamic rule for {rule.source_tile!r} cannot push at {rule.push_level} "
                f"from {rule.pre_scope}"
            )

        event = event_map[rule.event]
        event_coord_bounds = _event_coord_bounds(
            f"dynamic rule for {rule.source_tile!r}",
            event,
            rule.event_coord,
            compile_env=env.compile_env,
            known_bounds=known_bounds,
            tile_bounds=tile_bounds,
        )

        tile_index_bounds = tuple(
            _expr_interval(
                index,
                compile_env=env.compile_env,
                known_bounds=known_bounds,
                tile_bounds=tile_bounds,
                var_bounds={"push_idx": (0, count_hi - 1)},
            )
            for index in rule.tile_indices
        )
        target = None if rule.target_tile is None else tile_map[rule.target_tile]
        target_job = JobType.END.value if target is None else target.job_type
        if not 0 <= target_job < MAX_TASK_TYPE:
            raise ValueError(f"dynamic rule for {rule.source_tile!r} overflows packed task type")
        for axis, ((lower, upper), limit) in enumerate(
            zip(tile_index_bounds, _PACKED_INDEX_LIMITS, strict=True)
        ):
            if lower < 0 or upper >= limit:
                raise ValueError(
                    f"dynamic rule for {rule.source_tile!r} overflows packed tile indices"
                )
            if target is not None and upper >= target.scheduled_upper_bounds[axis]:
                raise ValueError(
                    f"dynamic rule for {rule.source_tile!r} maps outside target tile "
                    f"{rule.target_tile!r} axis {axis}"
                )

        trigger_upper_bound = reduce(
            mul, (upper - lower + 1 for lower, upper in event_coord_bounds), 1
        )
        plans.append(
            DynamicDispatchPlan(
                rule,
                trigger_upper_bound,
                count_lo,
                count_hi,
                trigger_upper_bound * count_hi,
                event_coord_bounds,
                tile_index_bounds,
            )
        )
    return tuple(plans)


def _validate_dynamic_protocol_links(
    tiles: tuple[TilePlan, ...], dispatch_plans: tuple[DynamicDispatchPlan, ...]
):
    tile_map = {tile.spec.name: tile for tile in tiles}
    for dispatch in dispatch_plans:
        rule = dispatch.rule
        source = tile_map[rule.source_tile]
        if rule.target_tile is None:
            if source.notifies:
                raise ValueError(
                    f"terminal tile {source.spec.name!r} must only pre-notify its drain event"
                )
            continue
        if len(source.notifies) != 1:
            raise ValueError(
                f"dynamic tile {source.spec.name!r} must have one completion notification"
            )
        notify = source.notifies[0]
        pre_scope_multiplier = _SCOPE_INSTANCES[rule.pre_scope] if rule.pre_scope_id == -1 else 1
        post_scope_multiplier = _SCOPE_INSTANCES[notify.scope] if notify.scope_id == -1 else 1
        if (
            notify.event != rule.event
            or notify.coord != rule.event_coord
            or notify.count != rule.pre_count
            or notify.rank != rule.rank
            or pre_scope_multiplier != post_scope_multiplier
        ):
            raise ValueError(
                f"dynamic tile {source.spec.name!r} pre/post notifications are inconsistent"
            )


def _validate_dispatch_coverage(
    env: MoeLoweringEnv,
    tiles: tuple[TilePlan, ...],
    dispatch_plans: tuple[DynamicDispatchPlan, ...],
):
    """Prove that every non-seed tile is pushed exactly once at its upper bound."""

    tile_map = {tile.spec.name: tile for tile in tiles}
    incoming: dict[str, int] = {}
    for dispatch in dispatch_plans:
        rule = dispatch.rule
        if rule.target_tile is None:
            continue
        incoming[rule.target_tile] = incoming.get(rule.target_tile, 0) + 1

        if any(isinstance(node, TileIndexExpr) for node in walk_expr(rule.count)):
            raise ValueError(
                f"dynamic rule for {rule.source_tile!r} has a tile-dependent push count"
            )
        event_tile_axes: dict[int, int] = {}
        for coord_axis, coord in enumerate(rule.event_coord):
            tile_nodes = [node for node in walk_expr(coord) if isinstance(node, TileIndexExpr)]
            if any(isinstance(node, ScalarLoadExpr) for node in walk_expr(coord)):
                raise ValueError(
                    f"dynamic rule for {rule.source_tile!r} has a runtime event coordinate"
                )
            if tile_nodes:
                if len(tile_nodes) != 1 or coord != tile_nodes[0]:
                    raise ValueError(
                        f"dynamic rule for {rule.source_tile!r} event coordinate is not "
                        "directly enumerable"
                    )
                tile_axis = tile_nodes[0].axis
                if tile_axis in event_tile_axes.values():
                    raise ValueError(
                        f"dynamic rule for {rule.source_tile!r} repeats a source tile axis"
                    )
                event_tile_axes[coord_axis] = tile_axis

        for index in rule.tile_indices:
            for node in walk_expr(index):
                if isinstance(node, ScalarLoadExpr):
                    raise ValueError(
                        f"dynamic rule for {rule.source_tile!r} has a runtime tile mapping"
                    )
                if isinstance(node, TileIndexExpr) and node.axis not in event_tile_axes.values():
                    raise ValueError(
                        f"dynamic rule for {rule.source_tile!r} maps a source tile axis "
                        "that is not fixed by its event coordinate"
                    )

        generated = []
        coord_ranges = [range(lower, upper + 1) for lower, upper in dispatch.event_coord_bounds]
        for event_coord in product(*coord_ranges):
            source_tile = [0, 0, 0]
            for coord_axis, tile_axis in event_tile_axes.items():
                source_tile[tile_axis] = event_coord[coord_axis]
            for push_idx in range(dispatch.count_upper_bound):
                eval_env = {
                    "vars": {**env.compile_env, "push_idx": push_idx},
                    "tiles": {rule.source_tile: tuple(source_tile)},
                }
                generated.append(tuple(index.evaluate(eval_env) for index in rule.tile_indices))

        target = tile_map[rule.target_tile]
        expected = set(
            product(*(range(upper_bound) for upper_bound in target.scheduled_upper_bounds))
        )
        if (
            len(generated) != dispatch.enqueue_upper_bound
            or len(generated) != len(set(generated))
            or set(generated) != expected
        ):
            raise ValueError(
                f"dynamic rule for {rule.source_tile!r} does not cover target tile "
                f"{rule.target_tile!r} exactly once"
            )

    expected_targets = {tile.spec.name for tile in tiles if tile.spec.name != "gating"}
    if set(incoming) != expected_targets or any(count != 1 for count in incoming.values()):
        raise ValueError("dynamic dispatch graph must have one incoming rule per non-seed tile")


def _dynamic_dispatch_rules(
    env: MoeLoweringEnv, down_dispatch_groups: int
) -> tuple[_DynamicDispatchRule, ...]:
    """Derive dynamic queue transitions from the five logical dependency edges."""

    push_idx = VarExpr("push_idx")
    gate_up_m = TileIndexExpr("gate_up_silu", 0)
    return (
        _DynamicDispatchRule(
            "gating",
            "gating_done",
            (ConstExpr(0),),
            "topk",
            ConstExpr(KernelConfig.SM_NUMBER),
            (push_idx, ConstExpr(0), ConstExpr(0)),
            "warpgroup",
            "warpgroup",
            pre_scope_id=0,
        ),
        _DynamicDispatchRule(
            "topk",
            "topk_done",
            (ConstExpr(0),),
            "align",
            ConstExpr(1),
            (ConstExpr(0), ConstExpr(0), ConstExpr(0)),
            "thread",
            "thread",
        ),
        _DynamicDispatchRule(
            "align",
            "align_done",
            (ConstExpr(0),),
            "count_sort",
            ConstExpr(KernelConfig.SM_NUMBER),
            (push_idx, ConstExpr(0), ConstExpr(0)),
            "cta",
            "cta",
        ),
        _DynamicDispatchRule(
            "count_sort",
            "count_sort_done",
            (ConstExpr(0),),
            "gate_up_silu",
            env.routed_rows * 12,
            (push_idx // 12, push_idx % 12, ConstExpr(0)),
            "cta",
            "cta",
        ),
        _DynamicDispatchRule(
            "gate_up_silu",
            "gate_up_done",
            (gate_up_m,),
            "down",
            ConstExpr(down_dispatch_groups),
            (gate_up_m, push_idx, ConstExpr(0)),
            "warp",
            "warp",
        ),
        _DynamicDispatchRule(
            "down",
            "down_dispatch_done",
            (ConstExpr(0),),
            None,
            ConstExpr(KernelConfig.SM_NUMBER),
            (ConstExpr(0), ConstExpr(0), ConstExpr(0)),
            "warp",
            "warp",
        ),
    )


def _validate_policy_edges(tiles: tuple[TilePlan, ...], rules: tuple[_DynamicDispatchRule, ...]):
    """Prove each non-terminal policy transition implements one logical edge."""

    tile_map = {tile.spec.name: tile for tile in tiles}
    for rule in rules:
        source = tile_map[rule.source_tile]
        if rule.target_tile is None:
            if rule.event != "down_dispatch_done" or source.notifies:
                raise ValueError("terminal dynamic rule must use the synthesized drain event")
            continue
        target = tile_map[rule.target_tile]
        matching_notifies = [notify for notify in source.notifies if notify.event == rule.event]
        matching_waits = [wait for wait in target.waits if wait.event == rule.event]
        if len(matching_notifies) != 1 or len(matching_waits) != 1:
            raise ValueError(
                f"dynamic rule {rule.source_tile!r} -> {rule.target_tile!r} "
                "does not match one logical notify/wait edge"
            )
        notify = matching_notifies[0]
        wait = matching_waits[0]
        if (
            notify.logical_spec.event is not wait.logical_spec.event
            or notify.coord != rule.event_coord
        ):
            raise ValueError(
                f"dynamic rule for {rule.source_tile!r} is inconsistent with its logical edge"
            )


class MoePolicy:
    name = "base"
    is_dynamic = False
    unfused = False

    def __init__(self, *, queue_capacity: int | None = None):
        self.queue_capacity = queue_capacity

    def normalize(self, spec: KernelSpec) -> NormalizedPlan:
        raise NotImplementedError


class StaticPolicy(MoePolicy):
    name = "static"

    def normalize(self, spec: KernelSpec) -> NormalizedPlan:
        env = MoeLoweringEnv(spec)
        events = _event_plans(env, is_dynamic=False, unfused=self.unfused, down_dispatch_groups=16)
        tiles = _normalize_tiles(
            env, is_dynamic=False, unfused=self.unfused, down_coalescing=1, runtime_init_tiles=set()
        )
        _validate_packed_tiles(tiles)
        by_name = {tile.spec.name: tile for tile in tiles}
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
            if tile.spec.name != "gating":
                central.extend(_enumerate_tile(tile))
        queue_columns = (len(central) + KernelConfig.SM_NUMBER - 1) // KernelConfig.SM_NUMBER + 1
        capacity = (
            StaticTileScheduler.MAX_TASKS if self.queue_capacity is None else self.queue_capacity
        )
        if capacity != StaticTileScheduler.MAX_TASKS or queue_columns > capacity:
            raise ValueError(
                f"static host queue requires {queue_columns} columns, capacity is {capacity}"
            )
        return NormalizedPlan(
            spec=spec,
            env=env,
            policy_name=self.name,
            is_dynamic=False,
            unfused=self.unfused,
            events=events,
            tiles=tiles,
            dispatch_plans=(),
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
        env = MoeLoweringEnv(spec)
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
        events = _event_plans(
            env, is_dynamic=True, unfused=False, down_dispatch_groups=down_dispatch_groups
        )
        tiles = _normalize_tiles(
            env,
            is_dynamic=True,
            unfused=False,
            down_coalescing=coalescing,
            runtime_init_tiles={
                event.runtime_init.tile for event in events if event.runtime_init is not None
            },
        )
        _validate_packed_tiles(tiles)
        dispatch_rules = _dynamic_dispatch_rules(env, down_dispatch_groups)
        _validate_policy_edges(tiles, dispatch_rules)
        dispatch_plans = _normalize_dispatch(env, tiles, dispatch_rules, events)
        by_name = {tile.spec.name: tile for tile in tiles}
        seed = [
            HostTask(JobType.INIT_ETENSOR.value, event_idx, 0, 0)
            for event_idx in range(len(events))
        ]
        seed.extend(_enumerate_tile(by_name["gating"]))
        queue_upper_bound = len(seed) + sum(
            dispatch.enqueue_upper_bound for dispatch in dispatch_plans
        )
        if queue_upper_bound > capacity:
            raise ValueError(
                f"dynamic queue upper bound {queue_upper_bound} exceeds capacity {capacity}"
            )
        if capacity != DynamicTileScheduler.MAX_TASKS:
            raise ValueError(
                f"dynamic queue capacity must remain {DynamicTileScheduler.MAX_TASKS}; got {capacity}"
            )
        return NormalizedPlan(
            spec=spec,
            env=env,
            policy_name=self.name,
            is_dynamic=True,
            unfused=False,
            events=events,
            tiles=tiles,
            dispatch_plans=dispatch_plans,
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
    graph = build_moe_graph(config, batch_size)
    return MoeLowerer(policy_for_scheduler(scheduler)).lower(graph)


class MoeLowerer:
    """Lower a normalized graph into event setup, dispatch, and opaque tile calls."""

    def __init__(self, policy: MoePolicy, owner=None):
        self.policy = policy
        self.owner = owner
        self.plan: NormalizedPlan | None = None

    def lower(self, spec: KernelSpec) -> NormalizedPlan:
        self.plan = self.policy.normalize(spec)
        return self.plan

    def _require_plan(self) -> NormalizedPlan:
        if self.plan is None:
            raise RuntimeError("MoeLowerer must lower a KernelSpec before code emission")
        return self.plan

    def register_tiles(self):
        """Register TileImpl-held tasks in the existing lifecycle order."""

        plan = self._require_plan()
        if self.owner is None:
            raise RuntimeError("tile registration requires a MegaKernelMOE owner")
        self.owner.reset()
        for tile in plan.tiles:
            tile.spec.impl.register(self.owner)

    def bind_context(self, context):
        """Bind lowering buffers to every concrete TileImpl adapter."""

        for tile in self._require_plan().tiles:
            tile.spec.impl.bind_context(context)

    def init_events(self, semaphore_cls, etensor_workspace_global):
        plan = self._require_plan()
        if self.owner is None:
            raise RuntimeError("event lowering requires a MegaKernelMOE owner")
        for event in plan.user_events:
            initializer = None if event.init_count is None else f_init_const(event.init_count)
            semaphore = self.owner.add_etensor(
                semaphore_cls, etensor_workspace_global, shape=list(event.shape), f_init=initializer
            )
            setattr(self.owner, _EVENT_ATTRS[event.name], semaphore)
        self.owner.set_events_complete(plan.is_dynamic, semaphore_cls, etensor_workspace_global)
        self.owner.num_etensors[plan.is_dynamic] = len(self.owner.etensor_and_f_init_pairs)
        if self.owner.etensor_workspace_offset != plan.workspace_size:
            raise ValueError(
                "DSL event workspace layout diverged from its normalized plan: "
                f"{self.owner.etensor_workspace_offset} != {plan.workspace_size}"
            )

    def _expr_env(self, tile: TilePlan, context, *, push_idx=None):
        plan = self._require_plan()
        variables = dict(plan.env.compile_env)
        if push_idx is not None:
            variables["push_idx"] = push_idx
        scheduler = self.owner.tile_scheduler
        return {
            "vars": variables,
            "tensors": context,
            "tiles": {tile.spec.name: (scheduler.m_idx, scheduler.n_idx, scheduler.k_idx)},
        }

    def _event(self, name: str):
        return getattr(self.owner, _EVENT_ATTRS[name])

    def _emit_pre_notify(self, tile: TilePlan, context):
        plan = self._require_plan()
        rule = plan.dispatch(tile.spec.name)
        event = self._event(rule.event)
        notify_env = self._expr_env(tile, context)

        def notify_fn(notify_idx):
            del notify_idx
            return (
                rule.pre_count.lower(notify_env),
                rule.rank,
                *(coord.lower(notify_env) for coord in rule.event_coord),
            )

        target_job = (
            JobType.END.value if rule.target_tile is None else plan.tile(rule.target_tile).job_type
        )

        def trigger_fn(trigger_idx):
            del trigger_idx

            def push_fn(push_idx):
                push_env = self._expr_env(tile, context, push_idx=push_idx)
                return (
                    target_job,
                    rule.count.lower(push_env),
                    *(index.lower(push_env) for index in rule.tile_indices),
                )

            return push_fn

        self.owner.tile_scheduler.pre_notify_and_push(
            event,
            notify_fn,
            trigger_fn,
            rule.push_level,
            rule.pre_scope,
            scope_id=rule.pre_scope_id,
        )

    def _emit_waits(self, tile: TilePlan, context):
        env = self._expr_env(tile, context)
        for wait in tile.waits:
            coord = tuple(index.lower(env) for index in wait.coord)
            if wait.mask == 0xFFFFFFFF:
                self.owner.tile_scheduler.wait(
                    self._event(wait.event), *coord, wait_level=wait.level
                )
            else:
                self.owner.tile_scheduler.wait(
                    self._event(wait.event), *coord, wait_level=wait.level, mask=wait.mask
                )

    def _emit_notifies(self, tile: TilePlan, context):
        env = self._expr_env(tile, context)
        for notify in tile.notifies:
            event = self._event(notify.event)

            def notify_fn(notify_idx, notify=notify):
                del notify_idx
                return (
                    notify.count.lower(env),
                    notify.rank,
                    *(coord.lower(env) for coord in notify.coord),
                )

            self.owner.tile_scheduler.notify(
                event,
                notify_fn,
                scope=notify.scope,
                scope_id=notify.scope_id,
                release=notify.release,
            )

    def _run_tile_impl(self, tile: TilePlan, context):
        scheduler = self.owner.tile_scheduler
        implementation = tile.implementation
        if implementation == "gate_up_silu":

            def run_gate_up():
                tile.spec.impl.run(scheduler.m_idx, scheduler.n_idx, scheduler.k_idx)

            plan = self._require_plan()
            if_frame = T.If(
                T.Or(
                    T.bool(plan.is_dynamic),
                    scheduler.m_idx < context["num_tokens_post_pad"][0] // 128,
                )
            )
            if_frame.__enter__()
            with T.Then():
                run_gate_up()
            if_frame.__exit__(None, None, None)
        elif implementation == "down":
            plan = self._require_plan()

            def run_down():
                with T.serial(plan.down_coalescing) as index:
                    tile.spec.impl.run(
                        scheduler.m_idx,
                        scheduler.n_idx * plan.down_coalescing + index,
                        scheduler.k_idx,
                    )

            if_frame = T.If(
                T.Or(
                    T.bool(plan.is_dynamic),
                    scheduler.m_idx < context["num_tokens_post_pad"][0] // 128,
                )
            )
            if_frame.__enter__()
            with T.Then():
                run_down()
            if_frame.__exit__(None, None, None)
        else:
            tile.spec.impl.run(scheduler.m_idx, scheduler.n_idx, scheduler.k_idx)

    @T.inline
    def _emit_runtime_event_init(
        self, tile, context, event_name, runtime_init, runtime_init_scope_id, tid
    ):
        if tid == runtime_init_scope_id:
            if runtime_init is not None:
                self._event(event_name).sem[0] = runtime_init.value.lower(
                    self._expr_env(tile, context)
                )

    @T.inline
    def _emit_align_tile(
        self,
        tile,
        context,
        emit_pre,
        emit_wait,
        emit_init,
        runtime_event_name,
        runtime_init,
        runtime_init_scope_id,
        emit_post,
    ):
        tid = T.thread_id([KernelConfig.NUM_THREADS])
        if emit_pre:
            self._emit_pre_notify(tile, context)
        if emit_wait:
            self._emit_waits(tile, context)
        self._run_tile_impl(tile, context)
        T.cuda.cta_sync()
        if emit_init:
            self._emit_runtime_event_init(
                tile, context, runtime_event_name, runtime_init, runtime_init_scope_id, tid
            )
        if emit_post:
            self._emit_notifies(tile, context)

    def _emit_tile(self, tile: TilePlan, context):
        if tile.spec.name == "align":
            steps = tile.execution_steps
            runtime_events = [
                event
                for event in self._require_plan().events
                if event.runtime_init is not None and event.runtime_init.tile == tile.spec.name
            ]
            if len(runtime_events) > 1:
                raise ValueError("a tile may initialize at most one runtime event in the MoE MVP")
            runtime_event = runtime_events[0] if runtime_events else None
            self._emit_align_tile(
                tile,
                context,
                _STEP_PRE_NOTIFY in steps,
                _STEP_WAIT in steps,
                _STEP_RUNTIME_EVENT_INIT in steps,
                None if runtime_event is None else runtime_event.name,
                None if runtime_event is None else runtime_event.runtime_init,
                0 if runtime_event is None else runtime_event.runtime_init.scope_id,
                _STEP_POST_NOTIFY in steps,
            )
            return
        for step in tile.execution_steps:
            if step == _STEP_PRE_NOTIFY:
                self._emit_pre_notify(tile, context)
            elif step == _STEP_WAIT:
                self._emit_waits(tile, context)
            elif step == _STEP_RUN:
                self._run_tile_impl(tile, context)
            elif step == _STEP_POST_NOTIFY:
                self._emit_notifies(tile, context)
            else:
                raise ValueError(f"unsupported execution step {step!r} for tile {tile.spec.name!r}")

    def dispatch_loop_body(self, context):
        """Emit the task-type dispatch chain from normalized tile bindings."""

        plan = self._require_plan()
        self.bind_context(context)
        entries: list[tuple[int, TilePlan | str]] = [(tile.job_type, tile) for tile in plan.tiles]
        entries.extend(
            [
                (JobType.INIT_ETENSOR.value, "init_event"),
                (JobType.WAIT_ETENSOR_INIT.value, "wait_event_init"),
            ]
        )
        task_type = self.owner.tile_scheduler.task_type
        if_frames = [T.If(task_type == job_type) for job_type, _ in entries]
        then_frames = [T.Then() for _ in entries]
        else_frames = [T.Else() for _ in entries]
        for index, (_, entry) in enumerate(entries):
            if_frames[index].__enter__()
            with then_frames[index]:
                if isinstance(entry, TilePlan):
                    self._emit_tile(entry, context)
                elif entry == "init_event":
                    self.owner.task_impl_init_etensor(plan.is_dynamic)
                else:
                    self.owner.task_impl_wait_etensor_init_complete(plan.is_dynamic)
            else_frames[index].__enter__()
        T.evaluate(T.cuda.trap_when_assert_failed(False))
        for index in range(len(entries) - 1, -1, -1):
            else_frames[index].__exit__(None, None, None)
            if_frames[index].__exit__(None, None, None)
