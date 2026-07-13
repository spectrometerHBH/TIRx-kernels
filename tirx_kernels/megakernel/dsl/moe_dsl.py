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
from dataclasses import dataclass, replace
from functools import reduce
from operator import mul
from typing import Any

import numpy as np

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

from .expr import ConstExpr, Expr, ScalarLoadExpr, TileIndexExpr, VarExpr, ceildiv
from .spec import (
    DispatchSpec,
    EventSpec,
    KernelSpec,
    NotifySpec,
    TaskDomain,
    TaskSpec,
    TensorSpec,
    TileBinding,
    WaitSpec,
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

    batch = VarExpr("B")
    max_rows = _max_rows(batch_size)
    max_tokens = max_rows * 128
    num_tokens_post_pad = TensorSpec("num_tokens_post_pad", (1,), "int32", "intermediate")
    runtime_rows = num_tokens_post_pad.scalar(0) // 128
    relaxed_rows = batch * 8 // 128 + 129
    push_idx = VarExpr("push_idx")

    tensors = (
        TensorSpec("hidden_state", (batch, 2048), "float16", "input"),
        TensorSpec("residual", (batch, 2048), "float16", "input"),
        TensorSpec("output", (batch, 2048), "float16", "output"),
        TensorSpec("gate_weight", (128, 2048), "float16", "input"),
        TensorSpec("gate_up_weight", (128, 1536, 2048), "float16", "input"),
        TensorSpec("down_weight", (128, 2048, 768), "float16", "input"),
        TensorSpec("gating_output", (batch, 128), "float32", "intermediate"),
        TensorSpec("topk_weights", (batch, 8), "float32", "intermediate"),
        TensorSpec("topk_indices", (batch, 8), "int32", "intermediate"),
        TensorSpec("sorted_token_ids", (max_tokens,), "int32", "intermediate"),
        TensorSpec("expert_ids", (max_rows,), "int32", "intermediate"),
        TensorSpec("num_valid_tokens", (max_rows,), "int32", "intermediate"),
        num_tokens_post_pad,
        TensorSpec("cumsum_buffer", (129,), "int32", "intermediate"),
        TensorSpec("reordered_hidden_state", (max_tokens, 2048), "float16", "intermediate"),
        TensorSpec("silu_mul_output", (max_tokens, 768), "float16", "intermediate"),
        TensorSpec("topk_reduce_output", (batch, 2048), "float16", "intermediate"),
    )

    events = (
        EventSpec("gating_done", (1,), 4 * ceildiv(batch, 128)),
        EventSpec("topk_done", (1,), 148),
        EventSpec("align_done", (1,), 1),
        EventSpec("count_sort_done", (1,), 148),
        EventSpec("gate_up_done", (relaxed_rows,), 12),
        EventSpec("down_dispatch_done", (1,), max_rows * 16),
    )

    gate_up_m = TileIndexExpr("gate_up_silu", 0)
    down_m = TileIndexExpr("down", 0)

    tasks = (
        TaskSpec(
            "gating",
            TaskDomain((ceildiv(batch, 128), 1, 4), (ceildiv(batch, 128), 1, 4)),
            TileBinding(JobType.MOE_GATING.value, "gating"),
            reads=("hidden_state", "gate_weight"),
            writes=("gating_output",),
            notifies=(NotifySpec("gating_done", (0,), "warpgroup", 0),),
        ),
        TaskSpec(
            "topk",
            TaskDomain((148, 1, 1), (148, 1, 1)),
            TileBinding(JobType.MOE_TOPK_SOFTMAX.value, "topk"),
            reads=("gating_output",),
            writes=("topk_weights", "topk_indices"),
            waits=(WaitSpec("gating_done", (0,), "cta"),),
            notifies=(NotifySpec("topk_done", (0,), "cta", 0),),
        ),
        TaskSpec(
            "align",
            TaskDomain((1, 1, 1), (1, 1, 1)),
            TileBinding(JobType.MOE_ALIGN.value, "align"),
            reads=("topk_indices",),
            writes=(
                "sorted_token_ids",
                "expert_ids",
                "num_valid_tokens",
                "num_tokens_post_pad",
                "cumsum_buffer",
            ),
            waits=(WaitSpec("topk_done", (0,), "cta"),),
            notifies=(NotifySpec("align_done", (0,), "thread", 0),),
        ),
        TaskSpec(
            "count_sort",
            TaskDomain((148, 1, 1), (148, 1, 1)),
            TileBinding(JobType.MOE_COUNT_AND_SORT.value, "count_sort"),
            reads=(
                "topk_indices",
                "sorted_token_ids",
                "cumsum_buffer",
                "hidden_state",
                "num_tokens_post_pad",
            ),
            writes=("reordered_hidden_state",),
            waits=(WaitSpec("align_done", (0,), "cta"),),
            notifies=(NotifySpec("count_sort_done", (0,), "cta", 0),),
        ),
        TaskSpec(
            "gate_up_silu",
            TaskDomain((runtime_rows, 12, 1), (max_rows, 12, 1)),
            TileBinding(JobType.MOE_GROUP_GEMM_GATE_UP_SILU.value, "gate_up_silu"),
            reads=(
                "reordered_hidden_state",
                "gate_up_weight",
                "topk_weights",
                "sorted_token_ids",
                "expert_ids",
                "num_valid_tokens",
                "num_tokens_post_pad",
            ),
            writes=("silu_mul_output",),
            waits=(WaitSpec("count_sort_done", (0,), "warp"),),
            notifies=(NotifySpec("gate_up_done", (gate_up_m,), "warpgroup", 0),),
        ),
        TaskSpec(
            "down",
            TaskDomain((runtime_rows, 16, 1), (max_rows, 16, 1)),
            TileBinding(JobType.MOE_GROUP_GEMM_DOWN.value, "down"),
            reads=(
                "silu_mul_output",
                "down_weight",
                "expert_ids",
                "topk_weights",
                "sorted_token_ids",
                "num_valid_tokens",
                "num_tokens_post_pad",
            ),
            writes=("topk_reduce_output",),
            waits=(WaitSpec("gate_up_done", (down_m,), "warp"),),
        ),
    )

    dispatch = (
        DispatchSpec(
            "gating",
            "gating_done",
            (0,),
            "topk",
            148,
            (push_idx, 0, 0),
            "warpgroup",
            "warpgroup",
            pre_scope_id=0,
        ),
        DispatchSpec("topk", "topk_done", (0,), "align", 1, (0, 0, 0), "thread", "thread"),
        DispatchSpec(
            "align", "align_done", (0,), "count_sort", 148, (push_idx, 0, 0), "cta", "cta"
        ),
        DispatchSpec(
            "count_sort",
            "count_sort_done",
            (0,),
            "gate_up_silu",
            runtime_rows * 12,
            (push_idx // 12, push_idx % 12, 0),
            "cta",
            "cta",
        ),
        DispatchSpec(
            "gate_up_silu",
            "gate_up_done",
            (gate_up_m,),
            "down",
            16,
            (gate_up_m, push_idx, 0),
            "warp",
            "warp",
        ),
        DispatchSpec("down", "down_dispatch_done", (0,), None, 148, (0, 0, 0), "warp", "warp"),
    )

    return KernelSpec(
        "qwen3_30b_a3b_moe", tensors, events, tasks, dispatch, compile_env={"B": batch_size}
    )


@dataclass(frozen=True)
class EventPlan:
    name: str
    shape: tuple[int, ...]
    init_count: int | None
    workspace_offset: int
    runtime_init_task: str | None = None

    @property
    def size(self) -> int:
        return reduce(mul, self.shape, 1)


@dataclass(frozen=True)
class TaskPlan:
    spec: TaskSpec
    upper_bounds: tuple[int, int, int]
    scheduled_upper_bounds: tuple[int, int, int]


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
    policy_name: str
    is_dynamic: bool
    unfused: bool
    events: tuple[EventPlan, ...]
    tasks: tuple[TaskPlan, ...]
    dispatch_rules: tuple[DispatchSpec, ...]
    central_tasks: tuple[HostTask, ...]
    seed_tasks: tuple[HostTask, ...]
    down_coalescing: int
    down_dispatch_groups: int
    queue_capacity: int
    queue_upper_bound: int
    persistent_ctas: int
    pre_before_wait: bool
    fifo_drain: bool

    @property
    def user_events(self) -> tuple[EventPlan, ...]:
        return tuple(event for event in self.events if event.name != "event_init_complete")

    @property
    def workspace_size(self) -> int:
        return sum(event.size for event in self.events)

    def event(self, name: str) -> EventPlan:
        return next(event for event in self.events if event.name == name)

    def task(self, name: str) -> TaskPlan:
        return next(task for task in self.tasks if task.spec.name == name)

    def dispatch(self, source_task: str) -> DispatchSpec:
        return next(rule for rule in self.dispatch_rules if rule.source_task == source_task)

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
                    "runtime_init_task": event.runtime_init_task,
                }
                for event in self.events
            ],
            "tasks": [
                {
                    "name": task.spec.name,
                    "upper_bounds": task.upper_bounds,
                    "scheduled_upper_bounds": task.scheduled_upper_bounds,
                    "waits": [
                        {
                            "event": wait.event,
                            "coord": [coord.to_data() for coord in wait.coord],
                            "level": wait.level,
                            "mask": wait.mask,
                        }
                        for wait in task.spec.waits
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
                        for notify in task.spec.notifies
                    ],
                }
                for task in self.tasks
            ],
            "central_task_count": len(self.central_tasks),
            "seed_tasks": [task.as_manual_tuple() for task in self.seed_tasks],
            "dispatch": [
                {
                    "source_task": rule.source_task,
                    "event": rule.event,
                    "target_task": rule.target_task,
                    "count": rule.count.to_data(),
                    "tile_indices": [index.to_data() for index in rule.tile_indices],
                    "push_level": rule.push_level,
                    "pre_scope": rule.pre_scope,
                    "pre_scope_id": rule.pre_scope_id,
                }
                for rule in self.dispatch_rules
            ],
            "down_coalescing": self.down_coalescing,
            "down_dispatch_groups": self.down_dispatch_groups,
            "queue_capacity": self.queue_capacity,
            "queue_upper_bound": self.queue_upper_bound,
            "persistent_ctas": self.persistent_ctas,
            "pre_before_wait": self.pre_before_wait,
            "fifo_drain": self.fifo_drain,
        }


def _evaluate(expr: Expr, env: Mapping[str, int], label: str) -> int:
    try:
        value = expr.evaluate(env)
    except ValueError as err:
        raise ValueError(f"{label} is not statically evaluable") from err
    if not isinstance(value, int):
        raise ValueError(f"{label} must evaluate to an integer")
    return value


def _normalize_tasks(
    spec: KernelSpec, *, unfused: bool, down_coalescing: int
) -> tuple[TaskPlan, ...]:
    plans = []
    for task in spec.tasks:
        normalized = task
        if unfused and task.name == "gate_up_silu":
            normalized = replace(
                task,
                notifies=tuple(
                    replace(notify, coord=(ConstExpr(0),))
                    if notify.event == "gate_up_done"
                    else notify
                    for notify in task.notifies
                ),
            )
        elif unfused and task.name == "down":
            normalized = replace(
                task,
                waits=tuple(
                    replace(wait, coord=(ConstExpr(0),)) if wait.event == "gate_up_done" else wait
                    for wait in task.waits
                ),
            )
        upper = tuple(
            _evaluate(bound, spec.compile_env, f"task {task.name} upper bound")
            for bound in task.domain.upper_bounds
        )
        scheduled = upper
        if task.name == "down" and down_coalescing != 1:
            scheduled = (upper[0], upper[1] // down_coalescing, upper[2])
        plans.append(TaskPlan(normalized, upper, scheduled))
    return tuple(plans)


def _event_plans(spec: KernelSpec, *, is_dynamic: bool, unfused: bool) -> tuple[EventPlan, ...]:
    offset = 0
    plans = []
    max_rows = _max_rows(spec.compile_env["B"])
    for event in spec.events:
        shape = tuple(
            _evaluate(extent, spec.compile_env, f"event {event.name} shape")
            for extent in event.shape
        )
        count = (
            None
            if event.init_count is None
            else _evaluate(event.init_count, spec.compile_env, f"event {event.name} count")
        )
        runtime_init_task = None
        if event.name == "gate_up_done" and unfused:
            shape = (1,)
            count = max_rows * 12
        if event.name == "down_dispatch_done" and is_dynamic:
            count = None
            runtime_init_task = "align"
        plan = EventPlan(event.name, shape, count, offset, runtime_init_task)
        plans.append(plan)
        offset += plan.size
    if not is_dynamic:
        complete = EventPlan(
            "event_init_complete", (1,), len(plans) + 1 + KernelConfig.SM_NUMBER, offset
        )
        plans.append(complete)
    return tuple(plans)


def _enumerate_task(task: TaskPlan) -> list[HostTask]:
    result = []
    for m_idx in range(task.scheduled_upper_bounds[0]):
        for n_idx in range(task.scheduled_upper_bounds[1]):
            for k_idx in range(task.scheduled_upper_bounds[2]):
                result.append(HostTask(task.spec.tile_binding.job_type, m_idx, n_idx, k_idx))
    return result


def _validate_packed_tasks(tasks: tuple[TaskPlan, ...]):
    for task in tasks:
        job_type = task.spec.tile_binding.job_type
        m_extent, n_extent, k_extent = task.scheduled_upper_bounds
        if not 0 <= job_type < MAX_TASK_TYPE:
            raise ValueError(f"task {task.spec.name!r} overflows packed task type")
        if m_extent > MAX_M_IDX or n_extent > MAX_N_IDX or k_extent > MAX_K_IDX:
            raise ValueError(f"task {task.spec.name!r} overflows packed tile indices")


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
        spec.validate()
        events = _event_plans(spec, is_dynamic=False, unfused=self.unfused)
        tasks = _normalize_tasks(spec, unfused=self.unfused, down_coalescing=1)
        _validate_packed_tasks(tasks)
        by_name = {task.spec.name: task for task in tasks}
        central = [
            HostTask(JobType.INIT_ETENSOR.value, event_idx, 0, 0)
            for event_idx in range(len(events))
        ]
        central.extend(_enumerate_task(by_name["gating"]))
        central.extend(
            HostTask(JobType.WAIT_ETENSOR_INIT.value, cta, 0, 0)
            for cta in range(KernelConfig.SM_NUMBER)
        )
        for name in ("topk", "align", "count_sort", "gate_up_silu", "down"):
            central.extend(_enumerate_task(by_name[name]))
        queue_columns = (len(central) + KernelConfig.SM_NUMBER - 1) // KernelConfig.SM_NUMBER + 1
        capacity = self.queue_capacity or StaticTileScheduler.MAX_TASKS
        if capacity != StaticTileScheduler.MAX_TASKS or queue_columns > capacity:
            raise ValueError(
                f"static host queue requires {queue_columns} columns, capacity is {capacity}"
            )
        return NormalizedPlan(
            spec,
            self.name,
            False,
            self.unfused,
            events,
            tasks,
            (),
            tuple(central),
            (),
            1,
            16,
            capacity,
            len(central),
            KernelConfig.SM_NUMBER,
            True,
            True,
        )


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
        spec.validate()
        if any(notify.release for task in spec.tasks for notify in task.notifies):
            raise ValueError("dynamic notification release is outside the MoE DSL MVP")
        batch_size = spec.compile_env["B"]
        expected_coalescing = 1 if batch_size < 4 else 4
        coalescing = expected_coalescing if self.down_coalescing is None else self.down_coalescing
        if coalescing != expected_coalescing or coalescing <= 0 or 16 % coalescing:
            raise ValueError(
                f"illegal dynamic down coalescing q={coalescing} for batch {batch_size}"
            )
        capacity = self.queue_capacity
        if capacity is None or capacity <= 0 or capacity & (capacity - 1):
            raise ValueError("dynamic queue capacity must be a positive power of two")
        events = _event_plans(spec, is_dynamic=True, unfused=False)
        tasks = _normalize_tasks(spec, unfused=False, down_coalescing=coalescing)
        _validate_packed_tasks(tasks)
        dispatch = tuple(
            replace(rule, count=ConstExpr(16 // coalescing))
            if rule.source_task == "gate_up_silu"
            else rule
            for rule in spec.dynamic_dispatch
        )
        by_name = {task.spec.name: task for task in tasks}
        seed = [
            HostTask(JobType.INIT_ETENSOR.value, event_idx, 0, 0)
            for event_idx in range(len(events))
        ]
        seed.extend(_enumerate_task(by_name["gating"]))
        queue_upper_bound = (
            len(seed)
            + 148
            + 1
            + 148
            + reduce(mul, by_name["gate_up_silu"].scheduled_upper_bounds, 1)
            + reduce(mul, by_name["down"].scheduled_upper_bounds, 1)
            + 148
        )
        if queue_upper_bound > capacity:
            raise ValueError(
                f"dynamic queue upper bound {queue_upper_bound} exceeds capacity {capacity}"
            )
        fanout = {
            rule.source_task: _evaluate(rule.count, spec.compile_env, "dispatch count")
            for rule in dispatch
            if not any(isinstance(node, ScalarLoadExpr) for node in _walk(rule.count))
        }
        if (
            KernelConfig.SM_NUMBER != 148
            or fanout.get("gating") != 148
            or fanout.get("align") != 148
            or fanout.get("down") != 148
        ):
            raise ValueError("dynamic policy violates persistent CTA saturation invariants")
        return NormalizedPlan(
            spec,
            self.name,
            True,
            False,
            events,
            tasks,
            dispatch,
            (),
            tuple(seed),
            coalescing,
            16 // coalescing,
            capacity,
            queue_upper_bound,
            KernelConfig.SM_NUMBER,
            True,
            True,
        )


def _walk(expr: Expr):
    yield expr
    if hasattr(expr, "lhs"):
        yield from _walk(expr.lhs)
        yield from _walk(expr.rhs)
    elif isinstance(expr, ScalarLoadExpr):
        yield from _walk(expr.index)


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
    return graph.lower(MoeLowerer(policy_for_scheduler(scheduler)))


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

    def _expr_env(self, task: TaskPlan, context, *, push_idx=None):
        plan = self._require_plan()
        variables = dict(plan.spec.compile_env)
        if push_idx is not None:
            variables["push_idx"] = push_idx
        scheduler = self.owner.tile_scheduler
        return {
            "vars": variables,
            "tensors": context,
            "tiles": {task.spec.name: (scheduler.m_idx, scheduler.n_idx, scheduler.k_idx)},
        }

    def _event(self, name: str):
        return getattr(self.owner, _EVENT_ATTRS[name])

    def _emit_pre_notify(self, task: TaskPlan, context):
        plan = self._require_plan()
        rule = plan.dispatch(task.spec.name)
        event = self._event(rule.event)
        notify_env = self._expr_env(task, context)

        def notify_fn(notify_idx):
            del notify_idx
            return (
                rule.pre_count.lower(notify_env),
                rule.rank,
                *(coord.lower(notify_env) for coord in rule.event_coord),
            )

        target_job = (
            JobType.END.value
            if rule.target_task is None
            else plan.task(rule.target_task).spec.tile_binding.job_type
        )

        def trigger_fn(trigger_idx):
            del trigger_idx

            def push_fn(push_idx):
                push_env = self._expr_env(task, context, push_idx=push_idx)
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

    def _emit_waits(self, task: TaskPlan, context):
        env = self._expr_env(task, context)
        for wait in task.spec.waits:
            coord = tuple(index.lower(env) for index in wait.coord)
            if wait.mask == 0xFFFFFFFF:
                self.owner.tile_scheduler.wait(
                    self._event(wait.event), *coord, wait_level=wait.level
                )
            else:
                self.owner.tile_scheduler.wait(
                    self._event(wait.event), *coord, wait_level=wait.level, mask=wait.mask
                )

    def _emit_notifies(self, task: TaskPlan, context):
        plan = self._require_plan()
        env = self._expr_env(task, context)
        for notify in task.spec.notifies:
            event = self._event(notify.event)

            def notify_fn(notify_idx, notify=notify):
                del notify_idx
                return (
                    notify.count.lower(env),
                    notify.rank,
                    *(coord.lower(env) for coord in notify.coord),
                )

            if plan.is_dynamic:
                if notify.release:
                    raise ValueError("dynamic notification release is outside the MoE DSL MVP")
                self.owner.tile_scheduler.notify(
                    event, notify_fn, scope=notify.scope, scope_id=notify.scope_id
                )
            else:
                self.owner.tile_scheduler.notify(
                    event,
                    notify_fn,
                    scope=notify.scope,
                    scope_id=notify.scope_id,
                    release=notify.release,
                )

    def _run_opaque_tile(self, task: TaskPlan, context):
        scheduler = self.owner.tile_scheduler
        implementation = task.spec.tile_binding.implementation
        if implementation == "gating":
            self.owner.run_tile(
                self.owner.gate,
                scheduler.m_idx,
                scheduler.n_idx,
                scheduler.k_idx,
                context["hidden_state"],
                context["gate_weight"],
                context["gating_output"],
                self.owner.profiler,
            )
        elif implementation == "topk":
            self.owner.run_tile(
                self.owner.topk_softmax,
                scheduler.m_idx,
                scheduler.n_idx,
                scheduler.k_idx,
                context["gating_output"],
                context["topk_weights"],
                context["topk_indices"],
                renormalize=False,
            )
        elif implementation == "align":
            self.owner.run_tile(
                self.owner.align,
                scheduler.m_idx,
                scheduler.n_idx,
                scheduler.k_idx,
                context["topk_indices_flat"],
                context["sorted_token_ids"],
                context["expert_ids"],
                context["num_tokens_post_pad"],
                context["cumsum_buffer"],
                context["num_valid_tokens"],
            )
        elif implementation == "count_sort":
            self.owner.run_tile(
                self.owner.count_and_sort_expert_tokens,
                scheduler.m_idx,
                scheduler.n_idx,
                scheduler.k_idx,
                context["topk_indices_flat"],
                context["sorted_token_ids"],
                context["cumsum_buffer"],
                context["hidden_state"],
                context["reordered_hidden_state"],
            )
        elif implementation == "gate_up_silu":

            def run_gate_up():
                self.owner.run_tile(
                    self.owner.group_gemm_gate_up_silu,
                    scheduler.m_idx,
                    scheduler.n_idx,
                    scheduler.k_idx,
                    context["reordered_hidden_state"],
                    context["gate_up_weight"],
                    context["silu_mul_output"],
                    context["expert_ids"],
                    context["topk_weights_flat"],
                    context["sorted_token_ids"],
                    context["num_valid_tokens"],
                    self.owner.profiler,
                )

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
                    self.owner.run_tile(
                        self.owner.group_gemm_down,
                        scheduler.m_idx,
                        scheduler.n_idx * plan.down_coalescing + index,
                        scheduler.k_idx,
                        context["silu_mul_output"],
                        context["down_weight"],
                        context["topk_reduce_output"],
                        context["expert_ids"],
                        context["topk_weights_flat"],
                        context["sorted_token_ids"],
                        context["num_valid_tokens"],
                        self.owner.profiler,
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
            raise ValueError(f"unknown opaque tile implementation {implementation!r}")

    @T.inline
    def _emit_align_task(self, task: TaskPlan, context, is_dynamic):
        plan = T.meta_var(self._require_plan())
        tid = T.thread_id([KernelConfig.NUM_THREADS])
        if is_dynamic:
            self._emit_pre_notify(task, context)
        self._emit_waits(task, context)
        self._run_opaque_tile(task, context)
        T.cuda.cta_sync()
        if tid == 0:
            if is_dynamic:
                self._event("down_dispatch_done").sem[0] = (
                    (self._event("down_dispatch_done").base + 1)
                    * (context["num_tokens_post_pad"][0] // 128)
                    * plan.down_dispatch_groups
                )
        self._emit_notifies(task, context)

    def _emit_task(self, task: TaskPlan, context):
        plan = self._require_plan()
        if task.spec.name == "align":
            self._emit_align_task(task, context, plan.is_dynamic)
            return
        if plan.is_dynamic:
            self._emit_pre_notify(task, context)
        self._emit_waits(task, context)
        self._run_opaque_tile(task, context)
        self._emit_notifies(task, context)

    def dispatch_loop_body(self, context):
        """Emit the task-type dispatch chain from normalized task bindings."""

        plan = self._require_plan()
        entries: list[tuple[int, TaskPlan | str]] = [
            (task.spec.tile_binding.job_type, task) for task in plan.tasks
        ]
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
                if isinstance(entry, TaskPlan):
                    self._emit_task(entry, context)
                elif entry == "init_event":
                    self.owner.task_impl_init_etensor(plan.is_dynamic)
                else:
                    self.owner.task_impl_wait_etensor_init_complete(plan.is_dynamic)
            else_frames[index].__enter__()
        T.evaluate(T.cuda.trap_when_assert_failed(False))
        for index in range(len(entries) - 1, -1, -1):
            else_frames[index].__exit__(None, None, None)
            if_frames[index].__exit__(None, None, None)
