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

from .expr import (
    BinaryExpr,
    CeilDivExpr,
    ConstExpr,
    Expr,
    ScalarLoadExpr,
    TileIndexExpr,
    VarExpr,
    ceildiv,
    walk_expr,
)
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
class RuntimeEventInitPlan:
    """A post-tile store of a raw dynamic semaphore value."""

    task: str
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

    @property
    def size(self) -> int:
        return reduce(mul, self.shape, 1)

    @property
    def runtime_init_task(self) -> str | None:
        return None if self.runtime_init is None else self.runtime_init.task


@dataclass(frozen=True)
class TaskPlan:
    spec: TaskSpec
    upper_bounds: tuple[int, int, int]
    scheduled_extents: tuple[Expr, Expr, Expr]
    scheduled_upper_bounds: tuple[int, int, int]
    execution_steps: tuple[str, ...]


@dataclass(frozen=True)
class DispatchPlan:
    """A dispatch rule plus statically proven trigger, count, and index ranges."""

    rule: DispatchSpec
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
    policy_name: str
    is_dynamic: bool
    unfused: bool
    events: tuple[EventPlan, ...]
    tasks: tuple[TaskPlan, ...]
    dispatch_plans: tuple[DispatchPlan, ...]
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
    def dispatch_rules(self) -> tuple[DispatchSpec, ...]:
        return tuple(dispatch.rule for dispatch in self.dispatch_plans)

    @property
    def pre_before_wait(self) -> bool:
        if not self.is_dynamic:
            return True
        for task in self.tasks:
            steps = task.execution_steps
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
        for task in self.tasks:
            steps = task.execution_steps
            has_post = _STEP_POST_NOTIFY in steps
            if has_post != bool(task.spec.notifies):
                return False
            if has_post and steps.index(_STEP_POST_NOTIFY) < steps.index(_STEP_RUN):
                return False
        return True

    @property
    def fifo_drain(self) -> bool:
        if not self.is_dynamic or self.protocol is None:
            return True
        terminal = [
            dispatch for dispatch in self.dispatch_plans if dispatch.rule.target_task is None
        ]
        if len(terminal) != 1:
            return False
        dispatch = terminal[0]
        source = self.task(dispatch.rule.source_task)
        return (
            self.protocol.queue_discipline == "fifo"
            and dispatch.trigger_upper_bound == 1
            and dispatch.count_lower_bound == self.persistent_ctas
            and dispatch.count_upper_bound == self.persistent_ctas
            and not source.spec.notifies
            and _STEP_POST_NOTIFY not in source.execution_steps
        )

    def event(self, name: str) -> EventPlan:
        return next(event for event in self.events if event.name == name)

    def task(self, name: str) -> TaskPlan:
        return next(task for task in self.tasks if task.spec.name == name)

    def dispatch(self, source_task: str) -> DispatchSpec:
        return next(rule for rule in self.dispatch_rules if rule.source_task == source_task)

    def dispatch_plan(self, source_task: str) -> DispatchPlan:
        return next(
            dispatch for dispatch in self.dispatch_plans if dispatch.rule.source_task == source_task
        )

    def validate(self) -> NormalizedPlan:
        offset = 0
        runtime_inits: dict[str, list[EventPlan]] = {}
        for event in self.events:
            if event.workspace_offset != offset:
                raise ValueError(f"event {event.name!r} has a non-contiguous workspace offset")
            offset += event.size
            if event.runtime_init is not None:
                runtime_inits.setdefault(event.runtime_init.task, []).append(event)

        task_names = {task.spec.name for task in self.tasks}
        if set(runtime_inits) - task_names:
            raise ValueError("runtime event initialization references an unknown task")
        if any(len(events) > 1 for events in runtime_inits.values()):
            raise ValueError("a task may initialize at most one runtime event in the MoE MVP")
        for task in self.tasks:
            steps = task.execution_steps
            if any(step not in _EXECUTION_STEPS for step in steps) or len(steps) != len(set(steps)):
                raise ValueError(f"task {task.spec.name!r} has an invalid execution plan")
            if steps.count(_STEP_RUN) != 1:
                raise ValueError(f"task {task.spec.name!r} must execute its tile exactly once")
            if (_STEP_WAIT in steps) != bool(task.spec.waits):
                raise ValueError(f"task {task.spec.name!r} wait step does not match its waits")
            if _STEP_WAIT in steps and steps.index(_STEP_WAIT) > steps.index(_STEP_RUN):
                raise ValueError(f"task {task.spec.name!r} must wait before tile execution")
            if (_STEP_POST_NOTIFY in steps) != bool(task.spec.notifies):
                raise ValueError(f"task {task.spec.name!r} post step does not match its notifies")
            if (_STEP_RUNTIME_EVENT_INIT in steps) != (
                task.spec.tile_binding.implementation == "align"
            ):
                raise ValueError(
                    f"task {task.spec.name!r} has an invalid runtime initialization slot"
                )
            if task.spec.tile_binding.implementation == "align":
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
                raise ValueError(f"task {task.spec.name!r} has an unsupported CTA synchronization")
            for event in runtime_inits.get(task.spec.name, ()):
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
        _validate_task_event_accesses(self.spec, self.tasks, self.events)
        _validate_event_notification_counts(self.spec, self.tasks, self.events)

        if self.is_dynamic:
            if self.central_tasks or not self.seed_tasks:
                raise ValueError("dynamic plan must use seed tasks instead of a central queue")
            expected_seed = tuple(
                [
                    HostTask(JobType.INIT_ETENSOR.value, event_idx, 0, 0)
                    for event_idx in range(len(self.events))
                ]
                + _enumerate_task(self.task("gating"))
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
                len(self.dispatch_plans) != len(task_names)
                or {dispatch.rule.source_task for dispatch in self.dispatch_plans} != task_names
            ):
                raise ValueError("dynamic plan does not have one dispatch rule per task")
            expected_dispatch = _normalize_dispatch(
                self.spec, self.tasks, self.dispatch_rules, self.events
            )
            if self.dispatch_plans != expected_dispatch:
                raise ValueError("dynamic dispatch bounds do not match their expressions")
            _validate_dynamic_protocol_links(self.tasks, self.dispatch_plans)
            _validate_dispatch_coverage(self.spec, self.tasks, self.dispatch_plans)
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
                * self.task("down").scheduled_extents[0]
                * self.down_dispatch_groups
            )
            if (
                runtime_init is None
                or runtime_init.task != "align"
                or runtime_init.scope != "thread"
                or runtime_init.scope_id != 0
                or runtime_init.value != expected_value
            ):
                raise ValueError("dynamic down event has an invalid runtime initialization")
        else:
            if self.dispatch_plans or self.seed_tasks or self.protocol is not None:
                raise ValueError("static plan contains dynamic scheduling state")
            if any(_STEP_PRE_NOTIFY in task.execution_steps for task in self.tasks):
                raise ValueError("static task cannot contain a pre-notification step")
            if runtime_inits:
                raise ValueError("static plan cannot contain runtime event initialization")
            expected_central = [
                HostTask(JobType.INIT_ETENSOR.value, event_idx, 0, 0)
                for event_idx in range(len(self.events))
            ]
            expected_central.extend(_enumerate_task(self.task("gating")))
            expected_central.extend(
                HostTask(JobType.WAIT_ETENSOR_INIT.value, cta, 0, 0)
                for cta in range(self.persistent_ctas)
            )
            for task in self.tasks:
                if task.spec.name != "gating":
                    expected_central.extend(_enumerate_task(task))
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
                    "runtime_init_task": event.runtime_init_task,
                    "runtime_init": (
                        None
                        if event.runtime_init is None
                        else {
                            "task": event.runtime_init.task,
                            "value": event.runtime_init.value.to_data(),
                            "scope": event.runtime_init.scope,
                            "scope_id": event.runtime_init.scope_id,
                            "after_step": event.runtime_init.after_step,
                        }
                    ),
                }
                for event in self.events
            ],
            "tasks": [
                {
                    "name": task.spec.name,
                    "job_type": task.spec.tile_binding.job_type,
                    "implementation": task.spec.tile_binding.implementation,
                    "upper_bounds": task.upper_bounds,
                    "scheduled_extents": [extent.to_data() for extent in task.scheduled_extents],
                    "scheduled_upper_bounds": task.scheduled_upper_bounds,
                    "execution_steps": task.execution_steps,
                    "reads": list(task.spec.reads),
                    "writes": list(task.spec.writes),
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
            "central_tasks": [task.as_manual_tuple() for task in self.central_tasks],
            "seed_tasks": [task.as_manual_tuple() for task in self.seed_tasks],
            "dispatch": [
                {
                    "source_task": dispatch.rule.source_task,
                    "event": dispatch.rule.event,
                    "event_coord": [coord.to_data() for coord in dispatch.rule.event_coord],
                    "target_task": dispatch.rule.target_task,
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


def _execution_steps(task: TaskSpec, *, is_dynamic: bool, runtime_init: bool) -> tuple[str, ...]:
    steps = []
    if is_dynamic:
        steps.append(_STEP_PRE_NOTIFY)
    if task.waits:
        steps.append(_STEP_WAIT)
    steps.append(_STEP_RUN)
    if task.tile_binding.implementation == "align":
        steps.append(_STEP_CTA_SYNC)
        steps.append(_STEP_RUNTIME_EVENT_INIT)
    elif runtime_init:
        raise ValueError("only the align task may initialize a runtime event in the MoE MVP")
    if task.notifies:
        steps.append(_STEP_POST_NOTIFY)
    return tuple(steps)


def _normalize_tasks(
    spec: KernelSpec,
    *,
    is_dynamic: bool,
    unfused: bool,
    down_coalescing: int,
    runtime_init_tasks: set[str],
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
        scheduled_extents = normalized.domain.extents
        scheduled = upper
        if task.name == "down" and down_coalescing != 1:
            scheduled_extents = (
                normalized.domain.extents[0],
                normalized.domain.extents[1] // down_coalescing,
                normalized.domain.extents[2],
            )
            scheduled = (upper[0], upper[1] // down_coalescing, upper[2])
        plans.append(
            TaskPlan(
                normalized,
                upper,
                scheduled_extents,
                scheduled,
                _execution_steps(
                    normalized,
                    is_dynamic=is_dynamic,
                    runtime_init=normalized.name in runtime_init_tasks,
                ),
            )
        )
    return tuple(plans)


def _event_plans(
    spec: KernelSpec, *, is_dynamic: bool, unfused: bool, down_dispatch_groups: int
) -> tuple[EventPlan, ...]:
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
        runtime_init = None
        if event.name == "gate_up_done" and unfused:
            shape = (1,)
            count = max_rows * 12
        if event.name == "down_dispatch_done" and is_dynamic:
            count = None
            runtime_rows = next(task for task in spec.tasks if task.name == "down").domain.extents[
                0
            ]
            runtime_init = RuntimeEventInitPlan(
                "align", ConstExpr(SemaphoreBase.base + 1) * runtime_rows * down_dispatch_groups
            )
        plan = EventPlan(event.name, shape, count, offset, runtime_init)
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


def _known_extent_bounds(tasks: tuple[TaskPlan, ...]) -> dict[Expr, int]:
    bounds: dict[Expr, int] = {}
    for task in tasks:
        for extent, upper_bound in zip(
            task.scheduled_extents, task.scheduled_upper_bounds, strict=True
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


def _validate_task_event_accesses(
    spec: KernelSpec, tasks: tuple[TaskPlan, ...], events: tuple[EventPlan, ...]
):
    event_map = {event.name: event for event in events}
    tile_bounds = {task.spec.name: task.scheduled_upper_bounds for task in tasks}
    known_bounds = _known_extent_bounds(tasks)
    for task in tasks:
        for wait in task.spec.waits:
            if (
                isinstance(wait.mask, bool)
                or not isinstance(wait.mask, int)
                or not 0 <= wait.mask <= 0xFFFFFFFF
            ):
                raise ValueError(f"task {task.spec.name!r} has an invalid wait mask")
            _event_coord_bounds(
                f"task {task.spec.name!r}",
                event_map[wait.event],
                wait.coord,
                compile_env=spec.compile_env,
                known_bounds=known_bounds,
                tile_bounds=tile_bounds,
            )
        for notify in task.spec.notifies:
            if not isinstance(notify.release, bool):
                raise ValueError(f"task {task.spec.name!r} has a non-boolean release flag")
            _event_coord_bounds(
                f"task {task.spec.name!r}",
                event_map[notify.event],
                notify.coord,
                compile_env=spec.compile_env,
                known_bounds=known_bounds,
                tile_bounds=tile_bounds,
            )
            count_bounds = _expr_interval(
                notify.count,
                compile_env=spec.compile_env,
                known_bounds=known_bounds,
                tile_bounds=tile_bounds,
            )
            _validate_scope(
                f"task {task.spec.name!r}", notify.scope, notify.scope_id, count_bounds, notify.rank
            )


def _validate_event_notification_counts(
    spec: KernelSpec, tasks: tuple[TaskPlan, ...], events: tuple[EventPlan, ...]
):
    event_map = {event.name: event for event in events}
    tile_bounds = {task.spec.name: task.scheduled_upper_bounds for task in tasks}
    known_bounds = _known_extent_bounds(tasks)
    for task in tasks:
        task_volume = reduce(mul, task.scheduled_upper_bounds, 1)
        for notify in task.spec.notifies:
            event = event_map[notify.event]
            if event.init_count is None:
                raise ValueError(
                    f"event {event.name!r} is notified without an initialization count"
                )
            coord_bounds = _event_coord_bounds(
                f"task {task.spec.name!r}",
                event,
                notify.coord,
                compile_env=spec.compile_env,
                known_bounds=known_bounds,
                tile_bounds=tile_bounds,
            )
            coord_count = reduce(mul, (upper - lower + 1 for lower, upper in coord_bounds), 1)
            count_bounds = _expr_interval(
                notify.count,
                compile_env=spec.compile_env,
                known_bounds=known_bounds,
                tile_bounds=tile_bounds,
            )
            if count_bounds[0] != count_bounds[1] or task_volume % coord_count:
                raise ValueError(
                    f"task {task.spec.name!r} notification coverage is not statically uniform"
                )
            scope_multiplier = _SCOPE_INSTANCES[notify.scope] if notify.scope_id == -1 else 1
            expected_count = task_volume // coord_count * count_bounds[0] * scope_multiplier
            if expected_count != event.init_count:
                raise ValueError(
                    f"event {event.name!r} expects {event.init_count} notifications per "
                    f"coordinate, but task {task.spec.name!r} provides {expected_count}"
                )


def _normalize_dispatch(
    spec: KernelSpec,
    tasks: tuple[TaskPlan, ...],
    rules: tuple[DispatchSpec, ...],
    events: tuple[EventPlan, ...],
) -> tuple[DispatchPlan, ...]:
    task_map = {task.spec.name: task for task in tasks}
    event_map = {event.name: event for event in events}
    tile_bounds = {task.spec.name: task.scheduled_upper_bounds for task in tasks}
    known_bounds = _known_extent_bounds(tasks)
    plans = []
    for rule in rules:
        count_lo, count_hi = _expr_interval(
            rule.count,
            compile_env=spec.compile_env,
            known_bounds=known_bounds,
            tile_bounds=tile_bounds,
        )
        if count_lo < 0 or count_hi <= 0:
            raise ValueError(f"dynamic rule for {rule.source_task!r} has an invalid count range")
        pre_count_bounds = _expr_interval(
            rule.pre_count,
            compile_env=spec.compile_env,
            known_bounds=known_bounds,
            tile_bounds=tile_bounds,
        )
        _validate_scope(
            f"dynamic rule for {rule.source_task!r}",
            rule.pre_scope,
            rule.pre_scope_id,
            pre_count_bounds,
            rule.rank,
        )
        if _SCOPE_ORDER[rule.push_level] > _SCOPE_ORDER[rule.pre_scope]:
            raise ValueError(
                f"dynamic rule for {rule.source_task!r} cannot push at {rule.push_level} "
                f"from {rule.pre_scope}"
            )

        event = event_map[rule.event]
        event_coord_bounds = _event_coord_bounds(
            f"dynamic rule for {rule.source_task!r}",
            event,
            rule.event_coord,
            compile_env=spec.compile_env,
            known_bounds=known_bounds,
            tile_bounds=tile_bounds,
        )

        tile_index_bounds = tuple(
            _expr_interval(
                index,
                compile_env=spec.compile_env,
                known_bounds=known_bounds,
                tile_bounds=tile_bounds,
                var_bounds={"push_idx": (0, count_hi - 1)},
            )
            for index in rule.tile_indices
        )
        target = None if rule.target_task is None else task_map[rule.target_task]
        target_job = JobType.END.value if target is None else target.spec.tile_binding.job_type
        if not 0 <= target_job < MAX_TASK_TYPE:
            raise ValueError(f"dynamic rule for {rule.source_task!r} overflows packed task type")
        for axis, ((lower, upper), limit) in enumerate(
            zip(tile_index_bounds, _PACKED_INDEX_LIMITS, strict=True)
        ):
            if lower < 0 or upper >= limit:
                raise ValueError(
                    f"dynamic rule for {rule.source_task!r} overflows packed tile indices"
                )
            if target is not None and upper >= target.scheduled_upper_bounds[axis]:
                raise ValueError(
                    f"dynamic rule for {rule.source_task!r} maps outside target task "
                    f"{rule.target_task!r} axis {axis}"
                )

        trigger_upper_bound = reduce(
            mul, (upper - lower + 1 for lower, upper in event_coord_bounds), 1
        )
        plans.append(
            DispatchPlan(
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
    tasks: tuple[TaskPlan, ...], dispatch_plans: tuple[DispatchPlan, ...]
):
    task_map = {task.spec.name: task for task in tasks}
    for dispatch in dispatch_plans:
        rule = dispatch.rule
        source = task_map[rule.source_task]
        if rule.target_task is None:
            if source.spec.notifies:
                raise ValueError(
                    f"terminal task {source.spec.name!r} must only pre-notify its drain event"
                )
            continue
        if len(source.spec.notifies) != 1:
            raise ValueError(
                f"dynamic task {source.spec.name!r} must have one completion notification"
            )
        notify = source.spec.notifies[0]
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
                f"dynamic task {source.spec.name!r} pre/post notifications are inconsistent"
            )


def _validate_dispatch_coverage(
    spec: KernelSpec, tasks: tuple[TaskPlan, ...], dispatch_plans: tuple[DispatchPlan, ...]
):
    """Prove that every non-seed task is pushed exactly once at its upper bound."""

    task_map = {task.spec.name: task for task in tasks}
    incoming: dict[str, int] = {}
    for dispatch in dispatch_plans:
        rule = dispatch.rule
        if rule.target_task is None:
            continue
        incoming[rule.target_task] = incoming.get(rule.target_task, 0) + 1

        if any(isinstance(node, TileIndexExpr) for node in walk_expr(rule.count)):
            raise ValueError(
                f"dynamic rule for {rule.source_task!r} has a tile-dependent push count"
            )
        event_tile_axes: dict[int, int] = {}
        for coord_axis, coord in enumerate(rule.event_coord):
            tile_nodes = [node for node in walk_expr(coord) if isinstance(node, TileIndexExpr)]
            if any(isinstance(node, ScalarLoadExpr) for node in walk_expr(coord)):
                raise ValueError(
                    f"dynamic rule for {rule.source_task!r} has a runtime event coordinate"
                )
            if tile_nodes:
                if len(tile_nodes) != 1 or coord != tile_nodes[0]:
                    raise ValueError(
                        f"dynamic rule for {rule.source_task!r} event coordinate is not "
                        "directly enumerable"
                    )
                tile_axis = tile_nodes[0].axis
                if tile_axis in event_tile_axes.values():
                    raise ValueError(
                        f"dynamic rule for {rule.source_task!r} repeats a source tile axis"
                    )
                event_tile_axes[coord_axis] = tile_axis

        for index in rule.tile_indices:
            for node in walk_expr(index):
                if isinstance(node, ScalarLoadExpr):
                    raise ValueError(
                        f"dynamic rule for {rule.source_task!r} has a runtime tile mapping"
                    )
                if isinstance(node, TileIndexExpr) and node.axis not in event_tile_axes.values():
                    raise ValueError(
                        f"dynamic rule for {rule.source_task!r} maps a source tile axis "
                        "that is not fixed by its event coordinate"
                    )

        generated = []
        coord_ranges = [range(lower, upper + 1) for lower, upper in dispatch.event_coord_bounds]
        for event_coord in product(*coord_ranges):
            source_tile = [0, 0, 0]
            for coord_axis, tile_axis in event_tile_axes.items():
                source_tile[tile_axis] = event_coord[coord_axis]
            for push_idx in range(dispatch.count_upper_bound):
                env = {
                    "vars": {**spec.compile_env, "push_idx": push_idx},
                    "tiles": {rule.source_task: tuple(source_tile)},
                }
                generated.append(tuple(index.evaluate(env) for index in rule.tile_indices))

        target = task_map[rule.target_task]
        expected = set(
            product(*(range(upper_bound) for upper_bound in target.scheduled_upper_bounds))
        )
        if (
            len(generated) != dispatch.enqueue_upper_bound
            or len(generated) != len(set(generated))
            or set(generated) != expected
        ):
            raise ValueError(
                f"dynamic rule for {rule.source_task!r} does not cover target task "
                f"{rule.target_task!r} exactly once"
            )

    expected_targets = {task.spec.name for task in tasks if task.spec.name != "gating"}
    if set(incoming) != expected_targets or any(count != 1 for count in incoming.values()):
        raise ValueError("dynamic dispatch graph must have one incoming rule per non-seed task")


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
        events = _event_plans(spec, is_dynamic=False, unfused=self.unfused, down_dispatch_groups=16)
        tasks = _normalize_tasks(
            spec,
            is_dynamic=False,
            unfused=self.unfused,
            down_coalescing=1,
            runtime_init_tasks=set(),
        )
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
        for task in tasks:
            if task.spec.name != "gating":
                central.extend(_enumerate_task(task))
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
            policy_name=self.name,
            is_dynamic=False,
            unfused=self.unfused,
            events=events,
            tasks=tasks,
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
        spec.validate()
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
        down_dispatch_groups = 16 // coalescing
        events = _event_plans(
            spec, is_dynamic=True, unfused=False, down_dispatch_groups=down_dispatch_groups
        )
        tasks = _normalize_tasks(
            spec,
            is_dynamic=True,
            unfused=False,
            down_coalescing=coalescing,
            runtime_init_tasks={
                event.runtime_init.task for event in events if event.runtime_init is not None
            },
        )
        _validate_packed_tasks(tasks)
        dispatch_rules = tuple(
            replace(rule, count=ConstExpr(down_dispatch_groups))
            if rule.source_task == "gate_up_silu"
            else rule
            for rule in spec.dynamic_dispatch
        )
        dispatch_plans = _normalize_dispatch(spec, tasks, dispatch_rules, events)
        by_name = {task.spec.name: task for task in tasks}
        seed = [
            HostTask(JobType.INIT_ETENSOR.value, event_idx, 0, 0)
            for event_idx in range(len(events))
        ]
        seed.extend(_enumerate_task(by_name["gating"]))
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
            policy_name=self.name,
            is_dynamic=True,
            unfused=False,
            events=events,
            tasks=tasks,
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
                self.owner.tile_scheduler.notify(
                    event,
                    notify_fn,
                    scope=notify.scope,
                    scope_id=notify.scope_id,
                    release=notify.release,
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
    def _emit_runtime_event_init(
        self, task, context, event_name, runtime_init, runtime_init_scope_id, tid
    ):
        if tid == runtime_init_scope_id:
            if runtime_init is not None:
                self._event(event_name).sem[0] = runtime_init.value.lower(
                    self._expr_env(task, context)
                )

    @T.inline
    def _emit_align_task(
        self,
        task,
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
            self._emit_pre_notify(task, context)
        if emit_wait:
            self._emit_waits(task, context)
        self._run_opaque_tile(task, context)
        T.cuda.cta_sync()
        if emit_init:
            self._emit_runtime_event_init(
                task, context, runtime_event_name, runtime_init, runtime_init_scope_id, tid
            )
        if emit_post:
            self._emit_notifies(task, context)

    def _emit_task(self, task: TaskPlan, context):
        if task.spec.name == "align":
            steps = task.execution_steps
            runtime_events = [
                event
                for event in self._require_plan().events
                if event.runtime_init is not None and event.runtime_init.task == task.spec.name
            ]
            if len(runtime_events) > 1:
                raise ValueError("a task may initialize at most one runtime event in the MoE MVP")
            runtime_event = runtime_events[0] if runtime_events else None
            self._emit_align_task(
                task,
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
        for step in task.execution_steps:
            if step == _STEP_PRE_NOTIFY:
                self._emit_pre_notify(task, context)
            elif step == _STEP_WAIT:
                self._emit_waits(task, context)
            elif step == _STEP_RUN:
                self._run_opaque_tile(task, context)
            elif step == _STEP_POST_NOTIFY:
                self._emit_notifies(task, context)
            else:
                raise ValueError(f"unsupported execution step {step!r} for task {task.spec.name!r}")

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
