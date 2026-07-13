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
"""CPU-only validation tests for the MoE megakernel DSL."""

from dataclasses import replace

import numpy as np
import pytest

from tirx_kernels.megakernel.dsl import (
    ConstExpr,
    DynamicPolicy,
    EventSpec,
    MoeLowerer,
    NotifySpec,
    StaticPolicy,
    TaskDomain,
    TensorSpec,
    TileIndexExpr,
    VarExpr,
    WaitSpec,
    build_moe_graph,
    ceildiv,
    make_moe_plan,
)
from tirx_kernels.megakernel.utils.config import MEGAKERNEL_MOE_BENCH_CONFIG, JobType, KernelConfig
from tirx_kernels.megakernel.utils.support import generate_exec_queue_moe, push_moe_tasks
from tirx_kernels.megakernel.utils.utils import MAX_M_IDX
from tvm.tirx import PrimExpr
from tvm.tirx.expr import Var


def _max_rows(batch_size: int) -> int:
    routed = batch_size * 8
    if routed < 128:
        return routed
    return 128 + (routed - 128 + 127) // 128


def _replace_task(graph, name, **changes):
    tasks = tuple(replace(task, **changes) if task.name == name else task for task in graph.tasks)
    return replace(graph, tasks=tasks)


def _replace_event(graph, name, **changes):
    events = tuple(
        replace(event, **changes) if event.name == name else event for event in graph.events
    )
    return replace(graph, events=events)


def test_expression_host_and_tir_lowering():
    batch = VarExpr("B")
    expr = ceildiv(batch * 3 + 1, 4) * 2 - 1
    assert expr.evaluate({"B": 5}) == 7

    tir_batch = Var("batch", "int32")
    lowered = expr.lower({"vars": {"B": tir_batch}})
    assert isinstance(lowered, PrimExpr)

    scalar = TensorSpec("runtime", (1,), "int32", "intermediate").scalar(0)
    assert scalar.evaluate({"tensors": {"runtime": [384]}}) == 384
    assert scalar.lower({"tensors": {"runtime": [tir_batch]}}).same_as(tir_batch)

    tile = TileIndexExpr("gate_up_silu", 0)
    assert tile.evaluate({"tiles": {"gate_up_silu": (9, 2, 0)}}) == 9
    with pytest.raises(TypeError, match="callables"):
        batch + (lambda: 1)


def test_complete_six_stage_graph_validates():
    graph = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 512)
    assert graph.validate() is graph
    assert [task.name for task in graph.tasks] == [
        "gating",
        "topk",
        "align",
        "count_sort",
        "gate_up_silu",
        "down",
    ]
    assert [event.name for event in graph.events] == [
        "gating_done",
        "topk_done",
        "align_done",
        "count_sort_done",
        "gate_up_done",
        "down_dispatch_done",
    ]
    assert len(graph.dynamic_dispatch) == len(graph.tasks)
    assert graph.normalized_data()["tasks"][4]["implementation"] == "gate_up_silu"
    assert len(graph.normalized_data()["dynamic_dispatch"]) == 6


def test_graph_wait_notify_and_dispatch_scopes():
    graph = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    tasks = {task.name: task for task in graph.tasks}

    assert [(wait.event, wait.level) for wait in tasks["topk"].waits] == [("gating_done", "cta")]
    assert [(wait.event, wait.level) for wait in tasks["align"].waits] == [("topk_done", "cta")]
    assert [(wait.event, wait.level) for wait in tasks["count_sort"].waits] == [
        ("align_done", "cta")
    ]
    assert [(wait.event, wait.level) for wait in tasks["gate_up_silu"].waits] == [
        ("count_sort_done", "warp")
    ]
    assert [(wait.event, wait.level) for wait in tasks["down"].waits] == [("gate_up_done", "warp")]
    assert [
        (task.name, notify.event, notify.scope, notify.scope_id)
        for task in graph.tasks
        for notify in task.notifies
    ] == [
        ("gating", "gating_done", "warpgroup", 0),
        ("topk", "topk_done", "cta", 0),
        ("align", "align_done", "thread", 0),
        ("count_sort", "count_sort_done", "cta", 0),
        ("gate_up_silu", "gate_up_done", "warpgroup", 0),
    ]
    assert [
        (rule.source_task, rule.push_level, rule.pre_scope, rule.pre_scope_id)
        for rule in graph.dynamic_dispatch
    ] == [
        ("gating", "warpgroup", "warpgroup", 0),
        ("topk", "thread", "thread", 0),
        ("align", "cta", "cta", 0),
        ("count_sort", "cta", "cta", 0),
        ("gate_up_silu", "warp", "warp", 0),
        ("down", "warp", "warp", 0),
    ]


@pytest.mark.parametrize("batch_size", [1, 4, 128, 512, 2048, 4096])
def test_policy_event_layout_and_domains(batch_size):
    max_rows = _max_rows(batch_size)
    relaxed_rows = batch_size * 8 // 128 + 129
    static = make_moe_plan(MEGAKERNEL_MOE_BENCH_CONFIG, batch_size, "static")
    unfused = make_moe_plan(MEGAKERNEL_MOE_BENCH_CONFIG, batch_size, "unfused")
    dynamic = make_moe_plan(MEGAKERNEL_MOE_BENCH_CONFIG, batch_size, "dynamic")

    assert [event.name for event in static.events] == [
        "gating_done",
        "topk_done",
        "align_done",
        "count_sort_done",
        "gate_up_done",
        "down_dispatch_done",
        "event_init_complete",
    ]
    assert len(unfused.events) == 7
    assert len(dynamic.events) == 6
    assert static.event("gate_up_done").shape == (relaxed_rows,)
    assert static.event("gate_up_done").init_count == 12
    assert static.event("down_dispatch_done").init_count == max_rows * 16
    assert static.event("event_init_complete").init_count == 155
    assert unfused.event("gate_up_done").shape == (1,)
    assert unfused.event("gate_up_done").init_count == max_rows * 12
    assert dynamic.event("gate_up_done").shape == (relaxed_rows,)
    assert dynamic.event("down_dispatch_done").init_count is None
    assert dynamic.event("down_dispatch_done").runtime_init_task == "align"
    assert [event.workspace_offset for event in static.events] == [
        0,
        1,
        2,
        3,
        4,
        4 + relaxed_rows,
        5 + relaxed_rows,
    ]
    assert [event.workspace_offset for event in unfused.events] == [0, 1, 2, 3, 4, 5, 6]
    assert [event.workspace_offset for event in dynamic.events] == [0, 1, 2, 3, 4, 4 + relaxed_rows]

    assert static.task("gate_up_silu").upper_bounds == (max_rows, 12, 1)
    assert static.task("down").upper_bounds == (max_rows, 16, 1)
    expected_q = 1 if batch_size < 4 else 4
    assert dynamic.down_coalescing == expected_q
    assert dynamic.task("down").scheduled_upper_bounds == (max_rows, 16 // expected_q, 1)
    assert dynamic.persistent_ctas == 148
    assert dynamic.pre_before_wait
    assert dynamic.fifo_drain


@pytest.mark.parametrize("scheduler", ["static", "unfused"])
@pytest.mark.parametrize("batch_size", [1, 4, 512, 2048])
def test_static_host_queue_matches_manual_sequence(batch_size, scheduler):
    plan = make_moe_plan(MEGAKERNEL_MOE_BENCH_CONFIG, batch_size, scheduler)
    manual = [
        (event_idx, 0, 0, JobType.INIT_ETENSOR.value) for event_idx in range(len(plan.events))
    ]
    push_moe_tasks(manual, batch_size, MEGAKERNEL_MOE_BENCH_CONFIG, insert_wait_etensor_init=True)
    assert [task.as_manual_tuple() for task in plan.central_tasks] == manual

    queue = plan.make_static_queue()
    assert queue.shape == (KernelConfig.SM_NUMBER, 128)
    assert queue.dtype == np.int32


@pytest.mark.parametrize("batch_size", [1, 4, 512, 2048])
def test_dynamic_seed_and_push_mapping(batch_size):
    plan = make_moe_plan(MEGAKERNEL_MOE_BENCH_CONFIG, batch_size, "dynamic")
    init = plan.seed_tasks[:6]
    assert [task.job_type for task in init] == [JobType.INIT_ETENSOR.value] * 6
    assert [task.m_idx for task in init] == list(range(6))
    assert len(plan.seed_tasks) == 6 + ((batch_size + 127) // 128) * 4

    queue = plan.make_dynamic_queue()
    assert int(queue.tail[0]) == len(plan.seed_tasks)
    assert plan.queue_capacity == 32768
    assert plan.queue_upper_bound <= plan.queue_capacity
    manual_queue = generate_exec_queue_moe(
        batch_size, MEGAKERNEL_MOE_BENCH_CONFIG, len(plan.events), "dynamic"
    )
    np.testing.assert_array_equal(queue.tasks, manual_queue.tasks)
    np.testing.assert_array_equal(queue.head, manual_queue.head)
    np.testing.assert_array_equal(queue.tail, manual_queue.tail)

    graph = plan.spec
    count_rule = plan.dispatch("count_sort")
    runtime_env = {
        "tensors": {"num_tokens_post_pad": [7 * 128]},
        "vars": {**graph.compile_env, "push_idx": 25},
        "tiles": {"count_sort": (0, 0, 0)},
    }
    assert count_rule.count.evaluate(runtime_env) == 7 * 12
    assert tuple(index.evaluate(runtime_env) for index in count_rule.tile_indices) == (2, 1, 0)

    gate_rule = plan.dispatch("gate_up_silu")
    gate_env = {"vars": {**graph.compile_env, "push_idx": 3}, "tiles": {"gate_up_silu": (11, 5, 0)}}
    assert gate_rule.count.evaluate(gate_env) == 16 // plan.down_coalescing
    assert tuple(index.evaluate(gate_env) for index in gate_rule.tile_indices) == (11, 3, 0)
    assert plan.dispatch("down").target_task is None
    assert plan.dispatch("down").count.evaluate(graph.compile_env) == 148


def test_unfused_collapses_gate_up_coordinates():
    plan = make_moe_plan(MEGAKERNEL_MOE_BENCH_CONFIG, 512, "unfused")
    gate_notify = plan.task("gate_up_silu").spec.notifies[0]
    down_wait = plan.task("down").spec.waits[0]
    assert gate_notify.coord == (ConstExpr(0),)
    assert down_wait.coord == (ConstExpr(0),)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda graph: _replace_task(graph, "gating", domain=TaskDomain((1, 1), (1, 1))),
            "three-axis",
        ),
        (
            lambda graph: _replace_task(
                graph, "topk", waits=(WaitSpec("gating_done", (0, 0), "cta"),)
            ),
            "coordinate rank",
        ),
        (
            lambda graph: replace(
                _replace_task(graph, "topk", waits=(WaitSpec("orphan", (0,), "cta"),)),
                events=(*graph.events, EventSpec("orphan", (1,), 1)),
            ),
            "unique producer",
        ),
        (
            lambda graph: _replace_task(
                graph, "gating", waits=(WaitSpec("down_dispatch_done", (0,), "cta"),)
            ),
            "cycle",
        ),
        (
            lambda graph: _replace_event(
                graph,
                "gate_up_done",
                shape=(
                    next(t for t in graph.tensors if t.name == "num_tokens_post_pad").scalar(0),
                ),
            ),
            "runtime shape",
        ),
        (
            lambda graph: _replace_task(
                graph,
                "topk",
                waits=(WaitSpec("gating_done", (0,), "cta"), WaitSpec("align_done", (0,), "cta")),
            ),
            "fan-in",
        ),
        (
            lambda graph: _replace_task(
                graph,
                "topk",
                notifies=(
                    *next(task for task in graph.tasks if task.name == "topk").notifies,
                    NotifySpec("gating_done", (0,), "cta", 0),
                ),
            ),
            "multiple producers",
        ),
    ],
)
def test_graph_validator_rejects_invalid_graphs(mutate, message):
    graph = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    with pytest.raises(ValueError, match=message):
        mutate(graph).validate()


def test_validator_rejects_foreign_tile_index_and_callable():
    graph = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    foreign = _replace_task(
        graph,
        "down",
        waits=(WaitSpec("gate_up_done", (TileIndexExpr("gate_up_silu", 0),), "warp"),),
    )
    with pytest.raises(ValueError, match="owned by task"):
        foreign.validate()

    task = next(task for task in graph.tasks if task.name == "gating")
    callable_graph = _replace_task(graph, "gating", tile_binding=lambda: task.tile_binding)
    with pytest.raises(ValueError, match="callables"):
        callable_graph.validate()


def test_validator_rejects_invalid_runtime_scalar():
    graph = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    tensors = tuple(
        replace(tensor, dtype="float32") if tensor.name == "num_tokens_post_pad" else tensor
        for tensor in graph.tensors
    )
    with pytest.raises(ValueError, match="integer tensor"):
        replace(graph, tensors=tensors).validate()

    scalar = next(
        tensor for tensor in graph.tensors if tensor.name == "num_tokens_post_pad"
    ).scalar(0)
    gating = next(task for task in graph.tasks if task.name == "gating")
    before_producer = _replace_task(
        graph,
        "gating",
        domain=TaskDomain((scalar, 1, 4), gating.domain.upper_bounds),
        reads=(*gating.reads, "num_tokens_post_pad"),
    )
    with pytest.raises(ValueError, match="before producer completion"):
        before_producer.validate()


def test_validator_rejects_missing_dynamic_rule():
    graph = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    missing = replace(
        graph,
        dynamic_dispatch=tuple(
            rule for rule in graph.dynamic_dispatch if rule.source_task != "down"
        ),
    )
    with pytest.raises(ValueError, match="exactly one dynamic dispatch"):
        missing.validate()

    with pytest.raises(ValueError, match="exactly one dynamic dispatch"):
        replace(graph, dynamic_dispatch=()).validate()


def test_validator_rejects_runtime_upper_bound():
    graph = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    gate = next(task for task in graph.tasks if task.name == "gate_up_silu")
    runtime_bound = next(
        tensor for tensor in graph.tensors if tensor.name == "num_tokens_post_pad"
    ).scalar(0)
    invalid = _replace_task(
        graph, "gate_up_silu", domain=TaskDomain(gate.domain.extents, (runtime_bound, 12, 1))
    )
    with pytest.raises(ValueError, match="static upper bound"):
        invalid.validate()


def test_policy_rejects_coalescing_capacity_and_packed_overflow():
    graph = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 512)
    with pytest.raises(ValueError, match="coalescing"):
        graph.lower(MoeLowerer(DynamicPolicy(down_coalescing=1)))
    with pytest.raises(ValueError, match="exceeds capacity"):
        graph.lower(MoeLowerer(DynamicPolicy(queue_capacity=512)))
    with pytest.raises(ValueError, match="columns"):
        graph.lower(MoeLowerer(StaticPolicy(queue_capacity=1)))

    gate = next(task for task in graph.tasks if task.name == "gate_up_silu")
    overflow = _replace_task(
        graph, "gate_up_silu", domain=TaskDomain(gate.domain.extents, (MAX_M_IDX + 1, 12, 1))
    )
    with pytest.raises(ValueError, match="packed tile indices"):
        overflow.lower(MoeLowerer(StaticPolicy()))
