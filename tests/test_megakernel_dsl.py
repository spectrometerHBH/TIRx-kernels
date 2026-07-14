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
"""CPU-only validation tests for the TVM-native MoE megakernel DSL."""

import importlib.util
import inspect
from dataclasses import replace

import numpy as np
import pytest

import tirx_kernels.megakernel.dsl as megakernel_dsl
import tvm.megakernel.dsl as tvm_dsl
from tirx_kernels.megakernel.dsl import (
    AlignTileImpl,
    CountSortTileImpl,
    DownTileImpl,
    DynamicPolicy,
    GateUpSiluTileImpl,
    GatingTileImpl,
    MoeLowerer,
    MoeLoweringEnv,
    StaticPolicy,
    TopkTileImpl,
    VarSpec,
    build_moe_graph,
    make_moe_plan,
)
from tirx_kernels.megakernel.dsl._expr import ConstExpr, ScalarLoadExpr, walk_expr
from tirx_kernels.megakernel.utils.config import MEGAKERNEL_MOE_BENCH_CONFIG, JobType, KernelConfig
from tirx_kernels.megakernel.utils.support import generate_exec_queue_moe, push_moe_tasks
from tirx_kernels.megakernel.utils.utils import MAX_M_IDX

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


def _max_rows(batch_size: int) -> int:
    routed = batch_size * 8
    if routed < 128:
        return routed
    return 128 + (routed - 128 + 127) // 128


def _tile(spec, name):
    return next(tile for tile in spec.tiles if tile.name == name)


def _replace_plan_tile(plan, name, **changes):
    tiles = tuple(
        replace(tile, **changes) if tile.spec.name == name else tile for tile in plan.tiles
    )
    return replace(plan, tiles=tiles)


def _replace_plan_event(plan, name, **changes):
    events = tuple(
        replace(event, **changes) if event.name == name else event for event in plan.events
    )
    return replace(plan, events=events)


def _replace_dispatch(plan, source_tile, dispatch):
    dispatches = tuple(
        dispatch if item.rule.source_tile == source_tile else item for item in plan.dispatch_plans
    )
    return replace(plan, dispatch_plans=dispatches)


def _collect_attr_keys(value):
    keys = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(key)
            keys.update(_collect_attr_keys(item))
    elif isinstance(value, tuple | list):
        for item in value:
            keys.update(_collect_attr_keys(item))
    return keys


def test_public_spec_types_are_tvm_owned_and_legacy_model_is_removed():
    for name in (
        "VarSpec",
        "TensorSpec",
        "EventSpec",
        "DependencySpec",
        "TileSpec",
        "TileImpl",
        "KernelSpec",
    ):
        assert getattr(megakernel_dsl, name) is getattr(tvm_dsl, name)

    for removed in (
        "TaskSpec",
        "TaskDomain",
        "TileBinding",
        "WaitSpec",
        "NotifySpec",
        "DispatchSpec",
        "ConstExpr",
        "ScalarLoadExpr",
        "TileIndexExpr",
    ):
        assert not hasattr(megakernel_dsl, removed)
    assert importlib.util.find_spec("tirx_kernels.megakernel.dsl.expr") is None


def test_complete_six_stage_graph_is_pure_logical_native_dsl():
    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 512)
    assert isinstance(spec, tvm_dsl.KernelSpec)
    assert spec.validate() is spec
    assert [tile.name for tile in spec.tiles] == [
        "gating",
        "topk",
        "align",
        "count_sort",
        "gate_up_silu",
        "down",
    ]
    assert list(spec.events) == [
        "gating_done",
        "topk_done",
        "align_done",
        "count_sort_done",
        "gate_up_done",
    ]
    assert len(spec.events) == 5
    assert "down_dispatch_done" not in spec.events
    assert _tile(spec, "gate_up_silu").tile_num[0] == VarSpec("routed_rows")
    assert _tile(spec, "down").tile_num[0] == VarSpec("routed_rows")
    assert not hasattr(spec.tensors["hidden_state"], "role")

    attr_keys = _collect_attr_keys(spec.attrs)
    for event in spec.events.values():
        attr_keys.update(_collect_attr_keys(event.attrs))
    for tile in spec.tiles:
        attr_keys.update(_collect_attr_keys(tile.attrs))
    assert not attr_keys & _FORBIDDEN_SPEC_FIELDS


def test_graph_has_five_logical_edges_and_callable_coordinate_maps():
    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    producers = {
        id(dependency.event): tile.name for tile in spec.tiles for dependency in tile.notifies
    }
    edges = [
        (producers[id(dependency.event)], tile.name, dependency.event.name)
        for tile in spec.tiles
        for dependency in tile.waits
    ]
    assert edges == [
        ("gating", "topk", "gating_done"),
        ("topk", "align", "topk_done"),
        ("align", "count_sort", "align_done"),
        ("count_sort", "gate_up_silu", "count_sort_done"),
        ("gate_up_silu", "down", "gate_up_done"),
    ]
    for tile in spec.tiles:
        for dependency in (*tile.waits, *tile.notifies):
            assert callable(dependency.coord_map)
            coord = dependency.coord_map(2, 3, 4)
            assert isinstance(coord, tuple)
            assert len(coord) == len(dependency.event.shape)


def test_six_concrete_tile_impls_hold_existing_tasks_and_only_compute_in_run():
    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 4)
    expected = (
        GatingTileImpl,
        TopkTileImpl,
        AlignTileImpl,
        CountSortTileImpl,
        GateUpSiluTileImpl,
        DownTileImpl,
    )
    assert tuple(type(tile.impl) for tile in spec.tiles) == expected
    assert all(tile.impl.tile_task is not None for tile in spec.tiles)
    assert len({tile.impl.job_type for tile in spec.tiles}) == 6
    for tile in spec.tiles:
        source = inspect.getsource(type(tile.impl).run)
        assert all(word not in source for word in ("wait(", "notify(", "dispatch", "coalesc"))


def test_moe_lowering_env_resolves_runtime_rows_and_producer_order():
    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 512)
    env = MoeLoweringEnv(spec)
    assert env.batch_size == 512
    assert env.rmax == _max_rows(512)
    assert env.tensor_producers["num_tokens_post_pad"] == "align"
    assert env.routed_rows.evaluate({"tensors": {"num_tokens_post_pad": [7 * 128]}}) == 7
    assert env.upper_bound(VarSpec("routed_rows")) == _max_rows(512)


@pytest.mark.parametrize("batch_size", [1, 4, 128, 512, 2048, 4096])
def test_policy_event_layout_domains_and_execution_steps(batch_size):
    max_rows = _max_rows(batch_size)
    relaxed_rows = batch_size * 8 // 128 + 129
    expected_q = 1 if batch_size < 4 else 4
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
    assert [event.is_logical for event in static.events] == [True] * 5 + [False, False]
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
    assert dynamic.event("down_dispatch_done").runtime_init_tile == "align"
    runtime_init = dynamic.event("down_dispatch_done").runtime_init
    assert runtime_init is not None
    assert runtime_init.value.evaluate({"tensors": {"num_tokens_post_pad": [max_rows * 128]}}) == (
        2**16 + 1
    ) * max_rows * (16 // expected_q)
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

    assert static.tile("gate_up_silu").upper_bounds == (max_rows, 12, 1)
    assert static.tile("down").upper_bounds == (max_rows, 16, 1)
    assert any(
        isinstance(node, ScalarLoadExpr)
        for node in walk_expr(static.tile("gate_up_silu").runtime_extents[0])
    )
    assert dynamic.down_coalescing == expected_q
    assert dynamic.tile("down").scheduled_upper_bounds == (max_rows, 16 // expected_q, 1)
    assert dynamic.persistent_ctas == 148
    assert dynamic.pre_before_wait and dynamic.post_after_run and dynamic.fifo_drain
    assert dynamic.tile("align").execution_steps == (
        "pre_notify",
        "wait",
        "run",
        "cta_sync",
        "runtime_event_init",
        "post_notify",
    )
    assert dynamic.tile("down").execution_steps == ("pre_notify", "wait", "run")
    assert static.tile("align").execution_steps == (
        "wait",
        "run",
        "cta_sync",
        "runtime_event_init",
        "post_notify",
    )
    assert dynamic.protocol is not None
    assert (dynamic.protocol.pre_decrement, dynamic.protocol.post_decrement) == (1, 2**16)
    assert dynamic.protocol.scheduler_warp == 7
    normalized = dynamic.normalized_data()
    assert normalized["events"][-1]["runtime_init"]["value"] == runtime_init.value.to_data()
    assert normalized["tiles"][2]["execution_steps"] == dynamic.tile("align").execution_steps
    assert normalized["dispatch"][0]["enqueue_upper_bound"] == 148


def test_physical_wait_notify_scopes_and_dynamic_rules_are_policy_owned():
    plan = make_moe_plan(MEGAKERNEL_MOE_BENCH_CONFIG, 128, "dynamic")
    assert [
        (tile.spec.name, wait.event, wait.level) for tile in plan.tiles for wait in tile.waits
    ] == [
        ("topk", "gating_done", "cta"),
        ("align", "topk_done", "cta"),
        ("count_sort", "align_done", "cta"),
        ("gate_up_silu", "count_sort_done", "warp"),
        ("down", "gate_up_done", "warp"),
    ]
    assert [
        (tile.spec.name, notify.event, notify.scope, notify.scope_id)
        for tile in plan.tiles
        for notify in tile.notifies
    ] == [
        ("gating", "gating_done", "warpgroup", 0),
        ("topk", "topk_done", "cta", 0),
        ("align", "align_done", "thread", 0),
        ("count_sort", "count_sort_done", "cta", 0),
        ("gate_up_silu", "gate_up_done", "warpgroup", 0),
    ]
    assert [
        (rule.source_tile, rule.target_tile, rule.event, rule.push_level, rule.pre_scope)
        for rule in plan.dispatch_rules
    ] == [
        ("gating", "topk", "gating_done", "warpgroup", "warpgroup"),
        ("topk", "align", "topk_done", "thread", "thread"),
        ("align", "count_sort", "align_done", "cta", "cta"),
        ("count_sort", "gate_up_silu", "count_sort_done", "cta", "cta"),
        ("gate_up_silu", "down", "gate_up_done", "warp", "warp"),
        ("down", None, "down_dispatch_done", "warp", "warp"),
    ]


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
def test_dynamic_seed_queue_and_dispatch_mapping(batch_size):
    plan = make_moe_plan(MEGAKERNEL_MOE_BENCH_CONFIG, batch_size, "dynamic")
    init = plan.seed_tasks[:6]
    assert [task.job_type for task in init] == [JobType.INIT_ETENSOR.value] * 6
    assert [task.m_idx for task in init] == list(range(6))
    assert len(plan.seed_tasks) == 6 + ((batch_size + 127) // 128) * 4

    queue = plan.make_dynamic_queue()
    assert int(queue.tail[0]) == len(plan.seed_tasks)
    assert plan.queue_capacity == 32768
    assert plan.queue_upper_bound == len(plan.seed_tasks) + sum(
        dispatch.enqueue_upper_bound for dispatch in plan.dispatch_plans
    )
    manual_queue = generate_exec_queue_moe(
        batch_size, MEGAKERNEL_MOE_BENCH_CONFIG, len(plan.events), "dynamic"
    )
    np.testing.assert_array_equal(queue.tasks, manual_queue.tasks)
    np.testing.assert_array_equal(queue.head, manual_queue.head)
    np.testing.assert_array_equal(queue.tail, manual_queue.tail)

    count_rule = plan.dispatch("count_sort")
    runtime_env = {
        "tensors": {"num_tokens_post_pad": [7 * 128]},
        "vars": {**plan.env.compile_env, "push_idx": 25},
        "tiles": {"count_sort": (0, 0, 0)},
    }
    assert count_rule.count.evaluate(runtime_env) == 7 * 12
    assert tuple(index.evaluate(runtime_env) for index in count_rule.tile_indices) == (2, 1, 0)

    gate_rule = plan.dispatch("gate_up_silu")
    gate_env = {
        "vars": {**plan.env.compile_env, "push_idx": 3},
        "tiles": {"gate_up_silu": (11, 5, 0)},
    }
    assert gate_rule.count.evaluate(gate_env) == 16 // plan.down_coalescing
    assert tuple(index.evaluate(gate_env) for index in gate_rule.tile_indices) == (11, 3, 0)
    assert plan.dispatch("down").target_tile is None
    assert plan.dispatch("down").count.evaluate(plan.env.compile_env) == 148


def test_unfused_collapses_gate_up_coordinates_only_in_physical_plan():
    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 512)
    assert _tile(spec, "gate_up_silu").notifies[0].coord_map(7, 0, 0) == (7,)
    assert _tile(spec, "down").waits[0].coord_map(7, 0, 0) == (7,)
    plan = MoeLowerer(megakernel_dsl.UnfusedPolicy()).lower(spec)
    assert plan.tile("gate_up_silu").notifies[0].coord == (ConstExpr(0),)
    assert plan.tile("down").waits[0].coord == (ConstExpr(0),)


def test_native_and_moe_validators_reject_invalid_graphs():
    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    _tile(spec, "gating").tile_num = (1, 1)
    with pytest.raises(ValueError, match="three axes"):
        spec.validate()

    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    _tile(spec, "topk").waits[0] = tvm_dsl.DependencySpec(spec.events["gating_done"], (0, 0))
    with pytest.raises(ValueError, match="rank"):
        spec.validate()

    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    _tile(spec, "gating").notifies.clear()
    with pytest.raises(ValueError, match="no notifier"):
        spec.validate()

    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    _tile(spec, "gating").wait(spec.events["gate_up_done"], lambda m, n, k: (m,))
    with pytest.raises(ValueError, match="acyclic"):
        spec.validate()

    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    spec.attrs["scheduler"] = {"scope": "cta"}
    with pytest.raises(ValueError, match="scheduler field"):
        MoeLoweringEnv(spec)


def test_validator_rejects_impure_callable_foreign_tensor_and_non_impl():
    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    state = {"value": 0}

    def impure(m, n, k):
        state["value"] += 1
        return (state["value"],)

    _tile(spec, "topk").waits[0] = tvm_dsl.DependencySpec(spec.events["gating_done"], impure)
    with pytest.raises(ValueError, match="pure deterministic"):
        spec.validate()

    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    foreign = tvm_dsl.TensorSpec("hidden_state", (128, 2048), "float16")
    _tile(spec, "gating").reads[0] = foreign
    with pytest.raises(ValueError, match="outside this kernel"):
        spec.validate()

    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    _tile(spec, "gating").impl = object()
    with pytest.raises(TypeError, match="TileImpl"):
        spec.validate()


def test_moe_env_rejects_invalid_runtime_binding_and_upper_bound():
    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    object.__setattr__(spec.tensors["num_tokens_post_pad"], "dtype", "float32")
    with pytest.raises(ValueError, match="integer tensor"):
        MoeLoweringEnv(spec)

    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    runtime_tensor = spec.tensors["num_tokens_post_pad"]
    _tile(spec, "align").writes.remove(runtime_tensor)
    with pytest.raises(ValueError, match="produced by the align"):
        MoeLoweringEnv(spec)

    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    _tile(spec, "gate_up_silu").tile_num = (VarSpec("unknown"), 12, 1)
    with pytest.raises(ValueError, match="routed_rows"):
        MoeLoweringEnv(spec)

    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    _tile(spec, "gating").tile_num = (MAX_M_IDX + 1, 1, 4)
    with pytest.raises(ValueError, match="packed tile indices"):
        MoeLowerer(StaticPolicy()).lower(spec)


def test_policy_rejects_coalescing_and_queue_capacity_regressions():
    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 512)
    with pytest.raises(ValueError, match="coalescing"):
        MoeLowerer(DynamicPolicy(down_coalescing=1)).lower(spec)
    with pytest.raises(ValueError, match="exceeds capacity"):
        MoeLowerer(DynamicPolicy(queue_capacity=512)).lower(spec)
    with pytest.raises(ValueError, match="must remain 32768"):
        MoeLowerer(DynamicPolicy(queue_capacity=65536)).lower(spec)
    with pytest.raises(ValueError, match="columns"):
        MoeLowerer(StaticPolicy(queue_capacity=1)).lower(spec)


def test_normalized_plan_rejects_execution_runtime_init_and_terminal_drain_regressions():
    plan = make_moe_plan(MEGAKERNEL_MOE_BENCH_CONFIG, 512, "dynamic")
    align = plan.tile("align")
    invalid = _replace_plan_tile(
        plan,
        "align",
        execution_steps=(
            "wait",
            "pre_notify",
            "run",
            "cta_sync",
            "runtime_event_init",
            "post_notify",
        ),
    )
    with pytest.raises(ValueError, match="pre-notification"):
        invalid.validate()

    invalid = _replace_plan_tile(
        plan,
        "align",
        execution_steps=(
            "pre_notify",
            "run",
            "wait",
            "cta_sync",
            "runtime_event_init",
            "post_notify",
        ),
    )
    with pytest.raises(ValueError, match="wait before"):
        invalid.validate()

    with pytest.raises(ValueError, match="derived from dispatch"):
        replace(plan, queue_upper_bound=plan.queue_upper_bound + 1).validate()

    down_event = plan.event("down_dispatch_done")
    invalid_runtime = replace(down_event.runtime_init, value=ConstExpr(1))
    invalid = _replace_plan_event(plan, "down_dispatch_done", runtime_init=invalid_runtime)
    with pytest.raises(ValueError, match="invalid runtime initialization"):
        invalid.validate()

    terminal = plan.dispatch_plan("down")
    invalid_terminal = replace(
        terminal,
        rule=replace(terminal.rule, count=ConstExpr(147)),
        count_lower_bound=147,
        count_upper_bound=147,
        enqueue_upper_bound=147,
    )
    invalid = _replace_dispatch(plan, "down", invalid_terminal)
    invalid = replace(invalid, queue_upper_bound=plan.queue_upper_bound - 1)
    with pytest.raises(ValueError, match="FIFO terminal drain"):
        invalid.validate()
    assert align.spec is _tile(plan.spec, "align")


def test_plan_rejects_scope_mask_notification_and_dispatch_coverage_regressions():
    plan = make_moe_plan(MEGAKERNEL_MOE_BENCH_CONFIG, 128, "dynamic")
    align = plan.tile("align")
    oversized = replace(align.notifies[0], count=ConstExpr(2))
    invalid = _replace_plan_tile(plan, "align", notifies=(oversized,))
    with pytest.raises(ValueError, match="exceeds its thread scope"):
        invalid.validate()

    gating = plan.tile("gating")
    all_warpgroups = replace(gating.notifies[0], scope_id=-1)
    invalid = _replace_plan_tile(plan, "gating", notifies=(all_warpgroups,))
    with pytest.raises(ValueError, match="notifications per coordinate"):
        invalid.validate()

    cross_rank = replace(gating.notifies[0], rank=0)
    invalid = _replace_plan_tile(plan, "gating", notifies=(cross_rank,))
    with pytest.raises(ValueError, match="cross-rank"):
        invalid.validate()

    topk = plan.tile("topk")
    invalid_wait = replace(topk.waits[0], mask=-1)
    invalid = _replace_plan_tile(plan, "topk", waits=(invalid_wait,))
    with pytest.raises(ValueError, match="invalid wait mask"):
        invalid.validate()

    gating_dispatch = plan.dispatch_plan("gating")
    duplicate_rule = replace(
        gating_dispatch.rule, tile_indices=(ConstExpr(0), ConstExpr(0), ConstExpr(0))
    )
    duplicate_dispatch = replace(
        gating_dispatch, rule=duplicate_rule, tile_index_bounds=((0, 0), (0, 0), (0, 0))
    )
    invalid = _replace_dispatch(plan, "gating", duplicate_dispatch)
    with pytest.raises(ValueError, match="cover target tile"):
        invalid.validate()


def test_plan_rejects_invalid_event_coordinate_packed_index_and_protocol_link():
    plan = make_moe_plan(MEGAKERNEL_MOE_BENCH_CONFIG, 512, "dynamic")
    gate = plan.dispatch_plan("gate_up_silu")
    invalid_coord = replace(gate.rule, event_coord=(ConstExpr(10_000),))
    invalid = _replace_dispatch(plan, "gate_up_silu", replace(gate, rule=invalid_coord))
    with pytest.raises(ValueError, match="event coordinate"):
        invalid.validate()

    terminal = plan.dispatch_plan("down")
    overflow_rule = replace(
        terminal.rule, tile_indices=(ConstExpr(MAX_M_IDX), ConstExpr(0), ConstExpr(0))
    )
    invalid = _replace_dispatch(plan, "down", replace(terminal, rule=overflow_rule))
    with pytest.raises(ValueError, match="packed tile indices"):
        invalid.validate()

    mismatched_rule = replace(gate.rule, event_coord=(ConstExpr(0),))
    mismatched = replace(
        gate,
        rule=mismatched_rule,
        trigger_upper_bound=1,
        event_coord_bounds=((0, 0),),
        enqueue_upper_bound=gate.count_upper_bound,
    )
    invalid = _replace_dispatch(plan, "gate_up_silu", mismatched)
    invalid = replace(
        invalid,
        queue_upper_bound=plan.queue_upper_bound
        - gate.enqueue_upper_bound
        + mismatched.enqueue_upper_bound,
    )
    with pytest.raises(ValueError, match="pre/post notifications are inconsistent"):
        invalid.validate()


def test_dynamic_plan_preserves_release_notifications():
    plan = make_moe_plan(MEGAKERNEL_MOE_BENCH_CONFIG, 128, "dynamic")
    gating = plan.tile("gating")
    released = replace(gating.notifies[0], release=True)
    updated = _replace_plan_tile(plan, "gating", notifies=(released,))
    assert updated.validate().tile("gating").notifies[0].release
