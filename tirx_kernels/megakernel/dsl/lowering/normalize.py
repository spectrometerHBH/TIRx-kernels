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

"""Normalize and validate logical MoE graphs into physical plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import reduce
from itertools import product
from operator import mul

from tirx_kernels.megakernel.utils.base import SemaphoreBase
from tirx_kernels.megakernel.utils.config import JobType, KernelConfig
from tirx_kernels.megakernel.utils.utils import MAX_K_IDX, MAX_M_IDX, MAX_N_IDX, MAX_TASK_TYPE

from .._expr import (
    BinaryExpr,
    CeilDivExpr,
    ConstExpr,
    Expr,
    ScalarLoadExpr,
    TileIndexExpr,
    VarExpr,
    walk_expr,
)
from ..spec import TileSpec
from ._constants import _PACKED_INDEX_LIMITS, _SCOPE_INSTANCES, _SCOPE_ORDER, _SCOPE_WIDTHS
from .model import (
    DynamicDispatchPlan,
    EventPlan,
    HostTask,
    MoeLoweringEnv,
    NotifyPlan,
    RuntimeEventInitPlan,
    TilePlan,
    WaitPlan,
    _DynamicDispatchRule,
)


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
            event=dependency[0].name,
            coord=env.coord(tile, dependency),
            level=_WAIT_LEVEL_BY_TILE[tile.name],
        )
        for dependency in tile.waits
    )
    notifies = tuple(
        NotifyPlan(
            logical_spec=dependency,
            event=dependency[0].name,
            coord=env.coord(tile, dependency),
            scope=_NOTIFY_SCOPE_BY_TILE[tile.name][0],
            scope_id=_NOTIFY_SCOPE_BY_TILE[tile.name][1],
        )
        for dependency in tile.notifies
    )
    return waits, notifies


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
        if notify.logical_spec[0] is not wait.logical_spec[0] or notify.coord != rule.event_coord:
            raise ValueError(
                f"dynamic rule for {rule.source_tile!r} is inconsistent with its logical edge"
            )
