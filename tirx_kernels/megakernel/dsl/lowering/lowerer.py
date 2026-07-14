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

"""TIRX emission adapter for normalized MoE plans."""

from __future__ import annotations

from tirx_kernels.megakernel.utils.config import JobType, KernelConfig
from tirx_kernels.megakernel.utils.utils import f_init_const
from tvm.script import tirx as T

from ..spec import KernelSpec
from ._constants import (
    _EVENT_ATTRS,
    _STEP_POST_NOTIFY,
    _STEP_PRE_NOTIFY,
    _STEP_RUN,
    _STEP_RUNTIME_EVENT_INIT,
    _STEP_WAIT,
)
from .model import NormalizedPlan, TilePlan
from .policies import MoePolicy


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
