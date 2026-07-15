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
from tvm.megakernel.transform import (
    BarrierAction,
    EmissionContext,
    ExecutionPlan,
    MegakernelBackend,
    NotifyAction,
    ProfileAction,
    QueuePushAction,
    RunAction,
    RuntimeEventInitAction,
    SmemEnterAction,
    SmemExitAction,
    TileEmitter,
    WaitAction,
)
from tvm.script import tirx as T

from ..spec import KernelSpec
from ._constants import _EVENT_ATTRS
from .model import NormalizedPlan, NotifyPlan, TilePlan, WaitPlan
from .policies import MoePolicy


class _MoeActionBackend(MegakernelBackend):
    """Emit one MoE tile program inside the existing scheduler dispatch scope."""

    def __init__(self, lowerer: MoeLowerer, buffers):
        self.lowerer = lowerer
        self.buffers = buffers
        self.tid = None
        self.pending_smem_enter = None
        self.active_smem_tile = None
        self.pending_profile_begin = None
        self.pending_run = None

    def begin_action_program(self, context: EmissionContext) -> None:
        del context

    def set_tid(self, tid) -> None:
        self.tid = tid

    def emit_action(self, action, context: EmissionContext) -> None:
        if context.tile is None:
            raise ValueError("MoE actions must be emitted inside a tile program")
        tile = self.lowerer._require_plan().tile(context.tile.name)
        if self.pending_run is not None and not (
            isinstance(action, ProfileAction) and action.phase == "end"
        ):
            self._flush_run(tile, None)
        if isinstance(action, QueuePushAction):
            self.lowerer._emit_pre_notify(tile, action.payload, self.buffers)
        elif isinstance(action, WaitAction):
            self.lowerer._emit_wait(tile, action.payload, self.buffers)
        elif isinstance(action, SmemEnterAction):
            if action.tile is not tile.spec:
                raise ValueError("shared-memory enter action references a different MoE tile")
            if self.pending_smem_enter is not None or self.active_smem_tile is not None:
                raise ValueError("duplicate MoE shared-memory enter action")
            self.pending_smem_enter = action
        elif isinstance(action, ProfileAction):
            if action.phase == "begin":
                if self.pending_profile_begin is not None:
                    raise ValueError("nested MoE profiling actions are unsupported")
                self.pending_profile_begin = action
            elif action.phase == "end":
                if self.pending_profile_begin is None:
                    raise ValueError("profile end requires a preceding profile begin action")
                if self.pending_run is None:
                    raise ValueError("profile end requires a preceding run action")
                self._flush_run(tile, action)
            else:
                raise ValueError(f"unsupported MoE profile phase {action.phase!r}")
        elif isinstance(action, RunAction):
            if self.pending_run is not None:
                raise ValueError("duplicate MoE run action")
            self.pending_run = action
        elif isinstance(action, BarrierAction):
            if action.kind != "cta":
                raise ValueError(f"unsupported MoE barrier {action.kind!r}")
            self.lowerer._emit_cta_barrier()
        elif isinstance(action, RuntimeEventInitAction):
            if self.tid is None:
                raise ValueError("runtime event initialization requires an align thread id")
            runtime_init = action.payload
            scope_id = action.predicate[1] if runtime_init is None else runtime_init.scope_id
            self.lowerer._emit_runtime_event_init(
                tile,
                self.buffers,
                None if action.event is None else action.event.name,
                runtime_init,
                scope_id,
                self.tid,
            )
        elif isinstance(action, NotifyAction):
            self.lowerer._emit_notify(tile, action.payload, self.buffers)
        elif isinstance(action, SmemExitAction):
            if self.active_smem_tile is None:
                raise ValueError("shared-memory exit action has no matching enter")
            if action.tile is not self.active_smem_tile or action.tile is not tile.spec:
                raise ValueError("shared-memory exit action references a different MoE tile")
            self.lowerer.owner.smem_manager.exit_tile_runtime()
            self.active_smem_tile = None
        else:
            raise ValueError(f"unsupported MoE action {type(action).__name__}")

    def end_action_program(self, context: EmissionContext) -> None:
        if self.pending_run is not None:
            tile = self.lowerer._require_plan().tile(context.tile.name)
            self._flush_run(tile, None)
        if self.pending_profile_begin is not None:
            raise ValueError("MoE action program ended with an open profiling scope")
        if self.pending_smem_enter is not None:
            raise ValueError("MoE action program ended with an open shared-memory scope")
        if self.active_smem_tile is not None:
            raise ValueError("MoE action program ended with an open shared-memory scope")
        self.tid = None

    def _flush_run(self, tile: TilePlan, profile_end: ProfileAction | None) -> None:
        if self.pending_run is None:
            return
        smem_enter = self.pending_smem_enter
        self.lowerer._emit_run_action(
            self.pending_run,
            tile,
            self.buffers,
            smem_enter=smem_enter,
            profile_begin=self.pending_profile_begin,
            profile_end=profile_end,
        )
        if smem_enter is not None:
            self.active_smem_tile = smem_enter.tile
        self.pending_smem_enter = None
        self.pending_profile_begin = None
        self.pending_run = None


class MoeLowerer:
    """Lower a normalized graph into event setup, dispatch, and opaque tile calls."""

    def __init__(self, policy: MoePolicy, owner=None):
        self.policy = policy
        self.owner = owner
        self.plan: NormalizedPlan | None = None
        self.execution: ExecutionPlan | None = None

    def lower(self, spec: KernelSpec) -> NormalizedPlan:
        self.plan = self.policy.normalize(spec)
        self.execution = self.plan.execution_plan()
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
            self.owner._add_tile(tile.spec.impl, tile.spec.impl.profile_event_type)

    def _bind_tensors(self, context):
        """Automatically bind constructor-held TensorSpecs to lowering buffers."""

        for tile in self._require_plan().tiles:
            impl = tile.spec.impl
            for attribute, binding in impl.tensor_bindings.items():
                if isinstance(binding, tuple):
                    tensor, flatten = binding
                else:
                    tensor, flatten = binding, False
                key = f"{tensor.name}_flat" if flatten else tensor.name
                setattr(impl, attribute, context[key])
            # Keep the profiler as an opaque parser resource.  Exposing it as
            # an implementation attribute lets script naming derive stable
            # names from that Python access path, which changes CUDA despite
            # structurally identical TIR.
            impl.profiler = T.meta_var(self.owner.profiler)

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

    def _emit_pre_notify(self, tile: TilePlan, dispatch, context):
        if dispatch is None or dispatch.rule.source_tile != tile.spec.name:
            raise ValueError("queue-push action must carry its tile's dispatch plan")
        plan = self._require_plan()
        rule = dispatch.rule
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

    def _emit_wait(self, tile: TilePlan, wait: WaitPlan, context):
        env = self._expr_env(tile, context)
        coord = tuple(index.lower(env) for index in wait.coord)
        if wait.mask == 0xFFFFFFFF:
            self.owner.tile_scheduler.wait(self._event(wait.event), *coord, wait_level=wait.level)
        else:
            self.owner.tile_scheduler.wait(
                self._event(wait.event), *coord, wait_level=wait.level, mask=wait.mask
            )

    def _emit_notify(self, tile: TilePlan, notify: NotifyPlan, context):
        env = self._expr_env(tile, context)
        event = self._event(notify.event)

        def notify_fn(notify_idx):
            del notify_idx
            return (
                notify.count.lower(env),
                notify.rank,
                *(coord.lower(env) for coord in notify.coord),
            )

        self.owner.tile_scheduler.notify(
            event, notify_fn, scope=notify.scope, scope_id=notify.scope_id, release=notify.release
        )

    def _emit_run_action(
        self,
        action: RunAction,
        tile: TilePlan,
        context,
        *,
        smem_enter: SmemEnterAction | None,
        profile_begin: ProfileAction | None,
        profile_end: ProfileAction | None,
    ):
        scheduler = self.owner.tile_scheduler
        if action.tile is not tile.spec:
            raise ValueError("run action references a different MoE tile")
        if isinstance(action.repeat, bool) or not isinstance(action.repeat, int):
            raise TypeError("MoE run repeat must be a positive integer")
        if action.repeat <= 0:
            raise ValueError("MoE run repeat must be positive")

        if (profile_begin is None) != (profile_end is None):
            raise ValueError("MoE run requires matched profile begin/end actions")
        if profile_begin is not None and profile_begin.event != profile_end.event:
            raise ValueError("MoE profile begin/end actions must use the same event")
        if smem_enter is not None and smem_enter.tile is not tile.spec:
            raise ValueError("shared-memory enter action references a different MoE tile")

        def run_once(m_idx, n_idx, k_idx):
            self._emit_run_once(
                action.tile.impl,
                None if smem_enter is None else smem_enter.tile.impl,
                None if profile_begin is None else profile_begin.event,
                m_idx,
                n_idx,
                k_idx,
            )

        def run():
            if action.index_map is None:
                if action.repeat != 1:
                    raise ValueError("a repeated MoE run requires an index map")
                run_once(scheduler.m_idx, scheduler.n_idx, scheduler.k_idx)
            else:
                if action.index_map != "expand_down_n":
                    raise ValueError(f"unsupported MoE run index map {action.index_map!r}")
                with T.serial(action.repeat) as index:
                    run_once(
                        scheduler.m_idx, scheduler.n_idx * action.repeat + index, scheduler.k_idx
                    )

        if action.predicate is None:
            run()
        elif action.predicate == "dynamic_or_routed_row":
            plan = self._require_plan()
            if_frame = T.If(
                T.Or(
                    T.bool(plan.is_dynamic),
                    scheduler.m_idx < context["num_tokens_post_pad"][0] // 128,
                )
            )
            if_frame.__enter__()
            with T.Then():
                run()
            if_frame.__exit__(None, None, None)
        else:
            raise ValueError(f"unsupported MoE run predicate {action.predicate!r}")

    @T.inline
    def _emit_run_once(self, tile_impl, smem_tile_impl, profile_event, m_idx, n_idx, k_idx):
        if smem_tile_impl is not None:
            self.owner.smem_manager.enter_tile_runtime(smem_tile_impl)
        if profile_event is not None:
            lane_id = T.lane_id([32])
            if self.owner.profiler_on:
                self.owner.profiler.start(profile_event, lane_id == 0)
            tile_impl.run(m_idx, n_idx, k_idx)
            if self.owner.profiler_on:
                self.owner.profiler.end(profile_event, lane_id == 0)
        else:
            tile_impl.run(m_idx, n_idx, k_idx)

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
    def _emit_cta_barrier(self):
        T.cuda.cta_sync()

    @T.inline
    def _emit_action_program(self, emitter, emission_context, actions):
        if emission_context.tile.name == "align":
            # Preserve the oracle's thread binding position before the first action.
            tid = T.thread_id([KernelConfig.NUM_THREADS])
            emitter.backend.set_tid(tid)
        emitter.emit_program(emission_context, actions)

    def dispatch_loop_body(self, context):
        """Emit the task-type dispatch chain from normalized tile bindings."""

        plan = self._require_plan()
        if self.execution is None:
            raise RuntimeError("MoE execution plan is unavailable")
        self._bind_tensors(context)
        entries: list[tuple[int, TilePlan | str]] = [(tile.job_type, tile) for tile in plan.tiles]
        entries.extend(
            [
                (JobType.INIT_ETENSOR.value, "init_event"),
                (JobType.WAIT_ETENSOR_INIT.value, "wait_event_init"),
            ]
        )
        task_type = self.owner.tile_scheduler.task_type
        region = self.execution.device_regions[0]
        programs = {program.tile.name: program for program in region.tile_programs}
        emitter = TileEmitter(_MoeActionBackend(self, context))
        if_frames = [T.If(task_type == job_type) for job_type, _ in entries]
        then_frames = [T.Then() for _ in entries]
        else_frames = [T.Else() for _ in entries]
        for index, (_, entry) in enumerate(entries):
            if_frames[index].__enter__()
            with then_frames[index]:
                if isinstance(entry, TilePlan):
                    program = programs[entry.spec.name]
                    self._emit_action_program(
                        emitter,
                        EmissionContext(self.execution, region, "tile_action", program.tile),
                        program.actions,
                    )
                elif entry == "init_event":
                    self.owner.task_impl_init_etensor(plan.is_dynamic)
                else:
                    self.owner.task_impl_wait_etensor_init_complete(plan.is_dynamic)
            else_frames[index].__enter__()
        T.evaluate(T.cuda.trap_when_assert_failed(False))
        for index in range(len(entries) - 1, -1, -1):
            else_frames[index].__exit__(None, None, None)
            if_frames[index].__exit__(None, None, None)
