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
    BarrierStep,
    NotifyStep,
    RunStep,
    RuntimeEventInitStep,
    WaitStep,
)
from tvm.script import tirx as T

from ..spec import KernelSpec
from ._constants import _EVENT_ATTRS
from .model import DynamicDispatchStep, EventPlan, MoeTileProgram, NormalizedPlan
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

    @property
    def execution(self):
        """Return the normalized plan's single physical execution."""

        return self._require_plan().execution

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
        for program in plan.programs:
            self.owner._add_tile(program.tile.impl, program.tile.impl.profile_event_type)

    def _bind_tensors(self, context):
        """Automatically bind constructor-held TensorSpecs to lowering buffers."""

        for program in self._require_plan().programs:
            impl = program.tile.impl
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

    def _expr_env(self, program: MoeTileProgram, context, *, push_idx=None):
        plan = self._require_plan()
        variables = dict(plan.env.compile_env)
        if push_idx is not None:
            variables["push_idx"] = push_idx
        scheduler = self.owner.tile_scheduler
        return {
            "vars": variables,
            "tensors": context,
            "tiles": {program.tile.name: (scheduler.m_idx, scheduler.n_idx, scheduler.k_idx)},
        }

    def _event(self, name: str):
        return getattr(self.owner, _EVENT_ATTRS[name])

    def _emit_pre_notify(self, program: MoeTileProgram, dispatch: DynamicDispatchStep, context):
        if dispatch.rule.source_tile != program.tile.name:
            raise ValueError("dynamic dispatch step must match its tile program")
        plan = self._require_plan()
        rule = dispatch.rule
        event = self._event(rule.event)
        notify_env = self._expr_env(program, context)

        def notify_fn(notify_idx):
            del notify_idx
            return (
                rule.pre_count.lower(notify_env),
                rule.rank,
                *(coord.lower(notify_env) for coord in rule.event_coord),
            )

        target_job = (
            JobType.END.value
            if rule.target_tile is None
            else plan.program(rule.target_tile).job_type
        )

        def trigger_fn(trigger_idx):
            del trigger_idx

            def push_fn(push_idx):
                push_env = self._expr_env(program, context, push_idx=push_idx)
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

    def _emit_wait(self, program: MoeTileProgram, wait: WaitStep, context):
        env = self._expr_env(program, context)
        coord = tuple(index.lower(env) for index in wait.coord_map)
        if wait.mask == 0xFFFFFFFF:
            self.owner.tile_scheduler.wait(
                self._event(wait.event.name), *coord, wait_level=wait.level
            )
        else:
            self.owner.tile_scheduler.wait(
                self._event(wait.event.name), *coord, wait_level=wait.level, mask=wait.mask
            )

    def _emit_notify(self, program: MoeTileProgram, notify: NotifyStep, context):
        env = self._expr_env(program, context)
        event = self._event(notify.event.name)

        def notify_fn(notify_idx):
            del notify_idx
            return (
                notify.count.lower(env),
                notify.rank,
                *(coord.lower(env) for coord in notify.coord_map),
            )

        self.owner.tile_scheduler.notify(
            event, notify_fn, scope=notify.scope, scope_id=notify.scope_id, release=notify.release
        )

    def _emit_run_step(self, step: RunStep, program: MoeTileProgram, context):
        scheduler = self.owner.tile_scheduler
        if isinstance(step.repeat, bool) or not isinstance(step.repeat, int):
            raise TypeError("MoE run repeat must be a positive integer")
        if step.repeat <= 0:
            raise ValueError("MoE run repeat must be positive")

        def run_once(m_idx, n_idx, k_idx):
            self._emit_run_once(program.tile.impl, step.profile_event, m_idx, n_idx, k_idx)

        def run():
            if step.index_map is None:
                if step.repeat != 1:
                    raise ValueError("a repeated MoE run requires an index map")
                run_once(scheduler.m_idx, scheduler.n_idx, scheduler.k_idx)
            else:
                if step.index_map != "expand_down_n":
                    raise ValueError(f"unsupported MoE run index map {step.index_map!r}")
                with T.serial(step.repeat) as index:
                    run_once(
                        scheduler.m_idx, scheduler.n_idx * step.repeat + index, scheduler.k_idx
                    )

        if step.predicate is None:
            run()
        elif step.predicate == "dynamic_or_routed_row":
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
            raise ValueError(f"unsupported MoE run predicate {step.predicate!r}")

    @T.inline
    def _emit_run_once(self, tile_impl, profile_event, m_idx, n_idx, k_idx):
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
    def _emit_runtime_event_init(self, program, context, step, tid):
        if step.scope != "thread":
            raise ValueError(f"unsupported MoE runtime init scope {step.scope!r}")
        if tid == step.scope_id:
            if step.event is not None:
                if not isinstance(step.event, EventPlan) or step.value is None:
                    raise ValueError("MoE runtime init step has an invalid event or value")
                self._event(step.event.name).sem[0] = step.value.lower(
                    self._expr_env(program, context)
                )

    @T.inline
    def _emit_cta_barrier(self):
        T.cuda.cta_sync()

    @T.inline
    def _emit_tile_program(self, program, context):
        if program.tile.name == "align":
            # Preserve the oracle's thread binding position before the first step.
            tid = T.thread_id([KernelConfig.NUM_THREADS])
            self._emit_program_steps(program, context, tid)
        else:
            self._emit_program_steps(program, context, 0)

    def _emit_program_steps(self, program, context, tid):
        if program.smem_scope == "program":
            self.owner.smem_manager.enter_tile_runtime(program.tile.impl)
        for step in program.steps:
            if isinstance(step, DynamicDispatchStep):
                self._emit_pre_notify(program, step, context)
            elif isinstance(step, WaitStep):
                self._emit_wait(program, step, context)
            elif isinstance(step, RunStep):
                if program.smem_scope == "run_to_end":
                    self.owner.smem_manager.enter_tile_runtime(program.tile.impl)
                self._emit_run_step(step, program, context)
            elif isinstance(step, BarrierStep):
                if step.kind != "cta":
                    raise ValueError(f"unsupported MoE barrier {step.kind!r}")
                self._emit_cta_barrier()
            elif isinstance(step, RuntimeEventInitStep):
                if program.tile.name != "align":
                    raise ValueError("runtime event initialization requires an align thread id")
                self._emit_runtime_event_init(program, context, step, tid)
            elif isinstance(step, NotifyStep):
                self._emit_notify(program, step, context)
            else:
                raise ValueError(f"unsupported MoE program step {type(step).__name__}")
        if program.smem_scope in ("program", "run_to_end"):
            self.owner.smem_manager.exit_tile_runtime()

    def dispatch_loop_body(self, context):
        """Emit the task-type dispatch chain from normalized tile bindings."""

        plan = self._require_plan()
        self._bind_tensors(context)
        entries: list[tuple[int, MoeTileProgram | str]] = [
            (program.job_type, program) for program in plan.programs
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
                if isinstance(entry, MoeTileProgram):
                    self._emit_tile_program(entry, context)
                elif entry == "init_event":
                    self.owner.task_impl_init_etensor(plan.is_dynamic)
                else:
                    self.owner.task_impl_wait_etensor_init_complete(plan.is_dynamic)
            else_frames[index].__enter__()
        T.evaluate(T.cuda.trap_when_assert_failed(False))
        for index in range(len(entries) - 1, -1, -1):
            else_frames[index].__exit__(None, None, None)
            if_frames[index].__exit__(None, None, None)
