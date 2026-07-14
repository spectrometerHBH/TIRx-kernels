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

"""Workload-specific lowering from logical graphs to existing TIRx ABIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tvm.megakernel.dsl import KernelSpec

from .. import _allgather_gemm_impl as ag_impl
from .model import GemmCommPlan
from .policies import GemmCommPolicy


@dataclass(frozen=True)
class LoweredGemmComm:
    """Physical module plus host/device entrypoints selected from one plan."""

    plan: GemmCommPlan
    module: Any | None
    device_entrypoints: tuple[str, ...]
    host_entrypoints: tuple[str, ...]


class GemmCommLowerer:
    """Lower a pure logical graph without teaching TVM about Disco or NVSHMEM."""

    def __init__(self, policy: GemmCommPolicy):
        self.policy = policy

    def lower(self, spec: KernelSpec, *, plan_only: bool = False) -> LoweredGemmComm:
        plan = self.policy.normalize(spec)
        if not plan.lowerable and not plan_only:
            raise NotImplementedError(plan.unsupported_reason)

        device_entrypoints = tuple(
            tile.impl.entrypoint for tile in spec.tiles if tile.impl.execution_space == "device"
        )
        host_entrypoints = tuple(
            tile.impl.entrypoint for tile in spec.tiles if tile.impl.execution_space == "host"
        )
        module = None
        if not plan_only:
            device_tasks = [
                tile.impl.tile_task
                for tile in spec.tiles
                if tile.impl.execution_space == "device"
                and tile.impl.tile_task.module_factory is not None
            ]
            if not device_tasks:
                raise ValueError("distributed GEMM graph has no device TileTask")
            modules = [task.lower_module(plan.policy_name) for task in device_tasks]
            module = modules[0]
            if any(candidate is not module for candidate in modules[1:]):
                raise ValueError("device TileTasks lowered to incompatible TIRx modules")
        return LoweredGemmComm(plan, module, device_entrypoints, host_entrypoints)


def make_allgather_dynamic_queue(
    plan: GemmCommPlan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize the existing queue ABI entirely from a normalized DSL plan."""

    if plan.workload != "allgather_gemm" or not plan.is_dynamic:
        raise ValueError("AllGather queue materialization requires its dynamic plan")
    task_types = np.full((plan.world_size, ag_impl.CAPACITY), -1, dtype=np.int32)
    task_indices = np.zeros(
        (plan.world_size, ag_impl.CAPACITY, ag_impl.TASK_IDX_LEN), dtype=np.int32
    )
    heads = np.zeros((plan.world_size, 1), dtype=np.int32)
    tails = np.zeros((plan.world_size, 1), dtype=np.int32)
    for schedule in plan.rank_schedules:
        tasks = schedule.shared_queue
        if len(tasks) > ag_impl.CAPACITY:
            raise ValueError("AllGather+GEMM dynamic queue exceeds its physical capacity")
        task_types[schedule.rank, : len(tasks)] = ag_impl.TaskType.GEMM.value
        task_indices[schedule.rank, : len(tasks)] = [(task.m, task.n) for task in tasks]
        tails[schedule.rank, 0] = len(tasks)
    return task_types, task_indices, heads, tails


__all__ = ["GemmCommLowerer", "LoweredGemmComm", "make_allgather_dynamic_queue"]
