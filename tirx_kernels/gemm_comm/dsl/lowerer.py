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

"""Lower logical graphs through the concrete TileImpls in each kernel module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tvm.megakernel.dsl import KernelSpec
from tvm.megakernel.transform import ExecutionPlan

from .. import allgather_gemm as ag_kernel
from .model import GemmCommPlan
from .policies import GemmCommPolicy


@dataclass(frozen=True)
class LoweredGemmComm:
    """Physical module plus host/device entrypoints selected from one plan."""

    plan: GemmCommPlan
    module: Any | None

    @property
    def execution(self) -> ExecutionPlan:
        return self.plan.execution

    @property
    def device_entrypoints(self) -> tuple[str, ...]:
        return tuple(
            self.plan.entrypoint_for(region.name) for region in self.execution.device_regions
        )

    @property
    def host_entrypoints(self) -> tuple[str, ...]:
        return tuple(
            self.plan.entrypoint_for(region.name) for region in self.execution.host_regions
        )


class GemmCommLowerer:
    """Lower a pure logical graph without teaching TVM about Disco or NVSHMEM."""

    def __init__(self, policy: GemmCommPolicy):
        self.policy = policy

    def lower(self, spec: KernelSpec, *, plan_only: bool = False) -> LoweredGemmComm:
        plan = self.policy.normalize(spec)
        if not plan.lowerable and not plan_only:
            raise NotImplementedError(plan.unsupported_reason)

        module = None
        if not plan_only:
            builders = [
                tile.impl.build_module
                for tile in plan.spec.tiles
                if callable(getattr(tile.impl, "build_module", None))
            ]
            if len(builders) != 1:
                raise ValueError("a distributed execution plan requires exactly one module backend")
            module = builders[0](plan.execution)
        return LoweredGemmComm(plan, module)


def make_allgather_dynamic_queue(
    plan: GemmCommPlan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize the existing queue ABI entirely from a normalized DSL plan."""

    if plan.workload != "allgather_gemm" or not plan.is_dynamic:
        raise ValueError("AllGather queue materialization requires its dynamic plan")
    task_types = np.full((plan.world_size, ag_kernel.CAPACITY), -1, dtype=np.int32)
    task_indices = np.zeros(
        (plan.world_size, ag_kernel.CAPACITY, ag_kernel.TASK_IDX_LEN), dtype=np.int32
    )
    heads = np.zeros((plan.world_size, 1), dtype=np.int32)
    tails = np.zeros((plan.world_size, 1), dtype=np.int32)
    for schedule in plan.rank_schedules:
        tasks = schedule.shared_queue
        if len(tasks) > ag_kernel.CAPACITY:
            raise ValueError("AllGather+GEMM dynamic queue exceeds its physical capacity")
        task_types[schedule.rank, : len(tasks)] = ag_kernel.TaskType.GEMM.value
        task_indices[schedule.rank, : len(tasks)] = [(task.m, task.n) for task in tasks]
        tails[schedule.rank, 0] = len(tasks)
    return task_types, task_indices, heads, tails


__all__ = ["GemmCommLowerer", "LoweredGemmComm", "make_allgather_dynamic_queue"]
