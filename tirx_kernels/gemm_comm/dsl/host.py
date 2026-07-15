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

"""TIRx-owned host execution for distributed GEMM region plans."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tvm.megakernel.transform import DeviceRegionPlan, HostCallStep, HostSyncStep

from .model import GemmCommPlan


@dataclass(frozen=True)
class GemmCommRuntimeBindings:
    """Backend callbacks that bind abstract regions to Disco/NVSHMEM runtime calls."""

    launch_device: Callable[[str], None]
    launch_host: Callable[[str], None]
    communication_barrier: Callable[[], None]
    communication_to_compute_sync: Callable[[], None]


class GemmCommHostExecutor:
    """Execute one region DAG without conflating launch order and completion."""

    def __init__(self, bindings: GemmCommRuntimeBindings):
        self.bindings = bindings

    def execute(self, plan: GemmCommPlan) -> None:
        plan.validate()
        if not plan.lowerable:
            raise NotImplementedError(plan.unsupported_reason)
        for region in plan.execution.regions_in_dependency_order():
            entrypoint = plan.entrypoint_for(region.name)
            if isinstance(region, DeviceRegionPlan):
                self.bindings.launch_device(entrypoint)
                continue
            for step in region.steps:
                if isinstance(step, HostCallStep):
                    if step.name != "collective":
                        raise ValueError(f"unsupported GemmComm host call {step.name!r}")
                    self.bindings.launch_host(entrypoint)
                elif isinstance(step, HostSyncStep):
                    if step.kind != "communication_completion":
                        raise ValueError(f"unsupported host sync kind {step.kind!r}")
                    self.bindings.communication_barrier()
                    self.bindings.communication_to_compute_sync()
                else:
                    raise ValueError(f"unsupported GemmComm host step {type(step).__name__}")


__all__ = ["GemmCommHostExecutor", "GemmCommRuntimeBindings"]
