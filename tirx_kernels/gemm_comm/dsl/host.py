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

from tvm.megakernel.transform import (
    FetchGuardAction,
    HostCallAction,
    HostEdgeAction,
    MegakernelBackend,
    MidBodyPortAction,
    RunAction,
    TileEmitter,
)

from .model import GemmCommPlan


@dataclass(frozen=True)
class GemmCommRuntimeBindings:
    """Backend callbacks that bind abstract regions to Disco/NVSHMEM runtime calls."""

    launch_device: Callable[[str], None]
    launch_host: Callable[[str], None]
    communication_barrier: Callable[[], None]
    communication_to_compute_sync: Callable[[], None]


class _GemmCommHostBackend(MegakernelBackend):
    def __init__(self, bindings: GemmCommRuntimeBindings):
        self.bindings = bindings

    def begin_device_region(self, plan, region) -> None:
        del plan
        entrypoint = region.attrs.get("entrypoint")
        if not isinstance(entrypoint, str) or not entrypoint:
            raise ValueError(f"device region {region.name!r} has no backend entrypoint")
        self.bindings.launch_device(entrypoint)

    def emit_device_action(self, action, context) -> None:
        if isinstance(action, RunAction | FetchGuardAction | MidBodyPortAction):
            # These actions are realized inside their device kernel and do not
            # add a host-side synchronization point.
            return
        else:
            raise ValueError(f"unsupported GemmComm device action {type(action).__name__}")

    def emit_host_action(self, action, context) -> None:
        if isinstance(action, HostCallAction):
            self.bindings.launch_host(action.name)
        elif isinstance(action, HostEdgeAction):
            if action.kind != "completion":
                raise ValueError(f"unsupported host edge kind {action.kind!r}")
            self.bindings.communication_barrier()
            self.bindings.communication_to_compute_sync()
        else:
            raise ValueError(f"unsupported GemmComm host action {type(action).__name__}")


class GemmCommHostExecutor:
    """Execute one region DAG without conflating launch order and completion."""

    def __init__(self, bindings: GemmCommRuntimeBindings):
        self.bindings = bindings

    def execute(self, plan: GemmCommPlan) -> None:
        if not plan.lowerable:
            raise NotImplementedError(plan.unsupported_reason)
        TileEmitter(_GemmCommHostBackend(self.bindings)).emit(plan.execution_plan())


__all__ = ["GemmCommHostExecutor", "GemmCommRuntimeBindings"]
