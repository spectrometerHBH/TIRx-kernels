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

"""Concrete TileImpl adapters for distributed GEMM stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from tvm.megakernel.dsl import TileImpl

from .. import _allgather_gemm_impl as ag_impl
from .. import _gemm_reduce_scatter_impl as rs_impl

ExecutionSpace = Literal["device", "host"]


@dataclass(frozen=True)
class GemmCommTileTask:
    """One implementation stage held directly by a logical ``TileImpl``."""

    name: str
    execution_space: ExecutionSpace
    entrypoint: str
    physical_m_coalescing: int = 1
    module_factory: Callable[[str], Any] | None = None

    def lower_module(self, scheduler: str) -> Any:
        if self.module_factory is None:
            raise TypeError(f"host TileTask {self.name!r} does not own a device module")
        return self.module_factory(scheduler)


def _lower_allgather_gemm(scheduler: str) -> Any:
    return ag_impl.get_kernel(scheduler)


def _lower_reduce_scatter(scheduler: str) -> Any:
    if scheduler != "static":
        raise NotImplementedError("GEMM+ReduceScatter device lowering is static only")
    return rs_impl.ReduceScatter


class _GemmCommTileImpl(TileImpl):
    implementation: str
    job_type: int

    def __init__(self, tile_task: GemmCommTileTask):
        super().__init__()
        self.tile_task = tile_task
        self._runner: Callable[[Any, Any, Any], Any] | None = None

    @property
    def execution_space(self) -> ExecutionSpace:
        return self.tile_task.execution_space

    @property
    def entrypoint(self) -> str:
        return self.tile_task.entrypoint

    def bind_runner(self, runner: Callable[[Any, Any, Any], Any]) -> None:
        """Bind the stage emitter selected by the workload-specific lowerer."""

        self._runner = runner

    def run(self, m_idx, n_idx, k_idx):
        if self._runner is None:
            raise RuntimeError(
                f"{type(self).__name__} must be bound by GemmCommLowerer before run()"
            )
        return self._runner(m_idx, n_idx, k_idx)


class AllGatherTileImpl(_GemmCommTileImpl):
    implementation = "allgather"
    job_type = 0

    def __init__(self):
        super().__init__(
            GemmCommTileTask("allgather", "host", "runtime.disco.transfer_to_peers_all_gather")
        )


class AllGatherGemmTileImpl(_GemmCommTileImpl):
    implementation = "allgather_gemm"
    job_type = 1

    def __init__(self):
        super().__init__(
            GemmCommTileTask(
                "allgather_gemm",
                "device",
                "test_mma_ss_tma_2sm_persistent",
                module_factory=_lower_allgather_gemm,
            )
        )


class PartialGemmTileImpl(_GemmCommTileImpl):
    implementation = "partial_gemm"
    job_type = 2

    def __init__(self):
        super().__init__(
            GemmCommTileTask(
                "partial_gemm",
                "device",
                "test_mma_ss_tma_2sm_persistent",
                physical_m_coalescing=4,
                module_factory=_lower_reduce_scatter,
            )
        )


class ReduceScatterTileImpl(_GemmCommTileImpl):
    implementation = "reduce_scatter_transfer"
    job_type = 3

    def __init__(self):
        super().__init__(
            GemmCommTileTask(
                "reduce_scatter_transfer", "host", "runtime.disco.transfer_to_peers_reduce_scatter"
            )
        )


class ReduceSumTileImpl(_GemmCommTileImpl):
    implementation = "reduce_sum"
    job_type = 4

    def __init__(self):
        super().__init__(
            GemmCommTileTask(
                "reduce_sum", "device", "reduce_sum", module_factory=_lower_reduce_scatter
            )
        )


__all__ = [
    "AllGatherGemmTileImpl",
    "AllGatherTileImpl",
    "GemmCommTileTask",
    "PartialGemmTileImpl",
    "ReduceScatterTileImpl",
    "ReduceSumTileImpl",
]
