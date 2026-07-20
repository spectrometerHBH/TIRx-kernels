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

"""Logical tile implementations backed by the direct GemmComm builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from tvm.megakernel.dsl import TileImpl
from tvm.megakernel.transform import ExecutionPlan


class _LogicalTileImpl(TileImpl):
    """Describe one source-level tile without duplicating its fused TIRx body."""

    def __init__(self, tensor_specs: Mapping[str, Any], config: Any):
        super().__init__()
        self.tensor_specs = dict(tensor_specs)
        self.config = config

    def run(self, m_idx, n_idx, k_idx):
        del m_idx, n_idx, k_idx
        raise RuntimeError("GemmComm tiles are emitted by the implementation-preserving builder")


class _BuilderTileImpl(_LogicalTileImpl):
    def __init__(
        self,
        tensor_specs: Mapping[str, Any],
        config: Any,
        module_builder: Callable[[ExecutionPlan], Any] | None,
    ):
        super().__init__(tensor_specs, config)
        self._module_builder = module_builder

    def build_module(self, execution: ExecutionPlan):
        if self._module_builder is None:
            raise ValueError("this logical graph has no attached GemmComm module builder")
        return self._module_builder(execution)


class AllGatherTileImpl(_LogicalTileImpl):
    """Logical publication of one rank-local activation shard."""


class AllGatherGemmTileImpl(_BuilderTileImpl):
    """Logical AllGather+GEMM tile backed by the direct persistent kernel."""


class PartialGemmTileImpl(_BuilderTileImpl):
    """One logical partial-GEMM tile backed by the fused dynamic GemmRS kernel."""


class MultimemReduceScatterTileImpl(_LogicalTileImpl):
    """One local multimem ReduceScatter tile in the fused GemmRS kernel."""


__all__ = [
    "AllGatherGemmTileImpl",
    "AllGatherTileImpl",
    "MultimemReduceScatterTileImpl",
    "PartialGemmTileImpl",
]
