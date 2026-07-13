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
"""IR equivalence gates for the MoE DSL and manual fallback."""

import re

import pytest

import tvm
from tirx_kernels.megakernel.moe import MegaKernelMOE
from tirx_kernels.megakernel.utils.config import MEGAKERNEL_MOE_BENCH_CONFIG
from tirx_kernels.megakernel.utils.utils import get_source


def _build_module(batch_size: int, scheduler: str, lowering: str):
    kernel = MegaKernelMOE(config=MEGAKERNEL_MOE_BENCH_CONFIG, world_size=1, profiler_on=False)
    kernel._compile_batch_size = batch_size
    return kernel.get_module(scheduler, lowering=lowering)


def _normalize_cuda_source(source: str) -> str:
    # TIRX may choose either an ``i_N`` or ``v_N`` spelling for an anonymous
    # loop variable even when the input PrimFuncs are structurally equal.
    source = re.sub(r"\b(?:i|v)_\d+\b", "generated_symbol", source)
    return " ".join(source.split())


@pytest.mark.parametrize("batch_size", [1, 4, 128, 512, 2048])
@pytest.mark.parametrize("scheduler", ["static", "unfused", "dynamic"])
def test_manual_and_dsl_are_structurally_equal(batch_size, scheduler):
    manual = _build_module(batch_size, scheduler, "manual")
    dsl = _build_module(batch_size, scheduler, "dsl")
    tvm.ir.assert_structural_equal(manual, dsl, map_free_vars=True)


@pytest.mark.parametrize("batch_size", [1, 4, 128, 512, 2048])
@pytest.mark.parametrize("scheduler", ["static", "unfused", "dynamic"])
def test_manual_and_dsl_generate_identical_cuda(batch_size, scheduler):
    manual_source, manual_lib = get_source(_build_module(batch_size, scheduler, "manual"))
    dsl_source, dsl_lib = get_source(_build_module(batch_size, scheduler, "dsl"))

    # Keep both runtime modules alive until after comparison.
    assert manual_lib is not None and dsl_lib is not None
    assert _normalize_cuda_source(manual_source) == _normalize_cuda_source(dsl_source)
