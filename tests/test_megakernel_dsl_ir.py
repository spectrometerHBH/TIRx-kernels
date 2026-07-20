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
"""IR equivalence gates for the MoE DSL and private migration oracle."""

import hashlib
import json
import re
from pathlib import Path
from unittest import SkipTest

import pytest
import tvm_ffi

import tvm
from tirx_kernels.megakernel.moe import MegaKernelMOE, _require_cuda_sm100
from tirx_kernels.megakernel.utils.config import MEGAKERNEL_MOE_BENCH_CONFIG
from tirx_kernels.megakernel.utils.utils import get_source

_GOLDEN = json.loads(Path(__file__).with_name("megakernel_oracles.json").read_text())["oracles"][
    "moe"
]["cases"]
_CASES = [
    (batch_size, scheduler, False)
    for batch_size in [1, 4, 128, 512, 2048, 4096]
    for scheduler in ["static", "unfused", "dynamic"]
] + [(4, "dynamic", True)]


def _build_module(
    batch_size: int, scheduler: str, *, profiler_on: bool = False, oracle: bool = False
):
    kernel = MegaKernelMOE(
        config=MEGAKERNEL_MOE_BENCH_CONFIG,
        batch_size=batch_size,
        world_size=1,
        profiler_on=profiler_on,
    )
    if oracle:
        return kernel._get_manual_oracle_module(scheduler)
    return kernel.get_module(scheduler)


def _normalize_cuda_source(source: str) -> str:
    # TIRX may choose either an ``i_N`` or ``v_N`` spelling for an anonymous
    # loop variable even when the input PrimFuncs are structurally equal.
    source = re.sub(r"\b(?:i|v)_\d+\b", "generated_symbol", source)
    return " ".join(source.split())


def _golden_key(batch_size: int, scheduler: str, profiler_on: bool) -> str:
    return f"moe_b{batch_size}_{scheduler}_prof{int(profiler_on)}"


@pytest.fixture(scope="module")
def cuda_sm100():
    """Skip CUDA source equivalence when the required compiler target is unavailable."""

    try:
        _require_cuda_sm100()
    except SkipTest as err:
        pytest.skip(str(err))


@pytest.mark.parametrize("batch_size,scheduler,profiler_on", _CASES)
def test_manual_and_dsl_are_structurally_equal(batch_size, scheduler, profiler_on):
    manual = _build_module(batch_size, scheduler, profiler_on=profiler_on, oracle=True)
    dsl = _build_module(batch_size, scheduler, profiler_on=profiler_on)
    tvm.ir.assert_structural_equal(manual, dsl, map_free_vars=True)
    expected = _GOLDEN[_golden_key(batch_size, scheduler, profiler_on)]["structural_hash"]
    assert tvm_ffi.structural_hash(manual, map_free_vars=True) == expected
    assert tvm_ffi.structural_hash(dsl, map_free_vars=True) == expected


@pytest.mark.parametrize("batch_size,scheduler,profiler_on", _CASES)
def test_manual_and_dsl_generate_identical_cuda(batch_size, scheduler, profiler_on, cuda_sm100):
    del cuda_sm100
    manual_source, manual_lib = get_source(
        _build_module(batch_size, scheduler, profiler_on=profiler_on, oracle=True)
    )
    dsl_source, dsl_lib = get_source(_build_module(batch_size, scheduler, profiler_on=profiler_on))

    # Keep both runtime modules alive until after comparison.
    assert manual_lib is not None and dsl_lib is not None
    manual_source = _normalize_cuda_source(manual_source)
    dsl_source = _normalize_cuda_source(dsl_source)
    assert manual_source == dsl_source
    expected = _GOLDEN[_golden_key(batch_size, scheduler, profiler_on)]["cuda_sha256"]
    assert hashlib.sha256(manual_source.encode()).hexdigest() == expected
    assert hashlib.sha256(dsl_source.encode()).hexdigest() == expected
