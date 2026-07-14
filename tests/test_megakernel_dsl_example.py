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

"""CPU checks for the runnable MoE DSL example."""

import inspect

import pytest

from tirx_kernels.megakernel import dsl
from tirx_kernels.megakernel.dsl import moe_dsl
from tirx_kernels.megakernel.examples.moe_dsl import (
    build_example,
    describe_graph,
    describe_plan,
    main,
)


def test_moe_dsl_is_split_into_logical_and_lowering_modules():
    assert dsl.build_moe_graph.__module__.endswith(".moe_spec")
    assert dsl.MoeLoweringEnv.__module__.endswith(".lowering.model")
    assert dsl.DynamicPolicy.__module__.endswith(".lowering.policies")
    assert dsl.MoeLowerer.__module__.endswith(".lowering.lowerer")
    assert moe_dsl.build_moe_graph is dsl.build_moe_graph
    assert moe_dsl.MoeLowerer is dsl.MoeLowerer
    assert len(inspect.getsource(moe_dsl).splitlines()) < 100


def test_example_describes_the_complete_logical_graph():
    spec = build_example(128)
    description = describe_graph(spec, 128)

    assert "logical events (5)" in description
    assert "tiles (6)" in description
    assert "gate_up_silu: GateUpSiluTileImpl tile_num=(routed_rows, 12, 1)" in description
    assert "down_dispatch_done" not in description


@pytest.mark.parametrize("scheduler", ["static", "unfused", "dynamic"])
def test_example_lowers_the_same_spec_through_each_policy(scheduler):
    spec = build_example(4)
    description = describe_plan(spec, scheduler)

    assert f"scheduler: {scheduler}" in description
    assert "queue upper bound:" in description


def test_example_module_entrypoint(capsys):
    main(["--batch-size", "128", "--scheduler", "dynamic"])
    output = capsys.readouterr().out

    assert "kernel: qwen3_30b_a3b_moe" in output
    assert "scheduler: dynamic" in output
