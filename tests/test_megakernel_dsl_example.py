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
from tirx_kernels.megakernel.dsl import KernelSpec, VarSpec, build_moe_graph
from tirx_kernels.megakernel.examples.moe import build_example, describe_graph, describe_plan, main
from tirx_kernels.megakernel.utils.config import MEGAKERNEL_MOE_BENCH_CONFIG


def _extent_signature(extent):
    if isinstance(extent, VarSpec):
        return ("var", extent.name)
    return extent


def _shape_signature(shape):
    if isinstance(shape, tuple | list):
        return tuple(_extent_signature(extent) for extent in shape)
    return (_extent_signature(shape),)


def _dependency_signature(dependency):
    samples = ((0, 0, 0), (1, 2, 3), (7, 5, 3))
    event, coord_map = dependency
    coordinates = tuple(tuple(coord_map(*sample)) for sample in samples)
    return (event.name, coordinates)


def _graph_signature(spec: KernelSpec):
    return {
        "name": spec.name,
        "attrs": spec.attrs,
        "tensors": tuple(
            (name, _shape_signature(tensor.shape), tensor.dtype)
            for name, tensor in spec.tensors.items()
        ),
        "events": tuple(
            (name, _shape_signature(event.shape), event.init_count, event.dtype, event.attrs)
            for name, event in spec.events.items()
        ),
        "tiles": tuple(
            (
                tile.name,
                type(tile.impl).__name__,
                tuple(_extent_signature(extent) for extent in tile.tile_num),
                tuple(tensor.name for tensor in tile.reads),
                tuple(tensor.name for tensor in tile.writes),
                tuple(_dependency_signature(dependency) for dependency in tile.waits),
                tuple(_dependency_signature(dependency) for dependency in tile.notifies),
                tile.attrs,
            )
            for tile in spec.tiles
        ),
    }


def test_moe_dsl_public_api_uses_split_modules():
    assert dsl.build_moe_graph.__module__.endswith(".moe_spec")
    assert dsl.MoeLoweringEnv.__module__.endswith(".lowering.model")
    assert dsl.DynamicPolicy.__module__.endswith(".lowering.policies")
    assert dsl.MoeLowerer.__module__.endswith(".lowering.lowerer")


def test_example_contains_complete_dsl_and_matches_production_graph():
    source = inspect.getsource(build_example)
    assert "KernelSpec(" in source
    assert ".tile(" in source
    assert "build_moe_graph" not in source

    example = build_example(128)
    production = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128).validate()
    assert _graph_signature(example) == _graph_signature(production)


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
