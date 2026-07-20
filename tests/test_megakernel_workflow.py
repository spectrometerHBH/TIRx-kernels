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
"""Stage-1 workflow contract checks for the six-stage MoE graph."""

from pathlib import Path

import yaml

from tirx_kernels._attrs import nested_attr_keys
from tirx_kernels.megakernel.dsl import VarSpec, build_moe_graph
from tirx_kernels.megakernel.utils.config import MEGAKERNEL_MOE_BENCH_CONFIG

_PLAN_PATH = (
    Path(__file__).parents[1]
    / "tirx_kernels"
    / "megakernel"
    / "workflow"
    / "qwen3_30b_a3b_moe_stage1.yaml"
)
_FORBIDDEN_KEYS = {
    "dispatch",
    "job_type",
    "level",
    "mask",
    "queue",
    "rank",
    "release",
    "runtime_init",
    "scope",
    "scope_id",
}


def _load_plan():
    with _PLAN_PATH.open(encoding="utf-8") as plan_file:
        return yaml.safe_load(plan_file)


def test_stage1_yaml_matches_native_graph_stages_tensors_and_events():
    plan = _load_plan()
    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)

    assert list(plan) == ["tiles", "tensors", "events", "dependencies", "validation"]
    assert list(plan["tiles"]) == [tile.name for tile in spec.tiles]
    assert set(plan["tensors"]) == set(spec.tensors)
    assert list(plan["events"]) == list(spec.events)
    assert "down_dispatch_done" not in yaml.safe_dump(plan)
    assert not nested_attr_keys(plan) & _FORBIDDEN_KEYS

    for tile in spec.tiles:
        planned = plan["tiles"][tile.name]
        assert planned["tile_impl"] == type(tile.impl).__name__
        assert planned["index_axes"] == ["m", "n", "k"]
        assert len(planned["tile_num"]) == len(tile.tile_num) == 3
        assert planned["reads"] == [tensor.name for tensor in tile.reads]
        assert planned["writes"] == [tensor.name for tensor in tile.writes]
        for planned_extent, extent in zip(planned["tile_num"], tile.tile_num, strict=True):
            if isinstance(extent, VarSpec):
                assert planned_extent == extent.name

    for event in spec.events.values():
        planned = plan["events"][event.name]
        assert planned["kind"] == "count"
        assert planned["dtype"] == event.dtype
        assert len(planned["shape"]) == len(event.shape)


def test_stage1_yaml_tensor_roles_and_flow_match_native_graph():
    plan = _load_plan()
    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    producers = {name: None for name in spec.tensors}
    consumers = {name: [] for name in spec.tensors}
    for tile in spec.tiles:
        for tensor in tile.writes:
            producers[tensor.name] = tile.name
        for tensor in tile.reads:
            consumers[tensor.name].append(tile.name)

    for name, tensor in spec.tensors.items():
        planned = plan["tensors"][name]
        assert planned["role"] in {"input", "intermediate", "output"}
        assert planned["producer"] == producers[name]
        assert planned["consumers"] == consumers[name]
        assert planned["dtype"] == tensor.dtype
        assert not hasattr(tensor, "role")


def test_stage1_yaml_five_edges_and_coordinate_ranks_match_native_graph():
    plan = _load_plan()
    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, 128)
    notifiers = {id(event): tile.name for tile in spec.tiles for event, _ in tile.notifies}
    graph_edges = [
        (notifiers[id(event)], tile.name, event.name)
        for tile in spec.tiles
        for event, _ in tile.waits
    ]
    planned_edges = [
        (dependency["producer"], dependency["consumer"], dependency["event"])
        for dependency in plan["dependencies"]
    ]
    assert len(graph_edges) == len(planned_edges) == 5
    assert planned_edges == graph_edges

    for dependency in plan["dependencies"]:
        rank = len(plan["events"][dependency["event"]]["shape"])
        assert dependency["notify"]["tile"] == dependency["producer"]
        assert dependency["wait"]["tile"] == dependency["consumer"]
        assert len(dependency["notify"]["coord_map"]) == rank
        assert len(dependency["wait"]["coord_map"]) == rank
