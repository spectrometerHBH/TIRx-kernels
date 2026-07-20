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

from __future__ import annotations

import inspect

import numpy as np
import pytest

import tvm
from tirx_kernels.gemm_comm import allgather_gemm, gemm_reduce_scatter
from tirx_kernels.gemm_comm.dsl import (
    GemmCommLowerer,
    build_allgather_gemm_graph,
    build_gemm_reduce_scatter_graph,
    make_allgather_dynamic_queue,
    policy_for_scheduler,
)
from tirx_kernels.megakernel.examples.allgather_gemm import build_example as build_ag_example
from tirx_kernels.megakernel.examples.gemm_reduce_scatter import build_example as build_rs_example
from tvm.megakernel.transform import FetchGuardStep, QueuePushStep, RunStep


def _graph_shape(spec):
    return (
        spec.name,
        tuple(spec.tensors),
        tuple(spec.events),
        tuple((tile.name, tuple(tile.tile_num)) for tile in spec.tiles),
    )


@pytest.mark.parametrize(
    "example_builder,production_builder",
    [
        (build_ag_example, build_allgather_gemm_graph),
        (build_rs_example, build_gemm_reduce_scatter_graph),
    ],
)
def test_standalone_examples_match_production_graph_shape(
    example_builder, production_builder
) -> None:
    assert _graph_shape(example_builder()) == _graph_shape(production_builder())


def test_allgather_dynamic_plan_matches_the_direct_queue() -> None:
    config = allgather_gemm.derive_config()
    spec = build_allgather_gemm_graph(config)
    plan = policy_for_scheduler("dynamic").normalize(spec)

    assert len(plan.execution.device_regions) == 1
    assert len(plan.execution.host_regions) == 1
    assert isinstance(plan.execution.device_regions[0].fetch_steps[0], FetchGuardStep)
    assert plan.task_count_per_rank == config.task_count

    actual = make_allgather_dynamic_queue(plan)
    expected = allgather_gemm._queue_state(config)
    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_gemm_rs_is_one_fused_device_region_with_complete_queue_coverage() -> None:
    config = gemm_reduce_scatter.derive_config()
    spec = build_gemm_reduce_scatter_graph(config)
    plan = policy_for_scheduler("dynamic").normalize(spec)

    assert len(plan.execution.device_regions) == 1
    assert not plan.execution.host_regions
    region = plan.execution.device_regions[0]
    assert [program.tile.name for program in region.tile_programs] == [
        "partial_gemm",
        "reduce_scatter",
    ]
    assert tuple(type(step) for step in region.tile_programs[0].steps) == (RunStep, QueuePushStep)
    assert region.tile_programs[0].steps[0].repeat == 2
    assert plan.task_count_per_rank == config.gemm_task_count
    assert plan.pushed_task_count_per_rank == config.rs_task_count

    queue_state = gemm_reduce_scatter._queue_state(config)
    gemm_indices = queue_state[1]
    gemm_tails = queue_state[3]
    for schedule in plan.rank_schedules:
        tail = int(gemm_tails[schedule.rank, 0])
        expected = [tuple(index) for index in gemm_indices[schedule.rank, :tail]]
        assert [(task.m, task.n) for task in schedule.tasks] == expected
        assert {(task.m, task.n) for task in schedule.pushed_tasks} == {
            (m_idx, n_idx)
            for m_idx in range(config.rs_m_clusters)
            for n_idx in range(config.rs_n_clusters)
        }


@pytest.mark.parametrize("module", [allgather_gemm, gemm_reduce_scatter])
@pytest.mark.parametrize("world_size", [1, 4])
def test_dsl_and_manual_paths_share_the_direct_builder(module, world_size: int) -> None:
    kwargs = {
        "M": module.M,
        "N": module.N,
        "K": module.K if module is allgather_gemm else module.TOTAL_K,
        "world_size": world_size,
        "dtype": "float16",
        "scheduler": "dynamic",
    }
    dsl_module = module.get_kernel(**kwargs, use_dsl=True)
    manual_module = module.get_kernel(**kwargs, use_dsl=False)
    tvm.ir.assert_structural_equal(dsl_module, manual_module, map_free_vars=True)


@pytest.mark.parametrize("builder", [build_allgather_gemm_graph, build_gemm_reduce_scatter_graph])
def test_lowerer_builds_exactly_one_attached_module(builder) -> None:
    sentinel = object()
    seen = []

    def module_builder(execution):
        seen.append(execution)
        return sentinel

    spec = builder(module_builder=module_builder)
    lowered = GemmCommLowerer(policy_for_scheduler("dynamic")).lower(spec)
    assert lowered.module is sentinel
    assert seen == [lowered.execution]


def test_static_gemm_comm_policy_is_explicitly_rejected() -> None:
    with pytest.raises(ValueError, match="only scheduler='dynamic'"):
        policy_for_scheduler("static").normalize(build_allgather_gemm_graph())


def test_direct_kernel_files_retain_runtime_and_builder_ownership() -> None:
    for module in (allgather_gemm, gemm_reduce_scatter):
        source = inspect.getsource(module)
        assert "def run_test(" in source
        assert "def run_bench(" in source
        assert "_gemm_reduce_scatter_runner" not in source
        assert "_allgather_gemm_runner" not in source
