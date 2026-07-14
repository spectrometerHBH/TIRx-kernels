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

from collections.abc import Mapping

import numpy as np
import pytest

import tvm
from tirx_kernels.gemm_comm import allgather_gemm, gemm_reduce_scatter
from tirx_kernels.gemm_comm.dsl import (
    GemmCommLowerer,
    build_allgather_gemm_graph,
    build_gemm_reduce_scatter_graph,
    policy_for_scheduler,
)
from tirx_kernels.gemm_comm.examples.gemm_comm_dsl import main
from tvm.megakernel.dsl import TileImpl

_FORBIDDEN = {
    "dispatch",
    "job_type",
    "level",
    "mask",
    "queue",
    "rank",
    "release",
    "runtime_init",
    "scheduler",
    "scope",
    "scope_id",
}


def _keys(value):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, tuple | list):
        for item in value:
            yield from _keys(item)


@pytest.mark.parametrize(
    "builder, tile_names, event_names",
    [
        (build_allgather_gemm_graph, ["allgather", "gemm"], ["shard_ready"]),
        (
            build_gemm_reduce_scatter_graph,
            ["partial_gemm", "transfer", "reduce"],
            ["partial_shard_ready", "staging_ready"],
        ),
    ],
)
def test_logical_graphs_use_tvm_tile_impls_without_scheduler_attrs(
    builder, tile_names, event_names
) -> None:
    spec = builder().validate()
    assert [tile.name for tile in spec.tiles] == tile_names
    assert list(spec.events) == event_names
    assert all(isinstance(tile.impl, TileImpl) for tile in spec.tiles)
    assert all(tile.impl.tile_task is not None for tile in spec.tiles)
    assert all(
        tile.impl.tile_task.module_factory is not None
        for tile in spec.tiles
        if tile.impl.execution_space == "device"
    )

    attrs = [spec.attrs]
    attrs.extend(event.attrs for event in spec.events.values())
    attrs.extend(tile.attrs for tile in spec.tiles)
    assert not (_FORBIDDEN & set(key for attr in attrs for key in _keys(attr)))


@pytest.mark.parametrize("scheduler", ["static", "dynamic"])
def test_allgather_policies_cover_every_gemm_tile_once(scheduler: str) -> None:
    spec = build_allgather_gemm_graph()
    plan = policy_for_scheduler(scheduler).normalize(spec)

    assert plan.lowerable
    assert plan.task_count_per_rank == 1024
    assert plan.persistent_clusters == 74
    assert all(len(schedule.tasks) == 1024 for schedule in plan.rank_schedules)
    if scheduler == "static":
        counts = [len(queue) for queue in plan.rank_schedules[0].worker_queues]
        assert max(counts) - min(counts) <= 1
    else:
        assert all(schedule.shared_queue for schedule in plan.rank_schedules)


def test_dynamic_allgather_queue_is_generated_from_dsl_and_matches_manual_oracle() -> None:
    actual = allgather_gemm._queue_state()
    expected = allgather_gemm._manual_queue_state()
    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)


@pytest.mark.parametrize("scheduler", ["static", "dynamic"])
def test_allgather_dsl_and_manual_paths_emit_identical_ir(scheduler: str) -> None:
    dsl_kernel = allgather_gemm.get_kernel(scheduler=scheduler, use_dsl=True)
    manual_kernel = allgather_gemm.get_kernel(scheduler=scheduler, use_dsl=False)
    tvm.ir.assert_structural_equal(dsl_kernel, manual_kernel, map_free_vars=True)


def test_reduce_scatter_static_dsl_preserves_existing_ir() -> None:
    dsl_module = gemm_reduce_scatter.get_kernel(scheduler="static", use_dsl=True)
    manual_module = gemm_reduce_scatter.get_kernel(scheduler="static", use_dsl=False)
    tvm.ir.assert_structural_equal(dsl_module, manual_module, map_free_vars=True)


def test_reduce_scatter_dynamic_plan_is_explicit_but_not_mislowered() -> None:
    spec = build_gemm_reduce_scatter_graph()
    lowerer = GemmCommLowerer(policy_for_scheduler("dynamic"))
    plan = lowerer.lower(spec, plan_only=True).plan

    assert not plan.lowerable
    assert plan.physical_scheduler == "planned_mpmc_queue"
    assert plan.task_count_per_rank == 1536
    assert "serialize the inter-tile pipeline" in plan.unsupported_reason
    with pytest.raises(NotImplementedError, match="serialize the inter-tile pipeline"):
        lowerer.lower(spec)


@pytest.mark.parametrize(
    "workload,scheduler,expected",
    [
        ("allgather_gemm", "dynamic", "physical scheduler: mpmc_queue"),
        ("allgather_gemm", "static", "physical scheduler: rank_aware_grid_stride"),
        ("gemm_reduce_scatter", "static", "lowerable: true"),
        ("gemm_reduce_scatter", "dynamic", "lowerable: false"),
    ],
)
def test_dsl_example_entrypoint(workload: str, scheduler: str, expected: str, capsys) -> None:
    main(["--workload", workload, "--scheduler", scheduler])
    assert expected in capsys.readouterr().out
