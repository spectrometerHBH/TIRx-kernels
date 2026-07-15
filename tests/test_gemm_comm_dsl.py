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

import hashlib
import importlib
import inspect
import json
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tvm_ffi

import tirx_kernels.gemm_comm.allgather_gemm as allgather_gemm
import tirx_kernels.gemm_comm.gemm_reduce_scatter as gemm_reduce_scatter
import tvm
from tirx_kernels.gemm_comm.dsl import (
    GemmCommHostExecutor,
    GemmCommLowerer,
    GemmCommRuntimeBindings,
    build_allgather_gemm_graph,
    build_gemm_reduce_scatter_graph,
    policy_for_scheduler,
)
from tirx_kernels.megakernel.examples.allgather_gemm import build_example as build_allgather_example
from tirx_kernels.megakernel.examples.allgather_gemm import main as allgather_main
from tirx_kernels.megakernel.examples.gemm_reduce_scatter import (
    build_example as build_reduce_scatter_example,
)
from tirx_kernels.megakernel.examples.gemm_reduce_scatter import main as reduce_scatter_main
from tvm.megakernel.dsl import TileImpl
from tvm.megakernel.transform import (
    FetchGuardStep,
    HostCallStep,
    HostSyncStep,
    MidBodyPortStep,
    RunStep,
)

_allgather_gemm_runner = importlib.import_module("tirx_kernels.gemm_comm._allgather_gemm_runner")
_gemm_reduce_scatter_runner = importlib.import_module(
    "tirx_kernels.gemm_comm._gemm_reduce_scatter_runner"
)

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
_GOLDEN = json.loads(Path(__file__).with_name("megakernel_oracles.json").read_text())["oracles"][
    "gemm_comm"
]["cases"]


def _keys(value):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, tuple | list):
        for item in value:
            yield from _keys(item)


def _shape_signature(shape):
    if isinstance(shape, tuple | list):
        return tuple(shape)
    return (shape,)


def _dependency_signature(dependency):
    samples = ((0, 0, 0), (1, 2, 3), (7, 5, 3))
    event, coord_map = dependency
    coordinates = tuple(tuple(coord_map(*sample)) for sample in samples)
    return (event.name, coordinates)


def _graph_signature(spec):
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
                tuple(tile.tile_num),
                tuple(tensor.name for tensor in tile.reads),
                tuple(tensor.name for tensor in tile.writes),
                tuple(_dependency_signature(dependency) for dependency in tile.waits),
                tuple(_dependency_signature(dependency) for dependency in tile.notifies),
                tile.attrs,
            )
            for tile in spec.tiles
        ),
    }


def _tile_run_source(tile_impl):
    run = type(tile_impl).run
    closure = inspect.getclosurevars(run).nonlocals
    inline = closure.get("obj")
    return inspect.getsource(inline.func if inline is not None else run)


def _inline_source(function):
    inline = inspect.getclosurevars(function).nonlocals.get("obj")
    return inspect.getsource(inline.func if inline is not None else function)


def _cuda_sha256(module) -> str:
    sources = []
    previous_postproc = tvm.get_global_func("tvm_callback_cuda_postproc", allow_missing=True)

    @tvm.register_global_func("tvm_callback_cuda_postproc", override=True)
    def capture_source(code, target):
        del target
        sources.append(code)
        return code

    try:
        tvm.compile(module, target=tvm.target.Target("cuda"), tir_pipeline="tirx")
    except RuntimeError:
        # NVSHMEM headers and libraries are runtime-environment dependencies;
        # CUDA source is already complete when that external compilation fails.
        if not sources:
            raise
    finally:
        tvm.register_global_func(
            "tvm_callback_cuda_postproc",
            previous_postproc or (lambda code, target: code),
            override=True,
        )

    source = re.sub(r"\b(?:i|v)_\d+\b", "generated_symbol", sources[-1])
    source = " ".join(source.split())
    return hashlib.sha256(source.encode()).hexdigest()


@pytest.mark.parametrize(
    "example_builder,production_builder,production_name",
    [
        (build_allgather_example, build_allgather_gemm_graph, "build_allgather_gemm_graph"),
        (
            build_reduce_scatter_example,
            build_gemm_reduce_scatter_graph,
            "build_gemm_reduce_scatter_graph",
        ),
    ],
)
def test_standalone_examples_contain_complete_dsl_and_match_production_graphs(
    example_builder, production_builder, production_name
) -> None:
    source = inspect.getsource(example_builder)
    assert "KernelSpec(" in source
    assert ".tile(" in source
    assert production_name not in source
    assert _graph_signature(example_builder()) == _graph_signature(production_builder().validate())


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
def test_logical_graphs_use_concrete_kernel_tile_impls_without_scheduler_attrs(
    builder, tile_names, event_names
) -> None:
    spec = builder().validate()
    execution = policy_for_scheduler("static").normalize(spec).execution
    device_tile_names = {
        program.tile.name for region in execution.device_regions for program in region.tile_programs
    }
    assert [tile.name for tile in spec.tiles] == tile_names
    assert list(spec.events) == event_names
    assert all(isinstance(tile.impl, TileImpl) for tile in spec.tiles)
    assert all(not hasattr(tile.impl, "tile_task") for tile in spec.tiles)
    assert {type(tile.impl).__module__ for tile in spec.tiles} <= {
        "tirx_kernels.gemm_comm.allgather_gemm",
        "tirx_kernels.gemm_comm.gemm_reduce_scatter",
    }
    for tile in spec.tiles:
        source = _tile_run_source(tile.impl)
        if tile.name in device_tile_names:
            assert "T." in source or "Tx." in source
        else:
            assert "GemmCommHostExecutor" in source
        assert not hasattr(tile.impl, "execution_space")
        assert not hasattr(tile.impl, "entrypoint")

    attrs = [spec.attrs]
    attrs.extend(event.attrs for event in spec.events.values())
    attrs.extend(tile.attrs for tile in spec.tiles)
    assert not (_FORBIDDEN & set(key for attr in attrs for key in _keys(attr)))


@pytest.mark.parametrize(
    "builder,scheduler",
    [
        (build_allgather_gemm_graph, "static"),
        (build_allgather_gemm_graph, "dynamic"),
        (build_gemm_reduce_scatter_graph, "static"),
    ],
)
def test_lowerer_binds_and_executes_attached_device_tile_impls(builder, scheduler) -> None:
    spec = builder()
    lowered = GemmCommLowerer(policy_for_scheduler(scheduler)).lower(spec)
    device_impls = [
        program.tile.impl
        for region in lowered.execution.device_regions
        for program in region.tile_programs
    ]

    assert lowered.module is not None
    assert all(tile_impl._resources_initialized for tile_impl in device_impls)


@pytest.mark.parametrize(
    "builder,scheduler",
    [
        (build_allgather_gemm_graph, "static"),
        (build_allgather_gemm_graph, "dynamic"),
        (build_gemm_reduce_scatter_graph, "static"),
    ],
)
def test_region_entrypoints_are_derived_from_execution_regions(builder, scheduler) -> None:
    spec = builder()
    lowered = GemmCommLowerer(policy_for_scheduler(scheduler)).lower(spec, plan_only=True)

    assert lowered.device_entrypoints == tuple(
        region.attrs["entrypoint"] for region in lowered.execution.device_regions
    )
    assert lowered.host_entrypoints == tuple(
        region.attrs["entrypoint"] for region in lowered.execution.host_regions
    )
    assert lowered.execution is lowered.plan.execution
    assert not hasattr(lowered.plan, "execution_plan")
    assert not hasattr(lowered.plan, "launch_steps")
    assert not hasattr(lowered.plan, "region_entrypoints")
    normalized = lowered.plan.normalized_data()
    assert "launch_steps" not in normalized and "region_entrypoints" not in normalized
    assert [region["name"] for region in normalized["regions"]] == [
        region.name for region in lowered.execution.regions_in_dependency_order()
    ]
    assert all(
        not hasattr(tile.impl, "execution_space") and not hasattr(tile.impl, "entrypoint")
        for tile in spec.tiles
    )

    region = lowered.execution.device_regions[0]
    attrs = {key: value for key, value in region.attrs.items() if key != "entrypoint"}
    execution = replace(
        lowered.execution,
        device_regions=(replace(region, attrs=attrs), *lowered.execution.device_regions[1:]),
    )
    with pytest.raises(ValueError, match="entrypoint"):
        replace(lowered.plan, execution=execution).validate()


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
def test_allgather_shard_ready_is_bound_to_fetch_before_publish(scheduler: str) -> None:
    plan = policy_for_scheduler(scheduler).normalize(build_allgather_gemm_graph())
    execution = plan.execution
    fetch = execution.device_regions[0].fetch_steps
    assert len(fetch) == 1 and isinstance(fetch[0], FetchGuardStep)
    assert tuple(type(step) for step in execution.device_regions[0].tile_programs[0].steps) == (
        RunStep,
    )
    placement = execution.edge_placements()[0]
    assert (placement.location, placement.region) == ("fetch", "gemm_device")

    if scheduler == "dynamic":
        source = _inline_source(allgather_gemm.GEMMMPMCQueue.dequeue)
        assert source.index("fetch_program.emit") < source.index("while_ld_global_acquire")
    else:
        source = _inline_source(allgather_gemm.SingleStaticTileScheduler._update_current_tile)
        assert source.index("fetch_program.emit") < source.index("self.fetched_task_type[0] = -1")


def test_allgather_fetch_predicate_is_authoritative() -> None:
    plan = policy_for_scheduler("dynamic").normalize(build_allgather_gemm_graph())
    execution = plan.execution
    region = execution.device_regions[0]
    step = replace(region.fetch_steps[0], predicate="invalid_mutated_predicate")
    execution = replace(execution, device_regions=(replace(region, fetch_steps=(step,)),))

    with pytest.raises(tvm.error.DiagnosticError, match="invalid_mutated_predicate"):
        allgather_gemm.build_kernel("dynamic", execution_plan=execution)


def test_reduce_scatter_private_port_and_host_completion_order() -> None:
    plan = policy_for_scheduler("static").normalize(build_gemm_reduce_scatter_graph())
    execution = plan.execution
    partial_program = execution.device_regions[0].tile_programs[0]
    assert tuple(type(step) for step in partial_program.steps) == (RunStep, MidBodyPortStep)
    assert any(
        isinstance(step, HostSyncStep) for region in execution.host_regions for step in region.steps
    )
    assert any(
        isinstance(step, HostCallStep) for region in execution.host_regions for step in region.steps
    )

    source = _tile_run_source(
        next(tile.impl for tile in plan.spec.tiles if tile.name == "partial_gemm")
    )
    assert source.rindex("partial_ready_port.emit") < source.rindex("mma2ld_pipe.advance")

    trace = []
    bindings = GemmCommRuntimeBindings(
        launch_device=lambda name: trace.append(("device", name)),
        launch_host=lambda name: trace.append(("host", name)),
        communication_barrier=lambda: trace.append(("barrier",)),
        communication_to_compute_sync=lambda: trace.append(("sync",)),
    )
    GemmCommHostExecutor(bindings).execute(plan)
    assert trace == [
        ("device", "test_mma_ss_tma_2sm_persistent"),
        ("host", "runtime.disco.transfer_to_peers_reduce_scatter"),
        ("barrier",),
        ("sync",),
        ("device", "reduce_sum"),
    ]


def test_reduce_scatter_port_and_host_sync_steps_are_authoritative() -> None:
    plan = policy_for_scheduler("static").normalize(build_gemm_reduce_scatter_graph())
    execution = plan.execution
    region = execution.device_regions[0]
    program = region.tile_programs[0]
    port = next(step for step in program.steps if isinstance(step, MidBodyPortStep))
    steps = tuple(
        replace(step, port="after_pipeline_advance") if step is port else step
        for step in program.steps
    )
    mutated = replace(
        execution,
        device_regions=(
            replace(region, tile_programs=(replace(program, steps=steps),)),
            *execution.device_regions[1:],
        ),
    )
    with pytest.raises(ValueError, match="approved GEMM epilogue port"):
        replace(plan, execution=mutated).validate()

    host = execution.host_regions[0]
    host_steps = tuple(
        replace(step, kind="invalid_sync") if isinstance(step, HostSyncStep) else step
        for step in host.steps
    )
    mutated = replace(execution, host_regions=(replace(host, steps=host_steps),))
    bindings = GemmCommRuntimeBindings(
        launch_device=lambda name: None,
        launch_host=lambda name: None,
        communication_barrier=lambda: None,
        communication_to_compute_sync=lambda: None,
    )
    with pytest.raises(ValueError, match="invalid_sync"):
        GemmCommHostExecutor(bindings).execute(replace(plan, execution=mutated))


def test_production_launches_use_region_executor_and_tileimpl_resource_hooks() -> None:
    allgather_launch = inspect.getsource(_allgather_gemm_runner._Case.launch)
    reduce_scatter_launch = inspect.getsource(_gemm_reduce_scatter_runner._Case.launch)
    assert "GemmCommHostExecutor" in allgather_launch
    assert "GemmCommHostExecutor" in reduce_scatter_launch

    ag_storage = _inline_source(allgather_gemm.AllGatherGemmTileImpl.init_storage)
    ag_device_init = _inline_source(allgather_gemm.AllGatherGemmTileImpl.device_init)
    partial_storage = _inline_source(gemm_reduce_scatter.PartialGemmTileImpl.init_storage)
    reduce_storage = _inline_source(gemm_reduce_scatter.ReduceSumTileImpl.init_storage)
    assert "T.decl_buffer" in ag_storage and "BarTMA2MMA" in ag_device_init
    assert "T.decl_buffer" in partial_storage and "TMA2MMAPipeline" in partial_storage
    assert "T.decl_buffer" in reduce_storage and "ReducePipe" in reduce_storage
    assert "init_instance_resources" not in inspect.getsource(allgather_gemm.build_kernel)
    assert "init_instance_resources" not in inspect.getsource(gemm_reduce_scatter.build_kernel)


@pytest.mark.parametrize("scheduler", ["static", "dynamic"])
def test_allgather_dsl_and_manual_paths_emit_identical_ir(scheduler: str) -> None:
    dsl_kernel = allgather_gemm.get_kernel(scheduler=scheduler)
    manual_kernel = _allgather_gemm_runner._get_manual_oracle_kernel(scheduler)
    tvm.ir.assert_structural_equal(dsl_kernel, manual_kernel, map_free_vars=True)
    expected = _GOLDEN[f"allgather_gemm_{scheduler}"]["structural_hash"]
    assert tvm_ffi.structural_hash(dsl_kernel, map_free_vars=True) == expected
    assert tvm_ffi.structural_hash(manual_kernel, map_free_vars=True) == expected


def test_reduce_scatter_static_dsl_preserves_existing_ir() -> None:
    dsl_module = gemm_reduce_scatter.get_kernel(scheduler="static")
    manual_module = _gemm_reduce_scatter_runner._get_manual_oracle_kernel("static")
    tvm.ir.assert_structural_equal(dsl_module, manual_module, map_free_vars=True)
    expected = _GOLDEN["gemm_reduce_scatter_static"]["structural_hash"]
    assert tvm_ffi.structural_hash(dsl_module, map_free_vars=True) == expected
    assert tvm_ffi.structural_hash(manual_module, map_free_vars=True) == expected


@pytest.mark.parametrize(
    "case_name,builder",
    [
        ("allgather_gemm_static", lambda: allgather_gemm.get_kernel(scheduler="static")),
        ("allgather_gemm_dynamic", lambda: allgather_gemm.get_kernel(scheduler="dynamic")),
        ("gemm_reduce_scatter_static", lambda: gemm_reduce_scatter.get_kernel(scheduler="static")),
    ],
)
def test_gemm_comm_cuda_matches_frozen_oracle(case_name, builder) -> None:
    assert _cuda_sha256(builder()) == _GOLDEN[case_name]["cuda_sha256"]


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
    "entrypoint,scheduler,expected",
    [
        (allgather_main, "dynamic", "physical scheduler: mpmc_queue"),
        (allgather_main, "static", "physical scheduler: rank_aware_grid_stride"),
        (reduce_scatter_main, "static", "lowerable: true"),
        (reduce_scatter_main, "dynamic", "lowerable: false"),
    ],
)
def test_dsl_example_entrypoint(entrypoint, scheduler: str, expected: str, capsys) -> None:
    entrypoint(["--scheduler", scheduler])
    assert expected in capsys.readouterr().out
