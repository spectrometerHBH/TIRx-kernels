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

"""CPU tests for the MoE DSL example module surface."""

import pytest

import tirx_kernels.megakernel.dsl.examples as examples
import tirx_kernels.megakernel.dsl.examples.moe as moe_example
from tirx_kernels.megakernel.dsl.examples.moe import (
    build_moe_graph,
    build_moe_kernel,
    moe_lowering_options,
)
from tirx_kernels.megakernel.utils.config import MEGAKERNEL_MOE_BENCH_CONFIG
from tvm.megakernel.transform import LoweringOptions, RuntimeKernelBuild

_CONFIG = MEGAKERNEL_MOE_BENCH_CONFIG


def test_example_module_exports_the_runner_surface():
    assert set(moe_example.__all__) == {
        "build_moe_graph",
        "build_moe_kernel",
        "moe_lowering_options",
    }
    assert examples.__doc__


def test_build_moe_graph_input_validation():
    with pytest.raises(ValueError, match="Qwen3-30B-A3B"):
        build_moe_graph({}, 4)
    for bad_batch in (0, -1, 1.5, "4"):
        with pytest.raises(ValueError, match="batch_size"):
            build_moe_graph(_CONFIG, bad_batch)


def test_moe_lowering_options_returns_fresh_options():
    first = moe_lowering_options("dynamic", 128)
    second = moe_lowering_options("dynamic", 128)
    assert isinstance(first, LoweringOptions)
    assert first == second
    assert first.attrs is not second.attrs


def test_build_moe_kernel_result_surface():
    for scheduler, expected in (
        ("static", "static"),
        ("unfused", "static"),
        ("dynamic", "dynamic"),
    ):
        build = build_moe_kernel(_CONFIG, 4, scheduler)
        assert isinstance(build, RuntimeKernelBuild)
        assert build.scheduler == expected
        assert build.module["qwen3_30b_a3b_moe"] is not None
        assert build.sm_count > 0
        assert build.event_workspace_size > 0
        assert not build.profiler_on
        if expected == "static":
            assert build.exec_queue is not None
            assert build.exec_queue.shape == (build.sm_count, build.max_tasks)
            assert build.queue_tasks is None
        else:
            assert build.exec_queue is None
            assert build.queue_tasks.shape == (build.max_tasks,)
            assert build.queue_head.shape == build.queue_tail.shape == (1,)
            assert len(build.drain_events) == 1


def test_build_moe_kernel_shares_one_spec_shape_across_schedulers():
    fused = build_moe_graph(_CONFIG, 128)
    unfused = build_moe_graph(_CONFIG, 128, unfused=True)
    # Only the gate-up event shape/count and its two coordinate maps change.
    assert [tile.name for tile in fused.tiles] == [tile.name for tile in unfused.tiles]
    assert [tensor.name for tensor in fused.tensors.values()] == [
        tensor.name for tensor in unfused.tensors.values()
    ]
    assert fused.events["gate_up_done"].shape != unfused.events["gate_up_done"].shape
    for name in ("gating_done", "topk_done", "align_done", "count_sort_done", "down_dispatch_done"):
        assert fused.events[name].shape == unfused.events[name].shape
        assert fused.events[name].init_count == unfused.events[name].init_count
