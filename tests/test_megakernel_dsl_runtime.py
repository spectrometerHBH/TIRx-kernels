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

"""Runtime correctness for the tvm-builder MoE DSL path (B200 GPU gate)."""

from unittest import SkipTest

import numpy as np
import pytest
import torch

import tvm
from tirx_kernels.megakernel.dsl.examples.moe import build_moe_kernel
from tirx_kernels.megakernel.moe import (
    _as_cuda_tensor,
    _compile_moe_schedulers,
    _make_tir_case,
    _require_cuda_sm100,
    _reset_prepare_data_cache,
    _reset_tir_case_for_cuda_graph,
    _validate_tir_case,
    prepare_data,
)
from tirx_kernels.megakernel.utils.config import MEGAKERNEL_MOE_BENCH_CONFIG
from tirx_kernels.megakernel.utils.utils import get_source

# Mirrors the graph-reset zero list built by _make_tir_case.
_ZERO_NAMES = (
    "output",
    "topk_weights",
    "topk_indices",
    "sorted_token_ids",
    "expert_ids",
    "num_valid_tokens",
    "num_tokens_post_pad",
    "cumsum_buffer",
    "reordered_hidden_state",
    "gate_up_output",
    "silu_mul_output",
    "etensor_workspace",
    "profiler_buffer",
)


def _use_build_products(case, build) -> None:
    """Feed a tvm-builder kernel and its host queues into the case harness."""

    dev = tvm.cuda(0)
    # The tvm-built kernel declares the workspace at its exact size.
    case["etensor_workspace"] = tvm.runtime.tensor(
        np.zeros((build.event_workspace_size,), dtype=np.int32), dev
    )
    case["graph_reset"]["zero"] = [torch.from_dlpack(case[name]) for name in _ZERO_NAMES]
    if build.scheduler == "static":
        case["exec_queue"] = tvm.runtime.tensor(build.exec_queue, dev)
        return

    case["graph_reset"].update(
        {
            "queue_tasks_source": _as_cuda_tensor(build.queue_tasks.copy()),
            "queue_head_source": _as_cuda_tensor(build.queue_head.copy()),
            "queue_tail_source": _as_cuda_tensor(build.queue_tail.copy()),
            "queue_tasks": [],
            "queue_head": [],
            "queue_tail": [],
        }
    )
    case["queue_tasks"] = []
    case["queue_head"] = []
    case["queue_tail"] = []
    for _ in range(case["launch_slots"]):
        tasks = tvm.runtime.tensor(build.queue_tasks.copy(), dev)
        head = tvm.runtime.tensor(build.queue_head.copy(), dev)
        tail = tvm.runtime.tensor(build.queue_tail.copy(), dev)
        case["queue_tasks"].append(tasks)
        case["queue_head"].append(head)
        case["queue_tail"].append(tail)
        case["graph_reset"]["queue_tasks"].append(torch.from_dlpack(tasks))
        case["graph_reset"]["queue_head"].append(torch.from_dlpack(head))
        case["graph_reset"]["queue_tail"].append(torch.from_dlpack(tail))


@pytest.mark.parametrize("batch_size", [1, 4, 128, 512, 2048])
@pytest.mark.parametrize("scheduler", ["static", "dynamic", "unfused"])
def test_megakernel_moe_dsl(batch_size, scheduler):
    try:
        _require_cuda_sm100()
    except SkipTest as err:
        pytest.skip(str(err))

    build = build_moe_kernel(MEGAKERNEL_MOE_BENCH_CONFIG, batch_size, scheduler)
    _, dsl_lib = get_source(build.module)

    manual_kernel, manual_libs = _compile_moe_schedulers((scheduler,), batch_size, 1, False)
    _reset_prepare_data_cache()
    data = dict(prepare_data(batch_size, manual_kernel))
    dsl_case = _make_tir_case(
        batch_size=batch_size,
        mk=manual_kernel,
        lib=manual_libs[scheduler],
        scheduler=scheduler,
        data=data,
        launch_slots=1,
    )
    dsl_case["kernel"] = dsl_lib["qwen3_30b_a3b_moe"]
    _use_build_products(dsl_case, build)
    manual_case = _make_tir_case(
        batch_size=batch_size,
        mk=manual_kernel,
        lib=manual_libs[scheduler],
        scheduler=scheduler,
        data=data,
        launch_slots=1,
    )

    _reset_tir_case_for_cuda_graph(dsl_case)
    _reset_tir_case_for_cuda_graph(manual_case)
    _validate_tir_case(
        {"tir": dsl_case, "tir_reference": manual_case, "cpu_data": data},
        manual_kernel,
        check_torch=batch_size <= 128,
    )
    torch.cuda.synchronize()
