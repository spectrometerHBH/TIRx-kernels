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

"""Runtime correctness for the standalone MoE DSL lowering."""

from unittest import SkipTest

import pytest
import torch

import tvm
from tirx_kernels.megakernel.dsl import MoeLowerer, build_moe_graph, policy_for_scheduler
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


def _use_plan_queue(case, plan) -> None:
    dev = tvm.cuda(0)
    if not plan.is_dynamic:
        case["exec_queue"] = tvm.runtime.tensor(plan.make_static_queue(), dev)
        return

    queue = plan.make_dynamic_queue()
    case["graph_reset"].update(
        {
            "queue_tasks_source": _as_cuda_tensor(queue.tasks.copy()),
            "queue_head_source": _as_cuda_tensor(queue.head.copy()),
            "queue_tail_source": _as_cuda_tensor(queue.tail.copy()),
            "queue_tasks": [],
            "queue_head": [],
            "queue_tail": [],
        }
    )
    case["queue_tasks"] = []
    case["queue_head"] = []
    case["queue_tail"] = []
    for _ in range(case["launch_slots"]):
        tasks = tvm.runtime.tensor(queue.tasks.copy(), dev)
        head = tvm.runtime.tensor(queue.head.copy(), dev)
        tail = tvm.runtime.tensor(queue.tail.copy(), dev)
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

    spec = build_moe_graph(MEGAKERNEL_MOE_BENCH_CONFIG, batch_size)
    lowerer = MoeLowerer(policy_for_scheduler(scheduler))
    plan = lowerer.lower(spec)
    _, dsl_lib = get_source(lowerer.build_module())
    dsl_kernel = lowerer.kernel

    manual_kernel, manual_libs = _compile_moe_schedulers((scheduler,), batch_size, 1, False)
    _reset_prepare_data_cache()
    data = dict(prepare_data(batch_size, dsl_kernel))
    dsl_case = _make_tir_case(
        batch_size=batch_size,
        mk=dsl_kernel,
        lib=dsl_lib,
        scheduler=scheduler,
        data=data,
        launch_slots=1,
    )
    _use_plan_queue(dsl_case, plan)
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
        dsl_kernel,
        check_torch=batch_size <= 128,
    )
    torch.cuda.synchronize()
