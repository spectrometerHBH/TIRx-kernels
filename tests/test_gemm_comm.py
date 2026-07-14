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

import json

import numpy as np
import pytest

from tirx_kernels.gemm_comm import _allgather_gemm_impl as ag_impl
from tirx_kernels.gemm_comm import _gemm_reduce_scatter_impl as rs_impl
from tirx_kernels.gemm_comm import allgather_gemm, gemm_reduce_scatter
from tirx_kernels.gemm_comm._baselines import _JSON_PREFIX, _decode_result
from tirx_kernels.gemm_comm._bench import iteration_counts, resolve_budget
from tirx_kernels.registry import discover_kernels


def test_gemm_comm_registry_entries() -> None:
    kernels = discover_kernels(category="gemm_comm")
    assert set(kernels) == {"allgather_gemm", "gemm_reduce_scatter"}
    assert all(module.KERNEL_META["compute_capability"] == 10 for module in kernels.values())


def test_tuned_tp4_configs_are_explicit() -> None:
    assert allgather_gemm.CONFIGS == [
        {
            "M": 8192,
            "N": 65536,
            "K": 8192,
            "world_size": 4,
            "dtype": "float16",
            "label": "tp4_m8192_n65536_k8192_fp16",
        }
    ]
    assert gemm_reduce_scatter.CONFIGS == [
        {
            "M": 16384,
            "N": 12288,
            "K": 49152,
            "world_size": 4,
            "dtype": "float16",
            "label": "tp4_m16384_n12288_k49152_fp16",
        }
    ]


@pytest.mark.parametrize(
    "module, overrides",
    [
        (allgather_gemm, {"M": ag_impl.M + 1}),
        (allgather_gemm, {"world_size": ag_impl.WORLD_SIZE - 1}),
        (gemm_reduce_scatter, {"K": rs_impl.TOTAL_K + 1}),
        (gemm_reduce_scatter, {"dtype": "bfloat16"}),
    ],
)
def test_tuned_kernels_reject_other_configs(module, overrides) -> None:
    with pytest.raises(ValueError, match="supports only"):
        module.get_kernel(**overrides)


def test_allgather_dynamic_queue_has_exact_coverage() -> None:
    task_types, task_indices, heads, tails = allgather_gemm._queue_state()
    expected_tasks = ag_impl.GEMM_M_CLUSTERS * ag_impl.GEMM_N_CLUSTERS
    expected_indices = {
        (m_index, n_index)
        for m_index in range(ag_impl.GEMM_M_CLUSTERS)
        for n_index in range(ag_impl.GEMM_N_CLUSTERS)
    }

    assert task_types.shape == (ag_impl.WORLD_SIZE, ag_impl.CAPACITY)
    assert task_indices.shape == (ag_impl.WORLD_SIZE, ag_impl.CAPACITY, ag_impl.TASK_IDX_LEN)
    np.testing.assert_array_equal(heads, 0)
    np.testing.assert_array_equal(tails, expected_tasks)
    for rank in range(ag_impl.WORLD_SIZE):
        np.testing.assert_array_equal(
            task_types[rank, :expected_tasks], ag_impl.TaskType.GEMM.value
        )
        assert set(map(tuple, task_indices[rank, :expected_tasks])) == expected_indices
        np.testing.assert_array_equal(task_types[rank, expected_tasks:], -1)


def test_distributed_benchmark_budgets_are_milliseconds() -> None:
    assert resolve_budget(None, 25, "warmup") == 25
    assert resolve_budget(7, 25, "warmup") == 7
    assert iteration_counts(2_000.0, 25, 100) == (12, 50)
    assert iteration_counts(0.0, 25, 100) == (1000, 1000)
    with pytest.raises(ValueError, match="positive integer"):
        resolve_budget(0, 25, "repeat")


def test_baseline_result_parser_ignores_library_logs() -> None:
    payload = {"status": "OK", "implementations": {"cublasmp_split_p2p": {}}}
    stdout = "NCCL INFO initialized\n" + _JSON_PREFIX + json.dumps(payload) + "\n"
    assert _decode_result(stdout) == payload
    with pytest.raises(RuntimeError, match="did not emit"):
        _decode_result("NCCL INFO initialized\n")
