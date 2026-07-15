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
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from tirx_kernels.gemm_comm import _baselines as baselines
from tirx_kernels.gemm_comm import _runtime as runtime
from tirx_kernels.gemm_comm import allgather_gemm, gemm_reduce_scatter
from tirx_kernels.registry import discover_kernels

ag_impl = allgather_gemm
rs_impl = gemm_reduce_scatter


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
            "scheduler": scheduler,
            "label": f"tp4_m8192_n65536_k8192_fp16_{scheduler}",
        }
        for scheduler in ("static", "dynamic")
    ]
    assert gemm_reduce_scatter.CONFIGS == [
        {
            "M": 16384,
            "N": 12288,
            "K": 49152,
            "world_size": 4,
            "dtype": "float16",
            "scheduler": "static",
            "label": "tp4_m16384_n12288_k49152_fp16_static",
        }
    ]


@pytest.mark.parametrize(
    "module, overrides",
    [
        (allgather_gemm, {"M": allgather_gemm.M + 1}),
        (allgather_gemm, {"world_size": allgather_gemm.WORLD_SIZE - 1}),
        (gemm_reduce_scatter, {"K": gemm_reduce_scatter.TOTAL_K + 1}),
        (gemm_reduce_scatter, {"dtype": "bfloat16"}),
    ],
)
def test_tuned_kernels_reject_other_configs(module, overrides) -> None:
    with pytest.raises(ValueError, match="supports only"):
        module.get_kernel(**overrides)


def test_allgather_dynamic_queue_has_exact_coverage() -> None:
    task_types, task_indices, heads, tails = allgather_gemm._queue_state()
    expected_tasks = allgather_gemm.GEMM_M_CLUSTERS * allgather_gemm.GEMM_N_CLUSTERS
    expected_indices = {
        (m_index, n_index)
        for m_index in range(allgather_gemm.GEMM_M_CLUSTERS)
        for n_index in range(allgather_gemm.GEMM_N_CLUSTERS)
    }

    assert task_types.shape == (allgather_gemm.WORLD_SIZE, allgather_gemm.CAPACITY)
    assert task_indices.shape == (
        allgather_gemm.WORLD_SIZE,
        allgather_gemm.CAPACITY,
        allgather_gemm.TASK_IDX_LEN,
    )
    np.testing.assert_array_equal(heads, 0)
    np.testing.assert_array_equal(tails, expected_tasks)
    for rank in range(allgather_gemm.WORLD_SIZE):
        np.testing.assert_array_equal(
            task_types[rank, :expected_tasks], allgather_gemm.TaskType.GEMM.value
        )
        assert set(map(tuple, task_indices[rank, :expected_tasks])) == expected_indices
        np.testing.assert_array_equal(task_types[rank, expected_tasks:], -1)


@pytest.mark.parametrize("module", [allgather_gemm, gemm_reduce_scatter])
def test_run_bench_delegates_to_rank_local_worker(module, monkeypatch: pytest.MonkeyPatch) -> None:
    kernel = object()
    captured = {}

    monkeypatch.setattr(module, "get_kernel", lambda *_args, **_kwargs: kernel)

    def fake_run_distributed(ir_module, **kwargs):
        captured.update(ir_module=ir_module, **kwargs)
        return {"status": "OK", "timer": "kineto"}

    monkeypatch.setattr(module, "run_distributed", fake_run_distributed)

    result = module.run_bench(timer="kineto", rounds=6, cooldown_s=0.25)

    assert result == {"status": "OK", "timer": "kineto"}
    assert captured["ir_module"] is kernel
    assert captured["world_size"] == 4
    assert captured["worker"] is module._run_worker
    assert captured["mode"] == "bench"
    assert captured["worker_kwargs"]["timer"] == "kineto"
    assert captured["worker_kwargs"]["rounds"] == 6
    assert captured["worker_kwargs"]["cooldown_s"] == 0.25


@pytest.mark.parametrize("module", [allgather_gemm, gemm_reduce_scatter])
def test_worker_pairs_tirx_with_shared_kineto_baselines(module) -> None:
    source = inspect.getsource(module._run_worker)
    assert "create_baseline_suite" in source
    assert "references=baselines.references()" in source
    assert 'prepare={"tirx": prepare}' in source
    assert 'result["ratio_definition"] = "baseline_us / tirx_us"' in source


def test_baseline_problem_and_ratio_contract() -> None:
    problem = baselines._problem("allgather_gemm", M=8192, N=65536, K=8192, world_size=4)
    assert (problem.local_m, problem.local_n, problem.local_k) == (2048, 16384, 2048)
    assert baselines.ratios(
        {"impls": {"tirx": 100.0, "cublas_nccl": 125.0, "cublasmp_split_p2p": 80.0}}
    ) == {"cublas_nccl": 1.25, "cublasmp_split_p2p": 0.8}


def test_baseline_references_have_stable_names() -> None:
    suite = object.__new__(baselines.GemmCommBaselineSuite)
    suite.cublasmp_algo = "split_p2p"
    references = suite.references()
    assert list(references) == ["cublas_nccl", "cublasmp_split_p2p"]


@pytest.mark.parametrize(
    "workload, expected_calls",
    [
        ("allgather_gemm", ["all_gather", "block_current_stream", "gemm"]),
        ("gemm_reduce_scatter", ["gemm", "reduce_scatter", "block_current_stream"]),
    ],
)
def test_cublas_nccl_joins_collective_to_timing_stream(
    workload, expected_calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class FakeWork:
        def block_current_stream(self):
            calls.append("block_current_stream")

        def wait(self):
            raise AssertionError("wait() does not join the timing stream")

    launch = object.__new__(baselines._CublasNcclLaunch)
    launch.runtime = SimpleNamespace(timing_stream=object())
    launch.problem = SimpleNamespace(workload=workload)
    launch.input = object()
    launch.weight = SimpleNamespace(T=object())
    launch.collective_buffer = object()
    launch.output = object()

    monkeypatch.setattr(baselines.torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(baselines.torch, "mm", lambda *_args, **_kwargs: calls.append("gemm"))
    monkeypatch.setattr(
        baselines.dist,
        "all_gather_into_tensor",
        lambda *_args, **_kwargs: (calls.append("all_gather"), FakeWork())[1],
    )
    monkeypatch.setattr(
        baselines.dist,
        "reduce_scatter_tensor",
        lambda *_args, **_kwargs: (calls.append("reduce_scatter"), FakeWork())[1],
    )

    launch.launch()

    assert calls == expected_calls


def test_rank_library_preload_is_scoped(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    libraries = []
    for name, env_name in runtime._LOCKED_LIBRARY_ENVS.items():
        library = tmp_path / f"lib{name}.so"
        library.touch()
        libraries.append(str(library))
        monkeypatch.setenv(env_name, str(library))
    monkeypatch.setenv("LD_PRELOAD", "/existing/libdependency.so")

    with runtime._rank_library_preload(required=True):
        assert runtime.os.environ["LD_PRELOAD"] == ":".join(
            [*libraries, "/existing/libdependency.so"]
        )

    assert runtime.os.environ["LD_PRELOAD"] == "/existing/libdependency.so"


def test_benchmark_library_locks_are_required(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_name in runtime._LOCKED_LIBRARY_ENVS.values():
        monkeypatch.delenv(env_name, raising=False)

    with pytest.raises(RuntimeError, match="explicit library locks"):
        runtime._locked_library_paths(required=True)


def test_library_provenance_records_loaded_versions_and_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    paths = {}
    symbol_paths = {}
    for name, env_name in runtime._LOCKED_LIBRARY_ENVS.items():
        path = tmp_path / f"lib{name}.so"
        path.touch()
        paths[name] = path
        symbol_paths[baselines._LIBRARY_SYMBOLS[name]] = path
        monkeypatch.setenv(env_name, str(path))

    monkeypatch.setattr(baselines, "_symbol_library_path", symbol_paths.__getitem__)
    monkeypatch.setattr(baselines, "_read_cublas_version", lambda: ("13.2.0", 130200))
    monkeypatch.setattr(baselines, "_read_nvshmem_version", lambda: ("3.4.5", "1.3"))
    monkeypatch.setattr(
        baselines,
        "_read_encoded_version",
        lambda symbol: {
            "ncclGetVersion": ("2.28.3", 22803),
            "cublasMpGetVersion": ("0.10.0", 1000),
        }[symbol],
    )

    assert baselines._library_provenance() == {
        "nccl": {"path": str(paths["nccl"]), "version": "2.28.3", "version_raw": 22803},
        "cublas": {"path": str(paths["cublas"]), "version": "13.2.0", "version_raw": 130200},
        "cublasmp": {"path": str(paths["cublasmp"]), "version": "0.10.0", "version_raw": 1000},
        "nvshmem": {"path": str(paths["nvshmem"]), "version": "3.4.5", "api_version": "1.3"},
    }


def test_parent_compiles_once_before_spawning_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_calls = []
    spawn_calls = []

    class FakeQueue:
        value = None

        def put(self, value):
            self.value = value

        def get(self):
            return self.value

    queue = FakeQueue()
    executable = SimpleNamespace(export_library=lambda path: compile_calls.append(("export", path)))

    def fake_compile(*args, **kwargs):
        compile_calls.append(("compile", args, kwargs))
        return executable

    monkeypatch.setattr(runtime, "require_sm100", lambda _world_size: None)
    monkeypatch.setattr(runtime, "_find_free_port", lambda: 12345)
    monkeypatch.setattr(runtime.tvm, "compile", fake_compile)
    monkeypatch.setattr(
        runtime.mp, "get_context", lambda _method: SimpleNamespace(SimpleQueue=lambda: queue)
    )

    def fake_spawn(target, args, nprocs, join):
        spawn_calls.append((target, args, nprocs, join))
        args[-1].put({"status": "OK"})

    monkeypatch.setattr(runtime.mp, "spawn", fake_spawn)

    result = runtime.run_distributed(
        object(),
        world_size=4,
        worker=lambda *_args: {"status": "OK"},
        mode="test",
        worker_kwargs={},
    )

    assert result == {"status": "OK"}
    assert [call[0] for call in compile_calls] == ["compile", "export"]
    assert len(spawn_calls) == 1
    assert spawn_calls[0][0] is runtime._rank_entry
    assert spawn_calls[0][2:] == (4, True)


def test_gemm_comm_runtime_has_no_disco_objects_or_private_timer() -> None:
    source = inspect.getsource(runtime)
    for forbidden in (
        "tvm.runtime.disco",
        "ProcessSession",
        "DModule",
        "DRef",
        "runtime.timer.",
        "debug_get_from_remote",
        "debug_copy_from",
    ):
        assert forbidden not in source

    # Apache TVM keeps these historical registry names even without a Disco session.
    assert '"runtime.disco.nvshmem.empty"' in source
