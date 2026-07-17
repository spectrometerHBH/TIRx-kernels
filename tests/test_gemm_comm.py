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
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from tirx_kernels.gemm_comm import _baselines as baselines
from tirx_kernels.gemm_comm import _gemm_reduce_scatter_runner as rs_runner
from tirx_kernels.gemm_comm import _runtime as runtime
from tirx_kernels.gemm_comm import allgather_gemm, gemm_reduce_scatter
from tirx_kernels.gemm_comm import rs_gemm_multimem_dynamic as rs_kernel
from tirx_kernels.gemm_comm._model_shapes import (
    ALLGATHER_GEMM_MODEL_SHAPES,
    GEMM_RS_MODEL_SHAPES,
    SUPPORTED_WORLD_SIZES,
    make_configs,
)
from tirx_kernels.registry import discover_kernels

ag_runner = allgather_gemm


def test_gemm_comm_registry_entries() -> None:
    kernels = discover_kernels(category="gemm_comm")
    assert set(kernels) == {"allgather_gemm", "gemm_reduce_scatter"}
    assert all(module.KERNEL_META["compute_capability"] == 10 for module in kernels.values())


def test_model_shape_configs_cover_tp1_tp2_tp4() -> None:
    assert allgather_gemm.CONFIGS == make_configs(ALLGATHER_GEMM_MODEL_SHAPES)
    assert gemm_reduce_scatter.CONFIGS == make_configs(GEMM_RS_MODEL_SHAPES)
    for configs in (allgather_gemm.CONFIGS, gemm_reduce_scatter.CONFIGS):
        assert len(configs) == 8 * len(SUPPORTED_WORLD_SIZES) == 24
        assert {config["world_size"] for config in configs} == set(SUPPORTED_WORLD_SIZES)
        assert len({config["label"] for config in configs}) == len(configs)


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
    with pytest.raises(ValueError, match="does not support|supports world_size|supports only"):
        module.get_kernel(**overrides)


def test_allgather_config_derivation_covers_every_registered_shape() -> None:
    for raw in allgather_gemm.CONFIGS:
        config = allgather_gemm.derive_config(
            raw["M"], raw["N"], raw["K"], raw["world_size"], raw["dtype"]
        )
        assert config.local_m == config.M // config.world_size
        assert config.local_n == config.N // config.world_size
        assert config.group_size == config.local_gemm_m_clusters == 16 // config.world_size
        assert config.task_count == config.gemm_m_clusters * config.gemm_n_clusters
        assert config.capacity >= max(2048, config.task_count)
        assert config.capacity & (config.capacity - 1) == 0
        assert config.capacity <= 8192


def test_reduce_scatter_config_derivation_covers_every_registered_shape() -> None:
    for raw in gemm_reduce_scatter.CONFIGS:
        config = rs_kernel.derive_config(
            raw["M"], raw["N"], raw["K"], raw["world_size"], raw["dtype"]
        )
        assert config.local_m == config.M // config.world_size
        assert config.k_local == config.total_k // config.world_size
        assert config.gemm_task_count == config.gemm_m_clusters * config.gemm_n_clusters
        assert config.rs_task_count == config.rs_m_clusters * config.rs_n_clusters
        assert config.completion_count == 2 * config.world_size
        assert max(config.gemm_task_count, config.rs_task_count) <= rs_kernel.CAPACITY


def test_all_registered_kernels_build_shape_specialized_ir() -> None:
    for module in (allgather_gemm, gemm_reduce_scatter):
        for raw in module.CONFIGS:
            kwargs = {key: value for key, value in raw.items() if key != "label"}
            assert module.get_kernel(**kwargs) is not None


def test_reduce_scatter_manual_config_and_queue_geometry() -> None:
    config = rs_kernel.derive_config()
    assert (config.k_local, config.local_m) == (6400, 2048)
    assert (config.gemm_task_count, config.rs_task_count, config.completion_count) == (320, 160, 8)

    state = rs_runner._queue_state(config)
    gemm_types, gemm_indices, gemm_heads, gemm_tails = state[:4]
    rs_types, _rs_indices, rs_heads, rs_tails = state[4:]
    expected_order = [
        (m_idx, n_idx)
        for group_begin in range(0, config.gemm_n_clusters, rs_kernel.GROUP_SIZE)
        for m_idx in range(config.gemm_m_clusters)
        for n_idx in range(
            group_begin, min(group_begin + rs_kernel.GROUP_SIZE, config.gemm_n_clusters)
        )
    ]
    assert list(map(tuple, gemm_indices[0, : config.gemm_task_count])) == expected_order
    np.testing.assert_array_equal(gemm_types[:, : config.gemm_task_count], 0)
    np.testing.assert_array_equal(gemm_heads, 0)
    np.testing.assert_array_equal(gemm_tails, config.gemm_task_count)
    np.testing.assert_array_equal(rs_types, -1)
    np.testing.assert_array_equal(rs_heads, 0)
    np.testing.assert_array_equal(rs_tails, 0)


@pytest.mark.parametrize("kwargs", [{"world_size": 8}, {"scheduler": "static"}, {"M": 1}])
def test_reduce_scatter_manual_port_rejects_unsupported_configs(kwargs) -> None:
    with pytest.raises(ValueError):
        gemm_reduce_scatter.get_kernel(**kwargs)


def test_reduce_scatter_remote_queue_uses_system_release_acquire() -> None:
    assert "atomicAdd_system(remote_tail, 1)" in rs_kernel.enqueue_remote
    assert "st.global.release.sys.b32" in rs_kernel.enqueue_remote
    assert "ld.global.acquire.sys.b32" in rs_kernel.ld_global_acquire
    assert "ld.global.acquire.sys.b32" in rs_kernel.while_ld_global_acquire


def test_reduce_scatter_publishes_gemm_output_before_completion_atomic() -> None:
    module_source = inspect.getsource(rs_kernel)
    notify_start = module_source.index("    def semaphore_notify(")
    notify_source = module_source[
        notify_start : module_source.index("\n\n@Tx.prim_func", notify_start)
    ]
    assert "__threadfence_system();" in rs_kernel.thread_fence_system
    assert notify_source.index('"thread_fence_system"') < notify_source.index(
        '"semaphore_notify_remote"'
    )
    assert "Tx.cuda.thread_fence()" not in notify_source


def test_gemm_comm_static_geometry_guards() -> None:
    assert allgather_gemm.GROUP_SIZE == allgather_gemm.LOCAL_GEMM_M_CLUSTERS
    assert rs_kernel.SMEM_SIZE == 230400
    assert rs_kernel.SMEM_SIZE <= rs_kernel.SM100_SMEM_CAPACITY == 232448


def test_reduce_scatter_reset_joins_peer_writers_first(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []

    class TensorProbe:
        def __init__(self, name: str) -> None:
            self.name = name

        def copy_(self, _value) -> None:
            events.append(f"copy:{self.name}")

        def zero_(self) -> None:
            events.append(f"zero:{self.name}")

        def fill_(self, _value) -> None:
            events.append(f"fill:{self.name}")

    monkeypatch.setattr(rs_runner, "torch_view", lambda tensor: tensor)
    monkeypatch.setattr(
        rs_runner, "barrier_on_compute_stream", lambda runtime: events.append(f"barrier:{runtime}")
    )
    case = SimpleNamespace(
        runtime="runtime",
        initial_queues=(object(),) * 8,
        gemm_task_types=TensorProbe("gemm_types"),
        gemm_task_idxs=TensorProbe("gemm_indices"),
        gemm_head_torch=TensorProbe("gemm_head"),
        gemm_tail_torch=TensorProbe("gemm_tail"),
        rs_task_types=TensorProbe("rs_types"),
        rs_task_idxs=TensorProbe("rs_indices"),
        rs_head_torch=TensorProbe("rs_head"),
        rs_tail_torch=TensorProbe("rs_tail"),
        semaphore_torch=TensorProbe("semaphore"),
        gemm_out_torch=TensorProbe("gemm_out"),
        out=TensorProbe("out"),
    )

    rs_runner._Case.reset(case)

    assert events[0] == "barrier:runtime"
    assert events[1:] == [
        "copy:gemm_types",
        "copy:gemm_indices",
        "copy:gemm_head",
        "copy:gemm_tail",
        "copy:rs_types",
        "copy:rs_indices",
        "copy:rs_head",
        "copy:rs_tail",
        "zero:semaphore",
        "fill:gemm_out",
        "fill:out",
    ]


def test_reduce_scatter_manual_kernel_has_no_legacy_profiler_argument() -> None:
    kernel = rs_kernel.build_kernel()
    func = kernel[rs_kernel.FUSED_DEVICE_ENTRYPOINT]
    assert len(func.params) == 13
    assert "profiler" not in str(func).lower()


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


def test_all_registered_queue_states_fit_and_cover_tiles() -> None:
    for raw in allgather_gemm.CONFIGS:
        config = allgather_gemm.derive_config(
            raw["M"], raw["N"], raw["K"], raw["world_size"], raw["dtype"]
        )
        task_types, task_indices, heads, tails = allgather_gemm._queue_state(config)
        assert task_types.shape == (config.world_size, config.capacity)
        assert task_indices.shape == (config.world_size, config.capacity, 2)
        np.testing.assert_array_equal(heads, 0)
        np.testing.assert_array_equal(tails, config.task_count)
        for rank in range(config.world_size):
            assert len(set(map(tuple, task_indices[rank, : config.task_count]))) == (
                config.task_count
            )

    for raw in gemm_reduce_scatter.CONFIGS:
        config = rs_kernel.derive_config(
            raw["M"], raw["N"], raw["K"], raw["world_size"], raw["dtype"]
        )
        state = rs_runner._queue_state(config)
        gemm_types, gemm_indices, gemm_heads, gemm_tails = state[:4]
        rs_types, _rs_indices, rs_heads, rs_tails = state[4:]
        np.testing.assert_array_equal(gemm_heads, 0)
        np.testing.assert_array_equal(gemm_tails, config.gemm_task_count)
        np.testing.assert_array_equal(rs_types, -1)
        np.testing.assert_array_equal(rs_heads, 0)
        np.testing.assert_array_equal(rs_tails, 0)
        for rank in range(config.world_size):
            assert len(set(map(tuple, gemm_indices[rank, : config.gemm_task_count]))) == (
                config.gemm_task_count
            )
            np.testing.assert_array_equal(
                gemm_types[rank, : config.gemm_task_count], rs_kernel.TaskType.GEMM.value
            )


@pytest.mark.parametrize(
    "module, runner", [(allgather_gemm, ag_runner), (gemm_reduce_scatter, rs_runner)]
)
def test_run_bench_delegates_to_rank_local_worker(
    module, runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel = object()
    captured = {}

    monkeypatch.setattr(runner, "_get_benchmark_kernel", lambda *_args, **_kwargs: kernel)

    def fake_run_distributed(ir_module, **kwargs):
        captured.update(ir_module=ir_module, **kwargs)
        return {"status": "OK", "timer": "event"}

    monkeypatch.setattr(runner, "run_distributed", fake_run_distributed)

    result = module.run_bench(timer="event", rounds=6, cooldown_s=0.25)

    assert result == {"status": "OK", "timer": "event"}
    assert captured["ir_module"] is kernel
    assert captured["world_size"] == 4
    assert captured["worker"] is runner._run_worker
    assert captured["mode"] == "bench"
    assert captured["worker_kwargs"]["timer"] == "event"
    assert captured["worker_kwargs"]["rounds"] == 6
    assert captured["worker_kwargs"]["cooldown_s"] == 0.25


@pytest.mark.parametrize("kwargs", [{"warmup": 1}, {"repeat": 1}])
@pytest.mark.parametrize("module", [allgather_gemm, gemm_reduce_scatter])
def test_distributed_event_bench_rejects_iteration_overrides(module, kwargs) -> None:
    with pytest.raises(ValueError, match="fixed iteration counts"):
        module.run_bench(timer="event", **kwargs)


@pytest.mark.parametrize("runner", [ag_runner, rs_runner])
def test_worker_pairs_event_baselines_and_named_kernel_diagnostic(runner) -> None:
    source = inspect.getsource(runner._run_worker)
    assert "create_baseline_suite" in source
    assert "references=baselines.references()" in source
    assert 'prepare={"tirx":' in source
    assert 'timer="event"' in source
    assert 'timer="kineto"' in source
    assert "kineto_kernel_names" in source
    assert 'result["ratio_definition"] = "baseline_us / tirx_us"' in source


def test_baseline_problem_and_ratio_contract() -> None:
    problem = baselines._problem("allgather_gemm", M=8192, N=65536, K=8192, world_size=4)
    assert (problem.local_m, problem.local_n, problem.local_k) == (2048, 16384, 2048)
    assert baselines.ratios(
        {"impls": {"tirx": 100.0, "cublas_nccl_cudagraph": 125.0, "cublasmp_split_p2p": 80.0}}
    ) == {"cublas_nccl_cudagraph": 1.25, "cublasmp_split_p2p": 0.8}


def test_baseline_references_have_stable_names() -> None:
    suite = object.__new__(baselines.GemmCommBaselineSuite)
    suite.cublasmp_algo = "split_p2p"
    references = suite.references()
    assert list(references) == ["cublas_nccl_cudagraph", "cublasmp_split_p2p"]


def test_cublas_nccl_capture_keeps_setup_outside_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class FakeStream:
        def synchronize(self):
            calls.append("synchronize")

    class FakeRuntime:
        timing_stream = FakeStream()

        @staticmethod
        def barrier():
            calls.append("barrier")

    class FakeGraph:
        def replay(self):
            calls.append("replay")

    @contextmanager
    def fake_capture(graph, *, stream):
        assert isinstance(graph, FakeGraph)
        assert stream is FakeRuntime.timing_stream
        calls.append("capture_begin")
        yield
        calls.append("capture_end")

    launch = object.__new__(baselines._CublasNcclLaunch)
    launch.runtime = FakeRuntime()
    launch._graph = None
    launch.launch = lambda: calls.append("launch")

    monkeypatch.setattr(baselines.torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(baselines.torch.cuda, "graph", fake_capture)
    monkeypatch.setattr(baselines.torch.cuda, "stream", lambda _stream: nullcontext())

    replay = launch.capture()

    assert calls == [
        "launch",
        "synchronize",
        "barrier",
        "capture_begin",
        "launch",
        "capture_end",
        "synchronize",
        "barrier",
        "replay",
        "synchronize",
        "barrier",
    ]
    replay()
    assert calls[-1] == "replay"


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


@pytest.mark.parametrize("runner", [ag_runner, rs_runner])
def test_bench_uses_one_direct_tirx_case(runner) -> None:
    source = inspect.getsource(runner._run_worker)
    assert '{"tirx": case.launch}' in source
    assert 'prepare={"tirx":' in source
    assert '"dsl"' not in source
    assert '"manual"' not in source


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
    assert spawn_calls[0][1][1].startswith("file://")


def test_rank_process_group_has_explicit_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class InitStopped(RuntimeError):
        pass

    def stop_after_init_call(**kwargs):
        captured.update(kwargs)
        raise InitStopped

    monkeypatch.setattr(runtime.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(runtime.dist, "init_process_group", stop_after_init_call)
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: False)

    with pytest.raises(InitStopped):
        runtime._rank_entry(0, 4, "tcp://127.0.0.1:12345", "kernel.so", None, "test", {}, None)

    assert captured["timeout"].total_seconds() == 60


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
