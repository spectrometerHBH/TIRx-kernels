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

"""Tensor-parallel AllGather + FP16 GEMM for SM100."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import allgather_gemm as impl
from ._baselines import run_external_baselines
from ._runtime import (
    DistributedRuntime,
    barrier_on_compute_stream,
    benchmark_slowest_rank,
    copy_ranked,
    create_runtime,
    fill_ranked,
    load_module,
    symmetric_empty,
    sync_compute_to_communication,
)

KERNEL_META = {"name": "allgather_gemm", "category": "gemm_comm", "compute_capability": 10}
_SUPPORTED_SCHEDULERS = ("static", "dynamic")

CONFIGS = [
    {
        "M": impl.M,
        "N": impl.N,
        "K": impl.K,
        "world_size": impl.WORLD_SIZE,
        "dtype": "float16",
        "scheduler": scheduler,
        "label": f"tp{impl.WORLD_SIZE}_m{impl.M}_n{impl.N}_k{impl.K}_fp16_{scheduler}",
    }
    for scheduler in _SUPPORTED_SCHEDULERS
]


def _check_config(M: int, N: int, K: int, world_size: int, dtype: str) -> None:
    expected = (impl.M, impl.N, impl.K, impl.WORLD_SIZE, "float16")
    actual = (M, N, K, world_size, dtype)
    if actual != expected:
        raise ValueError(f"this tuned kernel supports only {expected}, got {actual}")


def _check_scheduler(scheduler: str) -> None:
    if scheduler not in _SUPPORTED_SCHEDULERS:
        raise ValueError(f"unsupported AllGather+GEMM scheduler: {scheduler!r}")


def get_kernel(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    scheduler: str = "dynamic",
    **_kwargs: Any,
):
    _check_config(M, N, K, world_size, dtype)
    _check_scheduler(scheduler)
    from .dsl import GemmCommLowerer, build_allgather_gemm_graph, policy_for_scheduler

    lowered = GemmCommLowerer(policy_for_scheduler(scheduler)).lower(build_allgather_gemm_graph())
    return lowered.module


def _get_manual_oracle_kernel(scheduler: str):
    """Build the private pre-migration oracle for equivalence tests."""

    _check_scheduler(scheduler)
    return impl.build_kernel(scheduler)


def prepare_data(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    seed: int = 42,
    scale: float = 0.05,
    **_kwargs: Any,
) -> dict[str, np.ndarray]:
    """Create deterministic rank-local shards on the host."""

    _check_config(M, N, K, world_size, dtype)
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((world_size, impl.LOCAL_M, K)) * scale).astype(dtype)
    B = (rng.standard_normal((world_size, impl.LOCAL_N, K)) * scale).astype(dtype)
    return {"A": A, "B": B}


def _manual_queue_state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    task_types = np.full((impl.WORLD_SIZE, impl.CAPACITY), -1, dtype=np.int32)
    task_idxs = np.zeros((impl.WORLD_SIZE, impl.CAPACITY, impl.TASK_IDX_LEN), dtype=np.int32)
    heads = np.zeros((impl.WORLD_SIZE, 1), dtype=np.int32)
    tails = np.zeros((impl.WORLD_SIZE, 1), dtype=np.int32)

    for rank in range(impl.WORLD_SIZE):
        tasks: list[tuple[int, int]] = []
        offset = rank * impl.LOCAL_GEMM_M_CLUSTERS
        group_count = math.ceil(impl.GEMM_M_CLUSTERS / impl.GROUP_SIZE)
        for group in range(group_count):
            begin = group * impl.GROUP_SIZE
            end = min((group + 1) * impl.GROUP_SIZE, impl.GEMM_M_CLUSTERS)
            for n_idx in range(impl.GEMM_N_CLUSTERS):
                for m_idx in range(begin, end):
                    tasks.append(((offset + m_idx) % impl.GEMM_M_CLUSTERS, n_idx))
        if len(tasks) != impl.GEMM_M_CLUSTERS * impl.GEMM_N_CLUSTERS:
            raise AssertionError("incomplete AllGather+GEMM queue")
        if len(tasks) > impl.CAPACITY:
            raise AssertionError("AllGather+GEMM queue exceeds capacity")
        task_types[rank, : len(tasks)] = impl.TaskType.GEMM.value
        task_idxs[rank, : len(tasks)] = tasks
        tails[rank, 0] = len(tasks)
    return task_types, task_idxs, heads, tails


def _queue_state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from .dsl import build_allgather_gemm_graph, make_allgather_dynamic_queue, policy_for_scheduler

    plan = policy_for_scheduler("dynamic").normalize(build_allgather_gemm_graph())
    return make_allgather_dynamic_queue(plan)


@dataclass
class _Case:
    runtime: DistributedRuntime
    module: Any
    A: Any
    B: Any
    ag_out: Any
    semaphore: Any
    out: Any
    profiler: Any
    task_types: Any
    task_idxs: Any
    head: Any
    tail: Any
    plan: Any

    def reset(self) -> None:
        fill_ranked(self.semaphore, impl.WORLD_SIZE, (impl.WORLD_SIZE,), "uint64")
        task_types, task_idxs, heads, tails = _queue_state()
        copy_ranked(self.task_types, task_types)
        copy_ranked(self.task_idxs, task_idxs)
        copy_ranked(self.head, heads)
        copy_ranked(self.tail, tails)

    def prepare(self) -> None:
        barrier_on_compute_stream(self.runtime)
        sync_compute_to_communication(self.runtime)

    def launch(self) -> None:
        from .dsl import GemmCommHostExecutor, GemmCommRuntimeBindings

        def launch_host(name):
            if name != "runtime.disco.transfer_to_peers_all_gather":
                raise ValueError(f"unsupported AllGather host region {name!r}")
            self.runtime.session.get_global_func(name)(
                self.semaphore,
                self.A,
                self.ag_out,
                self.runtime.communication_stream,
                impl.M,
                impl.K,
                impl.WORLD_SIZE,
            )

        def launch_device(name):
            if name != "test_mma_ss_tma_2sm_persistent":
                raise ValueError(f"unsupported AllGather device region {name!r}")
            self.module[name](
                self.A,
                self.B,
                self.ag_out,
                self.semaphore,
                self.out,
                self.profiler,
                self.task_types,
                self.task_idxs,
                self.head,
                self.tail,
            )

        def unexpected_completion():
            raise ValueError("AllGather execution plan has no completion edge")

        GemmCommHostExecutor(
            GemmCommRuntimeBindings(
                launch_device=launch_device,
                launch_host=launch_host,
                communication_barrier=unexpected_completion,
                communication_to_compute_sync=unexpected_completion,
            )
        ).execute(self.plan)


def _allocate_case(
    runtime: DistributedRuntime, module: Any, data: dict[str, np.ndarray], scheduler: str
) -> _Case:
    from .dsl import build_allgather_gemm_graph, policy_for_scheduler

    session = runtime.session
    A = session.empty((impl.LOCAL_M, impl.K), impl.a_type)
    B = session.empty((impl.LOCAL_N, impl.K), impl.b_type)
    copy_ranked(A, data["A"])
    copy_ranked(B, data["B"])

    case = _Case(
        runtime=runtime,
        module=module,
        A=A,
        B=B,
        ag_out=symmetric_empty(session, (impl.M, impl.K), impl.a_type),
        semaphore=symmetric_empty(session, (impl.WORLD_SIZE,), "uint64"),
        out=session.empty((impl.M, impl.LOCAL_N), impl.d_type),
        profiler=session.empty((impl.PROFILER_BUFFER_SIZE,), "uint64"),
        task_types=session.empty((impl.CAPACITY,), "int32"),
        task_idxs=session.empty((impl.CAPACITY, impl.TASK_IDX_LEN), "int32"),
        head=session.empty((1,), "int32"),
        tail=session.empty((1,), "int32"),
        plan=policy_for_scheduler(scheduler).normalize(build_allgather_gemm_graph()),
    )
    case.reset()
    session._sync_all()
    return case


def _check_correctness(case: _Case, data: dict[str, np.ndarray]) -> None:
    import torch

    gathered_A = data["A"].reshape(impl.M, impl.K)
    gathered_ref = torch.from_numpy(gathered_A)
    A_cuda = gathered_ref.cuda(0)
    for rank in range(impl.WORLD_SIZE):
        ag_result = case.ag_out.debug_get_from_remote(rank).numpy()
        # The local shard is consumed directly from A and intentionally is not
        # copied into the symmetric AllGather workspace.
        local = slice(rank * impl.LOCAL_M, (rank + 1) * impl.LOCAL_M)
        ag_result[local] = gathered_A[local]
        np.testing.assert_array_equal(ag_result, gathered_A)

        B_cuda = torch.from_numpy(data["B"][rank]).cuda(0)
        reference = torch.matmul(A_cuda, B_cuda.T)
        result = torch.from_numpy(case.out.debug_get_from_remote(rank).numpy()).cuda(0)
        torch.testing.assert_close(result, reference, rtol=1e-3, atol=2e-2)


def run_test(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    seed: int = 42,
    scheduler: str = "dynamic",
    **_kwargs: Any,
) -> None:
    """Compile, launch on four GPUs, and compare the full result with PyTorch."""

    _check_config(M, N, K, world_size, dtype)
    _check_scheduler(scheduler)
    data = prepare_data(M, N, K, world_size, dtype, seed=seed)
    with create_runtime(world_size) as runtime:
        with load_module(
            runtime.session, get_kernel(M, N, K, world_size, dtype, scheduler=scheduler)
        ) as module:
            case = _allocate_case(runtime, module, data, scheduler)
            case.prepare()
            case.launch()
            runtime.session._sync_all()
            _check_correctness(case, data)


def run_bench(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    warmup: int | None = None,
    repeat: int | None = None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    baselines: bool = True,
    baseline_strict: bool = False,
    baseline_timeout: float = 900.0,
    cublasmp_algo: str = "split_p2p",
    scheduler: str = "dynamic",
    **_kwargs: Any,
) -> dict[str, Any]:
    """Benchmark the fused implementation using the slowest rank's CUDA event."""

    _check_config(M, N, K, world_size, dtype)
    _check_scheduler(scheduler)
    data = prepare_data(M, N, K, world_size, dtype)
    with create_runtime(world_size) as runtime:
        with load_module(
            runtime.session, get_kernel(M, N, K, world_size, dtype, scheduler=scheduler)
        ) as module:
            case = _allocate_case(runtime, module, data, scheduler)
            tirx_result = benchmark_slowest_rank(
                runtime.session,
                reset=case.reset,
                prepare=case.prepare,
                launch=case.launch,
                warmup=warmup,
                repeat=repeat,
                rounds=rounds,
                cooldown_s=cooldown_s,
            )

    result: dict[str, Any] = {
        "status": "OK",
        "impls": {"tirx": tirx_result["mean_us"]},
        "samples_us": {"tirx": tirx_result["samples_us"]},
        "round_samples": {"tirx": tirx_result["round_samples_us"]},
        "timing": {"tirx": tirx_result},
    }
    if baselines:
        baseline_result = run_external_baselines(
            "allgather_gemm",
            world_size=world_size,
            warmup=warmup,
            repeat=repeat,
            rounds=rounds,
            cooldown_s=cooldown_s,
            cublasmp_algo=cublasmp_algo,
            timeout=baseline_timeout,
            strict=baseline_strict,
        )
        result["baselines"] = baseline_result
        if baseline_result.get("status") == "OK":
            for name, measurement in baseline_result["implementations"].items():
                result["impls"][name] = measurement["mean_us"]
                result["samples_us"][name] = measurement["samples_us"]
                result["round_samples"][name] = measurement["round_samples_us"]
    return result


__all__ = ["CONFIGS", "KERNEL_META", "get_kernel", "prepare_data", "run_bench", "run_test"]
