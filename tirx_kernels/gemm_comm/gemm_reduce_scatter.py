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

"""Tensor-parallel FP16 GEMM + ReduceScatter for SM100."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _gemm_reduce_scatter_impl as impl
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
    sync_communication_to_compute,
    sync_compute_to_communication,
)

KERNEL_META = {"name": "gemm_reduce_scatter", "category": "gemm_comm", "compute_capability": 10}

CONFIGS = [
    {
        "M": impl.M,
        "N": impl.N,
        "K": impl.TOTAL_K,
        "world_size": impl.WORLD_SIZE,
        "dtype": "float16",
        "label": f"tp{impl.WORLD_SIZE}_m{impl.M}_n{impl.N}_k{impl.TOTAL_K}_fp16",
    }
]


def _check_config(M: int, N: int, K: int, world_size: int, dtype: str) -> None:
    expected = (impl.M, impl.N, impl.TOTAL_K, impl.WORLD_SIZE, "float16")
    actual = (M, N, K, world_size, dtype)
    if actual != expected:
        raise ValueError(f"this tuned kernel supports only {expected}, got {actual}")


def get_kernel(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.TOTAL_K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    **_kwargs: Any,
):
    _check_config(M, N, K, world_size, dtype)
    return impl.ReduceScatter


def prepare_data(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.TOTAL_K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    seed: int = 42,
    scale: float = 0.02,
    **_kwargs: Any,
) -> dict[str, np.ndarray]:
    """Create deterministic rank-local K shards on the host."""

    _check_config(M, N, K, world_size, dtype)
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((world_size, M, impl.K)) * scale).astype(dtype)
    B = (rng.standard_normal((world_size, N, impl.K)) * scale).astype(dtype)
    return {"A": A, "B": B}


@dataclass
class _Case:
    runtime: DistributedRuntime
    module: Any
    A: Any
    B: Any
    gemm_out: Any
    semaphore: Any
    staging: Any
    out: Any
    profiler: Any

    def reset(self) -> None:
        fill_ranked(self.semaphore, impl.WORLD_SIZE, (impl.WORLD_SIZE,), "uint64")

    def prepare(self) -> None:
        barrier_on_compute_stream(self.runtime)
        sync_compute_to_communication(self.runtime)

    def launch(self) -> None:
        self.module["test_mma_ss_tma_2sm_persistent"](
            self.A, self.B, self.gemm_out, self.semaphore, self.out, self.profiler
        )
        self.runtime.session.get_global_func("runtime.disco.transfer_to_peers_reduce_scatter")(
            self.semaphore,
            self.gemm_out,
            self.staging,
            self.runtime.communication_stream,
            impl.M,
            impl.N,
            impl.BLK_M,
            impl.BLK_N,
            impl.WORLD_SIZE,
        )
        self.runtime.session.get_global_func("runtime.disco.nvshmem.barrier_all_on_stream")(
            self.runtime.communication_stream
        )
        sync_communication_to_compute(self.runtime)
        self.module["reduce_sum"](self.staging, self.out)


def _allocate_case(runtime: DistributedRuntime, module: Any, data: dict[str, np.ndarray]) -> _Case:
    session = runtime.session
    A = session.empty((impl.M, impl.K), impl.a_type)
    B = session.empty((impl.N, impl.K), impl.b_type)
    copy_ranked(A, data["A"])
    copy_ranked(B, data["B"])

    case = _Case(
        runtime=runtime,
        module=module,
        A=A,
        B=B,
        gemm_out=symmetric_empty(session, (impl.M, impl.N), impl.d_type),
        semaphore=symmetric_empty(session, (impl.WORLD_SIZE,), "uint64"),
        staging=symmetric_empty(session, (impl.WORLD_SIZE, impl.LOCAL_M, impl.N), impl.d_type),
        out=session.empty((impl.LOCAL_M, impl.N), impl.d_type),
        profiler=session.empty((impl.PROFILER_BUFFER_SIZE,), "uint64"),
    )
    case.reset()
    session._sync_all()
    return case


def _check_correctness(case: _Case, data: dict[str, np.ndarray]) -> None:
    import torch

    total = torch.zeros((impl.M, impl.N), dtype=torch.float32, device="cuda:0")
    for rank in range(impl.WORLD_SIZE):
        A = torch.from_numpy(data["A"][rank]).cuda(0)
        B = torch.from_numpy(data["B"][rank]).cuda(0)
        partial = torch.matmul(A, B.T)
        gemm_result = torch.from_numpy(case.gemm_out.debug_get_from_remote(rank).numpy()).cuda(0)
        torch.testing.assert_close(gemm_result, partial, rtol=1e-3, atol=2e-2)
        total.add_(partial.float())

    expected = total.to(torch.float16)
    for rank in range(impl.WORLD_SIZE):
        begin = rank * impl.LOCAL_M
        end = begin + impl.LOCAL_M
        result = torch.from_numpy(case.out.debug_get_from_remote(rank).numpy()).cuda(0)
        torch.testing.assert_close(result, expected[begin:end], rtol=6e-2, atol=6e-2)


def run_test(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.TOTAL_K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    seed: int = 42,
    **_kwargs: Any,
) -> None:
    """Compile, launch on four GPUs, and validate GEMM and ReduceScatter."""

    _check_config(M, N, K, world_size, dtype)
    data = prepare_data(M, N, K, world_size, dtype, seed=seed)
    with create_runtime(world_size) as runtime:
        with load_module(runtime.session, get_kernel(M, N, K, world_size, dtype)) as module:
            case = _allocate_case(runtime, module, data)
            case.prepare()
            case.launch()
            runtime.session._sync_all()
            _check_correctness(case, data)


def run_bench(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.TOTAL_K,
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
    **_kwargs: Any,
) -> dict[str, Any]:
    """Benchmark the full GEMM + transfer + reduction pipeline."""

    _check_config(M, N, K, world_size, dtype)
    data = prepare_data(M, N, K, world_size, dtype)
    with create_runtime(world_size) as runtime:
        with load_module(runtime.session, get_kernel(M, N, K, world_size, dtype)) as module:
            case = _allocate_case(runtime, module, data)
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
            "gemm_reduce_scatter",
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
