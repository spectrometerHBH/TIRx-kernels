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

"""Rank-local runner for the DSL-lowered AllGather + FP16 GEMM."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

import tvm
from tvm.tirx import PrimFunc

from . import allgather_gemm as impl
from ._runtime import (
    DistributedRuntime,
    barrier_on_compute_stream,
    run_distributed,
    symmetric_empty,
    sync_communication_to_compute,
    sync_compute_to_communication,
    torch_view,
)

KERNEL_META = {"name": "allgather_gemm", "category": "gemm_comm", "compute_capability": 10}
_SUPPORTED_SCHEDULERS = ("static", "dynamic")
_DSL_PREFIX = "dsl"
_MANUAL_PREFIX = "manual"

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


def _prefix_modules(modules: dict[str, Any]) -> tvm.IRModule:
    functions = {}
    for prefix, module in modules.items():
        if isinstance(module, PrimFunc):
            entries = [(str(module.attrs["global_symbol"]), module)]
        elif isinstance(module, tvm.IRModule):
            entries = [
                (global_var.name_hint, function)
                for global_var, function in module.functions.items()
            ]
        else:
            raise TypeError(f"expected PrimFunc or IRModule, got {type(module).__name__}")
        for original_name, function in entries:
            name = f"{prefix}_{original_name}"
            if isinstance(function, PrimFunc):
                function = function.with_attr("global_symbol", name)
            functions[name] = function
    return tvm.IRModule(functions)


def _get_benchmark_kernel(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    scheduler: str = "dynamic",
) -> tvm.IRModule:
    return _prefix_modules(
        {
            _DSL_PREFIX: get_kernel(M, N, K, world_size, dtype, scheduler=scheduler),
            _MANUAL_PREFIX: _get_manual_oracle_kernel(scheduler),
        }
    )


def prepare_data(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    seed: int = 42,
    scale: float = 0.05,
    rank: int = 0,
    **_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Create deterministic inputs directly on one rank's CUDA device."""

    _check_config(M, N, K, world_size, dtype)
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    device = torch.device("cuda", rank)
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    A = torch.randn(
        (impl.LOCAL_M, K), dtype=torch.float16, device=device, generator=generator
    ).mul_(scale)
    B = torch.randn(
        (impl.LOCAL_N, K), dtype=torch.float16, device=device, generator=generator
    ).mul_(scale)
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
    entrypoint_prefix: str | None
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
    initial_task_types: Any
    initial_task_idxs: Any
    initial_tail: Any
    ag_out_torch: torch.Tensor
    semaphore_torch: torch.Tensor
    plan: Any

    def reset(self) -> None:
        self.semaphore_torch.zero_()
        self.task_types.copy_(self.initial_task_types)
        self.task_idxs.copy_(self.initial_task_idxs)
        self.head.zero_()
        self.tail.copy_(self.initial_tail)

    def prepare(self) -> None:
        barrier_on_compute_stream(self.runtime)
        sync_compute_to_communication(self.runtime)

    def _entrypoint(self, name: str) -> str:
        if self.entrypoint_prefix is None:
            return name
        return f"{self.entrypoint_prefix}_{name}"

    def launch(self) -> None:
        from .dsl import GemmCommHostExecutor, GemmCommRuntimeBindings

        def launch_host(name: str) -> None:
            if name != impl.ALLGATHER_HOST_ENTRYPOINT:
                raise ValueError(f"unsupported AllGather host region {name!r}")
            tvm.get_global_func(name)(
                self.semaphore,
                self.A,
                self.ag_out,
                self.runtime.communication_stream,
                impl.M,
                impl.K,
                impl.WORLD_SIZE,
            )

        def launch_device(name: str) -> None:
            if name != impl.GEMM_DEVICE_ENTRYPOINT:
                raise ValueError(f"unsupported AllGather device region {name!r}")
            self.module[self._entrypoint(name)](
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

        def unexpected_completion() -> None:
            raise ValueError("AllGather execution plan has no completion edge")

        GemmCommHostExecutor(
            GemmCommRuntimeBindings(
                launch_device=launch_device,
                launch_host=launch_host,
                communication_barrier=unexpected_completion,
                communication_to_compute_sync=unexpected_completion,
            )
        ).execute(self.plan)
        sync_communication_to_compute(self.runtime)


def _allocate_case(
    runtime: DistributedRuntime,
    module: Any,
    data: dict[str, torch.Tensor],
    scheduler: str,
    entrypoint_prefix: str | None,
) -> _Case:
    from .dsl import build_allgather_gemm_graph, policy_for_scheduler

    task_types, task_idxs, heads, tails = _queue_state()
    device = torch.device("cuda", runtime.rank)
    ag_out = symmetric_empty(runtime, (impl.M, impl.K), impl.a_type)
    semaphore = symmetric_empty(runtime, (impl.WORLD_SIZE,), "uint64")
    initial_task_types = torch.from_numpy(task_types[runtime.rank].copy()).to(device)
    initial_task_idxs = torch.from_numpy(task_idxs[runtime.rank].copy()).to(device)
    initial_tail = torch.from_numpy(tails[runtime.rank].copy()).to(device)
    case = _Case(
        runtime=runtime,
        module=module,
        entrypoint_prefix=entrypoint_prefix,
        A=data["A"],
        B=data["B"],
        ag_out=ag_out,
        semaphore=semaphore,
        out=torch.empty((impl.M, impl.LOCAL_N), dtype=torch.float16, device=device),
        profiler=torch.empty(impl.PROFILER_BUFFER_SIZE, dtype=torch.uint64, device=device),
        task_types=torch.empty_like(initial_task_types),
        task_idxs=torch.empty_like(initial_task_idxs),
        head=torch.from_numpy(heads[runtime.rank].copy()).to(device),
        tail=torch.empty_like(initial_tail),
        initial_task_types=initial_task_types,
        initial_task_idxs=initial_task_idxs,
        initial_tail=initial_tail,
        ag_out_torch=torch_view(ag_out),
        semaphore_torch=torch_view(semaphore),
        plan=policy_for_scheduler(scheduler).normalize(build_allgather_gemm_graph()),
    )
    with torch.cuda.stream(runtime.timing_stream):
        case.reset()
    torch.cuda.synchronize(runtime.rank)
    runtime.barrier()
    return case


def _check_correctness(case: _Case) -> None:
    gathered_A = torch.empty((impl.M, impl.K), dtype=torch.float16, device=case.A.device)
    with torch.cuda.stream(case.runtime.timing_stream):
        dist.all_gather_into_tensor(gathered_A, case.A)
    case.runtime.timing_stream.synchronize()

    local = slice(case.runtime.rank * impl.LOCAL_M, (case.runtime.rank + 1) * impl.LOCAL_M)
    case.ag_out_torch[local].copy_(case.A)
    torch.testing.assert_close(case.ag_out_torch, gathered_A, rtol=0, atol=0)

    reference = torch.matmul(gathered_A, case.B.T)
    torch.testing.assert_close(case.out, reference, rtol=1e-3, atol=2e-2)


def _run_worker(
    runtime: DistributedRuntime, module: Any, mode: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    scheduler = str(kwargs.get("scheduler", "dynamic"))
    data = prepare_data(rank=runtime.rank, **kwargs)

    if mode == "test":
        case = _allocate_case(runtime, module, data, scheduler, None)
        with torch.cuda.stream(runtime.timing_stream):
            case.prepare()
            case.launch()
        runtime.device.sync(runtime.compute_stream)
        _check_correctness(case)
        return {"status": "OK"}
    if mode != "bench":
        raise ValueError(f"unsupported distributed worker mode {mode!r}")

    from tvm.tirx.bench import bench

    dsl_case = _allocate_case(runtime, module, data, scheduler, _DSL_PREFIX)
    manual_case = _allocate_case(runtime, module, data, scheduler, _MANUAL_PREFIX)

    def prepare(case: _Case) -> None:
        case.reset()
        case.prepare()

    result = bench(
        {"dsl": dsl_case.launch, "manual": manual_case.launch},
        warmup=kwargs.get("warmup"),
        repeat=kwargs.get("repeat"),
        timer=kwargs.get("timer"),
        rounds=kwargs.get("rounds", 1),
        cooldown_s=kwargs.get("cooldown_s", 1.0),
        distributed=runtime.bench_context(),
        prepare={"dsl": lambda: prepare(dsl_case), "manual": lambda: prepare(manual_case)},
    )
    return {"status": "OK", **result}


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
    """Launch the DSL-lowered kernel on four GPUs and validate its result."""

    _check_config(M, N, K, world_size, dtype)
    _check_scheduler(scheduler)
    run_distributed(
        get_kernel(M, N, K, world_size, dtype, scheduler=scheduler),
        world_size=world_size,
        worker=_run_worker,
        mode="test",
        worker_kwargs={
            "M": M,
            "N": N,
            "K": K,
            "world_size": world_size,
            "dtype": dtype,
            "seed": seed,
            "scheduler": scheduler,
        },
    )


def run_bench(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    scheduler: str = "dynamic",
    **_kwargs: Any,
) -> dict[str, Any]:
    """Pair the DSL and manual paths with the shared distributed Kineto timer."""

    _check_config(M, N, K, world_size, dtype)
    _check_scheduler(scheduler)
    return run_distributed(
        _get_benchmark_kernel(M, N, K, world_size, dtype, scheduler),
        world_size=world_size,
        worker=_run_worker,
        mode="bench",
        worker_kwargs={
            "M": M,
            "N": N,
            "K": K,
            "world_size": world_size,
            "dtype": dtype,
            "scheduler": scheduler,
            "warmup": warmup,
            "repeat": repeat,
            "timer": timer,
            "rounds": rounds,
            "cooldown_s": cooldown_s,
        },
    )


__all__ = ["CONFIGS", "KERNEL_META", "get_kernel", "prepare_data", "run_bench", "run_test"]
