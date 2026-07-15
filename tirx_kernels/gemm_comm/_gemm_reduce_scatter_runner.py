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

"""Rank-local runner for the DSL-lowered FP16 GEMM + ReduceScatter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

import tvm
from tvm.tirx import PrimFunc

from . import gemm_reduce_scatter as impl
from ._runtime import (
    DistributedRuntime,
    barrier_on_communication_stream,
    barrier_on_compute_stream,
    run_distributed,
    symmetric_empty,
    sync_communication_to_compute,
    sync_compute_to_communication,
    torch_view,
)

KERNEL_META = {"name": "gemm_reduce_scatter", "category": "gemm_comm", "compute_capability": 10}
_DSL_PREFIX = "dsl"
_MANUAL_PREFIX = "manual"

CONFIGS = [
    {
        "M": impl.M,
        "N": impl.N,
        "K": impl.TOTAL_K,
        "world_size": impl.WORLD_SIZE,
        "dtype": "float16",
        "scheduler": "static",
        "label": f"tp{impl.WORLD_SIZE}_m{impl.M}_n{impl.N}_k{impl.TOTAL_K}_fp16_static",
    }
]


def _check_config(M: int, N: int, K: int, world_size: int, dtype: str) -> None:
    expected = (impl.M, impl.N, impl.TOTAL_K, impl.WORLD_SIZE, "float16")
    actual = (M, N, K, world_size, dtype)
    if actual != expected:
        raise ValueError(f"this tuned kernel supports only {expected}, got {actual}")


def _check_scheduler(scheduler: str) -> None:
    if scheduler not in {"static", "dynamic"}:
        raise ValueError(f"unsupported GEMM+ReduceScatter scheduler: {scheduler!r}")


def get_kernel(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.TOTAL_K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    scheduler: str = "static",
    **_kwargs: Any,
):
    _check_config(M, N, K, world_size, dtype)
    _check_scheduler(scheduler)
    from .dsl import GemmCommLowerer, build_gemm_reduce_scatter_graph, policy_for_scheduler

    lowered = GemmCommLowerer(policy_for_scheduler(scheduler)).lower(
        build_gemm_reduce_scatter_graph()
    )
    return lowered.module


def _get_manual_oracle_kernel(scheduler: str):
    """Return the private pre-migration static oracle."""

    _check_scheduler(scheduler)
    if scheduler != "static":
        raise NotImplementedError("the implementation-preserving manual path is static only")
    return impl.ReduceScatter


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
    K: int = impl.TOTAL_K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    scheduler: str = "static",
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
    K: int = impl.TOTAL_K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    seed: int = 42,
    scale: float = 0.02,
    rank: int = 0,
    **_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Create deterministic rank-local K shards directly on CUDA."""

    _check_config(M, N, K, world_size, dtype)
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    device = torch.device("cuda", rank)
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    A = torch.randn((M, K // world_size), dtype=torch.float16, device=device, generator=generator)
    B = torch.randn((N, K // world_size), dtype=torch.float16, device=device, generator=generator)
    A.mul_(scale)
    B.mul_(scale)
    return {"A": A, "B": B}


@dataclass
class _Case:
    runtime: DistributedRuntime
    module: Any
    entrypoint_prefix: str | None
    A: Any
    B: Any
    gemm_out: Any
    semaphore: Any
    staging: Any
    out: Any
    profiler: Any
    gemm_out_torch: torch.Tensor
    semaphore_torch: torch.Tensor
    plan: Any

    def reset(self) -> None:
        self.semaphore_torch.zero_()

    def prepare(self) -> None:
        barrier_on_compute_stream(self.runtime)
        sync_compute_to_communication(self.runtime)

    def _entrypoint(self, name: str) -> str:
        if self.entrypoint_prefix is None:
            return name
        return f"{self.entrypoint_prefix}_{name}"

    def launch(self) -> None:
        from .dsl import GemmCommHostExecutor, GemmCommRuntimeBindings

        def launch_device(name: str) -> None:
            entrypoint = self._entrypoint(name)
            if name == impl.PARTIAL_GEMM_DEVICE_ENTRYPOINT:
                self.module[entrypoint](
                    self.A, self.B, self.gemm_out, self.semaphore, self.out, self.profiler
                )
            elif name == impl.REDUCE_SUM_DEVICE_ENTRYPOINT:
                self.module[entrypoint](self.staging, self.out)
            else:
                raise ValueError(f"unsupported ReduceScatter device region {name!r}")

        def launch_host(name: str) -> None:
            if name != impl.REDUCE_SCATTER_HOST_ENTRYPOINT:
                raise ValueError(f"unsupported ReduceScatter host region {name!r}")
            tvm.get_global_func(name)(
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

        GemmCommHostExecutor(
            GemmCommRuntimeBindings(
                launch_device=launch_device,
                launch_host=launch_host,
                communication_barrier=lambda: barrier_on_communication_stream(self.runtime),
                communication_to_compute_sync=lambda: sync_communication_to_compute(self.runtime),
            )
        ).execute(self.plan)


def _allocate_case(
    runtime: DistributedRuntime,
    module: Any,
    data: dict[str, torch.Tensor],
    scheduler: str,
    entrypoint_prefix: str | None,
) -> _Case:
    from .dsl import build_gemm_reduce_scatter_graph, policy_for_scheduler

    device = torch.device("cuda", runtime.rank)
    gemm_out = symmetric_empty(runtime, (impl.M, impl.N), impl.d_type)
    semaphore = symmetric_empty(runtime, (impl.WORLD_SIZE,), "uint64")
    case = _Case(
        runtime=runtime,
        module=module,
        entrypoint_prefix=entrypoint_prefix,
        A=data["A"],
        B=data["B"],
        gemm_out=gemm_out,
        semaphore=semaphore,
        staging=symmetric_empty(runtime, (impl.WORLD_SIZE, impl.LOCAL_M, impl.N), impl.d_type),
        out=torch.empty((impl.LOCAL_M, impl.N), dtype=torch.float16, device=device),
        profiler=torch.empty(impl.PROFILER_BUFFER_SIZE, dtype=torch.uint64, device=device),
        gemm_out_torch=torch_view(gemm_out),
        semaphore_torch=torch_view(semaphore),
        plan=policy_for_scheduler(scheduler).normalize(build_gemm_reduce_scatter_graph()),
    )
    with torch.cuda.stream(runtime.timing_stream):
        case.reset()
    torch.cuda.synchronize(runtime.rank)
    runtime.barrier()
    return case


def _check_correctness(case: _Case) -> None:
    partial = torch.matmul(case.A, case.B.T)
    torch.testing.assert_close(case.gemm_out_torch, partial, rtol=1e-3, atol=2e-2)

    reference = partial.float()
    dist.all_reduce(reference, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(case.runtime.rank)
    expected = reference.to(torch.float16)
    begin = case.runtime.rank * impl.LOCAL_M
    end = begin + impl.LOCAL_M
    torch.testing.assert_close(case.out, expected[begin:end], rtol=6e-2, atol=6e-2)


def _run_worker(
    runtime: DistributedRuntime, module: Any, mode: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    scheduler = str(kwargs.get("scheduler", "static"))
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
    K: int = impl.TOTAL_K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    seed: int = 42,
    scheduler: str = "static",
    **_kwargs: Any,
) -> None:
    """Launch the DSL-lowered pipeline on four GPUs and validate its result."""

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
    K: int = impl.TOTAL_K,
    world_size: int = impl.WORLD_SIZE,
    dtype: str = "float16",
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    scheduler: str = "static",
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
