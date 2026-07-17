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

"""Rank-local runner for the directly ported dynamic-multimem GemmRS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

import tvm

from . import rs_gemm_multimem_dynamic as impl
from ._baselines import create_baseline_suite
from ._baselines import ratios as baseline_ratios
from ._model_shapes import GEMM_RS_MODEL_SHAPES, make_configs
from ._runtime import (
    DistributedRuntime,
    barrier_on_compute_stream,
    require_nvls_multicast,
    run_distributed,
    symmetric_empty,
    torch_view,
)

KERNEL_META = {"name": "gemm_reduce_scatter", "category": "gemm_comm", "compute_capability": 10}
_RELAUNCH_COUNT = 20

CONFIGS = make_configs(GEMM_RS_MODEL_SHAPES)


def _config(
    M: int, N: int, K: int, world_size: int, dtype: str, scheduler: str
) -> impl.GemmRSConfig:
    if scheduler != "dynamic":
        raise ValueError(f"GEMM+ReduceScatter supports only scheduler='dynamic'; got {scheduler!r}")
    return impl.derive_config(M, N, K, world_size, dtype)


def get_kernel(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.TOTAL_K,
    world_size: int = 4,
    dtype: str = impl.DTYPE,
    scheduler: str = "dynamic",
    **_kwargs: Any,
) -> tvm.IRModule:
    """Build the hand-transcribed fused kernel directly, without the megakernel DSL."""

    config = _config(M, N, K, world_size, dtype, scheduler)
    return impl.build_kernel(config)


def _get_benchmark_kernel(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.TOTAL_K,
    world_size: int = 4,
    dtype: str = impl.DTYPE,
    scheduler: str = "dynamic",
) -> tvm.IRModule:
    return get_kernel(M, N, K, world_size, dtype, scheduler=scheduler)


def prepare_data(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.TOTAL_K,
    world_size: int = 4,
    dtype: str = impl.DTYPE,
    scheduler: str = "dynamic",
    seed: int = 42,
    scale: float = 0.02,
    rank: int = 0,
    **_kwargs: Any,
) -> dict[str, torch.Tensor]:
    config = _config(M, N, K, world_size, dtype, scheduler)
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    device = torch.device("cuda", rank)
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    A = torch.randn(
        (config.M, config.k_local), dtype=torch.float16, device=device, generator=generator
    ).mul_(scale)
    B = torch.randn(
        (config.N, config.k_local), dtype=torch.float16, device=device, generator=generator
    ).mul_(scale)
    return {"A": A, "B": B}


def _queue_state(config: impl.GemmRSConfig) -> tuple[np.ndarray, ...]:
    tasks = [
        (m_idx, n_idx)
        for group_begin in range(0, config.gemm_n_clusters, impl.GROUP_SIZE)
        for m_idx in range(config.gemm_m_clusters)
        for n_idx in range(group_begin, min(group_begin + impl.GROUP_SIZE, config.gemm_n_clusters))
    ]
    if len(tasks) != config.gemm_task_count:
        raise AssertionError("GEMM queue does not have exact group-major coverage")

    gemm_types = np.full((config.world_size, impl.CAPACITY), -1, dtype=np.int32)
    gemm_indices = np.zeros((config.world_size, impl.CAPACITY, impl.TASK_IDX_LEN), dtype=np.int32)
    gemm_heads = np.zeros((config.world_size, 1), dtype=np.int32)
    gemm_tails = np.full((config.world_size, 1), config.gemm_task_count, dtype=np.int32)
    rs_types = np.full((config.world_size, impl.CAPACITY), -1, dtype=np.int32)
    rs_indices = np.zeros((config.world_size, impl.CAPACITY, impl.TASK_IDX_LEN), dtype=np.int32)
    rs_heads = np.zeros((config.world_size, 1), dtype=np.int32)
    rs_tails = np.zeros((config.world_size, 1), dtype=np.int32)
    for rank in range(config.world_size):
        gemm_types[rank, : config.gemm_task_count] = impl.TaskType.GEMM.value
        gemm_indices[rank, : config.gemm_task_count] = tasks
    return (
        gemm_types,
        gemm_indices,
        gemm_heads,
        gemm_tails,
        rs_types,
        rs_indices,
        rs_heads,
        rs_tails,
    )


_manual_queue_state = _queue_state


@dataclass
class _Case:
    runtime: DistributedRuntime
    module: Any
    config: impl.GemmRSConfig
    A: torch.Tensor
    B: torch.Tensor
    gemm_out: Any
    semaphore: Any
    out: torch.Tensor
    gemm_task_types: Any
    gemm_task_idxs: Any
    gemm_head: Any
    gemm_tail: Any
    rs_task_types: Any
    rs_task_idxs: Any
    rs_head: Any
    rs_tail: Any
    gemm_out_torch: torch.Tensor
    semaphore_torch: torch.Tensor
    gemm_types_torch: torch.Tensor
    gemm_head_torch: torch.Tensor
    gemm_tail_torch: torch.Tensor
    rs_types_torch: torch.Tensor
    rs_head_torch: torch.Tensor
    rs_tail_torch: torch.Tensor
    initial_queues: tuple[torch.Tensor, ...]

    def reset(self) -> None:
        # A completed local launch does not imply that peer kernels have stopped
        # writing this rank's symmetric queue state.  Join every rank before
        # overwriting it; prepare() supplies the matching post-reset barrier.
        barrier_on_compute_stream(self.runtime)
        (
            gemm_types,
            gemm_indices,
            gemm_heads,
            gemm_tails,
            rs_types,
            rs_indices,
            rs_heads,
            rs_tails,
        ) = self.initial_queues
        torch_view(self.gemm_task_types).copy_(gemm_types)
        torch_view(self.gemm_task_idxs).copy_(gemm_indices)
        self.gemm_head_torch.copy_(gemm_heads)
        self.gemm_tail_torch.copy_(gemm_tails)
        torch_view(self.rs_task_types).copy_(rs_types)
        torch_view(self.rs_task_idxs).copy_(rs_indices)
        self.rs_head_torch.copy_(rs_heads)
        self.rs_tail_torch.copy_(rs_tails)
        self.semaphore_torch.zero_()
        self.gemm_out_torch.fill_(float("nan"))
        self.out.fill_(float("nan"))

    def prepare(self) -> None:
        barrier_on_compute_stream(self.runtime)

    def launch(self) -> None:
        self.module[impl.FUSED_DEVICE_ENTRYPOINT](
            self.A,
            self.B,
            self.gemm_out,
            self.semaphore,
            self.out,
            self.gemm_task_types,
            self.gemm_task_idxs,
            self.gemm_head,
            self.gemm_tail,
            self.rs_task_types,
            self.rs_task_idxs,
            self.rs_head,
            self.rs_tail,
        )

    def assert_terminal_state(self) -> None:
        if int(self.gemm_tail_torch.item()) != self.config.gemm_task_count:
            raise AssertionError("GEMM queue tail changed unexpectedly")
        if int(self.rs_tail_torch.item()) != self.config.rs_task_count:
            raise AssertionError("RS queue did not publish every tile")
        if int(self.gemm_head_torch.item()) < self.config.gemm_task_count:
            raise AssertionError("GEMM queue did not consume every task")
        if int(self.rs_head_torch.item()) < self.config.rs_task_count:
            raise AssertionError("RS queue did not consume every task")
        if torch.any(self.gemm_types_torch[: self.config.gemm_task_count] != -1):
            raise AssertionError("GEMM queue retains an unconsumed task")
        if torch.any(self.rs_types_torch[: self.config.rs_task_count] != -1):
            raise AssertionError("RS queue retains an unconsumed task")
        torch.testing.assert_close(
            self.semaphore_torch,
            torch.full_like(self.semaphore_torch, self.config.completion_count),
        )
        if torch.isnan(self.gemm_out_torch).any() or torch.isnan(self.out).any():
            raise AssertionError("GemmRS output contains an uncovered tile")


def _allocate_case(
    runtime: DistributedRuntime,
    module: Any,
    data: dict[str, torch.Tensor],
    config: impl.GemmRSConfig,
) -> _Case:
    queue_state = _queue_state(config)
    device = torch.device("cuda", runtime.rank)
    gemm_out = symmetric_empty(runtime, (config.M, config.N), config.dtype)
    semaphore = symmetric_empty(runtime, (config.rs_m_clusters, config.rs_n_clusters), "uint64")
    gemm_task_types = torch.empty((impl.CAPACITY,), dtype=torch.int32, device=device)
    gemm_task_idxs = torch.empty(
        (impl.CAPACITY, impl.TASK_IDX_LEN), dtype=torch.int32, device=device
    )
    gemm_head = torch.empty((1,), dtype=torch.int32, device=device)
    gemm_tail = torch.empty((1,), dtype=torch.int32, device=device)
    rs_task_types = symmetric_empty(runtime, (impl.CAPACITY,), "int32")
    rs_task_idxs = symmetric_empty(runtime, (impl.CAPACITY, impl.TASK_IDX_LEN), "int32")
    rs_head = symmetric_empty(runtime, (1,), "int32")
    rs_tail = symmetric_empty(runtime, (1,), "int32")
    initial_queues = tuple(
        torch.from_numpy(array[runtime.rank].copy()).to(device) for array in queue_state
    )
    case = _Case(
        runtime=runtime,
        module=module,
        config=config,
        A=data["A"],
        B=data["B"],
        gemm_out=gemm_out,
        semaphore=semaphore,
        out=torch.empty((config.local_m, config.N), dtype=torch.float16, device=device),
        gemm_task_types=gemm_task_types,
        gemm_task_idxs=gemm_task_idxs,
        gemm_head=gemm_head,
        gemm_tail=gemm_tail,
        rs_task_types=rs_task_types,
        rs_task_idxs=rs_task_idxs,
        rs_head=rs_head,
        rs_tail=rs_tail,
        gemm_out_torch=torch_view(gemm_out),
        semaphore_torch=torch_view(semaphore),
        gemm_types_torch=torch_view(gemm_task_types),
        gemm_head_torch=torch_view(gemm_head),
        gemm_tail_torch=torch_view(gemm_tail),
        rs_types_torch=torch_view(rs_task_types),
        rs_head_torch=torch_view(rs_head),
        rs_tail_torch=torch_view(rs_tail),
        initial_queues=initial_queues,
    )
    if config.world_size > 1:
        require_nvls_multicast(runtime, case.gemm_out_torch)
    with torch.cuda.stream(runtime.timing_stream):
        case.reset()
    runtime.timing_stream.synchronize()
    runtime.barrier()
    return case


def _reference_outputs(
    runtime: DistributedRuntime, data: dict[str, torch.Tensor], config: impl.GemmRSConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    partial = torch.mm(data["A"], data["B"].T)
    expected = torch.empty(
        (config.local_m, config.N), dtype=torch.float16, device=f"cuda:{runtime.rank}"
    )
    dist.reduce_scatter_tensor(expected, partial, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(runtime.rank)
    return partial, expected


def _check_correctness(case: _Case, partial: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(case.gemm_out_torch, partial, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(case.out, expected, rtol=1e-2, atol=1e-2)
    case.assert_terminal_state()


def _run_worker(
    runtime: DistributedRuntime, module: Any, mode: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    config = _config(
        kwargs["M"],
        kwargs["N"],
        kwargs["K"],
        kwargs["world_size"],
        kwargs["dtype"],
        kwargs.get("scheduler", "dynamic"),
    )
    data = prepare_data(rank=runtime.rank, **kwargs)
    case = _allocate_case(runtime, module, data, config)

    if mode == "test":
        partial, expected = _reference_outputs(runtime, data, config)
        for _ in range(_RELAUNCH_COUNT):
            with torch.cuda.stream(runtime.timing_stream):
                case.reset()
                case.prepare()
                case.launch()
            runtime.timing_stream.synchronize()
            _check_correctness(case, partial, expected)
            runtime.barrier()
        return {
            "status": "OK",
            "relaunches": _RELAUNCH_COUNT,
            "gemm_tasks": config.gemm_task_count,
            "rs_tasks": config.rs_task_count,
        }
    if mode != "bench":
        raise ValueError(f"unsupported distributed worker mode {mode!r}")

    from tvm.tirx.bench import bench

    baselines = create_baseline_suite(
        runtime,
        data,
        workload="gemm_reduce_scatter",
        M=config.M,
        N=config.N,
        K=config.total_k,
        world_size=config.world_size,
    )
    try:

        def prepare() -> None:
            case.reset()
            case.prepare()

        result = bench(
            {"tirx": case.launch},
            references=baselines.references(),
            timer="event",
            rounds=kwargs.get("rounds", 1),
            cooldown_s=kwargs.get("cooldown_s", 1.0),
            distributed=runtime.bench_context(),
            prepare={"tirx": prepare},
        )
        kernel_only = bench(
            {"tirx": case.launch},
            timer="kineto",
            rounds=kwargs.get("rounds", 1),
            cooldown_s=kwargs.get("cooldown_s", 1.0),
            distributed=runtime.bench_context(),
            prepare={"tirx": prepare},
            kineto_kernel_names={"tirx": impl.FUSED_DEVICE_ENTRYPOINT},
        )
        result["baseline_metadata"] = baselines.metadata()
        result["ratio_definition"] = "baseline_us / tirx_us"
        result["ratios"] = baseline_ratios(result, tirx="tirx")
        result["kernel_only"] = kernel_only
        result["performance_gate"] = {
            "required_ratio": "> 1",
            "passed": all(
                ratio > 1 for name, ratio in result["ratios"].items() if name.startswith("cublas")
            ),
        }
        return {"status": "OK", **result}
    finally:
        baselines.close()


def run_test(
    M: int = impl.M,
    N: int = impl.N,
    K: int = impl.TOTAL_K,
    world_size: int = 4,
    dtype: str = impl.DTYPE,
    seed: int = 42,
    scheduler: str = "dynamic",
    **_kwargs: Any,
) -> None:
    """Validate the direct port for 20 reset/relaunch cycles."""

    _config(M, N, K, world_size, dtype, scheduler)
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
    world_size: int = 4,
    dtype: str = impl.DTYPE,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    scheduler: str = "dynamic",
    **_kwargs: Any,
) -> dict[str, Any]:
    """Benchmark the direct port and external baselines."""

    _config(M, N, K, world_size, dtype, scheduler)
    if timer not in {None, "event"}:
        raise ValueError("distributed GemmRS supports only timer='event'")
    if warmup is not None or repeat is not None:
        raise ValueError("timer='event' uses fixed iteration counts and rejects overrides")
    return run_distributed(
        _get_benchmark_kernel(M, N, K, world_size, dtype, scheduler=scheduler),
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
            "timer": "event",
            "rounds": rounds,
            "cooldown_s": cooldown_s,
        },
    )


__all__ = [
    "CONFIGS",
    "KERNEL_META",
    "_manual_queue_state",
    "_queue_state",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
