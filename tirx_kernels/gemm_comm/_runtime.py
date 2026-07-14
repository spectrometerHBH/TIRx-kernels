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

"""Shared Disco/NVSHMEM runtime support for distributed GEMM kernels."""

from __future__ import annotations

import contextlib
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tvm_ffi import Shape

import tvm
from tvm.runtime import disco as di

from ._bench import (
    CALIBRATION_ITERATIONS,
    DEFAULT_REPEAT_MS,
    DEFAULT_WARMUP_MS,
    iteration_counts,
    resolve_budget,
)


@dataclass
class DistributedRuntime:
    """Live TP session and the two streams used by the overlap protocol."""

    session: di.ProcessSession
    communication_stream: Any
    compute_stream: Any


def require_sm100(world_size: int) -> None:
    """Reject unsupported hosts before spawning Disco workers."""

    if di is None:
        raise RuntimeError("TVM was built without the Disco runtime")
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    import torch

    if torch.cuda.device_count() < world_size:
        raise RuntimeError(f"requires {world_size} CUDA devices, found {torch.cuda.device_count()}")
    for device_id in range(world_size):
        major, _minor = torch.cuda.get_device_capability(device_id)
        if major != 10:
            raise RuntimeError(f"device {device_id} has compute capability {major}, expected SM100")


@contextlib.contextmanager
def create_runtime(world_size: int) -> Iterator[DistributedRuntime]:
    """Create and reliably tear down a TP NCCL + NVSHMEM ProcessSession."""

    require_sm100(world_size)
    session = di.ProcessSession(num_workers=world_size)
    nvshmem_initialized = False
    try:
        devices = tuple(range(world_size))
        ccl = tvm.get_global_func("runtime.disco.compiled_ccl")()
        session.init_ccl(ccl, *devices)

        uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")()
        session.get_global_func("runtime.disco.nvshmem.init_nvshmem")(uid, world_size, 0)
        session.sync_worker_0()
        nvshmem_initialized = True

        communication_stream = session.get_global_func("runtime.disco.stream_create")()
        compute_stream = session.get_global_func("runtime.get_cuda_stream")()
        yield DistributedRuntime(session, communication_stream, compute_stream)
    finally:
        if nvshmem_initialized:
            try:
                session.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")()
                session.sync_worker_0()
            finally:
                session.shutdown()
        else:
            session.shutdown()


@contextlib.contextmanager
def load_module(session: di.ProcessSession, ir_module: Any) -> Iterator[di.DModule]:
    """Compile a TIRx module and load it into every Disco worker."""

    with tempfile.TemporaryDirectory(prefix="tirx-gemm-comm-") as tmpdir:
        library_path = Path(tmpdir) / "kernel.so"
        executable = tvm.compile(ir_module, target=tvm.target.Target("cuda"), tir_pipeline="tirx")
        executable.export_library(str(library_path))
        module = session.load_vm_module(str(library_path))
        session._sync_all()
        try:
            yield module
        finally:
            # Workers must finish opening the library before TemporaryDirectory
            # removes it.
            session._sync_all()


def symmetric_empty(session: di.ProcessSession, shape: Sequence[int], dtype: str):
    """Allocate the same NVSHMEM tensor on every rank."""

    return session.get_global_func("runtime.disco.nvshmem.empty")(Shape(tuple(shape)), dtype, None)


def copy_ranked(destination, arrays: Sequence[np.ndarray]) -> None:
    """Copy one host array into each rank of a DRef."""

    for rank, array in enumerate(arrays):
        destination.debug_copy_from(rank, np.ascontiguousarray(array))


def fill_ranked(destination, world_size: int, shape: Sequence[int], dtype: str, value=0) -> None:
    """Fill every rank independently without sharing mutable host storage."""

    for rank in range(world_size):
        destination.debug_copy_from(rank, np.full(tuple(shape), value, dtype=dtype))


def elapsed_ns_by_rank(session: di.ProcessSession, launch: Callable[[], None]) -> list[int]:
    """Measure one launch region with CUDA events on every worker."""

    timer = session.get_global_func("profiling.cuda.event.create")()
    session.get_global_func("profiling.cuda.event.start")(timer)
    launch()
    session.get_global_func("profiling.cuda.event.stop")(timer)
    elapsed = session.get_global_func("profiling.cuda.event.elapsed")(timer)
    return [int(elapsed.debug_get_from_remote(rank)) for rank in range(session.num_workers)]


def max_elapsed_us(session: di.ProcessSession, launch: Callable[[], None]) -> float:
    """Return the slowest rank's end-to-end GPU time in microseconds."""

    return max(elapsed_ns_by_rank(session, launch)) / 1_000.0


def benchmark_slowest_rank(
    session: di.ProcessSession,
    *,
    reset: Callable[[], None],
    prepare: Callable[[], None],
    launch: Callable[[], None],
    warmup: int | None,
    repeat: int | None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
) -> dict[str, Any]:
    """Benchmark a distributed launch using Triton-style millisecond budgets."""

    warmup_ms = resolve_budget(warmup, DEFAULT_WARMUP_MS, "warmup")
    repeat_ms = resolve_budget(repeat, DEFAULT_REPEAT_MS, "repeat")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if cooldown_s < 0:
        raise ValueError("cooldown_s must be non-negative")

    all_samples: list[float] = []
    round_samples: list[float] = []
    round_iteration_counts: list[dict[str, int]] = []

    def measure_once() -> float:
        reset()
        session._sync_all()
        prepare()
        return max_elapsed_us(session, launch)

    for _round in range(rounds):
        if cooldown_s:
            time.sleep(cooldown_s)
        calibration = [measure_once() for _ in range(CALIBRATION_ITERATIONS)]
        estimate_us = float(np.mean(calibration))
        warmup_count, repeat_count = iteration_counts(estimate_us, warmup_ms, repeat_ms)
        for _ in range(warmup_count):
            measure_once()
        samples = [measure_once() for _ in range(repeat_count)]
        all_samples.extend(samples)
        round_samples.append(float(np.median(samples)))
        round_iteration_counts.append(
            {"warmup": warmup_count, "repeat": repeat_count, "estimate_us": round(estimate_us)}
        )

    return {
        "mean_us": float(np.mean(round_samples)),
        "samples_us": all_samples,
        "round_samples_us": round_samples,
        "round_iteration_counts": round_iteration_counts,
        "warmup_ms": warmup_ms,
        "repeat_ms": repeat_ms,
    }


def barrier_on_compute_stream(runtime: DistributedRuntime) -> None:
    runtime.session.get_global_func("runtime.disco.nvshmem.barrier_all_on_stream")(
        runtime.compute_stream
    )


def sync_compute_to_communication(runtime: DistributedRuntime) -> None:
    runtime.session.get_global_func("runtime.disco.stream_sync")(
        runtime.compute_stream, runtime.communication_stream
    )


def sync_communication_to_compute(runtime: DistributedRuntime) -> None:
    runtime.session.get_global_func("runtime.disco.stream_sync")(
        runtime.communication_stream, runtime.compute_stream
    )
