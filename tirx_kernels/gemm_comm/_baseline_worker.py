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

"""Isolated multi-process cuBLASMp and cuBLAS+NCCL baselines.

This module intentionally has no third-party imports at module import time.  A
fresh Python process preloads one NCCL library before importing PyTorch,
NCCL4Py, or the low-level nvmath-python cuBLASMp binding.  This keeps the
baseline's NCCL version separate from the one linked into TVM's Disco runtime.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

from ._bench import CALIBRATION_ITERATIONS, DEFAULT_REPEAT_MS, DEFAULT_WARMUP_MS, iteration_counts

_WORLD_SIZE = 4
_JSON_PREFIX = "TIRX_GEMM_COMM_BASELINE="

_WORKLOADS = {
    "allgather_gemm": {"M": 8192, "N": 65536, "K": 8192, "input_scale": 0.05},
    "gemm_reduce_scatter": {"M": 16384, "N": 12288, "K": 49152, "input_scale": 0.02},
}

_ALGORITHMS = {
    "default": "DEFAULT",
    "split_p2p": "SPLIT_P2P",
    "split_multicast": "SPLIT_MULTICAST",
    "no_overlap": "NO_OVERLAP",
}


def _load_global(env_name: str, sonames: tuple[str, ...]) -> ctypes.CDLL:
    configured = os.environ.get(env_name)
    candidates = [configured] if configured else []
    candidates.extend(sonames)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    discovered = ctypes.util.find_library(sonames[0].removeprefix("lib").split(".so")[0])
    if discovered is not None:
        return ctypes.CDLL(discovered, mode=ctypes.RTLD_GLOBAL)
    raise RuntimeError(
        f"unable to load {sonames[0]}; set {env_name} to the absolute shared-library path"
    )


def _preload(*, cublasmp: bool) -> None:
    _load_global("TIRX_NCCL_LIBRARY", ("libnccl.so.2", "libnccl.so"))
    if cublasmp:
        _load_global(
            "TIRX_CUBLASMP_LIBRARY",
            ("libcublasmp.so.0", "libcublasmp.so", "libcublasMp.so.0", "libcublasMp.so"),
        )


@dataclass(frozen=True)
class _Problem:
    workload: str
    M: int
    N: int
    K: int
    world_size: int
    scale: float

    @property
    def local_m(self) -> int:
        return self.M // self.world_size

    @property
    def local_n(self) -> int:
        return self.N // self.world_size

    @property
    def local_k(self) -> int:
        return self.K // self.world_size


def _problem(workload: str, world_size: int) -> _Problem:
    if workload not in _WORKLOADS:
        raise ValueError(f"unknown workload: {workload}")
    if world_size != _WORLD_SIZE:
        raise ValueError(f"the tuned kernels require world_size={_WORLD_SIZE}, got {world_size}")
    config = _WORKLOADS[workload]
    return _Problem(
        workload=workload,
        M=int(config["M"]),
        N=int(config["N"]),
        K=int(config["K"]),
        world_size=world_size,
        scale=float(config["input_scale"]),
    )


def _make_inputs(problem: _Problem, rank: int, torch):
    generator = torch.Generator(device=rank)
    generator.manual_seed(42 + rank)
    if problem.workload == "allgather_gemm":
        input_shape = (problem.local_m, problem.K)
        weight_shape = (problem.local_n, problem.K)
    else:
        input_shape = (problem.M, problem.local_k)
        weight_shape = (problem.N, problem.local_k)

    input_tensor = torch.randn(
        input_shape, dtype=torch.float16, device=rank, generator=generator
    ).mul_(problem.scale)
    weight = torch.randn(weight_shape, dtype=torch.float16, device=rank, generator=generator).mul_(
        problem.scale
    )
    return input_tensor, weight


class _CublasMpMatmul:
    """One rank's direct cublasMpMatmul setup for a tensor-parallel fast path."""

    def __init__(self, problem: _Problem, comm, stream, input_tensor, weight, output, algo: str):
        import numpy as np
        from nvmath import CudaDataType
        from nvmath.bindings import cublas, cublasMp

        self._cublas_mp = cublasMp
        self._closed = False
        self._descriptors: list[int] = []
        self._grids: list[int] = []
        self._workspace = 0
        self._workspace_registered_grids: list[int] = []

        self._handle = cublasMp.create(stream.cuda_stream)
        self._matmul_desc = cublasMp.matmul_descriptor_create(cublas.ComputeType.COMPUTE_32F)

        self._set_attr(cublasMp.MatmulDescriptorAttribute.TRANSA, cublas.Operation.T)
        self._set_attr(cublasMp.MatmulDescriptorAttribute.TRANSB, cublas.Operation.N)
        self._set_attr(
            cublasMp.MatmulDescriptorAttribute.ALGO_TYPE,
            getattr(cublasMp.MatmulAlgoType, _ALGORITHMS[algo]),
        )
        if problem.workload == "gemm_reduce_scatter":
            self._set_attr(
                cublasMp.MatmulDescriptorAttribute.COMMUNICATION_TYPE, CudaDataType.CUDA_R_16F
            )

        grid_col = cublasMp.grid_create(
            problem.world_size, 1, cublasMp.GridLayout.COL_MAJOR, comm.ptr
        )
        grid_row = cublasMp.grid_create(
            1, problem.world_size, cublasMp.GridLayout.ROW_MAJOR, comm.ptr
        )
        self._grids.extend((grid_col, grid_row))

        fp16 = CudaDataType.CUDA_R_16F
        if problem.workload == "allgather_gemm":
            # Row-major Python computes input @ weight.T.  The same storage is
            # column-major (weight.T @ input.T) to cuBLASMp.
            matmul_m, matmul_n, matmul_k = problem.N, problem.M, problem.K
            desc_a = cublasMp.matrix_descriptor_create(
                problem.K, problem.N, problem.K, problem.local_n, 0, 0, problem.K, fp16, grid_row
            )
            desc_b = cublasMp.matrix_descriptor_create(
                problem.K, problem.M, problem.K, problem.local_m, 0, 0, problem.K, fp16, grid_row
            )
            desc_d = cublasMp.matrix_descriptor_create(
                problem.N,
                problem.M,
                problem.local_n,
                problem.local_m,
                0,
                0,
                problem.local_n,
                fp16,
                grid_col,
            )
            first_workspace_grid, other_workspace_grid = grid_row, grid_col
        else:
            # K is partitioned for the local GEMM.  cuBLASMp reduces the
            # column-major N x M result and scatters its M columns, which are
            # rows in the row-major Python view.
            matmul_m, matmul_n, matmul_k = problem.N, problem.M, problem.K
            desc_a = cublasMp.matrix_descriptor_create(
                problem.K,
                problem.N,
                problem.local_k,
                problem.N,
                0,
                0,
                problem.local_k,
                fp16,
                grid_col,
            )
            desc_b = cublasMp.matrix_descriptor_create(
                problem.K,
                problem.M,
                problem.local_k,
                problem.local_m,
                0,
                0,
                problem.local_k,
                fp16,
                grid_col,
            )
            desc_d = cublasMp.matrix_descriptor_create(
                problem.N, problem.M, problem.N, problem.local_m, 0, 0, problem.N, fp16, grid_row
            )
            first_workspace_grid, other_workspace_grid = grid_col, grid_row

        self._descriptors.extend((desc_a, desc_b, desc_d))
        self._dimensions = (matmul_m, matmul_n, matmul_k)
        self._pointers = (weight.data_ptr(), input_tensor.data_ptr(), output.data_ptr())
        self._alpha = np.array(1.0, dtype=np.float32)
        self._beta = np.array(0.0, dtype=np.float32)

        workspace_device, workspace_host = cublasMp.matmul_buffer_size(
            self._handle,
            self._matmul_desc,
            *self._dimensions,
            self._alpha.ctypes.data,
            self._pointers[0],
            1,
            1,
            desc_a,
            self._pointers[1],
            1,
            1,
            desc_b,
            self._beta.ctypes.data,
            0,
            1,
            1,
            desc_d,
            self._pointers[2],
            1,
            1,
            desc_d,
        )
        self.workspace_device_bytes = int(workspace_device)
        self.workspace_host_bytes = int(workspace_host)
        if self.workspace_device_bytes <= 0:
            raise RuntimeError("cuBLASMp returned an empty device workspace")

        self._workspace = cublasMp.malloc(first_workspace_grid, self.workspace_device_bytes)
        self._first_workspace_grid = first_workspace_grid
        cublasMp.buffer_register(other_workspace_grid, self._workspace, self.workspace_device_bytes)
        self._workspace_registered_grids.append(other_workspace_grid)
        self._host_workspace = np.empty(self.workspace_host_bytes, dtype=np.uint8)

    def _set_attr(self, attr, value) -> None:
        holder = ctypes.c_int(int(value))
        self._cublas_mp.matmul_descriptor_set_attribute(
            self._matmul_desc, attr, ctypes.addressof(holder), ctypes.sizeof(holder)
        )

    def launch(self) -> None:
        desc_a, desc_b, desc_d = self._descriptors
        host_workspace = self._host_workspace.ctypes.data if self.workspace_host_bytes > 0 else 0
        self._cublas_mp.matmul(
            self._handle,
            self._matmul_desc,
            *self._dimensions,
            self._alpha.ctypes.data,
            self._pointers[0],
            1,
            1,
            desc_a,
            self._pointers[1],
            1,
            1,
            desc_b,
            self._beta.ctypes.data,
            0,
            1,
            1,
            desc_d,
            self._pointers[2],
            1,
            1,
            desc_d,
            self._workspace,
            self.workspace_device_bytes,
            host_workspace,
            self.workspace_host_bytes,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for grid in reversed(self._workspace_registered_grids):
            self._cublas_mp.buffer_deregister(grid, self._workspace)
        self._cublas_mp.free(self._first_workspace_grid, self._workspace)
        for descriptor in reversed(self._descriptors):
            self._cublas_mp.matrix_descriptor_destroy(descriptor)
        self._cublas_mp.matmul_descriptor_destroy(self._matmul_desc)
        self._cublas_mp.destroy(self._handle)
        for grid in reversed(self._grids):
            self._cublas_mp.grid_destroy(grid)


def _barrier(comm, stream, barrier, nccl) -> None:
    with stream:
        barrier.zero_()
        comm.allreduce(barrier, barrier, nccl.SUM, stream=stream)


def _elapsed_us(comm, stream, barrier, nccl, torch, launch) -> float:
    _barrier(comm, stream, barrier, nccl)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with stream:
        start.record(stream)
        launch()
        end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end) * 1_000.0)


def _measure(
    comm, stream, barrier, nccl, torch, launch, warmup_ms: int, repeat_ms: int
) -> tuple[list[float], float, int, int]:
    calibration = [
        _elapsed_us(comm, stream, barrier, nccl, torch, launch)
        for _ in range(CALIBRATION_ITERATIONS)
    ]
    estimate = torch.tensor(
        [sum(calibration) / len(calibration)],
        dtype=torch.float32,
        device=torch.cuda.current_device(),
    )
    comm.allreduce(estimate, estimate, nccl.MAX, stream=stream)
    stream.synchronize()
    estimate_us = float(estimate.item())
    warmup_count, repeat_count = iteration_counts(estimate_us, warmup_ms, repeat_ms)

    for _ in range(warmup_count):
        _elapsed_us(comm, stream, barrier, nccl, torch, launch)
    samples = [_elapsed_us(comm, stream, barrier, nccl, torch, launch) for _ in range(repeat_count)]
    return samples, estimate_us, warmup_count, repeat_count


def _rank_main(rank: int, args: dict[str, Any], unique_id: bytes, result_queue) -> None:
    comm = None
    cublasmp_case = None
    try:
        _preload(cublasmp=True)
        import nccl.core as nccl
        import torch

        torch.cuda.set_device(rank)
        problem = _problem(args["workload"], args["world_size"])
        comm = nccl.Communicator.init(problem.world_size, rank, nccl.UniqueId.from_bytes(unique_id))
        stream = torch.cuda.Stream(device=rank)
        barrier = torch.zeros(1, dtype=torch.float32, device=rank)
        input_tensor, weight = _make_inputs(problem, rank, torch)

        if problem.workload == "allgather_gemm":
            output_shape = (problem.M, problem.local_n)
            collective_buffer = torch.empty(
                (problem.M, problem.K), dtype=torch.float16, device=rank
            )
        else:
            output_shape = (problem.local_m, problem.N)
            collective_buffer = torch.empty(
                (problem.M, problem.N), dtype=torch.float16, device=rank
            )

        cublasmp_output = torch.empty(output_shape, dtype=torch.float16, device=rank)
        nccl_output = torch.empty_like(cublasmp_output)
        cublasmp_case = _CublasMpMatmul(
            problem, comm, stream, input_tensor, weight, cublasmp_output, args["algo"]
        )

        def launch_cublasmp() -> None:
            cublasmp_case.launch()

        if problem.workload == "allgather_gemm":

            def launch_cublas_nccl() -> None:
                comm.allgather(input_tensor, collective_buffer, stream=stream)
                torch.mm(collective_buffer, weight.T, out=nccl_output)

        else:

            def launch_cublas_nccl() -> None:
                torch.mm(input_tensor, weight.T, out=collective_buffer)
                comm.reduce_scatter(collective_buffer, nccl_output, nccl.SUM, stream=stream)

        # Check the direct cuBLASMp path against the explicit cuBLAS+NCCL
        # composition using exactly the same local shards.
        _elapsed_us(comm, stream, barrier, nccl, torch, launch_cublasmp)
        _elapsed_us(comm, stream, barrier, nccl, torch, launch_cublas_nccl)
        torch.testing.assert_close(
            cublasmp_output,
            nccl_output,
            rtol=1e-3 if problem.workload == "allgather_gemm" else 6e-2,
            atol=2e-2 if problem.workload == "allgather_gemm" else 6e-2,
        )

        rank_results = {}
        for name, launch in (
            (f"cublasmp_{args['algo']}", launch_cublasmp),
            ("cublas_nccl", launch_cublas_nccl),
        ):
            rank_results[name] = {"rounds": []}
            for _round in range(args["rounds"]):
                if args["cooldown_s"]:
                    time.sleep(args["cooldown_s"])
                samples, estimate, n_warmup, n_repeat = _measure(
                    comm, stream, barrier, nccl, torch, launch, args["warmup_ms"], args["repeat_ms"]
                )
                rank_results[name]["rounds"].append(
                    {
                        "samples_us": samples,
                        "estimate_us": estimate,
                        "warmup_iterations": n_warmup,
                        "repeat_iterations": n_repeat,
                    }
                )

        result_queue.put(
            {
                "rank": rank,
                "results": rank_results,
                "metadata": {
                    "device": torch.cuda.get_device_name(rank),
                    "cublasmp_version": int(cublasmp_case._cublas_mp.get_version()),
                    "nccl_version": str(nccl.get_lib_version()),
                    "workspace_device_bytes": cublasmp_case.workspace_device_bytes,
                    "workspace_host_bytes": cublasmp_case.workspace_host_bytes,
                },
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "error": traceback.format_exc()})
        raise
    finally:
        if cublasmp_case is not None:
            try:
                cublasmp_case.close()
            except BaseException:
                pass
        if comm is not None and comm.is_valid:
            try:
                comm.destroy()
            except BaseException:
                comm.abort()


def _collect_results(args: argparse.Namespace) -> dict[str, Any]:
    _preload(cublasmp=True)
    import nccl.core as nccl

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    unique_id = nccl.get_unique_id().as_bytes
    worker_args = {
        "workload": args.workload,
        "world_size": args.world_size,
        "algo": args.cublasmp_algo,
        "warmup_ms": args.warmup,
        "repeat_ms": args.repeat,
        "rounds": args.rounds,
        "cooldown_s": args.cooldown,
    }
    processes = [
        context.Process(
            target=_rank_main, args=(rank, worker_args, unique_id, result_queue), daemon=False
        )
        for rank in range(args.world_size)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + args.timeout
    rank_results: dict[int, dict[str, Any]] = {}
    while len(rank_results) < args.world_size and time.monotonic() < deadline:
        try:
            item = result_queue.get(timeout=min(1.0, max(0.0, deadline - time.monotonic())))
        except queue.Empty:
            failed = [process for process in processes if process.exitcode not in (None, 0)]
            if failed:
                break
            continue
        rank_results[int(item["rank"])] = item
        if "error" in item:
            break

    # A rank reports after timing, then collectively deregisters a potentially
    # multi-gigabyte workspace and destroys both grids.  Give that cleanup its
    # own grace period instead of mistaking a reported result for a hung rank.
    cleanup_deadline = time.monotonic() + min(60.0, args.timeout)
    for process in processes:
        process.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
    unfinished = [process for process in processes if process.is_alive()]
    if (
        unfinished
        or len(rank_results) != args.world_size
        or any("error" in result for result in rank_results.values())
    ):
        for process in unfinished:
            process.terminate()
        for process in processes:
            process.join(timeout=5.0)
        errors = [
            f"rank {rank}:\n{result['error']}"
            for rank, result in sorted(rank_results.items())
            if "error" in result
        ]
        if not errors:
            errors.append(
                f"baseline workers timed out or exited before reporting: "
                f"received={sorted(rank_results)}, exitcodes={[p.exitcode for p in processes]}"
            )
        raise RuntimeError("\n".join(errors))

    implementations = next(iter(rank_results.values()))["results"].keys()
    aggregate: dict[str, Any] = {}
    for implementation in implementations:
        round_samples: list[float] = []
        all_slowest_samples: list[float] = []
        rank_samples_by_round: list[list[list[float]]] = []
        iteration_counts_by_round: list[dict[str, int]] = []
        for round_index in range(args.rounds):
            samples_by_rank = [
                rank_results[rank]["results"][implementation]["rounds"][round_index]["samples_us"]
                for rank in range(args.world_size)
            ]
            lengths = {len(samples) for samples in samples_by_rank}
            if len(lengths) != 1:
                raise RuntimeError(
                    f"ranks used different repeat counts for {implementation}: {sorted(lengths)}"
                )
            slowest_samples = [max(values) for values in zip(*samples_by_rank, strict=True)]
            ordered = sorted(slowest_samples)
            midpoint = len(ordered) // 2
            median = (
                ordered[midpoint]
                if len(ordered) % 2
                else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
            )
            round_samples.append(median)
            all_slowest_samples.extend(slowest_samples)
            rank_samples_by_round.append(samples_by_rank)
            rank_zero_round = rank_results[0]["results"][implementation]["rounds"][round_index]
            iteration_counts_by_round.append(
                {
                    "warmup": rank_zero_round["warmup_iterations"],
                    "repeat": rank_zero_round["repeat_iterations"],
                }
            )
        aggregate[implementation] = {
            "mean_us": sum(round_samples) / len(round_samples),
            "samples_us": all_slowest_samples,
            "round_samples_us": round_samples,
            "rank_samples_us": rank_samples_by_round,
            "round_iteration_counts": iteration_counts_by_round,
        }

    return {
        "status": "OK",
        "workload": args.workload,
        "world_size": args.world_size,
        "warmup_ms": args.warmup,
        "repeat_ms": args.repeat,
        "rounds": args.rounds,
        "cooldown_s": args.cooldown,
        "implementations": aggregate,
        "metadata": rank_results[0]["metadata"],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=tuple(_WORKLOADS), required=True)
    parser.add_argument("--world-size", type=int, default=_WORLD_SIZE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_MS)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT_MS)
    parser.add_argument("--cublasmp-algo", choices=tuple(_ALGORITHMS), default="split_p2p")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.warmup <= 0 or args.repeat <= 0:
        parser.error("--warmup and --repeat must be positive millisecond budgets")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.cooldown < 0:
        parser.error("--cooldown must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = _collect_results(args)
    except BaseException:
        result = {"status": "FAIL", "error": traceback.format_exc()}
        print(_JSON_PREFIX + json.dumps(result, separators=(",", ":")), flush=True)
        return 1
    print(_JSON_PREFIX + json.dumps(result, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
