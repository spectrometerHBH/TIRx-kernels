<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Distributed GEMM kernels

This package contains fixed TP4 SM100 implementations of AllGather+GEMM and
GEMM+ReduceScatter. The parent compiles each module once, then spawned ranks
initialize NCCL and NVSHMEM locally without a Disco session.

## Benchmark baselines

`run_bench()` compares TIRx with two implementations in the same rank-local
worker and paired distributed Kineto round:

- `cublasmp_split_p2p`: cuBLASMp with the official TP block-noncyclic
  distributions and `SPLIT_P2P` algorithm hint.
- `cublas_nccl`: explicit NCCL AllGather followed by cuBLAS GEMM, or cuBLAS
  GEMM followed by NCCL ReduceScatter.

Every implementation receives the same logical rank-local input shards and
owns independent mutable output and workspace state. Setup, allocation, and
correctness preflight are outside the timed scopes. The shared timer applies
fixed preflight/warmup/repeat counts, cold-cache setup, rank barriers, reversed
round ordering, and sample-wise slowest-rank aggregation. Each implementation
is captured in an isolated Kineto profiler session; pairing controls the
session execution order within a round.

cuBLASMp 0.10 requires nvmath-python, NCCL4Py, and a compatible recent NCCL.
Every benchmark requires absolute paths for all four runtime dependencies so a
loader-path change cannot silently alter the comparison:

```bash
export TIRX_NCCL_LIBRARY=/path/to/libnccl.so.2
export TIRX_CUBLAS_LIBRARY=/path/to/libcublas.so
export TIRX_CUBLASMP_LIBRARY=/path/to/libcublasmp.so.0
export TIRX_NVSHMEM_LIBRARY=/path/to/libnvshmem_host.so
export PYTHONPATH=/path/to/nvmath-python:/path/to/cublasmp-package:$PYTHONPATH
```

The selected files are preloaded only in newly spawned rank workers. Each
result records the actual shared object resolving the NCCL, cuBLAS, cuBLASMp,
and NVSHMEM API symbol together with its runtime version, and fails if any
loaded file differs from its configured lock. cuBLASMp builder failures remain
visible in `errors`, which bench-suite treats as a failed workload.

The result's `ratios` mapping is always `baseline_us / tirx_us`; values greater
than one mean TIRx is faster.

```bash
python -m tirx_kernels.test --kernel allgather_gemm
python -m tirx_kernels.test --kernel gemm_reduce_scatter

python -m tirx_kernels.bench --kernel allgather_gemm --timer kineto --rounds 6 --json
python -m tirx_kernels.bench --kernel gemm_reduce_scatter --timer kineto --rounds 6 --json
```
