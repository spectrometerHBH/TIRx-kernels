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

This category contains the SM100, tensor-parallel kernels recovered from the
megakernel performance branch. Both are fixed TP4 implementations:

| Registry name | Global operation | Rank-local result |
| --- | --- | --- |
| `allgather_gemm` | `A[M, K] @ W[N, K].T` after gathering row shards of `A` | `[M, N / 4]` |
| `gemm_reduce_scatter` | sum of rank-local `A[M, K / 4] @ W[N, K / 4].T`, scattered over `M` | `[M / 4, N]` |

The second source was historically called GEMM+AllReduce, but its actual
protocol and output shape are ReduceScatter. The public name reflects the
implemented operation.

The runtime uses one Disco process per rank, NCCL for rank setup, and NVSHMEM
for the overlapped communication protocol. Mutable queues, semaphores, and
outputs are reset independently on every rank before each measured launch.

## Megakernel DSL

Both workloads have scheduler-independent `tvm.megakernel.dsl.KernelSpec`
graphs under `tirx_kernels.gemm_comm.dsl`. The complete kernels and their
concrete implementations live together:

- `allgather_gemm.py`: `AllGatherTileImpl` and
  `AllGatherGemmTileImpl`.
- `gemm_reduce_scatter.py`: `PartialGemmTileImpl`,
  `ReduceScatterTileImpl`, and `ReduceSumTileImpl`.

The TVM `TileImpl.run(m, n, k)` API is the only task boundary; tirx-kernels
does not define another task model. The persistent kernels call those methods
directly for every scheduled tile, including the independent TMA, MMA,
epilogue, load, and reduction warp roles. The policy layer owns only rank-aware
ordering and queue assignment. Standalone examples contain the complete
logical DSL construction and are parity-tested against the production graphs.

```bash
python -m tirx_kernels.megakernel.examples.allgather_gemm --scheduler dynamic
python -m tirx_kernels.megakernel.examples.gemm_reduce_scatter --scheduler static
```

AllGather+GEMM lowers the same concrete GEMM TileImpl through both static
grid-stride and dynamic MPMC policies. GEMM+ReduceScatter lowers its concrete
partial-GEMM and reduction TileImpls through the existing static rank-aware
pipeline. Its dynamic policy is normalized and fully coverage-checked, but
physical lowering intentionally raises: the TMA, MMA, and epilogue roles
advance independently, so a CTA-wide dequeue would serialize the inter-tile
pipeline. The implementation-preserving path remains the default until a
pipelined multi-role dynamic dequeue is available.

`get_kernel(..., use_dsl=False)` keeps an explicit manual oracle for structural
and code-generation A/B checks.

## Baselines

`run_bench()` measures the slowest rank with CUDA events and then launches an
isolated process for two baselines using the same shapes and FP16 semantics:

- cuBLASMp `cublasMpMatmul` with a symmetric NCCL workspace and the official
  TP block-noncyclic distributions. `split_p2p` is the default algorithm hint.
- an explicit cuBLAS + NCCL composition: AllGather then GEMM, or GEMM then
  ReduceScatter. All tensors and GEMM outputs are preallocated outside the
  timed region.

The baseline worker requires cuBLASMp 0.10, nvmath-python bindings, NCCL4Py,
and NCCL 2.29.2 or newer. Point it at the selected libraries when they are not
installed in the default loader path:

```bash
export TIRX_NCCL_LIBRARY=/path/to/libnccl.so.2
export TIRX_CUBLAS_LIBRARY=/path/to/libcublas.so
export TIRX_CUBLASMP_LIBRARY=/path/to/libcublasmp.so.0
```

For a four-GPU B200 host:

```bash
python -m tirx_kernels.test --kernel allgather_gemm
python -m tirx_kernels.test --kernel gemm_reduce_scatter

python -m tirx_kernels.bench --kernel allgather_gemm --rounds 3 --json
python -m tirx_kernels.bench --kernel gemm_reduce_scatter --rounds 3 --json
```

`--warmup` and `--repeat` are millisecond budgets, consistent with the common
benchmark CLI. Each round calibrates iteration counts first, and the reported
value is the mean of per-round medians. Baseline allocation, initialization,
correctness comparison, barriers, and teardown are outside the timed region.
