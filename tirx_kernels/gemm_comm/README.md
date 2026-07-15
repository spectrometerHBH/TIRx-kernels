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

The parent compiles and exports one module, then `torch.multiprocessing.spawn`
starts one rank-local worker per GPU. NCCL bootstraps the process group and
broadcasts the NVSHMEM UID; every worker explicitly initializes NVSHMEM, loads
the local module, and owns its Device API streams. No Disco session or remote
runtime object is created. Mutable queues, semaphores, workspaces, and outputs
are reset independently on every rank before each measured launch.

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
epilogue, load, and reduction warp roles. The policy layer produces one
authoritative `ExecutionPlan` containing region entrypoints, ordered physical
steps, rank-aware ordering, and queue assignment. Device adapters and the host
executor interpret those steps directly. Standalone examples contain the
complete logical DSL construction and are parity-tested against the production
graphs.

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

The legacy builders are private test oracles used only for structural and
code-generation parity checks. Production entry points always lower the DSL.

## Validation and benchmarking

`run_bench()` measures the DSL and private manual oracle in the same worker and
paired Kineto round. They share the exact rank-local input tensors but own
independent mutable state. The shared distributed timer uses one preflight, 30
warmups, 30 measured launches, cold L2, rank barriers, AB/BA ordering, and
sample-wise slowest-rank aggregation. Reset and preparation remain outside the
timed launch scope.

For a four-GPU B200 host:

```bash
python -m tirx_kernels.test --kernel allgather_gemm
python -m tirx_kernels.test --kernel gemm_reduce_scatter

python -m tirx_kernels.bench --kernel allgather_gemm --timer kineto --rounds 6 --json
python -m tirx_kernels.bench --kernel gemm_reduce_scatter --timer kineto --rounds 6 --json
```

The distributed Kineto protocol has fixed iteration counts, so `--warmup` and
`--repeat` overrides are rejected. Six rounds give each AB and BA ordering three
samples; the reported value is the mean of round means.
