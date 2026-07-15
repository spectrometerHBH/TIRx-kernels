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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run the annotated FA4 kernel under NVIDIA IKET.

Example::

  TVM_IKET_OFFICIAL_PROFILE=cutlass-4.6.1 \
    run-iket --output-dir /tmp/fa4-iket --clobber \
      profile --postprocess all -- \
      python -m tirx_kernels.attention.profile_flash_attention4 \
        --seq-len 1024 --causal

The script launches only the target FA4 kernel.  Correctness and ordinary
kernel benchmarks remain in flash_attention4.run_test and run_bench; trace
allocation, patching, timing, and postprocessing are owned by run-iket.
"""

from __future__ import annotations

import argparse

import torch

import tvm
from tirx_kernels.attention.flash_attention4 import get_flash_attention4_kernel, prepare_data
from tvm.tirx.bench import IketProfiler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of traced FA4 launches; setup and compilation remain outside the loop",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    func = get_flash_attention4_kernel(
        args.batch_size,
        args.seq_len,
        args.seq_len,
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
        is_causal=args.causal,
    )
    executable = IketProfiler().compile(
        tvm.IRModule({"main": func}),
        target=tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}),
        tir_pipeline="tirx",
    )

    q, k, v, _ = prepare_data(
        args.batch_size,
        args.seq_len,
        args.seq_len,
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
    )
    q, k, v = q.cuda(), k.cuda(), v.cuda()
    out = torch.empty(
        (args.batch_size, args.seq_len, args.num_qo_heads, args.head_dim),
        dtype=torch.float16,
        device="cuda",
    )

    for _ in range(args.repeat):
        executable(q, k, v, out)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
