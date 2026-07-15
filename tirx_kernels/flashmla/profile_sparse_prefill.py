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
"""Profile the three SM100 sparse FlashMLA prefill kernels with NVIDIA IKET.

Example::

  TVM_IKET_OFFICIAL_PROFILE=cutlass-4.6.1 \
    run-iket --output-dir /tmp/flashmla-iket --clobber \
      profile --postprocess all -- \
      python -m tirx_kernels.flashmla.profile_sparse_prefill

The default workload launches one readable representative of each dispatch:
head64, regular head128, and small-topk head128.  Compilation and allocation
remain outside the traced launch loop; run-iket owns patching and trace output.
"""

from __future__ import annotations

import argparse
from types import ModuleType
from typing import Any

import torch

import tvm
from tirx_kernels.flashmla import sparse_prefill_head64_phase1 as head64
from tirx_kernels.flashmla import sparse_prefill_head128_phase1 as head128
from tirx_kernels.flashmla import sparse_prefill_head128_small_topk_phase1 as head128_small
from tvm.tirx.bench import IketProfiler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        choices=("all", "head64", "head128", "head128-small"),
        default="all",
        help="Sparse prefill implementation to trace (default: all three)",
    )
    parser.add_argument("--s-q", type=int, default=1, help="Query rows per implementation")
    parser.add_argument("--s-kv", type=int, default=8192, help="KV sequence length")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Traced launches per selected implementation"
    )
    return parser.parse_args()


def _configs(args: argparse.Namespace) -> list[tuple[str, ModuleType, dict[str, Any]]]:
    configs = [
        (
            "head64",
            head64,
            {
                "s_q": args.s_q,
                "s_kv": args.s_kv,
                "topk": 512,
                "d_qk": 576,
                "h_q": 64,
                "have_attn_sink": True,
            },
        ),
        (
            "head128",
            head128,
            {
                "s_q": args.s_q,
                "s_kv": args.s_kv,
                "topk": 2048,
                "d_qk": 576,
                "h_q": 128,
                "have_attn_sink": True,
            },
        ),
        (
            "head128-small",
            head128_small,
            {"s_q": args.s_q, "s_kv": args.s_kv, "topk": 1280, "h_q": 128, "have_attn_sink": True},
        ),
    ]
    if args.kernel == "all":
        return configs
    return [config for config in configs if config[0] == args.kernel]


def _launch_args(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        case["q"],
        case["kv"].reshape(-1),
        case["indices"].reshape(-1),
        case["attn_sink"],
        case["topk_length"],
        case["out"],
        case["max_logits"],
        case["lse"],
    )


def main() -> None:
    args = _parse_args()
    if args.s_q <= 0:
        raise ValueError("--s-q must be positive")
    if args.s_kv <= 0:
        raise ValueError("--s-kv must be positive")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    launches = []
    for name, module, config in _configs(args):
        executable = IketProfiler().compile(
            tvm.IRModule({"main": module.get_kernel(**config)}), target=target, tir_pipeline="tirx"
        )
        case = module.prepare_data(**config)
        launches.append((name, executable, _launch_args(case)))

    for name, executable, launch_args in launches:
        for _ in range(args.repeat):
            executable(*launch_args)
        print(f"profiled {name}: {args.repeat} launch(es)")
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
