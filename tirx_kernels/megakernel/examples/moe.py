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

"""Standalone six-stage MoE example written with the TVM megakernel DSL.

Run from the tirx-kernels checkout with the paired TVM checkout on
``PYTHONPATH``::

    python -m tirx_kernels.megakernel.examples.moe \
        --batch-size 128 --scheduler dynamic

The complete ``KernelSpec`` construction is intentionally visible in this
file.  It does not call the production ``build_moe_graph`` helper.
"""

from __future__ import annotations

import argparse

from tirx_kernels.megakernel.dsl import (
    AlignTileImpl,
    CountSortTileImpl,
    DownTileImpl,
    GateUpSiluTileImpl,
    GatingTileImpl,
    KernelSpec,
    MoeLowerer,
    TopkTileImpl,
    VarSpec,
    policy_for_scheduler,
)
from tirx_kernels.megakernel.utils.config import MEGAKERNEL_MOE_BENCH_CONFIG

_NUM_EXPERTS = 128
_TOP_K = 8
_ROUTE_BLOCK = 128


def _max_rows(batch_size: int) -> int:
    routed = batch_size * _TOP_K
    if routed < _NUM_EXPERTS:
        return routed
    return _NUM_EXPERTS + (routed - _NUM_EXPERTS + _ROUTE_BLOCK - 1) // _ROUTE_BLOCK


def build_example(batch_size: int = 128) -> KernelSpec:
    """Construct and validate the complete six-stage logical MoE graph."""

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive compile-time integer")

    config = MEGAKERNEL_MOE_BENCH_CONFIG
    max_rows = _max_rows(batch_size)
    max_tokens = max_rows * _ROUTE_BLOCK
    relaxed_rows = batch_size * _TOP_K // _ROUTE_BLOCK + _NUM_EXPERTS + 1
    routed_rows = VarSpec("routed_rows")
    kernel = KernelSpec(
        "qwen3_30b_a3b_moe", attrs={"source": "Qwen3-30B-A3B six-stage MoE pipeline"}
    )

    hidden_state = kernel.input("hidden_state", (batch_size, 2048), "float16")
    kernel.input("residual", (batch_size, 2048), "float16")
    kernel.output("output", (batch_size, 2048), "float16")
    gate_weight = kernel.input("gate_weight", (128, 2048), "float16")
    gate_up_weight = kernel.input("gate_up_weight", (128, 1536, 2048), "float16")
    down_weight = kernel.input("down_weight", (128, 2048, 768), "float16")
    gating_output = kernel.intermediate("gating_output", (batch_size, 128), "float32")
    topk_weights = kernel.intermediate("topk_weights", (batch_size, 8), "float32")
    topk_indices = kernel.intermediate("topk_indices", (batch_size, 8), "int32")
    sorted_token_ids = kernel.intermediate("sorted_token_ids", (max_tokens,), "int32")
    expert_ids = kernel.intermediate("expert_ids", (max_rows,), "int32")
    num_valid_tokens = kernel.intermediate("num_valid_tokens", (max_rows,), "int32")
    num_tokens_post_pad = kernel.intermediate("num_tokens_post_pad", (1,), "int32")
    cumsum_buffer = kernel.intermediate("cumsum_buffer", (129,), "int32")
    reordered_hidden_state = kernel.intermediate(
        "reordered_hidden_state", (max_tokens, 2048), "float16"
    )
    silu_mul_output = kernel.intermediate("silu_mul_output", (max_tokens, 768), "float16")
    topk_reduce_output = kernel.output("topk_reduce_output", (batch_size, 2048), "float16")

    gating_done = kernel.event(
        "gating_done",
        (1,),
        4 * ((batch_size + 127) // 128),
        attrs={"meaning": "all split-K gating tiles are complete"},
    )
    topk_done = kernel.event(
        "topk_done", (1,), 148, attrs={"meaning": "all persistent top-k tiles are complete"}
    )
    align_done = kernel.event(
        "align_done", (1,), 1, attrs={"meaning": "token-to-expert alignment metadata is ready"}
    )
    count_sort_done = kernel.event(
        "count_sort_done", (1,), 148, attrs={"meaning": "all count-and-sort tiles are complete"}
    )
    gate_up_done = kernel.event(
        "gate_up_done",
        (relaxed_rows,),
        12,
        attrs={"meaning": "all gate-up projections for one routed row are complete"},
    )

    (
        kernel.tile(
            "gating",
            impl=GatingTileImpl(config),
            tile_num=((batch_size + 127) // 128, 1, 4),
            attrs={
                "source_stage": "gating_output = hidden_state @ gate_weight.T",
                "purpose": "compute split-K expert logits",
            },
        )
        .read(hidden_state, gate_weight)
        .write(gating_output)
        .notify(gating_done, lambda m, n, k: (0,))
    )
    (
        kernel.tile(
            "topk",
            impl=TopkTileImpl(config, batch_size),
            tile_num=(148, 1, 1),
            attrs={
                "source_stage": "topk_weights, topk_indices = topk(gating_output)",
                "purpose": "select experts and routing weights",
            },
        )
        .read(gating_output)
        .write(topk_weights, topk_indices)
        .wait(gating_done, lambda m, n, k: (0,))
        .notify(topk_done, lambda m, n, k: (0,))
    )
    (
        kernel.tile(
            "align",
            impl=AlignTileImpl(config, batch_size),
            tile_num=(1, 1, 1),
            attrs={
                "source_stage": "align tokens by expert",
                "purpose": "produce padded routing metadata",
            },
        )
        .read(topk_indices)
        .write(sorted_token_ids, expert_ids, num_valid_tokens, num_tokens_post_pad, cumsum_buffer)
        .wait(topk_done, lambda m, n, k: (0,))
        .notify(align_done, lambda m, n, k: (0,))
    )
    (
        kernel.tile(
            "count_sort",
            impl=CountSortTileImpl(config, batch_size),
            tile_num=(148, 1, 1),
            attrs={
                "source_stage": "reorder hidden states by expert",
                "purpose": "count and scatter routed tokens",
            },
        )
        .read(topk_indices, sorted_token_ids, cumsum_buffer, hidden_state, num_tokens_post_pad)
        .write(reordered_hidden_state)
        .wait(align_done, lambda m, n, k: (0,))
        .notify(count_sort_done, lambda m, n, k: (0,))
    )
    (
        kernel.tile(
            "gate_up_silu",
            impl=GateUpSiluTileImpl(config, batch_size),
            tile_num=(routed_rows, 12, 1),
            attrs={
                "source_stage": "silu(gate) * up",
                "purpose": "compute routed gate-up projections and SiLU",
            },
        )
        .read(
            reordered_hidden_state,
            gate_up_weight,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_valid_tokens,
            num_tokens_post_pad,
        )
        .write(silu_mul_output)
        .wait(count_sort_done, lambda m, n, k: (0,))
        .notify(gate_up_done, lambda m, n, k: (m,))
    )
    (
        kernel.tile(
            "down",
            impl=DownTileImpl(config, batch_size),
            tile_num=(routed_rows, 16, 1),
            attrs={
                "source_stage": "topk_reduce_output = down(silu_mul_output)",
                "purpose": "compute and accumulate routed down projections",
            },
        )
        .read(
            silu_mul_output,
            down_weight,
            expert_ids,
            topk_weights,
            sorted_token_ids,
            num_valid_tokens,
            num_tokens_post_pad,
        )
        .write(topk_reduce_output)
        .wait(gate_up_done, lambda m, n, k: (m,))
    )
    return kernel.validate()


def _extent_text(extent: int | VarSpec) -> str:
    return extent.name if isinstance(extent, VarSpec) else str(extent)


def _tensor_names(tensors) -> str:
    return ", ".join(tensor.name for tensor in tensors) or "-"


def _event_names(dependencies) -> str:
    return ", ".join(dependency.event.name for dependency in dependencies) or "-"


def describe_graph(spec: KernelSpec, batch_size: int) -> str:
    """Render the scheduler-independent logical graph."""

    lines = [
        f"kernel: {spec.name}",
        f"batch_size: {batch_size}",
        f"logical tensors: {len(spec.tensors)}",
        f"logical events ({len(spec.events)}): {', '.join(spec.events)}",
        f"tiles ({len(spec.tiles)}):",
    ]
    for tile in spec.tiles:
        tile_num = ", ".join(_extent_text(extent) for extent in tile.tile_num)
        lines.extend(
            [
                f"  - {tile.name}: {type(tile.impl).__name__} tile_num=({tile_num})",
                f"    reads: {_tensor_names(tile.reads)}",
                f"    writes: {_tensor_names(tile.writes)}",
                f"    waits: {_event_names(tile.waits)}",
                f"    notifies: {_event_names(tile.notifies)}",
            ]
        )
    return "\n".join(lines)


def describe_plan(spec: KernelSpec, scheduler: str) -> str:
    """Lower through one policy and render its private physical-plan boundary."""

    plan = MoeLowerer(policy_for_scheduler(scheduler)).lower(spec)
    return "\n".join(
        [
            f"scheduler: {plan.policy_name}",
            f"physical events ({len(plan.events)}): "
            + ", ".join(event.name for event in plan.events),
            f"dispatch rules: {len(plan.dispatch_plans)}",
            f"queue upper bound: {plan.queue_upper_bound}",
            f"down coalescing: {plan.down_coalescing}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--scheduler", choices=("static", "unfused", "dynamic"))
    args = parser.parse_args(argv)

    spec = build_example(args.batch_size)
    print(describe_graph(spec, args.batch_size))
    if args.scheduler is not None:
        print()
        print(describe_plan(spec, args.scheduler))


if __name__ == "__main__":
    main()
