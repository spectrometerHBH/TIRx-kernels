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

"""The six-stage MoE kernel written with the megakernel DSL.

``build_moe_graph`` records the scheduler-independent logical graph: tensors,
events, tiles, and wait/notify edges.  Every physical fact (job ids, endpoint
scopes, run predicates, the drain event) is declared on the spec or on the
tile implementations, so ``tvm.megakernel.transform.build_runtime_kernel``
emits a standalone kernel for the static and dynamic schedulers (and the
unfused static variant).  The hand-written kernel in
``tirx_kernels.megakernel.moe`` remains an independent numerical reference.

Usage::

    build = build_moe_kernel(MEGAKERNEL_MOE_BENCH_CONFIG, batch_size, "static")
    _, lib = get_source(build.module)          # CUDA compile
    kernel = lib["qwen3_30b_a3b_moe"]
    exec_queue = build.exec_queue              # static central queue
    # dynamic: build.queue_tasks/head/tail seed arrays instead
    workspace = np.zeros((build.event_workspace_size,), dtype=np.int32)

Physical-fact ownership:

- spec tile attrs: ``megakernel.job_id`` (production ``JobType`` values) and
  ``megakernel.run_predicate`` (the static routed-row guard);
- spec event attrs: ``megakernel.drain`` on ``down_dispatch_done`` (static
  host init, dynamic runtime init in the align tile);
- tile impl class attributes (``dsl/tile_impl.py``): ``wait_level``,
  ``wait_mask``, ``notify_scope``, ``pre_notify_scope``, ``notify_release``,
  ``profile_event``, ``class_group``, ``hoisted_views``;
- caller policy (``moe_lowering_options``): reserved job ids matching
  ``JobType``, the profiler-parameter ABI flag, and the dynamic down-tile
  coalescing factor.

Host queues come from ``build_runtime_kernel`` itself: the static central
task list dealt round-robin into the per-SM queue, or the dynamic MPMC seed
arrays (event-init tasks plus the entry gating grid).  Both are byte-identical
to the production ``generate_exec_queue_moe`` arrays for the same batch size.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tirx_kernels.megakernel.utils.config import JobType, KernelConfig
from tvm.megakernel.dsl import KernelSpec
from tvm.megakernel.transform import LoweringOptions, RuntimeKernelBuild, build_runtime_kernel

_EXPECTED_CONFIG = {
    "HIDDEN_SIZE": 2048,
    "INTERMEDIATE_SIZE": 768,
    "NUM_EXPERTS": 128,
    "NUM_EXPERTS_PER_TOK": 8,
    "GATING_SPLIT_K_FACTOR": 4,
}

_SCHEDULERS = ("static", "dynamic", "unfused")


def _max_rows(batch_size: int) -> int:
    routed = batch_size * _EXPECTED_CONFIG["NUM_EXPERTS_PER_TOK"]
    experts = _EXPECTED_CONFIG["NUM_EXPERTS"]
    if routed < experts:
        return routed
    return experts + (routed - experts + 127) // 128


def _validate_mvp_config(config: Mapping[str, Any]):
    mismatch = {
        name: (config.get(name), expected)
        for name, expected in _EXPECTED_CONFIG.items()
        if config.get(name) != expected
    }
    if mismatch:
        raise ValueError(f"MoE megakernel DSL MVP only supports Qwen3-30B-A3B: {mismatch}")


def build_moe_graph(
    config: Mapping[str, Any], batch_size: int, *, unfused: bool = False
) -> KernelSpec:
    """Build the scheduler-independent six-stage MoE graph.

    ``unfused`` selects the static unfused variant: the gate-up completion
    event collapses to a single cell counted at the padded upper bound and
    both gemm tiles address it at coordinate ``(0,)``.
    """

    from ..tile_impl import (
        AlignTileImpl,
        CountSortTileImpl,
        DownTileImpl,
        GateUpSiluTileImpl,
        GatingTileImpl,
        TopkTileImpl,
    )

    _validate_mvp_config(config)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive compile-time integer")

    max_rows = _max_rows(batch_size)
    max_tokens = max_rows * 128
    relaxed_rows = batch_size * 8 // 128 + 129
    kernel = KernelSpec(
        "qwen3_30b_a3b_moe", attrs={"source": "Qwen3-30B-A3B six-stage MoE pipeline"}
    )

    # Tensor registration order matches the hand-written kernel's ABI.
    hidden_state = kernel.tensor("hidden_state", (batch_size, 2048), "float16")
    kernel.tensor("residual", (batch_size, 2048), "float16")
    kernel.tensor("output", (batch_size, 2048), "float16")
    gate_weight = kernel.tensor("gate_weight", (128, 2048), "float16")
    gate_up_weight = kernel.tensor("gate_up_weight", (128, 1536, 2048), "float16")
    down_weight = kernel.tensor("down_weight", (128, 2048, 768), "float16")
    gating_output = kernel.tensor("gating_output", (batch_size, 128), "float32")
    topk_weights = kernel.tensor("topk_weights", (batch_size, 8), "float32")
    topk_indices = kernel.tensor("topk_indices", (batch_size, 8), "int32")
    sorted_token_ids = kernel.tensor("sorted_token_ids", (max_tokens,), "int32")
    expert_ids = kernel.tensor("expert_ids", (max_rows,), "int32")
    num_valid_tokens = kernel.tensor("num_valid_tokens", (max_rows,), "int32")
    num_tokens_post_pad = kernel.tensor("num_tokens_post_pad", (1,), "int32")
    cumsum_buffer = kernel.tensor("cumsum_buffer", (129,), "int32")
    reordered_hidden_state = kernel.tensor("reordered_hidden_state", (max_tokens, 2048), "float16")
    kernel.tensor("gate_up_output", (max_tokens, 1536), "float16")
    silu_mul_output = kernel.tensor("silu_mul_output", (max_tokens, 768), "float16")
    topk_reduce_output = kernel.tensor("topk_reduce_output", (batch_size, 2048), "float16")

    # The routed-row count is a runtime scalar: the align tile publishes the
    # padded token count and every downstream consumer divides by the block
    # size.  Padded token counts are always >= 128 (one full block).
    padded_tokens = kernel.scalar(
        "num_tokens_post_pad", source=(num_tokens_post_pad, (0,)), range=(128, max_tokens)
    )
    routed_rows = padded_tokens // 128

    gating_done = kernel.event(
        "gating_done",
        (1,),
        4 * ((batch_size + 127) // 128),
        attrs={"meaning": "all split-K gating tiles are complete"},
    )
    topk_done = kernel.event(
        "topk_done",
        (1,),
        KernelConfig.SM_NUMBER,
        attrs={"meaning": "all persistent top-k tiles are complete"},
    )
    align_done = kernel.event(
        "align_done", (1,), 1, attrs={"meaning": "token-to-expert alignment metadata is ready"}
    )
    count_sort_done = kernel.event(
        "count_sort_done",
        (1,),
        KernelConfig.SM_NUMBER,
        attrs={"meaning": "all count-and-sort tiles are complete"},
    )
    if unfused:
        gate_up_done = kernel.event(
            "gate_up_done",
            (1,),
            max_rows * 12,
            attrs={"meaning": "all gate-up projections are complete (unfused single cell)"},
        )
    else:
        gate_up_done = kernel.event(
            "gate_up_done",
            (relaxed_rows,),
            12,
            attrs={"meaning": "all gate-up projections for one routed row are complete"},
        )
    # The terminal down tile's drain event.  Static scheduling initializes it
    # from the host at the padded upper bound (it is never waited on there);
    # dynamic scheduling runtime-initializes it in the align tile and pushes
    # the END tasks from the down tile's last pre-notify.
    kernel.event(
        "down_dispatch_done",
        (1,),
        max_rows * 16,
        attrs={"meaning": "all routed down projections are complete", "megakernel.drain": True},
    )

    run_while_routed = (0, "lt", routed_rows)
    (
        kernel.tile(
            "gating",
            impl=GatingTileImpl(config, kernel.tensors),
            tile_num=((batch_size + 127) // 128, 1, 4),
            reads=[hidden_state, gate_weight],
            writes=[gating_output],
            attrs={
                "source_stage": "gating_output = hidden_state @ gate_weight.T",
                "purpose": "compute split-K expert logits",
                "megakernel.job_id": JobType.MOE_GATING.value,
            },
        ).notify(gating_done, lambda m, n, k: (0,))
    )
    (
        kernel.tile(
            "topk",
            impl=TopkTileImpl(config, batch_size, kernel.tensors),
            tile_num=(KernelConfig.SM_NUMBER, 1, 1),
            reads=[gating_output],
            writes=[topk_weights, topk_indices],
            attrs={
                "source_stage": "topk_weights, topk_indices = topk(gating_output)",
                "purpose": "select experts and routing weights",
                "megakernel.job_id": JobType.MOE_TOPK_SOFTMAX.value,
            },
        )
        .wait(gating_done, lambda m, n, k: (0,))
        .notify(topk_done, lambda m, n, k: (0,))
    )
    (
        kernel.tile(
            "align",
            impl=AlignTileImpl(config, batch_size, kernel.tensors),
            tile_num=(1, 1, 1),
            reads=[topk_indices],
            writes=[
                sorted_token_ids,
                expert_ids,
                num_valid_tokens,
                num_tokens_post_pad,
                cumsum_buffer,
            ],
            attrs={
                "source_stage": "align tokens by expert",
                "purpose": "produce padded routing metadata",
                "megakernel.job_id": JobType.MOE_ALIGN.value,
            },
        )
        .wait(topk_done, lambda m, n, k: (0,))
        .notify(align_done, lambda m, n, k: (0,))
    )
    (
        kernel.tile(
            "count_sort",
            impl=CountSortTileImpl(config, batch_size, kernel.tensors),
            tile_num=(KernelConfig.SM_NUMBER, 1, 1),
            reads=[
                topk_indices,
                sorted_token_ids,
                cumsum_buffer,
                hidden_state,
                num_tokens_post_pad,
            ],
            writes=[reordered_hidden_state],
            attrs={
                "source_stage": "reorder hidden states by expert",
                "purpose": "count and scatter routed tokens",
                "megakernel.job_id": JobType.MOE_COUNT_AND_SORT.value,
            },
        )
        .wait(align_done, lambda m, n, k: (0,))
        .notify(count_sort_done, lambda m, n, k: (0,))
    )
    (
        kernel.tile(
            "gate_up_silu",
            impl=GateUpSiluTileImpl(config, batch_size, kernel.tensors),
            tile_num=(routed_rows, 12, 1),
            reads=[
                reordered_hidden_state,
                gate_up_weight,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_valid_tokens,
                num_tokens_post_pad,
            ],
            writes=[silu_mul_output],
            attrs={
                "source_stage": "silu(gate) * up",
                "purpose": "compute routed gate-up projections and SiLU",
                "megakernel.job_id": JobType.MOE_GROUP_GEMM_GATE_UP_SILU.value,
                "megakernel.run_predicate": run_while_routed,
            },
        )
        .wait(count_sort_done, lambda m, n, k: (0,))
        .notify(gate_up_done, (lambda m, n, k: (0,)) if unfused else (lambda m, n, k: (m,)))
    )
    (
        kernel.tile(
            "down",
            impl=DownTileImpl(config, batch_size, kernel.tensors),
            tile_num=(routed_rows, 16, 1),
            reads=[
                silu_mul_output,
                down_weight,
                expert_ids,
                topk_weights,
                sorted_token_ids,
                num_valid_tokens,
                num_tokens_post_pad,
            ],
            writes=[topk_reduce_output],
            attrs={
                "source_stage": "topk_reduce_output = down(silu_mul_output)",
                "purpose": "compute and accumulate routed down projections",
                "megakernel.job_id": JobType.MOE_GROUP_GEMM_DOWN.value,
                "megakernel.run_predicate": run_while_routed,
            },
        ).wait(gate_up_done, (lambda m, n, k: (0,)) if unfused else (lambda m, n, k: (m,)))
    )
    return kernel


def moe_lowering_options(scheduler: str, batch_size: int) -> LoweringOptions:
    """Lowering options that reproduce the hand-written kernel's physical ABI.

    The reserved job ids match ``JobType`` so the packed queue bytes are
    identical to the production host queues, and the profiler parameter stays
    in the kernel signature (unused with profiling off) for call-site parity.
    The dynamic down-tile coalescing is the production batch policy.
    """

    if scheduler not in _SCHEDULERS:
        raise ValueError(f"unsupported MoE scheduler {scheduler!r}; expected one of {_SCHEDULERS}")
    attrs: dict[str, Any] = {
        "init_event_job_id": JobType.INIT_ETENSOR.value,
        "wait_event_init_job_id": JobType.WAIT_ETENSOR_INIT.value,
        "end_job_id": JobType.END.value,
        "emit_profiler_param": True,
    }
    # The down tile's per-task run loop: the dynamic scheduler amortizes
    # dispatch overhead by the production batch factor, static keeps 1.
    down_task_size = (4 if batch_size >= 4 else 1) if scheduler == "dynamic" else 1
    attrs["tile_coalescing"] = {"down": down_task_size}
    return LoweringOptions(scheduler="dynamic" if scheduler == "dynamic" else "static", attrs=attrs)


def build_moe_kernel(
    config: Mapping[str, Any], batch_size: int, scheduler: str
) -> RuntimeKernelBuild:
    """Build one MoE kernel module and its host queues through the tvm builder.

    ``scheduler`` is ``"static"``, ``"dynamic"``, or ``"unfused"`` (a static
    debug variant).  The returned ``RuntimeKernelBuild`` carries the IRModule
    (kernel symbol ``qwen3_30b_a3b_moe``), the derived host queue arrays
    (``exec_queue`` for static, ``queue_tasks``/``queue_head``/``queue_tail``
    for dynamic), and the exact ``event_workspace_size`` to allocate and zero
    per launch.  Re-upload the queue arrays and re-zero the workspace between
    launches; the device mutates them.
    """

    spec = build_moe_graph(config, batch_size, unfused=scheduler == "unfused")
    spec.validate()
    return build_runtime_kernel(spec, moe_lowering_options(scheduler, batch_size))


__all__ = ["build_moe_graph", "build_moe_kernel", "moe_lowering_options"]
