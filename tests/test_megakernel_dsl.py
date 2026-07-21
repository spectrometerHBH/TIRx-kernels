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

"""CPU-only validation tests for the tvm-builder MoE megakernel DSL path."""

import importlib.util
import re

import numpy as np
import pytest

import tirx_kernels.megakernel.dsl as megakernel_dsl
import tvm.ir
import tvm.megakernel.dsl as tvm_dsl
from tirx_kernels.megakernel.dsl import (
    AlignTileImpl,
    CountSortTileImpl,
    DownTileImpl,
    GateUpSiluTileImpl,
    GatingTileImpl,
    TopkTileImpl,
)
from tirx_kernels.megakernel.dsl.examples.moe import (
    _max_rows,
    build_moe_graph,
    build_moe_kernel,
    moe_lowering_options,
)
from tirx_kernels.megakernel.moe import MegaKernelMOE
from tirx_kernels.megakernel.utils.config import MEGAKERNEL_MOE_BENCH_CONFIG, JobType
from tirx_kernels.megakernel.utils.support import generate_exec_queue_moe, push_moe_tasks
from tvm.megakernel.transform import build_runtime_kernel
from tvm.tirx import IfThenElse, SeqStmt, While

_CONFIG = MEGAKERNEL_MOE_BENCH_CONFIG
_KERNEL_NAME = "qwen3_30b_a3b_moe"


def _tile(spec, name):
    return next(tile for tile in spec.tiles if tile.name == name)


def _relaxed_rows(batch_size: int) -> int:
    return batch_size * 8 // 128 + 129


def test_public_spec_types_are_tvm_owned_and_moe_lowering_is_removed():
    for name in ("VarSpec", "TensorSpec", "EventSpec", "TileSpec", "TileImpl", "KernelSpec"):
        assert getattr(megakernel_dsl, name) is getattr(tvm_dsl, name)
    assert megakernel_dsl.ScalarSpec is tvm_dsl.ScalarSpec

    for removed_module in (
        "tirx_kernels.megakernel.dsl.spec",
        "tirx_kernels.megakernel.dsl.kernel",
        "tirx_kernels.megakernel.dsl._expr",
        "tirx_kernels.megakernel.dsl.lowering",
    ):
        assert importlib.util.find_spec(removed_module) is None
    for removed in (
        "MoeLowerer",
        "NormalizedPlan",
        "lower_moe_to_tirx",
        "make_moe_plan",
        "policy_for_scheduler",
        "ConstExpr",
        "ScalarLoadExpr",
    ):
        assert not hasattr(megakernel_dsl, removed)


def test_complete_six_stage_graph_is_pure_logical_native_dsl():
    batch_size = 512
    spec = build_moe_graph(_CONFIG, batch_size)
    max_rows = _max_rows(batch_size)
    assert isinstance(spec, tvm_dsl.KernelSpec)
    assert spec.validate() is spec
    assert [tile.name for tile in spec.tiles] == [
        "gating",
        "topk",
        "align",
        "count_sort",
        "gate_up_silu",
        "down",
    ]
    assert list(spec.events) == [
        "gating_done",
        "topk_done",
        "align_done",
        "count_sort_done",
        "gate_up_done",
        "down_dispatch_done",
    ]
    # The production kernel parameter order, including the unused buffers.
    assert list(spec.tensors) == [
        "hidden_state",
        "residual",
        "output",
        "gate_weight",
        "gate_up_weight",
        "down_weight",
        "gating_output",
        "topk_weights",
        "topk_indices",
        "sorted_token_ids",
        "expert_ids",
        "num_valid_tokens",
        "num_tokens_post_pad",
        "cumsum_buffer",
        "reordered_hidden_state",
        "gate_up_output",
        "silu_mul_output",
        "topk_reduce_output",
    ]
    # The routed-row count is a runtime scalar over the align-published count.
    scalar = spec.scalars["num_tokens_post_pad"]
    assert scalar.source == (spec.tensors["num_tokens_post_pad"], (0,))
    assert scalar.range == (128, max_rows * 128)
    routed_rows = scalar // 128
    assert _tile(spec, "gate_up_silu").tile_num == (routed_rows, 12, 1)
    assert _tile(spec, "down").tile_num == (routed_rows, 16, 1)
    assert not spec.vars
    # The drain event is declared once and referenced by no tile.
    drain = spec.events["down_dispatch_done"]
    assert drain.shape == (1,)
    assert drain.init_count == max_rows * 16
    assert drain.attrs["megakernel.drain"] is True
    for tile in spec.tiles:
        assert all(event is not drain for event, _ in (*tile.waits, *tile.notifies))
    # Static gate-up event facts.
    assert spec.events["gate_up_done"].shape == (_relaxed_rows(batch_size),)
    assert spec.events["gate_up_done"].init_count == 12
    # Physical facts live in tile attrs.
    assert _tile(spec, "gating").attrs["megakernel.job_id"] == JobType.MOE_GATING.value
    assert _tile(spec, "down").attrs["megakernel.job_id"] == JobType.MOE_GROUP_GEMM_DOWN.value
    assert (
        _tile(spec, "gate_up_silu").attrs["megakernel.job_id"]
        == JobType.MOE_GROUP_GEMM_GATE_UP_SILU.value
    )
    for name in ("gate_up_silu", "down"):
        assert _tile(spec, name).attrs["megakernel.run_predicate"] == (0, "lt", routed_rows)


def test_six_concrete_tile_impls_extend_tasks_and_carry_scope_metadata():
    spec = build_moe_graph(_CONFIG, 4)
    assert tuple(type(tile.impl) for tile in spec.tiles) == (
        GatingTileImpl,
        TopkTileImpl,
        AlignTileImpl,
        CountSortTileImpl,
        GateUpSiluTileImpl,
        DownTileImpl,
    )
    scopes = {tile.name: tile.impl for tile in spec.tiles}
    assert scopes["topk"].wait_level == "cta"
    assert scopes["align"].wait_level == "cta"
    assert scopes["count_sort"].wait_level == "cta"
    assert scopes["gate_up_silu"].wait_level == "warp"
    assert scopes["down"].wait_level == "warp"
    assert scopes["gating"].notify_scope == ("warpgroup", 0)
    assert scopes["topk"].notify_scope == ("cta", 0)
    assert scopes["align"].notify_scope == ("thread", 0)
    assert scopes["count_sort"].notify_scope == ("cta", 0)
    assert scopes["gate_up_silu"].notify_scope == ("warpgroup", 0)
    # The dynamic pre-notify scopes diverge from the notify scopes only where
    # the hand-written kernel hand-tunes them (topk, align, gate_up, down).
    assert scopes["gating"].pre_notify_scope is None
    assert scopes["count_sort"].pre_notify_scope is None
    assert scopes["topk"].pre_notify_scope == ("thread", 0)
    assert scopes["align"].pre_notify_scope == ("cta", 0)
    assert scopes["gate_up_silu"].pre_notify_scope == ("warp", 0)
    assert scopes["down"].pre_notify_scope == ("warp", 0)
    assert all(impl.notify_release for impl in scopes.values())
    assert len({impl.job_type for impl in scopes.values()}) == 6
    # The gemm family shares one class-resource group, as in production.
    assert (
        scopes["gating"].class_group
        is scopes["gate_up_silu"].class_group
        is scopes["down"].class_group
    )
    assert scopes["topk"].class_group is None
    assert scopes["align"].hoisted_views == (("topk_indices_flat", "topk_indices", (-1,)),)
    assert scopes["down"].hoisted_views == (("topk_weights_flat", "topk_weights", (-1,)),)


def test_unfused_spec_collapses_only_the_gate_up_event():
    batch_size = 512
    spec = build_moe_graph(_CONFIG, batch_size, unfused=True)
    max_rows = _max_rows(batch_size)
    assert spec.events["gate_up_done"].shape == (1,)
    assert spec.events["gate_up_done"].init_count == max_rows * 12
    assert _tile(spec, "gate_up_silu").notifies[0][1](7, 0, 0) == (0,)
    assert _tile(spec, "down").waits[0][1](7, 0, 0) == (0,)
    fused = build_moe_graph(_CONFIG, batch_size)
    assert _tile(fused, "gate_up_silu").notifies[0][1](7, 0, 0) == (7,)
    assert _tile(fused, "down").waits[0][1](7, 0, 0) == (7,)
    # Every other fact is unchanged.
    assert [event.name for event in spec.events.values()] == [
        event.name for event in fused.events.values()
    ]
    assert spec.validate() is spec


def test_moe_lowering_options_match_production_abi():
    options = moe_lowering_options("dynamic", 512)
    assert options.scheduler == "dynamic"
    assert options.attrs["init_event_job_id"] == JobType.INIT_ETENSOR.value
    assert options.attrs["wait_event_init_job_id"] == JobType.WAIT_ETENSOR_INIT.value
    assert options.attrs["end_job_id"] == JobType.END.value
    assert options.attrs["emit_profiler_param"] is True
    assert options.attrs["tile_coalescing"] == {"down": 4}
    assert moe_lowering_options("dynamic", 1).attrs["tile_coalescing"] == {"down": 1}
    for scheduler in ("static", "unfused"):
        options = moe_lowering_options(scheduler, 512)
        assert options.scheduler == "static"
        assert options.attrs["tile_coalescing"] == {"down": 1}
    with pytest.raises(ValueError, match="scheduler"):
        moe_lowering_options("bogus", 1)


@pytest.mark.parametrize("scheduler", ["static", "unfused"])
@pytest.mark.parametrize("batch_size", [1, 4, 128, 512, 2048])
def test_static_host_queue_matches_production(batch_size, scheduler):
    build = build_moe_kernel(_CONFIG, batch_size, scheduler)
    relaxed = _relaxed_rows(batch_size)

    manual_tasks = [(event_idx, 0, 0, JobType.INIT_ETENSOR.value) for event_idx in range(7)]
    push_moe_tasks(manual_tasks, batch_size, _CONFIG, insert_wait_etensor_init=True)
    assert list(build.central_tasks) == manual_tasks

    manual_queue = generate_exec_queue_moe(batch_size, _CONFIG, 7, "static")
    np.testing.assert_array_equal(build.exec_queue, manual_queue.numpy())
    expected_workspace = (relaxed + 6) if scheduler == "static" else 7
    assert build.event_workspace_size == expected_workspace
    assert build.scheduler == "static"
    assert build.end_task_type == JobType.END.value
    assert build.init_event_job_id == JobType.INIT_ETENSOR.value
    assert build.wait_event_init_job_id == JobType.WAIT_ETENSOR_INIT.value


@pytest.mark.parametrize("batch_size", [1, 4, 128, 512, 2048])
def test_dynamic_seed_queue_matches_production(batch_size):
    build = build_moe_kernel(_CONFIG, batch_size, "dynamic")
    relaxed = _relaxed_rows(batch_size)

    manual_queue = generate_exec_queue_moe(batch_size, _CONFIG, 6, "dynamic")
    np.testing.assert_array_equal(build.queue_tasks, manual_queue.tasks)
    np.testing.assert_array_equal(build.queue_head, manual_queue.head)
    np.testing.assert_array_equal(build.queue_tail, manual_queue.tail)
    # Six event-init seeds (five logical events plus the drain) plus gating.
    assert len(build.central_tasks) == 6 + ((batch_size + 127) // 128) * 4
    assert build.event_workspace_size == relaxed + 5
    (drain,) = build.drain_events
    assert drain.name == "down_dispatch_done"
    assert drain.workspace_offset == relaxed + 4
    assert drain.static_count is None
    assert drain.runtime_initialized


@pytest.mark.parametrize("scheduler", ["static", "unfused", "dynamic"])
def test_build_produces_production_abi_modules(scheduler):
    build = build_moe_kernel(_CONFIG, 4, scheduler)
    func = build.module[_KERNEL_NAME]
    # 18 tensors + event workspace + queue arrays + profiler buffer.
    expected_params = 18 + 1 + (1 if scheduler != "dynamic" else 3) + 1
    assert len(func.params) == expected_params
    assert not build.profiler_on


def test_builder_rejects_invalid_specs():
    with pytest.raises(ValueError, match="Qwen3-30B-A3B"):
        build_moe_graph({**_CONFIG, "HIDDEN_SIZE": 4096}, 4)

    # A run predicate must guard a scalar-dependent tile axis.
    spec = build_moe_graph(_CONFIG, 4)
    _tile(spec, "gate_up_silu").attrs["megakernel.run_predicate"] = (1, "lt", 1)
    with pytest.raises(ValueError, match="run_predicate"):
        build_runtime_kernel(spec, moe_lowering_options("static", 4))

    # The dynamic builder rejects dependency cycles.
    spec = build_moe_graph(_CONFIG, 4)
    _tile(spec, "topk").wait(spec.events["gate_up_done"], lambda m, n, k: (0,))
    with pytest.raises(ValueError, match="acyclic"):
        build_runtime_kernel(spec, moe_lowering_options("dynamic", 4))


# ---------------------------------------------------------------------------
# Structural parity with the hand-written kernel
# ---------------------------------------------------------------------------

#: Tolerated divergence: production declares the max-tokens-shaped tensors
#: with the symbolic ``max_num_tokens_padded`` var while the builder
#: concretizes every shape, so internal view declarations differ in shape.
_SHAPE_ONLY_RE = re.compile(r"(\.shape\[\d+\]|\.def\.extents\[\d+\])$")

#: Tolerated divergence: the DSL build publishes completions with
#: device-scope release atomics while the zero-diff manual path keeps its
#: plain ``atomicAdd`` (F2, intentional memory-model fix).
_RELEASE_CALL_RE = re.compile(r"atomic_add_int32(_release)?\b")

#: Tolerated divergence: the migrated ``stg_local`` store helper dropped the
#: unused ``pe`` parameter, so tvm-side calls have two arguments against the
#: manual three (LOW; call sites only, inside dynamic push paths).
_STG_CALL_RE = re.compile(r"\bstg_local\b")

#: The push block is nine statements: six axis definitions, the
#: ``new_scope_id`` local and its store, and the push branch itself.
_PUSH_BLOCK_LEN = 9


def _unwrap_seq(body):
    while not hasattr(body, "seq"):
        body = body.body
    return body


def _loop_index(seq_stmt):
    return next(i for i, stmt in enumerate(seq_stmt.seq) if isinstance(stmt, While))


def _first_divergence_path(lhs, rhs):
    try:
        tvm.ir.assert_structural_equal(lhs, rhs, map_free_vars=True)
        return None
    except Exception as err:
        message = str(err)
        if "Access path:" not in message:
            return "<no path>"
        return message.split("Access path:")[1].split("\n")[0].strip()


def _dispatch_branches(chain):
    branches = []
    while hasattr(chain, "then_case"):
        branches.append(chain.then_case)
        chain = getattr(chain, "else_case", None)
        if chain is None:
            break
    return branches


def _classify_pair(tvm_stmt, manual_stmt):
    """Classify one statement pair: equal, a tolerated class, or a violation."""

    path = _first_divergence_path(tvm_stmt, manual_stmt)
    if path is None:
        return "equal"
    if _SHAPE_ONLY_RE.search(path):
        return "shape"
    # Push bodies contain both the (matching) pre-notify atomic and the
    # store helper; the release fence only ever differs in notify bodies.
    if any(_STG_CALL_RE.search(repr(stmt)) for stmt in (tvm_stmt, manual_stmt)):
        return "stg_local"
    if any(_RELEASE_CALL_RE.search(repr(stmt)) for stmt in (tvm_stmt, manual_stmt)):
        return "release"
    return f"VIOLATION:{path}"


def _branch_census(tvm_branch, manual_branch):
    """Classify every aligned statement pair of two dispatch branches."""

    assert len(tvm_branch.seq) == len(manual_branch.seq), (
        f"branch statement counts diverge: {len(tvm_branch.seq)} vs {len(manual_branch.seq)}"
    )
    return [
        (index, _classify_pair(tvm_stmt, manual_stmt))
        for index, (tvm_stmt, manual_stmt) in enumerate(zip(tvm_branch.seq, manual_branch.seq))
    ]


def _relocate_push_to_front(branch):
    """Move a post-run pre-notify-and-push block back to the branch front (F1)."""

    stmts = list(branch.seq)
    push_index = next(
        i
        for i, stmt in enumerate(stmts)
        if isinstance(stmt, IfThenElse) and "new_scope_id" in repr(stmt)
    )
    start = push_index - _PUSH_BLOCK_LEN + 1
    block = stmts[start : push_index + 1]
    expected = (["ScopeIdDefStmt"] * 6) + ["AllocBuffer", "BufferStore", "IfThenElse"]
    assert [type(stmt).__name__ for stmt in block] == expected
    return SeqStmt(block + stmts[:start] + stmts[push_index + 1 :])


@pytest.mark.parametrize("scheduler", ["static", "unfused", "dynamic"])
def test_structural_parity_with_manual_kernel(scheduler):
    batch_size = 4
    build = build_moe_kernel(_CONFIG, batch_size, scheduler)
    tvm_body = _unwrap_seq(build.module[_KERNEL_NAME].body)
    mk = MegaKernelMOE(_CONFIG, 1, False)
    mk._compile_batch_size = batch_size
    manual_body = _unwrap_seq(mk.get_module(scheduler)["main"].body)
    tvm_loop = _loop_index(tvm_body)
    manual_loop = _loop_index(manual_body)

    # Everything around the dispatch loop is structurally identical: wrapper
    # lifecycle, class/device init, event setup, scheduler init, hoisted
    # views, scheduler advance, and the finalize tail.
    segments = (
        (tvm_body.seq[:tvm_loop], manual_body.seq[:manual_loop], "prefix"),
        (tvm_body.seq[tvm_loop].body.seq[1:], manual_body.seq[manual_loop].body.seq[1:], "tail"),
        (tvm_body.seq[tvm_loop + 1 :], manual_body.seq[manual_loop + 1 :], "post"),
    )
    for tvm_stmts, manual_stmts, label in segments:
        path = _first_divergence_path(SeqStmt(list(tvm_stmts)), SeqStmt(list(manual_stmts)))
        assert path is None, f"{label} diverges at {path}"

    # Per dispatch branch: identical except the pinned intentional classes.
    names = ["gating", "topk", "align", "count_sort", "gate_up_silu", "down", "init", "wait_init"]
    tvm_branches = _dispatch_branches(tvm_body.seq[tvm_loop].body.seq[0])
    manual_branches = _dispatch_branches(manual_body.seq[manual_loop].body.seq[0])
    dynamic = scheduler == "dynamic"
    allowed = {"shape", "release"} | ({"stg_local"} if dynamic else set())
    seen = set()
    for name, tvm_branch, manual_branch in zip(names, tvm_branches, manual_branches):
        if dynamic and name == "count_sort":
            # F1: the scalar-count push fires post-run with the full-count
            # trigger, so the push block sits after the run on the tvm side.
            tvm_branch = _relocate_push_to_front(tvm_branch)
        census = _branch_census(tvm_branch, manual_branch)
        violations = [(index, kind) for index, kind in census if kind not in allowed | {"equal"}]
        assert not violations, f"branch {name} diverges: {violations}"
        seen.update(kind for _, kind in census if kind != "equal")
    # The pinned classes must not only be tolerated but present (they are the
    # contract of this comparison): shape views in the align/gemm branches,
    # release atomics in every notify path, and the dynamic-only store-arity
    # divergence in the push paths.
    assert "shape" in seen
    assert "release" in seen
    if dynamic:
        assert "stg_local" in seen
