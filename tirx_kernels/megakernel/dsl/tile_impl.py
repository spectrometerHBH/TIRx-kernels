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
"""Concrete MoE adapters for logical megakernel tiles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tirx_kernels.megakernel.tile_tasks import (
    CountAndSortExpertTokens,
    GemmTile,
    GroupGEMMSiluTile,
    GroupGEMMTileSM100,
    MOEAlignTile,
    TopkSoftmaxTile,
)
from tirx_kernels.megakernel.utils.config import JobType, ProfileEventType
from tvm.megakernel.dsl import TileImpl


class _MoeTileImpl(TileImpl):
    """Bind one existing tile task to the native logical ``TileImpl`` API."""

    implementation: str
    job_type: int
    profile_event_type: ProfileEventType
    owner_attribute: str

    def __init__(self, tile_task):
        super().__init__()
        self.tile_task = tile_task
        self._owner = None
        self._context = None

    def register(self, owner):
        """Register the held task with the existing megakernel lifecycle."""

        self._owner = owner
        registered = owner._add_tile(self.tile_task, self.profile_event_type)
        setattr(owner, self.owner_attribute, registered)

    def bind_context(self, context: Mapping[str, Any]):
        self._context = context

    def _bound(self):
        if self._owner is None or self._context is None:
            raise RuntimeError(f"{type(self).__name__} must be bound before run()")
        return self._owner, self._context


class GatingTileImpl(_MoeTileImpl):
    implementation = "gating"
    job_type = JobType.MOE_GATING.value
    profile_event_type = ProfileEventType.MOE_GATING
    owner_attribute = "gate"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(
            GemmTile(
                config["NUM_EXPERTS"],
                config["HIDDEN_SIZE"],
                "float16",
                "float16",
                config["GATING_SPLIT_K_FACTOR"],
                128,
                128,
                use_tma_reduce=True,
            )
        )

    def run(self, m_idx, n_idx, k_idx):
        owner, context = self._bound()
        owner.run_tile(
            self.tile_task,
            m_idx,
            n_idx,
            k_idx,
            context["hidden_state"],
            context["gate_weight"],
            context["gating_output"],
            owner.profiler,
        )


class TopkTileImpl(_MoeTileImpl):
    implementation = "topk"
    job_type = JobType.MOE_TOPK_SOFTMAX.value
    profile_event_type = ProfileEventType.TOPK_SOFTMAX
    owner_attribute = "topk_softmax"

    def __init__(self, config: Mapping[str, Any], batch_size: int):
        super().__init__(
            TopkSoftmaxTile(
                config["NUM_EXPERTS"], batch_size, config["NUM_EXPERTS_PER_TOK"], dtype="float32"
            )
        )

    def run(self, m_idx, n_idx, k_idx):
        owner, context = self._bound()
        owner.run_tile(
            self.tile_task,
            m_idx,
            n_idx,
            k_idx,
            context["gating_output"],
            context["topk_weights"],
            context["topk_indices"],
            renormalize=False,
        )


class AlignTileImpl(_MoeTileImpl):
    implementation = "align"
    job_type = JobType.MOE_ALIGN.value
    profile_event_type = ProfileEventType.MOE_ALIGN
    owner_attribute = "align"

    def __init__(self, config: Mapping[str, Any], batch_size: int):
        numel = config["NUM_EXPERTS_PER_TOK"] * batch_size
        super().__init__(MOEAlignTile(config["NUM_EXPERTS"], numel, 128, pad_sorted_token_ids=True))

    def run(self, m_idx, n_idx, k_idx):
        owner, context = self._bound()
        owner.run_tile(
            self.tile_task,
            m_idx,
            n_idx,
            k_idx,
            context["topk_indices_flat"],
            context["sorted_token_ids"],
            context["expert_ids"],
            context["num_tokens_post_pad"],
            context["cumsum_buffer"],
            context["num_valid_tokens"],
        )


class CountSortTileImpl(_MoeTileImpl):
    implementation = "count_sort"
    job_type = JobType.MOE_COUNT_AND_SORT.value
    profile_event_type = ProfileEventType.COUNT_AND_SORT
    owner_attribute = "count_and_sort_expert_tokens"

    def __init__(self, config: Mapping[str, Any], batch_size: int):
        numel = config["NUM_EXPERTS_PER_TOK"] * batch_size
        super().__init__(
            CountAndSortExpertTokens(numel, config["HIDDEN_SIZE"], config["NUM_EXPERTS_PER_TOK"])
        )

    def run(self, m_idx, n_idx, k_idx):
        owner, context = self._bound()
        owner.run_tile(
            self.tile_task,
            m_idx,
            n_idx,
            k_idx,
            context["topk_indices_flat"],
            context["sorted_token_ids"],
            context["cumsum_buffer"],
            context["hidden_state"],
            context["reordered_hidden_state"],
        )


class GateUpSiluTileImpl(_MoeTileImpl):
    implementation = "gate_up_silu"
    job_type = JobType.MOE_GROUP_GEMM_GATE_UP_SILU.value
    profile_event_type = ProfileEventType.GROUP_GEMM_GATE_UP_SILU
    owner_attribute = "group_gemm_gate_up_silu"

    def __init__(self, config: Mapping[str, Any], batch_size: int):
        numel = config["NUM_EXPERTS_PER_TOK"] * batch_size
        super().__init__(
            GroupGEMMSiluTile(
                config["INTERMEDIATE_SIZE"] * 2,
                config["HIDDEN_SIZE"],
                config["NUM_EXPERTS"],
                config["NUM_EXPERTS_PER_TOK"],
                numel,
                "float16",
                "float16",
                low_batch=batch_size < 2048,
            )
        )

    def run(self, m_idx, n_idx, k_idx):
        owner, context = self._bound()
        owner.run_tile(
            self.tile_task,
            m_idx,
            n_idx,
            k_idx,
            context["reordered_hidden_state"],
            context["gate_up_weight"],
            context["silu_mul_output"],
            context["expert_ids"],
            context["topk_weights_flat"],
            context["sorted_token_ids"],
            context["num_valid_tokens"],
            owner.profiler,
        )


class DownTileImpl(_MoeTileImpl):
    implementation = "down"
    job_type = JobType.MOE_GROUP_GEMM_DOWN.value
    profile_event_type = ProfileEventType.GROUP_GEMM_DOWN
    owner_attribute = "group_gemm_down"

    def __init__(self, config: Mapping[str, Any], batch_size: int):
        numel = config["NUM_EXPERTS_PER_TOK"] * batch_size
        super().__init__(
            GroupGEMMTileSM100(
                config["HIDDEN_SIZE"],
                config["INTERMEDIATE_SIZE"],
                config["NUM_EXPERTS"],
                config["NUM_EXPERTS_PER_TOK"],
                numel,
                "float16",
                "float16",
                acc_output=True,
                low_batch=batch_size < 2048,
            )
        )

    def run(self, m_idx, n_idx, k_idx):
        owner, context = self._bound()
        owner.run_tile(
            self.tile_task,
            m_idx,
            n_idx,
            k_idx,
            context["silu_mul_output"],
            context["down_weight"],
            context["topk_reduce_output"],
            context["expert_ids"],
            context["topk_weights_flat"],
            context["sorted_token_ids"],
            context["num_valid_tokens"],
            owner.profiler,
        )


__all__ = [
    "AlignTileImpl",
    "CountSortTileImpl",
    "DownTileImpl",
    "GateUpSiluTileImpl",
    "GatingTileImpl",
    "TopkTileImpl",
]
