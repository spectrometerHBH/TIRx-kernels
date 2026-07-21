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

"""Concrete MoE TileImpls that directly extend the existing physical tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tirx_kernels.megakernel.kernels import (
    CountAndSortExpertTokens,
    GemmTile,
    GroupGEMMSiluTile,
    GroupGEMMTileSM100,
    MOEAlignTile,
    TopkSoftmaxTile,
)
from tirx_kernels.megakernel.utils.config import JobType, ProfileEventType
from tvm.megakernel.dsl import TensorSpec, TileImpl


class _MoeTileMetadataMixin(TileImpl):
    """Attach shared MoE metadata to a directly inherited physical tile."""

    implementation: str
    job_type: int
    profile_event_type: ProfileEventType

    def _init_logical(
        self,
        config: Mapping[str, Any],
        tensor_bindings: Mapping[str, TensorSpec | tuple[TensorSpec, bool]],
    ) -> None:
        self.config = dict(config)
        self.tensor_bindings = dict(tensor_bindings)


class GatingTileImpl(GemmTile, _MoeTileMetadataMixin):
    implementation = "gating"
    job_type = JobType.MOE_GATING.value
    profile_event_type = ProfileEventType.MOE_GATING

    def __init__(self, config: Mapping[str, Any], tensors: Mapping[str, TensorSpec]):
        GemmTile.__init__(
            self,
            config["NUM_EXPERTS"],
            config["HIDDEN_SIZE"],
            "float16",
            "float16",
            config["GATING_SPLIT_K_FACTOR"],
            128,
            128,
            use_tma_reduce=True,
        )
        self._init_logical(
            config,
            {
                "hidden_state": tensors["hidden_state"],
                "gate_weight": tensors["gate_weight"],
                "gating_output": tensors["gating_output"],
            },
        )

    def run(self, m_idx, n_idx, k_idx):
        GemmTile.run(
            self,
            m_idx,
            n_idx,
            k_idx,
            self.hidden_state,
            self.gate_weight,
            self.gating_output,
            self.profiler,
        )


class TopkTileImpl(TopkSoftmaxTile, _MoeTileMetadataMixin):
    implementation = "topk"
    job_type = JobType.MOE_TOPK_SOFTMAX.value
    profile_event_type = ProfileEventType.TOPK_SOFTMAX

    def __init__(
        self, config: Mapping[str, Any], batch_size: int, tensors: Mapping[str, TensorSpec]
    ):
        TopkSoftmaxTile.__init__(
            self, config["NUM_EXPERTS"], batch_size, config["NUM_EXPERTS_PER_TOK"], dtype="float32"
        )
        self._init_logical(
            config,
            {
                "gating_output": tensors["gating_output"],
                "topk_weights": tensors["topk_weights"],
                "topk_indices": tensors["topk_indices"],
            },
        )

    def run(self, m_idx, n_idx, k_idx):
        TopkSoftmaxTile.run(
            self,
            m_idx,
            n_idx,
            k_idx,
            self.gating_output,
            self.topk_weights,
            self.topk_indices,
            renormalize=False,
        )


class AlignTileImpl(MOEAlignTile, _MoeTileMetadataMixin):
    implementation = "align"
    job_type = JobType.MOE_ALIGN.value
    profile_event_type = ProfileEventType.MOE_ALIGN

    def __init__(
        self, config: Mapping[str, Any], batch_size: int, tensors: Mapping[str, TensorSpec]
    ):
        numel = config["NUM_EXPERTS_PER_TOK"] * batch_size
        MOEAlignTile.__init__(self, config["NUM_EXPERTS"], numel, 128, pad_sorted_token_ids=True)
        self._init_logical(
            config,
            {
                "topk_indices": (tensors["topk_indices"], True),
                "sorted_token_ids": tensors["sorted_token_ids"],
                "expert_ids": tensors["expert_ids"],
                "num_tokens_post_pad": tensors["num_tokens_post_pad"],
                "cumsum_buffer": tensors["cumsum_buffer"],
                "num_valid_tokens": tensors["num_valid_tokens"],
            },
        )

    def run(self, m_idx, n_idx, k_idx):
        MOEAlignTile.run(
            self,
            m_idx,
            n_idx,
            k_idx,
            self.topk_indices,
            self.sorted_token_ids,
            self.expert_ids,
            self.num_tokens_post_pad,
            self.cumsum_buffer,
            self.num_valid_tokens,
        )


class CountSortTileImpl(CountAndSortExpertTokens, _MoeTileMetadataMixin):
    implementation = "count_sort"
    job_type = JobType.MOE_COUNT_AND_SORT.value
    profile_event_type = ProfileEventType.COUNT_AND_SORT

    def __init__(
        self, config: Mapping[str, Any], batch_size: int, tensors: Mapping[str, TensorSpec]
    ):
        numel = config["NUM_EXPERTS_PER_TOK"] * batch_size
        CountAndSortExpertTokens.__init__(
            self, numel, config["HIDDEN_SIZE"], config["NUM_EXPERTS_PER_TOK"]
        )
        self._init_logical(
            config,
            {
                "topk_indices": (tensors["topk_indices"], True),
                "sorted_token_ids": tensors["sorted_token_ids"],
                "cumsum_buffer": tensors["cumsum_buffer"],
                "hidden_state": tensors["hidden_state"],
                "reordered_hidden_state": tensors["reordered_hidden_state"],
            },
        )

    def run(self, m_idx, n_idx, k_idx):
        CountAndSortExpertTokens.run(
            self,
            m_idx,
            n_idx,
            k_idx,
            self.topk_indices,
            self.sorted_token_ids,
            self.cumsum_buffer,
            self.hidden_state,
            self.reordered_hidden_state,
        )


class GateUpSiluTileImpl(GroupGEMMSiluTile, _MoeTileMetadataMixin):
    implementation = "gate_up_silu"
    job_type = JobType.MOE_GROUP_GEMM_GATE_UP_SILU.value
    profile_event_type = ProfileEventType.GROUP_GEMM_GATE_UP_SILU

    def __init__(
        self, config: Mapping[str, Any], batch_size: int, tensors: Mapping[str, TensorSpec]
    ):
        numel = config["NUM_EXPERTS_PER_TOK"] * batch_size
        GroupGEMMSiluTile.__init__(
            self,
            config["INTERMEDIATE_SIZE"] * 2,
            config["HIDDEN_SIZE"],
            config["NUM_EXPERTS"],
            config["NUM_EXPERTS_PER_TOK"],
            numel,
            "float16",
            "float16",
            low_batch=batch_size < 2048,
        )
        self._init_logical(
            config,
            {
                "reordered_hidden_state": tensors["reordered_hidden_state"],
                "gate_up_weight": tensors["gate_up_weight"],
                "silu_mul_output": tensors["silu_mul_output"],
                "expert_ids": tensors["expert_ids"],
                "topk_weights": (tensors["topk_weights"], True),
                "sorted_token_ids": tensors["sorted_token_ids"],
                "num_valid_tokens": tensors["num_valid_tokens"],
            },
        )

    def run(self, m_idx, n_idx, k_idx):
        GroupGEMMSiluTile.run(
            self,
            m_idx,
            n_idx,
            k_idx,
            self.reordered_hidden_state,
            self.gate_up_weight,
            self.silu_mul_output,
            self.expert_ids,
            self.topk_weights,
            self.sorted_token_ids,
            self.num_valid_tokens,
            self.profiler,
        )


class DownTileImpl(GroupGEMMTileSM100, _MoeTileMetadataMixin):
    implementation = "down"
    job_type = JobType.MOE_GROUP_GEMM_DOWN.value
    profile_event_type = ProfileEventType.GROUP_GEMM_DOWN

    def __init__(
        self, config: Mapping[str, Any], batch_size: int, tensors: Mapping[str, TensorSpec]
    ):
        numel = config["NUM_EXPERTS_PER_TOK"] * batch_size
        GroupGEMMTileSM100.__init__(
            self,
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
        self._init_logical(
            config,
            {
                "silu_mul_output": tensors["silu_mul_output"],
                "down_weight": tensors["down_weight"],
                "topk_reduce_output": tensors["topk_reduce_output"],
                "expert_ids": tensors["expert_ids"],
                "topk_weights": (tensors["topk_weights"], True),
                "sorted_token_ids": tensors["sorted_token_ids"],
                "num_valid_tokens": tensors["num_valid_tokens"],
            },
        )

    def run(self, m_idx, n_idx, k_idx):
        GroupGEMMTileSM100.run(
            self,
            m_idx,
            n_idx,
            k_idx,
            self.silu_mul_output,
            self.down_weight,
            self.topk_reduce_output,
            self.expert_ids,
            self.topk_weights,
            self.sorted_token_ids,
            self.num_valid_tokens,
            self.profiler,
        )


__all__ = [
    "AlignTileImpl",
    "CountSortTileImpl",
    "DownTileImpl",
    "GateUpSiluTileImpl",
    "GatingTileImpl",
    "TopkTileImpl",
]
