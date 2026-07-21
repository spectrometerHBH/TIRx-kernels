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

"""Concrete MoE TileImpls that directly extend the existing physical tasks.

The adapters carry only metadata and tensor plumbing; all scheduling facts
(endpoint scopes, profiler events) are class attributes consumed by
``tvm.megakernel.transform.build_runtime_kernel``.  The values mirror the
hand-written kernel in ``tirx_kernels.megakernel.moe``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from tirx_kernels.megakernel.tile_tasks import (
    CountAndSortExpertTokens,
    GemmTile,
    GroupGEMMSiluTile,
    GroupGEMMTileSM100,
    MOEAlignTile,
    TopkSoftmaxTile,
)
from tirx_kernels.megakernel.utils.config import JobType, ProfileEventType
from tvm.megakernel.dsl import TensorSpec, TileImpl
from tvm.script import tirx as T


class _GemmTileClassGroup(GroupGEMMTileSM100):
    """Class-resource group for the gemm-family MoE tiles.

    The hand-written kernel registers every ``GemmTile`` instance under
    ``GroupGEMMTile`` so the shared tcgen05/barrier resources initialize once
    for gating, gate_up_silu, and down.  This group class exposes the tvm
    wrapper's class hooks for that shared registration.
    """

    @classmethod
    @T.inline
    def init_shared_resources(cls, smem_manager):
        GroupGEMMTileSM100.class_init(smem_manager)

    @classmethod
    @T.inline
    def finalize_shared_resources(cls, smem_manager):
        GroupGEMMTileSM100.class_finalize()


class _MoeTileMetadataMixin(TileImpl):
    """Bridge the tvm runtime wrapper hooks onto the production tile tasks."""

    implementation: str
    job_type: int
    profile_event: ProfileEventType
    # The hand-written kernel notifies without a release fence (plain
    # atomicAdd); ordering comes from scope-level syncs and acquire waits.
    notify_release: ClassVar[bool] = False

    @classmethod
    @T.inline
    def init_shared_resources(cls, smem_manager):
        cls.class_init(smem_manager)

    @classmethod
    @T.inline
    def finalize_shared_resources(cls, smem_manager):
        cls.class_finalize()

    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        self.init(smem_manager)

    @T.inline
    def prefetch(self, m_idx, n_idx, k_idx):
        """The hand-written MoE kernel issues no prefetches."""

    def _init_logical(self, config: Mapping[str, Any], batch_size: int) -> None:
        self.config = dict(config)
        self.batch_size = batch_size
        # Mirrors the hand-written kernel: the gemm tiles read the runtime
        # num_valid_tokens buffer only at batch >= 512, otherwise they use
        # their static numel upper bound (None at parse time).
        self.dynamic_gemm_size = batch_size >= 512
        # Rebound by the builder to the wrapper profiler (None when off).
        self.profiler = None


class GatingTileImpl(_MoeTileMetadataMixin, GemmTile):
    implementation = "gating"
    job_type = JobType.MOE_GATING.value
    profile_event = ProfileEventType.MOE_GATING
    notify_scope: ClassVar[tuple[str, int]] = ("warpgroup", 0)
    class_group: ClassVar[type | None] = _GemmTileClassGroup

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
        self._init_logical(config, 0)
        self.hidden_state = tensors["hidden_state"]
        self.gate_weight = tensors["gate_weight"]
        self.gating_output = tensors["gating_output"]

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


class TopkTileImpl(_MoeTileMetadataMixin, TopkSoftmaxTile):
    implementation = "topk"
    job_type = JobType.MOE_TOPK_SOFTMAX.value
    profile_event = ProfileEventType.TOPK_SOFTMAX
    notify_scope: ClassVar[tuple[str, int]] = ("cta", 0)
    pre_notify_scope: ClassVar[tuple[str, int] | None] = ("thread", 0)

    def __init__(
        self, config: Mapping[str, Any], batch_size: int, tensors: Mapping[str, TensorSpec]
    ):
        TopkSoftmaxTile.__init__(
            self, config["NUM_EXPERTS"], batch_size, config["NUM_EXPERTS_PER_TOK"], dtype="float32"
        )
        self._init_logical(config, batch_size)
        self.gating_output = tensors["gating_output"]
        self.topk_weights = tensors["topk_weights"]
        self.topk_indices = tensors["topk_indices"]

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


class AlignTileImpl(_MoeTileMetadataMixin, MOEAlignTile):
    implementation = "align"
    job_type = JobType.MOE_ALIGN.value
    profile_event = ProfileEventType.MOE_ALIGN
    notify_scope: ClassVar[tuple[str, int]] = ("thread", 0)
    pre_notify_scope: ClassVar[tuple[str, int] | None] = ("cta", 0)
    hoisted_views: ClassVar[tuple] = (("topk_indices_flat", "topk_indices", (-1,)),)

    def __init__(
        self, config: Mapping[str, Any], batch_size: int, tensors: Mapping[str, TensorSpec]
    ):
        numel = config["NUM_EXPERTS_PER_TOK"] * batch_size
        MOEAlignTile.__init__(self, config["NUM_EXPERTS"], numel, 128, pad_sorted_token_ids=True)
        self._init_logical(config, batch_size)
        self.topk_indices = tensors["topk_indices"]
        self.topk_indices_flat = None
        self.sorted_token_ids = tensors["sorted_token_ids"]
        self.expert_ids = tensors["expert_ids"]
        self.num_tokens_post_pad = tensors["num_tokens_post_pad"]
        self.cumsum_buffer = tensors["cumsum_buffer"]
        self.num_valid_tokens = tensors["num_valid_tokens"]

    def run(self, m_idx, n_idx, k_idx):
        MOEAlignTile.run(
            self,
            m_idx,
            n_idx,
            k_idx,
            self.topk_indices_flat,
            self.sorted_token_ids,
            self.expert_ids,
            self.num_tokens_post_pad,
            self.cumsum_buffer,
            self.num_valid_tokens,
        )


class CountSortTileImpl(_MoeTileMetadataMixin, CountAndSortExpertTokens):
    implementation = "count_sort"
    job_type = JobType.MOE_COUNT_AND_SORT.value
    profile_event = ProfileEventType.COUNT_AND_SORT
    notify_scope: ClassVar[tuple[str, int]] = ("cta", 0)
    hoisted_views: ClassVar[tuple] = (("topk_indices_flat", "topk_indices", (-1,)),)

    def __init__(
        self, config: Mapping[str, Any], batch_size: int, tensors: Mapping[str, TensorSpec]
    ):
        numel = config["NUM_EXPERTS_PER_TOK"] * batch_size
        CountAndSortExpertTokens.__init__(
            self, numel, config["HIDDEN_SIZE"], config["NUM_EXPERTS_PER_TOK"]
        )
        self._init_logical(config, batch_size)
        self.topk_indices = tensors["topk_indices"]
        self.topk_indices_flat = None
        self.sorted_token_ids = tensors["sorted_token_ids"]
        self.cumsum_buffer = tensors["cumsum_buffer"]
        self.hidden_state = tensors["hidden_state"]
        self.reordered_hidden_state = tensors["reordered_hidden_state"]

    def run(self, m_idx, n_idx, k_idx):
        CountAndSortExpertTokens.run(
            self,
            m_idx,
            n_idx,
            k_idx,
            self.topk_indices_flat,
            self.sorted_token_ids,
            self.cumsum_buffer,
            self.hidden_state,
            self.reordered_hidden_state,
        )


class GateUpSiluTileImpl(_MoeTileMetadataMixin, GroupGEMMSiluTile):
    implementation = "gate_up_silu"
    job_type = JobType.MOE_GROUP_GEMM_GATE_UP_SILU.value
    profile_event = ProfileEventType.GROUP_GEMM_GATE_UP_SILU
    wait_level: ClassVar[str] = "warp"
    notify_scope: ClassVar[tuple[str, int]] = ("warpgroup", 0)
    pre_notify_scope: ClassVar[tuple[str, int] | None] = ("warp", 0)
    class_group: ClassVar[type | None] = _GemmTileClassGroup
    hoisted_views: ClassVar[tuple] = (("topk_weights_flat", "topk_weights", (-1,)),)

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
        self._init_logical(config, batch_size)
        self.reordered_hidden_state = tensors["reordered_hidden_state"]
        self.gate_up_weight = tensors["gate_up_weight"]
        self.silu_mul_output = tensors["silu_mul_output"]
        self.expert_ids = tensors["expert_ids"]
        self.topk_weights = tensors["topk_weights"]
        self.topk_weights_flat = None
        self.sorted_token_ids = tensors["sorted_token_ids"]
        self.num_valid_tokens = tensors["num_valid_tokens"]

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
            self.topk_weights_flat,
            self.sorted_token_ids,
            self.num_valid_tokens if self.dynamic_gemm_size else None,
            self.profiler,
        )


class DownTileImpl(_MoeTileMetadataMixin, GroupGEMMTileSM100):
    implementation = "down"
    job_type = JobType.MOE_GROUP_GEMM_DOWN.value
    profile_event = ProfileEventType.GROUP_GEMM_DOWN
    wait_level: ClassVar[str] = "warp"
    # The down tile has no completion notify; this scopes the dynamic drain
    # pre-notify-and-push instead (the hand-written kernel uses warp/warp).
    pre_notify_scope: ClassVar[tuple[str, int] | None] = ("warp", 0)
    class_group: ClassVar[type | None] = _GemmTileClassGroup
    hoisted_views: ClassVar[tuple] = (("topk_weights_flat", "topk_weights", (-1,)),)

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
        self._init_logical(config, batch_size)
        self.silu_mul_output = tensors["silu_mul_output"]
        self.down_weight = tensors["down_weight"]
        self.topk_reduce_output = tensors["topk_reduce_output"]
        self.expert_ids = tensors["expert_ids"]
        self.topk_weights = tensors["topk_weights"]
        self.topk_weights_flat = None
        self.sorted_token_ids = tensors["sorted_token_ids"]
        self.num_valid_tokens = tensors["num_valid_tokens"]

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
            self.topk_weights_flat,
            self.sorted_token_ids,
            self.num_valid_tokens if self.dynamic_gemm_size else None,
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
