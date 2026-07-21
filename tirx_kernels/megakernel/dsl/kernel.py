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

"""Standalone TIRX module emission for the MoE DSL."""

from __future__ import annotations

from tirx_kernels.megakernel.tile_tasks import (
    CountAndSortExpertTokens,
    GemmTile,
    GroupGEMMTileSM100,
    MOEAlignTile,
    TopkSoftmaxTile,
)
from tirx_kernels.megakernel.utils import dynamic_scheduler, static_scheduler
from tirx_kernels.megakernel.utils.base import MegaKernelWrapper
from tirx_kernels.megakernel.utils.config import KernelConfig
from tirx_kernels.megakernel.utils.dynamic_scheduler import DynamicTileScheduler
from tirx_kernels.megakernel.utils.static_scheduler import StaticTileScheduler
from tvm.script import tirx as T

from .lowering._constants import _WARP_SIZE
from .lowering.lowerer import MoeLowerer
from .lowering.policies import policy_for_scheduler
from .spec import KernelSpec


class _MoeDSLKernel(MegaKernelWrapper):
    """Own the complete device module emitted from one normalized MoE plan."""

    MOE_M_PAD_SIZE = 128

    def __init__(self, lowerer: MoeLowerer, profiler_on: bool):
        plan = lowerer._require_plan()
        config = dict(plan.programs[0].tile.impl.config)
        if any(program.tile.impl.config != config for program in plan.programs):
            raise ValueError("all MoE TileImpls must use the same kernel configuration")
        super().__init__(config, 1, profiler_on)
        self.lowerer = lowerer
        self.batch_size = plan.env.batch_size
        self.world_size = 1
        self.HIDDEN_SIZE = config["HIDDEN_SIZE"]
        self.INTERMEDIATE_SIZE = config["INTERMEDIATE_SIZE"]
        self.NUM_EXPERTS = config["NUM_EXPERTS"]
        self.NUM_EXPERTS_PER_TOK = config["NUM_EXPERTS_PER_TOK"]

    def _add_tile(self, tile, profiler_event_type, predicate=True):
        self.tile_attr[tile] = (profiler_event_type, predicate)
        if isinstance(tile, GemmTile):
            tile_class = GroupGEMMTileSM100
        elif isinstance(tile, TopkSoftmaxTile):
            tile_class = TopkSoftmaxTile
        elif isinstance(tile, MOEAlignTile):
            tile_class = MOEAlignTile
        elif isinstance(tile, CountAndSortExpertTokens):
            tile_class = CountAndSortExpertTokens
        else:
            tile_class = tile.__class__
        self.class_list.add(tile_class)
        return tile

    @T.inline
    def fused_body(
        self,
        hidden_state_global,
        residual_global,
        output_global,
        gate_weight_global,
        grp_gate_up_weight_global,
        grp_down_weight_global,
        gating_output_global,
        topk_weights_global,
        topk_indices_global,
        sorted_token_ids_global,
        expert_ids_global,
        num_valid_tokens_global,
        num_tokens_post_pad_global,
        cumsum_buffer_global,
        reordered_hidden_state_global,
        gate_up_output_global,
        silu_mul_output_global,
        topk_reduce_output_global,
        etensor_workspace_global,
        profiler_buffer,
        exec_queue,
        exec_task,
        exec_head,
        exec_tail,
        is_dynamic_sch,
        semaphore_cls: type[static_scheduler.Semaphore | dynamic_scheduler.Semaphore],
        scheduler_cls: type[StaticTileScheduler | DynamicTileScheduler],
    ):
        self.lowerer.register_tiles()
        self.host_init_all()

        T.device_entry()
        lane_id = T.lane_id([_WARP_SIZE])
        T.cta_id([KernelConfig.SM_NUMBER])
        T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
        T.warpgroup_id([KernelConfig.WG_NUMBER])
        T.thread_id([KernelConfig.NUM_THREADS])
        T.thread_id_in_wg([KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER])
        T.alloc_buffer([1], "uint32", scope="local", align=8)
        T.alloc_buffer([1], "uint64", scope="local", align=8)
        self.init_profiler(profiler_buffer)
        smem = T.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
        self.set_smem_manager(KernelConfig.MAX_SMEM_SIZE, 16384, smem.data)

        self.device_init_all(self.smem_manager)
        self.class_init_all(self.smem_manager)
        self.lowerer.init_events(semaphore_cls, etensor_workspace_global)

        if is_dynamic_sch:
            self.init_tile_scheduler(
                True,
                scheduler_cls,
                exec_task,
                exec_head,
                exec_tail,
                self.smem_manager,
                self.profiler,
            )
        else:
            self.init_tile_scheduler(False, scheduler_cls, "moe", exec_queue, self.smem_manager)
        self.smem_manager.init()

        context = T.meta_var(
            {
                "hidden_state": hidden_state_global,
                "residual": residual_global,
                "output": output_global,
                "gate_weight": gate_weight_global,
                "gate_up_weight": grp_gate_up_weight_global,
                "down_weight": grp_down_weight_global,
                "gating_output": gating_output_global,
                "topk_weights": topk_weights_global,
                "topk_indices": topk_indices_global,
                "topk_indices_flat": topk_indices_global.view(-1),
                "topk_weights_flat": topk_weights_global.view(-1),
                "sorted_token_ids": sorted_token_ids_global,
                "expert_ids": expert_ids_global,
                "num_valid_tokens": num_valid_tokens_global,
                "num_tokens_post_pad": num_tokens_post_pad_global,
                "cumsum_buffer": cumsum_buffer_global,
                "reordered_hidden_state": reordered_hidden_state_global,
                "gate_up_output": gate_up_output_global,
                "silu_mul_output": silu_mul_output_global,
                "topk_reduce_output": topk_reduce_output_global,
            }
        )
        while self.tile_scheduler.valid():
            self.lowerer.dispatch_loop_body(context)
            self.tile_scheduler.next_tile()
        if self.profiler_on:
            self.profiler.finalize(lane_id == 0)
        self.class_finalize_all()

    def get_func_static(self, unfused=False):
        compile_batch_size = self.batch_size

        # fmt: off
        @T.prim_func
        def main(
            hidden_state_ptr: T.handle,
            residual_ptr: T.handle,
            output_ptr: T.handle,
            gate_weight_ptr: T.handle,
            grp_gate_up_weight_ptr: T.handle,
            grp_down_weight_ptr: T.handle,
            gating_output_ptr: T.handle,
            topk_weights_ptr: T.handle,
            topk_indices_ptr: T.handle,
            sorted_token_ids_ptr: T.handle,
            expert_ids_ptr: T.handle,
            num_valid_tokens_ptr: T.handle,
            num_tokens_post_pad_ptr: T.handle,
            cumsum_buffer_ptr: T.handle,
            reordered_hidden_state_ptr: T.handle,
            gate_up_output_ptr: T.handle,
            silu_mul_output_ptr: T.handle,
            topk_reduce_output_ptr: T.handle,
            etensor_workspace_ptr: T.handle,
            exec_queue_ptr: T.handle,
            profiler_buffer: T.Buffer((self.PROFILER_BUFFER_SIZE,), "uint64"),
        ):
            T.func_attr({"global_symbol": "main", "target": T.target("cuda")})
            batch_size = T.meta_var(compile_batch_size)
            hidden_state_global = T.match_buffer(hidden_state_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            residual_global = T.match_buffer(residual_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            output_global = T.match_buffer(output_ptr, [batch_size, self.HIDDEN_SIZE], "float16")
            gate_weight_global = T.match_buffer(gate_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE], "float16", scope="global")
            grp_gate_up_weight_global = T.match_buffer(grp_gate_up_weight_ptr, [self.NUM_EXPERTS, self.INTERMEDIATE_SIZE * 2, self.HIDDEN_SIZE], "float16", scope="global")
            grp_down_weight_global = T.match_buffer(grp_down_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE], "float16", scope="global")
            gating_output_global = T.match_buffer(gating_output_ptr, [batch_size, self.NUM_EXPERTS], "float32", scope="global")
            topk_weights_global = T.match_buffer(topk_weights_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "float32", scope="global")
            topk_indices_global = T.match_buffer(topk_indices_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "int32", scope="global")
            max_num_tokens_padded = T.int32()
            sorted_token_ids_global = T.match_buffer(sorted_token_ids_ptr, [max_num_tokens_padded], "int32", scope="global")
            expert_ids_global = T.match_buffer(expert_ids_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_valid_tokens_global = T.match_buffer(num_valid_tokens_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_tokens_post_pad_global = T.match_buffer(num_tokens_post_pad_ptr, [1], "int32", scope="global")
            cumsum_buffer_global = T.match_buffer(cumsum_buffer_ptr, [self.NUM_EXPERTS + 1], "int32", scope="global")
            reordered_hidden_state_global = T.match_buffer(reordered_hidden_state_ptr, [max_num_tokens_padded, self.HIDDEN_SIZE], "float16", scope="global")
            gate_up_output_global = T.match_buffer(gate_up_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE * 2], "float16", scope="global")
            silu_mul_output_global = T.match_buffer(silu_mul_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE], "float16", scope="global")
            topk_reduce_output_global = T.match_buffer(topk_reduce_output_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            etensor_workspace_size = T.int32()
            etensor_workspace_global = T.match_buffer(etensor_workspace_ptr, [etensor_workspace_size], "int32", scope="global")
            exec_queue = T.match_buffer(exec_queue_ptr, [KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS], "int32", scope="global")

            @T.inline
            def run(dynamic_gemm_size):
                num_valid_tokens = T.meta_var(num_valid_tokens_global if dynamic_gemm_size else None)
                self.fused_body(
                    hidden_state_global, residual_global, output_global,
                    gate_weight_global, grp_gate_up_weight_global, grp_down_weight_global,
                    gating_output_global, topk_weights_global, topk_indices_global,
                    sorted_token_ids_global, expert_ids_global, num_valid_tokens,
                    num_tokens_post_pad_global, cumsum_buffer_global,
                    reordered_hidden_state_global, gate_up_output_global,
                    silu_mul_output_global, topk_reduce_output_global,
                    etensor_workspace_global, profiler_buffer,
                    exec_queue, None, None, None, False,
                    static_scheduler.Semaphore, StaticTileScheduler,
                )

            if compile_batch_size >= 512:
                run(dynamic_gemm_size=True)
            else:
                run(dynamic_gemm_size=False)
            # fmt: on
        return main

    def get_func_dynamic(self):
        compile_batch_size = self.batch_size

        # fmt: off
        @T.prim_func
        def main(
            hidden_state_ptr: T.handle,
            residual_ptr: T.handle,
            output_ptr: T.handle,
            gate_weight_ptr: T.handle,
            grp_gate_up_weight_ptr: T.handle,
            grp_down_weight_ptr: T.handle,
            gating_output_ptr: T.handle,
            topk_weights_ptr: T.handle,
            topk_indices_ptr: T.handle,
            sorted_token_ids_ptr: T.handle,
            expert_ids_ptr: T.handle,
            num_valid_tokens_ptr: T.handle,
            num_tokens_post_pad_ptr: T.handle,
            cumsum_buffer_ptr: T.handle,
            reordered_hidden_state_ptr: T.handle,
            gate_up_output_ptr: T.handle,
            silu_mul_output_ptr: T.handle,
            topk_reduce_output_ptr: T.handle,
            etensor_workspace_ptr: T.handle,
            queue_tasks_ptr: T.handle,
            queue_head_ptr: T.handle,
            queue_tail_ptr: T.handle,
            profiler_buffer: T.Buffer((self.PROFILER_BUFFER_SIZE,), "uint64"),
        ):
            T.func_attr({"global_symbol": "main", "target": T.target("cuda")})
            batch_size = T.meta_var(compile_batch_size)
            hidden_state_global = T.match_buffer(hidden_state_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            residual_global = T.match_buffer(residual_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            output_global = T.match_buffer(output_ptr, [batch_size, self.HIDDEN_SIZE], "float16")
            gate_weight_global = T.match_buffer(gate_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE], "float16", scope="global")
            grp_gate_up_weight_global = T.match_buffer(grp_gate_up_weight_ptr, [self.NUM_EXPERTS, self.INTERMEDIATE_SIZE * 2, self.HIDDEN_SIZE], "float16", scope="global")
            grp_down_weight_global = T.match_buffer(grp_down_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE], "float16", scope="global")
            gating_output_global = T.match_buffer(gating_output_ptr, [batch_size, self.NUM_EXPERTS], "float32", scope="global")
            topk_weights_global = T.match_buffer(topk_weights_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "float32", scope="global")
            topk_indices_global = T.match_buffer(topk_indices_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "int32", scope="global")
            max_num_tokens_padded = T.int32()
            sorted_token_ids_global = T.match_buffer(sorted_token_ids_ptr, [max_num_tokens_padded], "int32", scope="global")
            expert_ids_global = T.match_buffer(expert_ids_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_valid_tokens_global = T.match_buffer(num_valid_tokens_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_tokens_post_pad_global = T.match_buffer(num_tokens_post_pad_ptr, [1], "int32", scope="global")
            cumsum_buffer_global = T.match_buffer(cumsum_buffer_ptr, [self.NUM_EXPERTS + 1], "int32", scope="global")
            reordered_hidden_state_global = T.match_buffer(reordered_hidden_state_ptr, [max_num_tokens_padded, self.HIDDEN_SIZE], "float16", scope="global")
            gate_up_output_global = T.match_buffer(gate_up_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE * 2], "float16", scope="global")
            silu_mul_output_global = T.match_buffer(silu_mul_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE], "float16", scope="global")
            topk_reduce_output_global = T.match_buffer(topk_reduce_output_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            etensor_workspace_size = T.int32()
            etensor_workspace_global = T.match_buffer(etensor_workspace_ptr, [etensor_workspace_size], "int32", scope="global")
            queue_tasks_global = T.match_buffer(queue_tasks_ptr, [DynamicTileScheduler.MAX_TASKS], "int32", scope="global", offset_factor=1)
            queue_head_global = T.match_buffer(queue_head_ptr, [1], "int32", scope="global", offset_factor=1)
            queue_tail_global = T.match_buffer(queue_tail_ptr, [1], "int32", scope="global", offset_factor=1)

            @T.inline
            def run(dynamic_gemm_size):
                num_valid_tokens = T.meta_var(num_valid_tokens_global if dynamic_gemm_size else None)
                self.fused_body(
                    hidden_state_global, residual_global, output_global,
                    gate_weight_global, grp_gate_up_weight_global, grp_down_weight_global,
                    gating_output_global, topk_weights_global, topk_indices_global,
                    sorted_token_ids_global, expert_ids_global, num_valid_tokens,
                    num_tokens_post_pad_global, cumsum_buffer_global,
                    reordered_hidden_state_global, gate_up_output_global,
                    silu_mul_output_global, topk_reduce_output_global,
                    etensor_workspace_global, profiler_buffer,
                    None, queue_tasks_global, queue_head_global, queue_tail_global, True,
                    dynamic_scheduler.Semaphore, DynamicTileScheduler,
                )

            if compile_batch_size >= 512:
                run(dynamic_gemm_size=True)
            else:
                run(dynamic_gemm_size=False)
            # fmt: on
        return main


def lower_moe_to_tirx(spec: KernelSpec, scheduler: str, *, profiler_on: bool = False):
    """Lower a logical MoE specification through a complete DSL-owned TIRX backend."""

    lowerer = MoeLowerer(policy_for_scheduler(scheduler))
    lowerer.lower(spec)
    return lowerer.build_module(profiler_on=profiler_on)


__all__ = ["lower_moe_to_tirx"]
