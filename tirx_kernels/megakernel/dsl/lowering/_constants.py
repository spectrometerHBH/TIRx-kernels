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

"""Private constants shared by MoE normalization and TIRX emission."""

from tirx_kernels.megakernel.utils.config import KernelConfig
from tirx_kernels.megakernel.utils.utils import MAX_K_IDX, MAX_M_IDX, MAX_N_IDX

_EVENT_ATTRS = {
    "gating_done": "evt_gating",
    "topk_done": "evt_topk_softmax",
    "align_done": "evt_moe_align",
    "count_sort_done": "evt_count_and_sort",
    "gate_up_done": "evt_group_gemm_gate_up",
    "down_dispatch_done": "evt_group_gemm_down",
}
_STEP_PRE_NOTIFY = "pre_notify"
_STEP_WAIT = "wait"
_STEP_RUN = "run"
_STEP_CTA_SYNC = "cta_sync"
_STEP_RUNTIME_EVENT_INIT = "runtime_event_init"
_STEP_POST_NOTIFY = "post_notify"
_EXECUTION_STEPS = {
    _STEP_PRE_NOTIFY,
    _STEP_WAIT,
    _STEP_RUN,
    _STEP_CTA_SYNC,
    _STEP_RUNTIME_EVENT_INIT,
    _STEP_POST_NOTIFY,
}
_PACKED_INDEX_LIMITS = (MAX_M_IDX, MAX_N_IDX, MAX_K_IDX)
_SCOPE_WIDTHS = {
    "thread": 1,
    "warp": 32,
    "warpgroup": KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER,
    "cta": KernelConfig.NUM_THREADS,
}
_SCOPE_INSTANCES = {
    "thread": KernelConfig.NUM_THREADS,
    "warp": KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER,
    "warpgroup": KernelConfig.WG_NUMBER,
    "cta": 1,
}
_SCOPE_ORDER = {"thread": 0, "warp": 1, "warpgroup": 2, "cta": 3}
