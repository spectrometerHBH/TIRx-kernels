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

"""Thin import facade for the split MoE DSL and lowering modules.

New code should import public names from tirx_kernels.megakernel.dsl.
The scheduler-independent graph lives in moe_spec; all physical planning
and emission code lives under lowering.
"""

from .lowering import (
    DynamicDispatchPlan,
    DynamicPolicy,
    DynamicProtocolPlan,
    EventPlan,
    MoeLowerer,
    MoeLoweringEnv,
    MoePolicy,
    NormalizedPlan,
    NotifyPlan,
    StaticPolicy,
    TilePlan,
    UnfusedPolicy,
    WaitPlan,
    make_moe_plan,
    policy_for_scheduler,
)
from .lowering.model import HostTask, RuntimeEventInitPlan
from .moe_spec import build_moe_graph

__all__ = [
    "DynamicDispatchPlan",
    "DynamicPolicy",
    "DynamicProtocolPlan",
    "EventPlan",
    "HostTask",
    "MoeLowerer",
    "MoeLoweringEnv",
    "MoePolicy",
    "NormalizedPlan",
    "NotifyPlan",
    "RuntimeEventInitPlan",
    "StaticPolicy",
    "TilePlan",
    "UnfusedPolicy",
    "WaitPlan",
    "build_moe_graph",
    "make_moe_plan",
    "policy_for_scheduler",
]
