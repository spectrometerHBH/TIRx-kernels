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

"""Shared timing-budget helpers for distributed GEMM benchmarks."""

from __future__ import annotations

DEFAULT_WARMUP_MS = 25
DEFAULT_REPEAT_MS = 100
CALIBRATION_ITERATIONS = 5


def resolve_budget(value: int | None, default: int, name: str) -> int:
    budget = default if value is None else value
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError(f"{name} must be a positive integer millisecond budget")
    return budget


def iteration_counts(estimate_us: float, warmup_ms: int, repeat_ms: int) -> tuple[int, int]:
    if estimate_us <= 0:
        return 1000, 1000
    return (
        max(1, int(warmup_ms * 1_000.0 / estimate_us)),
        max(1, int(repeat_ms * 1_000.0 / estimate_us)),
    )


__all__ = [
    "CALIBRATION_ITERATIONS",
    "DEFAULT_REPEAT_MS",
    "DEFAULT_WARMUP_MS",
    "iteration_counts",
    "resolve_budget",
]
