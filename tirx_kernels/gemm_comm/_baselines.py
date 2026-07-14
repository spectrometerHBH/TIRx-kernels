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

"""Subprocess boundary for optional cuBLASMp and cuBLAS+NCCL baselines."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from ._bench import DEFAULT_REPEAT_MS, DEFAULT_WARMUP_MS, resolve_budget

_JSON_PREFIX = "TIRX_GEMM_COMM_BASELINE="


def _decode_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_JSON_PREFIX):
            value = json.loads(line.removeprefix(_JSON_PREFIX))
            if not isinstance(value, dict):
                raise RuntimeError("baseline worker returned a non-object JSON result")
            return value
    raise RuntimeError("baseline worker did not emit a result record")


def run_external_baselines(
    workload: str,
    *,
    world_size: int,
    warmup: int | None,
    repeat: int | None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    cublasmp_algo: str = "split_p2p",
    timeout: float = 900.0,
    strict: bool = False,
) -> dict[str, Any]:
    """Run baselines in a fresh process so their NCCL cannot collide with Disco."""

    warmup_ms = resolve_budget(warmup, DEFAULT_WARMUP_MS, "warmup")
    repeat_ms = resolve_budget(repeat, DEFAULT_REPEAT_MS, "repeat")
    if timeout <= 0:
        raise ValueError("baseline timeout must be positive")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if cooldown_s < 0:
        raise ValueError("cooldown_s must be non-negative")

    command = [
        sys.executable,
        "-m",
        "tirx_kernels.gemm_comm._baseline_worker",
        "--workload",
        workload,
        "--world-size",
        str(world_size),
        "--warmup",
        str(warmup_ms),
        "--repeat",
        str(repeat_ms),
        "--cublasmp-algo",
        cublasmp_algo,
        "--rounds",
        str(rounds),
        "--cooldown",
        str(cooldown_s),
        "--timeout",
        str(timeout),
    ]
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=timeout + 90.0
        )
        result = _decode_result(completed.stdout)
        if completed.returncode != 0 or result.get("status") != "OK":
            detail = result.get("error") or completed.stderr.strip() or "unknown baseline error"
            raise RuntimeError(detail)
        return result
    except BaseException as error:
        if strict:
            raise
        return {"status": "BASELINE_ERROR", "error": str(error)}


__all__ = ["run_external_baselines"]
