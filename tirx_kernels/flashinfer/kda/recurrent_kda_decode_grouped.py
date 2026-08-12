# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications Copyright (c) 2026 The TIRx Authors.
# Modifications are licensed under the Apache License, Version 2.0.
#
# This file is a TIRx port of FlashInfer's
# flashinfer/kda_kernels/recurrent_kda.py::_grouped_kda_kernel
# (flashinfer-ai/flashinfer @ f2e04400, v0.6.18).
# See LICENSE, NOTICE, and licenses/ for the applicable terms.

"""TIRx port of FlashInfer's grouped-CTA recurrent-KDA decode kernel.

Source CuTe DSL: ``flashinfer/kda_kernels/recurrent_kda.py``
(``_grouped_kda_kernel`` at line 482, host ``_grouped_kda_host`` at 729,
compile cache ``_get_grouped_compiled`` at 836, dispatch at 2090-2198).

This is the sibling of :mod:`recurrent_kda_decode_one_warp`.  FlashInfer's host
routes here whenever the one-warp predicate fails:

* ``NUM_TOKENS == 1`` with ``sequence_heads = N * HV < 128`` -- small-batch decode;
* **every** ``NUM_TOKENS > 1`` call -- SGLang ``target_verify`` (MTP/speculative),
  regardless of N and HV.

Unlike the one-warp kernel, this one is a genuine multi-warp CTA: it allocates
four shared-memory buffers and has exactly **one** ``__syncthreads()`` splitting
a token-preprocessing phase from a barrier-free sequential recurrence.  The
barrier exists because the two phases use *different* thread-to-element
mappings: phase A walks ``d = tid % D`` over ``EPT`` elements, phase B walks the
``part``-granule view of the state tile.

The host picks the tiling by a hard rule (``recurrent_kda.py:859-860``):
``KS = 4 if T == 1 else 2`` and ``VSPLIT = 4``.  That yields exactly two shapes:

===========  ====  ====  ====  ===  ===  ===
mode          T     KS    NT    G   EPT   SW
===========  ====  ====  ====  ===  ===  ===
decode        1     4     128   4    1    4
verify       >1     2     64    8    2    2
===========  ====  ====  ====  ===  ===  ===

Both gate modes SGLang can select are covered: ``GATE_MODE=2``
(``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``, Kimi K3, ``-5.0``) and
``GATE_MODE=1`` (softplus with the ``x > 20`` linear guard, Kimi Linear).

Out of scope, because unreachable from the production call path or handled by
the sibling: ``GATE_MODE=0`` (pre-computed gate), ``USE_SRC=1``
(``initial_state_source``), ``BETA_LOGIT=1``, ``USE_L2=0``, ``HEAD_DIM != 128``,
GQA (``HV != H``), and single-token decode with ``N * HV >= 128`` (one-warp).
"""

from __future__ import annotations

from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T

# Sibling import: the one-warp port is the canonical home for the shared PTX,
# bf16-conversion and shuffle helpers (mamba's convention, e.g.
# selective_state_update_stp_vertical.py:34 importing _simple).
from . import recurrent_kda_decode_one_warp as _one_warp

HEAD_DIM = _one_warp.HEAD_DIM  # 128
VSPLIT = 4  # recurrent_kda.py:860 -- fixed by the host
ONE_WARP_MIN_SEQUENCE_HEADS = _one_warp.ONE_WARP_MIN_SEQUENCE_HEADS  # 128
L2_EPS = _one_warp.L2_EPS  # 1e-6, hardcoded in the source at :663-664
DEFAULT_LOWER_BOUND = _one_warp.DEFAULT_LOWER_BOUND  # -5.0 (Kimi K3)

# recurrent_kda.py:479
LOG2_E = 1.4426950408889634
# recurrent_kda.py:620 -- softplus switches to the linear branch above this
SOFTPLUS_LINEAR_THRESHOLD = 20.0

GATE_MODE_PRECOMPUTED = 0
GATE_MODE_SOFTPLUS = 1
GATE_MODE_LOWER_BOUND = 2


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _row_lengths(num_seqs: int, num_tokens: int, empty_rows: int) -> list[int]:
    """Per-row token counts.

    Spec mode requires ``cu_seqlens`` to step by exactly ``T`` for *every* row,
    padded rows included (recurrent_kda.py:1658-1661); padding is signalled by
    ``ssm_state_indices == -1``, never by a short row.  That contract is load
    bearing rather than cosmetic: a row with ``seq_len < T`` leaves output
    positions ``token_base + seq_len .. token_base + T - 1`` written by nobody
    -- phase B's zero-fill is guarded by ``in_row`` (:714-717) and the orphan
    loop only covers ``pos >= cu[n_seq]`` (:719-725).  The source therefore has
    no defined result for a short spec row, so it cannot be a correctness case.

    The one in-contract way to make ``in_row`` false is a **zero-length row**
    in ``T = 1`` decode mode: every remaining output position is still owned by
    some row, so the result stays fully defined.
    """
    if empty_rows and num_tokens != 1:
        raise ValueError(
            "empty rows are only in contract for T=1 decode; spec mode requires "
            "cu_seqlens to step by T for every row (recurrent_kda.py:1658-1661)"
        )
    lens = [num_tokens] * num_seqs
    for i in range(min(empty_rows, num_seqs)):
        lens[i] = 0
    return lens


def _grouped_tiling(num_tokens: int) -> dict[str, int]:
    """Reproduce the host's tiling rule (recurrent_kda.py:859-860, :516-521)."""
    ks = 4 if num_tokens == 1 else 2
    nt = (HEAD_DIM * ks) // VSPLIT
    return {
        "KS": ks,
        "KC": HEAD_DIM // ks,
        "G": (HEAD_DIM // ks) // 8,
        "CPB": HEAD_DIM // VSPLIT,
        "NT": nt,
        "EPT": max(HEAD_DIM // nt, 1),
        "SW": min(HEAD_DIM, nt) // 32,
    }


def _case(label: str, **overrides: Any) -> dict[str, Any]:
    """One config row.  ``mode`` selects the decode or verify input family."""
    config: dict[str, Any] = {
        "label": label,
        "mode": "verify",  # "decode" (T=1) or "verify" (T>1)
        "num_seqs": 8,
        "num_tokens": 8,  # T; must be 1 for mode="decode"
        "num_heads": 16,
        "num_value_heads": 16,
        "scratch_steps": None,  # None -> equals num_tokens (S == T)
        "pool_size": 512,
        "lower_bound": DEFAULT_LOWER_BOUND,  # None -> softplus gate
        "padded_slots": 0,  # rows whose ssm index is -1
        "empty_rows": 0,  # verify rows with seq_len < T
        "orphan_tokens": 0,  # q_total beyond cu[n_seq]
        "seed": 20260812,
    }
    config.update(overrides)
    return config


def _dec(label: str, **kw: Any) -> dict[str, Any]:
    kw.setdefault("mode", "decode")
    kw.setdefault("num_tokens", 1)
    return _case(label, **kw)


# All bench rows use the Kimi K3 gate (lower_bound = -5.0) and sit inside the
# grouped dispatch domain.  Decode rows need N*HV < 128 or the host would pick
# the one-warp kernel; verify rows reach grouped for every N and HV.
BENCH_CONFIGS = [
    # -- decode, T=1 (KS=4, NT=128) -------------------------------------------
    _dec("dec_hv16_b1", num_seqs=1),  # sequence_heads 16
    _dec("dec_hv16_b4", num_seqs=4),  # 64
    _dec("dec_hv12_b4", num_seqs=4, num_heads=12, num_value_heads=12),  # 48
    _dec("dec_hv12_b8", num_seqs=8, num_heads=12, num_value_heads=12),  # 96
    # -- verify, T=8 (KS=2, NT=64) -- production DSPARK gamma=7, sglang parity --
    _case("ver_t8_hv16_b1", num_seqs=1),
    _case("ver_t8_hv16_b4", num_seqs=4),
    _case("ver_t8_hv16_b16", num_seqs=16),
    _case("ver_t8_hv16_b32", num_seqs=32),
    _case("ver_t8_hv16_b64", num_seqs=64),
    _case("ver_t8_hv16_b128", num_seqs=128),
    _case("ver_t8_hv12_b16", num_seqs=16, num_heads=12, num_value_heads=12),
    _case("ver_t8_hv12_b64", num_seqs=64, num_heads=12, num_value_heads=12),
    # -- verify, other draft-window sizes -------------------------------------
    _case("ver_t4_hv16_b32", num_seqs=32, num_tokens=4),  # test parity (32, 3)
    _case("ver_t2_hv16_b8", num_seqs=8, num_tokens=2),  # legal floor
]

CONFIGS = [dict(cfg) for cfg in BENCH_CONFIGS] + [
    # Softplus gate (Kimi Linear): GATE_MODE=1, incl. the x>20 linear guard.
    _dec("dec_hv16_b4_sp", num_seqs=4, lower_bound=None),
    _case("ver_t8_hv16_b4_sp", num_seqs=4, lower_bound=None),
    _case("ver_t2_hv16_b8_sp", num_seqs=8, num_tokens=2, lower_bound=None),
    # CUDA-graph padding: negative ssm_state_indices rows must produce zeroed
    # output and leave their state slots untouched.
    _dec("dec_hv16_b7_padded", num_seqs=7, padded_slots=3),  # 112 < 128
    _case("ver_t8_hv16_b4_padded", num_seqs=4, padded_slots=2),
    # Zero-length decode row: the one in-contract way to make in_row False.
    # (A short *spec* row is out of contract -- see _row_lengths.)
    _dec("dec_hv16_b6_empty", num_seqs=6, empty_rows=1),
    # Allocated scratch stride S > T (tests use T+2): ssm index stride is
    # scratch_steps, not T.
    _case("ver_t4_hv16_b8_s6", num_seqs=8, num_tokens=4, scratch_steps=6),
    # Orphan packed suffix: q_total > cu[n_seq] must be zero-filled by the
    # kernel's tail loop (:719-725).  Decode-only -- spec mode sizes out_buf as
    # N*NUM_TOKENS (:1810) while cu must step by T, so q_total == cu[n_seq]
    # there and the tail loop is unreachable (it would write out of bounds).
    _dec("dec_hv16_b4_orphan", num_seqs=4, orphan_tokens=3),
    # Boundary: the largest decode shape still inside the grouped domain
    # (sequence_heads 112 < 128).
    _dec("dec_hv16_b7", num_seqs=7),
]

KERNEL_META = {
    "name": "recurrent_kda_decode_grouped",
    "category": "flashinfer",
    "compute_capability": 10,
}


def _specialization(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Derive the constexpr set, mirroring the source host dispatch."""
    mode = str(kwargs.get("mode", "verify"))
    num_seqs = int(kwargs["num_seqs"])
    num_tokens = int(kwargs["num_tokens"])
    num_heads = int(kwargs["num_heads"])
    num_value_heads = int(kwargs["num_value_heads"])
    pool_size = int(kwargs["pool_size"])
    lower_bound = kwargs.get("lower_bound", DEFAULT_LOWER_BOUND)
    scratch_steps = kwargs.get("scratch_steps") or num_tokens
    orphan_tokens = int(kwargs.get("orphan_tokens", 0))

    if num_value_heads != num_heads:
        raise ValueError("GQA (HV != H) is outside this port's scope")
    if mode == "decode" and num_tokens != 1:
        raise ValueError("mode='decode' requires num_tokens == 1")
    if mode == "verify" and num_tokens < 2:
        raise ValueError("mode='verify' requires num_tokens >= 2")
    if scratch_steps < num_tokens:
        raise ValueError("scratch_steps must be >= num_tokens")

    # Grouped dispatch domain (recurrent_kda.py:1797-1798).  T>1 always lands
    # here; T==1 only when the one-warp predicate fails.
    sequence_heads = num_seqs * num_value_heads
    if num_tokens == 1 and sequence_heads >= ONE_WARP_MIN_SEQUENCE_HEADS:
        raise ValueError(
            f"sequence_heads={sequence_heads} dispatches to the one-warp kernel; "
            f"the grouped port requires < {ONE_WARP_MIN_SEQUENCE_HEADS} at T=1 "
            "(recurrent_kda.py:1798)"
        )

    tiling = _grouped_tiling(num_tokens)
    gate_mode = (
        GATE_MODE_SOFTPLUS if lower_bound is None else GATE_MODE_LOWER_BOUND
    )  # recurrent_kda.py:2140

    empty_rows = int(kwargs.get("empty_rows", 0))
    q_total = sum(_row_lengths(num_seqs, num_tokens, empty_rows)) + orphan_tokens
    # Shared memory: three T*D f32 planes plus the T*16 f32 reduce scratch,
    # each 16-byte aligned (recurrent_kda.py:526-530).
    plane = 4 * num_tokens * HEAD_DIM
    off_eg = 0
    off_kr = _align_up(off_eg + plane, 16)
    off_qr = _align_up(off_kr + plane, 16)
    off_red = _align_up(off_qr + plane, 16)
    shared_bytes = _align_up(off_red + 4 * num_tokens * 16, 16)

    # State pool: committed [pool, HV, D, D] for decode; scratch viewed as
    # [N*S, HV, D, D] for verify (recurrent_kda.py kda_flashinfer.py:307-309).
    state_slots = pool_size if mode == "decode" else num_seqs * scratch_steps
    slot_stride = num_value_heads * HEAD_DIM * HEAD_DIM

    return {
        "NUM_SEQS": num_seqs,
        "NUM_TOKENS": num_tokens,
        "NUM_HEADS": num_heads,
        "NUM_VALUE_HEADS": num_value_heads,
        "RATIO": num_value_heads // num_heads,
        "KS": tiling["KS"],
        "G": tiling["G"],
        "CPB": tiling["CPB"],
        "NT": tiling["NT"],
        "EPT": tiling["EPT"],
        "SW": tiling["SW"],
        "VSPLIT": VSPLIT,
        "GATE_MODE": gate_mode,
        "Q_TOTAL": q_total,
        "QK_ELEMENTS": q_total * num_heads * HEAD_DIM,
        "V_ELEMENTS": q_total * num_value_heads * HEAD_DIM,
        "BETA_ELEMENTS": q_total * num_value_heads,
        "STATE_ELEMENTS": state_slots * slot_stride,
        "STATE_SLOT_STRIDE": slot_stride,
        "A_LOG_ELEMENTS": num_heads,
        "DT_BIAS_ELEMENTS": num_heads * HEAD_DIM,
        "CU_ELEMENTS": num_seqs + 1,
        "SSM_IDX_ELEMENTS": num_seqs * num_tokens,
        "NAT_ELEMENTS": num_seqs,
        "OFF_EG": off_eg,
        "OFF_KR": off_kr,
        "OFF_QR": off_qr,
        "OFF_RED": off_red,
        "SHARED_BYTES": shared_bytes,
    }


@T.jit
def _recurrent_kda_decode_grouped(
    q_h: T.handle,
    k_h: T.handle,
    v_h: T.handle,
    g_h: T.handle,
    beta_h: T.handle,
    a_log_h: T.handle,
    dt_bias_h: T.handle,
    cu_h: T.handle,
    ssm_idx_h: T.handle,
    nat_h: T.handle,
    state_h: T.handle,
    out_h: T.handle,
    q_total: T.int32,
    g_stride_q: T.int32,
    state_slot_stride: T.int64,
    scale: T.float32,
    lower_bound: T.float32,
    *,
    NUM_SEQS: T.constexpr,
    NUM_TOKENS: T.constexpr,
    NUM_HEADS: T.constexpr,
    NUM_VALUE_HEADS: T.constexpr,
    RATIO: T.constexpr,
    KS: T.constexpr,
    G: T.constexpr,
    CPB: T.constexpr,
    NT: T.constexpr,
    EPT: T.constexpr,
    SW: T.constexpr,
    VSPLIT: T.constexpr,
    GATE_MODE: T.constexpr,
    Q_TOTAL: T.constexpr,
    QK_ELEMENTS: T.constexpr,
    V_ELEMENTS: T.constexpr,
    BETA_ELEMENTS: T.constexpr,
    STATE_ELEMENTS: T.constexpr,
    STATE_SLOT_STRIDE: T.constexpr,
    A_LOG_ELEMENTS: T.constexpr,
    DT_BIAS_ELEMENTS: T.constexpr,
    CU_ELEMENTS: T.constexpr,
    SSM_IDX_ELEMENTS: T.constexpr,
    NAT_ELEMENTS: T.constexpr,
    OFF_EG: T.constexpr,
    OFF_KR: T.constexpr,
    OFF_QR: T.constexpr,
    OFF_RED: T.constexpr,
    SHARED_BYTES: T.constexpr,
):
    q = T.match_buffer(q_h, (QK_ELEMENTS,), "bfloat16", scope="global")
    k = T.match_buffer(k_h, (QK_ELEMENTS,), "bfloat16", scope="global")
    v = T.match_buffer(v_h, (V_ELEMENTS,), "bfloat16", scope="global")
    g = T.match_buffer(g_h, (V_ELEMENTS,), "bfloat16", scope="global")
    beta = T.match_buffer(beta_h, (BETA_ELEMENTS,), "bfloat16", scope="global")
    a_log = T.match_buffer(a_log_h, (A_LOG_ELEMENTS,), "float32", scope="global")
    dt_bias = T.match_buffer(dt_bias_h, (DT_BIAS_ELEMENTS,), "float32", scope="global")
    cu = T.match_buffer(cu_h, (CU_ELEMENTS,), "int32", scope="global")
    ssm_idx = T.match_buffer(ssm_idx_h, (SSM_IDX_ELEMENTS,), "int32", scope="global")
    nat = T.match_buffer(nat_h, (NAT_ELEMENTS,), "int32", scope="global")
    state = T.match_buffer(state_h, (STATE_ELEMENTS,), "bfloat16", scope="global")
    out = T.match_buffer(out_h, (V_ELEMENTS,), "bfloat16", scope="global")
    T.device_entry()
    # TIRX_TRANSCRIBE_START recurrent_kda_decode_grouped
    # Scaffold stage: the kernel body is written in the correctness gate, from
    # the sketch-reviewer-approved sketch at
    # .agents/sketch/flashinfer/kda/recurrent_kda_decode_grouped.md
    T.evaluate(0)


def get_kernel(**kwargs: Any):
    """Return the specialized grouped-CTA recurrent-KDA decode PrimFunc."""
    return _recurrent_kda_decode_grouped.specialize(**_specialization(kwargs))


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Build one deterministic case with independent TIRx / reference state.

    Two families, matching the two SGLang call sites:

    * ``mode="decode"`` -- ``kda_flashinfer.py::decode`` with a small batch, so
      ``N * HV < 128`` and the host picks the grouped kernel.  The committed
      pool is updated in place.
    * ``mode="verify"`` -- ``kda_flashinfer.py::target_verify``: packed
      ``[1, N*T, ...]`` inputs, draft-stride ``cu_seqlens``, and the scratch
      ``intermediate_ssm`` pool as ``initial_state`` with step 0 python-seeded
      from the committed pool (``:299-304``).  ``ssm_state_indices`` is the
      ``[N, T]`` matrix ``base_rows * scratch_steps + arange(T)`` (``:293-296``),
      so its stride is the *allocated* step count, which may exceed ``T``.
    """
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for grouped recurrent-KDA decode")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(
            f"grouped recurrent-KDA decode targets compute capability 10.x, got {capability}"
        )

    spec = _specialization(kwargs)
    mode = str(kwargs.get("mode", "verify"))
    n_seq = spec["NUM_SEQS"]
    tokens = spec["NUM_TOKENS"]
    heads = spec["NUM_HEADS"]
    vheads = spec["NUM_VALUE_HEADS"]
    q_total = spec["Q_TOTAL"]
    slot_stride = spec["STATE_SLOT_STRIDE"]
    pool_size = int(kwargs["pool_size"])
    scratch_steps = kwargs.get("scratch_steps") or tokens
    lower_bound = kwargs.get("lower_bound", DEFAULT_LOWER_BOUND)
    padded_slots = int(kwargs.get("padded_slots", 0))
    empty_rows = int(kwargs.get("empty_rows", 0))
    orphan_tokens = int(kwargs.get("orphan_tokens", 0))

    gen = torch.Generator(device=device)
    gen.manual_seed(int(kwargs["seed"]))

    def randn(*shape: int, dtype=torch.bfloat16, gain: float = 1.0):
        raw = torch.randn(shape, device=device, dtype=torch.float32, generator=gen)
        return (gain * raw).to(dtype)

    # Magnitudes follow sglang's make_verify_inputs / make_decode_inputs
    # (bench_kda_flashinfer_mtp.py:50-121).
    q = randn(1, q_total, heads, HEAD_DIM, gain=0.5)
    k = randn(1, q_total, heads, HEAD_DIM, gain=0.5)
    v = randn(1, q_total, vheads, HEAD_DIM, gain=0.5)
    g = (
        0.5
        * torch.randn(
            (1, q_total, vheads, HEAD_DIM), device=device, dtype=torch.float32, generator=gen
        )
        - 1.0
    ).to(torch.bfloat16)
    # Softplus has a linear branch above x > 20 (recurrent_kda.py:620); push a
    # slice of the gate past it so that specialization is actually exercised.
    if lower_bound is None and q_total > 0:
        g_f32 = g.float()
        g_f32[:, :, :, :8] = 25.0
        g = g_f32.to(torch.bfloat16)
    beta_logit = randn(1, q_total, vheads, dtype=torch.float32, gain=0.5)
    beta = torch.sigmoid(beta_logit).to(torch.bfloat16)  # sglang pre-sigmoids
    a_log = randn(heads, dtype=torch.float32, gain=0.2)
    dt_bias = randn(heads * HEAD_DIM, dtype=torch.float32, gain=0.1)

    # cu_seqlens: decode = one token per row (a zero-length row is allowed and
    # is what exercises in_row); verify = draft stride T for every row, padded
    # rows included (recurrent_kda.py:1658-1661).
    if mode == "decode":
        offsets = [0]
        for row_len in _row_lengths(n_seq, tokens, empty_rows):
            offsets.append(offsets[-1] + row_len)
        cu = torch.tensor(offsets, device=device, dtype=torch.int32)
    else:
        lens = _row_lengths(n_seq, tokens, 0)
        offsets = [0]
        for row_len in lens:
            offsets.append(offsets[-1] + row_len)
        cu = torch.tensor(offsets, device=device, dtype=torch.int32)

    committed = randn(pool_size, vheads, HEAD_DIM, HEAD_DIM, gain=0.01)

    if mode == "decode":
        slots = torch.arange(n_seq, device=device, dtype=torch.int32)
        if padded_slots:
            slots[n_seq - padded_slots :] = -1
        ssm_idx = slots.reshape(n_seq, 1)
        tirx_state_raw = committed.reshape(-1).clone()
        reference_state_raw = tirx_state_raw.clone()
        state_slots = pool_size
    else:
        base_rows = torch.arange(n_seq, device=device, dtype=torch.int32)
        ssm_idx = (
            base_rows[:, None] * scratch_steps
            + torch.arange(tokens, device=device, dtype=torch.int32)[None, :]
        ).contiguous()
        if padded_slots:
            ssm_idx[n_seq - padded_slots :, :] = -1
        scratch = torch.zeros(
            n_seq * scratch_steps * slot_stride, device=device, dtype=torch.bfloat16
        )
        # Step-0 seeding, exactly as the sglang wrapper does before the launch.
        seed_src = committed[:n_seq].reshape(n_seq, slot_stride)
        scratch_view = scratch.reshape(n_seq, scratch_steps, slot_stride)
        scratch_view[:, 0, :] = seed_src
        tirx_state_raw = scratch
        reference_state_raw = scratch.clone()
        state_slots = n_seq * scratch_steps

    initial_state_raw = tirx_state_raw.clone()
    nat = torch.ones(n_seq, device=device, dtype=torch.int32)  # sglang's fallback
    tirx_out = torch.empty((1, q_total, vheads, HEAD_DIM), device=device, dtype=torch.bfloat16)

    return {
        "spec": spec,
        "config": dict(kwargs),
        "mode": mode,
        "device": device,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "cu_seqlens": cu,
        "ssm_state_indices": ssm_idx,
        "nat": nat,
        "committed": committed,
        "tirx_state_raw": tirx_state_raw,
        "reference_state_raw": reference_state_raw,
        "initial_state_raw": initial_state_raw,
        "state_slots": state_slots,
        "scratch_steps": scratch_steps,
        "tirx_out": tirx_out,
        "scale": HEAD_DIM**-0.5,
        "eps": L2_EPS,
        "lower_bound": lower_bound,
        "q_total": q_total,
        "orphan_tokens": orphan_tokens,
    }


def _state_view(raw: torch.Tensor, case: dict[str, Any]) -> torch.Tensor:
    spec = case["spec"]
    return raw.as_strided(
        (case["state_slots"], spec["NUM_VALUE_HEADS"], HEAD_DIM, HEAD_DIM),
        (spec["STATE_SLOT_STRIDE"], HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    spec = case["spec"]
    lower_bound = case["lower_bound"]
    return (
        case["q"].reshape(-1),
        case["k"].reshape(-1),
        case["v"].reshape(-1),
        case["g"].reshape(-1),
        case["beta"].reshape(-1),
        case["a_log"],
        case["dt_bias"],
        case["cu_seqlens"],
        case["ssm_state_indices"].reshape(-1),
        case["nat"],
        case["tirx_state_raw"],
        case["tirx_out"].reshape(-1),
        int(case["q_total"]),
        int(spec["NUM_VALUE_HEADS"] * HEAD_DIM),  # g token stride (contiguous here)
        int(spec["STATE_SLOT_STRIDE"]),
        float(case["scale"]),
        float(lower_bound if lower_bound is not None else 0.0),
    )


def _flashinfer_reference(case: dict[str, Any]) -> torch.Tensor:
    """Run the FlashInfer CuTe DSL source on the reference state pool.

    ``run_recurrent_kda`` derives ``NUM_TOKENS = 1 + num_spec_tokens``
    (recurrent_kda.py:1779), so the verify family must pass it; the decode
    family must not (:1298-1299).  ``num_accepted_tokens`` is deliberately left
    unset, matching SGLang's ``target_verify`` -- FlashInfer then substitutes a
    cached ones vector, which makes ``ic = 0`` and ``slot0 = ssm_idx[n, 0]``.
    """
    import importlib

    # flashinfer.kda_kernels.__init__ rebinds ``recurrent_kda`` to the run
    # function, so the submodule must be imported explicitly.
    fi = importlib.import_module("flashinfer.kda_kernels.recurrent_kda")
    spec = case["spec"]
    tokens = spec["NUM_TOKENS"]
    out, _ = fi.run_recurrent_kda(
        q=case["q"],
        k=case["k"],
        v=case["v"],
        g=case["g"],
        beta=case["beta"],
        A_log=case["a_log"],
        dt_bias=case["dt_bias"],
        scale=case["scale"],
        initial_state=_state_view(case["reference_state_raw"], case),
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=case["lower_bound"],
        cu_seqlens=case["cu_seqlens"],
        ssm_state_indices=case["ssm_state_indices"],
        num_spec_tokens=(tokens - 1) if tokens > 1 else None,
    )
    return out


def run_test(**kwargs: Any) -> None:
    raise SkipTest(
        "recurrent_kda_decode_grouped is in the scaffold stage; the kernel body "
        "and its correctness checks land in the correctness gate"
    )


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    raise SkipTest(
        "recurrent_kda_decode_grouped is in the scaffold stage; benchmarking "
        "lands in the performance gate"
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
