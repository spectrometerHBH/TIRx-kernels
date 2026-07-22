from __future__ import annotations

import ctypes
import math
import random
from dataclasses import dataclass, fields
from enum import Enum
from functools import lru_cache, partial
from typing import Any
from unittest import SkipTest

import torch

from tirx_kernels.flashmla._gemm import tcgen05_config
from tirx_kernels.flashmla._tma import tma_config
from tvm.backend.cuda.operator.tile_primitive.tma_utils import SwizzleMode
from tvm.ir import PointerType, PrimType
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.lang.pipeline import MBarrier, PipelineState, TCGen05Bar, TMABar
from tvm.tirx.layout import ComposeLayout, S, TileLayout, laneid, wid_in_wg

B_H = 64
B_TOPK = 64
D_V = 512
NUM_BUFS = 2
NUM_INDEX_BUFS = 4
NUM_THREADS = 384
BF16_BYTES = 2
MAX_INIT_VAL = -1.0e30
LOG_2_E = math.log2(math.e)
LN_2 = math.log(2.0)

# config.h's unscoped NamedBarriers enum is passed to CUTLASS's uint32_t
# user-barrier overload.  barrier.h therefore adds FirstUserBarrier (8) before
# emitting PTX; use those physical IDs rather than the logical enum values.
CUTLASS_USER_BARRIER_BASE = 8
BAR_EVERYONE_SYNC = CUTLASS_USER_BARRIER_BASE + 4
BAR_WG0_SYNC = CUTLASS_USER_BARRIER_BASE + 1
BAR_WG0_WARP02 = CUTLASS_USER_BARRIER_BASE + 2

LAUNCH_TAGS = (
    "blockIdx.x",
    "blockIdx.y",
    "blockIdx.z",
    "threadIdx.x",
    "tirx.use_dyn_shared_memory",
)
COMBINE_LAUNCH_TAGS = (
    "blockIdx.x",
    "blockIdx.y",
    "blockIdx.z",
    "threadIdx.x",
    "tirx.use_programtic_dependent_launch",
)
NULLABLE_MAIN_BUFFER_PARAMS = (
    "topk_length_h",
    "attn_sink_h",
    "extra_kv_h",
    "extra_indices_h",
    "extra_topk_length_h",
)
NULLABLE_COMBINE_BUFFER_PARAMS = ("attn_sink_h",)
_TMA_PREFETCH_SEQUENCE = "flashmla_q_sw128,flashmla_o,flashmla_q_sw64"

_mma_config = partial(tcgen05_config, cta_group=1)


# CUDA kerutils/device/sm100/intrinsics.cuh:54-76.  TIRx's ordinary vector
# elementwise lowering requests rounding/FTZ spellings that are observably
# different from these two source instructions, so keep the exact packed PTX
# locally at the four source call classes (peer add, sum add, and O scaling).
_ADD_F32X2_SRC = r"""
__device__ __forceinline__ unsigned long long tirx_flashmla_add_f32x2(
    unsigned long long a, unsigned long long b) {
  unsigned long long out;
  asm volatile("add.f32x2 %0, %1, %2;" : "=l"(out) : "l"(a), "l"(b));
  return out;
}
"""

_MUL_F32X2_SRC = r"""
__device__ __forceinline__ unsigned long long tirx_flashmla_mul_f32x2(
    unsigned long long a, unsigned long long b) {
  unsigned long long out;
  asm volatile("mul.f32x2 %0, %1, %2;" : "=l"(out) : "l"(a), "l"(b));
  return out;
}
"""


def _add_f32x2(a, b):
    return T.cuda.func_call(
        "tirx_flashmla_add_f32x2", a, b, source_code=_ADD_F32X2_SRC, return_type="uint64"
    )


def _mul_f32x2(a, b):
    return T.cuda.func_call(
        "tirx_flashmla_mul_f32x2", a, b, source_code=_MUL_F32X2_SRC, return_type="uint64"
    )


# kernel.cuh:600/634 selects between two CUtensorMap addresses as ``const
# void*`` before ku::tma_gather4 converts the selected pointer to the PTX u64
# operand.  Retain that pointer-typed select locally: selecting two already
# integerized addresses makes nvcc generate a measurably worse instruction
# sequence on SM100.
_SELECT_TENSORMAP_SRC = r"""
__device__ __forceinline__ unsigned long long tirx_flashmla_select_tensormap(
    bool use_extra, unsigned long long extra_addr, unsigned long long normal_addr) {
  const void* extra_ptr = reinterpret_cast<const void*>(extra_addr);
  const void* normal_ptr = reinterpret_cast<const void*>(normal_addr);
  return reinterpret_cast<unsigned long long>(use_extra ? extra_ptr : normal_ptr);
}
"""


def _select_tensormap(use_extra, extra_addr, normal_addr):
    return T.cuda.func_call(
        "tirx_flashmla_select_tensormap",
        use_extra,
        extra_addr,
        normal_addr,
        source_code=_SELECT_TENSORMAP_SRC,
        return_type="uint64",
    )


# CUDA kerutils/device/sm100/intrinsics.cuh:7-23.  The source converts the
# shared destination and mbarrier pointers to uint32 before issuing the exact
# CTA-group::1 gather.  Keeping those operands explicit lets each producer
# retain one current-stage shared base instead of rebuilding a generic 64-bit
# pointer at every source-unrolled call site.
_TMA_GATHER4_SHARED_ADDR_SRC = r"""
__device__ __forceinline__ void tirx_flashmla_tma_gather4_shared_addr(
    unsigned int dst_addr, unsigned long long desc_addr, int col_idx,
    int row0, int row1, int row2, int row3, unsigned int mbar_addr,
    unsigned long long cache_hint) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."
      "mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
      :
      : "r"(dst_addr), "l"(desc_addr), "r"(col_idx),
        "r"(row0), "r"(row1), "r"(row2), "r"(row3),
        "r"(mbar_addr), "l"(cache_hint)
      : "memory");
}
"""


# CUDA kernel.cuh:704-710.  CUDA exposes the float2->e8m0x2 conversion only
# through a CUDA intrinsic; keep the exact no-saturation/round-zero operation
# in a minimal local helper instead of replacing it with exponent arithmetic.
_F32X4_TO_E8M0X4_SRC = r"""
#include <cuda_fp8.h>
__device__ __forceinline__ unsigned int tirx_flashmla_f32x4_to_e8m0x4(
    float x, float y, float z, float w) {
  unsigned int out;
  unsigned short lo = __nv_cvt_float2_to_e8m0x2(
      make_float2(x, y), __NV_NOSAT, cudaRoundZero);
  unsigned short hi = __nv_cvt_float2_to_e8m0x2(
      make_float2(z, w), __NV_NOSAT, cudaRoundZero);
  out = static_cast<unsigned int>(lo) |
        (static_cast<unsigned int>(hi) << 16);
  return out;
}
"""


def _f32x4_to_e8m0x4(x, y, z, w):
    return T.cuda.func_call(
        "tirx_flashmla_f32x4_to_e8m0x4",
        x,
        y,
        z,
        w,
        source_code=_F32X4_TO_E8M0X4_SRC,
        return_type="uint32",
    )


# CUDA kernel.cuh:785-789/811-817.  One source intrinsic converts one packed
# e8m0 pair into two bf16 register values; WG2 invokes it exactly two times for
# V32 and four times for MODEL1, once per row and outside the column loop.
_E8M0X2_TO_BF16X2_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_fp8.h>
__device__ __forceinline__ unsigned int tirx_flashmla_e8m0x2_to_bf16x2(
    unsigned short packed) {
  __nv_bfloat162_raw out = __nv_cvt_e8m0x2_to_bf162raw(packed);
  return static_cast<unsigned int>(out.x) |
         (static_cast<unsigned int>(out.y) << 16);
}
"""


def _e8m0x2_to_bf16x2(packed):
    return T.cuda.func_call(
        "tirx_flashmla_e8m0x2_to_bf16x2",
        packed,
        source_code=_E8M0X2_TO_BF16X2_SRC,
        return_type="uint32",
    )


# CUDA kernel.cuh:777-779.  The source retains one 32-bit shared base after
# selecting the raw KV stage, then ptxas encodes every unrolled byte offset in
# LDS.64.  Taking the shared address directly avoids rebuilding a generic
# 64-bit pointer and repeating cvta.to.shared for each prefetched fp8x8 value.
_LD_SHARED_U64_SRC = r"""
__device__ __forceinline__ unsigned long long tirx_flashmla_ld_shared_u64(
    unsigned int smem_addr) {
  unsigned long long value;
  asm volatile("ld.shared.u64 %0, [%1];"
               : "=l"(value)
               : "r"(smem_addr));
  return value;
}
"""


def _ld_shared_u64(smem_addr):
    return T.cuda.func_call(
        "tirx_flashmla_ld_shared_u64",
        smem_addr,
        source_code=_LD_SHARED_U64_SRC,
        return_type="uint64",
    )


# CUDA helpers.h:25-32 and kernel.cuh:769-775/791-832.  Scale conversion is
# deliberately not folded into this helper: the source keeps all converted
# scales live and reusable.  The selected shared base is already a uint32 and
# is converted once per block.  The weak store has no memory clobber, matching
# the source inline-asm contract exactly.
_DEQUANT_ST128_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_fp8.h>
__device__ __forceinline__ void tirx_flashmla_dequant_st128(
    unsigned int smem_addr, unsigned long long raw, unsigned short scale_bits) {
  __nv_bfloat16 scale;
  *reinterpret_cast<unsigned short*>(&scale) = scale_bits;
  unsigned int packed[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_fp8x2_e4m3 data;
    data.__x = static_cast<__nv_fp8x2_storage_t>((raw >> (i * 16)) & 0xffffull);
    float2 value = static_cast<float2>(data);
    __nv_bfloat162 rounded = __float22bfloat162_rn(value);
    __nv_bfloat162 scaled{rounded.x * scale, rounded.y * scale};
    packed[i] = *reinterpret_cast<unsigned int*>(&scaled);
  }
  unsigned __int128 packed128 = *reinterpret_cast<unsigned __int128*>(packed);
  asm volatile("st.weak.shared::cta.b128 [%0], %1;"
               :: "r"(smem_addr), "q"(packed128));
}
"""


def _dequant_st128(smem_addr, raw, scale_bits):
    T.evaluate(
        T.cuda.func_call(
            "tirx_flashmla_dequant_st128",
            smem_addr,
            raw,
            scale_bits,
            source_code=_DEQUANT_ST128_SRC,
        )
    )


class ModelType(str, Enum):
    """The two CUDA template instances of the single head64 implementation."""

    V32 = "V32"
    MODEL1 = "MODEL1"


@dataclass(frozen=True)
class SparseFlashMLADecodeHead64Config:
    label: str
    model_type: ModelType | str
    b: int
    s_q: int
    s_kv: int
    topk: int
    page_block_size: int
    h_q: int = B_H
    h_kv: int = 1
    d_v: int = D_V
    have_attn_sink: bool = False
    have_topk_length: bool = False
    inject_invalid_indices: bool = False
    is_varlen: bool = True
    is_all_indices_invalid: bool = False
    have_zero_seqlen_k: bool = False
    extra_s_kv: int = 0
    extra_topk: int = 0
    extra_page_block_size: int = 0
    have_extra_topk_length: bool = False
    seed: int = 0

    @property
    def normalized_model_type(self) -> ModelType:
        return ModelType(self.model_type)

    @property
    def d_qk(self) -> int:
        # d_qk is deliberately derived from MODEL_TYPE, matching config.h.
        return 576 if self.normalized_model_type is ModelType.V32 else 512

    def validate(self) -> None:
        if self.h_kv != 1 or self.d_v != D_V:
            raise ValueError("head64 sparse decode requires h_kv=1 and d_v=512")
        if self.h_q == 128 and self.normalized_model_type is not ModelType.V32:
            raise ValueError("h_q=128,d_qk=512 dispatches to the out-of-scope head128 kernel")
        if self.h_q not in (B_H, 2 * B_H):
            raise ValueError("this port covers h_q=64 direct and V32 h_q=128 head64x2 dispatch")
        if self.b <= 0 or self.s_q <= 0 or self.s_kv <= 0 or self.page_block_size <= 0:
            raise ValueError("b, s_q, s_kv, and page_block_size must be positive")
        if self.topk <= 0 or self.topk % B_TOPK != 0:
            raise ValueError("topk must be a positive multiple of 64")
        if self.extra_topk % B_TOPK != 0:
            raise ValueError("extra_topk must be a multiple of 64")
        if self.extra_topk and not self.extra_s_kv:
            raise ValueError("extra_s_kv is required when extra_topk is nonzero")
        if self.extra_topk and self.extra_page_block_size <= 0:
            raise ValueError("extra_page_block_size must be positive with an extra KV cache")
        if self.have_extra_topk_length and not self.extra_topk:
            raise ValueError("extra_topk_length requires an extra KV cache")


def _dispatch_record(model_type: ModelType, h_q: int) -> str:
    d_qk = 576 if model_type is ModelType.V32 else 512
    if h_q == B_H:
        head_dispatch = "h_q=64 -> direct head64 launch"
    else:
        head_dispatch = "h_q=128,d_qk=576 -> host launches the same head64 kernel twice"
    return (
        f"sm100f + sparse FP8 + {head_dispatch} + h_kv=1 + d_v=512 + topk>0 -> "
        "sm100::decode::head64::run_flash_splitkv_mla_fp8_sparse_kernel<"
        f"ModelType::{model_type.value}>; d_qk={d_qk} is derived from MODEL_TYPE"
    )


_UPSTREAM_REGULAR_SHAPES = (
    (512, 64, 2),
    (512, 64, 64),
    (512, 64, 69),
    (1024, 576, 2),
    (1024, 576, 61),
    (2046, 2048, 2),
    (2046, 2048, 64),
    (2046, 2048, 576),
)
_UPSTREAM_CORNER_SHAPES = ((512, 64, 61), (650, 576, 53))


def _raw_upstream_case(
    *,
    group: str,
    b: int,
    h_q: int,
    s_q: int,
    s_kv: int,
    is_varlen: bool,
    topk: int,
    d_qk: int,
    page_block_size: int = 64,
    is_all_indices_invalid: bool = False,
    have_zero_seqlen_k: bool = False,
    have_topk_length: bool = False,
    have_attn_sink: bool = True,
    extra_s_kv: int | None = None,
    extra_topk: int | None = None,
    extra_page_block_size: int | None = None,
    have_extra_topk_length: bool = False,
) -> dict[str, Any]:
    return {
        "upstream_group": group,
        "b": b,
        "h_q": h_q,
        "s_q": s_q,
        "s_kv": s_kv,
        "is_varlen": is_varlen,
        "topk": topk,
        "d_qk": d_qk,
        "page_block_size": page_block_size,
        "is_all_indices_invalid": is_all_indices_invalid,
        "have_zero_seqlen_k": have_zero_seqlen_k,
        "have_topk_length": have_topk_length,
        "have_attn_sink": have_attn_sink,
        "extra_s_kv": extra_s_kv,
        "extra_topk": extra_topk,
        "extra_page_block_size": extra_page_block_size,
        "have_extra_topk_length": have_extra_topk_length,
    }


def _upstream_label(source_index: int, raw: dict[str, Any]) -> str:
    model = "v32" if raw["d_qk"] == 576 else "model1"
    label = (
        f"up_{raw['upstream_group'][:4]}_{source_index:04d}_{model}"
        f"_hq{raw['h_q']}_b{raw['b']}_sq{raw['s_q']}"
        f"_sk{raw['s_kv']}_topk{raw['topk']}_page{raw['page_block_size']}"
    )
    if raw["extra_topk"] is not None:
        label += (
            f"_xsk{raw['extra_s_kv']}_xtopk{raw['extra_topk']}_xpage{raw['extra_page_block_size']}"
        )
    modes = (
        "var" if raw["is_varlen"] else "fixed",
        "topklen" if raw["have_topk_length"] else "fulltopk",
        "xtopklen" if raw["have_extra_topk_length"] else "fullxtopk",
        "allinvalid" if raw["is_all_indices_invalid"] else "mixedindices",
        "zeroseqlen" if raw["have_zero_seqlen_k"] else "nonzeroseqlen",
        "sink" if raw["have_attn_sink"] else "nosink",
    )
    return f"{label}_{'_'.join(modes)}"


def _materialize_upstream_case(source_index: int, raw: dict[str, Any]) -> dict[str, Any]:
    model_type = ModelType.V32 if raw["d_qk"] == 576 else ModelType.MODEL1
    return {
        "label": _upstream_label(source_index, raw),
        "model_type": model_type.value,
        "b": raw["b"],
        "s_q": raw["s_q"],
        "s_kv": raw["s_kv"],
        "topk": raw["topk"],
        "page_block_size": raw["page_block_size"],
        "h_q": raw["h_q"],
        "have_attn_sink": raw["have_attn_sink"],
        "have_topk_length": raw["have_topk_length"],
        "is_varlen": raw["is_varlen"],
        "is_all_indices_invalid": raw["is_all_indices_invalid"],
        "have_zero_seqlen_k": raw["have_zero_seqlen_k"],
        "extra_s_kv": raw["extra_s_kv"] or 0,
        "extra_topk": raw["extra_topk"] or 0,
        "extra_page_block_size": raw["extra_page_block_size"] or 0,
        "have_extra_topk_length": raw["have_extra_topk_length"],
        # FlashMLA's Counter assigns this same ordinal as the deterministic seed.
        "seed": source_index,
        "upstream_group": raw["upstream_group"],
        "upstream_case_index": source_index,
        "dispatch_reason": _dispatch_record(model_type, raw["h_q"]),
    }


def _generate_upstream_sparse_decode_matrix() -> tuple[list[dict[str, Any]], int]:
    """Transcribe tests/test_flash_mla_sparse_decoding.py::gen_testcase."""

    correctness: list[dict[str, Any]] = []
    corners: list[dict[str, Any]] = []
    for d_qk in (576, 512):
        extra_modes = (False, True) if d_qk == 512 else (False,)
        for have_extra_kv in extra_modes:
            extra_length_modes = (False, True) if have_extra_kv else (False,)
            for have_extra_topk_length in extra_length_modes:
                topk_length_modes = (False, True) if d_qk == 512 else (False,)
                for have_topk_length in topk_length_modes:
                    for h_q in (64, 128):
                        for s_kv, topk, page_block_size in _UPSTREAM_REGULAR_SHAPES:
                            extra_shapes = (
                                _UPSTREAM_REGULAR_SHAPES if have_extra_kv else ((None, None, None),)
                            )
                            for extra_s_kv, extra_topk, extra_page_block_size in extra_shapes:
                                for b in (4, 74, 321):
                                    for s_q in (1, 3):
                                        varlen_modes = (
                                            (True, False)
                                            if b == 74
                                            and not have_topk_length
                                            and not have_extra_topk_length
                                            else (True,)
                                        )
                                        for is_varlen in varlen_modes:
                                            correctness.append(
                                                _raw_upstream_case(
                                                    group="correctness",
                                                    b=b,
                                                    h_q=h_q,
                                                    s_q=s_q,
                                                    s_kv=s_kv,
                                                    is_varlen=is_varlen,
                                                    topk=topk,
                                                    d_qk=d_qk,
                                                    page_block_size=page_block_size,
                                                    have_topk_length=have_topk_length,
                                                    extra_s_kv=extra_s_kv,
                                                    extra_topk=extra_topk,
                                                    extra_page_block_size=extra_page_block_size,
                                                    have_extra_topk_length=(have_extra_topk_length),
                                                )
                                            )

                        for s_kv, topk, page_block_size in _UPSTREAM_CORNER_SHAPES:
                            extra_shapes = (
                                _UPSTREAM_CORNER_SHAPES if have_extra_kv else ((None, None, None),)
                            )
                            for extra_s_kv, extra_topk, extra_page_block_size in extra_shapes:
                                for b in (4, 74, 321):
                                    varlen_modes = (
                                        (True, False)
                                        if b == 74
                                        and not have_topk_length
                                        and not have_extra_topk_length
                                        else (True,)
                                    )
                                    for is_varlen in varlen_modes:
                                        for is_all_indices_invalid in (True, False):
                                            for have_zero_seqlen_k in (True, False):
                                                for have_attn_sink in (True, False):
                                                    if not (
                                                        is_all_indices_invalid
                                                        or have_zero_seqlen_k
                                                        or have_attn_sink
                                                    ):
                                                        continue
                                                    corners.append(
                                                        _raw_upstream_case(
                                                            group="corner",
                                                            b=b,
                                                            h_q=h_q,
                                                            s_q=3,
                                                            s_kv=s_kv,
                                                            is_varlen=is_varlen,
                                                            topk=topk,
                                                            d_qk=d_qk,
                                                            page_block_size=page_block_size,
                                                            is_all_indices_invalid=(
                                                                is_all_indices_invalid
                                                            ),
                                                            have_zero_seqlen_k=(have_zero_seqlen_k),
                                                            have_topk_length=have_topk_length,
                                                            have_attn_sink=have_attn_sink,
                                                            extra_s_kv=extra_s_kv,
                                                            extra_topk=extra_topk,
                                                            extra_page_block_size=(
                                                                extra_page_block_size
                                                            ),
                                                            have_extra_topk_length=(
                                                                have_extra_topk_length
                                                            ),
                                                        )
                                                    )

    performance: list[dict[str, Any]] = []
    production = (
        (
            _raw_upstream_case(
                group="performance",
                b=0,
                h_q=128,
                s_q=2,
                s_kv=32768,
                is_varlen=True,
                topk=2048,
                d_qk=576,
            ),
            (2, 64, 74, 128),
        ),
        (
            _raw_upstream_case(
                group="performance",
                b=0,
                h_q=64,
                s_q=2,
                s_kv=16384,
                is_varlen=True,
                topk=128,
                d_qk=512,
                page_block_size=256,
                extra_s_kv=16384,
                extra_topk=512,
                extra_page_block_size=64,
            ),
            (2, 64, 74, 128, 148, 256),
        ),
        (
            _raw_upstream_case(
                group="performance",
                b=0,
                h_q=128,
                s_q=2,
                s_kv=16384,
                is_varlen=True,
                topk=128,
                d_qk=512,
                page_block_size=256,
                extra_s_kv=16384,
                extra_topk=1024,
                extra_page_block_size=64,
            ),
            (2, 64, 74, 128, 148, 256),
        ),
        (
            _raw_upstream_case(
                group="performance",
                b=0,
                h_q=64,
                s_q=2,
                s_kv=16384,
                is_varlen=True,
                topk=128,
                d_qk=512,
                page_block_size=256,
                extra_s_kv=16384,
                extra_topk=1024,
                extra_page_block_size=2,
                have_extra_topk_length=True,
            ),
            (2, 64, 74, 128, 148, 256),
        ),
        (
            _raw_upstream_case(
                group="performance",
                b=0,
                h_q=128,
                s_q=2,
                s_kv=16384,
                is_varlen=True,
                topk=128,
                d_qk=512,
                page_block_size=256,
                extra_s_kv=16384,
                extra_topk=1024,
                extra_page_block_size=2,
                have_extra_topk_length=True,
            ),
            (2, 64, 74, 128, 148, 256),
        ),
    )
    for base, batch_sizes in production:
        for b in batch_sizes:
            performance.append({**base, "b": b})
    for h_q in (64, 128):
        for d_qk in (512, 576):
            performance.append(
                _raw_upstream_case(
                    group="performance",
                    b=148,
                    h_q=h_q,
                    s_q=2,
                    s_kv=32768,
                    is_varlen=True,
                    topk=16384,
                    d_qk=d_qk,
                )
            )

    all_cases = correctness + corners + performance
    in_scope = [
        _materialize_upstream_case(source_index, raw)
        for source_index, raw in enumerate(all_cases)
        if raw["h_q"] == B_H
    ]
    return in_scope, len(all_cases)


UPSTREAM_TEST_CONFIGS, UPSTREAM_TOTAL_CASES = _generate_upstream_sparse_decode_matrix()
UPSTREAM_CONFIGS = [
    config for config in UPSTREAM_TEST_CONFIGS if config["upstream_group"] == "performance"
]

# The task requires this h_q=64 DeepSeek-V4 case to remain the primary benchmark,
# even though upstream's production V3.2 row uses the head64x2 h_q=128 host path.
PRIMARY_BENCH_CONFIG = {
    "label": "deepseek_v4_v32_hq64_b128_sq2_sk32768_topk2048",
    "model_type": ModelType.V32.value,
    "b": 128,
    "s_q": 2,
    "s_kv": 32768,
    "topk": 2048,
    "page_block_size": 64,
    "h_q": 64,
    "have_attn_sink": True,
    "is_varlen": True,
    "seed": UPSTREAM_TOTAL_CASES,
    "upstream_group": "required_primary",
    "upstream_case_index": None,
    "dispatch_reason": _dispatch_record(ModelType.V32, 64),
}

# CONFIGS and BENCH_CONFIGS contain the 14 upstream performance cases whose
# public h_q is 64, plus the required h_q=64 DeepSeek-V4 primary.  Upstream's
# 2,358 h_q=64 correctness/corner cases have num_runs=0 and are not benchmark
# shapes.  Public h_q=128 is outside the clarified scope.
CONFIGS = [PRIMARY_BENCH_CONFIG, *UPSTREAM_CONFIGS]
BENCH_CONFIGS = CONFIGS

assert UPSTREAM_TOTAL_CASES == 4748
assert len(UPSTREAM_TEST_CONFIGS) == 2372
assert len(UPSTREAM_CONFIGS) == 14
assert sum(config["model_type"] == ModelType.MODEL1.value for config in UPSTREAM_CONFIGS) == 13
assert sum(config["model_type"] == ModelType.V32.value for config in UPSTREAM_CONFIGS) == 1
assert len(BENCH_CONFIGS) == 15

KERNEL_META = {
    "name": "sparse_flashmla_decode_head64",
    "category": "flashmla",
    "compute_capability": 10,
}


def _cfg(**kwargs: Any) -> SparseFlashMLADecodeHead64Config:
    cfg_fields = {field.name for field in fields(SparseFlashMLADecodeHead64Config)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    cfg_kwargs.setdefault("label", "custom")
    cfg = SparseFlashMLADecodeHead64Config(**cfg_kwargs)
    cfg.validate()
    return cfg


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _device_sm_count(device: torch.device | str) -> int:
    device_obj = torch.device(device)
    device_index = device_obj.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def _kv_storage_spec(
    model_type: ModelType, num_blocks: int, page_block_size: int
) -> tuple[int, int, int, int]:
    """Return API bytes/token, TMA stride, block stride, and TMA rows."""

    if model_type is ModelType.V32:
        bytes_per_token = 512 + 4 * 4 + 64 * BF16_BYTES
        tma_k_stride = 656
        # tests/quant.py intentionally allocates one padding row per block.
        stride_kv_block = (page_block_size + 1) * tma_k_stride
    else:
        bytes_per_token = 448 + 64 * BF16_BYTES + 7 + 1
        tma_k_stride = 576
        stride_kv_block = _ceil_div(page_block_size * bytes_per_token, tma_k_stride) * tma_k_stride
    num_tma_rows = num_blocks * (stride_kv_block // tma_k_stride)
    return bytes_per_token, tma_k_stride, stride_kv_block, num_tma_rows


class _AlignedTensorMap:
    """Own one source-compatible, 64-byte-aligned CUtensorMap value."""

    def __init__(self) -> None:
        # CUtensorMap is 128 bytes.  ctypes zero-initializes the backing store,
        # which also gives the exact all-zero optional descriptor used when the
        # source's extra_topk is zero.
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_sparse_kv_tensormap(
    *,
    storage: torch.Tensor,
    base_offset_bytes: int,
    tensor_dtype: str,
    global_inner_dim: int,
    global_outer_dim: int,
    global_outer_stride_bytes: int,
    box_inner_dim: int,
    swizzle: int,
) -> _AlignedTensorMap:
    """Encode kernel.cuh:909-937's two-dimensional KV TensorMaps."""

    import tvm

    desc = _AlignedTensorMap()
    encode_tensormap = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode_tensormap(
        desc.ptr,
        tensor_dtype,
        2,
        ctypes.c_void_p(int(storage.data_ptr()) + base_offset_bytes),
        global_inner_dim,
        global_outer_dim,
        global_outer_stride_bytes,
        box_inner_dim,
        1,
        1,
        1,
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        swizzle,
        2,  # CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )
    return desc


def _build_sparse_kv_tensormaps(case: dict[str, Any]) -> dict[str, _AlignedTensorMap]:
    """Build the four explicit CUtensorMap members in the CUDA TmaParams."""

    cfg: SparseFlashMLADecodeHead64Config = case["config"]
    shape: dict[str, int] = case["shape"]
    is_v32 = cfg.normalized_model_type is ModelType.V32
    d_nope = 512 if is_v32 else 448
    tma_k_stride = 656 if is_v32 else 576
    rope_offset_bytes = d_nope + (16 if is_v32 else 0)
    rope_tile = 32 if is_v32 else 64
    rope_swizzle = 2 if is_v32 else 3

    def encode_pair(
        storage: torch.Tensor, num_tma_rows: int
    ) -> tuple[_AlignedTensorMap, _AlignedTensorMap]:
        nope = _encode_sparse_kv_tensormap(
            storage=storage,
            base_offset_bytes=0,
            tensor_dtype="int64",
            global_inner_dim=d_nope // 8,
            global_outer_dim=num_tma_rows,
            global_outer_stride_bytes=tma_k_stride,
            box_inner_dim=d_nope // 8,
            swizzle=0,
        )
        rope = _encode_sparse_kv_tensormap(
            storage=storage,
            base_offset_bytes=rope_offset_bytes,
            tensor_dtype="bfloat16",
            global_inner_dim=64,
            global_outer_dim=num_tma_rows,
            global_outer_stride_bytes=tma_k_stride,
            box_inner_dim=rope_tile,
            swizzle=rope_swizzle,
        )
        return nope, rope

    kv_nope, kv_rope = encode_pair(case["kv_storage"], shape["num_tma_rows"])
    if cfg.extra_topk:
        extra_nope, extra_rope = encode_pair(case["extra_kv_storage"], shape["extra_num_tma_rows"])
    else:
        extra_nope = _AlignedTensorMap()
        extra_rope = _AlignedTensorMap()
    return {
        "kv_nope": kv_nope,
        "kv_rope": kv_rope,
        "extra_kv_nope": extra_nope,
        "extra_kv_rope": extra_rope,
    }


def _max_splits_bucket(num_sm_parts: int) -> int:
    # combine.cu:176-187 MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, ...).
    for bucket in (32, 64, 96, 128, 160):
        if num_sm_parts <= bucket:
            return bucket
    raise ValueError(f"FlashMLA combine supports at most 160 SM partitions, got {num_sm_parts}")


def _build_decode_scheduler(
    cfg: SparseFlashMLADecodeHead64Config,
    topk_length_cpu: torch.Tensor,
    extra_topk_length_cpu: torch.Tensor,
    num_sm_parts: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host replica of get_mla_metadata_kernel for sparse decode.

    The CUDA implementation uses one warp only to build these small metadata
    arrays.  Keeping this out of the timed kernel matches FlashMLA's reuse of
    FlashMLASchedMeta across decode invocations.
    """

    block_size_n = B_TOPK
    fixed_overhead_num_blocks = 5
    seqlens_k: list[int] = []
    num_blocks: list[int] = []
    first_block_idx: list[int] = []
    last_block_idx: list[int] = []
    total_num_blocks = 0

    for batch_idx in range(cfg.b):
        cur_s_k = int(topk_length_cpu[batch_idx]) if cfg.have_topk_length else cfg.topk
        if cur_s_k == 0:
            cur_s_k = 1
        if cfg.extra_topk:
            cur_s_k = _ceil_div(cur_s_k, block_size_n) * block_size_n
            cur_s_k += (
                int(extra_topk_length_cpu[batch_idx])
                if cfg.have_extra_topk_length
                else cfg.extra_topk
            )
        seqlens_k.append(cur_s_k)
        first = 0
        last = max(cur_s_k - 1, 0) // block_size_n
        blocks = last - first + 1
        first_block_idx.append(first)
        last_block_idx.append(last)
        num_blocks.append(blocks)
        total_num_blocks += blocks + fixed_overhead_num_blocks

    payload = _ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks
    metadata = torch.zeros((num_sm_parts, 8), dtype=torch.int32)
    num_splits = torch.zeros((cfg.b + 1,), dtype=torch.int32)
    now_req_idx = 0
    now_block = 0
    now_n_split_idx = 0
    cum_num_splits = 0

    for partition_idx in range(num_sm_parts):
        # CUDA leaves partitions after all requests with begin_req_idx >= b;
        # only that field is consumed because the main kernel returns at once.
        if now_req_idx >= cfg.b:
            metadata[partition_idx, 0] = cfg.b
            continue

        begin_req_idx = now_req_idx
        begin_block_idx = now_block + first_block_idx[now_req_idx]
        begin_split_idx = now_n_split_idx
        is_first_req_splitted = int(now_block != 0)
        remain_payload = payload
        while now_req_idx < cfg.b:
            now_remain_blocks = num_blocks[now_req_idx] - now_block
            if remain_payload >= now_remain_blocks + fixed_overhead_num_blocks:
                cum_num_splits += now_n_split_idx + 1
                num_splits[now_req_idx + 1] = cum_num_splits
                remain_payload -= now_remain_blocks + fixed_overhead_num_blocks
                now_req_idx += 1
                now_block = 0
                now_n_split_idx = 0
            else:
                if remain_payload - fixed_overhead_num_blocks > 0:
                    now_block += remain_payload - fixed_overhead_num_blocks
                    now_n_split_idx += 1
                    remain_payload = 0
                break

        end_req_idx = now_req_idx if now_block > 0 else now_req_idx - 1
        if now_block > 0:
            end_block_idx = now_block + first_block_idx[now_req_idx]
        else:
            prev_req_idx = now_req_idx - 1
            end_block_idx = 0 if seqlens_k[prev_req_idx] == 0 else last_block_idx[prev_req_idx] + 1
        is_last_req_splitted = int(
            end_block_idx != last_block_idx[end_req_idx] + 1 and seqlens_k[end_req_idx] != 0
        )
        if begin_req_idx == end_req_idx:
            split = int(bool(is_first_req_splitted or is_last_req_splitted))
            is_first_req_splitted = split
            is_last_req_splitted = split
        metadata[partition_idx] = torch.tensor(
            [
                begin_req_idx,
                end_req_idx,
                begin_block_idx,
                end_block_idx,
                begin_split_idx,
                is_first_req_splitted,
                is_last_req_splitted,
                0,
            ],
            dtype=torch.int32,
        )

    if not (now_req_idx == cfg.b and now_block == 0 and now_n_split_idx == 0):
        raise RuntimeError("host scheduler did not consume every sparse decode request")
    return metadata.to(device=device), num_splits.to(device=device)


@torch.inference_mode()
def _quantize_fp8_kv_cache(
    source: torch.Tensor, model_type: ModelType
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Pack the exact V32 or MODEL1 sparse-FP8 cache byte layout."""

    num_blocks, page_block_size, h_kv, d_qk = source.shape
    if h_kv != 1:
        raise ValueError("sparse decode FP8 cache requires h_kv=1")
    expected_d_qk = 576 if model_type is ModelType.V32 else 512
    if d_qk != expected_d_qk:
        raise ValueError(f"{model_type.value} requires d_qk={expected_d_qk}")

    bytes_per_token, _, stride_kv_block, num_tma_rows = _kv_storage_spec(
        model_type, num_blocks, page_block_size
    )
    storage = torch.empty((num_blocks * stride_kv_block,), dtype=torch.uint8, device=source.device)
    source_rows = source[:, :, 0, :]

    if model_type is ModelType.V32:
        d_nope, tile_size, num_tiles = 512, 128, 4
        physical_rows = storage.as_strided(
            (num_blocks, page_block_size, 656), (stride_kv_block, 656, 1)
        )
        scale_view = physical_rows[:, :, 512:528].view(torch.float32)
        physical_rows[:, :, 528:656].view(torch.bfloat16).copy_(source_rows[:, :, d_nope:])
        for tile_idx in range(num_tiles):
            values = source_rows[:, :, tile_idx * tile_size : (tile_idx + 1) * tile_size].float()
            scale = torch.pow(
                2.0, (values.abs().amax(dim=-1) / 448.0).clamp_min(1.0e-4).log2().ceil()
            )
            quantized = (values / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
            physical_rows[:, :, tile_idx * tile_size : (tile_idx + 1) * tile_size].copy_(
                quantized.view(torch.uint8)
            )
            scale_view[:, :, tile_idx].copy_(scale)
    else:
        d_nope, tile_size, num_tiles = 448, 64, 7
        # MODEL1 is not token-interleaved: all 576-byte NoPE/RoPE rows come
        # first, followed by a page tail of 8 scale bytes per token.
        physical_rows = storage.as_strided(
            (num_blocks, page_block_size, 576), (stride_kv_block, 576, 1)
        )
        scale_rows = storage.as_strided(
            (num_blocks, page_block_size, 8),
            (stride_kv_block, 8, 1),
            storage_offset=page_block_size * 576,
        )
        physical_rows[:, :, d_nope:576].view(torch.bfloat16).copy_(source_rows[:, :, d_nope:])
        for tile_idx in range(num_tiles):
            values = source_rows[:, :, tile_idx * tile_size : (tile_idx + 1) * tile_size].float()
            scale = torch.pow(
                2.0, (values.abs().amax(dim=-1) / 448.0).clamp_min(1.0e-4).log2().ceil()
            )
            quantized = (values / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
            physical_rows[:, :, tile_idx * tile_size : (tile_idx + 1) * tile_size].copy_(
                quantized.view(torch.uint8)
            )
            scale_rows[:, :, tile_idx].copy_(scale.to(torch.float8_e8m0fnu).view(torch.uint8))

    # The public API validates this logical shape/stride while the kernel uses
    # the same allocation as a flat byte address and applies MODEL_TYPE layout.
    public_view = storage.view(torch.float8_e4m3fn).as_strided(
        (num_blocks, page_block_size, 1, bytes_per_token),
        (stride_kv_block, bytes_per_token, bytes_per_token, 1),
    )
    return public_view, storage, stride_kv_block, num_tma_rows


def _noncontiguous_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Mirror tests/kernelkit/generate.py::non_contiguousify."""

    padded_shape = [
        dim + 128 if dim_idx == tensor.ndim - 1 else dim + 1
        for dim_idx, dim in enumerate(tensor.shape)
    ]
    storage = torch.empty(padded_shape, dtype=tensor.dtype, device=tensor.device)
    view = storage[tuple(slice(0, dim) for dim in tensor.shape)]
    view.copy_(tensor)
    return view


def _noncontiguous_randn(
    shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    """Mirror tests/kernelkit/generate.py::gen_non_contiguous_randn_tensor."""

    padded_shape = [
        dim + 128 if dim_idx == len(shape) - 1 else dim + 1 for dim_idx, dim in enumerate(shape)
    ]
    storage = torch.randn(padded_shape, dtype=dtype, device=device, generator=generator)
    return storage[tuple(slice(0, dim) for dim in shape)]


def _batched_randperm(
    permutation_ranges: torch.Tensor, permutation_size: int, generator: torch.Generator
) -> torch.Tensor:
    """Mirror FlashMLA's `_randperm_batch(..., paddings=[-1])`."""

    batch_size = permutation_ranges.numel()
    max_range = max(int(permutation_ranges.max().item()), permutation_size)
    random_values = torch.rand(
        (batch_size, max_range),
        dtype=torch.float32,
        device=permutation_ranges.device,
        generator=generator,
    )
    positions = torch.arange(max_range, device=permutation_ranges.device)
    random_values.masked_fill_(positions.view(1, -1) >= permutation_ranges.view(-1, 1), -math.inf)
    result = random_values.topk(permutation_size, dim=-1, sorted=True).indices.to(torch.int32)
    result.masked_fill_(result >= permutation_ranges.view(-1, 1), -1)
    return result


@torch.inference_mode()
def _prepare_upstream_kv_scope(
    cfg: SparseFlashMLADecodeHead64Config,
    *,
    s_kv: int,
    topk: int,
    page_block_size: int,
    have_topk_length: bool,
    device: torch.device,
    device_generator: torch.Generator,
    cpu_generator: torch.Generator,
    python_rng: random.Random,
) -> dict[str, Any]:
    """Reproduce generate_testcase_for_decode's physical paged-KV scope."""

    cache_seqlens_cpu = torch.full((cfg.b,), s_kv, dtype=torch.int32)
    if cfg.is_varlen:
        for batch_idx in range(cfg.b):
            cache_seqlens_cpu[batch_idx] = int(
                max(python_rng.normalvariate(s_kv, s_kv / 2), cfg.s_q)
            )
    if cfg.have_zero_seqlen_k:
        zero_mask = torch.randn((cfg.b,), dtype=torch.float32, generator=cpu_generator) > 0
        cache_seqlens_cpu[zero_mask] = 0

    max_seqlen_alignment = 4 * page_block_size
    max_seqlen_pad = (
        max(_ceil_div(int(cache_seqlens_cpu.max().item()), max_seqlen_alignment), 1)
        * max_seqlen_alignment
    )
    blocks_per_sequence = max_seqlen_pad // page_block_size
    num_blocks = cfg.b * blocks_per_sequence

    block_ids = torch.arange(num_blocks, dtype=torch.int32, device=device)
    block_table = block_ids.index_select(
        0, torch.randperm(num_blocks, device=device, generator=device_generator)
    ).view(cfg.b, blocks_per_sequence)

    source = _noncontiguous_randn(
        (num_blocks, page_block_size, cfg.h_kv, cfg.d_qk),
        dtype=torch.bfloat16,
        device=device,
        generator=device_generator,
    )
    source = source / 10
    source.clamp_(min=-1.0, max=1.0)

    if cfg.is_all_indices_invalid:
        absolute_indices = torch.full((cfg.b, cfg.s_q, topk), -1, dtype=torch.int32, device=device)
    else:
        ranges = cache_seqlens_cpu.to(device=device).repeat_interleave(cfg.s_q)
        absolute_indices = _batched_randperm(ranges, topk, device_generator).view(
            cfg.b, cfg.s_q, topk
        )

    safe_indices = absolute_indices.clamp_min(0)
    batch_block_offsets = (
        torch.arange(cfg.b, dtype=torch.int32, device=device) * blocks_per_sequence
    ).view(cfg.b, 1, 1)
    block_lookup = safe_indices // page_block_size + batch_block_offsets
    physical_blocks = block_table.view(-1).index_select(0, block_lookup.view(-1).long())
    indices = (
        physical_blocks.view(cfg.b, cfg.s_q, topk) * page_block_size
        + safe_indices % page_block_size
    )
    indices.masked_fill_(absolute_indices < 0, -1)

    if have_topk_length:
        topk_length = torch.randint(
            0, topk + 1, (cfg.b,), dtype=torch.int32, device=device, generator=device_generator
        )
        topk_length_cpu = topk_length.cpu()
    else:
        topk_length = torch.zeros((cfg.b,), dtype=torch.int32, device=device)
        topk_length_cpu = torch.zeros((cfg.b,), dtype=torch.int32)

    masked_indices = indices
    if have_topk_length:
        masked_indices = indices.clone()
        positions = torch.arange(topk, device=device).view(1, 1, topk)
        masked_indices.masked_fill_(positions >= topk_length.view(cfg.b, 1, 1), -1)
    nonused_tokens = torch.ones((num_blocks * page_block_size,), dtype=torch.bool, device=device)
    used_tokens = masked_indices.long()
    nonused_tokens[used_tokens] = False
    source.view(-1, cfg.d_qk)[nonused_tokens] = float("nan")

    kv, kv_storage, stride_kv_block, num_tma_rows = _quantize_fp8_kv_cache(
        source, cfg.normalized_model_type
    )
    del source
    indices = _noncontiguous_copy(indices)
    return {
        "kv": kv,
        "kv_storage": kv_storage,
        "indices": indices,
        "topk_length": topk_length,
        "topk_length_cpu": topk_length_cpu,
        "cache_seqlens_cpu": cache_seqlens_cpu,
        "num_blocks": num_blocks,
        "stride_kv_block": stride_kv_block,
        "num_tma_rows": num_tma_rows,
    }


@lru_cache(maxsize=2)
def _make_sparse_decode_head64_kernel(model_type: ModelType):
    """One CUDA-structure transcription, configured by the MODEL_TYPE enum."""

    is_v32 = model_type is ModelType.V32
    d_qk = 576 if is_v32 else 512
    d_nope = 512 if is_v32 else 448
    num_scales = 4 if is_v32 else 8
    tma_k_stride = 656 if is_v32 else 576
    q_tail_start = 256 if is_v32 else 224
    rope_tile = 32 if is_v32 else 64
    rows_per_group = B_TOPK // (128 // 8)
    cols_per_group = d_nope // (8 * 8)
    k_rope_swizzle = SwizzleMode.SWIZZLE_64B_ATOM if is_v32 else SwizzleMode.SWIZZLE_128B_ATOM

    def allocate_model_kv_union(pool, u_base):
        """Select config.h's MODEL_TYPE union layout at parser time."""

        if is_v32:
            v32_stage_elems = B_TOPK * (D_V + 64)
            k_full = pool.alloc(
                (NUM_BUFS, B_TOPK, D_V),
                "bfloat16",
                align=1024,
                layout=ComposeLayout(
                    3,
                    3,
                    3,
                    TileLayout(
                        S[(NUM_BUFS, B_TOPK, 8, 64) : (v32_stage_elems, 64, B_TOPK * 64, 1)]
                    ),
                ),
            )
            pool.move_base_to(u_base + B_TOPK * D_V * BF16_BYTES)
            k_rope = pool.alloc(
                (NUM_BUFS, B_TOPK, 64),
                "bfloat16",
                align=1024,
                layout=ComposeLayout(
                    3,
                    2,
                    3,
                    TileLayout(
                        S[(NUM_BUFS, B_TOPK, 2, 32) : (v32_stage_elems, 32, B_TOPK * 32, 1)]
                    ),
                ),
            )
            pool.move_base_to(u_base + NUM_BUFS * v32_stage_elems * BF16_BYTES)
        else:
            k_full = pool.alloc_tcgen05_mma_AB((NUM_BUFS, B_TOPK, D_V), "bfloat16")
            k_full_end = pool.offset
            # MODEL1's RoPE aliases k_full[..., 448:512].
            pool.move_base_to(u_base)
            k_rope = pool.alloc_tcgen05_mma_AB(
                (NUM_BUFS, B_TOPK, 64), "bfloat16", swizzle_mode=k_rope_swizzle
            )
            pool.move_base_to(k_full_end)
        return k_full, k_rope

    def allocate_q_sw64_tail(pool, q_sw128_end):
        """Allocate only V32's nonzero q-SW64 member at parser time."""

        if is_v32:
            pool.move_base_to(q_sw128_end)
            q_sw64 = pool.alloc_tcgen05_mma_AB(
                (B_H, 64), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_64B_ATOM
            )
            q_with_tail_end = pool.offset
            pool.move_base_to(q_with_tail_end)
            return q_sw64
        # config.h:163: bf16 q_sw64[B_H * 0] reserves no storage.
        pool.move_base_to(q_sw128_end)
        return None

    def rope_stage_base_ptr(k_full, k_rope, stage):
        """Return the unswizzled physical base of one RoPE ring stage."""

        if is_v32:
            return k_rope.ptr_to([stage, 0, 0])
        return k_full.ptr_to([stage, 0, d_nope])

    def emit_no_split_epilogue(
        o_epi_frag,
        o_epi_bf16_frag,
        o_epi,
        o_win,
        o_smem_win,
        o_smem,
        out_strided,
        output_scale_pair,
        warp_idx,
        batch_idx,
        s_q_idx,
    ):
        """Meta-expand CUDA's CUTE_UNROLL before TMA layout dispatch."""

        # TilePrimitiveDispatch runs before a TIR ``T.unroll`` pass.  Execute
        # this ordinary Python helper eagerly so every SW128 shared-memory
        # region has the same concrete 64-column origin that CuTe sees after
        # CUTE_UNROLL, while batch/query coordinates remain runtime values.
        for epi_i in range((D_V // 2) // 64):
            Tx.wg.copy_async(o_epi_frag[:, :], o_win.chunk((None, (D_V // 2) // 64))[:, epi_i])
            T.evaluate(T.ptx.tcgen05.wait.ld())
            for scale_i in range(64 // 2):
                scaled_pair = T.Bind(
                    _mul_f32x2(
                        T.cuda.make_float2(o_epi[scale_i * 2], o_epi[scale_i * 2 + 1]),
                        output_scale_pair,
                    )
                )
                T.buffer_store(o_epi, T.cuda.float2_x(scaled_pair), [scale_i * 2])
                T.buffer_store(o_epi, T.cuda.float2_y(scaled_pair), [scale_i * 2 + 1])
            Tx.wg.cast(o_epi_bf16_frag[:, :], o_epi_frag[:, :])
            col_base = (D_V // 2 if epi_i * 64 >= D_V // 4 else 0) + (epi_i * 64) % (D_V // 4)
            Tx.wg.copy(o_smem_win.chunk((None, (D_V // 2) // 64))[:, epi_i], o_epi_bf16_frag[:, :])
            T.evaluate(T.ptx.fence.proxy_async("shared::cta"))
            T.evaluate(T.ptx.bar.sync(BAR_WG0_SYNC, 128))
            with T.If(warp_idx == 0):
                with T.Then():
                    with T.If(T.ptx.elect_sync() != T.uint32(0)):
                        with T.Then():
                            Tx.copy_async(
                                out_strided[batch_idx, s_q_idx, :, col_base : col_base + 64],
                                o_smem[:, col_base : col_base + 64],
                                **tma_config(
                                    prefetch_tensormap=False,
                                    tensormap_l2_promotion="L2::128B",
                                    tensormap_label="flashmla_o",
                                    tensormap_assume_aligned_strides=True,
                                ),
                            )
            warp1_col_base = col_base + D_V // 4
            with T.If(warp_idx == 1):
                with T.Then():
                    with T.If(T.ptx.elect_sync() != T.uint32(0)):
                        with T.Then():
                            Tx.copy_async(
                                out_strided[
                                    batch_idx, s_q_idx, :, warp1_col_base : warp1_col_base + 64
                                ],
                                o_smem[:, warp1_col_base : warp1_col_base + 64],
                                **tma_config(
                                    prefetch_tensormap=False,
                                    tensormap_l2_promotion="L2::128B",
                                    tensormap_label="flashmla_o",
                                    tensormap_assume_aligned_strides=True,
                                ),
                            )

    def assign_int4_registers(dst, src):
        """Emit CUDA's local ``int4`` value assignment as four register lanes."""

        # Both operands are thread-local register fragments.  A memory-vector
        # dispatcher would incorrectly require an address space; eager lane
        # stores preserve the C++ aggregate assignment and let SSA/codegen
        # coalesce or eliminate the register moves, including the final dead
        # assignment retained at each source loop tail.
        for int4_lane in range(4):
            T.buffer_store(dst, src[int4_lane], [int4_lane])

    def issue_gather4(dst_addr, mbar_addr, tensor_map_addr, column, indices):
        """Emit ku::tma_gather4's single CTA-group source instruction."""

        T.evaluate(
            T.cuda.func_call(
                "tirx_flashmla_tma_gather4_shared_addr",
                dst_addr,
                tensor_map_addr,
                column,
                indices[0],
                indices[1],
                indices[2],
                indices[3],
                mbar_addr,
                T.uint64(0x14F0000000000000),
                source_code=_TMA_GATHER4_SHARED_ADDR_SRC,
            )
        )

    @T.jit
    def _kernel(
        q_h: T.handle,
        kv_h: T.handle,
        indices_h: T.handle,
        topk_length_h: T.handle,
        attn_sink_h: T.handle,
        lse_h: T.handle,
        out_h: T.handle,
        lse_accum_h: T.handle,
        o_accum_h: T.handle,
        tile_scheduler_metadata_h: T.handle,
        num_splits_h: T.handle,
        extra_kv_h: T.handle,
        extra_indices_h: T.handle,
        extra_topk_length_h: T.handle,
        tensor_map_kv_nope: T.TensorMap(),
        tensor_map_kv_rope: T.TensorMap(),
        tensor_map_extra_kv_nope: T.TensorMap(),
        tensor_map_extra_kv_rope: T.TensorMap(),
        sm_scale_div_log2: T.float32,
        stride_q_b: T.int32,
        stride_q_s_q: T.int32,
        stride_q_h_q: T.int32,
        stride_kv_block: T.int32,
        stride_kv_row: T.int32,
        stride_indices_b: T.int32,
        stride_indices_s_q: T.int32,
        stride_lse_b: T.int32,
        stride_lse_s_q: T.int32,
        stride_o_b: T.int32,
        stride_o_s_q: T.int32,
        stride_o_h_q: T.int32,
        stride_extra_kv_block: T.int32,
        stride_extra_kv_row: T.int32,
        stride_extra_indices_b: T.int32,
        stride_extra_indices_s_q: T.int32,
        stride_lse_accum_split: T.int32,
        stride_lse_accum_s_q: T.int32,
        stride_o_accum_split: T.int32,
        stride_o_accum_s_q: T.int32,
        stride_o_accum_h_q: T.int32,
        b: T.int32,
        s_q: T.int32,
        topk: T.int32,
        extra_topk: T.int32,
        num_blocks: T.int32,
        extra_num_blocks: T.int32,
        page_block_size: T.int32,
        extra_page_block_size: T.int32,
        num_sm_parts: T.int32,
    ):
        # params.h:71-99.  Optional tensors stay nullable runtime handles, and
        # every source stride remains a runtime kernel operand.  The views are
        # descriptor/addressing views over the caller's storage, not packed
        # substitutes.
        q = T.match_buffer(
            q_h,
            ((b - 1) * stride_q_b + (s_q - 1) * stride_q_s_q + (B_H - 1) * stride_q_h_q + d_qk,),
            "bfloat16",
            scope="global",
        )
        kv = T.match_buffer(kv_h, (num_blocks * stride_kv_block,), "uint8", scope="global")
        indices = T.match_buffer(
            indices_h,
            ((b - 1) * stride_indices_b + (s_q - 1) * stride_indices_s_q + topk,),
            "int32",
            scope="global",
        )
        topk_length = T.match_buffer(topk_length_h, (b,), "int32", scope="global")
        attn_sink = T.match_buffer(attn_sink_h, (B_H,), "float32", scope="global")
        lse = T.match_buffer(
            lse_h,
            ((b - 1) * stride_lse_b + (s_q - 1) * stride_lse_s_q + B_H,),
            "float32",
            scope="global",
        )
        out = T.match_buffer(
            out_h,
            ((b - 1) * stride_o_b + (s_q - 1) * stride_o_s_q + (B_H - 1) * stride_o_h_q + D_V,),
            "bfloat16",
            scope="global",
        )
        lse_accum = T.match_buffer(
            lse_accum_h,
            (
                (b + num_sm_parts - 1) * stride_lse_accum_split
                + (s_q - 1) * stride_lse_accum_s_q
                + B_H,
            ),
            "float32",
            scope="global",
        )
        o_accum = T.match_buffer(
            o_accum_h,
            (
                (b + num_sm_parts - 1) * stride_o_accum_split
                + (s_q - 1) * stride_o_accum_s_q
                + (B_H - 1) * stride_o_accum_h_q
                + D_V,
            ),
            "float32",
            scope="global",
        )
        tile_scheduler_metadata = T.match_buffer(
            tile_scheduler_metadata_h, (num_sm_parts, 8), "int32", scope="global"
        )
        num_splits = T.match_buffer(num_splits_h, (b + 1,), "int32", scope="global")
        extra_kv = T.match_buffer(
            extra_kv_h, (extra_num_blocks * stride_extra_kv_block,), "uint8", scope="global"
        )
        extra_indices = T.match_buffer(
            extra_indices_h,
            ((b - 1) * stride_extra_indices_b + (s_q - 1) * stride_extra_indices_s_q + extra_topk,),
            "int32",
            scope="global",
        )
        extra_topk_length = T.match_buffer(extra_topk_length_h, (b,), "int32", scope="global")

        T.device_entry()
        T.attr(
            {
                "tirx.launch_bounds_min_blocks_per_sm": 1,
                "tirx.launch_bounds_max_blocks_per_cluster": 1,
            }
        )
        # MODEL_TYPE is the sole implementation specialization.  Keep its
        # branch selector and exact SharedMemoryPlan size as parser/meta-time
        # Python values; binding either as an ordinary scalar would turn it
        # into a TIR expression and destroy the C++ if-constexpr structure.
        model_is_v32 = T.meta_var(is_v32)
        source_smem_size = T.meta_var(232192 if model_is_v32 else 218848)
        q_strided = q.view(
            b,
            s_q,
            B_H,
            d_qk,
            layout=TileLayout(S[(b, s_q, B_H, d_qk) : (stride_q_b, stride_q_s_q, stride_q_h_q, 1)]),
        )
        out_strided = out.view(
            b,
            s_q,
            B_H,
            D_V,
            layout=TileLayout(S[(b, s_q, B_H, D_V) : (stride_o_b, stride_o_s_q, stride_o_h_q, 1)]),
        )

        # kernel.cuh:25-33.  Grid is exactly (s_q, num_sm_parts, 1), with
        # three 128-thread warpgroups and the same canonical role indices.
        s_q_idx, partition_idx, _ = T.cta_id([s_q, num_sm_parts, 1])
        thread_idx = T.thread_id([NUM_THREADS])
        warpgroup_idx = T.warpgroup_id([3])
        warp_idx_in_wg = T.warp_id_in_wg([4])
        lane_idx = T.lane_id([32])
        idx_in_warpgroup = T.thread_id_in_wg([128])
        warp_idx: T.let = warpgroup_idx * 4 + warp_idx_in_wg

        # config.h:159-193.  Recreate SharedMemoryPlan's two unions.  V32 is
        # physically interleaved per stage as nope0/rope0/nope1/rope1; the
        # custom stage strides keep the source SW128/SW64 address maps while
        # exposing the same logical ring dimensions to MMA and TMA.
        pool = T.SMEMPool()
        u_base = T.meta_var(pool.offset)
        model_kv_union = T.meta_var(allocate_model_kv_union(pool, u_base))
        k_full = T.meta_var(model_kv_union[0])
        k_rope = T.meta_var(model_kv_union[1])
        raw_nope = pool.alloc((NUM_BUFS, B_TOPK, d_nope // 8), "uint64", align=1024)
        kv_union_end = T.meta_var(pool.offset)

        pool.move_base_to(u_base)
        q_sw128 = pool.alloc_tcgen05_mma_AB((B_H, 512), "bfloat16")
        q_sw128_end = T.meta_var(pool.offset)
        q_sw64 = T.meta_var(allocate_q_sw64_tail(pool, q_sw128_end))
        o_union_base = T.meta_var(pool.offset)
        o_smem = pool.alloc_tcgen05_mma_AB((B_H, D_V), "bfloat16")
        o_bf16_end = T.meta_var(pool.offset)
        pool.move_base_to(o_union_base)
        o_accum_storage = pool.alloc(((B_H - 1) * (D_V + 8) + D_V,), "float32", align=1024)
        o_accum_smem = o_accum_storage.view(
            B_H, D_V, layout=TileLayout(S[(B_H, D_V) : (D_V + 8, 1)])
        )
        qo_union_end = T.meta_var(pool.offset)
        pool.move_base_to(max(kv_union_end, qo_union_end, o_bf16_end))

        sp_union_base = T.meta_var(pool.offset)
        p_exchange = pool.alloc((4, 32 * (B_TOPK // 2)), "float32", align=16)
        sp_union_end = T.meta_var(pool.offset)
        pool.move_base_to(sp_union_base)
        s_smem_gemm = pool.alloc_tcgen05_mma_AB(
            (B_H, B_TOPK), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_NONE
        )
        pool.move_base_to(sp_union_end)

        rowwise_buf = pool.alloc((128,), "float32", align=16)
        is_token_valid = pool.alloc((NUM_INDEX_BUFS, B_TOPK // 8), "int8", align=16)
        tma_coord = pool.alloc((NUM_INDEX_BUFS, B_TOPK), "int32", align=16)
        scales_e8m0 = pool.alloc((NUM_INDEX_BUFS, B_TOPK * num_scales), "uint8", align=16)
        # array_aligned<uint32_t,1> occupies a complete 16-byte slot.
        tmem_start_addr = pool.alloc((4,), "uint32", align=16)

        # kernel.cuh:45-60.  Arrival counts encode the original producer and
        # consumer roles, including 258 arrivals on valid/coord/scale free.
        bar_last_store_done = MBarrier(pool, 1)
        bar_q_tma = TMABar(pool, 1)
        bar_q_utccp = TCGen05Bar(pool, 1)
        bar_rope_ready = TMABar(pool, NUM_BUFS)
        bar_nope_ready = MBarrier(pool, NUM_BUFS)
        bar_raw_ready = TMABar(pool, NUM_BUFS)
        bar_raw_free = MBarrier(pool, NUM_BUFS)
        bar_valid_ready = MBarrier(pool, NUM_INDEX_BUFS)
        bar_valid_free = MBarrier(pool, NUM_INDEX_BUFS)
        bar_qk_done = TCGen05Bar(pool, NUM_BUFS)
        bar_so_ready = MBarrier(pool, NUM_BUFS)
        bar_sv_done = TCGen05Bar(pool, NUM_BUFS)
        # sizeof(SharedMemoryPlan), including the final struct-alignment pad.
        pool.commit(size=source_smem_size)

        # config.h:71-80.  Preserve fixed TMEM columns O=0, Q=256, P=400.
        tmem_pool = T.TMEMPool(
            pool, total_cols=512, cta_group=1, tmem_addr=tmem_start_addr, sync_after_alloc=False
        )
        o_tmem = tmem_pool.alloc_tcgen05_mma_D(
            (B_H, D_V), "float32", M=64, cta_group=1, ws=True, group=(2, 2, 128)
        )
        o_win = o_tmem.rearrange("h (a b c) -> (b h) (a c)", a=2, b=2, c=128)
        tmem_pool.move_base_to(256)
        q_tmem = tmem_pool.alloc_tcgen05_mma_A(
            (2, B_H, d_qk // 2), "bfloat16", M=64, cta_group=1, ws=True
        )
        tmem_pool.move_base_to(400)
        p_tmem = tmem_pool.alloc_tcgen05_mma_D(
            (2, B_H, B_TOPK), "float32", M=64, cta_group=1, ws=True
        )

        @T.inline
        def load_scheduler_meta(dst):
            # kernel.cuh:80-88 / KU_LDG_256.  Keep one 32-byte operation,
            # including its cache operators and L2 prefetch size; the eighth
            # int32 word is intentionally loaded even though it is reserved.
            T.ptx.ld_global_nc(
                tile_scheduler_metadata.view("uint64").ptr_to([partition_idx, 0]),
                "uint64",
                "u64",
                dst=dst.ptr_to([0]),
                vec="v4",
                l1_evict="L1::no_allocate",
                l2_evict="L2::evict_normal",
                prefetch_size="L2::256B",
            )

        # kernel.cuh:35-67.  Q/O/Q-tail descriptors are ordered by the lowering
        # sequence attached to their copy sites.  The four KV maps mirror the
        # explicit CUtensorMap members of TmaParams; prefetch only the two normal
        # maps, exactly as the source's _TMA_PREFETCH_SEQUENCE does.
        if warp_idx == 0:
            if T.ptx.elect_sync() != T.uint32(0):
                T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_kv_nope)))
                T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_kv_rope)))
                T.ptx.mbarrier.init(bar_last_store_done.ptr_to([0]), 128)
                T.ptx.mbarrier.init(bar_q_tma.ptr_to([0]), 1)
                T.ptx.mbarrier.init(bar_q_utccp.ptr_to([0]), 1)
                for stage in T.unroll(NUM_BUFS):
                    T.ptx.mbarrier.init(bar_rope_ready.ptr_to([stage]), 1)
                    T.ptx.mbarrier.init(bar_nope_ready.ptr_to([stage]), 128)
                    T.ptx.mbarrier.init(bar_raw_ready.ptr_to([stage]), 1)
                    T.ptx.mbarrier.init(bar_raw_free.ptr_to([stage]), 128)
                    T.ptx.mbarrier.init(bar_qk_done.ptr_to([stage]), 1)
                    T.ptx.mbarrier.init(bar_so_ready.ptr_to([stage]), 128)
                    T.ptx.mbarrier.init(bar_sv_done.ptr_to([stage]), 1)
                for index_stage in T.unroll(NUM_INDEX_BUFS):
                    T.ptx.mbarrier.init(bar_valid_ready.ptr_to([index_stage]), 32)
                    T.ptx.mbarrier.init(bar_valid_free.ptr_to([index_stage]), 258)
                T.ptx.fence.mbarrier_init()
            T.ptx.tcgen05.alloc(T.address_of(tmem_start_addr[0]), n_cols=512, cta_group=1)
            T.cuda.trap_when_assert_failed(tmem_start_addr[0] == T.uint32(0))
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
        T.cuda.cta_sync()

        if warpgroup_idx == 0:
            # kernel.cuh:134-150.  Scale/exp warpgroup and its 224-register
            # allocation.  The output and S register/shared layouts match the
            # fixed dual-GEMM TMEM datapath used by the CUDA source.
            T.ptx.setmaxnreg(True, 224)
            rs_buf = PipelineState(NUM_BUFS, phase=0)
            rs_index = PipelineState(NUM_INDEX_BUFS, phase=0)
            p_tmem_win = p_tmem.rearrange("b h t -> (b h) t")
            s_frag_layout = TileLayout(
                S[(2, 32, 2, 32) : (1 @ wid_in_wg, 1 @ laneid, 2 @ wid_in_wg, 1)]
            )
            o_smem_win = o_smem.rearrange("h (a b c) -> (b h) (a c)", a=2, b=2, c=128)
            scale_pair: T.let = T.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)
            attn_sink_log2: T.float32 = T.float32(-float("inf"))
            if not T.isnullptr(attn_sink_h):
                attn_sink_log2 = (
                    T.cuda.ldg(attn_sink.ptr_to([idx_in_warpgroup % B_H]), "float32") * LOG_2_E
                )

            # kernel.cuh:77-118 expanded at the role call site to avoid the
            # register spilling explicitly called out by the CUDA source.
            sched_words = T.alloc_local((4,), "uint64")
            load_scheduler_meta(sched_words)
            sched_i32 = sched_words.view("int32")
            sched_begin_req: T.let = sched_i32[0]
            sched_end_req: T.let = sched_i32[1]
            sched_begin_block: T.let = sched_i32[2]
            sched_end_block: T.let = sched_i32[3]
            sched_begin_split: T.let = sched_i32[4]
            sched_first_split: T.let = sched_i32[5]
            sched_last_split: T.let = sched_i32[6]
            batch_bar_phase: T.int32 = 0

            # The CUDA return exits only the local run_main_loop lambda.  Guard
            # its body so inactive partitions still reach WG0's TMEM dealloc.
            if sched_begin_req < b:
                for batch_idx in T.serial(sched_begin_req, sched_end_req + 1, unroll=False):
                    topk_len: T.int32 = topk
                    if not T.isnullptr(topk_length_h):
                        topk_len = T.cuda.ldg(topk_length.ptr_to([batch_idx]), "int32")
                    orig_topk_padded: T.let = T.max(
                        ((topk_len + B_TOPK - 1) // B_TOPK) * B_TOPK, B_TOPK
                    )
                    extra_topk_len: T.int32 = extra_topk
                    if not T.isnullptr(extra_topk_length_h):
                        extra_topk_len = T.cuda.ldg(extra_topk_length.ptr_to([batch_idx]), "int32")
                    total_topk_padded: T.let = (
                        orig_topk_padded + ((extra_topk_len + B_TOPK - 1) // B_TOPK) * B_TOPK
                    )
                    start_block: T.let = T.if_then_else(
                        batch_idx == sched_begin_req, sched_begin_block, 0
                    )
                    end_block: T.let = T.if_then_else(
                        batch_idx == sched_end_req, sched_end_block, total_topk_padded // B_TOPK
                    )
                    is_split: T.bool = T.cast(
                        T.if_then_else(
                            batch_idx == sched_begin_req,
                            sched_first_split,
                            T.if_then_else(batch_idx == sched_end_req, sched_last_split, 0),
                        ),
                        "bool",
                    )
                    is_no_split: T.bool = not is_split
                    n_split_idx: T.let = T.if_then_else(
                        batch_idx == sched_begin_req,
                        T.cuda.ldg(num_splits.ptr_to([batch_idx]), "int32") + sched_begin_split,
                        T.cuda.ldg(num_splits.ptr_to([batch_idx]), "int32"),
                    )
                    num_orig_blocks: T.let = orig_topk_padded // B_TOPK
                    is_last_batch: T.bool = batch_idx == sched_end_req

                    # kernel.cuh:151-159.  Retire prior TMA stores before the
                    # aliased Q/O shared region is reused for this batch.
                    T.ptx.cp_async.bulk.wait_group(0, read=True)
                    bar_last_store_done.arrive(0)
                    mi: T.float32 = MAX_INIT_VAL
                    li: T.float32 = 0.0
                    real_mi: T.float32 = T.float32(-float("inf"))

                    # kernel.cuh:160-299.  P load, dual-warp exchange, mask,
                    # online softmax, S staging, and conditional O rescale.
                    for block_idx in T.serial(start_block, end_block, unroll=False):
                        T.ptx.bar.sync(BAR_WG0_SYNC, 128)
                        bar_valid_ready.wait(rs_index.stage, rs_index.phase)
                        bar_qk_done.wait(rs_buf.stage, rs_buf.phase)
                        T.ptx.tcgen05.fence.after_thread_sync()

                        p_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, B_TOPK // 2), "float32")
                        p_peer_frag = T.alloc_tcgen05_ldst_frag(
                            "32x32b", (128, B_TOPK // 2), "float32"
                        )
                        p = p_frag.local()
                        p_peer = p_peer_frag.local()
                        if warp_idx < 2:
                            Tx.wg.copy_async(p_frag[:, :], p_tmem_win.chunk((None, 2))[:, 0])
                            Tx.wg.copy_async(p_peer_frag[:, :], p_tmem_win.chunk((None, 2))[:, 1])
                        else:
                            Tx.wg.copy_async(p_peer_frag[:, :], p_tmem_win.chunk((None, 2))[:, 0])
                            Tx.wg.copy_async(p_frag[:, :], p_tmem_win.chunk((None, 2))[:, 1])
                        T.ptx.tcgen05.wait.ld()
                        T.ptx.tcgen05.fence.before_thread_sync()

                        for exchange_i in T.unroll((B_TOPK // 2) // 4):
                            exchange_offset: T.let = exchange_i * 32 * 4 + lane_idx * 4
                            Tx.copy(
                                p_exchange[warp_idx ^ 2, exchange_offset : exchange_offset + 4],
                                p_peer[exchange_i * 4 : exchange_i * 4 + 4],
                                dispatch="vec_128b",
                            )
                        T.ptx.bar.sync(BAR_WG0_WARP02 + T.bitwise_and(warp_idx, T.int32(1)), 64)
                        for exchange_i in T.unroll((B_TOPK // 2) // 4):
                            exchange_offset: T.let = exchange_i * 32 * 4 + lane_idx * 4
                            peer_tmp = T.alloc_local((4,), "float32")
                            Tx.copy(
                                peer_tmp[0:4],
                                p_exchange[warp_idx, exchange_offset : exchange_offset + 4],
                                dispatch="vec_128b",
                            )
                            pair0: T.let = _add_f32x2(
                                T.cuda.make_float2(p[exchange_i * 4], p[exchange_i * 4 + 1]),
                                T.cuda.make_float2(peer_tmp[0], peer_tmp[1]),
                            )
                            pair1: T.let = _add_f32x2(
                                T.cuda.make_float2(p[exchange_i * 4 + 2], p[exchange_i * 4 + 3]),
                                T.cuda.make_float2(peer_tmp[2], peer_tmp[3]),
                            )
                            p[exchange_i * 4] = T.cuda.float2_x(pair0)
                            p[exchange_i * 4 + 1] = T.cuda.float2_y(pair0)
                            p[exchange_i * 4 + 2] = T.cuda.float2_x(pair1)
                            p[exchange_i * 4 + 3] = T.cuda.float2_y(pair1)

                        valid_word: T.let = is_token_valid.view("uint32")[
                            rs_index.stage, T.if_then_else(idx_in_warpgroup >= 64, 1, 0)
                        ]
                        for p_i in T.unroll(B_TOPK // 2):
                            if T.bitwise_and(
                                T.shift_right(valid_word, T.cast(p_i, "uint32")), T.uint32(1)
                            ) == T.uint32(0):
                                p[p_i] = T.float32(-float("inf"))

                        cur_pi_max: T.float32 = T.float32(-float("inf"))
                        for p_i in T.unroll(B_TOPK // 2):
                            cur_pi_max = T.max(cur_pi_max, p[p_i])
                        cur_pi_max = cur_pi_max * sm_scale_div_log2
                        rowwise_buf[idx_in_warpgroup] = cur_pi_max
                        T.ptx.bar.sync(BAR_WG0_SYNC, 128)
                        bar_valid_free.arrive(rs_index.stage)
                        cur_pi_max = T.max(cur_pi_max, rowwise_buf[idx_in_warpgroup ^ 64])
                        real_mi = T.max(real_mi, cur_pi_max)
                        should_scale_o: T.let = (
                            T.ptx.any_sync(T.uint32(0xFFFFFFFF), cur_pi_max - mi > 6.0) != 0
                        )
                        new_max: T.float32
                        scale_for_old: T.float32
                        if not should_scale_o:
                            scale_for_old = 1.0
                            new_max = mi
                        else:
                            new_max = T.max(cur_pi_max, mi)
                            scale_for_old = T.ptx.exp2(mi - new_max)
                        mi = new_max

                        s_frag = T.alloc_buffer(
                            (B_H, B_TOPK), "bfloat16", scope="local", layout=s_frag_layout
                        )
                        s_pack = s_frag.local().view("uint32")
                        cur_sum_pair: T.uint64 = T.cuda.make_float2(0.0, 0.0)
                        neg_max_pair: T.let = T.cuda.make_float2(-new_max, -new_max)
                        for s_i in T.unroll((B_TOPK // 2) // 2):
                            p_pair: T.let = T.cuda.make_float2(p[s_i * 2], p[s_i * 2 + 1])
                            soft_pair: T.let = T.ptx.fma_f32x2(
                                p_pair, scale_pair, neg_max_pair, dps=False
                            )
                            sx: T.let = T.ptx.exp2(T.cuda.float2_x(soft_pair))
                            sy: T.let = T.ptx.exp2(T.cuda.float2_y(soft_pair))
                            cur_sum_pair = _add_f32x2(cur_sum_pair, T.cuda.make_float2(sx, sy))
                            s_pack[s_i] = T.cuda.float22bfloat162_rn(sx, sy)
                        cur_sum: T.let = T.cuda.float2_x(cur_sum_pair) + T.cuda.float2_y(
                            cur_sum_pair
                        )
                        li_next: T.float32
                        T.ptx.fma_f32(T.address_of(li_next), li, scale_for_old, cur_sum)
                        li = li_next

                        Tx.wg.copy(s_smem_gemm[:, :], s_frag[:, :])
                        if T.And(block_idx != start_block, should_scale_o):
                            scale_for_old_pair: T.let = T.cuda.make_float2(
                                scale_for_old, scale_for_old
                            )
                            T.ptx.tcgen05.fence.after_thread_sync()
                            o_rescale_frag = T.alloc_tcgen05_ldst_frag(
                                "32x32b", (128, 64), "float32"
                            )
                            o_rescale = o_rescale_frag.local()
                            for o_chunk in T.unroll((D_V // 2) // 64):
                                Tx.wg.copy_async(
                                    o_rescale_frag[:, :],
                                    o_win.chunk((None, (D_V // 2) // 64))[:, o_chunk],
                                )
                                T.ptx.tcgen05.wait.ld()
                                for scale_i in T.unroll(64 // 2):
                                    scaled_pair: T.let = _mul_f32x2(
                                        T.cuda.make_float2(
                                            o_rescale[scale_i * 2], o_rescale[scale_i * 2 + 1]
                                        ),
                                        scale_for_old_pair,
                                    )
                                    o_rescale[scale_i * 2] = T.cuda.float2_x(scaled_pair)
                                    o_rescale[scale_i * 2 + 1] = T.cuda.float2_y(scaled_pair)
                                Tx.wg.copy_async(
                                    o_win.chunk((None, (D_V // 2) // 64))[:, o_chunk],
                                    o_rescale_frag[:, :],
                                )
                                T.ptx.tcgen05.wait.st()
                            T.ptx.tcgen05.fence.before_thread_sync()

                        T.ptx.fence.proxy_async("shared::cta")
                        bar_so_ready.arrive(rs_buf.stage)
                        if block_idx != end_block - 1:
                            rs_buf.advance()
                            rs_index.advance()

                    # kernel.cuh:301-333.  Empty-row repair, li exchange, LSE
                    # store, final SV wait, ring advance, and launch dependency.
                    if real_mi == T.float32(-float("inf")):
                        li = 0.0
                        mi = T.float32(-float("inf"))
                    rowwise_buf[idx_in_warpgroup] = li
                    T.ptx.bar.sync(BAR_WG0_SYNC, 128)
                    li = li + rowwise_buf[idx_in_warpgroup ^ 64]
                    if idx_in_warpgroup < B_H:
                        if is_no_split:
                            cur_lse: T.float32
                            T.ptx.fma_f32(T.address_of(cur_lse), mi, T.float32(LN_2), T.log(li))
                            lse[
                                batch_idx * stride_lse_b
                                + s_q_idx * stride_lse_s_q
                                + idx_in_warpgroup
                            ] = T.if_then_else(
                                cur_lse == T.float32(-float("inf")),
                                T.float32(float("inf")),
                                cur_lse,
                            )
                        else:
                            lse_accum[
                                n_split_idx * stride_lse_accum_split
                                + s_q_idx * stride_lse_accum_s_q
                                + idx_in_warpgroup
                            ] = T.log2(li) + mi
                    bar_sv_done.wait(rs_buf.stage, rs_buf.phase)
                    rs_buf.advance()
                    rs_index.advance()
                    T.ptx.tcgen05.fence.after_thread_sync()
                    if is_last_batch:
                        T.ptx.griddepcontrol.launch_dependents()

                    # kernel.cuh:335-421.  Keep no-split TMA output and split
                    # fp32 bulk output as distinct epilogues; attn_sink is only
                    # applied here for no-split and is deferred to combine for
                    # split output exactly as in the CUDA source.
                    if is_no_split:
                        output_scale: T.let = T.if_then_else(
                            li == 0.0,
                            0.0,
                            T.cuda.fdividef(1.0, li + T.ptx.exp2(attn_sink_log2 - mi)),
                        )
                        output_scale_pair: T.let = T.cuda.make_float2(output_scale, output_scale)
                        o_epi_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, 64), "float32")
                        o_epi_bf16_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, 64), "bfloat16")
                        o_epi = o_epi_frag.local()
                        emit_no_split_epilogue(
                            o_epi_frag,
                            o_epi_bf16_frag,
                            o_epi,
                            o_win,
                            o_smem_win,
                            o_smem,
                            out_strided,
                            output_scale_pair,
                            warp_idx,
                            batch_idx,
                            s_q_idx,
                        )
                        T.ptx.cp_async.bulk.commit_group()
                    else:
                        output_scale: T.let = T.if_then_else(
                            li == 0.0, 0.0, T.cuda.fdividef(1.0, li)
                        )
                        output_scale_pair: T.let = T.cuda.make_float2(output_scale, output_scale)
                        split_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, 64), "float32")
                        split_local = split_frag.local()
                        for epi_i in T.unroll((D_V // 2) // 64):
                            Tx.wg.copy_async(
                                split_frag[:, :], o_win.chunk((None, (D_V // 2) // 64))[:, epi_i]
                            )
                            T.ptx.tcgen05.wait.ld()
                            for scale_i in T.unroll(64 // 2):
                                scaled_pair: T.let = _mul_f32x2(
                                    T.cuda.make_float2(
                                        split_local[scale_i * 2], split_local[scale_i * 2 + 1]
                                    ),
                                    output_scale_pair,
                                )
                                split_local[scale_i * 2] = T.cuda.float2_x(scaled_pair)
                                split_local[scale_i * 2 + 1] = T.cuda.float2_y(scaled_pair)
                            col_base: T.let = (
                                (idx_in_warpgroup // 64) * 128
                                + T.if_then_else(epi_i * 64 >= D_V // 4, D_V // 2, 0)
                                + (epi_i * 64) % (D_V // 4)
                            )
                            for j in T.unroll(64 // 4):
                                Tx.copy(
                                    o_accum_smem[
                                        idx_in_warpgroup % 64,
                                        col_base + j * 4 : col_base + j * 4 + 4,
                                    ],
                                    split_local[j * 4 : j * 4 + 4],
                                    dispatch="vec_128b",
                                )
                        T.ptx.fence.proxy_async("shared::cta")
                        T.ptx.bar.sync(BAR_WG0_SYNC, 128)
                        if T.ptx.elect_sync() != T.uint32(0):
                            for local_row in T.unroll(B_H // 4):
                                smem_row: T.let = local_row * 4 + warp_idx
                                T.ptx.cp_async_bulk_s2g(
                                    o_accum.ptr_to(
                                        [
                                            n_split_idx * stride_o_accum_split
                                            + s_q_idx * stride_o_accum_s_q
                                            + smem_row * stride_o_accum_h_q
                                        ]
                                    ),
                                    o_accum_smem.ptr_to([smem_row, 0]),
                                    T.uint32(D_V * 4),
                                )
                            T.ptx.cp_async.bulk.commit_group()

                    # kernel.cuh:116 uses the unaligned spelling because the
                    # elected WG1 producer lanes reach this named barrier via
                    # control flow distinct from the empty-role lanes.
                    T.ptx.barrier.sync(BAR_EVERYONE_SYNC, NUM_THREADS)
                    batch_bar_phase = batch_bar_phase ^ 1

            if warp_idx == 0:
                T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=512, cta_group=1)

        elif warpgroup_idx == 1:
            # kernel.cuh:427-430.  The producer/MMA warpgroup deliberately
            # gives registers back, then recomputes the synchronized canonical
            # warp id; retaining the earlier value is known to spill registers.
            T.ptx.setmaxnreg(False, 72)
            wg1_warp_idx: T.let = T.tvm_warp_shuffle(
                T.uint32(0xFFFFFFFF), T.cuda.thread_rank() // 32, 0, 32, 32
            )

            # kernel.cuh:431-746.  Mirror each CUDA run_main_loop(lambda)
            # call as a separate parser-time specialization.  This keeps the
            # scheduler and its registers inside the selected warp role.
            @T.inline
            def run_wg1_role(role: T.constexpr):
                rs_buf = PipelineState(NUM_BUFS, phase=0)
                rs_index = PipelineState(NUM_INDEX_BUFS, phase=0)
                q_sw128_tmem = q_tmem.sub[:, :, 0:256]
                q_sw128_tmem_cp = q_sw128_tmem.rearrange("b h (dc di) -> h dc b di", di=64)
                k_full_tiled = k_full.rearrange(
                    "b r (dc h ci) -> b (h r) (dc ci)", dc=4, h=2, ci=64
                )
                q_tail_tmem = q_tmem.sub[:, :, q_tail_start : q_tail_start + 32]
                q_tail_tmem_cp = q_tail_tmem.rearrange("b h k -> h (b k)")
                k_rope_tiled = k_rope.rearrange("b r (h ci) -> b (h r) ci", h=2, ci=32)

                # kernel.cuh:657-667.  These warp-7 invariants are deliberately
                # materialized before the scheduler traversal.  Pointer bases are
                # held as byte addresses so the per-token paths only add offsets.
                tma_coords_step_per_token: T.int32 = 0
                tma_coords_step_per_block: T.int32 = 0
                tma_coords_step_per_extra_block: T.int32 = 0
                k_scales_ptr_u64: T.uint64 = T.uint64(0)
                extra_k_scales_ptr_u64: T.uint64 = T.uint64(0)
                if role == 7:
                    tma_coords_step_per_token = (656 if model_is_v32 else 576) // tma_k_stride
                    tma_coords_step_per_block = stride_kv_block // tma_k_stride
                    tma_coords_step_per_extra_block = stride_extra_kv_block // tma_k_stride
                    k_scales_ptr_u64 = T.reinterpret(
                        "uint64",
                        kv.ptr_to(
                            [d_nope if model_is_v32 else page_block_size * (d_nope + 2 * 64)]
                        ),
                    )
                    extra_k_scales_ptr_u64 = T.reinterpret(
                        "uint64",
                        extra_kv.ptr_to(
                            [d_nope if model_is_v32 else extra_page_block_size * (d_nope + 2 * 64)]
                        ),
                    )
                # kernel.cuh:77-118, expanded once for all WG1 threads.  Non-elected
                # lanes still execute the empty role and participate in the 384-way
                # per-batch named barrier, matching the CUDA else branch at 744.
                sched_words = T.alloc_local((4,), "uint64")
                load_scheduler_meta(sched_words)
                sched_i32 = sched_words.view("int32")
                sched_begin_req: T.let = sched_i32[0]
                sched_end_req: T.let = sched_i32[1]
                sched_begin_block: T.let = sched_i32[2]
                sched_end_block: T.let = sched_i32[3]
                sched_begin_split: T.let = sched_i32[4]
                sched_first_split: T.let = sched_i32[5]
                sched_last_split: T.let = sched_i32[6]
                batch_bar_phase: T.int32 = 0

                # The CUDA return exits only this role's run_main_loop lambda.
                if sched_begin_req < b:
                    for batch_idx in T.serial(sched_begin_req, sched_end_req + 1, unroll=False):
                        topk_len: T.int32 = topk
                        if not T.isnullptr(topk_length_h):
                            topk_len = T.cuda.ldg(topk_length.ptr_to([batch_idx]), "int32")
                        orig_topk_padded: T.let = T.max(
                            ((topk_len + B_TOPK - 1) // B_TOPK) * B_TOPK, B_TOPK
                        )
                        extra_topk_len: T.int32 = extra_topk
                        if not T.isnullptr(extra_topk_length_h):
                            extra_topk_len = T.cuda.ldg(
                                extra_topk_length.ptr_to([batch_idx]), "int32"
                            )
                        total_topk_padded: T.let = (
                            orig_topk_padded + ((extra_topk_len + B_TOPK - 1) // B_TOPK) * B_TOPK
                        )
                        start_block: T.let = T.if_then_else(
                            batch_idx == sched_begin_req, sched_begin_block, 0
                        )
                        end_block: T.let = T.if_then_else(
                            batch_idx == sched_end_req, sched_end_block, total_topk_padded // B_TOPK
                        )
                        is_split: T.bool = T.cast(
                            T.if_then_else(
                                batch_idx == sched_begin_req,
                                sched_first_split,
                                T.if_then_else(batch_idx == sched_end_req, sched_last_split, 0),
                            ),
                            "bool",
                        )
                        is_no_split: T.bool = not is_split
                        num_orig_blocks: T.let = orig_topk_padded // B_TOPK
                        n_split_idx: T.let = T.if_then_else(
                            batch_idx == sched_begin_req,
                            T.cuda.ldg(num_splits.ptr_to([batch_idx]), "int32") + sched_begin_split,
                            T.cuda.ldg(num_splits.ptr_to([batch_idx]), "int32"),
                        )
                        is_last_batch: T.bool = batch_idx == sched_end_req

                        if role == 4:
                            # kernel.cuh:431-527.  Warp 4 issues both Q TMAs, then
                            # the same SW128/SW64 UTCCP deposits at TMEM Q=256.
                            T.cuda.trap_when_assert_failed(start_block < end_block)
                            # SmemLayoutQ_SW128 is tile_to_shape of a 64x64
                            # atom.  CuTe therefore partitions the 64x512 Q
                            # copy into eight source-order TMA boxes instead of
                            # issuing one monolithic 64 KiB transaction.
                            for q_tile in T.unroll(512 // 64):
                                Tx.copy_async(
                                    q_sw128[:, q_tile * 64 : (q_tile + 1) * 64],
                                    q_strided[
                                        batch_idx, s_q_idx, :, q_tile * 64 : (q_tile + 1) * 64
                                    ],
                                    **tma_config(
                                        mbar=bar_q_tma.ptr_to([0]),
                                        cta_group=1,
                                        cache_hint="evict_first",
                                        prefetch_tensormap=False,
                                        tensormap_l2_promotion="L2::128B",
                                        tensormap_label="flashmla_q_sw128",
                                        tensormap_assume_aligned_strides=True,
                                        zero_tensormap_labels=(
                                            None if model_is_v32 else "flashmla_q_sw64"
                                        ),
                                        prefetch_tensormap_sequence=(
                                            None if model_is_v32 else _TMA_PREFETCH_SEQUENCE
                                        ),
                                    ),
                                )
                            if model_is_v32:
                                Tx.copy_async(
                                    q_sw64[:, :],
                                    q_strided[batch_idx, s_q_idx, :, 512:576],
                                    **tma_config(
                                        mbar=bar_q_tma.ptr_to([0]),
                                        cta_group=1,
                                        cache_hint="evict_first",
                                        prefetch_tensormap=False,
                                        tensormap_l2_promotion="L2::128B",
                                        tensormap_label="flashmla_q_sw64",
                                        tensormap_assume_aligned_strides=True,
                                        prefetch_tensormap_sequence=_TMA_PREFETCH_SEQUENCE,
                                    ),
                                )
                            bar_q_tma.arrive(0, tx_count=B_H * d_qk * BF16_BYTES)
                            bar_q_tma.wait(0, batch_bar_phase)
                            T.ptx.tcgen05.fence.after_thread_sync()
                            Tx.copy_async(
                                q_sw128_tmem_cp[:, :, :, :],
                                q_sw128.view(B_H, 4, 2, 64)[:, :, :, :],
                                shape="128x256b",
                                cta_group=1,
                            )
                            if model_is_v32:
                                Tx.copy_async(q_tail_tmem_cp[:, :], q_sw64[:, :])
                            bar_q_utccp.arrive(0)
                            bar_q_utccp.wait(0, batch_bar_phase)
                            T.ptx.tcgen05.fence.after_thread_sync()

                            # kernel.cuh:529-584.  MODEL_TYPE only selects how the
                            # shared K latent is interpreted; both instances issue
                            # the same dual-head P and SxV pipelines.
                            for block_idx in T.serial(start_block, end_block, unroll=False):
                                if model_is_v32:
                                    bar_rope_ready.wait(rs_buf.stage, rs_buf.phase)
                                    T.ptx.tcgen05.fence.after_thread_sync()
                                    Tx.gemm_async(
                                        p_tmem[:, :, :],
                                        q_tail_tmem[:, :, :],
                                        k_rope_tiled[rs_buf.stage, :, :],
                                        **_mma_config(accum=T.uint32(0)),
                                    )
                                    bar_nope_ready.wait(rs_buf.stage, rs_buf.phase)
                                    T.ptx.tcgen05.fence.after_thread_sync()
                                    Tx.gemm_async(
                                        p_tmem[:, :, :],
                                        q_sw128_tmem[:, :, :],
                                        k_full_tiled[rs_buf.stage, :, :],
                                        **_mma_config(accum=T.uint32(1)),
                                    )
                                else:
                                    bar_rope_ready.wait(rs_buf.stage, rs_buf.phase)
                                    bar_nope_ready.wait(rs_buf.stage, rs_buf.phase)
                                    T.ptx.tcgen05.fence.after_thread_sync()
                                    Tx.gemm_async(
                                        p_tmem[:, :, :],
                                        q_sw128_tmem[:, :, :],
                                        k_full_tiled[rs_buf.stage, :, :],
                                        **_mma_config(accum=T.uint32(0)),
                                    )
                                bar_qk_done.arrive(rs_buf.stage)

                                bar_so_ready.wait(rs_buf.stage, rs_buf.phase)
                                T.ptx.tcgen05.fence.after_thread_sync()
                                mma_o_accum: T.let = T.if_then_else(
                                    block_idx == start_block, T.uint32(0), T.uint32(1)
                                )
                                Tx.gemm_async(
                                    o_tmem.sub[:, 0 : D_V // 2],
                                    s_smem_gemm[:, :],
                                    k_full[rs_buf.stage, :, 0 : D_V // 2],
                                    transB=True,
                                    **_mma_config(accum=mma_o_accum),
                                )
                                Tx.gemm_async(
                                    o_tmem.sub[:, D_V // 2 : D_V],
                                    s_smem_gemm[:, :],
                                    k_full[rs_buf.stage, :, D_V // 2 : D_V],
                                    transB=True,
                                    **_mma_config(accum=mma_o_accum),
                                )
                                bar_sv_done.arrive(rs_buf.stage)
                                rs_buf.advance()
                                rs_index.advance()

                        elif role == 5:
                            # kernel.cuh:586-615.  One gather4 producer loads raw
                            # fp8 NoPE as int64, retaining the two-stage raw ring.
                            bar_q_utccp.wait(0, batch_bar_phase)
                            bar_last_store_done.wait(0, batch_bar_phase)
                            for block_idx in T.serial(start_block, end_block, unroll=False):
                                bar_valid_ready.wait(rs_index.stage, rs_index.phase)
                                bar_raw_free.wait(rs_buf.stage, rs_buf.phase ^ 1)
                                cur_raw_stage_addr: T.uint32 = T.cuda.cvta_generic_to_shared(
                                    raw_nope.ptr_to([rs_buf.stage, 0, 0])
                                )
                                cur_raw_mbar_addr: T.uint32 = T.cuda.cvta_generic_to_shared(
                                    bar_raw_ready.ptr_to([rs_buf.stage])
                                )
                                cur_indices = T.alloc_local((4,), "int32")
                                next_indices = T.alloc_local((4,), "int32")
                                Tx.copy(
                                    cur_indices[0:4],
                                    tma_coord[rs_index.stage, 0:4],
                                    dispatch="vec_128b",
                                )
                                for row4 in T.unroll(B_TOPK // 4):
                                    row: T.let = row4 * 4
                                    if row + 4 < B_TOPK:
                                        Tx.copy(
                                            next_indices[0:4],
                                            tma_coord[rs_index.stage, row + 4 : row + 8],
                                            dispatch="vec_128b",
                                        )
                                    issue_gather4(
                                        cur_raw_stage_addr + T.cast(row * d_nope, "uint32"),
                                        cur_raw_mbar_addr,
                                        _select_tensormap(
                                            block_idx >= num_orig_blocks,
                                            T.address_of(tensor_map_extra_kv_nope),
                                            T.address_of(tensor_map_kv_nope),
                                        ),
                                        0,
                                        cur_indices,
                                    )
                                    assign_int4_registers(cur_indices, next_indices)
                                bar_raw_ready.arrive(rs_buf.stage, tx_count=B_TOPK * d_nope)
                                bar_valid_free.arrive(rs_index.stage)
                                rs_buf.advance()
                                rs_index.advance()

                        elif role == 6:
                            # kernel.cuh:616-652.  RoPE remains bf16 and uses the
                            # model-specific SW64 (two 32-col gathers) or SW128
                            # (one 64-col gather) destination.
                            rope_stage0_base_u64: T.let = T.reinterpret(
                                "uint64", rope_stage_base_ptr(k_full, k_rope, 0)
                            )
                            rope_stage1_base_u64: T.let = T.reinterpret(
                                "uint64", rope_stage_base_ptr(k_full, k_rope, 1)
                            )
                            bar_q_utccp.wait(0, batch_bar_phase)
                            bar_last_store_done.wait(0, batch_bar_phase)
                            for block_idx in T.serial(start_block, end_block, unroll=False):
                                bar_valid_ready.wait(rs_index.stage, rs_index.phase)
                                if model_is_v32:
                                    bar_qk_done.wait(rs_buf.stage, rs_buf.phase ^ 1)
                                else:
                                    bar_sv_done.wait(rs_buf.stage, rs_buf.phase ^ 1)
                                # kernel.cuh:640 selects the current ring-stage
                                # ``rope.data()`` once for this dynamic block.
                                # Keep this mutable scalar outside both
                                # source-unrolled loops so lowering does not
                                # clone the stage select into every gather.
                                cur_rope_stage_base_u64: T.uint64 = rope_stage0_base_u64
                                if rs_buf.stage != 0:
                                    cur_rope_stage_base_u64 = rope_stage1_base_u64
                                cur_rope_stage_addr: T.uint32 = T.cuda.cvta_generic_to_shared(
                                    T.reinterpret(
                                        PointerType(PrimType("bfloat16")), cur_rope_stage_base_u64
                                    )
                                )
                                cur_rope_mbar_addr: T.uint32 = T.cuda.cvta_generic_to_shared(
                                    bar_rope_ready.ptr_to([rs_buf.stage])
                                )
                                cur_indices = T.alloc_local((4,), "int32")
                                next_indices = T.alloc_local((4,), "int32")
                                Tx.copy(
                                    cur_indices[0:4],
                                    tma_coord[rs_index.stage, 0:4],
                                    dispatch="vec_128b",
                                )
                                for row4 in T.unroll(B_TOPK // 4):
                                    row: T.let = row4 * 4
                                    if row + 4 < B_TOPK:
                                        Tx.copy(
                                            next_indices[0:4],
                                            tma_coord[rs_index.stage, row + 4 : row + 8],
                                            dispatch="vec_128b",
                                        )
                                    for rope_part in T.unroll(64 // rope_tile):
                                        rope_dst_addr: T.let = cur_rope_stage_addr + T.cast(
                                            (rope_part * B_TOPK + row) * rope_tile * BF16_BYTES,
                                            "uint32",
                                        )
                                        issue_gather4(
                                            rope_dst_addr,
                                            cur_rope_mbar_addr,
                                            _select_tensormap(
                                                block_idx >= num_orig_blocks,
                                                T.address_of(tensor_map_extra_kv_rope),
                                                T.address_of(tensor_map_kv_rope),
                                            ),
                                            rope_part * rope_tile,
                                            cur_indices,
                                        )
                                    assign_int4_registers(cur_indices, next_indices)
                                bar_rope_ready.arrive(
                                    rs_buf.stage, tx_count=B_TOPK * 64 * BF16_BYTES
                                )
                                bar_valid_free.arrive(rs_index.stage)
                                rs_buf.advance()
                                rs_index.advance()

                        elif role == 7:
                            # kernel.cuh:653-743.  All 32 lanes transform exactly
                            # two indices, form TMA coordinates, load/convert the
                            # model-specific scales, and construct each 8-bit mask.
                            indices_base: T.let = (
                                batch_idx * stride_indices_b + s_q_idx * stride_indices_s_q
                            )
                            extra_indices_base: T.let = (
                                batch_idx * stride_extra_indices_b
                                + s_q_idx * stride_extra_indices_s_q
                            )

                            @T.inline
                            def process_index_block(cur_block, is_extra: T.constexpr):
                                abs_pos: T.let = T.if_then_else(
                                    is_extra,
                                    (cur_block - num_orig_blocks) * B_TOPK + lane_idx * 2,
                                    cur_block * B_TOPK + lane_idx * 2,
                                )
                                cur_page_size: T.let = T.if_then_else(
                                    is_extra, extra_page_block_size, page_block_size
                                )
                                cur_block_stride: T.let = T.if_then_else(
                                    is_extra, stride_extra_kv_block, stride_kv_block
                                )
                                cur_row_stride: T.let = T.if_then_else(
                                    is_extra, stride_extra_kv_row, stride_kv_row
                                )
                                cur_length: T.let = T.if_then_else(
                                    is_extra, extra_topk_len, topk_len
                                )
                                cur_k_scales_ptr_u64: T.let = T.if_then_else(
                                    is_extra, extra_k_scales_ptr_u64, k_scales_ptr_u64
                                )
                                cur_tma_coords_step_per_block: T.let = T.if_then_else(
                                    is_extra,
                                    tma_coords_step_per_extra_block,
                                    tma_coords_step_per_block,
                                )

                                pair_indices = T.alloc_local((2,), "int32")
                                if is_extra:
                                    Tx.copy(
                                        pair_indices[0:2],
                                        extra_indices[
                                            extra_indices_base + abs_pos : extra_indices_base
                                            + abs_pos
                                            + 2
                                        ],
                                        dispatch="vec_64b",
                                        cache="nc",
                                    )
                                else:
                                    Tx.copy(
                                        pair_indices[0:2],
                                        indices[
                                            indices_base + abs_pos : indices_base + abs_pos + 2
                                        ],
                                        dispatch="vec_64b",
                                        cache="nc",
                                    )
                                bar_valid_free.wait(rs_index.stage, rs_index.phase ^ 1)
                                coords = T.alloc_local((2,), "int32")
                                cache_blocks = T.alloc_local((2,), "uint32")
                                indices_in_block = T.alloc_local((2,), "uint32")
                                scale_words = T.alloc_local((2,), "uint64")
                                pair_token_valid = T.alloc_local((2,), "bool")
                                scale_f32 = T.alloc_local((2, 4), "float32")
                                scale_byte_offsets = T.alloc_local((2,), "uint64")
                                valid_mask: T.int8 = T.int8(0)
                                for pair_i in T.unroll(2):
                                    index_u32: T.let = T.cast(pair_indices[pair_i], "uint32")
                                    cache_blocks[pair_i] = index_u32 // T.cast(
                                        cur_page_size, "uint32"
                                    )
                                    indices_in_block[pair_i] = index_u32 % T.cast(
                                        cur_page_size, "uint32"
                                    )
                                    token_valid: T.let = T.And(
                                        pair_indices[pair_i] != -1, abs_pos + pair_i < cur_length
                                    )
                                    pair_token_valid[pair_i] = token_valid
                                    valid_mask = T.cast(
                                        T.bitwise_or(
                                            T.cast(valid_mask, "int32"),
                                            T.shift_left(
                                                T.cast(token_valid, "int32"),
                                                T.cast(pair_i, "int32"),
                                            ),
                                        ),
                                        "int8",
                                    )
                                    coords[pair_i] = T.if_then_else(
                                        pair_token_valid[pair_i],
                                        T.cast(cache_blocks[pair_i], "int32")
                                        * cur_tma_coords_step_per_block
                                        + T.cast(indices_in_block[pair_i], "int32")
                                        * tma_coords_step_per_token,
                                        -1,
                                    )
                                    if model_is_v32:
                                        # k_scales_ptr is kv+D_NOPE even for an
                                        # invalid token; invalid entries therefore
                                        # load token-0's scale float4 and are zeroed
                                        # only after conversion, exactly as CUDA.
                                        scale_byte_offsets[pair_i] = T.if_then_else(
                                            pair_token_valid[pair_i],
                                            T.cast(cache_blocks[pair_i], "uint64")
                                            * T.cast(cur_block_stride, "int64")
                                            + T.cast(indices_in_block[pair_i], "uint64")
                                            * T.cast(cur_row_stride, "int64"),
                                            T.uint64(0),
                                        )
                                        # The CUDA loop is source-unrolled.  Issue
                                        # both token float4 loads before consuming
                                        # either value in F2FP.E8 so the random
                                        # scale reads overlap in flight.
                                        T.cuda.ldg(
                                            T.reinterpret(
                                                PointerType(PrimType("float32")),
                                                cur_k_scales_ptr_u64 + scale_byte_offsets[pair_i],
                                            ),
                                            "float32",
                                            dst=(
                                                scale_f32.ptr_to([pair_i, 0]),
                                                scale_f32.ptr_to([pair_i, 1]),
                                                scale_f32.ptr_to([pair_i, 2]),
                                                scale_f32.ptr_to([pair_i, 3]),
                                            ),
                                            vec="v4",
                                        )
                                    else:
                                        scale_byte_offsets[pair_i] = (
                                            T.cast(cache_blocks[pair_i], "uint64")
                                            * T.cast(cur_block_stride, "int64")
                                            + T.cast(indices_in_block[pair_i], "uint64") * 8
                                        )
                                        scale_words[pair_i] = T.if_then_else(
                                            pair_token_valid[pair_i],
                                            T.cuda.ldg(
                                                T.reinterpret(
                                                    PointerType(PrimType("uint64")),
                                                    cur_k_scales_ptr_u64
                                                    + scale_byte_offsets[pair_i],
                                                ),
                                                "uint64",
                                            ),
                                            T.uint64(0),
                                        )

                                if model_is_v32:
                                    for pair_i in T.unroll(2):
                                        packed_scale: T.let = _f32x4_to_e8m0x4(
                                            scale_f32[pair_i, 0],
                                            scale_f32[pair_i, 1],
                                            scale_f32[pair_i, 2],
                                            scale_f32[pair_i, 3],
                                        )
                                        scale_words[pair_i] = T.if_then_else(
                                            pair_token_valid[pair_i],
                                            T.cast(packed_scale, "uint64"),
                                            T.uint64(0),
                                        )

                                valid_mask = T.cast(
                                    T.shift_left(
                                        T.cast(valid_mask, "int32"),
                                        T.cast((lane_idx % 4) * 2, "int32"),
                                    ),
                                    "int8",
                                )
                                valid_mask = T.cast(
                                    T.bitwise_or(
                                        T.cast(valid_mask, "int32"),
                                        T.cuda.__shfl_xor_sync(
                                            T.uint32(0xFFFFFFFF), T.cast(valid_mask, "int32"), 1, 32
                                        ),
                                    ),
                                    "int8",
                                )
                                valid_mask = T.cast(
                                    T.bitwise_or(
                                        T.cast(valid_mask, "int32"),
                                        T.cuda.__shfl_xor_sync(
                                            T.uint32(0xFFFFFFFF), T.cast(valid_mask, "int32"), 2, 32
                                        ),
                                    ),
                                    "int8",
                                )
                                if model_is_v32:
                                    scales_e8m0.view("uint64")[rs_index.stage, lane_idx] = (
                                        T.bitwise_or(
                                            scale_words[0],
                                            T.shift_left(scale_words[1], T.uint64(32)),
                                        )
                                    )
                                else:
                                    Tx.copy(
                                        scales_e8m0.view("uint64")[
                                            rs_index.stage, lane_idx * 2 : lane_idx * 2 + 2
                                        ],
                                        scale_words[0:2],
                                        dispatch="vec_128b",
                                    )
                                Tx.copy(
                                    tma_coord[rs_index.stage, lane_idx * 2 : lane_idx * 2 + 2],
                                    coords[0:2],
                                    dispatch="vec_64b",
                                )
                                if lane_idx % 4 == 0:
                                    is_token_valid[rs_index.stage, lane_idx // 4] = valid_mask
                                bar_valid_ready.arrive(rs_index.stage)
                                rs_buf.advance()
                                rs_index.advance()

                            for block_idx in T.serial(
                                start_block, T.min(num_orig_blocks, end_block), unroll=False
                            ):
                                process_index_block(block_idx, False)
                            for block_idx in T.serial(
                                T.max(start_block, num_orig_blocks), end_block, unroll=False
                            ):
                                process_index_block(block_idx, True)

                        T.ptx.barrier.sync(BAR_EVERYONE_SYNC, NUM_THREADS)
                        batch_bar_phase = batch_bar_phase ^ 1

            # kernel.cuh:431/586/616/653/744.  Election is evaluated only in
            # the matching warp.  Record the selected source branch first so
            # every non-elected lane reaches the one shared final ``else``
            # scheduler, rather than cloning that empty scheduler once for
            # each producer warp.
            selected_wg1_role: T.int32 = -1
            if wg1_warp_idx == 4:
                if T.ptx.elect_sync() != T.uint32(0):
                    selected_wg1_role = 4
            elif wg1_warp_idx == 5:
                if T.ptx.elect_sync() != T.uint32(0):
                    selected_wg1_role = 5
            elif wg1_warp_idx == 6:
                if T.ptx.elect_sync() != T.uint32(0):
                    selected_wg1_role = 6
            elif wg1_warp_idx == 7:
                selected_wg1_role = 7

            if selected_wg1_role == 4:
                run_wg1_role(4)
            elif selected_wg1_role == 5:
                run_wg1_role(5)
            elif selected_wg1_role == 6:
                run_wg1_role(6)
            elif selected_wg1_role == 7:
                run_wg1_role(7)
            else:
                run_wg1_role(-1)
        else:
            # kernel.cuh:747-759.  The dequant warpgroup keeps 208 registers
            # and assigns exactly eight threads per token group.
            T.ptx.setmaxnreg(True, 208)
            rs_buf = PipelineState(NUM_BUFS, phase=0)
            rs_index = PipelineState(NUM_INDEX_BUFS, phase=0)
            group_idx: T.let = idx_in_warpgroup // 8
            idx_in_group: T.let = idx_in_warpgroup % 8

            # kernel.cuh:751-758.  Keep both dequant-stage and raw-stage
            # per-thread bases live across the scheduler loop.  The source
            # selects one pointer from each pair once per block.
            nope0_base_u64: T.let = T.reinterpret(
                "uint64", k_full.ptr_to([0, group_idx, idx_in_group * 8])
            )
            nope1_base_u64: T.let = T.reinterpret(
                "uint64", k_full.ptr_to([1, group_idx, idx_in_group * 8])
            )
            raw_nope0_base_u64: T.let = T.reinterpret(
                "uint64", raw_nope.ptr_to([0, group_idx, idx_in_group])
            )
            raw_nope1_base_u64: T.let = T.reinterpret(
                "uint64", raw_nope.ptr_to([1, group_idx, idx_in_group])
            )

            # kernel.cuh:77-118 expanded for WG2, preserving scheduler loads
            # in this specialization to avoid cross-role register pressure.
            sched_words = T.alloc_local((4,), "uint64")
            load_scheduler_meta(sched_words)
            sched_i32 = sched_words.view("int32")
            sched_begin_req: T.let = sched_i32[0]
            sched_end_req: T.let = sched_i32[1]
            sched_begin_block: T.let = sched_i32[2]
            sched_end_block: T.let = sched_i32[3]
            sched_begin_split: T.let = sched_i32[4]
            sched_first_split: T.let = sched_i32[5]
            sched_last_split: T.let = sched_i32[6]
            batch_bar_phase: T.int32 = 0

            # The CUDA return exits only this role's run_main_loop lambda.
            if sched_begin_req < b:
                for batch_idx in T.serial(sched_begin_req, sched_end_req + 1, unroll=False):
                    topk_len: T.int32 = topk
                    if not T.isnullptr(topk_length_h):
                        topk_len = T.cuda.ldg(topk_length.ptr_to([batch_idx]), "int32")
                    orig_topk_padded: T.let = T.max(
                        ((topk_len + B_TOPK - 1) // B_TOPK) * B_TOPK, B_TOPK
                    )
                    extra_topk_len: T.int32 = extra_topk
                    if not T.isnullptr(extra_topk_length_h):
                        extra_topk_len = T.cuda.ldg(extra_topk_length.ptr_to([batch_idx]), "int32")
                    total_topk_padded: T.let = (
                        orig_topk_padded + ((extra_topk_len + B_TOPK - 1) // B_TOPK) * B_TOPK
                    )
                    start_block: T.let = T.if_then_else(
                        batch_idx == sched_begin_req, sched_begin_block, 0
                    )
                    end_block: T.let = T.if_then_else(
                        batch_idx == sched_end_req, sched_end_block, total_topk_padded // B_TOPK
                    )
                    is_split: T.bool = T.cast(
                        T.if_then_else(
                            batch_idx == sched_begin_req,
                            sched_first_split,
                            T.if_then_else(batch_idx == sched_end_req, sched_last_split, 0),
                        ),
                        "bool",
                    )
                    is_no_split: T.bool = not is_split
                    n_split_idx: T.let = T.if_then_else(
                        batch_idx == sched_begin_req,
                        T.cuda.ldg(num_splits.ptr_to([batch_idx]), "int32") + sched_begin_split,
                        T.cuda.ldg(num_splits.ptr_to([batch_idx]), "int32"),
                    )
                    num_orig_blocks: T.let = orig_topk_padded // B_TOPK
                    is_last_batch: T.bool = batch_idx == sched_end_req

                    # kernel.cuh:760-840.  Wait on Q, raw fp8 and the previous
                    # SxV use, then convert each fp8x8 with the exact ue8m0
                    # scale and weak shared b128 store from the source.
                    bar_q_utccp.wait(0, batch_bar_phase)
                    for block_idx in T.serial(start_block, end_block, unroll=False):
                        bar_valid_ready.wait(rs_index.stage, rs_index.phase)
                        bar_raw_ready.wait(rs_buf.stage, rs_buf.phase)
                        bar_sv_done.wait(rs_buf.stage, rs_buf.phase ^ 1)
                        cur_nope_base_u64: T.let = T.if_then_else(
                            rs_buf.stage == 0, nope0_base_u64, nope1_base_u64
                        )
                        cur_raw_nope_base_u64: T.let = T.if_then_else(
                            rs_buf.stage == 0, raw_nope0_base_u64, raw_nope1_base_u64
                        )
                        cur_nope_base_uint_addr: T.let = T.cuda.cvta_generic_to_shared(
                            T.reinterpret(PointerType(PrimType("bfloat16")), cur_nope_base_u64)
                        )
                        cur_raw_nope_base_uint_addr: T.let = T.cuda.cvta_generic_to_shared(
                            T.reinterpret(PointerType(PrimType("uint64")), cur_raw_nope_base_u64)
                        )
                        for local_row in T.unroll(rows_per_group):
                            row_idx: T.let = local_row * (128 // 8) + group_idx
                            scales_bf16_bits = T.alloc_local((num_scales,), "uint16")
                            if model_is_v32:
                                packed_scales: T.let = scales_e8m0.view("uint32")[
                                    rs_index.stage, row_idx
                                ]
                                for scale_pair_idx in T.unroll(2):
                                    converted_pair: T.let = _e8m0x2_to_bf16x2(
                                        T.cast(
                                            T.shift_right(
                                                packed_scales, T.cast(scale_pair_idx * 16, "uint32")
                                            ),
                                            "uint16",
                                        )
                                    )
                                    scales_bf16_bits[scale_pair_idx * 2] = T.cast(
                                        converted_pair, "uint16"
                                    )
                                    scales_bf16_bits[scale_pair_idx * 2 + 1] = T.cast(
                                        T.shift_right(converted_pair, T.uint32(16)), "uint16"
                                    )
                            else:
                                packed_scales: T.let = scales_e8m0.view("uint64")[
                                    rs_index.stage, row_idx
                                ]
                                for scale_pair_idx in T.unroll(4):
                                    converted_pair: T.let = _e8m0x2_to_bf16x2(
                                        T.cast(
                                            T.shift_right(
                                                packed_scales, T.cast(scale_pair_idx * 16, "uint64")
                                            ),
                                            "uint16",
                                        )
                                    )
                                    scales_bf16_bits[scale_pair_idx * 2] = T.cast(
                                        converted_pair, "uint16"
                                    )
                                    scales_bf16_bits[scale_pair_idx * 2 + 1] = T.cast(
                                        T.shift_right(converted_pair, T.uint32(16)), "uint16"
                                    )

                            cur_raw_fp8x8: T.uint64 = _ld_shared_u64(
                                cur_raw_nope_base_uint_addr
                                + T.cast(local_row * (128 // 8) * d_nope, "uint32")
                            )
                            for local_col in T.unroll(cols_per_group):
                                raw_fp8x8: T.let = cur_raw_fp8x8
                                if local_col + 1 < cols_per_group:
                                    cur_raw_fp8x8 = _ld_shared_u64(
                                        cur_raw_nope_base_uint_addr
                                        + T.cast(
                                            local_row * (128 // 8) * d_nope
                                            + (local_col + 1) * (8 * 8),
                                            "uint32",
                                        )
                                    )
                                scale_idx: T.let = (
                                    local_col // (cols_per_group // 4)
                                    if model_is_v32
                                    else local_col
                                )
                                _dequant_st128(
                                    cur_nope_base_uint_addr
                                    + T.cast(
                                        BF16_BYTES
                                        * (local_row * (128 // 8) * 64 + local_col * B_TOPK * 64),
                                        "uint32",
                                    ),
                                    raw_fp8x8,
                                    scales_bf16_bits[scale_idx],
                                )
                        T.ptx.fence.proxy_async("shared::cta")
                        bar_nope_ready.arrive(rs_buf.stage)
                        bar_raw_free.arrive(rs_buf.stage)
                        bar_valid_free.arrive(rs_index.stage)
                        rs_buf.advance()
                        rs_index.advance()

                    T.ptx.barrier.sync(BAR_EVERYONE_SYNC, NUM_THREADS)
                    batch_bar_phase = batch_bar_phase ^ 1

    return _kernel


@T.jit
def _sparse_decode_head64_combine_kernel(
    lse_h: T.handle,
    out_h: T.handle,
    lse_accum_h: T.handle,
    o_accum_h: T.handle,
    num_splits_h: T.handle,
    attn_sink_h: T.handle,
    stride_lse_b: T.int32,
    stride_lse_s_q: T.int32,
    stride_o_b: T.int32,
    stride_o_s_q: T.int32,
    stride_o_h_q: T.int32,
    stride_lse_accum_split: T.int32,
    stride_lse_accum_s_q: T.int32,
    stride_o_accum_split: T.int32,
    stride_o_accum_s_q: T.int32,
    stride_o_accum_h_q: T.int32,
    b: T.int32,
    s_q: T.int32,
    h_q: T.int32,
    d_v: T.int32,
    num_sm_parts: T.int32,
    *,
    max_splits: T.constexpr,
):
    lse = T.match_buffer(lse_h, (b * s_q * h_q,), "float32", scope="global")
    out = T.match_buffer(out_h, (b * s_q * h_q * d_v,), "bfloat16", scope="global")
    lse_accum = T.match_buffer(
        lse_accum_h, ((b + num_sm_parts) * s_q * h_q,), "float32", scope="global"
    )
    o_accum = T.match_buffer(
        o_accum_h, ((b + num_sm_parts) * s_q * h_q * d_v,), "float32", scope="global"
    )
    num_splits = T.match_buffer(num_splits_h, (b + 1,), "int32", scope="global")
    attn_sink = T.match_buffer(attn_sink_h, (h_q,), "float32", scope="global")
    T.device_entry()
    # combine.cu:18-43.  Keep one warp per head, eight heads per CTA, the
    # early no-split return, and the MAX_SPLITS bucket selected by the host.
    batch_s_q_idx_expr, _, h_block_idx_expr = T.cta_id([b * s_q, 1, (h_q + 7) // 8])
    thread_idx_expr = T.thread_id([8 * 32])
    # Keep CUDA's prologue values as registers instead of substituting their
    # division/modulo expressions into every source-unrolled address.  These
    # are the exact const-int values at combine.cu:21-32.
    batch_s_q_idx: T.int32 = batch_s_q_idx_expr
    h_block_idx: T.int32 = h_block_idx_expr
    thread_idx: T.int32 = thread_idx_expr
    warp_idx: T.int32 = thread_idx // 32
    lane_idx: T.int32 = thread_idx % 32
    batch_idx: T.int32 = batch_s_q_idx // s_q
    query_idx: T.int32 = batch_s_q_idx - batch_idx * s_q
    h_block_base: T.int32 = h_block_idx * 8
    head_idx: T.int32 = h_block_base + warp_idx
    num_valid_heads: T.int32 = T.min(8, h_q - h_block_base)
    if warp_idx >= num_valid_heads:
        return 0

    start_split: T.int32 = T.cuda.ldg(num_splits.ptr_to([batch_idx]), "int32")
    end_split: T.int32 = T.cuda.ldg(num_splits.ptr_to([batch_idx + 1]), "int32")
    my_num_splits: T.int32 = end_split - start_split
    if my_num_splits == 1:
        return 0

    T.cuda.trap_when_assert_failed(my_num_splits <= max_splits)

    # combine.cu:45-54.  Preserve the source gLseAccum/gLse base views.
    g_lse_accum_offset: T.int32 = (
        start_split * stride_lse_accum_split + query_idx * stride_lse_accum_s_q + h_block_base
    )
    g_lse_offset: T.int32 = batch_idx * stride_lse_b + query_idx * stride_lse_s_q + h_block_base
    g_lse_accum = T.decl_buffer(
        (max_splits, 8),
        "float32",
        data=lse_accum.data,
        scope="global",
        elem_offset=g_lse_accum_offset,
        layout=TileLayout(S[(max_splits, 8) : (stride_lse_accum_split, 1)]),
    )
    g_lse = T.decl_buffer((8,), "float32", data=lse.data, scope="global", elem_offset=g_lse_offset)

    # combine.cu:56.  This is static shared storage; launch-time dynamic bytes
    # stay zero.
    lse_scales = T.alloc_buffer((8, max_splits), "float32", scope="shared")

    # combine.cu:58-69.  The PDL consumer wait remains after both early
    # returns, followed by the same four float4 prefetches per lane.
    T.evaluate(T.ptx.griddepcontrol.wait())
    oaccum_offset: T.int32 = (
        start_split * stride_o_accum_split
        + query_idx * stride_o_accum_s_q
        + head_idx * stride_o_accum_h_q
    )
    oaccum_ptr = T.decl_buffer(
        (num_sm_parts * stride_o_accum_split + D_V,),
        "float32",
        data=o_accum.data,
        scope="global",
        elem_offset=oaccum_offset,
    )
    datas = T.alloc_local((D_V // (32 * 4), 4), "float32")
    for elem_i in T.unroll(D_V // (32 * 4)):
        Tx.copy(
            datas[elem_i, 0:4],
            oaccum_ptr[lane_idx * 4 + elem_i * 128 : lane_idx * 4 + elem_i * 128 + 4],
            dispatch="vec_128b",
        )

    # combine.cu:71-119.  Gather log2 LSE, warp-reduce max/sum, store the
    # public natural-log LSE, then fold attn_sink only into normalization.
    local_lse = T.alloc_local(((max_splits + 31) // 32,), "float32")
    for lse_i in T.unroll((max_splits + 31) // 32):
        split_idx: T.let = lse_i * 32 + lane_idx
        local_lse[lse_i] = T.if_then_else(
            split_idx < my_num_splits, g_lse_accum[split_idx, warp_idx], T.float32(-float("inf"))
        )
    max_lse: T.float32 = T.float32(-float("inf"))
    for lse_i in T.unroll((max_splits + 31) // 32):
        max_lse = T.max(max_lse, local_lse[lse_i])
    for reduce_i in T.unroll(5):
        xor_offset: T.let = 16 >> reduce_i
        max_lse = T.max(
            max_lse, T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), max_lse, xor_offset, 32)
        )
    max_lse = T.if_then_else(max_lse == T.float32(-float("inf")), 0.0, max_lse)
    sum_lse: T.float32 = 0.0
    for lse_i in T.unroll((max_splits + 31) // 32):
        sum_lse = sum_lse + T.ptx.exp2(local_lse[lse_i] - max_lse)
    for reduce_i in T.unroll(5):
        xor_offset: T.let = 16 >> reduce_i
        sum_lse = sum_lse + T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), sum_lse, xor_offset, 32)
    global_lse: T.float32 = T.if_then_else(
        T.Or(sum_lse == 0.0, sum_lse == T.float32(-float("inf"))),
        T.float32(float("inf")),
        T.log2(sum_lse) + max_lse,
    )
    if lane_idx == 0:
        g_lse[warp_idx] = global_lse / T.float32(LOG_2_E)

    if not T.isnullptr(attn_sink_h):
        sink: T.let = T.cuda.ldg(attn_sink.ptr_to([head_idx]), "float32")
        if global_lse != T.float32(float("inf")):
            global_lse = global_lse + T.log2(1.0 + T.ptx.exp2(sink * LOG_2_E - global_lse))
        else:
            global_lse = T.if_then_else(
                sink == T.float32(-float("inf")), T.float32(float("inf")), sink * LOG_2_E
            )
    for lse_i in T.unroll((max_splits + 31) // 32):
        split_idx: T.let = lse_i * 32 + lane_idx
        lse_scales[warp_idx, split_idx] = T.ptx.exp2(local_lse[lse_i] - global_lse)
    T.cuda.warp_sync()

    # combine.cu:123-160.  Keep the unroll-1 split traversal and the
    # source's next-split float4 prefetch inside the accumulation loop.
    result = T.alloc_local((D_V // (32 * 4), 4), "float32")
    for elem_i in T.unroll(D_V // (32 * 4)):
        for vec_i in T.unroll(4):
            result[elem_i, vec_i] = 0.0
    for split_idx in T.serial(0, my_num_splits, unroll=False):
        lse_scale: T.let = lse_scales[warp_idx, split_idx]
        for elem_i in T.unroll(D_V // (32 * 4)):
            for vec_i in T.unroll(4):
                result[elem_i, vec_i] = result[elem_i, vec_i] + lse_scale * datas[elem_i, vec_i]
            if split_idx != my_num_splits - 1:
                Tx.copy(
                    datas[elem_i, 0:4],
                    oaccum_ptr[
                        (split_idx + 1) * stride_o_accum_split + lane_idx * 4 + elem_i * 128 : (
                            split_idx + 1
                        )
                        * stride_o_accum_split
                        + lane_idx * 4
                        + elem_i * 128
                        + 4
                    ],
                    dispatch="vec_128b",
                )

    out_offset: T.int32 = (
        batch_idx * stride_o_b + query_idx * stride_o_s_q + head_idx * stride_o_h_q
    )
    o_ptr = T.decl_buffer((D_V,), "bfloat16", data=out.data, scope="global", elem_offset=out_offset)
    for elem_i in T.unroll(D_V // (32 * 4)):
        data_converted = T.alloc_local((4,), "bfloat16")
        data_converted[0] = result[elem_i, 0]
        data_converted[1] = result[elem_i, 1]
        data_converted[2] = result[elem_i, 2]
        data_converted[3] = result[elem_i, 3]
        o_ptr.view("uint64")[(lane_idx * 4 + elem_i * 128) // 4] = data_converted.view("uint64")[0]


def _kernel_shape_params(
    cfg: SparseFlashMLADecodeHead64Config,
    device: torch.device | str,
    *,
    prepared_num_blocks: int | None = None,
    prepared_extra_num_blocks: int | None = None,
) -> dict[str, int]:
    num_sm_parts = _device_sm_count(device) // cfg.s_q
    num_sm_parts = max(num_sm_parts, 1)
    num_blocks = (
        prepared_num_blocks
        if prepared_num_blocks is not None
        else _ceil_div(cfg.s_kv, cfg.page_block_size)
    )
    _, tma_k_stride, stride_kv_block, num_tma_rows = _kv_storage_spec(
        cfg.normalized_model_type, num_blocks, cfg.page_block_size
    )

    if cfg.extra_topk:
        extra_num_blocks = (
            prepared_extra_num_blocks
            if prepared_extra_num_blocks is not None
            else _ceil_div(cfg.extra_s_kv, cfg.extra_page_block_size)
        )
        (_, extra_tma_k_stride, stride_extra_kv_block, extra_num_tma_rows) = _kv_storage_spec(
            cfg.normalized_model_type, extra_num_blocks, cfg.extra_page_block_size
        )
        if extra_tma_k_stride != tma_k_stride:
            raise AssertionError("original and extra KV caches must use one MODEL_TYPE")
        extra_page_block_size = cfg.extra_page_block_size
    else:
        # kernel.cuh:934-950 leaves both optional TensorMaps all-zero and all
        # optional runtime shape/stride fields at zero when extra KV is absent.
        extra_num_blocks = 0
        extra_page_block_size = 0
        stride_extra_kv_block = 0
        extra_num_tma_rows = 0

    return {
        "num_sm_parts": num_sm_parts,
        "num_blocks": num_blocks,
        "stride_kv_block": stride_kv_block,
        "num_tma_rows": num_tma_rows,
        "kv_bytes": num_blocks * stride_kv_block,
        "extra_num_blocks": extra_num_blocks,
        "extra_page_block_size": extra_page_block_size,
        "stride_extra_kv_block": stride_extra_kv_block,
        "extra_num_tma_rows": extra_num_tma_rows,
        "extra_kv_bytes": extra_num_blocks * stride_extra_kv_block,
        "extra_indices_elems": cfg.b * cfg.s_q * cfg.extra_topk,
        "split_rows": cfg.b + num_sm_parts,
        "max_splits": _max_splits_bucket(num_sm_parts),
    }


@lru_cache(maxsize=10)
def _specialized_decode_kernels(model_type: ModelType, max_splits: int):
    main = (
        _make_sparse_decode_head64_kernel(model_type)
        .specialize()
        .with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))
        .with_attr("tirx.nullable_buffer_params", list(NULLABLE_MAIN_BUFFER_PARAMS))
    )
    combine = _sparse_decode_head64_combine_kernel.specialize(max_splits=max_splits).with_attr(
        "tirx.kernel_launch_params", list(COMBINE_LAUNCH_TAGS)
    )
    combine = combine.with_attr("tirx.nullable_buffer_params", list(NULLABLE_COMBINE_BUFFER_PARAMS))
    return main, combine


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA decode")
    device = kwargs.get("device", "cuda")
    shape = _kernel_shape_params(cfg, device)
    return list(_specialized_decode_kernels(cfg.normalized_model_type, shape["max_splits"]))


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA decode")
    device = torch.device(kwargs.get("device", "cuda"))
    props = torch.cuda.get_device_properties(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    if props.major != 10:
        raise SkipTest(f"SM100f is required, got compute capability {props.major}.{props.minor}")

    device_generator = torch.Generator(device=device)
    device_generator.manual_seed(cfg.seed)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(cfg.seed)
    python_rng = random.Random(cfg.seed)

    q_contiguous = torch.randn(
        (cfg.b, cfg.s_q, cfg.h_q, cfg.d_qk),
        dtype=torch.bfloat16,
        device=device,
        generator=device_generator,
    )
    q_contiguous.clamp_(min=-1.0, max=1.0)

    if cfg.have_attn_sink:
        attn_sink = torch.randn(
            (cfg.h_q,), dtype=torch.float32, device=device, generator=device_generator
        )
        inf_mask = torch.randn(
            (cfg.h_q,), dtype=torch.float32, device=device, generator=device_generator
        )
        attn_sink[inf_mask > 0.5] = float("inf")
        attn_sink[inf_mask < -0.5] = -float("inf")
    else:
        attn_sink = torch.zeros((cfg.h_q,), dtype=torch.float32, device=device)

    kv_scope = _prepare_upstream_kv_scope(
        cfg,
        s_kv=cfg.s_kv,
        topk=cfg.topk,
        page_block_size=cfg.page_block_size,
        have_topk_length=cfg.have_topk_length,
        device=device,
        device_generator=device_generator,
        cpu_generator=cpu_generator,
        python_rng=python_rng,
    )

    if cfg.extra_topk:
        extra_scope = _prepare_upstream_kv_scope(
            cfg,
            s_kv=cfg.extra_s_kv,
            topk=cfg.extra_topk,
            page_block_size=cfg.extra_page_block_size,
            have_topk_length=cfg.have_extra_topk_length,
            device=device,
            device_generator=device_generator,
            cpu_generator=cpu_generator,
            python_rng=python_rng,
        )
    else:
        extra_scope = None

    shape = _kernel_shape_params(
        cfg,
        device,
        prepared_num_blocks=kv_scope["num_blocks"],
        prepared_extra_num_blocks=(extra_scope["num_blocks"] if extra_scope is not None else None),
    )
    if (
        kv_scope["stride_kv_block"] != shape["stride_kv_block"]
        or kv_scope["num_tma_rows"] != shape["num_tma_rows"]
    ):
        raise AssertionError("prepared KV layout disagrees with runtime shape")
    if extra_scope is not None and (
        extra_scope["stride_kv_block"] != shape["stride_extra_kv_block"]
        or extra_scope["num_tma_rows"] != shape["extra_num_tma_rows"]
    ):
        raise AssertionError("prepared extra KV layout disagrees with runtime shape")

    q = _noncontiguous_copy(q_contiguous)
    del q_contiguous
    indices = kv_scope["indices"]
    if cfg.inject_invalid_indices:
        indices[:, :, 0] = -1
        indices[:, :, -1] = -1
        if extra_scope is not None:
            extra_scope["indices"][:, :, 0] = -1
            extra_scope["indices"][:, :, -1] = -1

    topk_length_cpu = kv_scope["topk_length_cpu"]
    extra_topk_length_cpu = (
        extra_scope["topk_length_cpu"]
        if extra_scope is not None
        else torch.zeros((cfg.b,), dtype=torch.int32)
    )
    tile_scheduler_metadata, num_splits = _build_decode_scheduler(
        cfg, topk_length_cpu, extra_topk_length_cpu, shape["num_sm_parts"], device
    )
    out = torch.empty((cfg.b, cfg.s_q, cfg.h_q, D_V), dtype=torch.bfloat16, device=device)
    lse = torch.empty((cfg.b, cfg.s_q, cfg.h_q), dtype=torch.float32, device=device)
    lse_accum = torch.empty(
        (shape["split_rows"], cfg.s_q, cfg.h_q), dtype=torch.float32, device=device
    )
    o_accum = torch.empty(
        (shape["split_rows"], cfg.s_q, cfg.h_q, D_V), dtype=torch.float32, device=device
    )
    sm_scale = cfg.d_qk**-0.55
    case = {
        "config": cfg,
        "shape": shape,
        "q": q,
        "kv": kv_scope["kv"],
        "kv_storage": kv_scope["kv_storage"],
        "indices": indices,
        "topk_length": kv_scope["topk_length"],
        "cache_seqlens_cpu": kv_scope["cache_seqlens_cpu"],
        "attn_sink": attn_sink,
        "lse": lse,
        "out": out,
        "lse_accum": lse_accum,
        "o_accum": o_accum,
        "tile_scheduler_metadata": tile_scheduler_metadata,
        "num_splits": num_splits,
        "extra_kv": extra_scope["kv"] if extra_scope is not None else None,
        "extra_kv_storage": extra_scope["kv_storage"] if extra_scope is not None else None,
        "extra_indices": extra_scope["indices"] if extra_scope is not None else None,
        "extra_topk_length": (
            extra_scope["topk_length"]
            if extra_scope is not None
            else torch.zeros((cfg.b,), dtype=torch.int32, device=device)
        ),
        "extra_cache_seqlens_cpu": (
            extra_scope["cache_seqlens_cpu"] if extra_scope is not None else None
        ),
        "sm_scale": sm_scale,
        "sm_scale_div_log2": sm_scale * LOG_2_E,
        "stride_q_b": q.stride(0),
        "stride_q_s_q": q.stride(1),
        "stride_q_h_q": q.stride(2),
        "stride_kv_block": kv_scope["kv"].stride(0),
        "stride_kv_row": kv_scope["kv"].stride(1),
        "stride_indices_b": indices.stride(0),
        "stride_indices_s_q": indices.stride(1),
        "stride_lse_b": lse.stride(0),
        "stride_lse_s_q": lse.stride(1),
        "stride_o_b": out.stride(0),
        "stride_o_s_q": out.stride(1),
        "stride_o_h_q": out.stride(2),
        "stride_extra_kv_block": extra_scope["kv"].stride(0) if extra_scope is not None else 0,
        "stride_extra_kv_row": extra_scope["kv"].stride(1) if extra_scope is not None else 0,
        "stride_extra_indices_b": (
            extra_scope["indices"].stride(0) if extra_scope is not None else 0
        ),
        "stride_extra_indices_s_q": (
            extra_scope["indices"].stride(1) if extra_scope is not None else 0
        ),
        "stride_lse_accum_split": lse_accum.stride(0),
        "stride_lse_accum_s_q": lse_accum.stride(1),
        "stride_o_accum_split": o_accum.stride(0),
        "stride_o_accum_s_q": o_accum.stride(1),
        "stride_o_accum_h_q": o_accum.stride(2),
    }
    case["tensor_maps"] = _build_sparse_kv_tensormaps(case)
    _validate_tirx_launch_case(case)
    return case


def _validate_tirx_launch_case(case: dict[str, Any]) -> None:
    """Mirror kernel.cuh:859-868/909-912 before TensorMap creation."""

    cfg: SparseFlashMLADecodeHead64Config = case["config"]
    if cfg.normalized_model_type is ModelType.MODEL1 and case["stride_kv_row"] != 584:
        raise ValueError("MODEL1 sparse FP8 decode requires stride_kv_row == 584 bytes")
    if case["kv_storage"].data_ptr() % 16 != 0:
        raise ValueError("KV cache base must be 16-byte aligned")
    tma_k_stride = 656 if cfg.normalized_model_type is ModelType.V32 else 576
    if case["stride_kv_block"] % tma_k_stride:
        raise ValueError("KV block stride must be divisible by MODEL_TYPE TMA_K_STRIDE")

    if cfg.extra_topk:
        if case["extra_kv_storage"].data_ptr() % 16 != 0:
            raise ValueError("extra KV cache base must be 16-byte aligned")
        if case["stride_extra_kv_block"] % tma_k_stride:
            raise ValueError("extra KV block stride must be divisible by TMA_K_STRIDE")


def _flat_storage_alias(
    tensor: torch.Tensor, *, element_offset: int = 0, extent: int | None = None
) -> torch.Tensor:
    """Expose a raw-pointer span while retaining the source tensor's storage."""

    if extent is None:
        extent = tensor.numel() - element_offset
    return tensor.as_strided(
        (extent,), (1,), storage_offset=tensor.storage_offset() + element_offset
    )


def _tirx_main_args(case: dict[str, Any], start_head_idx: int) -> tuple[Any, ...]:
    cfg: SparseFlashMLADecodeHead64Config = case["config"]
    if start_head_idx % B_H or start_head_idx + B_H > cfg.h_q:
        raise ValueError(f"invalid head64 slice {start_head_idx} for h_q={cfg.h_q}")

    q_extent = (
        (cfg.b - 1) * case["stride_q_b"]
        + (cfg.s_q - 1) * case["stride_q_s_q"]
        + (B_H - 1) * case["stride_q_h_q"]
        + cfg.d_qk
    )
    indices_extent = (
        (cfg.b - 1) * case["stride_indices_b"]
        + (cfg.s_q - 1) * case["stride_indices_s_q"]
        + cfg.topk
    )
    lse_extent = (cfg.b - 1) * case["stride_lse_b"] + (cfg.s_q - 1) * case["stride_lse_s_q"] + B_H
    out_extent = (
        (cfg.b - 1) * case["stride_o_b"]
        + (cfg.s_q - 1) * case["stride_o_s_q"]
        + (B_H - 1) * case["stride_o_h_q"]
        + D_V
    )
    lse_accum_extent = (
        (case["shape"]["split_rows"] - 1) * case["stride_lse_accum_split"]
        + (cfg.s_q - 1) * case["stride_lse_accum_s_q"]
        + B_H
    )
    o_accum_extent = (
        (case["shape"]["split_rows"] - 1) * case["stride_o_accum_split"]
        + (cfg.s_q - 1) * case["stride_o_accum_s_q"]
        + (B_H - 1) * case["stride_o_accum_h_q"]
        + D_V
    )
    extra_indices_extent = (
        (cfg.b - 1) * case["stride_extra_indices_b"]
        + (cfg.s_q - 1) * case["stride_extra_indices_s_q"]
        + cfg.extra_topk
    )
    return (
        _flat_storage_alias(
            case["q"], element_offset=start_head_idx * case["stride_q_h_q"], extent=q_extent
        ),
        case["kv_storage"],
        _flat_storage_alias(case["indices"], extent=indices_extent),
        case["topk_length"] if cfg.have_topk_length else None,
        (case["attn_sink"][start_head_idx : start_head_idx + B_H] if cfg.have_attn_sink else None),
        _flat_storage_alias(case["lse"], element_offset=start_head_idx, extent=lse_extent),
        _flat_storage_alias(
            case["out"], element_offset=start_head_idx * case["stride_o_h_q"], extent=out_extent
        ),
        _flat_storage_alias(
            case["lse_accum"], element_offset=start_head_idx, extent=lse_accum_extent
        ),
        _flat_storage_alias(
            case["o_accum"],
            element_offset=start_head_idx * case["stride_o_accum_h_q"],
            extent=o_accum_extent,
        ),
        case["tile_scheduler_metadata"],
        case["num_splits"],
        case["extra_kv_storage"],
        (
            _flat_storage_alias(case["extra_indices"], extent=extra_indices_extent)
            if case["extra_indices"] is not None
            else None
        ),
        case["extra_topk_length"] if cfg.have_extra_topk_length else None,
        case["tensor_maps"]["kv_nope"].ptr,
        case["tensor_maps"]["kv_rope"].ptr,
        case["tensor_maps"]["extra_kv_nope"].ptr,
        case["tensor_maps"]["extra_kv_rope"].ptr,
        case["sm_scale_div_log2"],
        case["stride_q_b"],
        case["stride_q_s_q"],
        case["stride_q_h_q"],
        case["stride_kv_block"],
        case["stride_kv_row"],
        case["stride_indices_b"],
        case["stride_indices_s_q"],
        case["stride_lse_b"],
        case["stride_lse_s_q"],
        case["stride_o_b"],
        case["stride_o_s_q"],
        case["stride_o_h_q"],
        case["stride_extra_kv_block"],
        case["stride_extra_kv_row"],
        case["stride_extra_indices_b"],
        case["stride_extra_indices_s_q"],
        case["stride_lse_accum_split"],
        case["stride_lse_accum_s_q"],
        case["stride_o_accum_split"],
        case["stride_o_accum_s_q"],
        case["stride_o_accum_h_q"],
        cfg.b,
        cfg.s_q,
        cfg.topk,
        cfg.extra_topk,
        case["shape"]["num_blocks"],
        case["shape"]["extra_num_blocks"],
        cfg.page_block_size,
        case["shape"]["extra_page_block_size"],
        case["shape"]["num_sm_parts"],
    )


def _tirx_combine_args(case: dict[str, Any]) -> tuple[Any, ...]:
    cfg: SparseFlashMLADecodeHead64Config = case["config"]
    return (
        case["lse"].reshape(-1),
        case["out"].reshape(-1),
        case["lse_accum"].reshape(-1),
        case["o_accum"].reshape(-1),
        case["num_splits"],
        case["attn_sink"] if cfg.have_attn_sink else None,
        case["stride_lse_b"],
        case["stride_lse_s_q"],
        case["stride_o_b"],
        case["stride_o_s_q"],
        case["stride_o_h_q"],
        case["stride_lse_accum_split"],
        case["stride_lse_accum_s_q"],
        case["stride_o_accum_split"],
        case["stride_o_accum_s_q"],
        case["stride_o_accum_h_q"],
        cfg.b,
        cfg.s_q,
        cfg.h_q,
        cfg.d_v,
        case["shape"]["num_sm_parts"],
    )


@lru_cache(maxsize=10)
def _compile_decode_kernels_cached(model_type: ModelType, max_splits: int):
    from tirx_kernels.runner import compile_kernel

    main, combine = _specialized_decode_kernels(model_type, max_splits)
    return compile_kernel(main), compile_kernel(combine)


def _compile_decode_kernels(**kwargs: Any):
    cfg = _cfg(**kwargs)
    device = kwargs.get("device", "cuda")
    shape = _kernel_shape_params(cfg, device)
    return _compile_decode_kernels_cached(cfg.normalized_model_type, shape["max_splits"])


def _launch_tirx(case: dict[str, Any], executables: tuple[Any, Any]) -> None:
    main_ex, combine_ex = executables
    _validate_tirx_launch_case(case)
    cfg: SparseFlashMLADecodeHead64Config = case["config"]
    for start_head_idx in range(0, cfg.h_q, B_H):
        main_ex(*_tirx_main_args(case, start_head_idx))
    combine_ex(*_tirx_combine_args(case))


def run_test(**kwargs: Any) -> None:
    cfg = _cfg(**kwargs)
    # Upstream clears the allocator before every generated case; keep the
    # 15-case performance sweep from retaining cached pressure-shape blocks.
    torch.cuda.empty_cache()
    case = prepare_data(**kwargs)
    executables = _compile_decode_kernels(**kwargs)

    from tirx_kernels.flashmla._flashmla_bench import _import_flash_mla, run_flashmla_sparse_decode

    flash_mla = _import_flash_mla()
    sched_meta, _ = flash_mla.get_mla_metadata()
    ref_out, ref_lse = run_flashmla_sparse_decode(case, sched_meta)
    torch.cuda.synchronize()

    # Validate the host replica against the actual CUDA scheduler.  Inactive
    # rows only have a defined begin_req_idx; the remaining CUDA fields may
    # contain shared-memory tail values and are never consumed.
    ref_metadata = sched_meta.tile_scheduler_metadata
    ref_num_splits = sched_meta.num_splits
    if ref_metadata is None or ref_num_splits is None:
        raise AssertionError("FlashMLA did not initialize decode scheduler metadata")
    ours_metadata = case["tile_scheduler_metadata"]
    torch.testing.assert_close(ours_metadata[:, 0], ref_metadata[:, 0], rtol=0, atol=0)
    active = ours_metadata[:, 0] < cfg.b
    torch.testing.assert_close(ours_metadata[active, :7], ref_metadata[active, :7], rtol=0, atol=0)
    torch.testing.assert_close(case["num_splits"], ref_num_splits, rtol=0, atol=0)

    case["out"].fill_(float("nan"))
    case["lse"].fill_(float("nan"))
    _launch_tirx(case, executables)
    torch.cuda.synchronize()
    torch.testing.assert_close(case["out"], ref_out, rtol=2.01 / 128, atol=1.0e-3)
    torch.testing.assert_close(case["lse"], ref_lse.transpose(1, 2), rtol=8.01 / 65536, atol=1.0e-6)
    cfg.validate()


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    rounds = kwargs.pop("rounds", 1)
    cooldown_s = kwargs.pop("cooldown_s", 1.0)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA decode benchmark")

    from tvm.tirx.bench import bench

    executables = _compile_decode_kernels(**kwargs)
    # Allocate once outside both timed regions; both paths launch the exact
    # split-KV main kernel followed by their separate combine kernel.
    case = prepare_data(**kwargs)

    def tirx_decode():
        _launch_tirx(case, executables)

    from tirx_kernels.flashmla._flashmla_bench import flashmla_decode_reference_builder

    return bench(
        {"tirx": tirx_decode},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashmla": lambda: flashmla_decode_reference_builder(case)},
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "ModelType",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
