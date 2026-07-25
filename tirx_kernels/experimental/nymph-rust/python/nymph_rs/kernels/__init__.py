"""Kernels described with the IRBuilder (build to Rust IR)."""

from .bootstrap_gemm import build_bootstrap_gemm
from .fp16_bf16_gemm import Fp16Bf16GemmConfig, build_fp16_bf16_gemm
from .nvfp4_gemm import CONFIGS as NVFP4_CONFIGS
from .nvfp4_gemm import NvFp4GemmConfig, build_nvfp4_gemm, nvfp4_task_config

# The remaining kernels (flash_attention4, flash_bwd_sm100, fp8_blockwise_gemm,
# gdn_prefill) still use the removed TmemLayout-era builder API — they are the
# Stage-3 migration work and intentionally not importable yet.
