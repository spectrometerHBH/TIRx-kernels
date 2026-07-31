# bench-suite baseline view: `baseline.json`

- Timestamp: `18-nvfp4-8192-rechecked`
- Label:     `full-origin-dev-47456e5d-nvfp4-8192-rechecked`
- Git:       `{'tir': '0a348f2d', 'tirx-kernels': '47456e5d', 'tirx-bench-ci': None}`
- Workloads: 113 ok, 0 failed

Grouped workloads show one row per config and one timing column per implementation. Single-TIR workloads show ref/ours against the fastest reference implementation.

## allgather_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n106496_k16384_fp16_dynamic` | tirx | 20526.1473 | cublas_nccl_cudagraph | 18063.1157 | 0.880 | cublasmp_split_p2p=18291.2300 |
| `tp1_m8192_n24576_k4096_fp16_dynamic` | tirx | 1038.4812 | cublas_nccl_cudagraph | 1019.0285 | 0.981 | cublasmp_split_p2p=1095.2682 |
| `tp1_m8192_n51200_k5120_fp16_dynamic` | tirx | 2821.0918 | cublas_nccl_cudagraph | 2706.7880 | 0.959 | cublasmp_split_p2p=2782.3144 |
| `tp1_m8192_n57344_k8192_fp16_dynamic` | tirx | 5322.5258 | cublas_nccl_cudagraph | 4870.4921 | 0.915 | cublasmp_split_p2p=4960.7416 |

## deepgemm_fp8_fp4_mega_moe

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t64_m64_h7168_i3072_e384_k6_g1` | tirx | 1297.0000 | deepgemm | 1290.4000 | 0.995 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g1` | tirx | 3382.6000 | deepgemm | 3385.4000 | 1.001 | — |

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 37.3254 | deepgemm | 39.1732 | 1.050 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 181.4130 | deepgemm | 181.9449 | 1.003 | — |

## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.1025 | deepgemm | 6.4288 | 1.053 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.1568 | deepgemm | 5.1454 | 1.238 | — |

## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 39.3229 | deepgemm | 41.3615 | 1.052 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 178.8758 | deepgemm | 191.2600 | 1.069 | — |

## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.4949 | sglang_cutedsl | 6.7706 | 1.042 | deepgemm=6.9184 |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.3556 | sglang_cutedsl | 4.6155 | 1.060 | deepgemm=5.2630 |

## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m128_n24_k16384_s64` | tirx | 4.9256 | deepgemm | 5.1178 | 1.039 | — |
| `m137_n24_k7680_s16` | tirx | 5.1135 | deepgemm | 5.3407 | 1.044 | — |
| `m13_n24_k7168_s1` | tirx | 21.4308 | deepgemm | 20.8704 | 0.974 | — |
| `m4096_n24_k28672_s16` | tirx | 58.0437 | deepgemm | 62.7385 | 1.081 | — |
| `m4096_n24_k7168_s1` | tirx | 23.3416 | deepgemm | 23.5681 | 1.010 | — |
| `m64_n24_k28672_s112` | tirx | 4.9208 | deepgemm | 5.1150 | 1.039 | — |
| `m8192_n24_k16384_s1` | tirx | 50.7581 | deepgemm | 59.4272 | 1.171 | — |
| `m8192_n24_k28672_s1` | tirx | 83.2584 | deepgemm | 91.7941 | 1.103 | — |

## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.7662 | flashattn_sm100 | 20.2273 | 1.023 | — |
| `s1024_h32kv16_causal` | tir | 20.4522 | flashattn_sm100 | 20.5150 | 1.003 | — |
| `s1024_h32kv32` | tir | 20.3313 | flashattn_sm100 | 20.7544 | 1.021 | — |
| `s1024_h32kv32_causal` | tir | 20.8467 | flashattn_sm100 | 21.2008 | 1.017 | — |
| `s1024_h32kv4` | tir | 19.2377 | flashattn_sm100 | 19.7583 | 1.027 | — |
| `s1024_h32kv4_causal` | tir | 18.7694 | flashattn_sm100 | 19.6503 | 1.047 | — |
| `s1024_h32kv8` | tir | 19.4938 | flashattn_sm100 | 20.0183 | 1.027 | — |
| `s1024_h32kv8_causal` | tir | 19.0725 | flashattn_sm100 | 19.8677 | 1.042 | — |
| `s2048_h32kv16` | tir | 58.0888 | flashattn_sm100 | 58.3591 | 1.005 | — |
| `s2048_h32kv16_causal` | tir | 36.6109 | flashattn_sm100 | 38.7870 | 1.059 | — |
| `s2048_h32kv32` | tir | 59.8467 | flashattn_sm100 | 60.4474 | 1.010 | — |
| `s2048_h32kv32_causal` | tir | 39.8321 | flashattn_sm100 | 40.5273 | 1.017 | — |
| `s2048_h32kv4` | tir | 55.7150 | flashattn_sm100 | 56.0427 | 1.006 | — |
| `s2048_h32kv4_causal` | tir | 35.2954 | flashattn_sm100 | 37.8719 | 1.073 | — |
| `s2048_h32kv8` | tir | 57.0068 | flashattn_sm100 | 57.2701 | 1.005 | — |
| `s2048_h32kv8_causal` | tir | 36.2357 | flashattn_sm100 | 38.1621 | 1.053 | — |
| `s4096_h32kv16` | tir | 213.3766 | flashattn_sm100 | 216.2025 | 1.013 | — |
| `s4096_h32kv16_causal` | tir | 113.3341 | flashattn_sm100 | 119.2567 | 1.052 | — |
| `s4096_h32kv32` | tir | 214.0958 | flashattn_sm100 | 217.0692 | 1.014 | — |
| `s4096_h32kv32_causal` | tir | 118.3576 | flashattn_sm100 | 119.9245 | 1.013 | — |
| `s4096_h32kv4` | tir | 206.6479 | flashattn_sm100 | 207.7699 | 1.005 | — |
| `s4096_h32kv4_causal` | tir | 110.9520 | flashattn_sm100 | 116.0709 | 1.046 | — |
| `s4096_h32kv8` | tir | 207.3313 | flashattn_sm100 | 209.1818 | 1.009 | — |
| `s4096_h32kv8_causal` | tir | 110.6451 | flashattn_sm100 | 115.9943 | 1.048 | — |
| `s8192_h32kv16` | tir | 783.7915 | flashattn_sm100 | 777.1477 | 0.992 | — |
| `s8192_h32kv16_causal` | tir | 412.8636 | flashattn_sm100 | 422.8170 | 1.024 | — |
| `s8192_h32kv32` | tir | 782.9433 | flashattn_sm100 | 793.5758 | 1.014 | — |
| `s8192_h32kv32_causal` | tir | 433.1099 | flashattn_sm100 | 429.7134 | 0.992 | — |
| `s8192_h32kv4` | tir | 770.9662 | flashattn_sm100 | 771.5551 | 1.001 | — |
| `s8192_h32kv4_causal` | tir | 409.2400 | flashattn_sm100 | 419.7684 | 1.026 | — |
| `s8192_h32kv8` | tir | 774.5814 | flashattn_sm100 | 778.2181 | 1.005 | — |
| `s8192_h32kv8_causal` | tir | 410.8781 | flashattn_sm100 | 421.2932 | 1.025 | — |

## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.1799 | torch-cublas | 5.7809 | 0.935 | deepgemm-bf16=6.7465, deepgemm-cublaslt=5.8433 |
| `bf16_16384x16384x16384` | tir | 5524.0931 | torch-cublas | 5528.5352 | 1.001 | deepgemm-bf16=6279.4991, deepgemm-cublaslt=5536.4112 |
| `bf16_2048x2048x2048` | tir | 15.9030 | torch-cublas | 15.6107 | 0.982 | deepgemm-bf16=17.1687, deepgemm-cublaslt=15.6158 |
| `bf16_4096x4096x4096` | tir | 87.6913 | deepgemm-bf16 | 87.4284 | 0.997 | deepgemm-cublaslt=88.3761, torch-cublas=88.3697 |
| `bf16_8192x8192x8192` | tir | 691.7817 | deepgemm-cublaslt | 708.4753 | 1.024 | deepgemm-bf16=743.7710, torch-cublas=723.8874 |
| `fp16_1024x1024x1024` | tir | 6.2197 | torch-cublas | 5.7301 | 0.921 | — |
| `fp16_16384x16384x16384` | tir | 5629.1218 | torch-cublas | 5782.0455 | 1.027 | — |
| `fp16_2048x2048x2048` | tir | 16.1465 | torch-cublas | 15.6908 | 0.972 | — |
| `fp16_4096x4096x4096` | tir | 91.7258 | torch-cublas | 91.6638 | 0.999 | — |
| `fp16_8192x8192x8192` | tir | 739.4496 | torch-cublas | 753.0291 | 1.018 | — |

## fp8_blockwise_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 42.0010 | deepgemm | 49.2636 | 1.173 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 112.2873 | deepgemm | 113.2982 | 1.009 | — |
| `deepgemm_m4096_n32768_k512` | tir | 68.4478 | deepgemm | 71.3123 | 1.042 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 80.8552 | deepgemm | 79.8202 | 0.987 | — |
| `deepgemm_m4096_n576_k7168` | tir | 18.5506 | deepgemm | 19.0195 | 1.025 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 334.1922 | deepgemm | 330.8403 | 0.990 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 42.2477 | deepgemm | 42.2353 | 1.000 | — |

## gemm_reduce_scatter

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n16384_k53248_fp16_dynamic` | tirx | 9885.3501 | cublas_nccl_cudagraph | 9384.5946 | 0.949 | cublasmp_split_p2p=10196.1220 |
| `tp1_m8192_n4096_k12288_fp16_dynamic` | tirx | 594.7183 | cublas_nccl_cudagraph | 563.4575 | 0.947 | cublasmp_split_p2p=753.3096 |
| `tp1_m8192_n5120_k25600_fp16_dynamic` | tirx | 1515.9519 | cublas_nccl_cudagraph | 1446.0517 | 0.954 | cublasmp_split_p2p=1848.2512 |
| `tp1_m8192_n8192_k28672_fp16_dynamic` | tirx | 2496.0102 | cublas_nccl_cudagraph | 2450.7368 | 0.982 | cublasmp_split_p2p=2885.6015 |

## grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `large_g4_m8192_n4096_k2048` | tir | 163.7314 | deepgemm | 169.0523 | 1.032 | — |
| `large_g4_m8192_n4096_k4096` | tir | 369.1755 | deepgemm | 371.4189 | 1.006 | — |
| `large_g4_m8192_n6144_k7168` | tir | 968.0299 | deepgemm | 997.6826 | 1.031 | — |
| `large_g4_m8192_n7168_k3072` | tir | 490.2145 | deepgemm | 523.6072 | 1.068 | — |
| `large_g8_m4096_n4096_k2048` | tir | 191.7040 | deepgemm | 199.0590 | 1.038 | — |
| `large_g8_m4096_n4096_k4096` | tir | 358.1131 | deepgemm | 354.2541 | 0.989 | — |
| `large_g8_m4096_n6144_k7168` | tir | 1113.6396 | deepgemm | 1135.6100 | 1.020 | — |
| `large_g8_m4096_n7168_k3072` | tir | 514.1661 | deepgemm | 542.5178 | 1.055 | — |

## megakernel_moe

| config | tir_static (µs) | tir_dynamic (µs) | tir_unfused (µs) | sglang_full (µs) | flashinfer_full (µs) |
|---|---:|---:|---:|---:|---:|
| `moe_a3b_bs1_all` | 33.6205 | 37.2778 | 33.3702 | 54.8706 | 62.4532 |
| `moe_a3b_bs8_all` | 97.1334 | 100.8026 | 101.7439 | 125.9843 | 129.1746 |
| `moe_a3b_bs32_all` | 189.0376 | 192.3222 | 198.8576 | 223.9538 | 214.8292 |
| `moe_a3b_bs128_all` | 223.5500 | 219.7560 | 228.9188 | 242.7548 | 243.2492 |
| `moe_a3b_bs512_all` | 237.1914 | 229.3849 | 243.9618 | 288.0264 | 274.6339 |
| `moe_a3b_bs1024_all` | 257.6564 | 252.6185 | 273.0748 | 337.5818 | 310.2052 |
| `moe_a3b_bs2048_all` | 328.6593 | 329.3971 | 341.3998 | 431.7911 | 392.7344 |
| `moe_a3b_bs4096_all` | 529.2070 | 539.5501 | 543.8347 | 659.4472 | 602.7343 |

## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 4.6684 | cublaslt_nvfp4 | 4.4276 | 0.948 | flashinfer=4.4969 |
| `16384x16384x16384` | tir | 1466.5449 | flashinfer | 1404.0958 | 0.957 | cublaslt_nvfp4=1457.2260 |
| `2048x2048x2048` | tir | 7.8410 | flashinfer | 7.5764 | 0.966 | cublaslt_nvfp4=7.6110 |
| `4096x4096x4096` | tir | 28.8187 | cublaslt_nvfp4 | 27.7477 | 0.963 | flashinfer=29.4135 |
| `8192x8192x8192` | tir | 180.0340 | flashinfer | 174.9166 | 0.972 | cublaslt_nvfp4=179.3179 |

## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1710.0388 | flashmla | 1742.2043 | 1.019 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1875.1598 | flashmla | 1873.0610 | 0.999 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1682.6708 | flashmla | 1714.3549 | 1.019 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1807.1615 | flashmla | 1814.1524 | 1.004 | trtllm_gen=2045.3788 |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 2009.6952 | flashmla | 2004.4034 | 0.997 | trtllm_gen=2184.6203 |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1800.1651 | flashmla | 1805.6182 | 1.003 | trtllm_gen=2034.2320 |

## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1137.6732 | flashmla | 1156.2579 | 1.016 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1202.1687 | flashmla | 1212.6718 | 1.009 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1146.3653 | flashmla | 1158.6463 | 1.011 | — |

## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 371.9500 | flashmla | 381.8576 | 1.027 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 371.6401 | flashmla | 383.2295 | 1.031 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 377.8390 | flashmla | 388.8718 | 1.029 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 364.6633 | flashmla | 374.1866 | 1.026 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 384.3693 | flashmla | 400.7237 | 1.043 | trtllm_gen=478.6277 |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 390.1084 | flashmla | 408.2587 | 1.047 | trtllm_gen=472.8826 |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 403.0179 | flashmla | 418.8248 | 1.039 | trtllm_gen=491.6743 |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 373.7161 | flashmla | 382.2290 | 1.023 | trtllm_gen=464.4826 |
