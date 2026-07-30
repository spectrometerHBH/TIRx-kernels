# bench-suite baseline view: `baseline.json`

- Timestamp: `77`
- Label:     `pr45-26631487-full-sgl044-torch213`
- Git:       `{'tir': 'fb56ab11-dirty', 'tirx-kernels': '26631487-dirty', 'tirx-bench-ci': None}`
- Workloads: 113 ok, 0 failed

Grouped workloads show one row per config and one timing column per implementation. Single-TIR workloads show ref/ours against the fastest reference implementation.

## allgather_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n106496_k16384_fp16_dynamic` | tirx | 20347.1755 | cublas_nccl_cudagraph | 17985.0723 | 0.884 | cublasmp_split_p2p=18204.1503 |
| `tp1_m8192_n24576_k4096_fp16_dynamic` | tirx | 1036.6108 | cublas_nccl_cudagraph | 1018.7519 | 0.983 | cublasmp_split_p2p=1063.6670 |
| `tp1_m8192_n51200_k5120_fp16_dynamic` | tirx | 2825.9458 | cublas_nccl_cudagraph | 2707.5847 | 0.958 | cublasmp_split_p2p=2763.2361 |
| `tp1_m8192_n57344_k8192_fp16_dynamic` | tirx | 5428.8352 | cublas_nccl_cudagraph | 4935.6822 | 0.909 | cublasmp_split_p2p=5015.3208 |

## deepgemm_fp8_fp4_mega_moe

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t64_m64_h7168_i3072_e384_k6_g1` | tirx | 1296.0000 | deepgemm | 1286.0000 | 0.992 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g1` | tirx | 3410.4000 | deepgemm | 3410.6000 | 1.000 | — |

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 42.1491 | deepgemm | 39.3123 | 0.933 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 195.9240 | deepgemm | 183.3436 | 0.936 | — |

## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.3787 | deepgemm | 6.5043 | 1.020 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.0699 | deepgemm | 4.6242 | 1.136 | — |

## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 39.0420 | deepgemm | 41.0418 | 1.051 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 192.7095 | deepgemm | 186.8664 | 0.970 | — |

## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.6487 | sglang_cutedsl | 6.8659 | 1.033 | deepgemm=6.9456 |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2997 | sglang_cutedsl | 4.5680 | 1.062 | deepgemm=4.9827 |

## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m128_n24_k16384_s64` | tirx | 5.2594 | deepgemm | 5.1812 | 0.985 | — |
| `m137_n24_k7680_s16` | tirx | 5.5257 | deepgemm | 5.4588 | 0.988 | — |
| `m13_n24_k7168_s1` | tirx | 21.4789 | deepgemm | 20.8115 | 0.969 | — |
| `m4096_n24_k28672_s16` | tirx | 57.9906 | deepgemm | 62.6236 | 1.080 | — |
| `m4096_n24_k7168_s1` | tirx | 23.4001 | deepgemm | 23.7544 | 1.015 | — |
| `m64_n24_k28672_s112` | tirx | 5.1281 | deepgemm | 5.1043 | 0.995 | — |
| `m8192_n24_k16384_s1` | tirx | 50.6801 | deepgemm | 59.4274 | 1.173 | — |
| `m8192_n24_k28672_s1` | tirx | 84.8203 | deepgemm | 92.8472 | 1.095 | — |

## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.8064 | flashattn_sm100 | 20.1674 | 1.018 | — |
| `s1024_h32kv16_causal` | tir | 20.4449 | flashattn_sm100 | 20.4744 | 1.001 | — |
| `s1024_h32kv32` | tir | 20.2122 | flashattn_sm100 | 20.6242 | 1.020 | — |
| `s1024_h32kv32_causal` | tir | 21.0007 | flashattn_sm100 | 21.3521 | 1.017 | — |
| `s1024_h32kv4` | tir | 19.2721 | flashattn_sm100 | 20.0094 | 1.038 | — |
| `s1024_h32kv4_causal` | tir | 18.7786 | flashattn_sm100 | 19.4542 | 1.036 | — |
| `s1024_h32kv8` | tir | 19.6263 | flashattn_sm100 | 20.1915 | 1.029 | — |
| `s1024_h32kv8_causal` | tir | 19.4163 | flashattn_sm100 | 19.8671 | 1.023 | — |
| `s2048_h32kv16` | tir | 57.4676 | flashattn_sm100 | 57.6315 | 1.003 | — |
| `s2048_h32kv16_causal` | tir | 36.5735 | flashattn_sm100 | 38.7478 | 1.059 | — |
| `s2048_h32kv32` | tir | 59.2354 | flashattn_sm100 | 59.7353 | 1.008 | — |
| `s2048_h32kv32_causal` | tir | 39.8546 | flashattn_sm100 | 40.5481 | 1.017 | — |
| `s2048_h32kv4` | tir | 55.7974 | flashattn_sm100 | 55.9215 | 1.002 | — |
| `s2048_h32kv4_causal` | tir | 35.3721 | flashattn_sm100 | 37.7387 | 1.067 | — |
| `s2048_h32kv8` | tir | 56.5389 | flashattn_sm100 | 56.7749 | 1.004 | — |
| `s2048_h32kv8_causal` | tir | 35.6759 | flashattn_sm100 | 38.1033 | 1.068 | — |
| `s4096_h32kv16` | tir | 211.5813 | flashattn_sm100 | 215.3945 | 1.018 | — |
| `s4096_h32kv16_causal` | tir | 112.4831 | flashattn_sm100 | 117.5787 | 1.045 | — |
| `s4096_h32kv32` | tir | 217.5533 | flashattn_sm100 | 218.2915 | 1.003 | — |
| `s4096_h32kv32_causal` | tir | 120.7237 | flashattn_sm100 | 121.6142 | 1.007 | — |
| `s4096_h32kv4` | tir | 207.2906 | flashattn_sm100 | 207.2475 | 1.000 | — |
| `s4096_h32kv4_causal` | tir | 109.8672 | flashattn_sm100 | 114.8975 | 1.046 | — |
| `s4096_h32kv8` | tir | 211.3173 | flashattn_sm100 | 211.5027 | 1.001 | — |
| `s4096_h32kv8_causal` | tir | 110.8674 | flashattn_sm100 | 116.5797 | 1.052 | — |
| `s8192_h32kv16` | tir | 775.8115 | flashattn_sm100 | 776.4188 | 1.001 | — |
| `s8192_h32kv16_causal` | tir | 421.6543 | flashattn_sm100 | 432.9106 | 1.027 | — |
| `s8192_h32kv32` | tir | 780.2330 | flashattn_sm100 | 779.3744 | 0.999 | — |
| `s8192_h32kv32_causal` | tir | 436.0053 | flashattn_sm100 | 434.5562 | 0.997 | — |
| `s8192_h32kv4` | tir | 783.2991 | flashattn_sm100 | 768.1835 | 0.981 | — |
| `s8192_h32kv4_causal` | tir | 415.9150 | flashattn_sm100 | 424.4983 | 1.021 | — |
| `s8192_h32kv8` | tir | 773.2016 | flashattn_sm100 | 774.0533 | 1.001 | — |
| `s8192_h32kv8_causal` | tir | 414.6564 | flashattn_sm100 | 421.4241 | 1.016 | — |

## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.1857 | deepgemm-cublaslt | 5.8463 | 0.945 | deepgemm-bf16=6.7069, torch-cublas=5.8557 |
| `bf16_16384x16384x16384` | tir | 5541.9210 | deepgemm-cublaslt | 5528.1810 | 0.998 | deepgemm-bf16=6625.7403, torch-cublas=5572.7584 |
| `bf16_2048x2048x2048` | tir | 15.8669 | deepgemm-cublaslt | 15.6621 | 0.987 | deepgemm-bf16=17.2149, torch-cublas=15.6718 |
| `bf16_4096x4096x4096` | tir | 89.7640 | torch-cublas | 90.3669 | 1.007 | deepgemm-bf16=90.8218, deepgemm-cublaslt=90.7227 |
| `bf16_8192x8192x8192` | tir | 700.8799 | torch-cublas | 722.3591 | 1.031 | deepgemm-bf16=739.7395, deepgemm-cublaslt=730.3039 |
| `fp16_1024x1024x1024` | tir | 6.2860 | torch-cublas | 5.8934 | 0.938 | — |
| `fp16_16384x16384x16384` | tir | 6011.6638 | torch-cublas | 5900.7829 | 0.982 | — |
| `fp16_2048x2048x2048` | tir | 16.1207 | torch-cublas | 15.7391 | 0.976 | — |
| `fp16_4096x4096x4096` | tir | 92.0312 | torch-cublas | 92.2030 | 1.002 | — |
| `fp16_8192x8192x8192` | tir | 750.1237 | torch-cublas | 759.7881 | 1.013 | — |

## fp8_blockwise_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 50.1107 | deepgemm | 49.1787 | 0.981 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 113.9351 | deepgemm | 113.8368 | 0.999 | — |
| `deepgemm_m4096_n32768_k512` | tir | 68.9345 | deepgemm | 72.0699 | 1.045 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 82.2197 | deepgemm | 81.8118 | 0.995 | — |
| `deepgemm_m4096_n576_k7168` | tir | 20.0859 | deepgemm | 19.0355 | 0.948 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 334.4252 | deepgemm | 325.6543 | 0.974 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 42.4870 | deepgemm | 42.1623 | 0.992 | — |

## gemm_reduce_scatter

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n16384_k53248_fp16_dynamic` | tirx | 9376.4720 | cublas_nccl_cudagraph | 9047.8146 | 0.965 | cublasmp_split_p2p=9854.3384 |
| `tp1_m8192_n4096_k12288_fp16_dynamic` | tirx | 589.7098 | cublas_nccl_cudagraph | 565.0988 | 0.958 | cublasmp_split_p2p=751.8051 |
| `tp1_m8192_n5120_k25600_fp16_dynamic` | tirx | 1533.3093 | cublas_nccl_cudagraph | 1464.3912 | 0.955 | cublasmp_split_p2p=1865.8576 |
| `tp1_m8192_n8192_k28672_fp16_dynamic` | tirx | 2480.6625 | cublas_nccl_cudagraph | 2451.4849 | 0.988 | cublasmp_split_p2p=2887.7046 |

## grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `large_g4_m8192_n4096_k2048` | tir | 161.5973 | deepgemm | 166.7829 | 1.032 | — |
| `large_g4_m8192_n4096_k4096` | tir | 362.2023 | deepgemm | 362.3270 | 1.000 | — |
| `large_g4_m8192_n6144_k7168` | tir | 1017.2346 | deepgemm | 1030.7333 | 1.013 | — |
| `large_g4_m8192_n7168_k3072` | tir | 497.7478 | deepgemm | 527.8830 | 1.061 | — |
| `large_g8_m4096_n4096_k2048` | tir | 188.6174 | deepgemm | 194.2249 | 1.030 | — |
| `large_g8_m4096_n4096_k4096` | tir | 355.1739 | deepgemm | 361.9487 | 1.019 | — |
| `large_g8_m4096_n6144_k7168` | tir | 1077.3380 | deepgemm | 1093.9870 | 1.015 | — |
| `large_g8_m4096_n7168_k3072` | tir | 506.5762 | deepgemm | 539.8993 | 1.066 | — |

## megakernel_moe

| config | tir_static (µs) | tir_dynamic (µs) | tir_unfused (µs) | sglang_full (µs) | flashinfer_full (µs) |
|---|---:|---:|---:|---:|---:|
| `moe_a3b_bs1_all` | 33.3472 | 37.2991 | 32.4838 | 52.7438 | 61.4531 |
| `moe_a3b_bs8_all` | 96.7169 | 100.6797 | 103.5047 | 129.6171 | 134.4403 |
| `moe_a3b_bs32_all` | 191.0123 | 192.7385 | 196.8859 | 229.2989 | 217.7729 |
| `moe_a3b_bs128_all` | 223.7852 | 220.0046 | 226.3990 | 244.6572 | 246.6854 |
| `moe_a3b_bs512_all` | 236.0292 | 230.9385 | 243.4052 | 287.4995 | 275.3515 |
| `moe_a3b_bs1024_all` | 257.9789 | 252.6093 | 273.5757 | 335.6316 | 314.2841 |
| `moe_a3b_bs2048_all` | 328.7713 | 330.3569 | 345.5856 | 431.7734 | 393.3928 |
| `moe_a3b_bs4096_all` | 533.4655 | 535.3281 | 546.4082 | 666.1293 | 604.3348 |

## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 4.6652 | cublaslt_nvfp4 | 4.5274 | 0.970 | flashinfer=4.5538 |
| `16384x16384x16384` | tir | 1447.9414 | cublaslt_nvfp4 | 1430.7288 | 0.988 | flashinfer=1439.7707 |
| `2048x2048x2048` | tir | 7.8846 | flashinfer | 7.4537 | 0.945 | cublaslt_nvfp4=7.7022 |
| `4096x4096x4096` | tir | 28.5639 | cublaslt_nvfp4 | 27.8961 | 0.977 | flashinfer=29.0789 |
| `8192x8192x8192` | tir | 183.2869 | flashinfer | 176.9837 | 0.966 | cublaslt_nvfp4=178.5887 |

## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1725.2833 | flashmla | 1761.1250 | 1.021 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1893.8073 | flashmla | 1897.4448 | 1.002 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1689.9636 | flashmla | 1704.5504 | 1.009 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1817.3885 | flashmla | 1832.8135 | 1.008 | trtllm_gen=2053.5608 |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 2003.1962 | flashmla | 1995.7244 | 0.996 | trtllm_gen=2182.7194 |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1823.7447 | flashmla | 1828.6868 | 1.003 | trtllm_gen=2056.8754 |

## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1153.7651 | flashmla | 1172.7877 | 1.016 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1206.1585 | flashmla | 1217.3493 | 1.009 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1153.2283 | flashmla | 1167.8016 | 1.013 | — |

## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 368.6787 | flashmla | 378.1364 | 1.026 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 370.8843 | flashmla | 382.8325 | 1.032 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 377.1894 | flashmla | 388.4326 | 1.030 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 364.0831 | flashmla | 372.9145 | 1.024 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 382.3995 | flashmla | 394.1750 | 1.031 | trtllm_gen=466.4023 |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 390.5793 | flashmla | 403.2429 | 1.032 | trtllm_gen=475.7821 |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 405.1353 | flashmla | 419.4723 | 1.035 | trtllm_gen=490.9278 |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 374.3321 | flashmla | 382.3630 | 1.021 | trtllm_gen=462.9439 |
