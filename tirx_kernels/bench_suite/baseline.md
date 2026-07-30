# bench-suite baseline view: `baseline.json`

- Timestamp: `157`
- Label:     `pr45-final`
- Git:       `{'tir': 'fb56ab11-dirty', 'tirx-kernels': '47eb8943-dirty', 'tirx-bench-ci': None}`
- Workloads: 113 ok, 0 failed

Grouped workloads show one row per config and one timing column per implementation. Single-TIR workloads show ref/ours against the fastest reference implementation.

## allgather_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n106496_k16384_fp16_dynamic` | tirx | 19035.4892 | cublas_nccl_cudagraph | 18121.0029 | 0.952 | cublasmp_split_p2p=18366.2004 |
| `tp1_m8192_n24576_k4096_fp16_dynamic` | tirx | 1035.5806 | cublas_nccl_cudagraph | 1026.3873 | 0.991 | cublasmp_split_p2p=1072.1947 |
| `tp1_m8192_n51200_k5120_fp16_dynamic` | tirx | 2865.6280 | cublas_nccl_cudagraph | 2728.9279 | 0.952 | cublasmp_split_p2p=2784.6971 |
| `tp1_m8192_n57344_k8192_fp16_dynamic` | tirx | 5484.6730 | cublas_nccl_cudagraph | 4885.4983 | 0.891 | cublasmp_split_p2p=4974.1606 |

## deepgemm_fp8_fp4_mega_moe

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t64_m64_h7168_i3072_e384_k6_g1` | tirx | 1297.0000 | deepgemm | 1290.0000 | 0.995 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g1` | tirx | 3401.4000 | deepgemm | 3401.4000 | 1.000 | — |

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 37.4858 | deepgemm | 39.4054 | 1.051 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 182.1232 | deepgemm | 182.5651 | 1.002 | — |

## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.3685 | deepgemm | 6.5583 | 1.030 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.0776 | deepgemm | 4.9700 | 1.219 | — |

## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 38.9846 | deepgemm | 40.9416 | 1.050 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 181.1184 | deepgemm | 187.9073 | 1.037 | — |

## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.7307 | sglang_cutedsl | 6.9280 | 1.029 | deepgemm=6.9418 |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.5571 | sglang_cutedsl | 4.9052 | 1.076 | deepgemm=5.0422 |

## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m128_n24_k16384_s64` | tirx | 5.2831 | deepgemm | 5.1969 | 0.984 | — |
| `m137_n24_k7680_s16` | tirx | 5.4653 | deepgemm | 5.4258 | 0.993 | — |
| `m13_n24_k7168_s1` | tirx | 21.8838 | deepgemm | 21.3284 | 0.975 | — |
| `m4096_n24_k28672_s16` | tirx | 57.9550 | deepgemm | 62.7717 | 1.083 | — |
| `m4096_n24_k7168_s1` | tirx | 23.3083 | deepgemm | 23.5230 | 1.009 | — |
| `m64_n24_k28672_s112` | tirx | 5.2852 | deepgemm | 5.2854 | 1.000 | — |
| `m8192_n24_k16384_s1` | tirx | 50.7833 | deepgemm | 60.0844 | 1.183 | — |
| `m8192_n24_k28672_s1` | tirx | 83.7081 | deepgemm | 91.6873 | 1.095 | — |

## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.7023 | flashattn_sm100 | 20.2096 | 1.026 | — |
| `s1024_h32kv16_causal` | tir | 19.9980 | flashattn_sm100 | 20.2722 | 1.014 | — |
| `s1024_h32kv32` | tir | 20.5129 | flashattn_sm100 | 20.8115 | 1.015 | — |
| `s1024_h32kv32_causal` | tir | 21.0046 | flashattn_sm100 | 21.4597 | 1.022 | — |
| `s1024_h32kv4` | tir | 19.3340 | flashattn_sm100 | 19.9688 | 1.033 | — |
| `s1024_h32kv4_causal` | tir | 19.0127 | flashattn_sm100 | 19.8612 | 1.045 | — |
| `s1024_h32kv8` | tir | 19.4448 | flashattn_sm100 | 19.8315 | 1.020 | — |
| `s1024_h32kv8_causal` | tir | 19.4614 | flashattn_sm100 | 20.0129 | 1.028 | — |
| `s2048_h32kv16` | tir | 57.9021 | flashattn_sm100 | 57.8175 | 0.999 | — |
| `s2048_h32kv16_causal` | tir | 36.3622 | flashattn_sm100 | 38.6221 | 1.062 | — |
| `s2048_h32kv32` | tir | 59.4407 | flashattn_sm100 | 59.7029 | 1.004 | — |
| `s2048_h32kv32_causal` | tir | 39.9642 | flashattn_sm100 | 40.5845 | 1.016 | — |
| `s2048_h32kv4` | tir | 55.6885 | flashattn_sm100 | 55.8006 | 1.002 | — |
| `s2048_h32kv4_causal` | tir | 35.4333 | flashattn_sm100 | 38.0078 | 1.073 | — |
| `s2048_h32kv8` | tir | 56.8092 | flashattn_sm100 | 56.6679 | 0.998 | — |
| `s2048_h32kv8_causal` | tir | 35.5016 | flashattn_sm100 | 37.8814 | 1.067 | — |
| `s4096_h32kv16` | tir | 211.5064 | flashattn_sm100 | 213.3994 | 1.009 | — |
| `s4096_h32kv16_causal` | tir | 112.0787 | flashattn_sm100 | 118.3730 | 1.056 | — |
| `s4096_h32kv32` | tir | 218.1151 | flashattn_sm100 | 219.6977 | 1.007 | — |
| `s4096_h32kv32_causal` | tir | 119.3250 | flashattn_sm100 | 119.8426 | 1.004 | — |
| `s4096_h32kv4` | tir | 210.7095 | flashattn_sm100 | 209.0348 | 0.992 | — |
| `s4096_h32kv4_causal` | tir | 111.5006 | flashattn_sm100 | 115.6392 | 1.037 | — |
| `s4096_h32kv8` | tir | 210.0037 | flashattn_sm100 | 211.2078 | 1.006 | — |
| `s4096_h32kv8_causal` | tir | 110.9455 | flashattn_sm100 | 116.8375 | 1.053 | — |
| `s8192_h32kv16` | tir | 779.0893 | flashattn_sm100 | 776.9617 | 0.997 | — |
| `s8192_h32kv16_causal` | tir | 413.0516 | flashattn_sm100 | 426.6270 | 1.033 | — |
| `s8192_h32kv32` | tir | 783.1333 | flashattn_sm100 | 793.6872 | 1.013 | — |
| `s8192_h32kv32_causal` | tir | 434.0619 | flashattn_sm100 | 433.0868 | 0.998 | — |
| `s8192_h32kv4` | tir | 764.3664 | flashattn_sm100 | 768.0960 | 1.005 | — |
| `s8192_h32kv4_causal` | tir | 418.1305 | flashattn_sm100 | 424.5940 | 1.015 | — |
| `s8192_h32kv8` | tir | 771.5513 | flashattn_sm100 | 775.8581 | 1.006 | — |
| `s8192_h32kv8_causal` | tir | 411.1461 | flashattn_sm100 | 422.1195 | 1.027 | — |

## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.2070 | deepgemm-cublaslt | 5.8917 | 0.949 | deepgemm-bf16=6.7480, torch-cublas=5.8918 |
| `bf16_16384x16384x16384` | tir | 5452.0192 | torch-cublas | 5420.8317 | 0.994 | deepgemm-bf16=5812.4480, deepgemm-cublaslt=5439.5397 |
| `bf16_2048x2048x2048` | tir | 15.9003 | torch-cublas | 15.6392 | 0.984 | deepgemm-bf16=17.2324, deepgemm-cublaslt=15.6399 |
| `bf16_4096x4096x4096` | tir | 89.4518 | deepgemm-bf16 | 88.3152 | 0.987 | deepgemm-cublaslt=89.2192, torch-cublas=89.6141 |
| `bf16_8192x8192x8192` | tir | 684.4459 | deepgemm-cublaslt | 702.8859 | 1.027 | deepgemm-bf16=708.4234, torch-cublas=703.9724 |
| `fp16_1024x1024x1024` | tir | 6.2365 | torch-cublas | 5.9274 | 0.950 | — |
| `fp16_16384x16384x16384` | tir | 5643.1590 | torch-cublas | 5920.7061 | 1.049 | — |
| `fp16_2048x2048x2048` | tir | 16.1947 | torch-cublas | 15.6606 | 0.967 | — |
| `fp16_4096x4096x4096` | tir | 92.1835 | torch-cublas | 93.0357 | 1.009 | — |
| `fp16_8192x8192x8192` | tir | 720.7126 | torch-cublas | 727.5503 | 1.009 | — |

## fp8_blockwise_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 50.6205 | deepgemm | 49.6334 | 0.980 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 114.1395 | deepgemm | 113.9458 | 0.998 | — |
| `deepgemm_m4096_n32768_k512` | tir | 67.9121 | deepgemm | 71.8083 | 1.057 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 82.2233 | deepgemm | 81.4473 | 0.991 | — |
| `deepgemm_m4096_n576_k7168` | tir | 20.1080 | deepgemm | 19.0759 | 0.949 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 332.7540 | deepgemm | 336.6865 | 1.012 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 43.1168 | deepgemm | 42.3866 | 0.983 | — |

## gemm_reduce_scatter

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n16384_k53248_fp16_dynamic` | tirx | 9493.3588 | cublas_nccl_cudagraph | 9314.3908 | 0.981 | cublasmp_split_p2p=10107.8418 |
| `tp1_m8192_n4096_k12288_fp16_dynamic` | tirx | 595.6043 | cublas_nccl_cudagraph | 569.0537 | 0.955 | cublasmp_split_p2p=756.8610 |
| `tp1_m8192_n5120_k25600_fp16_dynamic` | tirx | 1496.5997 | cublas_nccl_cudagraph | 1433.0416 | 0.958 | cublasmp_split_p2p=1836.6789 |
| `tp1_m8192_n8192_k28672_fp16_dynamic` | tirx | 2501.2472 | cublas_nccl_cudagraph | 2461.8364 | 0.984 | cublasmp_split_p2p=2897.1493 |

## grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `large_g4_m8192_n4096_k2048` | tir | 161.9885 | deepgemm | 165.7490 | 1.023 | — |
| `large_g4_m8192_n4096_k4096` | tir | 364.0539 | deepgemm | 385.7832 | 1.060 | — |
| `large_g4_m8192_n6144_k7168` | tir | 1006.3684 | deepgemm | 1024.2809 | 1.018 | — |
| `large_g4_m8192_n7168_k3072` | tir | 503.5784 | deepgemm | 532.1796 | 1.057 | — |
| `large_g8_m4096_n4096_k2048` | tir | 195.0866 | deepgemm | 199.8450 | 1.024 | — |
| `large_g8_m4096_n4096_k4096` | tir | 361.5587 | deepgemm | 364.8858 | 1.009 | — |
| `large_g8_m4096_n6144_k7168` | tir | 1097.2625 | deepgemm | 1105.4624 | 1.007 | — |
| `large_g8_m4096_n7168_k3072` | tir | 523.3928 | deepgemm | 545.0444 | 1.041 | — |

## megakernel_moe

| config | tir_static (µs) | tir_dynamic (µs) | tir_unfused (µs) | sglang_full (µs) | flashinfer_full (µs) |
|---|---:|---:|---:|---:|---:|
| `moe_a3b_bs1_all` | 33.5470 | 37.2208 | 33.8733 | 55.4579 | 64.9824 |
| `moe_a3b_bs8_all` | 96.1016 | 100.8926 | 101.7110 | 125.9085 | 130.2768 |
| `moe_a3b_bs32_all` | 190.2761 | 192.7161 | 198.6271 | 227.4717 | 219.6533 |
| `moe_a3b_bs128_all` | 223.4229 | 219.3805 | 226.4836 | 245.1133 | 248.4980 |
| `moe_a3b_bs512_all` | 237.9336 | 230.3061 | 242.4455 | 287.7520 | 276.8443 |
| `moe_a3b_bs1024_all` | 256.1143 | 249.9446 | 270.0376 | 338.1065 | 315.2151 |
| `moe_a3b_bs2048_all` | 327.6949 | 329.4998 | 343.6046 | 430.7744 | 394.4439 |
| `moe_a3b_bs4096_all` | 531.8767 | 545.7722 | 548.6232 | 665.2427 | 602.7492 |

## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 4.6965 | flashinfer | 4.5039 | 0.959 | cublaslt_nvfp4=4.5309 |
| `16384x16384x16384` | tir | 1436.6070 | cublaslt_nvfp4 | 1440.5255 | 1.003 | flashinfer=1442.6932 |
| `2048x2048x2048` | tir | 7.9725 | flashinfer | 7.6929 | 0.965 | cublaslt_nvfp4=7.7367 |
| `4096x4096x4096` | tir | 28.8581 | cublaslt_nvfp4 | 27.9019 | 0.967 | flashinfer=28.9719 |
| `8192x8192x8192` | tir | 178.7241 | flashinfer | 175.0517 | 0.979 | cublaslt_nvfp4=177.0423 |

## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1745.6163 | flashmla | 1750.9986 | 1.003 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1892.9687 | flashmla | 1879.0310 | 0.993 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1701.6645 | flashmla | 1716.0421 | 1.008 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1803.8346 | flashmla | 1831.1934 | 1.015 | trtllm_gen=2061.5866 |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 2017.0968 | flashmla | 2004.5769 | 0.994 | trtllm_gen=2213.5642 |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1820.4899 | flashmla | 1834.8606 | 1.008 | trtllm_gen=2052.7717 |

## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1155.1400 | flashmla | 1161.9437 | 1.006 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1196.2900 | flashmla | 1207.7541 | 1.010 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1135.0424 | flashmla | 1153.7119 | 1.016 | — |

## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 366.3225 | flashmla | 379.0269 | 1.035 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 371.9328 | flashmla | 381.2608 | 1.025 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 376.3476 | flashmla | 386.8558 | 1.028 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 365.2739 | flashmla | 373.4348 | 1.022 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 382.8646 | flashmla | 396.9808 | 1.037 | trtllm_gen=467.1347 |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 389.0858 | flashmla | 404.9870 | 1.041 | trtllm_gen=473.4408 |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 402.5077 | flashmla | 417.4860 | 1.037 | trtllm_gen=481.9321 |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 374.5857 | flashmla | 383.2480 | 1.023 | trtllm_gen=462.7434 |
