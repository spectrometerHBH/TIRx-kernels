# bench-suite baseline view: `baseline.json`

- Timestamp: `2`
- Label:     `691feec4-dirty`
- Git:       `{'tir': 'e905776a', 'tirx-kernels': 'b0a0df4e-dirty', 'tirx-bench-ci': None}`
- Workloads: 128 ok, 0 failed

Grouped workloads show one row per config and one timing column per implementation. Single-TIR workloads show ref/ours against the fastest reference implementation.

## allgather_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n106496_k16384_fp16_dynamic` | tirx | 19295.3294 | cublas_nccl_cudagraph | 17885.3028 | 0.927 | cublasmp_split_p2p=18124.6783 |
| `tp1_m8192_n24576_k4096_fp16_dynamic` | tirx | 1050.5499 | cublas_nccl_cudagraph | 1013.5965 | 0.965 | cublasmp_split_p2p=1059.8718 |
| `tp1_m8192_n51200_k5120_fp16_dynamic` | tirx | 2872.7876 | cublas_nccl_cudagraph | 2674.3440 | 0.931 | cublasmp_split_p2p=2778.7565 |
| `tp1_m8192_n57344_k8192_fp16_dynamic` | tirx | 5463.5618 | cublas_nccl_cudagraph | 4846.3458 | 0.887 | cublasmp_split_p2p=4935.7216 |

## deepgemm_fp8_fp4_mega_moe

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t64_m64_h7168_i3072_e384_k6_g1` | tirx | 1297.2000 | deepgemm | 1289.2000 | 0.994 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g1` | tirx | 3467.0000 | deepgemm | 3468.0000 | 1.000 | — |

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 38.8288 | deepgemm | 39.0732 | 1.006 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 181.0958 | deepgemm | 182.3834 | 1.007 | — |

## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.1343 | deepgemm | 6.5764 | 1.072 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.2940 | deepgemm | 5.1011 | 1.188 | — |

## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 40.5116 | deepgemm | 41.0614 | 1.014 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 184.5412 | deepgemm | 189.0791 | 1.025 | — |

## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.7726 | sglang_cutedsl | 6.7549 | 0.997 | deepgemm=6.8962 |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.4133 | sglang_cutedsl | 4.4834 | 1.016 | deepgemm=5.0033 |

## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m128_n24_k16384_s64` | tirx | 5.2352 | deepgemm | 5.2289 | 0.999 | — |
| `m137_n24_k7680_s16` | tirx | 5.3441 | deepgemm | 5.3607 | 1.003 | — |
| `m13_n24_k7168_s1` | tirx | 20.8344 | deepgemm | 21.3015 | 1.022 | — |
| `m4096_n24_k28672_s16` | tirx | 57.4497 | deepgemm | 62.7967 | 1.093 | — |
| `m4096_n24_k7168_s1` | tirx | 22.4769 | deepgemm | 23.6052 | 1.050 | — |
| `m64_n24_k28672_s112` | tirx | 5.1296 | deepgemm | 5.1328 | 1.001 | — |
| `m8192_n24_k16384_s1` | tirx | 51.2118 | deepgemm | 60.2061 | 1.176 | — |
| `m8192_n24_k28672_s1` | tirx | 83.4529 | deepgemm | 91.7669 | 1.100 | — |

## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.8515 | flashattn_sm100 | 20.1252 | 1.014 | — |
| `s1024_h32kv16_causal` | tir | 20.3716 | flashattn_sm100 | 20.1331 | 0.988 | — |
| `s1024_h32kv32` | tir | 20.0932 | flashattn_sm100 | 20.2886 | 1.010 | — |
| `s1024_h32kv32_causal` | tir | 21.0061 | flashattn_sm100 | 21.2746 | 1.013 | — |
| `s1024_h32kv4` | tir | 19.1346 | flashattn_sm100 | 19.6906 | 1.029 | — |
| `s1024_h32kv4_causal` | tir | 19.3387 | flashattn_sm100 | 19.6489 | 1.016 | — |
| `s1024_h32kv8` | tir | 19.6321 | flashattn_sm100 | 19.9280 | 1.015 | — |
| `s1024_h32kv8_causal` | tir | 19.8337 | flashattn_sm100 | 20.1010 | 1.013 | — |
| `s2048_h32kv16` | tir | 57.2230 | flashattn_sm100 | 57.7924 | 1.010 | — |
| `s2048_h32kv16_causal` | tir | 36.2025 | flashattn_sm100 | 38.4455 | 1.062 | — |
| `s2048_h32kv32` | tir | 59.2902 | flashattn_sm100 | 59.7581 | 1.008 | — |
| `s2048_h32kv32_causal` | tir | 40.5775 | flashattn_sm100 | 39.8887 | 0.983 | — |
| `s2048_h32kv4` | tir | 55.2675 | flashattn_sm100 | 56.0876 | 1.015 | — |
| `s2048_h32kv4_causal` | tir | 35.2268 | flashattn_sm100 | 37.8547 | 1.075 | — |
| `s2048_h32kv8` | tir | 56.3359 | flashattn_sm100 | 56.9675 | 1.011 | — |
| `s2048_h32kv8_causal` | tir | 35.7064 | flashattn_sm100 | 37.8037 | 1.059 | — |
| `s4096_h32kv16` | tir | 212.8696 | flashattn_sm100 | 216.0092 | 1.015 | — |
| `s4096_h32kv16_causal` | tir | 112.6235 | flashattn_sm100 | 117.9478 | 1.047 | — |
| `s4096_h32kv32` | tir | 213.7173 | flashattn_sm100 | 218.7990 | 1.024 | — |
| `s4096_h32kv32_causal` | tir | 122.6408 | flashattn_sm100 | 121.9455 | 0.994 | — |
| `s4096_h32kv4` | tir | 208.7711 | flashattn_sm100 | 210.9965 | 1.011 | — |
| `s4096_h32kv4_causal` | tir | 108.9606 | flashattn_sm100 | 114.2291 | 1.048 | — |
| `s4096_h32kv8` | tir | 208.1914 | flashattn_sm100 | 212.9024 | 1.023 | — |
| `s4096_h32kv8_causal` | tir | 110.6651 | flashattn_sm100 | 115.6321 | 1.045 | — |
| `s8192_h32kv16` | tir | 772.8333 | flashattn_sm100 | 783.9207 | 1.014 | — |
| `s8192_h32kv16_causal` | tir | 464.9653 | flashattn_sm100 | 424.9841 | 0.914 | — |
| `s8192_h32kv32` | tir | 783.2524 | flashattn_sm100 | 799.4478 | 1.021 | — |
| `s8192_h32kv32_causal` | tir | 433.3247 | flashattn_sm100 | 430.1334 | 0.993 | — |
| `s8192_h32kv4` | tir | 764.4909 | flashattn_sm100 | 777.3582 | 1.017 | — |
| `s8192_h32kv4_causal` | tir | 406.1900 | flashattn_sm100 | 420.6036 | 1.035 | — |
| `s8192_h32kv8` | tir | 763.0542 | flashattn_sm100 | 777.3898 | 1.019 | — |
| `s8192_h32kv8_causal` | tir | 407.7918 | flashattn_sm100 | 420.1893 | 1.030 | — |

## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.8394 | torch-cublas | 5.9838 | 0.875 | deepgemm-bf16=6.8101, deepgemm-cublaslt=5.9939 |
| `bf16_16384x16384x16384` | tir | 5670.4596 | torch-cublas | 5427.3315 | 0.957 | deepgemm-bf16=6417.9533, deepgemm-cublaslt=5468.8178 |
| `bf16_2048x2048x2048` | tir | 16.2651 | torch-cublas | 15.8004 | 0.971 | deepgemm-bf16=17.1367, deepgemm-cublaslt=15.8199 |
| `bf16_4096x4096x4096` | tir | 93.6135 | deepgemm-bf16 | 89.5862 | 0.957 | deepgemm-cublaslt=90.0239, torch-cublas=89.7647 |
| `bf16_8192x8192x8192` | tir | 695.8847 | torch-cublas | 716.2781 | 1.029 | deepgemm-bf16=723.1890, deepgemm-cublaslt=718.5874 |
| `fp16_1024x1024x1024` | tir | 6.9834 | torch-cublas | 6.0855 | 0.871 | — |
| `fp16_16384x16384x16384` | tir | 6021.2726 | torch-cublas | 5877.8464 | 0.976 | — |
| `fp16_2048x2048x2048` | tir | 16.5189 | torch-cublas | 16.1061 | 0.975 | — |
| `fp16_4096x4096x4096` | tir | 95.6050 | torch-cublas | 91.6990 | 0.959 | — |
| `fp16_8192x8192x8192` | tir | 732.7519 | torch-cublas | 747.5038 | 1.020 | — |

## fp8_blockwise_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 50.4722 | deepgemm | 49.4849 | 0.980 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 115.0606 | deepgemm | 114.6086 | 0.996 | — |
| `deepgemm_m4096_n32768_k512` | tir | 68.2228 | deepgemm | 71.6053 | 1.050 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 81.1150 | deepgemm | 80.9344 | 0.998 | — |
| `deepgemm_m4096_n576_k7168` | tir | 20.2570 | deepgemm | 19.0895 | 0.942 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 327.8762 | deepgemm | 327.0052 | 0.997 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 43.3793 | deepgemm | 42.5185 | 0.980 | — |

## gemm_reduce_scatter

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n16384_k53248_fp16_dynamic` | tirx | 9356.5753 | cublas_nccl_cudagraph | 9029.7421 | 0.965 | cublasmp_split_p2p=9836.3170 |
| `tp1_m8192_n4096_k12288_fp16_dynamic` | tirx | 593.4697 | cublas_nccl_cudagraph | 561.6201 | 0.946 | cublasmp_split_p2p=747.7858 |
| `tp1_m8192_n5120_k25600_fp16_dynamic` | tirx | 1515.5371 | cublas_nccl_cudagraph | 1442.0078 | 0.951 | cublasmp_split_p2p=1841.0078 |
| `tp1_m8192_n8192_k28672_fp16_dynamic` | tirx | 2523.2398 | cublas_nccl_cudagraph | 2467.1696 | 0.978 | cublasmp_split_p2p=2902.8862 |

## grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `large_g4_m8192_n4096_k2048` | tir | 161.9105 | deepgemm | 162.3524 | 1.003 | — |
| `large_g4_m8192_n4096_k4096` | tir | 348.7676 | deepgemm | 365.8182 | 1.049 | — |
| `large_g4_m8192_n6144_k7168` | tir | 989.9543 | deepgemm | 1030.5004 | 1.041 | — |
| `large_g4_m8192_n7168_k3072` | tir | 505.8246 | deepgemm | 526.2587 | 1.040 | — |
| `large_g8_m4096_n4096_k2048` | tir | 188.7791 | deepgemm | 198.0719 | 1.049 | — |
| `large_g8_m4096_n4096_k4096` | tir | 349.4309 | deepgemm | 347.1907 | 0.994 | — |
| `large_g8_m4096_n6144_k7168` | tir | 1129.2648 | deepgemm | 1148.2563 | 1.017 | — |
| `large_g8_m4096_n7168_k3072` | tir | 528.6297 | deepgemm | 545.5813 | 1.032 | — |

## megakernel_moe

| config | tir_static (µs) | tir_dynamic (µs) | tir_unfused (µs) | sglang_full (µs) | flashinfer_full (µs) |
|---|---:|---:|---:|---:|---:|
| `moe_a3b_bs1_all` | 33.6316 | 37.8042 | 34.8430 | 53.8078 | 60.5378 |
| `moe_a3b_bs8_all` | 103.3933 | 102.6205 | 111.0531 | 137.1187 | 142.5176 |
| `moe_a3b_bs32_all` | 203.2012 | 203.6943 | 211.0378 | 238.3332 | 238.9275 |
| `moe_a3b_bs128_all` | 223.3124 | 220.2085 | 231.1850 | 254.8006 | 260.8025 |
| `moe_a3b_bs512_all` | 234.7884 | 232.5720 | 248.7551 | 308.0618 | 293.7048 |
| `moe_a3b_bs1024_all` | 254.5459 | 250.6894 | 271.1115 | 366.6708 | 340.0000 |
| `moe_a3b_bs2048_all` | 336.8534 | 337.8143 | 349.7044 | 453.5882 | 407.2323 |
| `moe_a3b_bs4096_all` | 526.0154 | 534.9026 | 539.4193 | 663.5941 | 598.7854 |

## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 5.2129 | cublaslt_nvfp4 | 4.5248 | 0.868 | flashinfer=4.6234 |
| `16384x16384x16384` | tir | 1515.0944 | flashinfer | 1434.6245 | 0.947 | cublaslt_nvfp4=1470.8436 |
| `2048x2048x2048` | tir | 8.5714 | cublaslt_nvfp4 | 7.4747 | 0.872 | flashinfer=7.5224 |
| `4096x4096x4096` | tir | 29.6264 | cublaslt_nvfp4 | 28.6867 | 0.968 | flashinfer=30.7572 |
| `8192x8192x8192` | tir | 185.8869 | flashinfer | 177.4658 | 0.955 | cublaslt_nvfp4=183.9112 |

## sparse_flashmla_decode_head64

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepseek_v4_v32_b128_sq2_sk32768_topk2048_p64` | tirx | 138.8959 | flashmla | 143.6941 | 1.035 | — |
| `model1_b128_sq2_sk16384_topk128_p256_xsk16384_xtopk1024_xp2_xtopklen` | tirx | 70.4264 | flashmla | 73.6416 | 1.046 | — |
| `model1_b128_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | tirx | 58.1189 | flashmla | 58.5720 | 1.008 | — |
| `model1_b148_sq2_sk16384_topk128_p256_xsk16384_xtopk1024_xp2_xtopklen` | tirx | 75.9245 | flashmla | 79.2599 | 1.044 | — |
| `model1_b148_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | tirx | 60.1992 | flashmla | 60.4661 | 1.004 | — |
| `model1_b148_sq2_sk32768_topk16384_p64` | tirx | 904.0643 | flashmla | 907.7006 | 1.004 | — |
| `model1_b256_sq2_sk16384_topk128_p256_xsk16384_xtopk1024_xp2_xtopklen` | tirx | 111.6659 | flashmla | 115.0202 | 1.030 | — |
| `model1_b256_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | tirx | 105.7925 | flashmla | 108.6594 | 1.027 | — |
| `model1_b2_sq2_sk16384_topk128_p256_xsk16384_xtopk1024_xp2_xtopklen` | tirx | 18.9377 | flashmla | 22.9930 | 1.214 | — |
| `model1_b2_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | tirx | 16.9030 | flashmla | 21.1933 | 1.254 | — |
| `model1_b64_sq2_sk16384_topk128_p256_xsk16384_xtopk1024_xp2_xtopklen` | tirx | 43.4416 | flashmla | 47.3502 | 1.090 | — |
| `model1_b64_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | tirx | 31.7378 | flashmla | 32.4080 | 1.021 | — |
| `model1_b74_sq2_sk16384_topk128_p256_xsk16384_xtopk1024_xp2_xtopklen` | tirx | 49.6371 | flashmla | 53.0040 | 1.068 | — |
| `model1_b74_sq2_sk16384_topk128_p256_xsk16384_xtopk512_xp64` | tirx | 34.3509 | flashmla | 34.5908 | 1.007 | — |
| `v32_b148_sq2_sk32768_topk16384_p64` | tirx | 957.4198 | flashmla | 957.6634 | 1.000 | — |

## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1720.3043 | flashmla | 1742.0049 | 1.013 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1884.2059 | flashmla | 1893.9394 | 1.005 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1724.0276 | flashmla | 1726.3319 | 1.001 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1829.0657 | flashmla | 1845.1190 | 1.009 | trtllm_gen=2068.9038 |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 1999.1795 | flashmla | 1988.3791 | 0.995 | trtllm_gen=2165.1378 |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1804.8583 | flashmla | 1818.1843 | 1.007 | trtllm_gen=2051.3738 |

## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1158.2862 | flashmla | 1175.4128 | 1.015 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1204.5303 | flashmla | 1214.1591 | 1.008 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1146.6569 | flashmla | 1160.8605 | 1.012 | — |

## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 368.6979 | flashmla | 380.8613 | 1.033 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 375.8817 | flashmla | 386.3110 | 1.028 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 376.3356 | flashmla | 387.9096 | 1.031 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 362.8557 | flashmla | 371.8570 | 1.025 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 381.8869 | flashmla | 395.3827 | 1.035 | trtllm_gen=466.9376 |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 394.7216 | flashmla | 410.2901 | 1.039 | trtllm_gen=477.0997 |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 400.9523 | flashmla | 416.1780 | 1.038 | trtllm_gen=487.7229 |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 377.6830 | flashmla | 387.2590 | 1.025 | trtllm_gen=464.1193 |
