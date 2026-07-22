# bench-suite baseline view: `baseline.json`

- Timestamp: `6`
- Label:     `6d46ac95-dirty`
- Git:       `{'tir': 'd25621f5', 'tirx-kernels': 'aa97050f', 'tirx-bench-ci': None}`
- Workloads: 348 ok, 0 failed

Grouped workloads show one row per config and one timing column per implementation. Single-TIR workloads show ref/ours against the fastest reference implementation.

## allgather_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n106496_k16384_fp16_dynamic` | tirx | 23208.0323 | cublas_nccl_cudagraph | 20919.9279 | 0.901 | cublasmp_split_p2p=21170.6322 |
| `tp1_m8192_n24576_k4096_fp16_dynamic` | tirx | 1067.0560 | cublas_nccl_cudagraph | 1024.1600 | 0.960 | cublasmp_split_p2p=1069.9253 |
| `tp1_m8192_n51200_k5120_fp16_dynamic` | tirx | 2976.4373 | cublas_nccl_cudagraph | 2814.2933 | 0.946 | cublasmp_split_p2p=2843.0613 |
| `tp1_m8192_n57344_k8192_fp16_dynamic` | tirx | 6234.5093 | cublas_nccl_cudagraph | 5514.5386 | 0.885 | cublasmp_split_p2p=5614.6373 |
| `tp4_m8192_n106496_k16384_fp16_dynamic` | tirx | 5298.3654 | cublasmp_split_p2p | 5636.5760 | 1.064 | cublas_nccl_cudagraph=5697.3734 |
| `tp4_m8192_n24576_k4096_fp16_dynamic` | tirx | 301.6747 | cublasmp_split_p2p | 305.0827 | 1.011 | cublas_nccl_cudagraph=375.8640 |
| `tp4_m8192_n51200_k5120_fp16_dynamic` | tirx | 706.5627 | cublasmp_split_p2p | 723.6240 | 1.024 | cublas_nccl_cudagraph=809.2107 |
| `tp4_m8192_n57344_k8192_fp16_dynamic` | tirx | 1316.6880 | cublasmp_split_p2p | 1425.2160 | 1.082 | cublas_nccl_cudagraph=1477.8853 |

## deepgemm_fp8_fp4_mega_moe

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `t64_m64_h7168_i3072_e384_k6_g1` | tirx | 1297.0000 | deepgemm | 1289.0000 | 0.994 | — |
| `t64_m64_h7168_i3072_e384_k6_g2` | tirx | 913.1806 | deepgemm | 902.8422 | 0.989 | — |
| `t64_m64_h7168_i3072_e384_k6_g4` | tirx | 598.2578 | deepgemm | 597.9520 | 0.999 | — |
| `t64_m64_h7168_i3072_e384_k6_g6` | tirx | 486.6194 | deepgemm | 474.4618 | 0.975 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g1` | tirx | 3372.2000 | deepgemm | 3375.8000 | 1.001 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g2` | tirx | 3183.8000 | deepgemm | 3182.2000 | 0.999 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g4` | tirx | 2797.6000 | deepgemm | 2765.8000 | 0.989 | — |
| `t8192_m8192_h7168_i3072_e384_k6_g6` | tirx | 2917.6000 | deepgemm | 2908.6000 | 0.997 | — |

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 40.2385 | deepgemm | 40.9725 | 1.018 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 52.3165 | deepgemm | 53.0854 | 1.015 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 40.6919 | deepgemm | 41.6422 | 1.023 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 53.0757 | deepgemm | 54.5139 | 1.027 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 40.2405 | deepgemm | 42.5854 | 1.058 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 51.8639 | deepgemm | 55.1669 | 1.064 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 38.9814 | deepgemm | 39.4920 | 1.013 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 50.3269 | deepgemm | 51.0228 | 1.014 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 70.1864 | deepgemm | 71.3029 | 1.016 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 107.4568 | deepgemm | 108.7462 | 1.012 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 70.7711 | deepgemm | 72.8295 | 1.029 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 108.2805 | deepgemm | 111.7695 | 1.032 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 69.6395 | deepgemm | 73.9586 | 1.062 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 106.6060 | deepgemm | 113.3015 | 1.063 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 66.9366 | deepgemm | 67.9318 | 1.015 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 101.6421 | deepgemm | 103.0185 | 1.014 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 70.9981 | deepgemm | 72.2997 | 1.018 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 71.0213 | deepgemm | 72.1035 | 1.015 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 71.6594 | deepgemm | 73.9819 | 1.032 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 71.6432 | deepgemm | 73.8289 | 1.031 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 70.8131 | deepgemm | 75.3752 | 1.064 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 70.5174 | deepgemm | 75.2302 | 1.067 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 68.3745 | deepgemm | 69.2559 | 1.013 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 68.2719 | deepgemm | 69.0591 | 1.012 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 125.5773 | deepgemm | 127.1024 | 1.012 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 179.4415 | deepgemm | 181.4407 | 1.011 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 126.8607 | deepgemm | 130.8271 | 1.031 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 181.3469 | deepgemm | 187.0450 | 1.031 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 124.8101 | deepgemm | 132.7996 | 1.064 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 177.8604 | deepgemm | 189.6516 | 1.066 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 119.3647 | deepgemm | 120.5930 | 1.010 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 169.6329 | deepgemm | 171.9794 | 1.014 | — |

## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.3976 | deepgemm | 5.6910 | 1.054 | — |
| `b16_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.6593 | deepgemm | 6.0020 | 1.061 | — |
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.1801 | deepgemm | 6.5945 | 1.067 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.9560 | deepgemm | 6.2784 | 1.054 | — |
| `b16_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.6495 | deepgemm | 5.0203 | 1.080 | — |
| `b16_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.6288 | deepgemm | 4.9992 | 1.080 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.3144 | deepgemm | 4.6783 | 1.084 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2473 | deepgemm | 4.6124 | 1.086 | — |
| `b16_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.8438 | deepgemm | 5.0990 | 1.053 | — |
| `b16_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.9167 | deepgemm | 5.2128 | 1.060 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.0571 | deepgemm | 5.3206 | 1.052 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.9009 | deepgemm | 5.1585 | 1.053 | — |
| `b16_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.6279 | deepgemm | 5.1562 | 1.114 | — |
| `b16_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.6019 | deepgemm | 5.1059 | 1.110 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7770 | deepgemm | 5.0091 | 1.049 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.8335 | deepgemm | 5.0221 | 1.039 | — |
| `b1_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.9172 | deepgemm | 5.3757 | 1.093 | — |
| `b1_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.4654 | deepgemm | 4.7560 | 1.065 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.8721 | deepgemm | 5.1906 | 1.065 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.9199 | deepgemm | 5.2404 | 1.065 | — |
| `b1_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.2968 | deepgemm | 5.1121 | 1.190 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.3454 | deepgemm | 5.2067 | 1.198 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 3.9484 | deepgemm | 5.0093 | 1.269 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 3.9447 | deepgemm | 4.9977 | 1.267 | — |
| `b1_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.4328 | deepgemm | 4.6970 | 1.060 | — |
| `b1_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.9973 | deepgemm | 5.1910 | 1.039 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.0568 | deepgemm | 5.3614 | 1.060 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.0175 | deepgemm | 5.3502 | 1.066 | — |
| `b1_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.3858 | deepgemm | 4.8430 | 1.104 | — |
| `b1_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 5.1503 | deepgemm | 5.5255 | 1.073 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.9235 | deepgemm | 5.2771 | 1.072 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.3910 | deepgemm | 4.7691 | 1.086 | — |
| `b2_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.0258 | deepgemm | 5.5107 | 1.096 | — |
| `b2_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.9924 | deepgemm | 5.2177 | 1.045 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.6073 | deepgemm | 4.8355 | 1.050 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.7942 | deepgemm | 4.8737 | 1.017 | — |
| `b2_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.1498 | deepgemm | 4.9194 | 1.185 | — |
| `b2_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.1069 | deepgemm | 4.4728 | 1.089 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.0528 | deepgemm | 4.4514 | 1.098 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.4742 | deepgemm | 5.1302 | 1.147 | — |
| `b2_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.4200 | deepgemm | 4.6950 | 1.062 | — |
| `b2_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.6661 | deepgemm | 5.0713 | 1.087 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.8902 | deepgemm | 5.4914 | 1.123 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.8619 | deepgemm | 5.3682 | 1.104 | — |
| `b2_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 5.0870 | deepgemm | 5.2752 | 1.037 | — |
| `b2_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.3693 | deepgemm | 4.5359 | 1.038 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.4506 | deepgemm | 4.6076 | 1.035 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.3868 | deepgemm | 4.5849 | 1.045 | — |
| `b4_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.0407 | deepgemm | 5.2868 | 1.049 | — |
| `b4_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.8087 | deepgemm | 4.9798 | 1.036 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.0977 | deepgemm | 5.3377 | 1.047 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.1576 | deepgemm | 5.4369 | 1.054 | — |
| `b4_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.2792 | deepgemm | 4.6261 | 1.081 | — |
| `b4_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.2279 | deepgemm | 4.6132 | 1.091 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.1190 | deepgemm | 4.5761 | 1.111 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.1564 | deepgemm | 4.5672 | 1.099 | — |
| `b4_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.5402 | deepgemm | 4.9964 | 1.100 | — |
| `b4_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.6975 | deepgemm | 4.9127 | 1.046 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.7544 | deepgemm | 5.0160 | 1.055 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.7952 | deepgemm | 4.9713 | 1.037 | — |
| `b4_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.4999 | deepgemm | 4.9703 | 1.105 | — |
| `b4_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.6812 | deepgemm | 4.8570 | 1.038 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7430 | deepgemm | 4.8596 | 1.025 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5115 | deepgemm | 4.9065 | 1.088 | — |
| `b8_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.0649 | deepgemm | 5.4131 | 1.069 | — |
| `b8_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.1038 | deepgemm | 5.3742 | 1.053 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.3795 | deepgemm | 5.6060 | 1.042 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.2641 | deepgemm | 5.4959 | 1.044 | — |
| `b8_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.1036 | deepgemm | 5.0562 | 1.232 | — |
| `b8_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.1556 | deepgemm | 4.6489 | 1.119 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.1097 | deepgemm | 4.5368 | 1.104 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.3586 | deepgemm | 4.8413 | 1.111 | — |
| `b8_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.5115 | deepgemm | 4.8425 | 1.073 | — |
| `b8_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.5129 | deepgemm | 4.8143 | 1.067 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.9666 | deepgemm | 5.3532 | 1.078 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.9656 | deepgemm | 5.3167 | 1.071 | — |
| `b8_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.8876 | deepgemm | 5.1534 | 1.054 | — |
| `b8_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.5975 | deepgemm | 5.0781 | 1.105 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.5044 | deepgemm | 4.7633 | 1.057 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.4509 | deepgemm | 4.7941 | 1.077 | — |

## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 41.5074 | deepgemm | 41.9998 | 1.012 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 53.1855 | deepgemm | 54.1207 | 1.018 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 40.7478 | deepgemm | 41.4634 | 1.018 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 52.0202 | deepgemm | 53.4391 | 1.027 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 40.3472 | deepgemm | 43.2757 | 1.073 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 51.6074 | deepgemm | 56.2354 | 1.090 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 40.4071 | deepgemm | 41.1222 | 1.018 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 51.6047 | deepgemm | 52.8810 | 1.025 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 68.9178 | deepgemm | 71.0441 | 1.031 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 106.2875 | deepgemm | 110.1922 | 1.037 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 67.5034 | deepgemm | 70.3156 | 1.042 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 105.3824 | deepgemm | 109.9165 | 1.043 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 66.9574 | deepgemm | 74.6005 | 1.114 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 104.8258 | deepgemm | 116.2738 | 1.109 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 67.2050 | deepgemm | 70.0128 | 1.042 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 104.8944 | deepgemm | 109.8541 | 1.047 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 71.8578 | deepgemm | 72.5572 | 1.010 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 71.9938 | deepgemm | 73.1139 | 1.016 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 70.8448 | deepgemm | 72.0169 | 1.017 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 70.7505 | deepgemm | 72.2434 | 1.021 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 70.2124 | deepgemm | 76.4848 | 1.089 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 70.1355 | deepgemm | 76.4741 | 1.090 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 70.1991 | deepgemm | 71.6256 | 1.020 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 70.8750 | deepgemm | 71.5191 | 1.009 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 128.3840 | deepgemm | 130.4801 | 1.016 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 185.8410 | deepgemm | 191.5684 | 1.031 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 126.0460 | deepgemm | 130.2969 | 1.034 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 182.2652 | deepgemm | 189.5423 | 1.040 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 125.3978 | deepgemm | 135.4542 | 1.080 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 183.4872 | deepgemm | 199.3847 | 1.087 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 124.9909 | deepgemm | 128.8084 | 1.031 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 184.0205 | deepgemm | 190.8961 | 1.037 | — |

## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.7374 | sglang_cutedsl | 6.7346 | 1.000 | deepgemm=6.9053 |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 6.8487 | sglang_cutedsl | 6.9363 | 1.013 | deepgemm=7.0919 |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.7145 | sglang_cutedsl | 4.8158 | 1.021 | deepgemm=5.0307 |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.3188 | deepgemm | 4.6230 | 1.070 | sglang_cutedsl=4.7201 |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.1971 | sglang_cutedsl | 5.3544 | 1.030 | deepgemm=5.4982 |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.3850 | sglang_cutedsl | 5.5222 | 1.025 | deepgemm=5.7048 |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.8658 | sglang_cutedsl | 4.9276 | 1.013 | deepgemm=5.2607 |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5744 | deepgemm | 4.8572 | 1.062 | sglang_cutedsl=5.0839 |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.8180 | sglang_cutedsl | 4.8033 | 0.997 | deepgemm=5.1168 |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.7503 | sglang_cutedsl | 4.7839 | 1.007 | deepgemm=5.0332 |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.4499 | sglang_cutedsl | 4.6743 | 1.050 | deepgemm=4.9277 |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.3090 | sglang_cutedsl | 4.4214 | 1.026 | deepgemm=4.9619 |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.4905 | sglang_cutedsl | 4.5994 | 1.024 | deepgemm=4.7014 |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.7353 | sglang_cutedsl | 4.6678 | 0.986 | deepgemm=5.0525 |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.8163 | sglang_cutedsl | 4.8467 | 1.006 | deepgemm=4.9921 |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.7519 | sglang_cutedsl | 4.8209 | 1.015 | deepgemm=4.9231 |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.9753 | sglang_cutedsl | 4.9756 | 1.000 | deepgemm=5.2809 |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.8251 | sglang_cutedsl | 4.8341 | 1.002 | deepgemm=5.0744 |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.6030 | sglang_cutedsl | 4.6618 | 1.013 | deepgemm=4.8804 |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.4796 | sglang_cutedsl | 4.7608 | 1.063 | deepgemm=4.7621 |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.7179 | sglang_cutedsl | 4.8663 | 1.031 | deepgemm=4.9262 |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.7880 | sglang_cutedsl | 4.7261 | 0.987 | deepgemm=4.9676 |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7506 | sglang_cutedsl | 4.8620 | 1.023 | deepgemm=4.9958 |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5038 | sglang_cutedsl | 4.6420 | 1.031 | deepgemm=4.6717 |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.1676 | sglang_cutedsl | 5.2940 | 1.024 | deepgemm=5.4410 |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.1434 | sglang_cutedsl | 5.1822 | 1.008 | deepgemm=5.4959 |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.6802 | sglang_cutedsl | 4.7739 | 1.020 | deepgemm=4.8308 |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2815 | deepgemm | 4.5151 | 1.055 | sglang_cutedsl=4.5460 |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.8812 | sglang_cutedsl | 4.8589 | 0.995 | deepgemm=5.2924 |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.5238 | deepgemm | 4.8054 | 1.062 | sglang_cutedsl=4.9230 |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.4447 | sglang_cutedsl | 4.5512 | 1.024 | deepgemm=4.6572 |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.4655 | sglang_cutedsl | 4.6265 | 1.036 | deepgemm=4.6648 |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.7556 | sglang_cutedsl | 5.9087 | 1.027 | deepgemm=6.0022 |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 6.2715 | sglang_cutedsl | 6.4577 | 1.030 | deepgemm=6.7115 |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.2628 | deepgemm | 4.5847 | 1.076 | sglang_cutedsl=4.6197 |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.7470 | sglang_cutedsl | 4.7773 | 1.006 | deepgemm=5.3349 |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.0783 | sglang_cutedsl | 5.2281 | 1.030 | deepgemm=5.5004 |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.0988 | sglang_cutedsl | 5.1999 | 1.020 | deepgemm=5.4408 |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.5682 | deepgemm | 4.7890 | 1.048 | sglang_cutedsl=4.8481 |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.7802 | sglang_cutedsl | 4.8574 | 1.016 | deepgemm=5.3739 |

## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m1024_n24_k16384_s9` | tirx | 11.1936 | deepgemm | 11.7035 | 1.046 | — |
| `m1024_n24_k28672_s9` | tirx | 15.5834 | deepgemm | 17.2816 | 1.109 | — |
| `m1152_n24_k16384_s8` | tirx | 11.8335 | deepgemm | 12.5681 | 1.062 | — |
| `m1152_n24_k28672_s8` | tirx | 16.5378 | deepgemm | 18.6946 | 1.130 | — |
| `m128_n24_k16384_s64` | tirx | 5.2159 | deepgemm | 5.2130 | 0.999 | — |
| `m128_n24_k28672_s74` | tirx | 6.1043 | deepgemm | 6.1058 | 1.000 | — |
| `m1344_n24_k16384_s7` | tirx | 13.0774 | deepgemm | 14.0627 | 1.075 | — |
| `m1344_n24_k28672_s7` | tirx | 18.6690 | deepgemm | 21.0603 | 1.128 | — |
| `m137_n24_k7680_s16` | tirx | 5.5236 | deepgemm | 5.5340 | 1.002 | — |
| `m13_n24_k7168_s1` | tirx | 20.8120 | deepgemm | 21.2523 | 1.021 | — |
| `m1536_n24_k16384_s6` | tirx | 14.2275 | deepgemm | 15.6021 | 1.097 | — |
| `m1536_n24_k28672_s6` | tirx | 20.2784 | deepgemm | 23.3684 | 1.152 | — |
| `m1856_n24_k16384_s5` | tirx | 15.8785 | deepgemm | 17.5964 | 1.108 | — |
| `m1856_n24_k28672_s5` | tirx | 23.7925 | deepgemm | 27.6106 | 1.160 | — |
| `m192_n24_k16384_s49` | tirx | 5.9263 | deepgemm | 5.9042 | 0.996 | — |
| `m192_n24_k28672_s49` | tirx | 7.2646 | deepgemm | 7.1676 | 0.987 | — |
| `m2048_n24_k16384_s4` | tirx | 17.0283 | deepgemm | 19.1798 | 1.126 | — |
| `m2048_n24_k28672_s4` | tirx | 25.4788 | deepgemm | 30.0829 | 1.181 | — |
| `m2368_n24_k16384_s4` | tirx | 18.7809 | deepgemm | 21.0921 | 1.123 | — |
| `m2368_n24_k28672_s4` | tirx | 27.9436 | deepgemm | 33.1991 | 1.188 | — |
| `m256_n24_k16384_s37` | tirx | 6.3434 | deepgemm | 6.2874 | 0.991 | — |
| `m256_n24_k28672_s37` | tirx | 8.0874 | deepgemm | 8.0091 | 0.990 | — |
| `m3136_n24_k16384_s3` | tirx | 22.3079 | deepgemm | 26.0216 | 1.166 | — |
| `m3136_n24_k28672_s3` | tirx | 35.1164 | deepgemm | 41.9163 | 1.194 | — |
| `m320_n24_k16384_s29` | tirx | 6.9334 | deepgemm | 6.8546 | 0.989 | — |
| `m320_n24_k28672_s29` | tirx | 8.9452 | deepgemm | 8.9655 | 1.002 | — |
| `m384_n24_k16384_s24` | tirx | 7.5023 | deepgemm | 7.4210 | 0.989 | — |
| `m384_n24_k28672_s24` | tirx | 9.3787 | deepgemm | 9.4723 | 1.010 | — |
| `m4096_n24_k16384_s2` | tirx | 29.4388 | deepgemm | 34.7980 | 1.182 | — |
| `m4096_n24_k28672_s16` | tirx | 57.3877 | deepgemm | 62.7546 | 1.094 | — |
| `m4096_n24_k28672_s2` | tirx | 45.5110 | deepgemm | 54.7713 | 1.203 | — |
| `m4096_n24_k7168_s1` | tirx | 22.5396 | deepgemm | 23.7073 | 1.052 | — |
| `m448_n24_k16384_s21` | tirx | 7.8129 | deepgemm | 7.7525 | 0.992 | — |
| `m448_n24_k28672_s21` | tirx | 10.2691 | deepgemm | 10.4447 | 1.017 | — |
| `m4736_n24_k16384_s2` | tirx | 32.6268 | deepgemm | 38.1094 | 1.168 | — |
| `m4736_n24_k28672_s2` | tirx | 50.1462 | deepgemm | 59.9067 | 1.195 | — |
| `m512_n24_k16384_s18` | tirx | 8.4608 | deepgemm | 8.3818 | 0.991 | — |
| `m512_n24_k28672_s18` | tirx | 10.5725 | deepgemm | 10.8598 | 1.027 | — |
| `m576_n24_k16384_s16` | tirx | 8.9757 | deepgemm | 8.9690 | 0.999 | — |
| `m576_n24_k28672_s16` | tirx | 11.2886 | deepgemm | 11.7030 | 1.037 | — |
| `m640_n24_k16384_s14` | tirx | 9.0559 | deepgemm | 9.0984 | 1.005 | — |
| `m640_n24_k28672_s14` | tirx | 11.7549 | deepgemm | 12.4882 | 1.062 | — |
| `m64_n24_k28672_s112` | tirx | 5.1248 | deepgemm | 5.1336 | 1.002 | — |
| `m704_n24_k16384_s13` | tirx | 9.5614 | deepgemm | 9.7263 | 1.017 | — |
| `m704_n24_k28672_s13` | tirx | 12.5522 | deepgemm | 13.4431 | 1.071 | — |
| `m768_n24_k16384_s12` | tirx | 10.2364 | deepgemm | 10.3852 | 1.015 | — |
| `m768_n24_k28672_s12` | tirx | 13.3162 | deepgemm | 14.4032 | 1.082 | — |
| `m8192_n24_k16384_s1` | tirx | 51.3634 | deepgemm | 60.3445 | 1.175 | — |
| `m8192_n24_k28672_s1` | tirx | 84.0846 | deepgemm | 92.3930 | 1.099 | — |
| `m832_n24_k16384_s11` | tirx | 10.2505 | deepgemm | 10.4482 | 1.019 | — |
| `m832_n24_k28672_s11` | tirx | 13.7321 | deepgemm | 15.0211 | 1.094 | — |
| `m896_n24_k16384_s10` | tirx | 10.6917 | deepgemm | 10.9667 | 1.026 | — |
| `m896_n24_k28672_s10` | tirx | 14.4350 | deepgemm | 15.8178 | 1.096 | — |

## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.8367 | flashattn_sm100 | 20.0614 | 1.011 | — |
| `s1024_h32kv16_causal` | tir | 20.3834 | flashattn_sm100 | 20.1342 | 0.988 | — |
| `s1024_h32kv32` | tir | 20.3717 | flashattn_sm100 | 20.4074 | 1.002 | — |
| `s1024_h32kv32_causal` | tir | 21.0437 | flashattn_sm100 | 21.2752 | 1.011 | — |
| `s1024_h32kv4` | tir | 19.2491 | flashattn_sm100 | 19.7954 | 1.028 | — |
| `s1024_h32kv4_causal` | tir | 19.3650 | flashattn_sm100 | 19.7143 | 1.018 | — |
| `s1024_h32kv8` | tir | 19.6570 | flashattn_sm100 | 19.9340 | 1.014 | — |
| `s1024_h32kv8_causal` | tir | 19.7342 | flashattn_sm100 | 19.8446 | 1.006 | — |
| `s2048_h32kv16` | tir | 57.5522 | flashattn_sm100 | 58.0016 | 1.008 | — |
| `s2048_h32kv16_causal` | tir | 36.3314 | flashattn_sm100 | 38.6439 | 1.064 | — |
| `s2048_h32kv32` | tir | 59.1279 | flashattn_sm100 | 59.6590 | 1.009 | — |
| `s2048_h32kv32_causal` | tir | 40.3784 | flashattn_sm100 | 40.1752 | 0.995 | — |
| `s2048_h32kv4` | tir | 55.4535 | flashattn_sm100 | 56.4356 | 1.018 | — |
| `s2048_h32kv4_causal` | tir | 35.0917 | flashattn_sm100 | 37.9437 | 1.081 | — |
| `s2048_h32kv8` | tir | 56.2735 | flashattn_sm100 | 56.8710 | 1.011 | — |
| `s2048_h32kv8_causal` | tir | 35.0869 | flashattn_sm100 | 37.8508 | 1.079 | — |
| `s4096_h32kv16` | tir | 213.1821 | flashattn_sm100 | 216.0909 | 1.014 | — |
| `s4096_h32kv16_causal` | tir | 111.9918 | flashattn_sm100 | 117.8540 | 1.052 | — |
| `s4096_h32kv32` | tir | 216.2620 | flashattn_sm100 | 219.5906 | 1.015 | — |
| `s4096_h32kv32_causal` | tir | 122.4922 | flashattn_sm100 | 121.2001 | 0.989 | — |
| `s4096_h32kv4` | tir | 205.8206 | flashattn_sm100 | 210.3724 | 1.022 | — |
| `s4096_h32kv4_causal` | tir | 109.9380 | flashattn_sm100 | 114.6183 | 1.043 | — |
| `s4096_h32kv8` | tir | 207.8017 | flashattn_sm100 | 210.4235 | 1.013 | — |
| `s4096_h32kv8_causal` | tir | 109.5520 | flashattn_sm100 | 115.4890 | 1.054 | — |
| `s8192_h32kv16` | tir | 773.8496 | flashattn_sm100 | 773.7255 | 1.000 | — |
| `s8192_h32kv16_causal` | tir | 464.3977 | flashattn_sm100 | 421.8541 | 0.908 | — |
| `s8192_h32kv32` | tir | 779.9157 | flashattn_sm100 | 783.6445 | 1.005 | — |
| `s8192_h32kv32_causal` | tir | 437.9537 | flashattn_sm100 | 432.4135 | 0.987 | — |
| `s8192_h32kv4` | tir | 762.3605 | flashattn_sm100 | 775.7643 | 1.018 | — |
| `s8192_h32kv4_causal` | tir | 412.1217 | flashattn_sm100 | 422.7869 | 1.026 | — |
| `s8192_h32kv8` | tir | 765.2638 | flashattn_sm100 | 779.5500 | 1.019 | — |
| `s8192_h32kv8_causal` | tir | 410.5884 | flashattn_sm100 | 421.3691 | 1.026 | — |

## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.8299 | torch-cublas | 5.9781 | 0.875 | deepgemm-bf16=6.7710, deepgemm-cublaslt=5.9788 |
| `bf16_16384x16384x16384` | tir | 5515.2305 | torch-cublas | 5578.0705 | 1.011 | deepgemm-bf16=6445.8065, deepgemm-cublaslt=5589.9630 |
| `bf16_2048x2048x2048` | tir | 16.3837 | deepgemm-cublaslt | 15.6981 | 0.958 | deepgemm-bf16=17.2570, torch-cublas=15.7171 |
| `bf16_4096x4096x4096` | tir | 93.4063 | torch-cublas | 89.1807 | 0.955 | deepgemm-bf16=89.5884, deepgemm-cublaslt=89.4314 |
| `bf16_8192x8192x8192` | tir | 678.5797 | deepgemm-bf16 | 694.3811 | 1.023 | deepgemm-cublaslt=708.8169, torch-cublas=705.4324 |
| `fp16_1024x1024x1024` | tir | 6.9783 | torch-cublas | 6.1004 | 0.874 | — |
| `fp16_16384x16384x16384` | tir | 5922.6479 | torch-cublas | 5831.1985 | 0.985 | — |
| `fp16_2048x2048x2048` | tir | 16.4003 | torch-cublas | 16.0398 | 0.978 | — |
| `fp16_4096x4096x4096` | tir | 94.2328 | torch-cublas | 91.2280 | 0.968 | — |
| `fp16_8192x8192x8192` | tir | 723.6137 | torch-cublas | 756.1902 | 1.045 | — |

## fp8_blockwise_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 50.3229 | deepgemm | 49.1405 | 0.977 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 116.0378 | deepgemm | 113.8894 | 0.981 | — |
| `deepgemm_m4096_n32768_k512` | tir | 68.3255 | deepgemm | 70.7560 | 1.036 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 82.4525 | deepgemm | 81.7692 | 0.992 | — |
| `deepgemm_m4096_n576_k7168` | tir | 20.0249 | deepgemm | 19.0833 | 0.953 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 337.1709 | deepgemm | 337.8438 | 1.002 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 43.2023 | deepgemm | 42.3016 | 0.979 | — |

## gemm_reduce_scatter

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `tp1_m8192_n16384_k53248_fp16_dynamic` | tirx | 10815.1573 | cublas_nccl_cudagraph | 10489.3521 | 0.970 | cublasmp_split_p2p=11427.4106 |
| `tp1_m8192_n4096_k12288_fp16_dynamic` | tirx | 593.3013 | cublas_nccl_cudagraph | 563.6560 | 0.950 | cublasmp_split_p2p=753.8853 |
| `tp1_m8192_n5120_k25600_fp16_dynamic` | tirx | 1526.6960 | cublas_nccl_cudagraph | 1459.0134 | 0.956 | cublasmp_split_p2p=1868.9120 |
| `tp1_m8192_n8192_k28672_fp16_dynamic` | tirx | 2482.2586 | cublas_nccl_cudagraph | 2450.5147 | 0.987 | cublasmp_split_p2p=2930.0027 |
| `tp4_m8192_n16384_k53248_fp16_dynamic` | tirx | 2604.8026 | cublasmp_split_p2p | 2743.7493 | 1.053 | cublas_nccl_cudagraph=2775.7653 |
| `tp4_m8192_n4096_k12288_fp16_dynamic` | tirx | 262.0027 | cublasmp_split_p2p | 249.3147 | 0.952 | cublas_nccl_cudagraph=252.5120 |
| `tp4_m8192_n5120_k25600_fp16_dynamic` | tirx | 478.0347 | cublasmp_split_p2p | 431.3307 | 0.902 | cublas_nccl_cudagraph=484.1093 |
| `tp4_m8192_n8192_k28672_fp16_dynamic` | tirx | 756.1147 | cublasmp_split_p2p | 783.8987 | 1.037 | cublas_nccl_cudagraph=840.0533 |

## grouped_fp8_gemm_contiguous

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `large_g4_m8192_n4096_k2048` | tir | 163.3434 | deepgemm | 169.8797 | 1.040 | — |
| `large_g4_m8192_n4096_k4096` | tir | 366.7465 | deepgemm | 371.5478 | 1.013 | — |
| `large_g4_m8192_n6144_k7168` | tir | 979.9913 | deepgemm | 986.3398 | 1.006 | — |
| `large_g4_m8192_n7168_k3072` | tir | 490.2328 | deepgemm | 506.3523 | 1.033 | — |
| `large_g8_m4096_n4096_k2048` | tir | 182.7027 | deepgemm | 192.1728 | 1.052 | — |
| `large_g8_m4096_n4096_k4096` | tir | 358.7952 | deepgemm | 360.0171 | 1.003 | — |
| `large_g8_m4096_n6144_k7168` | tir | 1092.7013 | deepgemm | 1144.8009 | 1.048 | — |
| `large_g8_m4096_n7168_k3072` | tir | 504.2643 | deepgemm | 536.9235 | 1.065 | — |

## megakernel_moe

| config | tir_static (µs) | tir_dynamic (µs) | tir_unfused (µs) | sglang_full (µs) | flashinfer_full (µs) |
|---|---:|---:|---:|---:|---:|
| `moe_a3b_bs1_all` | 34.3236 | 38.4509 | 34.3723 | 57.6844 | 64.9148 |
| `moe_a3b_bs8_all` | 101.7217 | 102.6952 | 110.0313 | 134.1214 | 146.5452 |
| `moe_a3b_bs32_all` | 205.4530 | 203.6978 | 207.2780 | 242.9599 | 244.0950 |
| `moe_a3b_bs128_all` | 222.6645 | 219.6306 | 229.0844 | 257.2693 | 260.5438 |
| `moe_a3b_bs512_all` | 235.0351 | 234.0581 | 242.3089 | 307.2538 | 294.8196 |
| `moe_a3b_bs1024_all` | 253.5294 | 251.7546 | 269.4263 | 368.9248 | 338.6725 |
| `moe_a3b_bs2048_all` | 338.1688 | 337.7938 | 351.4161 | 455.5760 | 413.7839 |
| `moe_a3b_bs4096_all` | 519.9968 | 529.6217 | 536.9673 | 659.9393 | 597.4913 |

## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 5.4792 | flashinfer | 4.5388 | 0.828 | cublaslt_nvfp4=4.5630 |
| `16384x16384x16384` | tir | 1519.7452 | cublaslt_nvfp4 | 1417.6958 | 0.933 | flashinfer=1436.0074 |
| `2048x2048x2048` | tir | 8.4931 | cublaslt_nvfp4 | 7.6337 | 0.899 | flashinfer=7.7482 |
| `4096x4096x4096` | tir | 29.5781 | cublaslt_nvfp4 | 28.7877 | 0.973 | flashinfer=31.0869 |
| `8192x8192x8192` | tir | 184.8647 | flashinfer | 181.1199 | 0.980 | cublaslt_nvfp4=182.3157 |

## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1721.0173 | flashmla | 1735.7986 | 1.009 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1862.4760 | flashmla | 1894.8346 | 1.017 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1709.0216 | flashmla | 1734.5634 | 1.015 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1802.4899 | flashmla | 1830.1270 | 1.015 | trtllm_gen=2049.0881 |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 2023.8198 | flashmla | 2000.9210 | 0.989 | trtllm_gen=2199.8237 |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1791.3010 | flashmla | 1793.8845 | 1.001 | trtllm_gen=2030.4356 |

## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1148.8238 | flashmla | 1167.9828 | 1.017 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1179.5890 | flashmla | 1209.7339 | 1.026 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1150.4679 | flashmla | 1166.8383 | 1.014 | — |

## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 369.8020 | flashmla | 378.8894 | 1.025 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 373.0716 | flashmla | 381.9309 | 1.024 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 381.5666 | flashmla | 387.6533 | 1.016 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 363.3967 | flashmla | 373.3128 | 1.027 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 383.7200 | flashmla | 401.1419 | 1.045 | trtllm_gen=467.9234 |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 388.7866 | flashmla | 402.5865 | 1.035 | trtllm_gen=470.4102 |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 401.6096 | flashmla | 417.1230 | 1.039 | trtllm_gen=484.8768 |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 369.2718 | flashmla | 381.5986 | 1.033 | trtllm_gen=459.3726 |
