# bench-suite baseline view: `tir.json + ref.json`

- Timestamp: `121`
- Label:     `cb2a7638-dirty`
- Git:       `{'tir': '51d56098-dirty', 'tirx-kernels': 'cb2a7638-dirty', 'tirx-bench-ci': None}`
- Workloads: 259 ok, 0 failed

Each row shows our impl's time (tir/tirx) and every reference impl, with ref/ours where ref = fastest non-ours impl. Higher ratio = ours is faster.

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 42.4598 | deepgemm | 41.8267 | 0.985 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 55.6399 | deepgemm | 54.4135 | 0.978 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 42.5336 | deepgemm | 40.9525 | 0.963 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 55.7432 | deepgemm | 52.9635 | 0.950 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 41.4287 | deepgemm | 43.5333 | 1.051 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 54.2862 | deepgemm | 57.0471 | 1.051 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 40.9842 | deepgemm | 39.2352 | 0.957 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 53.5868 | deepgemm | 50.8791 | 0.949 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 75.2619 | deepgemm | 72.8948 | 0.969 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 116.1340 | deepgemm | 111.1040 | 0.957 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 75.2587 | deepgemm | 71.1154 | 0.945 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 115.6379 | deepgemm | 108.2213 | 0.936 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 73.3461 | deepgemm | 76.4114 | 1.042 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 114.1203 | deepgemm | 117.7857 | 1.032 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 72.0895 | deepgemm | 68.0596 | 0.944 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 110.4869 | deepgemm | 103.5819 | 0.938 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 74.1822 | deepgemm | 74.3536 | 1.002 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 73.9630 | deepgemm | 74.2531 | 1.004 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 74.6696 | deepgemm | 72.7652 | 0.974 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 74.7328 | deepgemm | 72.8468 | 0.975 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 71.6966 | deepgemm | 76.1820 | 1.063 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 72.0743 | deepgemm | 76.4724 | 1.061 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 71.5979 | deepgemm | 68.6218 | 0.958 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 71.6523 | deepgemm | 68.6329 | 0.958 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 134.4507 | deepgemm | 131.0205 | 0.974 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 193.6365 | deepgemm | 187.0589 | 0.966 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 134.5889 | deepgemm | 127.6599 | 0.949 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 192.8499 | deepgemm | 181.2399 | 0.940 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 131.6639 | deepgemm | 137.6949 | 1.046 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 188.8640 | deepgemm | 196.5835 | 1.041 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 128.8373 | deepgemm | 121.5175 | 0.943 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 184.5801 | deepgemm | 173.2425 | 0.939 | — |
## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 5.5246 | deepgemm | 5.5426 | 1.003 | — |
| `b16_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.7904 | deepgemm | 5.8057 | 1.003 | — |
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.3654 | deepgemm | 6.3888 | 1.004 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 6.1057 | deepgemm | 6.1414 | 1.006 | — |
| `b16_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.3586 | deepgemm | 4.6851 | 1.075 | — |
| `b16_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 5.1782 | deepgemm | 5.7518 | 1.111 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.9541 | deepgemm | 5.4846 | 1.107 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.3572 | deepgemm | 4.6894 | 1.076 | — |
| `b16_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 5.3112 | deepgemm | 5.3316 | 1.004 | — |
| `b16_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.9235 | deepgemm | 4.8086 | 0.977 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.3172 | deepgemm | 5.3528 | 1.007 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.3135 | deepgemm | 5.3618 | 1.009 | — |
| `b16_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.7230 | deepgemm | 4.7352 | 1.003 | — |
| `b16_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.6620 | deepgemm | 4.6982 | 1.008 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7211 | deepgemm | 4.7645 | 1.009 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.7250 | deepgemm | 4.7685 | 1.009 | — |
| `b1_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.5801 | deepgemm | 4.6040 | 1.005 | — |
| `b1_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.6488 | deepgemm | 4.6773 | 1.006 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.6660 | deepgemm | 4.7271 | 1.013 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.6952 | deepgemm | 4.7204 | 1.005 | — |
| `b1_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.3167 | deepgemm | 4.6932 | 1.087 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.3106 | deepgemm | 4.6843 | 1.087 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.3340 | deepgemm | 4.6978 | 1.084 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.7838 | deepgemm | 5.4038 | 1.130 | — |
| `b1_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 5.0962 | deepgemm | 5.3594 | 1.052 | — |
| `b1_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 5.0952 | deepgemm | 5.3456 | 1.049 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.5081 | deepgemm | 4.5532 | 1.010 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.5370 | deepgemm | 4.5919 | 1.012 | — |
| `b1_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.7094 | deepgemm | 4.8427 | 1.028 | — |
| `b1_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.6505 | deepgemm | 4.8252 | 1.038 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 5.0426 | deepgemm | 5.3801 | 1.067 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5312 | deepgemm | 4.7171 | 1.041 | — |
| `b2_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.6544 | deepgemm | 4.6801 | 1.006 | — |
| `b2_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 4.6964 | deepgemm | 4.7057 | 1.002 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.7057 | deepgemm | 4.7258 | 1.004 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.7650 | deepgemm | 4.8008 | 1.008 | — |
| `b2_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.2318 | deepgemm | 4.4379 | 1.049 | — |
| `b2_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.2238 | deepgemm | 4.4502 | 1.054 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.2417 | deepgemm | 4.4368 | 1.046 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2240 | deepgemm | 4.4519 | 1.054 | — |
| `b2_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 5.1340 | deepgemm | 5.3967 | 1.051 | — |
| `b2_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.7488 | deepgemm | 4.7540 | 1.001 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.1581 | deepgemm | 5.4644 | 1.059 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.5079 | deepgemm | 4.5427 | 1.008 | — |
| `b2_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.5445 | deepgemm | 4.5624 | 1.004 | — |
| `b2_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 5.0335 | deepgemm | 5.3174 | 1.056 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.5303 | deepgemm | 4.5651 | 1.008 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.5357 | deepgemm | 4.5698 | 1.008 | — |
| `b4_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.9441 | deepgemm | 4.9820 | 1.008 | — |
| `b4_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.2130 | deepgemm | 5.2467 | 1.006 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.2510 | deepgemm | 5.2977 | 1.009 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.2198 | deepgemm | 5.2798 | 1.012 | — |
| `b4_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.3209 | deepgemm | 4.8934 | 1.132 | — |
| `b4_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.3114 | deepgemm | 4.8862 | 1.133 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.2834 | deepgemm | 4.7868 | 1.118 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2893 | deepgemm | 4.8345 | 1.127 | — |
| `b4_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.8403 | deepgemm | 4.8591 | 1.004 | — |
| `b4_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.7869 | deepgemm | 5.0746 | 1.060 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.9158 | deepgemm | 4.9527 | 1.008 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.9073 | deepgemm | 4.9407 | 1.007 | — |
| `b4_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.7130 | deepgemm | 4.9923 | 1.059 | — |
| `b4_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.7313 | deepgemm | 4.7790 | 1.010 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7853 | deepgemm | 4.8336 | 1.010 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.6404 | deepgemm | 4.9424 | 1.065 | — |
| `b8_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 4.9978 | deepgemm | 5.0098 | 1.002 | — |
| `b8_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 5.1416 | deepgemm | 5.1794 | 1.007 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.5304 | deepgemm | 5.5713 | 1.007 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 5.5149 | deepgemm | 5.5533 | 1.007 | — |
| `b8_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 4.2723 | deepgemm | 4.5951 | 1.076 | — |
| `b8_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 4.6580 | deepgemm | 5.2329 | 1.123 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.2493 | deepgemm | 4.4353 | 1.044 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.2547 | deepgemm | 4.4742 | 1.052 | — |
| `b8_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 4.7104 | deepgemm | 4.7247 | 1.003 | — |
| `b8_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 4.7314 | deepgemm | 4.7791 | 1.010 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.2007 | deepgemm | 5.3282 | 1.025 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.1798 | deepgemm | 5.2977 | 1.023 | — |
| `b8_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 4.6226 | deepgemm | 4.6469 | 1.005 | — |
| `b8_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 4.5850 | deepgemm | 4.6160 | 1.007 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.6514 | deepgemm | 4.7041 | 1.011 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.6094 | deepgemm | 4.6572 | 1.010 | — |
## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 43.2367 | deepgemm | 44.8829 | 1.038 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 56.6371 | deepgemm | 58.3084 | 1.030 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 42.8894 | deepgemm | 42.2019 | 0.984 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 56.4706 | deepgemm | 55.6460 | 0.985 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 43.3446 | deepgemm | 44.6351 | 1.030 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 56.4136 | deepgemm | 58.1625 | 1.031 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 42.7242 | deepgemm | 41.6104 | 0.974 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 56.3324 | deepgemm | 54.7785 | 0.972 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 76.0274 | deepgemm | 78.0064 | 1.026 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 114.7197 | deepgemm | 120.6304 | 1.052 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 74.4817 | deepgemm | 74.5032 | 1.000 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 114.7676 | deepgemm | 114.7454 | 1.000 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 76.3288 | deepgemm | 77.5976 | 1.017 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 116.7965 | deepgemm | 120.2671 | 1.030 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 74.3637 | deepgemm | 73.4574 | 0.988 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 115.7477 | deepgemm | 112.6860 | 0.974 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 75.7840 | deepgemm | 79.3970 | 1.048 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 75.9310 | deepgemm | 79.1240 | 1.042 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 75.1144 | deepgemm | 73.6916 | 0.981 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 75.1137 | deepgemm | 74.0107 | 0.985 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 75.9145 | deepgemm | 79.0995 | 1.042 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 76.1778 | deepgemm | 79.0002 | 1.037 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 75.1860 | deepgemm | 72.5095 | 0.964 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 75.0545 | deepgemm | 72.3152 | 0.964 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 136.4134 | deepgemm | 142.0493 | 1.041 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 197.7501 | deepgemm | 207.0028 | 1.047 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 134.6607 | deepgemm | 132.8556 | 0.987 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 194.5994 | deepgemm | 193.0987 | 0.992 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 136.1688 | deepgemm | 142.8017 | 1.049 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 198.9675 | deepgemm | 206.6323 | 1.039 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 133.5498 | deepgemm | 130.3368 | 0.976 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 197.3128 | deepgemm | 190.5352 | 0.966 | — |
## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 6.7739 | deepgemm | 6.7599 | 0.998 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 7.0251 | deepgemm | 7.0163 | 0.999 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.4765 | deepgemm | 4.5267 | 1.011 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.4973 | deepgemm | 4.4946 | 0.999 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.3600 | deepgemm | 5.3991 | 1.007 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.4878 | deepgemm | 5.5031 | 1.003 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 5.3105 | deepgemm | 5.3767 | 1.012 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.7440 | deepgemm | 4.7669 | 1.005 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 4.9611 | deepgemm | 5.2041 | 1.049 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.9416 | deepgemm | 5.1968 | 1.052 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.6505 | deepgemm | 4.8963 | 1.053 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.6474 | deepgemm | 4.8911 | 1.052 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.7081 | deepgemm | 4.9602 | 1.054 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.7903 | deepgemm | 5.0269 | 1.049 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.7657 | deepgemm | 5.2990 | 1.112 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.7575 | deepgemm | 4.7887 | 1.007 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.1130 | deepgemm | 5.3931 | 1.055 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.9724 | deepgemm | 5.1980 | 1.045 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.5746 | deepgemm | 4.8618 | 1.063 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.5643 | deepgemm | 4.8575 | 1.064 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.7589 | deepgemm | 5.0467 | 1.060 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.7952 | deepgemm | 5.0867 | 1.061 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.6460 | deepgemm | 4.9318 | 1.062 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.7669 | deepgemm | 4.7954 | 1.006 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.3689 | deepgemm | 5.3787 | 1.002 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 4.9118 | deepgemm | 4.9041 | 0.998 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.3919 | deepgemm | 4.4046 | 1.003 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.3649 | deepgemm | 4.3961 | 1.007 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 4.7270 | deepgemm | 4.7451 | 1.004 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 4.7078 | deepgemm | 4.7392 | 1.007 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 5.0769 | deepgemm | 5.3177 | 1.047 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 5.0932 | deepgemm | 5.3377 | 1.048 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 5.8325 | deepgemm | 5.8564 | 1.004 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 6.2541 | deepgemm | 6.2468 | 0.999 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 4.4646 | deepgemm | 4.4705 | 1.001 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 4.9751 | deepgemm | 5.2508 | 1.055 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 5.0635 | deepgemm | 5.1000 | 1.007 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 5.2189 | deepgemm | 5.2529 | 1.007 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 4.6838 | deepgemm | 4.7173 | 1.007 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 4.6195 | deepgemm | 4.6380 | 1.004 | — |
## deepgemm_sm100_tf32_hc_prenorm_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `m137_n24_k7680_s16` | tirx | 5.7587 | deepgemm | 5.4783 | 0.951 | — |
| `m13_n24_k7168_s1` | tirx | 24.3605 | deepgemm | 21.4179 | 0.879 | — |
| `m4096_n24_k28672_s16` | tirx | 65.3296 | deepgemm | 63.2366 | 0.968 | — |
| `m4096_n24_k7168_s1` | tirx | 25.8323 | deepgemm | 23.7160 | 0.918 | — |
## flash_attention4

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 19.8096 | flashattn_sm100 | 20.0760 | 1.013 | — |
| `s1024_h32kv16_causal` | tir | 20.4283 | flashattn_sm100 | 20.2782 | 0.993 | — |
| `s1024_h32kv32` | tir | 20.1870 | flashattn_sm100 | 20.4458 | 1.013 | — |
| `s1024_h32kv32_causal` | tir | 20.8159 | flashattn_sm100 | 21.3923 | 1.028 | — |
| `s1024_h32kv4` | tir | 19.2540 | flashattn_sm100 | 19.7927 | 1.028 | — |
| `s1024_h32kv4_causal` | tir | 19.5308 | flashattn_sm100 | 20.0070 | 1.024 | — |
| `s1024_h32kv8` | tir | 19.5045 | flashattn_sm100 | 19.5702 | 1.003 | — |
| `s1024_h32kv8_causal` | tir | 19.6599 | flashattn_sm100 | 19.7318 | 1.004 | — |
| `s2048_h32kv16` | tir | 57.3001 | flashattn_sm100 | 57.6297 | 1.006 | — |
| `s2048_h32kv16_causal` | tir | 36.2646 | flashattn_sm100 | 38.4182 | 1.059 | — |
| `s2048_h32kv32` | tir | 59.4427 | flashattn_sm100 | 59.7341 | 1.005 | — |
| `s2048_h32kv32_causal` | tir | 40.4139 | flashattn_sm100 | 40.0659 | 0.991 | — |
| `s2048_h32kv4` | tir | 55.4590 | flashattn_sm100 | 56.1430 | 1.012 | — |
| `s2048_h32kv4_causal` | tir | 35.1132 | flashattn_sm100 | 37.7379 | 1.075 | — |
| `s2048_h32kv8` | tir | 56.1573 | flashattn_sm100 | 56.9796 | 1.015 | — |
| `s2048_h32kv8_causal` | tir | 35.4236 | flashattn_sm100 | 37.8938 | 1.070 | — |
| `s4096_h32kv16` | tir | 213.4413 | flashattn_sm100 | 212.4653 | 0.995 | — |
| `s4096_h32kv16_causal` | tir | 111.9785 | flashattn_sm100 | 117.6295 | 1.050 | — |
| `s4096_h32kv32` | tir | 214.2722 | flashattn_sm100 | 217.6024 | 1.016 | — |
| `s4096_h32kv32_causal` | tir | 121.5008 | flashattn_sm100 | 120.1768 | 0.989 | — |
| `s4096_h32kv4` | tir | 205.9919 | flashattn_sm100 | 210.2510 | 1.021 | — |
| `s4096_h32kv4_causal` | tir | 109.2315 | flashattn_sm100 | 115.9414 | 1.061 | — |
| `s4096_h32kv8` | tir | 206.8473 | flashattn_sm100 | 208.3275 | 1.007 | — |
| `s4096_h32kv8_causal` | tir | 110.3818 | flashattn_sm100 | 115.4242 | 1.046 | — |
| `s8192_h32kv16` | tir | 767.7250 | flashattn_sm100 | 752.4508 | 0.980 | — |
| `s8192_h32kv16_causal` | tir | 465.8747 | flashattn_sm100 | 413.8001 | 0.888 | — |
| `s8192_h32kv32` | tir | 782.5778 | flashattn_sm100 | 769.3423 | 0.983 | — |
| `s8192_h32kv32_causal` | tir | 436.0990 | flashattn_sm100 | 430.6355 | 0.987 | — |
| `s8192_h32kv4` | tir | 772.7373 | flashattn_sm100 | 768.1529 | 0.994 | — |
| `s8192_h32kv4_causal` | tir | 404.4490 | flashattn_sm100 | 415.1800 | 1.027 | — |
| `s8192_h32kv8` | tir | 765.2810 | flashattn_sm100 | 776.3060 | 1.014 | — |
| `s8192_h32kv8_causal` | tir | 400.9411 | flashattn_sm100 | 419.6447 | 1.047 | — |
## fp16_bf16_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 6.7977 | deepgemm-cublaslt | 5.9876 | 0.881 | deepgemm-bf16=7.9204, torch-cublas=6.0023 |
| `bf16_16384x16384x16384` | tir | 5606.5364 | torch-cublas | 5587.3767 | 0.997 | deepgemm-bf16=6748.0801, deepgemm-cublaslt=5611.5691 |
| `bf16_2048x2048x2048` | tir | 16.3830 | deepgemm-cublaslt | 15.6939 | 0.958 | deepgemm-bf16=18.5112, torch-cublas=15.7318 |
| `bf16_4096x4096x4096` | tir | 93.4358 | torch-cublas | 89.9932 | 0.963 | deepgemm-bf16=91.2166, deepgemm-cublaslt=90.3658 |
| `bf16_8192x8192x8192` | tir | 679.0821 | deepgemm-cublaslt | 673.2049 | 0.991 | deepgemm-bf16=674.1139, torch-cublas=730.1406 |
| `fp16_1024x1024x1024` | tir | 6.9652 | deepgemm-cublaslt | 6.0668 | 0.871 | torch-cublas=6.0743 |
| `fp16_16384x16384x16384` | tir | 5713.6190 | deepgemm-cublaslt | 5683.7476 | 0.995 | torch-cublas=5832.5427 |
| `fp16_2048x2048x2048` | tir | 16.5389 | torch-cublas | 15.8654 | 0.959 | deepgemm-cublaslt=15.8686 |
| `fp16_4096x4096x4096` | tir | 93.9727 | torch-cublas | 92.0741 | 0.980 | deepgemm-cublaslt=92.9098 |
| `fp16_8192x8192x8192` | tir | 741.7718 | deepgemm-cublaslt | 760.2102 | 1.025 | torch-cublas=778.0312 |
## fp8_blockwise_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 50.4992 | deepgemm | 50.7831 | 1.006 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 113.3228 | deepgemm | 114.1334 | 1.007 | — |
| `deepgemm_m4096_n32768_k512` | tir | 68.8340 | deepgemm | 72.6083 | 1.055 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 81.2952 | deepgemm | 81.0871 | 0.997 | — |
| `deepgemm_m4096_n576_k7168` | tir | 20.0076 | deepgemm | 20.4505 | 1.022 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 340.1152 | deepgemm | 333.0350 | 0.979 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 43.1832 | deepgemm | 43.6777 | 1.011 | — |
## nvfp4_gemm

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 5.4714 | flashinfer | 4.5010 | 0.823 | cublaslt_nvfp4=4.5685 |
| `16384x16384x16384` | tir | 1490.2281 | cublaslt_nvfp4 | 1428.0843 | 0.958 | flashinfer=1447.4385 |
| `2048x2048x2048` | tir | 8.4138 | cublaslt_nvfp4 | 7.5407 | 0.896 | flashinfer=7.6777 |
| `4096x4096x4096` | tir | 29.6078 | cublaslt_nvfp4 | 28.8391 | 0.974 | flashinfer=29.9187 |
| `8192x8192x8192` | tir | 177.4519 | cublaslt_nvfp4 | 179.1572 | 1.010 | flashinfer=181.3831 |
## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1708.4414 | flashmla | 1699.0193 | 0.994 | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1818.7030 | flashmla | 1936.1847 | 1.065 | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1676.9131 | flashmla | 1758.1067 | 1.048 | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1733.5709 | flashmla | 1891.3820 | 1.091 | — |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 1969.8459 | flashmla | 1983.0878 | 1.007 | — |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1703.4234 | flashmla | 1816.4004 | 1.066 | — |
## sparse_flashmla_prefill_head128_small_topk_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_smalltopk_dqk512_hq128_s4096_kv32768_topk1280` | tirx | 1110.1777 | flashmla | 1141.2565 | 1.028 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv65536_topk1280` | tirx | 1152.5229 | flashmla | 1206.7187 | 1.047 | — |
| `bench_smalltopk_dqk512_hq128_s4096_kv8192_topk1280` | tirx | 1106.4839 | flashmla | 1145.3705 | 1.035 | — |
## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 374.2908 | flashmla | 379.3439 | 1.014 | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 373.7587 | flashmla | 382.0244 | 1.022 | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 375.5029 | flashmla | 389.1991 | 1.036 | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 368.5957 | flashmla | 371.3999 | 1.008 | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 387.3669 | flashmla | 400.9529 | 1.035 | — |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 387.7694 | flashmla | 408.4249 | 1.053 | — |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 395.0347 | flashmla | 419.9437 | 1.063 | — |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 373.9255 | flashmla | 384.2568 | 1.028 | — |
