/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"

namespace flash_nystrom {

// Scatter gradients from landmarks back to sequence positions.
// dQ_s[n,d] += dQ_tilde[l,d] / seg_len   for n in segment l
// Same for dK_s. No races — segments are disjoint.
// Grid: (BH, m), Block: min(256, D)

template <typename scalar_t>
__global__ void landmark_bwd_kernel(
    const float* __restrict__ dQ_tilde,  // (BH, m, D) FP32
    const float* __restrict__ dK_tilde,  // (BH, m, D) FP32
    scalar_t* __restrict__ dQ_s,         // (BH, N, D) accumulated
    scalar_t* __restrict__ dK_s,         // (BH, N, D) accumulated
    int N, int D, int m
) {
    const int bh = blockIdx.x;
    const int landmark = blockIdx.y;

    const int seg_len_floor = N / m;
    const int seg_start = landmark * seg_len_floor;
    const int seg_end = (landmark == m - 1) ? N : (seg_start + seg_len_floor);
    const int actual_len = seg_end - seg_start;
    if (actual_len <= 0) return;

    const float inv_len = 1.0f / static_cast<float>(actual_len);
    const float* dqt = dQ_tilde + bh * m * D + landmark * D;
    const float* dkt = dK_tilde + bh * m * D + landmark * D;
    scalar_t* dqs_base = dQ_s + bh * N * D;
    scalar_t* dks_base = dK_s + bh * N * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float dq_val = dqt[d] * inv_len;
        float dk_val = dkt[d] * inv_len;
        for (int n = seg_start; n < seg_end; n++) {
            // Accumulate: dQ_s already has contributions from kernel1_bwd
            float existing_q = to_float(dqs_base[n * D + d]);
            dqs_base[n * D + d] = from_float<scalar_t>(existing_q + dq_val);
            float existing_k = to_float(dks_base[n * D + d]);
            dks_base[n * D + d] = from_float<scalar_t>(existing_k + dk_val);
        }
    }
}

template <typename scalar_t>
void launch_landmark_bwd(
    const float* dQ_tilde, const float* dK_tilde,
    scalar_t* dQ_s, scalar_t* dK_s,
    int BH, int N, int D, int m, cudaStream_t stream
) {
    dim3 grid(BH, m);
    dim3 block(min(256, D));
    if (block.x < 32) block.x = 32;
    landmark_bwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
        dQ_tilde, dK_tilde, dQ_s, dK_s, N, D, m);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
