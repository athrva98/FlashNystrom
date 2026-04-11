/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once

#include "utils.h"

namespace flash_nystrom {

// landmark kernel: segment-mean over Q and K to produce the nystrom landmarks
// pretty simple — each landmark is just the average of its segment
//
// For each landmark l in [0, m):
//   segment = [l * seg_len, min((l+1) * seg_len, N))
//   Q_tilde[b,h,l,:] = mean(Q[b,h,segment,:]) * scale
//   K_tilde[b,h,l,:] = mean(K[b,h,segment,:]) * scale
//
// Grid:  (B*H, m)     — one CTA per (batch-head, landmark)
// Block: (256)         — threads stride over D dimension
//
// SMEM: none (pure register accumulation, D <= 256 so each thread handles <=1 elem)

template <typename scalar_t>
__global__ void landmark_kernel(
    const scalar_t* __restrict__ q,   // (B*H, N, D) row-major
    const scalar_t* __restrict__ k,   // (B*H, N, D)
    scalar_t* __restrict__ q_tilde,   // (B*H, m, D)
    scalar_t* __restrict__ k_tilde,   // (B*H, m, D)
    int N, int D, int m, float scale
) {
    const int bh = blockIdx.x;
    const int landmark = blockIdx.y;

    // Use floor division for segment length. Last segment absorbs remainder.
    // seg_len_floor = N / m (floor). Last segment: [seg_len_floor * (m-1), N)
    const int seg_len_floor = N / m;
    const int seg_start = landmark * seg_len_floor;
    const int seg_end = (landmark == m - 1) ? N : (seg_start + seg_len_floor);
    const int actual_len = seg_end - seg_start;
    if (actual_len <= 0) return;
    // Ignore the seg_len parameter, compute from N and m directly

    const float inv_len = 1.0f / static_cast<float>(actual_len);

    const scalar_t* q_bh = q + bh * N * D;
    const scalar_t* k_bh = k + bh * N * D;
    scalar_t* qt_out = q_tilde + bh * m * D + landmark * D;
    scalar_t* kt_out = k_tilde + bh * m * D + landmark * D;

    // Each thread handles one or more D dimensions
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;

        for (int n = seg_start; n < seg_end; n++) {
            q_sum += to_float(q_bh[n * D + d]);
            k_sum += to_float(k_bh[n * D + d]);
        }

        qt_out[d] = from_float<scalar_t>(q_sum * inv_len * scale);
        kt_out[d] = from_float<scalar_t>(k_sum * inv_len * scale);
    }
}

// In-place scaling: Q *= scale, K *= scale
// Grid: (ceil(total/256)), Block: 256

template <typename scalar_t>
__global__ void scale_inplace_kernel(
    scalar_t* __restrict__ data,    // flattened array
    int total_elements,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        data[idx] = from_float<scalar_t>(to_float(data[idx]) * scale);
    }
}

// -- launch wrapper --


template <typename scalar_t>
void launch_landmarks(
    const scalar_t* q, const scalar_t* k,
    scalar_t* q_tilde, scalar_t* k_tilde,
    int BH, int N, int D, int m, float scale,
    cudaStream_t stream
) {
    FN_CHECK(BH > 0 && N > 0 && D > 0 && m > 0, "launch_landmarks: invalid dims");
    FN_CHECK(m <= N, "launch_landmarks: m > N");

    dim3 grid(BH, m);
    dim3 block(min(256, D));  // No point having more threads than D
    // Ensure at least 32 threads (one warp)
    if (block.x < 32) block.x = 32;

    landmark_kernel<scalar_t><<<grid, block, 0, stream>>>(
        q, k, q_tilde, k_tilde, N, D, m, scale);
    FN_CUDA_KERNEL_CHECK();
}

template <typename scalar_t>
void launch_scale_inplace(
    scalar_t* data, int total_elements, float scale, cudaStream_t stream
) {
    if (total_elements <= 0) return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    scale_inplace_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        data, total_elements, scale);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
