/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once

#include "utils.h"

namespace flash_nystrom {

// DConv Residual: output += depthwise_conv1d(V, weight)
//
// Per-head 1D convolution over the sequence dimension with zero-padding.
// weight shape: (H, kernel_size). Each head is independent.
//
// Grid:  (ceil(N*D / 256), B*H)
// Block: (256)
//
// Each thread computes one output element at position (n, d):
//   conv_val = sum_{k=0}^{ks-1} weight[h, k] * V[b,h, n+k-pad, d]
//   output[b,h,n,d] += conv_val

template <typename scalar_t>
__global__ void dconv_residual_kernel(
    const scalar_t* __restrict__ v,          // (B*H, N, D)
    const scalar_t* __restrict__ conv_weight, // (num_heads, kernel_size)
    scalar_t*       __restrict__ output,     // (B*H, N, D), accumulated
    int N, int D, int num_heads, int kernel_size
) {
    const int bh = blockIdx.y;
    const int h = bh % num_heads;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * D;
    if (tid >= total) return;

    const int n = tid / D;
    const int d = tid % D;
    const int pad = kernel_size / 2;

    const scalar_t* v_bh = v + bh * N * D;
    const scalar_t* w_h = conv_weight + h * kernel_size;

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        int src_n = n + k - pad;
        if (src_n >= 0 && src_n < N) {
            sum += to_float(v_bh[src_n * D + d]) * to_float(w_h[k]);
        }
    }

    // Accumulate onto output
    scalar_t* out_bh = output + bh * N * D;
    float existing = to_float(out_bh[n * D + d]);
    out_bh[n * D + d] = from_float<scalar_t>(existing + sum);
}

template <typename scalar_t>
void launch_dconv_residual(
    const scalar_t* v, const scalar_t* conv_weight, scalar_t* output,
    int BH, int N, int D, int num_heads, int kernel_size,
    cudaStream_t stream
) {
    if (conv_weight == nullptr || kernel_size <= 0) return;

    int total = N * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dim3 grid(blocks, BH);
    dim3 block(threads);

    dconv_residual_kernel<scalar_t><<<grid, block, 0, stream>>>(
        v, conv_weight, output, N, D, num_heads, kernel_size);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
