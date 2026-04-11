/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"

namespace flash_nystrom {

// dV[n,d] += sum_k weight[h,k] * dO[n+k-pad, d]  (transposed conv)
template <typename scalar_t>
__global__ void dconv_dv_kernel(
    const scalar_t* __restrict__ dO,          // (BH, N, D)
    const scalar_t* __restrict__ conv_weight,  // (num_heads, kernel_size)
    scalar_t* __restrict__ dV,                // (BH, N, D) accumulated
    int N, int D, int num_heads, int kernel_size
) {
    const int bh = blockIdx.y;
    const int h = bh % num_heads;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;

    const int n = idx / D;
    const int d = idx % D;
    const int pad = kernel_size / 2;

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        // Forward: out[n] += w[k] * V[n+k-pad]
        // Backward dV: dV[n] += w[k] * dO[n-k+pad]  (flipped kernel)
        int src_n = n - k + pad;
        if (src_n >= 0 && src_n < N) {
            sum += to_float(conv_weight[h * kernel_size + k])
                 * to_float(dO[bh * N * D + src_n * D + d]);
        }
    }
    float existing = to_float(dV[bh * N * D + n * D + d]);
    dV[bh * N * D + n * D + d] = from_float<scalar_t>(existing + sum);
}

// dweight[h,k] = sum_{b,n,d} V[b,h,n+k-pad,d] * dO[b,h,n,d]
template <typename scalar_t>
__global__ void dconv_dweight_kernel(
    const scalar_t* __restrict__ V,           // (BH, N, D)
    const scalar_t* __restrict__ dO,          // (BH, N, D)
    float* __restrict__ dweight,              // (num_heads, kernel_size) FP32
    int N, int D, int num_heads, int batch_size, int kernel_size
) {
    const int h = blockIdx.x;
    const int k = blockIdx.y;
    const int tid = threadIdx.x;
    const int pad = kernel_size / 2;

    __shared__ float scratch[8];

    float local_sum = 0.0f;
    const int B = batch_size;
    const int BH_per_head = B;

    for (int b = 0; b < B; b++) {
        int bh = b * num_heads + h;
        for (int nd = tid; nd < N * D; nd += blockDim.x) {
            int n = nd / D;
            int d = nd % D;
            int src_n = n + k - pad;
            if (src_n >= 0 && src_n < N) {
                local_sum += to_float(V[bh * N * D + src_n * D + d])
                           * to_float(dO[bh * N * D + n * D + d]);
            }
        }
    }

    // Block reduction
    float result = block_reduce_sum(local_sum, scratch);
    if (tid == 0) {
        dweight[h * kernel_size + k] = result;
    }
}

template <typename scalar_t>
void launch_dconv_bwd(
    const scalar_t* dO, const scalar_t* V, const scalar_t* conv_weight,
    scalar_t* dV, float* dweight_f32,
    int BH, int N, int D, int num_heads, int batch_size, int kernel_size,
    cudaStream_t stream
) {
    if (kernel_size <= 0) return;

    // dV kernel
    {
        int total = N * D;
        dim3 grid((total + 255) / 256, BH);
        dim3 block(256);
        dconv_dv_kernel<scalar_t><<<grid, block, 0, stream>>>(
            dO, conv_weight, dV, N, D, num_heads, kernel_size);
        FN_CUDA_KERNEL_CHECK();
    }

    // dweight kernel
    {
        dim3 grid(num_heads, kernel_size);
        dim3 block(256);
        dconv_dweight_kernel<scalar_t><<<grid, block, 0, stream>>>(
            V, dO, dweight_f32, N, D, num_heads, batch_size, kernel_size);
        FN_CUDA_KERNEL_CHECK();
    }
}

} // namespace flash_nystrom
