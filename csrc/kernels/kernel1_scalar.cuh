/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once

// Scalar (non-tensor-core) implementation of kernel1 for FP32 fallback.
// Also used as reference for correctness validation.

#include "utils.h"

namespace flash_nystrom {

constexpr int kK1ScalarBr = 64;
constexpr int kK1ScalarThreads = 256;

template <typename scalar_t>
__global__ void kernel1_scalar_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_tilde,
    const scalar_t* __restrict__ step2,
    scalar_t*       __restrict__ output,
    float*          __restrict__ softmax1_lse_out,
    int N, int D, int m
) {
    const int tile_idx = blockIdx.x;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    constexpr int Br = kK1ScalarBr;

    const int row_start = tile_idx * Br;
    const int row_end = min(row_start + Br, N);
    const int tile_rows = row_end - row_start;
    if (tile_rows <= 0) return;

    extern __shared__ char smem_raw[];
    scalar_t* K_tilde_s = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* step2_s = K_tilde_s + m * D;
    scalar_t* Q_tile_s = step2_s + m * D;
    float* S_row = reinterpret_cast<float*>(Q_tile_s + Br * D);
    float* scratch = S_row + Br * m;

    const scalar_t* kt_ptr = k_tilde + bh * m * D;
    const scalar_t* s2_ptr = step2 + bh * m * D;
    for (int idx = tid; idx < m * D; idx += nthreads) {
        K_tilde_s[idx] = kt_ptr[idx];
        step2_s[idx] = s2_ptr[idx];
    }
    const scalar_t* q_ptr = q + bh * N * D + row_start * D;
    for (int idx = tid; idx < tile_rows * D; idx += nthreads) {
        Q_tile_s[idx] = q_ptr[idx];
    }
    __syncthreads();

    for (int idx = tid; idx < tile_rows * m; idx += nthreads) {
        int row = idx / m, col = idx % m;
        float acc = 0.0f;
        for (int d = 0; d < D; d++)
            acc += to_float(Q_tile_s[row * D + d]) * to_float(K_tilde_s[col * D + d]);
        S_row[idx] = acc;
    }
    __syncthreads();

    float* lse_out = softmax1_lse_out + bh * N + row_start;
    for (int row = 0; row < tile_rows; row++) {
        float local_max = -FLT_MAX;
        for (int j = tid; j < m; j += nthreads) local_max = fmaxf(local_max, S_row[row * m + j]);
        float rmax = block_reduce_max(local_max, scratch);
        float local_sum = 0.0f;
        for (int j = tid; j < m; j += nthreads) {
            float val = expf(S_row[row * m + j] - rmax);
            S_row[row * m + j] = val;
            local_sum += val;
        }
        float rsum = block_reduce_sum(local_sum, scratch);
        float inv_sum = 1.0f / (rsum + 1e-12f);
        for (int j = tid; j < m; j += nthreads) S_row[row * m + j] *= inv_sum;
        __syncthreads();
        if (tid == 0) lse_out[row] = rmax + logf(rsum + 1e-12f);
    }
    __syncthreads();

    scalar_t* out_ptr = output + bh * N * D + row_start * D;
    for (int idx = tid; idx < tile_rows * D; idx += nthreads) {
        int row = idx / D, d = idx % D;
        float acc = 0.0f;
        for (int j = 0; j < m; j++)
            acc += S_row[row * m + j] * to_float(step2_s[j * D + d]);
        out_ptr[idx] = from_float<scalar_t>(acc);
    }
}

template <typename scalar_t>
void launch_kernel1_scalar(
    const scalar_t* q, const scalar_t* k_tilde, const scalar_t* step2,
    scalar_t* output, float* softmax1_lse,
    int BH, int N, int D, int m,
    cudaStream_t stream
) {
    constexpr int Br = kK1ScalarBr;
    int num_tiles = (N + Br - 1) / Br;
    dim3 grid(num_tiles, BH);
    dim3 block(kK1ScalarThreads);

    size_t smem_bytes =
        m * D * sizeof(scalar_t) * 2      // K_tilde + step2
      + Br * D * sizeof(scalar_t)          // Q_tile
      + Br * m * sizeof(float)             // S_row
      + (kK1ScalarThreads / 32) * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        size_t max_smem = get_max_smem_per_block();
        FN_CHECK(smem_bytes <= max_smem, "kernel1_scalar: insufficient smem");
        FN_CUDA_CHECK(cudaFuncSetAttribute(
            kernel1_scalar_kernel<scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
    }
    kernel1_scalar_kernel<scalar_t><<<grid, block, smem_bytes, stream>>>(
        q, k_tilde, step2, output, softmax1_lse, N, D, m);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
