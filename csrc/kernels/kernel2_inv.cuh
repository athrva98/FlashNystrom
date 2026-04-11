/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once

#include "utils.h"

namespace flash_nystrom {

// kernel2inv: softmax(Q_tilde @ K_tilde^T) then newton-schulz pseudoinverse
// everything runs in shared memory since its just m x m matrices (m is small)
// 6 iterations of third-order N-S is more than enough for convergence
// All operations are on (m, m) matrices in shared memory.
//
// Grid:  (B*H)    — one CTA per (batch, head)
// Block: (256)    — threads cooperatively operate on (m, m) matrices
//
// SMEM layout (all FP32):
//   K2:    m*m   (kernel_2 after softmax, read-only during iterations)
//   Z:     m*m   (current Newton-Schulz iterate)
//   Zold:  m*m   (previous iterate, needed for final multiply)
//   T1:    m*m   (temporary)
//   T2:    m*m   (temporary)
//   scratch: 8 floats (reduction workspace)
//
// Total: 5 * m^2 * 4 + 32 bytes
//   m=64:  80KB  (fits SM80 opt-in of ~100KB)

template <typename scalar_t>
__global__ void kernel2_inv_kernel(
    const scalar_t* __restrict__ q_tilde,   // (B*H, m, D)
    const scalar_t* __restrict__ k_tilde,   // (B*H, m, D)
    float* __restrict__ kernel2_inv_out,     // (B*H, m, m) FP32
    float* __restrict__ softmax_lse_out,     // (B*H, m)
    float* __restrict__ ns_iterates_out,     // (B*H, newton_iter, m, m) or nullptr
    int D, int m, int newton_iter
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int mm = m * m;

    extern __shared__ float smem[];
    float* K2      = smem;
    float* Z       = smem + mm;
    float* Zold    = smem + 2 * mm;
    float* T1      = smem + 3 * mm;
    float* T2      = smem + 4 * mm;
    float* scratch = smem + 5 * mm;

    const scalar_t* qt = q_tilde + bh * m * D;
    const scalar_t* kt = k_tilde + bh * m * D;

    // Step 1: K2 = Q_tilde @ K_tilde^T (FP32 accumulation)
    for (int idx = tid; idx < mm; idx += nthreads) {
        int i = idx / m;
        int j = idx % m;
        float acc = 0.0f;
        for (int d = 0; d < D; d++) {
            acc += to_float(qt[i * D + d]) * to_float(kt[j * D + d]);
        }
        K2[idx] = acc;
    }
    __syncthreads();

    // Step 2: Row-wise softmax on K2, save logsumexp
    float* lse_out = softmax_lse_out + bh * m;

    for (int row = 0; row < m; row++) {
        float local_max = -FLT_MAX;
        for (int j = tid; j < m; j += nthreads) {
            local_max = fmaxf(local_max, K2[row * m + j]);
        }
        float row_max = block_reduce_max(local_max, scratch);

        float local_sum = 0.0f;
        for (int j = tid; j < m; j += nthreads) {
            float val = expf(K2[row * m + j] - row_max);
            K2[row * m + j] = val;
            local_sum += val;
        }
        float row_sum = block_reduce_sum(local_sum, scratch);

        float inv_sum = 1.0f / (row_sum + 1e-12f);
        for (int j = tid; j < m; j += nthreads) {
            K2[row * m + j] *= inv_sum;
        }
        __syncthreads();

        if (tid == 0) {
            lse_out[row] = row_max + logf(row_sum + 1e-12f);
        }
    }
    __syncthreads();

    // Step 3: Z_0 = K2^T / ||K2||_1
    // K2 is row-stochastic (softmax output), so ||K2||_inf = 1.
    // ||K2||_1 = max column sum.

    for (int j = tid; j < m; j += nthreads) {
        float col_sum = 0.0f;
        for (int i = 0; i < m; i++) col_sum += K2[i * m + j];
        T1[j] = col_sum;
    }
    __syncthreads();

    float local_col_max = -FLT_MAX;
    for (int j = tid; j < m; j += nthreads) {
        local_col_max = fmaxf(local_col_max, T1[j]);
    }
    float norm1 = block_reduce_max(local_col_max, scratch);
    float inv_norm = 1.0f / (norm1 + 1e-12f);

    for (int idx = tid; idx < mm; idx += nthreads) {
        int row = idx / m;
        int col = idx % m;
        Z[idx] = K2[col * m + row] * inv_norm;  // transpose
    }
    __syncthreads();

    // Step 4: Newton-Schulz iterations (third order)
    // Z_{j+1} = 0.25 * Z_j * (13I - K2*Z_j * (15I - K2*Z_j * (7I - K2*Z_j)))
    //
    // Requires Zold to preserve Z_j for the final 0.25 * Z_j @ (...) multiply.

    for (int iter = 0; iter < newton_iter; iter++) {
        // Save Zold = Z (both to SMEM and global for backward)
        for (int idx = tid; idx < mm; idx += nthreads) Zold[idx] = Z[idx];
        if (ns_iterates_out != nullptr) {
            float* iter_out = ns_iterates_out + bh * newton_iter * mm + iter * mm;
            for (int idx = tid; idx < mm; idx += nthreads) iter_out[idx] = Z[idx];
        }
        __syncthreads();

        // T1 = K2 @ Zold
        for (int idx = tid; idx < mm; idx += nthreads) {
            int row = idx / m, col = idx % m;
            float acc = 0.0f;
            for (int kk = 0; kk < m; kk++) acc += K2[row * m + kk] * Zold[kk * m + col];
            T1[idx] = acc;
        }
        __syncthreads();

        // T2 = 7I - T1
        for (int idx = tid; idx < mm; idx += nthreads) {
            int row = idx / m, col = idx % m;
            T2[idx] = ((row == col) ? 7.0f : 0.0f) - T1[idx];
        }
        __syncthreads();

        // Z = T1 @ T2   (= KZ * (7I - KZ))
        for (int idx = tid; idx < mm; idx += nthreads) {
            int row = idx / m, col = idx % m;
            float acc = 0.0f;
            for (int kk = 0; kk < m; kk++) acc += T1[row * m + kk] * T2[kk * m + col];
            Z[idx] = acc;
        }
        __syncthreads();

        // T2 = 15I - Z
        for (int idx = tid; idx < mm; idx += nthreads) {
            int row = idx / m, col = idx % m;
            T2[idx] = ((row == col) ? 15.0f : 0.0f) - Z[idx];
        }
        __syncthreads();

        // Z = T1 @ T2   (= KZ * (15I - KZ*(7I-KZ)))
        for (int idx = tid; idx < mm; idx += nthreads) {
            int row = idx / m, col = idx % m;
            float acc = 0.0f;
            for (int kk = 0; kk < m; kk++) acc += T1[row * m + kk] * T2[kk * m + col];
            Z[idx] = acc;
        }
        __syncthreads();

        // T2 = 13I - Z
        for (int idx = tid; idx < mm; idx += nthreads) {
            int row = idx / m, col = idx % m;
            T2[idx] = ((row == col) ? 13.0f : 0.0f) - Z[idx];
        }
        __syncthreads();

        // Z = 0.25 * Zold @ T2
        for (int idx = tid; idx < mm; idx += nthreads) {
            int row = idx / m, col = idx % m;
            float acc = 0.0f;
            for (int kk = 0; kk < m; kk++) acc += Zold[row * m + kk] * T2[kk * m + col];
            Z[idx] = 0.25f * acc;
        }
        __syncthreads();
    }

    // Step 5: Write output
    float* out = kernel2_inv_out + bh * mm;
    for (int idx = tid; idx < mm; idx += nthreads) {
        out[idx] = Z[idx];
    }
}

// -- launch wrapper --


template <typename scalar_t>
void launch_kernel2_inv(
    const scalar_t* q_tilde, const scalar_t* k_tilde,
    float* kernel2_inv, float* softmax_lse,
    float* ns_iterates,  // (BH, newton_iter, m, m) or nullptr
    int BH, int D, int m, int newton_iter,
    cudaStream_t stream
) {
    FN_CHECK(m > 0 && m <= kMaxLandmarks, "launch_kernel2_inv: m out of range");

    dim3 grid(BH);
    dim3 block(256);

    size_t smem_bytes = (5 * m * m + 8) * sizeof(float);

    size_t max_smem = get_max_smem_per_block();
    if (smem_bytes > 48 * 1024) {
        FN_CHECK(smem_bytes <= max_smem,
                 "kernel2_inv: m too large for available shared memory");
        FN_CUDA_CHECK(cudaFuncSetAttribute(
            kernel2_inv_kernel<scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
    }

    kernel2_inv_kernel<scalar_t><<<grid, block, smem_bytes, stream>>>(
        q_tilde, k_tilde, kernel2_inv, softmax_lse, ns_iterates, D, m, newton_iter);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
