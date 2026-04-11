/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"

namespace flash_nystrom {

// kernel2inv backward via implicit function theorem
// instead of unrolling all 6 N-S iterations backwards (nightmrae),
// we use: dK2 = -K2_inv^T @ dK2_inv @ K2_inv^T
// this is exact when N-S has converged (which it has after 6 iters)
//
// For a converged pseudoinverse K2_inv ≈ pinv(K2):
//   dK2 = -K2_inv^T @ dK2_inv @ K2_inv^T
//
// This is exact when N-S has converged and avoids the impossible task of
// backpropagating through 6 iterations of nested matrix products.
//
// Then backprop through softmax(Qt @ Kt^T) = K2:
//   dS2 = K2 * (dK2 - rowsum(dK2 * K2))
//   dQ_tilde += dS2 @ K_tilde
//   dK_tilde += dS2^T @ Q_tilde
//
// Grid: (BH), Block: 256
// SMEM: 4 × m² floats (K2, K2_inv_local, dK2, temp) + scratch

template <typename scalar_t>
__global__ void kernel2_inv_bwd_kernel(
    const scalar_t* __restrict__ q_tilde,     // (BH, m, D)
    const scalar_t* __restrict__ k_tilde,     // (BH, m, D)
    const float*    __restrict__ lse2,         // (BH, m)
    const float*    __restrict__ k2_inv,       // (BH, m, m) — the forward output
    const float*    __restrict__ dK2_inv_in,   // (BH, m, m) — gradient input
    float*          __restrict__ dQ_tilde,     // (BH, m, D) FP32, atomicAdd
    float*          __restrict__ dK_tilde,     // (BH, m, D) FP32, atomicAdd
    int D, int m
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int mm = m * m;

    extern __shared__ float smem[];
    float* K2      = smem;               // recomputed softmax output
    float* K2inv_s = smem + mm;          // copy of K2_inv
    float* dK2     = smem + 2 * mm;      // gradient w.r.t. K2
    float* T1      = smem + 3 * mm;      // temp
    float* scratch = smem + 4 * mm;

    const scalar_t* qt = q_tilde + bh * m * D;
    const scalar_t* kt = k_tilde + bh * m * D;

    // Recompute K2 = softmax(Qt @ Kt^T) from LSE2
    const float* lse2_bh = lse2 + bh * m;
    for (int idx = tid; idx < mm; idx += nthreads) {
        int i = idx / m, j = idx % m;
        float dot = 0.0f;
        for (int d = 0; d < D; d++)
            dot += to_float(qt[i * D + d]) * to_float(kt[j * D + d]);
        K2[idx] = expf(dot - lse2_bh[i]);
    }
    __syncthreads();

    // Load K2_inv into SMEM
    const float* k2inv_bh = k2_inv + bh * mm;
    for (int idx = tid; idx < mm; idx += nthreads) K2inv_s[idx] = k2inv_bh[idx];
    __syncthreads();

    // Implicit function theorem: dK2 = -K2_inv^T @ dK2_inv @ K2_inv^T
    const float* dk2i = dK2_inv_in + bh * mm;

    // Step 1: T1 = dK2_inv @ K2_inv^T
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        float acc = 0.0f;
        for (int k = 0; k < m; k++) acc += dk2i[r * m + k] * K2inv_s[c * m + k]; // K2inv^T[k,c] = K2inv[c,k]
        T1[idx] = acc;
    }
    __syncthreads();

    // Step 2: dK2 = -K2_inv^T @ T1
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        float acc = 0.0f;
        for (int k = 0; k < m; k++) acc += K2inv_s[k * m + r] * T1[k * m + c]; // K2inv^T[r,k] = K2inv[k,r]
        dK2[idx] = -acc;
    }
    __syncthreads();

    // Softmax backward: K2 = softmax(S2), dS2 = K2 * (dK2 - D2)
    // D2[i] = sum_j(dK2[i,j] * K2[i,j])
    for (int i = tid; i < m; i += nthreads) {
        float D_i = 0.0f;
        for (int j = 0; j < m; j++) D_i += dK2[i * m + j] * K2[i * m + j];
        for (int j = 0; j < m; j++)
            T1[i * m + j] = K2[i * m + j] * (dK2[i * m + j] - D_i);
    }
    // T1 = dS2
    __syncthreads();

    // dQ_tilde += dS2 @ K_tilde
    float* dQt_bh = dQ_tilde + bh * m * D;
    for (int idx = tid; idx < m * D; idx += nthreads) {
        int i = idx / D, d = idx % D;
        float sum = 0.0f;
        for (int j = 0; j < m; j++) sum += T1[i * m + j] * to_float(kt[j * D + d]);
        atomicAdd(&dQt_bh[idx], sum);
    }

    // dK_tilde += dS2^T @ Q_tilde
    float* dKt_bh = dK_tilde + bh * m * D;
    for (int idx = tid; idx < m * D; idx += nthreads) {
        int j = idx / D, d = idx % D;
        float sum = 0.0f;
        for (int i = 0; i < m; i++) sum += T1[i * m + j] * to_float(qt[i * D + d]);
        atomicAdd(&dKt_bh[idx], sum);
    }
}

template <typename scalar_t>
void launch_kernel2_inv_bwd(
    const scalar_t* q_tilde, const scalar_t* k_tilde,
    const float* lse2, const float* k2_inv, const float* dK2_inv,
    float* dQ_tilde, float* dK_tilde,
    int BH, int D, int m, int newton_iter, cudaStream_t stream
) {
    (void)newton_iter;  // Not needed for IFT approach

    dim3 grid(BH);
    dim3 block(256);
    size_t smem = (4 * m * m + 8) * sizeof(float);

    if (smem > 48 * 1024) {
        FN_CHECK(smem <= get_max_smem_per_block(), "kernel2_inv_bwd: insufficient smem");
        FN_CUDA_CHECK(cudaFuncSetAttribute(kernel2_inv_bwd_kernel<scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    }

    kernel2_inv_bwd_kernel<scalar_t><<<grid, block, smem, stream>>>(
        q_tilde, k_tilde, lse2, k2_inv, dK2_inv,
        dQ_tilde, dK_tilde, D, m);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
