/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once

#include "utils.h"

namespace flash_nystrom {

// kernel2_inv: K2 = softmax(Q_tilde @ K_tilde^T), then Newton-Schulz pseudoinverse.
//
// Forward Newton-Schulz iteration (third-order Higham polynomial):
//
//     M_j = K2 @ Z_j
//     Z_{j+1} = (1/4) * Z_j * (13*I - M_j*(15*I - M_j*(7*I - M_j)))
//             = (1/4) * Z_j * (13*I - 15*M_j + 7*M_j^2 - M_j^3)
//
// Initialization: Z_0 = K2^T / (||K2||_1 * ||K2||_inf).
//
// To enable the unrolled backward, ALL iterates Z_0, Z_1, ..., Z_J are
// written to ns_iterates_out (BH, J+1, m, m), where J = newton_iter. The
// forward output K2_inv = Z_J is also written to kernel2_inv_out separately
// for kernel3.
//
// Iteration count is `newton_iter` (default 6, capped at 20 by binding).
// For row-stochastic K2 with spread eigenvalues, 6 iters has residual O(1).
// The backward consistency does NOT require convergence — autograd-through-NS
// gives correct gradients of the actual approximation regardless.

template <typename scalar_t>
__global__ void kernel2_inv_kernel(
    const scalar_t* __restrict__ q_tilde,    // (B*H, m, D)
    const scalar_t* __restrict__ k_tilde,    // (B*H, m, D)
    float* __restrict__ kernel2_inv_out,      // (B*H, m, m) FP32 — final K2_inv = Z_J
    float* __restrict__ softmax_lse_out,      // (B*H, m)
    float* __restrict__ ns_iterates_out,      // (B*H, newton_iter+1, m, m) FP32 — Z_0..Z_J
    float* __restrict__ k2_softmax_out,       // (B*H, m, m) FP32 — the softmax K2 (for backward)
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

    // Step 1: K2_logits = Q_tilde @ K_tilde^T (FP32 accumulation)
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

    // Step 2: Row-wise softmax on K2, save LSE
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

    // Step 2.5: write softmax K2 to GMEM for the backward kernel.
    // The SMEM K2 will be overwritten by Z_0 below; preserve a copy.
    if (k2_softmax_out != nullptr) {
        float* k2sm_bh = k2_softmax_out + bh * mm;
        for (int idx = tid; idx < mm; idx += nthreads) k2sm_bh[idx] = K2[idx];
        __syncthreads();
    }

    // Step 3: Z_0 = K2^T / (||K2||_1 * ||K2||_inf).
    // Both norms are computed and used as a piecewise-constant scalar — the
    // gradient through their max() ops is handled in the backward final kernel.
    for (int j = tid; j < m; j += nthreads) {
        float col_sum = 0.0f;
        for (int i = 0; i < m; i++) col_sum += K2[i * m + j];   // K2 >= 0 (softmax)
        T1[j] = col_sum;
    }
    __syncthreads();

    float local_col_max = -FLT_MAX;
    for (int j = tid; j < m; j += nthreads) {
        local_col_max = fmaxf(local_col_max, T1[j]);
    }
    float norm1 = block_reduce_max(local_col_max, scratch);

    // ||K2||_inf = max row sum. For a row-stochastic K2 this is ~1 but we
    // compute it explicitly so the scaling matches the reference exactly.
    float local_row_max = -FLT_MAX;
    for (int i = tid; i < m; i += nthreads) {
        float row_sum = 0.0f;
        for (int j = 0; j < m; j++) row_sum += K2[i * m + j];
        local_row_max = fmaxf(local_row_max, row_sum);
    }
    float norm_inf = block_reduce_max(local_row_max, scratch);

    float inv_c = 1.0f / fmaxf(norm1 * norm_inf, 1e-12f);

    for (int idx = tid; idx < mm; idx += nthreads) {
        int row = idx / m;
        int col = idx % m;
        Z[idx] = K2[col * m + row] * inv_c;  // transpose
    }
    __syncthreads();

    // Save Z_0 as the first iterate (index 0 in ns_iterates_out)
    if (ns_iterates_out != nullptr) {
        float* iter_out = ns_iterates_out + bh * (newton_iter + 1) * mm;
        for (int idx = tid; idx < mm; idx += nthreads) iter_out[idx] = Z[idx];
        __syncthreads();
    }

    // Step 4: Newton-Schulz iterations (third order Higham)
    // Z_{j+1} = 0.25 * Z_j * (13I - K2*Z_j * (15I - K2*Z_j * (7I - K2*Z_j)))
    //
    // Save each iterate Z_{j+1} to GMEM (index j+1).
    for (int iter = 0; iter < newton_iter; iter++) {
        // Save Zold = Z (we'll need it for the final 0.25 * Z * (...) multiply)
        for (int idx = tid; idx < mm; idx += nthreads) Zold[idx] = Z[idx];
        __syncthreads();

        // T1 = K2 @ Zold = M
        for (int idx = tid; idx < mm; idx += nthreads) {
            int row = idx / m, col = idx % m;
            float acc = 0.0f;
            for (int kk = 0; kk < m; kk++) acc += K2[row * m + kk] * Zold[kk * m + col];
            T1[idx] = acc;
        }
        __syncthreads();

        // T2 = 7I - M
        for (int idx = tid; idx < mm; idx += nthreads) {
            int row = idx / m, col = idx % m;
            T2[idx] = ((row == col) ? 7.0f : 0.0f) - T1[idx];
        }
        __syncthreads();

        // Z = M @ T2 = M*(7I - M)
        for (int idx = tid; idx < mm; idx += nthreads) {
            int row = idx / m, col = idx % m;
            float acc = 0.0f;
            for (int kk = 0; kk < m; kk++) acc += T1[row * m + kk] * T2[kk * m + col];
            Z[idx] = acc;
        }
        __syncthreads();

        // T2 = 15I - Z = 15I - M*(7I - M)
        for (int idx = tid; idx < mm; idx += nthreads) {
            int row = idx / m, col = idx % m;
            T2[idx] = ((row == col) ? 15.0f : 0.0f) - Z[idx];
        }
        __syncthreads();

        // Z = M @ T2 = M*(15I - M*(7I - M))
        for (int idx = tid; idx < mm; idx += nthreads) {
            int row = idx / m, col = idx % m;
            float acc = 0.0f;
            for (int kk = 0; kk < m; kk++) acc += T1[row * m + kk] * T2[kk * m + col];
            Z[idx] = acc;
        }
        __syncthreads();

        // T2 = 13I - Z = 13I - M*(15I - M*(7I - M))
        for (int idx = tid; idx < mm; idx += nthreads) {
            int row = idx / m, col = idx % m;
            T2[idx] = ((row == col) ? 13.0f : 0.0f) - Z[idx];
        }
        __syncthreads();

        // Z = 0.25 * Zold @ T2 = (1/4) * Z_old * (13I - M*(15I - M*(7I - M)))
        for (int idx = tid; idx < mm; idx += nthreads) {
            int row = idx / m, col = idx % m;
            float acc = 0.0f;
            for (int kk = 0; kk < m; kk++) acc += Zold[row * m + kk] * T2[kk * m + col];
            Z[idx] = 0.25f * acc;
        }
        __syncthreads();

        // Save Z_{iter+1} to GMEM (index iter+1)
        if (ns_iterates_out != nullptr) {
            float* iter_out = ns_iterates_out + bh * (newton_iter + 1) * mm + (iter + 1) * mm;
            for (int idx = tid; idx < mm; idx += nthreads) iter_out[idx] = Z[idx];
            __syncthreads();
        }
    }

    // Step 5: Write final K2_inv (= Z_J) for kernel3
    float* out = kernel2_inv_out + bh * mm;
    for (int idx = tid; idx < mm; idx += nthreads) {
        out[idx] = Z[idx];
    }
}

template <typename scalar_t>
void launch_kernel2_inv(
    const scalar_t* q_tilde, const scalar_t* k_tilde,
    float* kernel2_inv, float* softmax_lse,
    float* ns_iterates,        // (BH, newton_iter+1, m, m) — REQUIRED for backward
    float* k2_softmax,         // (BH, m, m) — the softmax K2 (REQUIRED for backward)
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
        q_tilde, k_tilde, kernel2_inv, softmax_lse, ns_iterates, k2_softmax,
        D, m, newton_iter);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
