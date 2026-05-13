/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once

#include "utils.h"

namespace flash_nystrom {

// Scalar kernel3 for FP32 fallback
template <typename scalar_t>
__global__ void kernel3_scalar_kernel(
    const scalar_t* __restrict__ q_tilde,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float*    __restrict__ kernel2_inv,
    scalar_t*       __restrict__ step2_out,
    scalar_t*       __restrict__ b_out,       // (B*H, m, D) or nullptr
    float*          __restrict__ softmax3_lse_out,
    int N, int D, int m
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    constexpr int Bc = 64;

    extern __shared__ char smem_raw[];
    scalar_t* Q_tilde_s = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* K_tile_s  = Q_tilde_s + m * D;
    scalar_t* V_tile_s  = K_tile_s + Bc * D;
    float* S_tile       = reinterpret_cast<float*>(V_tile_s + Bc * D);
    float* O_acc        = S_tile + m * Bc;
    float* row_max_s    = O_acc + m * D;
    float* row_sum_s    = row_max_s + m;
    float* scratch      = row_sum_s + m;

    // Load Q_tilde
    const scalar_t* qt_ptr = q_tilde + bh * m * D;
    for (int idx = tid; idx < m * D; idx += nthreads) Q_tilde_s[idx] = qt_ptr[idx];
    for (int idx = tid; idx < m * D; idx += nthreads) O_acc[idx] = 0.0f;
    for (int idx = tid; idx < m; idx += nthreads) { row_max_s[idx] = -FLT_MAX; row_sum_s[idx] = 0.0f; }
    __syncthreads();

    const scalar_t* k_bh = k + bh * N * D;
    const scalar_t* v_bh = v + bh * N * D;
    int num_tiles = (N + Bc - 1) / Bc;

    for (int tile = 0; tile < num_tiles; tile++) {
        int ts = tile * Bc, te = min(ts + Bc, N), tl = te - ts;
        for (int idx = tid; idx < tl * D; idx += nthreads) { K_tile_s[idx] = k_bh[ts * D + idx]; V_tile_s[idx] = v_bh[ts * D + idx]; }
        __syncthreads();

        for (int idx = tid; idx < m * tl; idx += nthreads) {
            int r = idx / tl, c = idx % tl;
            float acc = 0.0f;
            for (int d = 0; d < D; d++) acc += to_float(Q_tilde_s[r * D + d]) * to_float(K_tile_s[c * D + d]);
            S_tile[r * Bc + c] = acc;
        }
        __syncthreads();

        for (int row = 0; row < m; row++) {
            float lm = -FLT_MAX;
            for (int j = tid; j < tl; j += nthreads) lm = fmaxf(lm, S_tile[row * Bc + j]);
            float tm = block_reduce_max(lm, scratch);
            float nm = fmaxf(row_max_s[row], tm);
            float a = expf(row_max_s[row] - nm);
            float ls = 0.0f;
            for (int j = tid; j < tl; j += nthreads) {
                float p = expf(S_tile[row * Bc + j] - nm);
                S_tile[row * Bc + j] = p;
                ls += p;
            }
            float ts_val = block_reduce_sum(ls, scratch);
            if (tid == 0) { row_sum_s[row] = a * row_sum_s[row] + ts_val; row_max_s[row] = nm; }
            __syncthreads();

            for (int d = tid; d < D; d += nthreads) {
                float pv = 0.0f;
                for (int j = 0; j < tl; j++) pv += S_tile[row * Bc + j] * to_float(V_tile_s[j * D + d]);
                O_acc[row * D + d] = a * O_acc[row * D + d] + pv;
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < m * D; idx += nthreads) {
        int r = idx / D;
        O_acc[idx] *= (row_sum_s[r] > 0.0f) ? (1.0f / row_sum_s[r]) : 0.0f;
    }
    __syncthreads();

    // Side output: write B = O_acc to GMEM if requested (saved for the bwd
    // compute_dk2inv so it does not need to N-walk to recompute B).
    if (b_out != nullptr) {
        scalar_t* b_bh = b_out + bh * m * D;
        for (int idx = tid; idx < m * D; idx += nthreads)
            b_bh[idx] = from_float<scalar_t>(O_acc[idx]);
    }

    // K2_inv @ O_acc
    const float* k2inv = kernel2_inv + bh * m * m;
    scalar_t* step2 = step2_out + bh * m * D;
    for (int idx = tid; idx < m * D; idx += nthreads) {
        int r = idx / D, d = idx % D;
        float acc = 0.0f;
        for (int j = 0; j < m; j++) acc += k2inv[r * m + j] * O_acc[j * D + d];
        step2[idx] = from_float<scalar_t>(acc);
    }

    float* lse = softmax3_lse_out + bh * m;
    for (int i = tid; i < m; i += nthreads)
        lse[i] = row_max_s[i] + logf(row_sum_s[i] + 1e-12f);
}

template <typename scalar_t>
void launch_kernel3_scalar(
    const scalar_t* q_tilde, const scalar_t* k, const scalar_t* v,
    const float* kernel2_inv, scalar_t* step2,
    scalar_t* b_out,                          // (BH, m, D) or nullptr
    float* softmax3_lse,
    int BH, int N, int D, int m, cudaStream_t stream
) {
    constexpr int Bc = 64;
    dim3 grid(BH); dim3 block(256);
    size_t smem = m * D * sizeof(scalar_t) + Bc * D * sizeof(scalar_t) * 2
                + m * Bc * sizeof(float) + m * D * sizeof(float) + 2 * m * sizeof(float) + 8 * sizeof(float);
    if (smem > 48 * 1024) {
        FN_CHECK(smem <= get_max_smem_per_block(), "kernel3_scalar: insufficient smem");
        FN_CUDA_CHECK(cudaFuncSetAttribute(kernel3_scalar_kernel<scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    }
    kernel3_scalar_kernel<scalar_t><<<grid, block, smem, stream>>>(
        q_tilde, k, v, kernel2_inv, step2, b_out, softmax3_lse, N, D, m);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
