/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"

namespace flash_nystrom {

// =============================================================================
// dK2_inv = ∂L/∂Z_N — exact, no approximation in NS convergence.
//
// Forward: step2 = Z_N @ B   where Z_N is the NS approximation of K2_inv,
//                            and B = softmax(Q_tilde @ K_s^T) @ V (the true
//                            "kernel3-without-Z_N" output).
// True backward: ∂L/∂Z_N = ∂L/∂step2 @ B^T.
//
// The previous version used dK2_inv = dstep2 @ (K2 @ step2)^T, which only
// equals the true backward when K2 @ Z_N = I (full NS convergence). At
// newton_iter=6 this introduces ~38% relative error in dK2_inv — that error
// then propagates through the entire NS unrolled backward.
//
// This kernel recomputes B exactly from saved q_tilde, k_s, v, lse3:
//   A[i, n] = exp(q_tilde[i] · k_s[n] - lse3[i])         (m × N)
//   B[i, d] = sum_n A[i, n] * V[n, d]                     (m × D)
//   dK2_inv[i, j] = sum_d dstep2[i, d] * B[j, d]          (m × m)
//
// Tiled over N with TILE_N = 32 to fit SMEM.
//
// SMEM (FP32 throughout for accuracy, m ≤ 64, D ∈ {64, 128}, TILE_N = 32):
//   sQ [m*D]        ─ q_tilde, persistent
//   sB [m*D]        ─ B accumulator, persistent
//   sKV [TILE_N*D]  ─ K_tile then V_tile (aliased), per tile
//   sA [m*TILE_N]   ─ A_tile, per tile
//   sLSE [m]        ─ lse3, persistent
// At m=64, D=128:  32 + 32 + 16 + 8 + 0.25 ≈ 88 KB (opt-in required).
// At m=64, D=64:   16 + 16 + 8 + 8 + 0.25 ≈ 48 KB.
// =============================================================================

template <typename scalar_t>
__global__ void compute_dk2inv_kernel(
    const scalar_t* __restrict__ q_tilde,  // (BH, m, D)    elem_type
    const scalar_t* __restrict__ k_s,      // (BH, N, D)    elem_type, scaled K
    const scalar_t* __restrict__ v,        // (BH, N, D)    elem_type
    const float*    __restrict__ lse3,     // (BH, m)        FP32
    const float*    __restrict__ dstep2,   // (BH, m, D)     FP32
    float*          __restrict__ dK2_inv,  // (BH, m, m)     FP32 output
    int N, int D, int m
) {
    constexpr int TILE_N = 32;
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    extern __shared__ float smem[];
    float* sQ   = smem;                             // (m, D)
    float* sB   = sQ   + m * D;                     // (m, D)
    float* sKV  = sB   + m * D;                     // (TILE_N, D) aliased K/V
    float* sA   = sKV  + TILE_N * D;                // (m, TILE_N)
    float* sLSE = sA   + m * TILE_N;                // (m)

    const scalar_t* qt   = q_tilde + bh * m * D;
    const scalar_t* ks   = k_s     + bh * N * D;
    const scalar_t* vv   = v       + bh * N * D;
    const float*    lse_g = lse3    + bh * m;
    const float*    ds2   = dstep2  + bh * m * D;
    float*          dki   = dK2_inv + bh * m * m;

    // Load q_tilde -> sQ (FP32), init sB = 0, load lse3 -> sLSE.
    for (int idx = tid; idx < m * D; idx += nthreads) sQ[idx] = to_float(qt[idx]);
    for (int idx = tid; idx < m * D; idx += nthreads) sB[idx] = 0.0f;
    for (int i = tid; i < m; i += nthreads)            sLSE[i] = lse_g[i];
    __syncthreads();

    // Tile over N
    for (int n0 = 0; n0 < N; n0 += TILE_N) {
        int tile_len = (N - n0 < TILE_N) ? (N - n0) : TILE_N;

        // 1. Load K_tile -> sKV (rows >= tile_len zeroed).
        for (int idx = tid; idx < TILE_N * D; idx += nthreads) {
            int n = idx / D, d = idx % D;
            sKV[idx] = (n < tile_len) ? to_float(ks[(n0 + n) * D + d]) : 0.0f;
        }
        __syncthreads();

        // 2. Compute A_tile[i, n] = exp(q_tilde[i] · K_tile[n] - lse[i]).
        //    For n >= tile_len, set A = 0 so it contributes nothing to B.
        //    For i >= m, no thread reads/writes (m guarded).
        for (int idx = tid; idx < m * TILE_N; idx += nthreads) {
            int i = idx / TILE_N, n = idx % TILE_N;
            if (n < tile_len) {
                float dot = 0.0f;
                for (int d = 0; d < D; d++) dot += sQ[i * D + d] * sKV[n * D + d];
                sA[idx] = expf(dot - sLSE[i]);
            } else {
                sA[idx] = 0.0f;
            }
        }
        __syncthreads();

        // 3. Load V_tile -> sKV (overwrites K).
        for (int idx = tid; idx < TILE_N * D; idx += nthreads) {
            int n = idx / D, d = idx % D;
            sKV[idx] = (n < tile_len) ? to_float(vv[(n0 + n) * D + d]) : 0.0f;
        }
        __syncthreads();

        // 4. sB[i, d] += sum_n A_tile[i, n] * V_tile[n, d]
        for (int idx = tid; idx < m * D; idx += nthreads) {
            int i = idx / D, d = idx % D;
            float acc = 0.0f;
            for (int n = 0; n < TILE_N; n++) acc += sA[i * TILE_N + n] * sKV[n * D + d];
            sB[idx] += acc;
        }
        __syncthreads();
    }

    // dK2_inv[i, j] = sum_d dstep2[i, d] * B[j, d]
    for (int idx = tid; idx < m * m; idx += nthreads) {
        int i = idx / m, j = idx % m;
        float acc = 0.0f;
        for (int d = 0; d < D; d++) acc += ds2[i * D + d] * sB[j * D + d];
        dki[idx] = acc;
    }
}

template <typename scalar_t>
void launch_compute_dk2inv(
    const scalar_t* q_tilde, const scalar_t* k_s, const scalar_t* v,
    const float* lse3, const float* dstep2, float* dK2_inv,
    int BH, int N, int D, int m, cudaStream_t stream
) {
    constexpr int TILE_N = 32;
    dim3 grid(BH);
    dim3 block(256);
    size_t smem = (m * D + m * D + TILE_N * D + m * TILE_N + m) * sizeof(float);
    if (smem > 48 * 1024) {
        FN_CHECK(smem <= get_max_smem_per_block(),
                 "compute_dk2inv: insufficient smem");
        FN_CUDA_CHECK(cudaFuncSetAttribute(compute_dk2inv_kernel<scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    }
    compute_dk2inv_kernel<scalar_t><<<grid, block, smem, stream>>>(
        q_tilde, k_s, v, lse3, dstep2, dK2_inv, N, D, m);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
