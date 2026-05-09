/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"

namespace flash_nystrom {

// =============================================================================
// dK2_inv = ∂L/∂Z_N    AND    D3[i] = sum_n A3[i, n] * (dO3[i, :] · V[n, :])
//
// Both outputs share the same B = softmax(Q_tilde @ K_s^T) @ V (the kernel3
// inner product before the K2_inv multiply). The kernel walks N once,
// accumulates B in SMEM, and emits both gradient pieces:
//
//   dK2_inv[i, j] = sum_d dstep2[i, d] * B[j, d]            (m × m output)
//   D3[i]         = sum_d B[i, d] * dO3[i, d]              (m vector output)
//
// The D3 identity comes from algebra:
//   D3[i] = sum_n A[i, n] * (dO3[i, :] · V[n, :])
//         = sum_n A[i, n] * sum_d dO3[i, d] * V[n, d]
//         = sum_d dO3[i, d] * sum_n A[i, n] * V[n, d]
//         = sum_d dO3[i, d] * B[i, d]
// so D3 is a free byproduct of B — one m·D pass over GMEM dO3 after the tile
// loop completes. This replaces the standalone precompute_d3 kernel that
// re-walked N for the same information.
//
// Tiled over N with TILE_N = 32 to fit SMEM:
//   sQ  [m·D]        ─ q_tilde, persistent
//   sB  [m·D]        ─ B accumulator, persistent (then read for dK2_inv & D3)
//   sKV [TILE_N·D]   ─ K_tile then V_tile (aliased), per tile
//   sA  [m·TILE_N]   ─ A_tile, per tile
//   sLSE [m]         ─ lse3, persistent
// At m=64, D=128:  32 + 32 + 16 + 8 + 0.25  ≈ 88 KB (opt-in required).
// At m=64, D=64:   16 + 16 +  8 + 8 + 0.25  ≈ 48 KB.
// =============================================================================

template <typename scalar_t>
__global__ void compute_dk2inv_kernel(
    const scalar_t* __restrict__ q_tilde,  // (BH, m, D)
    const scalar_t* __restrict__ k_s,      // (BH, N, D)  scaled K
    const scalar_t* __restrict__ v,        // (BH, N, D)
    const scalar_t* __restrict__ dO3,      // (BH, m, D)  precomputed K2_inv^T @ dstep2
    const float*    __restrict__ lse3,     // (BH, m)
    const float*    __restrict__ dstep2,   // (BH, m, D)
    float*          __restrict__ dK2_inv,  // (BH, m, m)  output
    float*          __restrict__ D3,       // (BH, m)     output
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

    const scalar_t* qt    = q_tilde + bh * m * D;
    const scalar_t* ks    = k_s     + bh * N * D;
    const scalar_t* vv    = v       + bh * N * D;
    const scalar_t* dO3_b = dO3     + bh * m * D;
    const float*    lse_g = lse3    + bh * m;
    const float*    ds2   = dstep2  + bh * m * D;
    float*          dki   = dK2_inv + bh * m * m;
    float*          d3_b  = D3      + bh * m;

    // Load q_tilde -> sQ (FP32), init sB = 0, load lse3 -> sLSE.
    for (int idx = tid; idx < m * D; idx += nthreads) sQ[idx] = to_float(qt[idx]);
    for (int idx = tid; idx < m * D; idx += nthreads) sB[idx] = 0.0f;
    for (int i = tid; i < m; i += nthreads)            sLSE[i] = lse_g[i];
    __syncthreads();

    // Tile over N — accumulate B[i, d] += sum_{n in tile} A[i, n] * V[n, d].
    for (int n0 = 0; n0 < N; n0 += TILE_N) {
        int tile_len = (N - n0 < TILE_N) ? (N - n0) : TILE_N;

        // 1. Load K_tile -> sKV (rows >= tile_len zeroed).
        for (int idx = tid; idx < TILE_N * D; idx += nthreads) {
            int n = idx / D, d = idx % D;
            sKV[idx] = (n < tile_len) ? to_float(ks[(n0 + n) * D + d]) : 0.0f;
        }
        __syncthreads();

        // 2. A_tile[i, n] = exp(q_tilde[i] · K_tile[n] - lse[i]).  Padded n -> 0.
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

    // D3[i] = sum_d B[i, d] * dO3[i, d]   — one m·D pass over GMEM dO3.
    // Each thread handles a stride of i values; the inner D loop reads B
    // from SMEM (cached) and dO3 from GMEM.
    for (int i = tid; i < m; i += nthreads) {
        float acc = 0.0f;
        const scalar_t* dO3_i = dO3_b + i * D;
        for (int d = 0; d < D; d++) acc += sB[i * D + d] * to_float(dO3_i[d]);
        d3_b[i] = acc;
    }
}

template <typename scalar_t>
void launch_compute_dk2inv(
    const scalar_t* q_tilde, const scalar_t* k_s, const scalar_t* v,
    const scalar_t* dO3,
    const float* lse3, const float* dstep2,
    float* dK2_inv, float* D3,
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
        q_tilde, k_s, v, dO3, lse3, dstep2, dK2_inv, D3, N, D, m);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
