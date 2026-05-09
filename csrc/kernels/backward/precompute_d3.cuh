/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"

namespace flash_nystrom {

// =============================================================================
// D3[bh, i] = sum_n A3[i, n] * dP3[i, n]
//          = sum_n exp(q_tilde[i] · k_s[n] - lse3[i]) * (dO3[i, :] · V[n, :])
//
// This is the global rowsum needed by the kernel3 softmax-backward stage:
//   dS3[i, j] = A3[i, j] * (dP3[i, j] - D3[i])
//
// Computing D3 inside kernel3_bwd's tile loop produces a *per-tile* rowsum,
// which equals the global rowsum only when N <= Bc (one tile). For N that
// spans multiple tiles, the per-tile D3 is wrong and the resulting dS3
// gradients are biased — visible as ~2% relerr in dQ/dK at FP32 for N=65,
// growing as N grows.
//
// This kernel computes D3 once, before kernel3_bwd, with a global rowsum.
//
// Grid:  (BH, m)  — one CTA per (batch-head, landmark row i)
// Block: 256
// SMEM:  2*D + 32 floats  (q_tilde[i] + dO3[i] + reduction scratch)
// =============================================================================

template <typename scalar_t>
__global__ void precompute_d3_kernel(
    const scalar_t* __restrict__ q_tilde,  // (BH, m, D)
    const scalar_t* __restrict__ k_s,       // (BH, N, D)
    const scalar_t* __restrict__ v,         // (BH, N, D)
    const scalar_t* __restrict__ dO3,       // (BH, m, D)
    const float*    __restrict__ lse3,      // (BH, m)
    float*          __restrict__ D3,        // (BH, m) — output
    int N, int D, int m
) {
    const int bh = blockIdx.x;
    const int i  = blockIdx.y;
    if (i >= m) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    extern __shared__ float smem[];
    float* sQ      = smem;          // (D,)
    float* sdO3    = sQ + D;        // (D,)
    float* scratch = sdO3 + D;      // (>= 32) for block_reduce_sum

    const scalar_t* qt_i  = q_tilde + (bh * m + i) * D;
    const scalar_t* dO3_i = dO3     + (bh * m + i) * D;
    const float lse_i     = lse3[bh * m + i];

    for (int d = tid; d < D; d += nthreads) {
        sQ[d]   = to_float(qt_i[d]);
        sdO3[d] = to_float(dO3_i[d]);
    }
    __syncthreads();

    const scalar_t* ks_bh = k_s + bh * N * D;
    const scalar_t* v_bh  = v   + bh * N * D;

    // Each thread accumulates partial sum over a stride of n.
    float partial = 0.0f;
    for (int n = tid; n < N; n += nthreads) {
        const scalar_t* k_row = ks_bh + n * D;
        const scalar_t* v_row = v_bh  + n * D;
        float dot_qk = 0.0f;
        float dot_dV = 0.0f;
        for (int d = 0; d < D; d++) {
            dot_qk += sQ[d]   * to_float(k_row[d]);
            dot_dV += sdO3[d] * to_float(v_row[d]);
        }
        partial += expf(dot_qk - lse_i) * dot_dV;
    }

    float total = block_reduce_sum(partial, scratch);
    if (tid == 0) D3[bh * m + i] = total;
}

template <typename scalar_t>
void launch_precompute_d3(
    const scalar_t* q_tilde, const scalar_t* k_s, const scalar_t* v,
    const scalar_t* dO3, const float* lse3,
    float* D3,
    int BH, int N, int D, int m, cudaStream_t stream
) {
    dim3 grid(BH, m);
    dim3 block(256);
    size_t smem = (2 * D + 32) * sizeof(float);
    precompute_d3_kernel<scalar_t><<<grid, block, smem, stream>>>(
        q_tilde, k_s, v, dO3, lse3, D3, N, D, m);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
