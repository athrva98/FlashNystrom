/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"

namespace flash_nystrom {

// Straight-through backward for the leverage-seeded Voronoi-mean landmarks
// (leverage_landmarks.cuh). Membership (the argmax assignment) and the cell
// counts are held FIXED at their forward values, so a landmark is a plain mean
// of its cell and the gradient is the transpose of that scatter-mean:
//
//   dQ_s[n,d] += dQ_tilde[c_q(n), d] / cnt_q[c_q(n)]     c_q(n) = q_assign[n]
//   dK_s[n,d] += dK_tilde[c_k(n), d] / cnt_k[c_k(n)]     c_k(n) = k_assign[n]
//
// Q and K carry INDEPENDENT assignments/counts (different Gumbel seeds ->
// different seeds -> different Voronoi cells). Rows with assign == -1 were not
// processed (subsample>1) and contribute no landmark-path gradient. Counts are
// always >= 1 (every seed lands in its own cell), so no divide-by-zero.
//
// Same vectorized layout as landmark_bwd_kernel: 16-byte column slice per
// thread, grid-stride over rows, RMW accumulate onto the kernel1/kernel3
// contributions already in dQ_s/dK_s. No races (each row written once).

template <typename scalar_t>
__global__ void landmark_bwd_voronoi_kernel(
    const float* __restrict__ dQ_tilde,  // (BH, m, D) FP32
    const float* __restrict__ dK_tilde,  // (BH, m, D) FP32
    const int*   __restrict__ q_assign,  // (BH, N) cell id per row, -1 = skip
    const int*   __restrict__ k_assign,  // (BH, N)
    const int*   __restrict__ q_cnt,     // (BH, m) cell counts (>=1)
    const int*   __restrict__ k_cnt,     // (BH, m)
    scalar_t* __restrict__ dQ_s,         // (BH, N, D) accumulated
    scalar_t* __restrict__ dK_s,         // (BH, N, D) accumulated
    int N, int D, int m
) {
    const int64_t bh = blockIdx.y;
    const float* dqt = dQ_tilde + bh * m * D;
    const float* dkt = dK_tilde + bh * m * D;
    const int*   qa  = q_assign + bh * N;
    const int*   ka  = k_assign + bh * N;
    const int*   qc  = q_cnt + bh * m;
    const int*   kc  = k_cnt + bh * m;
    scalar_t* dqs = dQ_s + bh * N * D;
    scalar_t* dks = dK_s + bh * N * D;

    constexpr int kEpt = 16 / (int)sizeof(scalar_t);  // elems per 16B vector
    const int threads_per_row = D / kEpt;
    const int rows_per_block = blockDim.x / threads_per_row;
    const int r_in_block = threadIdx.x / threads_per_row;
    const int d0 = (threadIdx.x % threads_per_row) * kEpt;
    if (r_in_block >= rows_per_block) return;

    const int row_step = gridDim.x * rows_per_block;
    for (int n = blockIdx.x * rows_per_block + r_in_block; n < N; n += row_step) {
        const int cq = qa[n];
        const int ck = ka[n];
        const float inv_q = (cq >= 0) ? 1.0f / static_cast<float>(qc[cq]) : 0.0f;
        const float inv_k = (ck >= 0) ? 1.0f / static_cast<float>(kc[ck]) : 0.0f;
        const float* dq_src = (cq >= 0) ? (dqt + cq * D + d0) : nullptr;
        const float* dk_src = (ck >= 0) ? (dkt + ck * D + d0) : nullptr;

        scalar_t* q = dqs + (int64_t)n * D + d0;
        scalar_t* k = dks + (int64_t)n * D + d0;
        uint4 qv = *reinterpret_cast<const uint4*>(q);
        uint4 kv = *reinterpret_cast<const uint4*>(k);
        auto* qe = reinterpret_cast<scalar_t*>(&qv);
        auto* ke = reinterpret_cast<scalar_t*>(&kv);
        #pragma unroll
        for (int e = 0; e < kEpt; e++) {
            if (cq >= 0) qe[e] = from_float<scalar_t>(to_float(qe[e]) + dq_src[e] * inv_q);
            if (ck >= 0) ke[e] = from_float<scalar_t>(to_float(ke[e]) + dk_src[e] * inv_k);
        }
        *reinterpret_cast<uint4*>(q) = qv;
        *reinterpret_cast<uint4*>(k) = kv;
    }
}

template <typename scalar_t>
void launch_landmark_bwd_voronoi(
    const float* dQ_tilde, const float* dK_tilde,
    const int* q_assign, const int* k_assign,
    const int* q_cnt, const int* k_cnt,
    scalar_t* dQ_s, scalar_t* dK_s,
    int BH, int N, int D, int m, cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int threads_per_row = D / (16 / (int)sizeof(scalar_t));
    const int rows_per_block = max(1, kBlock / max(1, threads_per_row));
    const int64_t blocks = ((int64_t)N + rows_per_block - 1) / rows_per_block;
    dim3 grid((unsigned)min<int64_t>(blocks, 1 << 20), (unsigned)BH);
    landmark_bwd_voronoi_kernel<scalar_t><<<grid, kBlock, 0, stream>>>(
        dQ_tilde, dK_tilde, q_assign, k_assign, q_cnt, k_cnt, dQ_s, dK_s, N, D, m);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
