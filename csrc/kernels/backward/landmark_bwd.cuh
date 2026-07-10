/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"

namespace flash_nystrom {

// Scatter gradients from landmarks back to sequence positions.
// dQ_s[n,d] += dQ_tilde[l,d] / seg_len   for n in segment l
// Same for dK_s. No races — segments are disjoint.
//
// Parallelized over (n, d) with a grid-stride loop and contiguous
// (coalesced) indexing; the per-bh dQ_tilde/dK_tilde reads (m*D floats)
// stay L2-resident. The previous (BH, m)-grid version looped serially
// over the whole segment per thread, which collapsed to a few thousand
// threads at low BH: 58.8 ms at (1, 4, N=2M, D=64, m=32) on a B200
// against a ~0.5 ms memory roofline. This form measures at roofline.

template <typename scalar_t>
__global__ void landmark_bwd_kernel(
    const float* __restrict__ dQ_tilde,  // (BH, m, D) FP32
    const float* __restrict__ dK_tilde,  // (BH, m, D) FP32
    scalar_t* __restrict__ dQ_s,         // (BH, N, D) accumulated
    scalar_t* __restrict__ dK_s,         // (BH, N, D) accumulated
    int N, int D, int m
) {
    const int64_t bh = blockIdx.y;
    const int seg_len_floor = N / m;

    const float* dqt = dQ_tilde + (int64_t)bh * m * D;
    const float* dkt = dK_tilde + (int64_t)bh * m * D;
    scalar_t* dqs = dQ_s + (int64_t)bh * N * D;
    scalar_t* dks = dK_s + (int64_t)bh * N * D;

    // Fixed 16-byte column slice per thread; stride over rows. Row-major
    // vectorized RMW keeps the kernel at the memory roofline (the flat-index
    // form paid a 64-bit div/mod per element, the scalar row form an RMW
    // per element). D in {64, 128} is always divisible by kEpt.
    constexpr int kEpt = 16 / (int)sizeof(scalar_t);  // elems per 16B vector
    const int threads_per_row = D / kEpt;
    const int rows_per_block = blockDim.x / threads_per_row;
    const int r_in_block = threadIdx.x / threads_per_row;
    const int d0 = (threadIdx.x % threads_per_row) * kEpt;
    if (r_in_block >= rows_per_block) return;

    const int row_step = gridDim.x * rows_per_block;
    for (int n = blockIdx.x * rows_per_block + r_in_block; n < N;
         n += row_step) {
        // Landmark segment containing position n; the last landmark takes
        // the remainder (identical partition to the segment-mean forward).
        const int l = (seg_len_floor == 0)
            ? (m - 1) : min(n / seg_len_floor, m - 1);
        const int seg_start = l * seg_len_floor;
        const int seg_end = (l == m - 1) ? N : (seg_start + seg_len_floor);
        const float inv_len = 1.0f / static_cast<float>(seg_end - seg_start);
        const float* dq_src = dqt + l * D + d0;
        const float* dk_src = dkt + l * D + d0;
        scalar_t* q = dqs + (int64_t)n * D + d0;
        scalar_t* k = dks + (int64_t)n * D + d0;
        uint4 qv = *reinterpret_cast<const uint4*>(q);
        uint4 kv = *reinterpret_cast<const uint4*>(k);
        auto* qe = reinterpret_cast<scalar_t*>(&qv);
        auto* ke = reinterpret_cast<scalar_t*>(&kv);
        // Accumulate: dQ_s already has contributions from kernel1_bwd
        #pragma unroll
        for (int e = 0; e < kEpt; e++) {
            qe[e] = from_float<scalar_t>(to_float(qe[e]) + dq_src[e] * inv_len);
            ke[e] = from_float<scalar_t>(to_float(ke[e]) + dk_src[e] * inv_len);
        }
        *reinterpret_cast<uint4*>(q) = qv;
        *reinterpret_cast<uint4*>(k) = kv;
    }
}

template <typename scalar_t>
void launch_landmark_bwd(
    const float* dQ_tilde, const float* dK_tilde,
    scalar_t* dQ_s, scalar_t* dK_s,
    int BH, int N, int D, int m, cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int threads_per_row = D / (16 / (int)sizeof(scalar_t));
    const int rows_per_block = max(1, kBlock / max(1, threads_per_row));
    const int64_t blocks = ((int64_t)N + rows_per_block - 1) / rows_per_block;
    dim3 grid((unsigned)min<int64_t>(blocks, 1 << 20), (unsigned)BH);
    landmark_bwd_kernel<scalar_t><<<grid, kBlock, 0, stream>>>(
        dQ_tilde, dK_tilde, dQ_s, dK_s, N, D, m);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
