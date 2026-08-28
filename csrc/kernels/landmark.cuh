/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once

#include "utils.h"

namespace flash_nystrom {

// landmark kernel: segment-mean over Q and K to produce the nystrom landmarks
// pretty simple — each landmark is just the average of its segment
//
// For each landmark l in [0, m):
//   segment = [l * seg_len, min((l+1) * seg_len, N))
//   Q_tilde[b,h,l,:] = mean(Q[b,h,segment,:]) * scale
//   K_tilde[b,h,l,:] = mean(K[b,h,segment,:]) * scale
//
// Grid:  (B*H, m)        — one CTA per (batch-head, landmark)
// Block: (tpd * D)       — tpd = 1024/D threads cooperate per output column d
//                          (so each landmark's segment is reduced by tpd*D
//                           threads, not one thread per d serially; see below)
//
// SMEM: 2 * tpd * D floats — scratch to reduce the tpd per-column partials
//                            (Q half then K half)

// Each landmark is the scaled segment-mean of Q (and K). The reduction over a
// segment of N/m rows is split across `tpd = blockDim.x / D` threads per output
// column d, so one landmark's segment is reduced by tpd*D threads instead of a
// single thread per d serially walking the whole segment. This matters at low
// batch*head with large N: there the grid has only BH*m CTAs and each segment
// is huge, so the old one-thread-per-d design was latency-bound (~13 ms at
// BH=4, N=2M). Splitting the segment makes it bandwidth-bound (sub-ms).
template <typename scalar_t>
__global__ void landmark_kernel(
    const scalar_t* __restrict__ q,   // (B*H, N, D) row-major
    const scalar_t* __restrict__ k,   // (B*H, N, D)
    scalar_t* __restrict__ q_tilde,   // (B*H, m, D)
    scalar_t* __restrict__ k_tilde,   // (B*H, m, D)
    int N, int D, int m, float scale
) {
    const int64_t bh = blockIdx.x;
    const int landmark = blockIdx.y;

    // Floor-division segments; the last landmark absorbs the remainder.
    const int seg_len_floor = N / m;
    const int seg_start = landmark * seg_len_floor;
    const int seg_end = (landmark == m - 1) ? N : (seg_start + seg_len_floor);
    const int seg_len = seg_end - seg_start;
    if (seg_len <= 0) return;
    const float inv_len = 1.0f / static_cast<float>(seg_len);

    const scalar_t* q_bh = q + static_cast<size_t>(bh) * N * D;
    const scalar_t* k_bh = k + static_cast<size_t>(bh) * N * D;
    scalar_t* qt_out = q_tilde + (static_cast<size_t>(bh) * m + landmark) * D;
    scalar_t* kt_out = k_tilde + (static_cast<size_t>(bh) * m + landmark) * D;

    // VECTORIZED over the head dim: each thread owns VEC consecutive columns
    // and loads them as one 16-byte access.
    //
    // The scalar version issued one 2-byte load per thread per row. That is
    // perfectly coalesced (two warps cover a 128-byte line) but it moves only
    // 64 bytes per warp-instruction, and this kernel is limited by load-issue
    // rate rather than by wasted bandwidth: it measured 3.69 passes over Q and
    // K where the algorithm needs 2. A 16-byte load moves 512 bytes per
    // warp-instruction, so the same issue rate carries 8x the traffic.
    constexpr int VEC = 16 / sizeof(scalar_t);   // 8 fp16/bf16, 4 fp32
    const int nvec = D / VEC;                    // vector-columns per row
    const int tpd  = blockDim.x / nvec;          // threads cooperating per column
    const int vc   = threadIdx.x % nvec;
    const int grp  = threadIdx.x / nvec;         // 0 .. tpd-1

    float qacc[VEC], kacc[VEC];
    #pragma unroll
    for (int j = 0; j < VEC; j++) { qacc[j] = 0.0f; kacc[j] = 0.0f; }

    for (int i = grp; i < seg_len; i += tpd) {
        const int n = seg_start + i;
        const uint4 qv = *reinterpret_cast<const uint4*>(q_bh + (size_t)n * D + vc * VEC);
        const uint4 kv = *reinterpret_cast<const uint4*>(k_bh + (size_t)n * D + vc * VEC);
        const scalar_t* qe = reinterpret_cast<const scalar_t*>(&qv);
        const scalar_t* ke = reinterpret_cast<const scalar_t*>(&kv);
        #pragma unroll
        for (int j = 0; j < VEC; j++) {
            qacc[j] += to_float(qe[j]);
            kacc[j] += to_float(ke[j]);
        }
    }

    // Reduce the tpd partials per column through SMEM.
    extern __shared__ float smem_lm[];        // [tpd*D] for Q, then [tpd*D] for K
    float* sq = smem_lm;
    float* sk = smem_lm + static_cast<size_t>(tpd) * D;
    #pragma unroll
    for (int j = 0; j < VEC; j++) {
        sq[grp * D + vc * VEC + j] = qacc[j];
        sk[grp * D + vc * VEC + j] = kacc[j];
    }
    __syncthreads();

    if (grp == 0) {
        #pragma unroll
        for (int j = 0; j < VEC; j++) {
            const int d = vc * VEC + j;
            float qs = 0.0f, ks = 0.0f;
            for (int g = 0; g < tpd; g++) {
                qs += sq[g * D + d];
                ks += sk[g * D + d];
            }
            qt_out[d] = from_float<scalar_t>(qs * inv_len * scale);
            kt_out[d] = from_float<scalar_t>(ks * inv_len * scale);
        }
    }
}

// Scaled copy: dst[i] = src[i] * scale, for the full flattened tensor.
//
// This replaces the old "clone then scale_inplace" two-pass sequence. The
// backward needs the SCALED q/k saved, and the forward already had to clone
// q/k (to avoid mutating the user's inputs) -- so folding the scalar multiply
// into that clone removes an entire redundant read+write of Q and K. At high
// batch*head that pass was ~44% of the whole forward (a separate in-place
// scale runs at ~0.87 TB/s, far below peak), so eliminating it is the single
// biggest forward win.
//
// Vectorized: each thread moves 16 bytes (a uint4 = 8 fp16/bf16 or 4 fp32) so
// the copy saturates HBM bandwidth instead of issuing 2-byte transactions.
// src == dst is allowed (in-place scale): each thread owns disjoint elements,
// so there is no aliasing hazard. Grid-stride so a capped grid is correct.
template <typename scalar_t>
__global__ void scaled_copy_kernel(
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ dst,
    int total, float scale
) {
    constexpr int VEC = static_cast<int>(16 / sizeof(scalar_t));  // 8 fp16/bf16, 4 fp32
    const int nvec = total / VEC;
    const int stride = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int v = tid; v < nvec; v += stride) {
        // Load through a uint4 local (16-byte aligned by type) so the
        // reinterpret to scalar_t* is properly aligned for element access.
        uint4 vec = *reinterpret_cast<const uint4*>(src + v * VEC);
        scalar_t* buf = reinterpret_cast<scalar_t*>(&vec);
        #pragma unroll
        for (int j = 0; j < VEC; j++)
            buf[j] = from_float<scalar_t>(to_float(buf[j]) * scale);
        *reinterpret_cast<uint4*>(dst + v * VEC) = vec;
    }
    // Scalar tail (total is divisible by D>=64 hence by VEC in all supported
    // shapes; this guard is defensive against future shapes).
    for (int i = nvec * VEC + tid; i < total; i += stride)
        dst[i] = from_float<scalar_t>(to_float(src[i]) * scale);
}

// -- launch wrapper --


template <typename scalar_t>
void launch_landmarks(
    const scalar_t* q, const scalar_t* k,
    scalar_t* q_tilde, scalar_t* k_tilde,
    int BH, int N, int D, int m, float scale,
    cudaStream_t stream
) {
    FN_CHECK(BH > 0 && N > 0 && D > 0 && m > 0, "launch_landmarks: invalid dims");
    FN_CHECK(m <= N, "launch_landmarks: m > N");

    // tpd threads per output column, block capped at 1024. D is 64 or 128, both
    // divide 1024, so block.x is an exact multiple of D (tpd = 16 or 8). More
    // threads per landmark = the long segment reduction is split, not serial.
    // The kernel is vectorized 16 bytes per thread, so a row is covered by
    // D/VEC threads, not D. Block 256 keeps the SMEM reduction buffer at
    // 2*tpd*D floats = 16 KB while still giving the segment loop enough
    // concurrent groups; grid is (BH, m) so the GPU is filled by blocks.
    constexpr int kVec = 16 / sizeof(scalar_t);
    FN_CHECK(D % kVec == 0, "launch_landmarks: head_dim must be a multiple of "
                            "the 16-byte vector width");
    const int nvec = D / kVec;
    const int block = 256;
    int tpd = block / nvec;
    if (tpd < 1) tpd = 1;
    dim3 grid(BH, m);
    const size_t smem = static_cast<size_t>(2) * tpd * D * sizeof(float);

    landmark_kernel<scalar_t><<<grid, block, smem, stream>>>(
        q, k, q_tilde, k_tilde, N, D, m, scale);
    FN_CUDA_KERNEL_CHECK();
}

// dst[i] = src[i] * scale over `total` elements. src == dst is allowed.
template <typename scalar_t>
void launch_scaled_copy(
    const scalar_t* src, scalar_t* dst, int total, float scale,
    cudaStream_t stream
) {
    if (total <= 0) return;
    constexpr int VEC = static_cast<int>(16 / sizeof(scalar_t));
    const int threads = 256;
    const int nvec = total / VEC;
    // Cap the grid; the kernel is grid-stride so any cap stays correct.
    const int kMaxBlocks = 65535;
    int blocks = (nvec + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    if (blocks > kMaxBlocks) blocks = kMaxBlocks;
    scaled_copy_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        src, dst, total, scale);
    FN_CUDA_KERNEL_CHECK();
}

// In-place scale: data[i] *= scale. Thin wrapper over the vectorized scaled
// copy (still used by the backward to scale dQ/dK).
template <typename scalar_t>
void launch_scale_inplace(
    scalar_t* data, int total_elements, float scale, cudaStream_t stream
) {
    launch_scaled_copy<scalar_t>(data, data, total_elements, scale, stream);
}

} // namespace flash_nystrom
