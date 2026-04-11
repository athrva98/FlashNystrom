/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"

namespace flash_nystrom {

// compute dK2_inv in FP32 to avoid the ~100x error amplification
// from the IFT backward. this was causing dK to be garbage when done in fp16.
// O3 = K2 @ step2, where K2 = softmax(Qt @ Kt^T) recomputed from LSE2.
//
// This avoids the tiled FP16 accumulation in kernel3_bwd that gets amplified
// 100x+ by the IFT backward through K2_inv.
//
// Grid: (BH), Block: 256

template <typename scalar_t>
__global__ void compute_dk2inv_kernel(
    const scalar_t* __restrict__ q_tilde,  // (BH, m, D)
    const scalar_t* __restrict__ k_tilde,  // (BH, m, D)
    const scalar_t* __restrict__ step2,    // (BH, m, D)
    const float*    __restrict__ lse2,     // (BH, m)
    const float*    __restrict__ dstep2,   // (BH, m, D) FP32
    float*          __restrict__ dK2_inv,  // (BH, m, m) FP32 output
    int D, int m
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    extern __shared__ float smem[];
    // K2: m*m, O3: m*D (all FP32)
    float* K2 = smem;
    float* O3 = smem + m * m;

    const scalar_t* qt = q_tilde + bh * m * D;
    const scalar_t* kt = k_tilde + bh * m * D;
    const scalar_t* s2 = step2 + bh * m * D;
    const float* lse = lse2 + bh * m;
    const float* ds2 = dstep2 + bh * m * D;

    // Recompute K2 = softmax(Qt @ Kt^T) from LSE2 (all FP32)
    for (int idx = tid; idx < m * m; idx += nthreads) {
        int i = idx / m, j = idx % m;
        float dot = 0.0f;
        for (int d = 0; d < D; d++)
            dot += to_float(qt[i * D + d]) * to_float(kt[j * D + d]);
        K2[idx] = expf(dot - lse[i]);
    }
    __syncthreads();

    // O3 = K2 @ step2 (m×m times m×D = m×D, FP32)
    for (int idx = tid; idx < m * D; idx += nthreads) {
        int i = idx / D, d = idx % D;
        float sum = 0.0f;
        for (int j = 0; j < m; j++)
            sum += K2[i * m + j] * to_float(s2[j * D + d]);
        O3[idx] = sum;
    }
    __syncthreads();

    // dK2_inv = dstep2 @ O3^T (m×D times D×m = m×m, FP32)
    float* dk2i = dK2_inv + bh * m * m;
    for (int idx = tid; idx < m * m; idx += nthreads) {
        int i = idx / m, j = idx % m;
        float sum = 0.0f;
        for (int d = 0; d < D; d++)
            sum += ds2[i * D + d] * O3[j * D + d];
        dk2i[idx] = sum;
    }
}

template <typename scalar_t>
void launch_compute_dk2inv(
    const scalar_t* q_tilde, const scalar_t* k_tilde, const scalar_t* step2,
    const float* lse2, const float* dstep2, float* dK2_inv,
    int BH, int D, int m, cudaStream_t stream
) {
    dim3 grid(BH);
    dim3 block(256);
    size_t smem = (m * m + m * D) * sizeof(float);
    if (smem > 48 * 1024) {
        FN_CHECK(smem <= get_max_smem_per_block(), "compute_dk2inv: insufficient smem");
        FN_CUDA_CHECK(cudaFuncSetAttribute(compute_dk2inv_kernel<scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    }
    compute_dk2inv_kernel<scalar_t><<<grid, block, smem, stream>>>(
        q_tilde, k_tilde, step2, lse2, dstep2, dK2_inv, D, m);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
