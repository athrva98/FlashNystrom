/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"

namespace flash_nystrom {

// Compute dO3 = K2_inv^T @ dstep2, output as FP16/BF16 to GMEM.
//
// Extracted from kernel3_bwd so the multi-CTA TC kernel can load dO3
// as a compact FP16 tensor instead of burning 32KB of SMEM on FP32 dO3.
//
// Grid: (BH), Block: 256
// Each thread computes one or more elements of the (m, D) output.

template <typename scalar_t>
__global__ void compute_dO3_kernel(
    const float*    __restrict__ k2_inv,   // (BH, m, m) FP32
    const float*    __restrict__ dstep2,   // (BH, m, D) FP32
    scalar_t*       __restrict__ dO3,      // (BH, m, D) output in elem_type
    int D, int m
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const float* k2i = k2_inv + bh * m * m;
    const float* ds2 = dstep2 + bh * m * D;
    scalar_t* out = dO3 + bh * m * D;

    // dO3[row, d] = sum_j K2_inv^T[row, j] * dstep2[j, d]
    //             = sum_j K2_inv[j, row] * dstep2[j, d]
    for (int idx = tid; idx < m * D; idx += nthreads) {
        int row = idx / D, d = idx % D;
        float sum = 0.0f;
        for (int j = 0; j < m; j++)
            sum += k2i[j * m + row] * ds2[j * D + d];
        out[idx] = from_float<scalar_t>(sum);
    }
}

template <typename scalar_t>
void launch_compute_dO3(
    const float* k2_inv, const float* dstep2, scalar_t* dO3,
    int BH, int D, int m, cudaStream_t stream
) {
    dim3 grid(BH);
    dim3 block(256);
    compute_dO3_kernel<scalar_t><<<grid, block, 0, stream>>>(
        k2_inv, dstep2, dO3, D, m);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
