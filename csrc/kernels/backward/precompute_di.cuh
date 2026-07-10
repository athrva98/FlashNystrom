/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"

namespace flash_nystrom {

// D1[i] = sum_d(dO[i,d] * O[i,d]) — per-row dot product for softmax backward
// Grid: (ceil(N/256), BH), Block: 256

template <typename scalar_t>
__global__ void precompute_di_kernel(
    const scalar_t* __restrict__ dO,  // (BH, N, D)
    const scalar_t* __restrict__ O,   // (BH, N, D)
    float* __restrict__ D1,           // (BH, N)
    int N, int D
) {
    const int64_t bh = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const scalar_t* dO_row = dO + bh * N * D + n * D;
    const scalar_t* O_row  = O  + bh * N * D + n * D;

    float dot = 0.0f;
    for (int d = 0; d < D; d++) {
        dot += to_float(dO_row[d]) * to_float(O_row[d]);
    }
    D1[bh * N + n] = dot;
}

template <typename scalar_t>
void launch_precompute_di(
    const scalar_t* dO, const scalar_t* O, float* D1,
    int BH, int N, int D, cudaStream_t stream
) {
    dim3 grid((N + 255) / 256, BH);
    dim3 block(256);
    precompute_di_kernel<scalar_t><<<grid, block, 0, stream>>>(dO, O, D1, N, D);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
