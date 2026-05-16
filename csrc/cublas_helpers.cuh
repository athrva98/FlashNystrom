/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 *
 * cuBLAS helpers for the m-bounded dense matmuls in the Newton-Schulz step.
 * The TC flash kernels (kernel1, kernel3 fwd/bwd) stay custom; only the
 * m x m x m matmuls in NS, where there is no softmax fusion to exploit,
 * delegate to cuBLAS so we are not competing with NVIDIA's tuned Sgemm at
 * shapes that fit cleanly in a single CTA.
 ******************************************************************************/
#pragma once

#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

#include "utils.h"

namespace flash_nystrom {

// Row-major batched GEMM: C_rm = alpha * op(A_rm) @ op(B_rm) + beta * C_rm.
//
// cuBLAS is column-major. The standard trick is to call it with the two
// inputs swapped, which yields C^T_cm = (A@B)^T_cm = B^T @ A^T. Since
// row-major data IS the column-major transpose of the same matrix, the
// data we hand cuBLAS already represents the transposes, so no actual
// transpose work happens.
//
// opA, opB are 'N' or 'T' in row-major math semantics:
//   opA = 'N'  ->  A is read as (M_rm x K_rm)
//   opA = 'T'  ->  A is read as (K_rm x M_rm) and used as A^T
// (M_rm, K_rm) is the post-op shape of A; (K_rm, N_rm) of B; C is (M_rm, N_rm).
//
// Strides are between consecutive batched matrices, expressed in floats.
inline void rm_sgemm_strided_batched(
    cublasHandle_t handle,
    char opA, char opB,
    int M_rm, int N_rm, int K_rm,
    float alpha,
    const float* A, long long strideA,
    const float* B, long long strideB,
    float beta,
    float* C, long long strideC,
    int batch
) {
    cublasOperation_t cb_opA = (opB == 'T' || opB == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cb_opB = (opA == 'T' || opA == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda_rm = (opA == 'N' || opA == 'n') ? K_rm : M_rm;  // row stride of A in row-major
    int ldb_rm = (opB == 'N' || opB == 'n') ? N_rm : K_rm;  // row stride of B
    int ldc_rm = N_rm;                                       // row stride of C

    auto status = cublasSgemmStridedBatched(handle,
        cb_opA, cb_opB,
        N_rm, M_rm, K_rm,
        &alpha,
        B, ldb_rm, strideB,
        A, lda_rm, strideA,
        &beta,
        C, ldc_rm, strideC,
        batch);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "[FlashNystrom] cublasSgemmStridedBatched failed with status "
            << static_cast<int>(status)
            << " (shape M_rm=" << M_rm << ", N_rm=" << N_rm
            << ", K_rm=" << K_rm << ", batch=" << batch << ")";
        throw std::runtime_error(oss.str());
    }
}

// out[bh, i, j] = c0 * (i == j ? 1 : 0)
//              + c1 * A[bh, i, j]
//              + (B ? c2 * B[bh, i, j] : 0)
//
// This handles all the linear-combination-with-identity-offset patterns
// that show up between the NS step matmuls (V = 15I - 7M + M^2,
// T = 13I - M*V, and similar). A and B may alias C and may be the same
// buffer; the kernel writes one element per thread so there are no races.
template <int BLOCK = 128>
__global__ void affine_with_identity_kernel(
    float* __restrict__ out,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float c0, float c1, float c2,
    int m
) {
    int bh = blockIdx.y;
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    int mm = m * m;
    if (idx >= mm) return;
    int row = idx / m;
    int col = idx - row * m;
    float val = c1 * A[bh * mm + idx];
    if (B != nullptr) val += c2 * B[bh * mm + idx];
    if (row == col)   val += c0;
    out[bh * mm + idx] = val;
}

inline void launch_affine_with_identity(
    float* out, const float* A, const float* B,
    float c0, float c1, float c2,
    int BH, int m, cudaStream_t stream
) {
    constexpr int BLOCK = 128;
    int mm = m * m;
    dim3 grid((mm + BLOCK - 1) / BLOCK, BH);
    affine_with_identity_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
        out, A, B, c0, c1, c2, m);
}

// out[bh, i, j] += scale * A[bh, i, j]  (plain add, NOT atomicAdd)
// Used to fold dV @ M^T into dM in place.
template <int BLOCK = 128>
__global__ void add_scaled_inplace_kernel(
    float* __restrict__ out,
    const float* __restrict__ A,
    float scale,
    int m
) {
    int bh = blockIdx.y;
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    int mm = m * m;
    if (idx >= mm) return;
    out[bh * mm + idx] += scale * A[bh * mm + idx];
}

inline void launch_add_scaled_inplace(
    float* out, const float* A, float scale,
    int BH, int m, cudaStream_t stream
) {
    constexpr int BLOCK = 128;
    int mm = m * m;
    dim3 grid((mm + BLOCK - 1) / BLOCK, BH);
    add_scaled_inplace_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
        out, A, scale, m);
}

// out[bh, i, j] += scale * A[bh, i, j]  via atomicAdd
// (used for accumulating dK2_contrib across NS iterations into dK2_acc).
template <int BLOCK = 128>
__global__ void atomic_add_scaled_kernel(
    float* __restrict__ out,
    const float* __restrict__ A,
    float scale,
    int m
) {
    int bh = blockIdx.y;
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    int mm = m * m;
    if (idx >= mm) return;
    atomicAdd(&out[bh * mm + idx], scale * A[bh * mm + idx]);
}

inline void launch_atomic_add_scaled(
    float* out, const float* A, float scale,
    int BH, int m, cudaStream_t stream
) {
    constexpr int BLOCK = 128;
    int mm = m * m;
    dim3 grid((mm + BLOCK - 1) / BLOCK, BH);
    atomic_add_scaled_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
        out, A, scale, m);
}

} // namespace flash_nystrom
