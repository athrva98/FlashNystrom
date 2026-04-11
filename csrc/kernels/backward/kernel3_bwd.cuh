/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"

namespace flash_nystrom {

// kernel3 backward — tiled over K/V like the forward
// dK2_inv computation was moved to a seperate kernel for numerical stability
//
// Given dstep2, produces dV, dK_s, dQ_tilde, dK2_inv.
// One CTA per batch-head. Tiles over K/V columns.

constexpr int kK3BwdBc = 64;
constexpr int kK3BwdThreads = 256;

template <typename scalar_t>
__global__ void kernel3_bwd_kernel(
    const scalar_t* __restrict__ q_tilde,
    const scalar_t* __restrict__ k_s,
    const scalar_t* __restrict__ v,
    const float*    __restrict__ k2_inv,
    const float*    __restrict__ lse3,
    const float*    __restrict__ dstep2,
    scalar_t* __restrict__ dV,
    scalar_t* __restrict__ dK_s,
    float*    __restrict__ dQ_tilde,
    float*    __restrict__ dK2_inv,
    int N, int D, int m
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    constexpr int Bc = kK3BwdBc;

    extern __shared__ char smem_raw[];
    scalar_t* sQt    = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* sKtile = sQt + m * D;
    scalar_t* sVtile = sKtile + Bc * D;
    float*    sdO3   = reinterpret_cast<float*>(sVtile + Bc * D);
    float*    sP     = sdO3 + m * D;
    float*    scratch = sP + m * Bc;

    // Load Q_tilde
    const scalar_t* qt_base = q_tilde + bh * m * D;
    for (int idx = tid; idx < m * D; idx += nthreads) sQt[idx] = qt_base[idx];

    // Compute dO3 = K2_inv^T @ dstep2
    const float* k2i = k2_inv + bh * m * m;
    const float* ds2 = dstep2 + bh * m * D;
    for (int idx = tid; idx < m * D; idx += nthreads) {
        int row = idx / D, d = idx % D;
        float sum = 0.0f;
        for (int j = 0; j < m; j++) sum += k2i[j * m + row] * ds2[j * D + d];
        sdO3[idx] = sum;
    }
    __syncthreads();

    // Zero dK2_inv accumulator
    float* dk2i = dK2_inv + bh * m * m;
    for (int idx = tid; idx < m * m; idx += nthreads) dk2i[idx] = 0.0f;
    __syncthreads();

    const float* lse3_bh = lse3 + bh * m;
    int num_tiles = (N + Bc - 1) / Bc;

    for (int tile = 0; tile < num_tiles; tile++) {
        int ts = tile * Bc, te = min(ts + Bc, N), tl = te - ts;

        // Load K_tile and V_tile
        const scalar_t* k_base = k_s + bh * N * D + ts * D;
        const scalar_t* v_base = v   + bh * N * D + ts * D;
        for (int idx = tid; idx < tl * D; idx += nthreads) {
            sKtile[idx] = k_base[idx];
            sVtile[idx] = v_base[idx];
        }
        for (int idx = tid + tl * D; idx < Bc * D; idx += nthreads) {
            sKtile[idx] = from_float<scalar_t>(0.0f);
            sVtile[idx] = from_float<scalar_t>(0.0f);
        }
        __syncthreads();

        // Recompute P3[i,j] = exp(Qt[i].K_tile[j] - LSE3[i])
        // Mask rows >= m and cols >= tl to zero
        for (int idx = tid; idx < m * Bc; idx += nthreads) {
            int i = idx / Bc, j = idx % Bc;
            float p = 0.0f;
            if (i < m && j < tl) {
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += to_float(sQt[i * D + d]) * to_float(sKtile[j * D + d]);
                p = expf(dot - lse3_bh[i]);
            }
            sP[idx] = p;
        }
        __syncthreads();

        // dV_tile = P3^T @ dO3
        scalar_t* dV_tile = dV + bh * N * D + ts * D;
        for (int idx = tid; idx < tl * D; idx += nthreads) {
            int j = idx / D, d = idx % D;
            float sum = 0.0f;
            for (int i = 0; i < m; i++) sum += sP[i * Bc + j] * sdO3[i * D + d];
            float existing = to_float(dV_tile[idx]);
            dV_tile[idx] = from_float<scalar_t>(existing + sum);
        }

        // Softmax backward: dS3 = P3 * (dP3 - D3)
        // dP3[i,j] = sum_d dO3[i,d] * V[j,d]
        // D3[i] = sum_j dP3[i,j] * P3[i,j]
        // dS3[i,j] = P3[i,j] * (dP3[i,j] - D3[i])
        // Process row-by-row. Each row is independent.
        for (int i = tid; i < m; i += nthreads) {
            // Compute D3[i] = sum_j P3[i,j] * dP3[i,j]
            float D3_i = 0.0f;
            for (int j = 0; j < tl; j++) {
                float dP_ij = 0.0f;
                for (int d = 0; d < D; d++)
                    dP_ij += sdO3[i * D + d] * to_float(sVtile[j * D + d]);
                D3_i += sP[i * Bc + j] * dP_ij;
            }
            // Now compute dS3[i,j] = P3[i,j] * (dP3[i,j] - D3[i])
            for (int j = 0; j < tl; j++) {
                float dP_ij = 0.0f;
                for (int d = 0; d < D; d++)
                    dP_ij += sdO3[i * D + d] * to_float(sVtile[j * D + d]);
                sP[i * Bc + j] = sP[i * Bc + j] * (dP_ij - D3_i);
            }
        }
        // sP now contains dS3
        __syncthreads();

        // dK_s_tile = dS3^T @ Q_tilde
        scalar_t* dK_tile = dK_s + bh * N * D + ts * D;
        for (int idx = tid; idx < tl * D; idx += nthreads) {
            int j = idx / D, d = idx % D;
            float sum = 0.0f;
            for (int i = 0; i < m; i++) sum += sP[i * Bc + j] * to_float(sQt[i * D + d]);
            float existing = to_float(dK_tile[idx]);
            dK_tile[idx] = from_float<scalar_t>(existing + sum);
        }

        // dQ_tilde += dS3 @ K_tile
        float* dQt_bh = dQ_tilde + bh * m * D;
        for (int idx = tid; idx < m * D; idx += nthreads) {
            int i = idx / D, d = idx % D;
            float sum = 0.0f;
            for (int j = 0; j < tl; j++)
                sum += sP[i * Bc + j] * to_float(sKtile[j * D + d]);
            atomicAdd(&dQt_bh[idx], sum);
        }

        // dK2_inv is now computed in a separate FP32 kernel (compute_dk2inv.cuh)
        // to avoid 100x+ error amplification from the IFT backward.
        __syncthreads();
    }
}

template <typename scalar_t>
void launch_kernel3_bwd(
    const scalar_t* q_tilde, const scalar_t* k_s, const scalar_t* v,
    const float* k2_inv, const float* lse3, const float* dstep2,
    scalar_t* dV, scalar_t* dK_s, float* dQ_tilde, float* dK2_inv,
    int BH, int N, int D, int m, cudaStream_t stream
) {
    constexpr int Bc = kK3BwdBc;
    dim3 grid(BH);
    dim3 block(kK3BwdThreads);

    size_t smem = m * D * sizeof(scalar_t)        // sQt
               + Bc * D * sizeof(scalar_t) * 2    // sKtile + sVtile
               + m * D * sizeof(float)            // sdO3
               + m * Bc * sizeof(float)           // sP
               + 8 * sizeof(float);               // scratch

    if (smem > 48 * 1024) {
        FN_CHECK(smem <= get_max_smem_per_block(), "kernel3_bwd: insufficient smem");
        FN_CUDA_CHECK(cudaFuncSetAttribute(kernel3_bwd_kernel<scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    }

    kernel3_bwd_kernel<scalar_t><<<grid, block, smem, stream>>>(
        q_tilde, k_s, v, k2_inv, lse3, dstep2, dV, dK_s, dQ_tilde, dK2_inv, N, D, m);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
