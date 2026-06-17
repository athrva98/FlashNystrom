/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"
#include "nystrom_utils.h"
#include "kernels/kernel3_output_fused.cuh"   // K3Traits (TC layouts/MMA)

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cutlass/numeric_types.h>

namespace flash_nystrom {

using namespace cute;

// =============================================================================
// dK2_inv = ∂L/∂Z_J  (J = newton_iter)  AND  D3[i] = sum_n A3[i, n] * (dO3[i, :] · V[n, :])
//
// Both outputs share B = softmax(Q_tilde @ K_s^T) @ V (the kernel3 inner
// product before the K2_inv multiply). The kernel walks N once, accumulates
// B in registers/SMEM, and emits both gradient pieces:
//   dK2_inv[i, j] = sum_d dstep2[i, d] * B[j, d]            (m × m output)
//   D3[i]         = sum_d B[i, d] * dO3[i, d]              (m vector)
//
// The D3 identity:
//   D3[i] = sum_n A[i, n] · (dO3[i, :] · V[n, :])
//         = sum_d dO3[i, d] · sum_n A[i, n] · V[n, d]
//         = sum_d dO3[i, d] · B[i, d]
// so D3 is a free byproduct of B — one m·D dot product per row.
//
// Two implementations:
//   * compute_dk2inv_tc    — FP16/BF16 path. Tensor-core GEMMs reuse K3Traits
//                            (4-warp M-distribution, 16x8x16 atom). The
//                            tile-loop GEMMs (S = Qt@K^T, B += P@V) run on TC
//                            with FP32 accumulators.
//   * compute_dk2inv_kernel — FP32 scalar fallback (no TC for FP32 input).
//
// Tile-loop output B is written back to FP32 SMEM via element-wise per-thread
// fragment stores (no FP32 SMEM swizzle). The trailing m×m matmul and the
// D3 dot product run scalar (small).
// =============================================================================

// -----------------------------------------------------------------------------
// B-reuse path: B = softmax(Q_tilde @ K^T) @ V was saved during the forward
// (see kernel3_output_fused.cuh / kernel3_scalar.cuh). The backward only
// needs the two small trailing matmuls:
//
//   dK2_inv[i, j] = sum_d dstep2[i, d] * B[j, d]            (m x m output)
//   D3[i]         = sum_d B[i, d] * dO3[i, d]              (m vector)
//
// Both run scalar in a single CTA per (batch, head). At m <= 64 and D in {64,
// 128} the work is m*m*D + m*D = 0.5M ops per BH at the largest config; one
// CTA at 256 threads finishes this in tens of microseconds. This kernel
// replaces the prior N-walking compute_dk2inv path for the FP16/BF16 default
// and saves O(m * N * D) compute per backward.
// -----------------------------------------------------------------------------

template <typename scalar_t>
__global__ void compute_dk2inv_from_b_kernel(
    const scalar_t* __restrict__ b,        // (BH, m, D) saved from fwd
    const scalar_t* __restrict__ dO3,      // (BH, m, D)
    const float*    __restrict__ dstep2,   // (BH, m, D)
    float*          __restrict__ dK2_inv,  // (BH, m, m) output
    float*          __restrict__ D3,       // (BH, m) output
    int D, int m
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const scalar_t* b_bh   = b      + bh * m * D;
    const scalar_t* dO3_bh = dO3    + bh * m * D;
    const float*    ds2_bh = dstep2 + bh * m * D;
    float* dki_bh = dK2_inv + bh * m * m;
    float* d3_bh  = D3      + bh * m;

    // dK2_inv[i, j] = sum_d dstep2[i, d] * B[j, d]
    for (int idx = tid; idx < m * m; idx += nthreads) {
        int i = idx / m, j = idx % m;
        float acc = 0.0f;
        for (int d = 0; d < D; d++)
            acc += ds2_bh[i * D + d] * to_float(b_bh[j * D + d]);
        dki_bh[idx] = acc;
    }

    // D3[i] = sum_d B[i, d] * dO3[i, d]
    for (int i = tid; i < m; i += nthreads) {
        float acc = 0.0f;
        for (int d = 0; d < D; d++)
            acc += to_float(b_bh[i * D + d]) * to_float(dO3_bh[i * D + d]);
        d3_bh[i] = acc;
    }
}

template <typename scalar_t>
static void launch_compute_dk2inv_from_b(
    const scalar_t* b, const scalar_t* dO3, const float* dstep2,
    float* dK2_inv, float* D3,
    int BH, int D, int m, cudaStream_t stream
) {
    dim3 grid(BH);
    dim3 block(256);
    compute_dk2inv_from_b_kernel<scalar_t><<<grid, block, 0, stream>>>(
        b, dO3, dstep2, dK2_inv, D3, D, m);
    FN_CUDA_KERNEL_CHECK();
}

// -----------------------------------------------------------------------------
// FP32 scalar fallback. Same algorithm, scalar inner loops.
// -----------------------------------------------------------------------------

template <typename scalar_t>
__global__ void compute_dk2inv_kernel(
    const scalar_t* __restrict__ q_tilde,  // (BH, m, D)
    const scalar_t* __restrict__ k_s,      // (BH, N, D)  scaled K
    const scalar_t* __restrict__ v,        // (BH, N, D)
    const scalar_t* __restrict__ dO3,      // (BH, m, D)
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
    float* sQ   = smem;
    float* sB   = sQ   + m * D;
    float* sKV  = sB   + m * D;
    float* sA   = sKV  + TILE_N * D;
    float* sLSE = sA   + m * TILE_N;

    const scalar_t* qt    = q_tilde + bh * m * D;
    const scalar_t* ks    = k_s     + bh * N * D;
    const scalar_t* vv    = v       + bh * N * D;
    const scalar_t* dO3_b = dO3     + bh * m * D;
    const float*    lse_g = lse3    + bh * m;
    const float*    ds2   = dstep2  + bh * m * D;
    float*          dki   = dK2_inv + bh * m * m;
    float*          d3_b  = D3      + bh * m;

    for (int idx = tid; idx < m * D; idx += nthreads) sQ[idx] = to_float(qt[idx]);
    for (int idx = tid; idx < m * D; idx += nthreads) sB[idx] = 0.0f;
    for (int i = tid; i < m; i += nthreads)            sLSE[i] = lse_g[i];
    __syncthreads();

    for (int n0 = 0; n0 < N; n0 += TILE_N) {
        int tile_len = (N - n0 < TILE_N) ? (N - n0) : TILE_N;

        for (int idx = tid; idx < TILE_N * D; idx += nthreads) {
            int n = idx / D, d = idx % D;
            sKV[idx] = (n < tile_len) ? to_float(ks[(n0 + n) * D + d]) : 0.0f;
        }
        __syncthreads();

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

        for (int idx = tid; idx < TILE_N * D; idx += nthreads) {
            int n = idx / D, d = idx % D;
            sKV[idx] = (n < tile_len) ? to_float(vv[(n0 + n) * D + d]) : 0.0f;
        }
        __syncthreads();

        for (int idx = tid; idx < m * D; idx += nthreads) {
            int i = idx / D, d = idx % D;
            float acc = 0.0f;
            for (int n = 0; n < TILE_N; n++) acc += sA[i * TILE_N + n] * sKV[n * D + d];
            sB[idx] += acc;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < m * m; idx += nthreads) {
        int i = idx / m, j = idx % m;
        float acc = 0.0f;
        for (int d = 0; d < D; d++) acc += ds2[i * D + d] * sB[j * D + d];
        dki[idx] = acc;
    }

    for (int i = tid; i < m; i += nthreads) {
        float acc = 0.0f;
        const scalar_t* dO3_i = dO3_b + i * D;
        for (int d = 0; d < D; d++) acc += sB[i * D + d] * to_float(dO3_i[d]);
        d3_b[i] = acc;
    }
}

// -----------------------------------------------------------------------------
// FP16/BF16 tensor-core path. Mirrors kernel3_fused_tc's tile loop but emits
// B in FP32 SMEM at the end and runs the trailing m×m matmul + D3 scalar.
//
// P stays in registers between GEMM1 (S = Qt @ K^T) and GEMM2 (B += P @ V).
// V is read through the transposed SMEM view so the K-axis of GEMM2 maps to
// the Bc axis of sKV.
//
// SMEM (m=64, D=128): sQt (16K) + sKV (16K) + sB-FP32 (32K) = 64 KB.
// -----------------------------------------------------------------------------

template <typename Traits>
__global__ void __launch_bounds__(Traits::kNThreads)
compute_dk2inv_tc(
    const typename Traits::Element* __restrict__ q_tilde_ptr,  // (BH, m, D)
    const typename Traits::Element* __restrict__ k_s_ptr,      // (BH, N, D)
    const typename Traits::Element* __restrict__ v_ptr,        // (BH, N, D)
    const typename Traits::Element* __restrict__ dO3_ptr,      // (BH, m, D)
    const float*                    __restrict__ lse3_ptr,     // (BH, m)
    const float*                    __restrict__ dstep2_ptr,   // (BH, m, D)
    float*                          __restrict__ dK2_inv_ptr,  // (BH, m, m)
    float*                          __restrict__ D3_ptr,       // (BH, m)
    int N, int D, int m
) {
    using Element = typename Traits::Element;
    constexpr int kBlockM  = Traits::kBlockM;   // 64 (m-row tile)
    constexpr int kBlockN  = Traits::kBlockN;   // 64 (Bc tile size)
    constexpr int kHeadDim = Traits::kHeadDim;

    const int bh   = blockIdx.x;
    const int tidx = threadIdx.x;

    // SMEM layout: sQt | sKV | sB(FP32)
    extern __shared__ char smem_[];
    Element* sQ_ptr   = reinterpret_cast<Element*>(smem_);
    Element* sKV_ptr  = sQ_ptr  + Traits::kSmemQElems;
    float*   sB_ptr   = reinterpret_cast<float*>(sKV_ptr + Traits::kSmemKVElems);

    Tensor sQ   = make_tensor(make_smem_ptr(sQ_ptr),  typename Traits::SmemLayoutQ{});
    Tensor sKV  = make_tensor(make_smem_ptr(sKV_ptr), typename Traits::SmemLayoutKV{});

    // Transposed views of sKV for use as GEMM2 B-operand (where K-axis is Bc)
    Tensor sKVt    = make_tensor(sKV.data(), typename Traits::SmemLayoutKVtransposed{});
    Tensor sKVtNS  = make_tensor(sKV.data().get(),
                                 typename Traits::SmemLayoutKVtransposedNoSwizzle{});

    // Zero-init Q (handles m < kBlockM padding) then load q_tilde
    for (int idx = tidx; idx < Traits::kSmemQElems; idx += Traits::kNThreads)
        sQ_ptr[idx] = Element(0);
    __syncthreads();

    {
        const Element* qt_base = q_tilde_ptr + bh * m * D;
        for (int idx = tidx; idx < m * kHeadDim; idx += Traits::kNThreads) {
            int r = idx / kHeadDim, c = idx % kHeadDim;
            if (c < D) sQ(r, c) = qt_base[r * D + c];
        }
    }
    __syncthreads();

    typename Traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);

    typename Traits::GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr = gmem_tiled_copy.get_thread_slice(tidx);

    // Pre-load Q fragments (reused every tile)
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    auto smem_copy_Q = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto thr_copy_Q  = smem_copy_Q.get_thread_slice(tidx);
    Tensor tCsQ      = thr_copy_Q.partition_S(sQ);
    Tensor tCrQ_view = thr_copy_Q.retile_D(tSrQ);
    #pragma unroll
    for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
        cute::copy(smem_copy_Q, tCsQ(_, _, ki), tCrQ_view(_, _, ki));
    }

    // Persistent acc_b register fragment for B = sum_n A[i, n] * V[n, :]
    Tensor acc_b = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_b);

    // Identity tensors for masking and write-back coords
    Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thr_mma.partition_C(cS);
    Tensor cB = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tBcB = thr_mma.partition_C(cB);

    const float* lse_bh = lse3_ptr + bh * m;
    const int num_tiles = (N + kBlockN - 1) / kBlockN;

    for (int tile = 0; tile < num_tiles; tile++) {
        const int tile_start = tile * kBlockN;
        const int tile_end   = min(tile_start + kBlockN, N);
        const int tile_len   = tile_end - tile_start;
        const bool full_tile = (tile_len == kBlockN);

        // Load K_tile into sKV (zero-init handles the partial-tile pad)
        for (int idx = tidx; idx < Traits::kSmemKVElems; idx += Traits::kNThreads)
            sKV_ptr[idx] = Element(0);
        __syncthreads();

        if (full_tile) {
            Tensor gK = make_tensor(
                make_gmem_ptr(k_s_ptr + bh * N * D + tile_start * D),
                Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            cute::copy(gmem_tiled_copy, gmem_thr.partition_S(gK), gmem_thr.partition_D(sKV));
            cp_async_fence();
            cp_async_wait<0>();
        } else {
            const Element* k_base = k_s_ptr + bh * N * D + tile_start * D;
            for (int idx = tidx; idx < tile_len * kHeadDim; idx += Traits::kNThreads) {
                int r = idx / kHeadDim, c = idx % kHeadDim;
                if (c < D) sKV(r, c) = k_base[r * D + c];
            }
        }
        __syncthreads();

        // GEMM1: S = Qt @ K_tile^T  (Q in regs, K in SMEM)
        Tensor tSrK = thr_mma.partition_fragment_B(sKV);
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);

        auto smem_copy_K = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
        auto thr_copy_K  = smem_copy_K.get_thread_slice(tidx);
        Tensor tCsK      = thr_copy_K.partition_S(sKV);
        {
            auto tCrK_view = thr_copy_K.retile_D(tSrK);
            #pragma unroll
            for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
                cute::copy(smem_copy_K, tCsK(_, _, ki), tCrK_view(_, _, ki));
                cute::gemm(tiled_mma, tSrQ(_, _, ki), tSrK(_, _, ki), acc_s);
            }
        }

        // Apply softmax with PRECOMPUTED lse3 (no online state needed):
        //   P[i, j] = exp(S[i, j] - lse3[i])    for i < m, j < tile_len
        //           = 0                          otherwise
        auto scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
        auto coords = make_tensor(tScS.data(), convert_layout_acc_rowcol(tScS.layout()));
        constexpr int nrow = decltype(size<0>(scores))::value;
        constexpr int ncol = decltype(size<1>(scores))::value;
        #pragma unroll
        for (int mi = 0; mi < nrow; mi++) {
            int row = get<0>(coords(mi, 0));
            float lse_val = (row < m) ? lse_bh[row] : 0.0f;
            #pragma unroll
            for (int ni = 0; ni < ncol; ni++) {
                int col = get<1>(coords(mi, ni));
                bool valid = (row < m) && (col < tile_len);
                scores(mi, ni) = valid ? expf(scores(mi, ni) - lse_val) : 0.0f;
            }
        }
        // acc_s holds P in registers (FP32)

        // Reload V_tile into sKV (overwrites K). Done BEFORE we touch P regs
        // to keep the syncthreads in the standard order.
        __syncthreads();  // GEMM1 done reading sKV(K)
        for (int idx = tidx; idx < Traits::kSmemKVElems; idx += Traits::kNThreads)
            sKV_ptr[idx] = Element(0);
        __syncthreads();

        if (full_tile) {
            Tensor gV = make_tensor(
                make_gmem_ptr(v_ptr + bh * N * D + tile_start * D),
                Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            cute::copy(gmem_tiled_copy, gmem_thr.partition_S(gV), gmem_thr.partition_D(sKV));
            cp_async_fence();
            cp_async_wait<0>();
        } else {
            const Element* v_base = v_ptr + bh * N * D + tile_start * D;
            for (int idx = tidx; idx < tile_len * kHeadDim; idx += Traits::kNThreads) {
                int r = idx / kHeadDim, c = idx % kHeadDim;
                if (c < D) sKV(r, c) = v_base[r * D + c];
            }
        }
        __syncthreads();

        // GEMM2: acc_b += P @ V_tile.  P in registers (converted FP32->Element),
        // V_tile read through the transposed sKV view so the K-axis of the MMA
        // maps to the Bc axis of V.
        {
            Tensor rP = convert_type<Element>(acc_s);
            Tensor tOrP = make_tensor(rP.data(),
                convert_layout_acc_Aregs<typename Traits::TiledMma>(rP.layout()));

            Tensor tOrVt = thr_mma.partition_fragment_B(sKVtNS);
            auto smem_copy_V = make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{}, tiled_mma);
            auto thr_copy_V  = smem_copy_V.get_thread_slice(tidx);
            Tensor tCsVt = thr_copy_V.partition_S(sKVt);

            gemm_rs(acc_b, tOrP, tOrVt, tCsVt, tiled_mma, smem_copy_V, thr_copy_V);
        }
    }

    // Write acc_b (FP32 register fragments) to sB (FP32 row-major SMEM).
    // Element-wise per-thread fragment store using identity-tensor coords.
    {
        for (int idx = tidx; idx < m * D; idx += Traits::kNThreads) sB_ptr[idx] = 0.0f;
        // Pad rows (i >= m) and cols (d >= D) are not written — sB layout is m*D row-major.
    }
    __syncthreads();
    {
        auto acc_b_rc = make_tensor(acc_b.data(), convert_layout_acc_rowcol(acc_b.layout()));
        auto tBcB_rc  = make_tensor(tBcB.data(), convert_layout_acc_rowcol(tBcB.layout()));
        constexpr int br = decltype(size<0>(acc_b_rc))::value;
        constexpr int bc = decltype(size<1>(acc_b_rc))::value;
        #pragma unroll
        for (int mi = 0; mi < br; mi++) {
            #pragma unroll
            for (int ni = 0; ni < bc; ni++) {
                int row = get<0>(tBcB_rc(mi, ni));
                int col = get<1>(tBcB_rc(mi, ni));
                if (row < m && col < D) sB_ptr[row * D + col] = acc_b_rc(mi, ni);
            }
        }
    }
    __syncthreads();

    // Trailing scalar phase: dK2_inv = dstep2 @ B^T   and   D3 = diag(B @ dO3^T)
    const float*    ds2_bh = dstep2_ptr + bh * m * D;
    const Element*  dO3_bh = dO3_ptr    + bh * m * D;
    float* dki_bh = dK2_inv_ptr + bh * m * m;
    float* d3_bh  = D3_ptr      + bh * m;

    for (int idx = tidx; idx < m * m; idx += Traits::kNThreads) {
        int i = idx / m, j = idx % m;
        float acc = 0.0f;
        for (int d = 0; d < D; d++) acc += ds2_bh[i * D + d] * sB_ptr[j * D + d];
        dki_bh[idx] = acc;
    }
    for (int i = tidx; i < m; i += Traits::kNThreads) {
        float acc = 0.0f;
        const Element* dO3_i = dO3_bh + i * D;
        for (int d = 0; d < D; d++) acc += sB_ptr[i * D + d] * to_float(dO3_i[d]);
        d3_bh[i] = acc;
    }
}

// -----------------------------------------------------------------------------
// Launch wrapper. Dispatches TC for FP16/BF16, scalar for FP32.
// -----------------------------------------------------------------------------

// Helper: run the FP32 scalar kernel. Always available, used as the default
// path. Inputs are cast to FP32 in SMEM, all matmuls run in FP32.
template <typename scalar_t>
static void launch_compute_dk2inv_scalar(
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

// Dispatch:
//   1. If `b` is non-null, use the B-reuse path. B was emitted by the forward
//      kernel3 (FP16/BF16 or FP32 scalar). compute_dk2inv collapses to two
//      tiny m-bounded matmuls and the N-walk is eliminated entirely. This is
//      the default path on all dtypes when the forward saved B.
//   2. Otherwise fall back to the prior N-walking path. The `fast_dk2inv`
//      flag still toggles between TC and scalar for that fallback; kept for
//      compatibility and for the FP32-input case where no saved B exists.
//
// FP32 input dtype falls into the scalar fallback for the N-walking variant
// (TC atom requires 16-bit operands).
template <typename scalar_t>
void launch_compute_dk2inv(
    const scalar_t* q_tilde, const scalar_t* k_s, const scalar_t* v,
    const scalar_t* b,                  // (BH, m, D) saved from forward, or nullptr
    const scalar_t* dO3,
    const float* lse3, const float* dstep2,
    float* dK2_inv, float* D3,
    int BH, int N, int D, int m, bool fast_dk2inv, cudaStream_t stream
) {
    if (b != nullptr) {
        launch_compute_dk2inv_from_b<scalar_t>(
            b, dO3, dstep2, dK2_inv, D3, BH, D, m, stream);
        return;
    }
    if constexpr (std::is_same_v<scalar_t, float>) {
        launch_compute_dk2inv_scalar<scalar_t>(
            q_tilde, k_s, v, dO3, lse3, dstep2, dK2_inv, D3,
            BH, N, D, m, stream);
    } else if (!fast_dk2inv) {
        launch_compute_dk2inv_scalar<scalar_t>(
            q_tilde, k_s, v, dO3, lse3, dstep2, dK2_inv, D3,
            BH, N, D, m, stream);
    } else {
        FN_CHECK(D == 64 || D == 128, "compute_dk2inv_tc: D must be 64 or 128");
        auto launch = [&](auto HeadDimTag) {
            constexpr int kHeadDim = decltype(HeadDimTag)::value;
            using Traits = K3Traits<kHeadDim, scalar_t>;
            dim3 grid(BH);
            dim3 block(Traits::kNThreads);
            size_t smem_elem  = (Traits::kSmemQElems + Traits::kSmemKVElems)
                                * sizeof(scalar_t);
            size_t smem_fp32  = static_cast<size_t>(Traits::kBlockM) *
                                static_cast<size_t>(kHeadDim) * sizeof(float);
            size_t smem = smem_elem + smem_fp32;
            if (smem > 48 * 1024) {
                FN_CHECK(smem <= get_max_smem_per_block(),
                         "compute_dk2inv_tc: insufficient smem");
                FN_CUDA_CHECK(cudaFuncSetAttribute(compute_dk2inv_tc<Traits>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(smem)));
            }
            compute_dk2inv_tc<Traits><<<grid, block, smem, stream>>>(
                q_tilde, k_s, v, dO3, lse3, dstep2, dK2_inv, D3, N, D, m);
        };
        if (D == 64) launch(Int<64>{}); else launch(Int<128>{});
        FN_CUDA_KERNEL_CHECK();
    }
}

} // namespace flash_nystrom
