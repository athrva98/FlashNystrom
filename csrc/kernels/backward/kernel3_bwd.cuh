/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"
#include "nystrom_utils.h"
#include "flash_nystrom.h"   // g_kernel3_bwd_sm100_hook
#include "kernels/kernel3_output_fused.cuh"

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cutlass/numeric_types.h>

namespace flash_nystrom {

using namespace cute;

// kernel3 backward — tiled over K/V like the forward.
// dK2_inv computation was moved to a separate kernel for numerical stability.
//
// FP32 path: scalar kernel, one CTA per batch-head, tiles over K/V serially.
// FP16/BF16 path: tensor core kernel (kernel3_bwd_tc), one CTA per (batch-head, tile),
//   full parallelism across tiles. Uses TiledMma (4,1,1) and TiledMmaDKV (2,2,1).

constexpr int kK3BwdBc = 64;
constexpr int kK3BwdThreads = 256;

template <typename scalar_t>
__global__ void kernel3_bwd_kernel(
    const scalar_t* __restrict__ q_tilde,
    const scalar_t* __restrict__ k_s,
    const scalar_t* __restrict__ v,
    const float*    __restrict__ k2_inv,
    const float*    __restrict__ lse3,
    const float*    __restrict__ D3_in,
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

    // dK2_inv is computed by a separate FP32 kernel (compute_dk2inv.cuh).
    // No zeroing needed here — the separate kernel overwrites it entirely.

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

        // Softmax backward: dS3[i,j] = P3[i,j] * (dP3[i,j] - D3[i])
        //   dP3[i,j] = sum_d dO3[i,d] * V[j,d]
        //   D3[i]    = sum_n_GLOBAL P3[i,n] * dP3[i,n]   (precomputed)
        //
        // D3 is the GLOBAL rowsum across all N positions and is precomputed
        // by launch_precompute_d3 — must NOT be recomputed per-tile (the
        // per-tile sum is incorrect whenever N > Bc).
        const float* D3_bh = D3_in + bh * m;
        for (int i = tid; i < m; i += nthreads) {
            float D3_i = D3_bh[i];
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

// -- tensor core kernel (fp16/bf16), multi-CTA --
//
// ALL 5 GEMMs use tensor cores. One CTA per (batch-head, tile).
//
// 3-buffer SMEM: sQB + sKV + sPdS = 40KB at D=128
//
// sQB is time-multiplexed between Qt (Phase 1, Phase 11) and dO3 (Phase 4, 7).
// Folding the two 16KB buffers into one (and paying one extra Qt reload from
// GMEM after Phase 7) drops SMEM from 56KB to 40KB and lets two CTAs share an
// SM. The Qt and dO3 lifetimes never overlap so the reuse is exact.
//
// TiledMma (4,1,1):    GEMM1 (S=Qt@K^T), GEMM_dP (dP=dO3@V^T), GEMM_dQt (dQt+=dS@K^T)
// TiledMmaDKV (2,2,1): GEMM_dK (dK=dS^T@Qt), GEMM_dV (dV=P^T@dO3)
//
// sQB lifecycle: Qt (Phase 1) -> dO3 (Phase 4, 7) -> Qt (Phase 11)
// sKV lifecycle: K (Phase 1) -> V (Phase 4, 7) -> K (Phase 10)
// sPdS lifecycle: P (Phase 7) -> dS (Phase 10, 11)

template <typename Traits, bool kWide>
__global__ void __launch_bounds__(Traits::kNThreads)
kernel3_bwd_tc(
    const typename Traits::Element* __restrict__ q_tilde_ptr,  // (BH, m, D)
    const typename Traits::Element* __restrict__ k_s_ptr,      // (BH, N, D)
    const typename Traits::Element* __restrict__ v_ptr,        // (BH, N, D)
    const float*                    __restrict__ lse3_ptr,     // (BH, m)
    const float*                    __restrict__ D3_ptr,       // (BH, m) precomputed global rowsum
    const typename Traits::Element* __restrict__ dO3_ptr,      // (BH, m, D) precomputed FP16
    typename Traits::Element*       __restrict__ dV_ptr,       // (BH, N, D) output
    typename Traits::Element*       __restrict__ dK_s_ptr,     // (BH, N, D) output
    // Split-K target for dQ_tilde. Shape (num_splits, BH, m, D) FP32, zero
    // initialized; each tile atomicAdds into slot (tile_idx % num_splits).
    // When num_splits == 0 the kernel writes straight to dQ_tilde_split_ptr
    // treated as the legacy (BH, m, D) buffer (single shared accumulator).
    float*                          __restrict__ dQ_tilde_split_ptr,
    int num_splits,
    int N, int D, int m
) {
    using Element = typename Traits::Element;
    constexpr int kBlockM  = Traits::kBlockM;   // 64 (m dimension)
    constexpr int kBlockN  = Traits::kBlockN;   // 64 (Bc tile size)
    constexpr int kHeadDim = Traits::kHeadDim;

    // Transposed view reuse requires SmemLayoutQ == SmemLayoutKV
    static_assert(kBlockM == kBlockN, "kernel3_bwd_tc requires kBlockM == kBlockN");

    const int tile_idx = blockIdx.x;
    const int bh = blockIdx.y;
    const int tidx = threadIdx.x;

    const int tile_start = tile_idx * kBlockN;
    if (tile_start >= N) return;
    const int tile_len = min(kBlockN, N - tile_start);
    const bool full_tile = (tile_len == kBlockN);

    // SMEM layout. Narrow (3 buffers): sQB holds Qt or dO3 depending on phase
    // and sKV holds K or V, forcing two mid-kernel swap stalls plus K and Qt
    // reloads from GMEM. Wide (5 buffers): dedicated sdO3 and sV, everything
    // loads once in the prologue, no swaps, no reloads. SmemLayoutQ ==
    // SmemLayoutKV under kBlockM == kBlockN, so partition_fragment and copy
    // atoms compose identically against either view.
    extern __shared__ char smem_[];
    Element* sQB_ptr  = reinterpret_cast<Element*>(smem_);
    Element* sKV_ptr  = sQB_ptr  + Traits::kSmemQElems;
    Element* sPdS_ptr = sKV_ptr  + Traits::kSmemKVElems;
    Element* sdO3_ptr = kWide ? sPdS_ptr + Traits::kSmemPdSElems : sQB_ptr;
    Element* sV_ptr   = kWide ? sdO3_ptr + Traits::kSmemQElems   : sKV_ptr;

    // Normal views (sdO3/sV alias sQB/sKV when narrow)
    auto sQt   = make_tensor(make_smem_ptr(sQB_ptr),  typename Traits::SmemLayoutQ{});
    auto sdO3  = make_tensor(make_smem_ptr(sdO3_ptr), typename Traits::SmemLayoutKV{});
    auto sKV   = make_tensor(make_smem_ptr(sKV_ptr),  typename Traits::SmemLayoutKV{});
    auto sV    = make_tensor(make_smem_ptr(sV_ptr),   typename Traits::SmemLayoutKV{});
    auto sPdS  = make_tensor(make_smem_ptr(sPdS_ptr), typename Traits::SmemLayoutPdS{});

    // Transposed views for TC GEMM B-operands and A-operands
    auto sKVt    = make_tensor(sKV.data(),        typename Traits::SmemLayoutKVtransposed{});
    auto sKVtNS  = make_tensor(sKV.data().get(),  typename Traits::SmemLayoutKVtransposedNoSwizzle{});
    auto sQtt    = make_tensor(sQt.data(),        typename Traits::SmemLayoutKVtransposed{});
    auto sQttNS  = make_tensor(sQt.data().get(),  typename Traits::SmemLayoutKVtransposedNoSwizzle{});
    auto sdO3t   = make_tensor(sdO3.data(),       typename Traits::SmemLayoutKVtransposed{});
    auto sdO3tNS = make_tensor(sdO3.data().get(), typename Traits::SmemLayoutKVtransposedNoSwizzle{});
    auto sPdSt   = make_tensor(sPdS.data(),       typename Traits::SmemLayoutPdStransposed{});
    auto sPdStNS = make_tensor(make_smem_ptr(sPdS_ptr), typename Traits::SmemLayoutPdStransposedNoSwizzle{});

    // Zero-init all buffers (handles partial tiles and m < kBlockM)
    constexpr int total_elems = kWide
        ? 2*Traits::kSmemQElems + 2*Traits::kSmemKVElems + Traits::kSmemPdSElems
        : Traits::kSmemBwdElems;
    for (int i = tidx; i < total_elems; i += Traits::kNThreads)
        reinterpret_cast<Element*>(smem_)[i] = Element(0);
    __syncthreads();

    // Load Q_tilde into sQB and K into sKV. Narrow loads dO3/V later by
    // swapping buffers and reloads Qt/K before Phases 10/11; wide issues dO3
    // and V here as a second cp.async group that completes under GEMM1.
    typename Traits::GmemTiledCopy gmem_copy;
    auto gmem_thr = gmem_copy.get_thread_slice(tidx);

    if (m == kBlockM) {
        auto gQt  = make_tensor(make_gmem_ptr(q_tilde_ptr + bh*m*D),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
        cute::copy(gmem_copy, gmem_thr.partition_S(gQt),  gmem_thr.partition_D(sQt));
    } else {
        auto* qt = q_tilde_ptr + bh*m*D;
        for (int i = tidx; i < m*kHeadDim; i += Traits::kNThreads) {
            int r = i / kHeadDim, c = i % kHeadDim;
            if (c < D) sQt(r,c) = qt[r*D+c];
        }
    }

    // Load K_tile into sKV
    if (full_tile) {
        auto gK = make_tensor(make_gmem_ptr(k_s_ptr + bh*N*D + tile_start*D),
                              Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
        cute::copy(gmem_copy, gmem_thr.partition_S(gK), gmem_thr.partition_D(sKV));
    } else {
        auto* k_base = k_s_ptr + bh*N*D + tile_start*D;
        for (int i = tidx; i < tile_len*kHeadDim; i += Traits::kNThreads) {
            int r = i / kHeadDim, c = i % kHeadDim;
            if (c < D) sKV(r,c) = k_base[r*D+c];
        }
    }
    if constexpr (kWide) {
        cp_async_fence();  // group 1: Qt + K (GEMM1's operands)
        // Issue V and dO3 now; they complete under GEMM1 and are waited
        // before Phase 4 (GEMM_dP).
        if (full_tile) {
            auto gV = make_tensor(make_gmem_ptr(v_ptr + bh*N*D + tile_start*D),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            cute::copy(gmem_copy, gmem_thr.partition_S(gV), gmem_thr.partition_D(sV));
        } else {
            auto* v_base = v_ptr + bh*N*D + tile_start*D;
            for (int i = tidx; i < tile_len*kHeadDim; i += Traits::kNThreads) {
                int r = i / kHeadDim, c = i % kHeadDim;
                if (c < D) sV(r,c) = v_base[r*D+c];
            }
        }
        if (m == kBlockM) {
            auto gdO3 = make_tensor(make_gmem_ptr(dO3_ptr + bh*m*D),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            cute::copy(gmem_copy, gmem_thr.partition_S(gdO3), gmem_thr.partition_D(sdO3));
        } else {
            auto* d3 = dO3_ptr + bh*m*D;
            for (int i = tidx; i < m*kHeadDim; i += Traits::kNThreads) {
                int r = i / kHeadDim, c = i % kHeadDim;
                if (c < D) sdO3(r,c) = d3[r*D+c];
            }
        }
        cp_async_fence();  // group 2: V + dO3
        cp_async_wait<1>();  // group 1 done; group 2 may still be in flight
    } else {
        cp_async_fence(); cp_async_wait<0>();
    }
    __syncthreads();

    // ======== TiledMma setup (GEMM1, GEMM_dP, GEMM_dQt) ========
    typename Traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);

    auto smem_copy_A  = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto thr_copy_A   = smem_copy_A.get_thread_slice(tidx);
    auto smem_copy_B  = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto thr_copy_B   = smem_copy_B.get_thread_slice(tidx);
    auto smem_copy_Bt = make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto thr_copy_Bt  = smem_copy_Bt.get_thread_slice(tidx);

    // Identity tensor for (kBlockM, kBlockN) accumulators — score masking
    auto cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    auto tScS = thr_mma.partition_C(cS);
    auto tScS_rc = make_tensor(tScS.data(), convert_layout_acc_rowcol(tScS.layout()));

    // Identity tensor for (kBlockM, kHeadDim) accumulators — dQt atomicAdd
    auto cDQt = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    auto tDQtc = thr_mma.partition_C(cDQt);

    // ======== TiledMmaDKV setup (GEMM_dK, GEMM_dV) ========
    typename Traits::TiledMmaDKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);

    auto smem_copy_PdSt  = make_tiled_copy_A(typename Traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto thr_copy_PdSt   = smem_copy_PdSt.get_thread_slice(tidx);
    auto smem_copy_QtT   = make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto thr_copy_QtT    = smem_copy_QtT.get_thread_slice(tidx);

    // Identity tensor for (kBlockN, kHeadDim) accumulators — dK/dV output
    auto cDKV = make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
    auto tDKVc = thr_mma_dkv.partition_C(cDKV);

    // ======== Phase 1: GEMM1 — S = Qt @ K^T (sKV has K_tile) ========
    auto acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);
    gemm_smem(acc_s, thr_mma.partition_fragment_A(sQt), thr_mma.partition_fragment_B(sKV),
              thr_copy_A.partition_S(sQt), thr_copy_B.partition_S(sKV),
              tiled_mma, smem_copy_A, smem_copy_B, thr_copy_A, thr_copy_B);

    // ======== Phase 2: P = exp(S - LSE3) ========
    auto scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    constexpr int nrow = decltype(size<0>(scores))::value;

    const float* lse = lse3_ptr + bh * m;
    #pragma unroll
    for (int mi = 0; mi < nrow; mi++) {
        int pr = get<0>(tScS_rc(mi, 0));
        float lse_val = (pr < m) ? lse[pr] : 0.0f;
        #pragma unroll
        for (int ni = 0; ni < size<1>(scores); ni++) {
            int pc = get<1>(tScS_rc(mi, ni));
            scores(mi, ni) = (pc < tile_len && pr < m) ? expf(scores(mi, ni) - lse_val) : 0.0f;
        }
    }
    // Eager FP32 -> FP16 conversion of P. Frees the 32 FP32 acc_s registers
    // before GEMM_dP and softmax_bwd, dropping live-register pressure across
    // the GEMM_dV peak. softmax_bwd upconverts per element below.
    Tensor rP = convert_type<Element>(acc_s);
    auto rP_rc = make_tensor(rP.data(), convert_layout_acc_rowcol(acc_s.layout()));

    // ======== Phase 3: make V and dO3 readable ========
    if constexpr (kWide) {
        // V and dO3 were issued in the prologue and have been loading under
        // GEMM1; Qt and K stay resident for Phases 10/11.
        cp_async_wait<0>();
        __syncthreads();
    } else {
        // Narrow: swap sKV (K -> V) and sQB (Qt -> dO3). Phase 4 needs V (for
        // dP = dO3 @ V^T) and dO3 (as A operand of GEMM_dP). sQt is dead after
        // Phase 1 (acc_s now holds S); sKV(K) is dead after GEMM1's inner-K
        // loop completes. Overwriting both is safe past this sync. Partial
        // tails stay zero from the kernel-start init.
        __syncthreads();  // GEMM1 done reading sQt and sKV(K)
        if (full_tile) {
            auto gV = make_tensor(make_gmem_ptr(v_ptr + bh*N*D + tile_start*D),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            cute::copy(gmem_copy, gmem_thr.partition_S(gV), gmem_thr.partition_D(sV));
        } else {
            auto* v_base = v_ptr + bh*N*D + tile_start*D;
            for (int i = tidx; i < tile_len*kHeadDim; i += Traits::kNThreads) {
                int r = i / kHeadDim, c = i % kHeadDim;
                if (c < D) sV(r,c) = v_base[r*D+c];
            }
        }
        if (m == kBlockM) {
            auto gdO3 = make_tensor(make_gmem_ptr(dO3_ptr + bh*m*D),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            cute::copy(gmem_copy, gmem_thr.partition_S(gdO3), gmem_thr.partition_D(sdO3));
        } else {
            auto* d3 = dO3_ptr + bh*m*D;
            for (int i = tidx; i < m*kHeadDim; i += Traits::kNThreads) {
                int r = i / kHeadDim, c = i % kHeadDim;
                if (c < D) sdO3(r,c) = d3[r*D+c];
            }
        }
        cp_async_fence(); cp_async_wait<0>(); __syncthreads();
    }

    // ======== Phase 4: GEMM_dP — dP = dO3 @ V^T ========
    auto acc_dp = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_dp);
    gemm_smem(acc_dp, thr_mma.partition_fragment_A(sdO3), thr_mma.partition_fragment_B(sV),
              thr_copy_A.partition_S(sdO3), thr_copy_B.partition_S(sV),
              tiled_mma, smem_copy_A, smem_copy_B, thr_copy_A, thr_copy_B);

    // ======== Phase 5: Softmax backward — dS = P * (dP - D3) ========
    // D3[i] is the GLOBAL rowsum sum_n A3[i, n] * dP3[i, n], precomputed by
    // launch_precompute_d3. P is read from rP_rc (FP16) and upconverted per
    // element; the FP32 acc_s storage was freed by the eager conversion above.
    auto dP_rc = make_tensor(acc_dp.data(), convert_layout_acc_rowcol(acc_dp.layout()));

    Tensor D3 = make_tensor<float>(Shape<Int<nrow>>{});
    const float* D3_bh = D3_ptr + bh * m;
    #pragma unroll
    for (int mi = 0; mi < nrow; mi++) {
        int pr = get<0>(tScS_rc(mi, 0));
        D3(mi) = (pr < m) ? D3_bh[pr] : 0.0f;
    }

    #pragma unroll
    for (int mi = 0; mi < nrow; mi++) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(dP_rc); ni++) {
            // FP32 P (scores), not the eagerly-downcast fp16 rP -- the
            // softmax-Jacobian cancellation needs full-precision P (see
            // kernel1_bwd). dS is still downcast for the GEMM (TC unchanged).
            float p_val = scores(mi, ni);
            dP_rc(mi, ni) = p_val * (dP_rc(mi, ni) - D3(mi));
        }
    }
    // Eager FP16 conversion of dS, mirror of acc_s above. Frees 32 FP32 regs
    // before GEMM_dV (which allocates a 64-FP32-reg acc_dv).
    Tensor rdS = convert_type<Element>(acc_dp);

    // ======== Phase 6: Write P (rP, already FP16) to sPdS ========
    {
        auto sc = make_tiled_copy_C(typename Traits::SmemCopyAtomPdS{}, tiled_mma);
        auto tc = sc.get_thread_slice(tidx);
        cute::copy(sc, tc.retile_S(rP), tc.partition_D(sPdS));
    }
    __syncthreads();

    // ======== Phase 7: GEMM_dV — dV += P^T @ dO3 (accumulate to existing) ========
    // dV is zero-initialized by the bwd caller; this is the only writer.
    // The += form (rather than =) keeps the kernel composable with any
    // future dV contribution that runs before this one.
    {
        auto acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
        clear(acc_dv);
        gemm_smem(acc_dv,
                  thr_mma_dkv.partition_fragment_A(sPdStNS),
                  thr_mma_dkv.partition_fragment_B(sdO3tNS),
                  thr_copy_PdSt.partition_S(sPdSt),
                  thr_copy_QtT.partition_S(sdO3t),
                  tiled_mma_dkv, smem_copy_PdSt, smem_copy_QtT,
                  thr_copy_PdSt, thr_copy_QtT);

        Element* dv_tile = dV_ptr + bh * N * D + tile_start * D;
        #pragma unroll
        for (int fi = 0; fi < size(acc_dv); fi++) {
            int row = get<0>(tDKVc(fi));
            int col = get<1>(tDKVc(fi));
            if (row < tile_len && col < D) {
                float existing = to_float(dv_tile[row * D + col]);
                dv_tile[row * D + col] = from_float<Element>(existing + acc_dv(fi));
            }
        }
    }

    // ======== Phase 8: Write dS (rdS, already FP16 from eager convert) to sPdS ========
    // GEMM_dV above reads P from sPdS; this store overwrites sPdS with dS.
    // Without a barrier a fast warp clobbers P with dS while a lagging warp is
    // still reading P in GEMM_dV's mainloop -- WAR on sPdS (racecheck). This is
    // the kernel3 twin of the kernel1 GEMM2->P-write barrier.
    __syncthreads();
    {
        auto sc = make_tiled_copy_C(typename Traits::SmemCopyAtomPdS{}, tiled_mma);
        auto tc = sc.get_thread_slice(tidx);
        cute::copy(sc, tc.retile_S(rdS), tc.partition_D(sPdS));
    }

    // ======== Phase 9: make dS visible (+ reload K and Qt when narrow) ========
    // The sync also protects dO3's last read (Phase 7) and the dS write to
    // sPdS (Phase 8) before Phases 10/11 consume them.
    __syncthreads();  // GEMM_dV done reading sdO3 and sPdS(P); dS write to sPdS complete
    if constexpr (!kWide) {
        // Narrow: V evicted K and dO3 evicted Qt, so reload both for Phases
        // 10 and 11. Partial tails stay zero. full_tile gates the K reload
        // (sKV size), m == kBlockM gates the Qt reload (sQB size).
        if (full_tile) {
            auto gK  = make_tensor(make_gmem_ptr(k_s_ptr + bh*N*D + tile_start*D),
                                   Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            cute::copy(gmem_copy, gmem_thr.partition_S(gK),  gmem_thr.partition_D(sKV));
        } else {
            auto* k_base = k_s_ptr + bh*N*D + tile_start*D;
            for (int i = tidx; i < tile_len*kHeadDim; i += Traits::kNThreads) {
                int r = i / kHeadDim, c = i % kHeadDim;
                if (c < D) sKV(r,c) = k_base[r*D+c];
            }
        }
        if (m == kBlockM) {
            auto gQt = make_tensor(make_gmem_ptr(q_tilde_ptr + bh*m*D),
                                   Shape<Int<kBlockM>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            cute::copy(gmem_copy, gmem_thr.partition_S(gQt), gmem_thr.partition_D(sQt));
        } else {
            auto* qt_base = q_tilde_ptr + bh*m*D;
            for (int i = tidx; i < m*kHeadDim; i += Traits::kNThreads) {
                int r = i / kHeadDim, c = i % kHeadDim;
                if (c < D) sQt(r,c) = qt_base[r*D+c];
            }
        }
        cp_async_fence(); cp_async_wait<0>(); __syncthreads();
    }

    // ======== Phase 10: GEMM_dQt — dQt += dS @ K^T (atomicAdd) ========
    {
        auto acc_dqt = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
        clear(acc_dqt);
        gemm_smem(acc_dqt, thr_mma.partition_fragment_A(sPdS), thr_mma.partition_fragment_B(sKVtNS),
                  thr_copy_A.partition_S(sPdS), thr_copy_Bt.partition_S(sKVt),
                  tiled_mma, smem_copy_A, smem_copy_Bt, thr_copy_A, thr_copy_Bt);

        // Split-K: write into slot (tile_idx % num_splits) so adjacent tiles
        // do not contend on the same FP32 cells in L2. With num_splits >=
        // num_tiles each slot has exactly one contributor (zero atomicAdd
        // contention); otherwise contention is num_tiles / num_splits, much
        // less than the prior num_tiles per cell.
        const int split_idx = (num_splits > 0) ? (tile_idx % num_splits) : 0;
        const long long bh_slot_stride = static_cast<long long>(m) * D;
        const long long split_slot_stride = static_cast<long long>(gridDim.y) * bh_slot_stride;
        float* dqt = dQ_tilde_split_ptr
                   + static_cast<long long>(split_idx) * split_slot_stride
                   + static_cast<long long>(bh) * bh_slot_stride;
        #pragma unroll
        for (int fi = 0; fi < size(acc_dqt); fi++) {
            int row = get<0>(tDQtc(fi));
            int col = get<1>(tDQtc(fi));
            if (row < m && col < D)
                atomicAdd(&dqt[row * D + col], acc_dqt(fi));
        }
    }

    // ======== Phase 11: GEMM_dK — dK = dS^T @ Qt (direct write) ========
    {
        auto acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
        clear(acc_dk);
        gemm_smem(acc_dk,
                  thr_mma_dkv.partition_fragment_A(sPdStNS),
                  thr_mma_dkv.partition_fragment_B(sQttNS),
                  thr_copy_PdSt.partition_S(sPdSt),
                  thr_copy_QtT.partition_S(sQtt),
                  tiled_mma_dkv, smem_copy_PdSt, smem_copy_QtT,
                  thr_copy_PdSt, thr_copy_QtT);

        Element* dk_tile = dK_s_ptr + bh * N * D + tile_start * D;
        #pragma unroll
        for (int fi = 0; fi < size(acc_dk); fi++) {
            int row = get<0>(tDKVc(fi));
            int col = get<1>(tDKVc(fi));
            if (row < tile_len && col < D)
                dk_tile[row * D + col] = from_float<Element>(acc_dk(fi));
        }
    }
}

// Reduce dQ_tilde_split (num_splits, BH, m, D) along the split axis into
// dQ_tilde (BH, m, D). One thread per (bh, row*D+col); each thread sums
// num_splits scattered reads, one per split slot.
//
// Templated on a dummy parameter to give the global function weak linkage
// so the header can be included from multiple translation units without
// LNK2005 multiply-defined errors.
template <int kBlock = 128>
__global__ void reduce_dQ_tilde_split_kernel(
    const float* __restrict__ dQ_tilde_split,
    float* __restrict__ dQ_tilde,
    int num_splits, int BH, int mD
) {
    int bh = blockIdx.y;
    int idx = blockIdx.x * kBlock + threadIdx.x;
    if (idx >= mD) return;
    const long long bh_stride    = static_cast<long long>(mD);
    const long long split_stride = static_cast<long long>(BH) * bh_stride;
    float acc = 0.0f;
    #pragma unroll 8
    for (int s = 0; s < num_splits; s++) {
        acc += dQ_tilde_split[
            static_cast<long long>(s) * split_stride
            + static_cast<long long>(bh) * bh_stride
            + idx];
    }
    dQ_tilde[bh * bh_stride + idx] = acc;
}

inline void launch_reduce_dQ_tilde_split(
    const float* dQ_tilde_split, float* dQ_tilde,
    int num_splits, int BH, int m, int D, cudaStream_t stream
) {
    int mD = m * D;
    constexpr int BLOCK = 128;
    dim3 grid((mD + BLOCK - 1) / BLOCK, BH);
    reduce_dQ_tilde_split_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
        dQ_tilde_split, dQ_tilde, num_splits, BH, mD);
    FN_CUDA_KERNEL_CHECK();
}


template <typename scalar_t>
void launch_kernel3_bwd(
    const scalar_t* q_tilde, const scalar_t* k_s, const scalar_t* v,
    const float* k2_inv, const float* lse3,
    const float* D3,                    // (BH, m) global precomputed rowsum
    const float* dstep2,
    scalar_t* dV, scalar_t* dK_s, float* dQ_tilde, float* dK2_inv,
    const scalar_t* dO3,                // precomputed dO3 in scalar_t (always supplied)
    // dQ_tilde split-K workspace: (num_splits, BH, m, D) FP32, zero-init by
    // caller. The kernel writes each tile's dQ_tilde contribution to slot
    // (tile_idx % num_splits), then the reduction below sums into dQ_tilde.
    // If null or num_splits == 0, dQ_tilde is used directly (legacy).
    float* dQ_tilde_split, int num_splits,
    int BH, int N, int D, int m, cudaStream_t stream
) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        // FP32: scalar fallback (single CTA per batch-head)
        constexpr int Bc = kK3BwdBc;
        dim3 grid(BH);
        dim3 block(kK3BwdThreads);
        size_t smem = m * D * sizeof(scalar_t)        // sQt
                   + Bc * D * sizeof(scalar_t) * 2    // sKtile + sVtile
                   + m * D * sizeof(float)            // sdO3
                   + m * Bc * sizeof(float);          // sP
        if (smem > 48 * 1024) {
            FN_CHECK(smem <= get_max_smem_per_block(), "kernel3_bwd: insufficient smem");
            FN_CUDA_CHECK(cudaFuncSetAttribute(kernel3_bwd_kernel<float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        }
        kernel3_bwd_kernel<float><<<grid, block, smem, stream>>>(
            q_tilde, k_s, v, k2_inv, lse3, D3, dstep2, dV, dK_s, dQ_tilde, dK2_inv, N, D, m);
    } else {
        // FP16/BF16: tensor core, multi-CTA
        FN_CHECK(D == 64 || D == 128, "kernel3_bwd: D must be 64 or 128");
        FN_CHECK(dO3 != nullptr, "kernel3_bwd TC requires precomputed dO3");

        // Blackwell-native path (fp16 only). The hook is registered by the
        // sm100 extension at import time on sm_100 devices and validates the
        // shape itself (m == 64, N % 128 == 0), returning false to fall
        // back. It writes dV/dK directly and atomicAdds dQ_tilde, so the
        // split+reduce below is skipped. FLASH_NYSTROM_SM100=0 disables.
        if constexpr (std::is_same_v<scalar_t, cutlass::half_t>) {
            static const bool sm100_enabled = [] {
                const char* env = std::getenv("FLASH_NYSTROM_SM100");
                return env == nullptr || env[0] != '0';
            }();
            if (sm100_enabled && g_kernel3_bwd_sm100_hook != nullptr &&
                g_kernel3_bwd_sm100_hook(q_tilde, k_s, v, lse3, D3, dO3,
                                         dV, dK_s, dQ_tilde,
                                         BH, N, D, m, stream)) {
                FN_CUDA_KERNEL_CHECK();
                return;
            }
        }
        const bool use_split_k = (dQ_tilde_split != nullptr && num_splits > 0);
        float* dqt_target = use_split_k ? dQ_tilde_split : dQ_tilde;
        int    eff_num_splits = use_split_k ? num_splits : 0;
        auto launch = [&](auto HeadDimTag) {
            constexpr int kHeadDim = decltype(HeadDimTag)::value;
            using Traits = K3Traits<kHeadDim, scalar_t>;
            constexpr int Bc = Traits::kBlockN;
            int num_tiles = (N + Bc - 1) / Bc;
            dim3 grid(num_tiles, BH);
            dim3 block(Traits::kNThreads);
            // Auto rule: wide only if it costs no resident CTAs vs narrow.
            // The runtime occupancy query accounts for both SMEM and register
            // limits (wide loses SMEM-bound residency on 100KB/SM consumer
            // parts but is free on A100/H100/B200, where registers bind first).
            // Interleaved A/B on an RTX 5060 and A100/H100 Modal runs both
            // match this rule. Cached once per Traits instantiation.
            static const bool wide_auto = [] {
                if ((size_t)Traits::kSmemBwdWideBytes > 48 * 1024) {
                    cudaFuncSetAttribute((const void*)kernel3_bwd_tc<Traits, true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        Traits::kSmemBwdWideBytes);
                }
                int bw = 0, bn = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bw,
                    (const void*)kernel3_bwd_tc<Traits, true>,
                    Traits::kNThreads, Traits::kSmemBwdWideBytes);
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bn,
                    (const void*)kernel3_bwd_tc<Traits, false>,
                    Traits::kNThreads, Traits::kSmemBwdBytes);
                return bw > 0 && bw >= bn;
            }();
            // FLASH_NYSTROM_BWD_WIDE overrides the auto rule ("0" narrow /
            // "1" wide), for A/B measurement; empty or unset = auto.
            bool wide_ok = wide_auto;
            if (const char* env = std::getenv("FLASH_NYSTROM_BWD_WIDE");
                env != nullptr && env[0] != '\0') {
                wide_ok = (env[0] != '0');
            }
            BOOL_SWITCH(wide_ok, kWide, [&] {
                size_t smem = kWide ? Traits::kSmemBwdWideBytes : Traits::kSmemBwdBytes;
                if (smem > 48 * 1024) {
                    FN_CHECK(smem <= get_max_smem_per_block(), "kernel3_bwd_tc: insufficient smem");
                    FN_CUDA_CHECK(cudaFuncSetAttribute((kernel3_bwd_tc<Traits, kWide>),
                        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
                }
                kernel3_bwd_tc<Traits, kWide><<<grid, block, smem, stream>>>(
                    q_tilde, k_s, v, lse3, D3, dO3, dV, dK_s,
                    dqt_target, eff_num_splits, N, D, m);
            });
        };
        if (D == 64) launch(Int<64>{}); else launch(Int<128>{});
        FN_CUDA_KERNEL_CHECK();

        if (use_split_k) {
            launch_reduce_dQ_tilde_split(
                dQ_tilde_split, dQ_tilde, num_splits, BH, m, D, stream);
        }
    }
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
