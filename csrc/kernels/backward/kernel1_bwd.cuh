/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"
#include "nystrom_utils.h"
#include "static_switch.h"
#include "kernels/kernel1_output_fused.cuh"
#include "kernels/backward/kernel3_bwd.cuh"  // reduce_dQ_tilde_split (shared split reducer)

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cutlass/numeric_types.h>

namespace flash_nystrom {

using namespace cute;

// -- launch wrapper --


template <typename scalar_t>
__global__ void kernel1_bwd_scalar_kernel(
    const scalar_t* __restrict__ q_s, const scalar_t* __restrict__ k_tilde,
    const scalar_t* __restrict__ step2, const float* __restrict__ lse1,
    const float* __restrict__ D1, const scalar_t* __restrict__ dO,
    scalar_t* __restrict__ dQ_s, float* __restrict__ dstep2, float* __restrict__ dK_tilde,
    int N, int D, int m
) {
    const int tile_idx = blockIdx.x; const int64_t bh = blockIdx.y;
    const int tid = threadIdx.x, nthreads = blockDim.x;
    constexpr int Br = 64;                          // rows of Q handled per CTA
    const int row_start = tile_idx * Br;
    const int row_end   = min(row_start + Br, N);
    const int tr        = row_end - row_start;      // rows in this tile
    if (tr <= 0) return;

    // SMEM layout: k_tilde (m,D) | step2 (m,D) | Q tile (Br,D) | dO tile (Br,D)
    // | P scratch (tr,m, FP32, reused for the softmax-Jacobian VJP).
    extern __shared__ char smem_raw[];
    scalar_t* sKt = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* sS2 = sKt + m * D;
    scalar_t* sQ  = sS2 + m * D;
    scalar_t* sdO = sQ  + Br * D;
    float*    sP  = reinterpret_cast<float*>(sdO + Br * D);

    // Load k_tilde and step2 (both (m,D)) for this batch-head.
    const scalar_t* kt_bh = k_tilde + bh * m * D;
    const scalar_t* s2_bh = step2   + bh * m * D;
    for (int i = tid; i < m * D; i += nthreads) {
        sKt[i] = kt_bh[i];
        sS2[i] = s2_bh[i];
    }
    // Load this tile's Q and dO rows ((tr,D)).
    const scalar_t* q_bh  = q_s + bh * N * D + row_start * D;
    const scalar_t* dO_bh = dO  + bh * N * D + row_start * D;
    for (int i = tid; i < tr * D; i += nthreads) {
        sQ[i]  = q_bh[i];
        sdO[i] = dO_bh[i];
    }
    __syncthreads();

    // P1[r,c] = softmax1 prob = exp(<Q[r], k_tilde[c]> - lse1[r]).
    const float* lse_bh = lse1 + bh * N + row_start;
    for (int i = tid; i < tr * m; i += nthreads) {
        const int r = i / m, c = i % m;
        float dot = 0.0f;
        for (int dd = 0; dd < D; dd++)
            dot += to_float(sQ[r * D + dd]) * to_float(sKt[c * D + dd]);
        sP[i] = expf(dot - lse_bh[r]);
    }
    __syncthreads();

    // dstep2 += P1^T @ dO (accumulated across row tiles -> atomicAdd into GMEM).
    float* dstep2_bh = dstep2 + bh * m * D;
    for (int i = tid; i < m * D; i += nthreads) {
        const int j = i / D, dd = i % D;
        float acc = 0.0f;
        for (int r = 0; r < tr; r++)
            acc += sP[r * m + j] * to_float(sdO[r * D + dd]);
        atomicAdd(&dstep2_bh[i], acc);
    }
    __syncthreads();

    // Softmax-Jacobian VJP, in place: dS1[r,c] = P1[r,c]*(<dO[r], step2[c]> - D1[r]).
    const float* D1_bh = D1 + bh * N + row_start;
    for (int i = tid; i < tr * m; i += nthreads) {
        const int r = i / m, c = i % m;
        float dp = 0.0f;
        for (int dd = 0; dd < D; dd++)
            dp += to_float(sdO[r * D + dd]) * to_float(sS2[c * D + dd]);
        sP[i] = sP[i] * (dp - D1_bh[r]);
    }
    __syncthreads();

    // dQ_s = dS1 @ k_tilde (one write per element, no atomics).
    scalar_t* dQ_bh = dQ_s + bh * N * D + row_start * D;
    for (int i = tid; i < tr * D; i += nthreads) {
        const int r = i / D, dd = i % D;
        float acc = 0.0f;
        for (int j = 0; j < m; j++)
            acc += sP[r * m + j] * to_float(sKt[j * D + dd]);
        dQ_bh[i] = from_float<scalar_t>(acc);
    }
    // dK_tilde += dS1^T @ Q_s (accumulated across row tiles -> atomicAdd into GMEM).
    float* dKt_bh = dK_tilde + bh * m * D;
    for (int i = tid; i < m * D; i += nthreads) {
        const int j = i / D, dd = i % D;
        float acc = 0.0f;
        for (int r = 0; r < tr; r++)
            acc += sP[r * m + j] * to_float(sQ[r * D + dd]);
        atomicAdd(&dKt_bh[i], acc);
    }
}

// -- tensor core kernel (fp16/bf16) --
//
// ALL 5 GEMMs use tensor cores.
//
// 3-buffer SMEM layout: slot (Q or dO) + sKt + sS2 (step2, then P/dS)
// D=128: 16KB + 16KB + 16KB = 48KB — fits 2 blocks/SM without opt-in
//
// TiledMma (4,1,1):    GEMM1 (S=Q@Kt), GEMM2 (dP=dO@S2), GEMM3 (dQ=dS@Kt)
// TiledMmaDKV (2,2,1): GEMM4 (dKt=dS^T@Q), GEMM5 (dS2=P^T@dO)
//
// Timeline:
//  GEMM1 → softmax → swap(Q→dO) → GEMM2 → softmax_bwd
//  → write P to sS2 → GEMM5(TC) → atomicAdd dstep2
//  → write dS to sS2 + reload Q → GEMM4(TC) → atomicAdd dKt
//  → GEMM3(TC) → write dQ


template <typename Traits, bool kWide>
__global__ void __launch_bounds__(Traits::kNThreads)
kernel1_bwd_tc(
    const typename Traits::Element* __restrict__ q_s_ptr,
    const typename Traits::Element* __restrict__ k_tilde_ptr,
    const typename Traits::Element* __restrict__ step2_ptr,
    const float* __restrict__ lse1_ptr,
    const float* __restrict__ D1_ptr,
    const typename Traits::Element* __restrict__ dO_ptr,
    typename Traits::Element* __restrict__ dQ_s_ptr,
    // dstep2 / dK_tilde targets. With num_splits == 0 these are the legacy
    // (BH, m, D) accumulators and every row tile atomicAdds the same cells
    // (N/64-way contention). With num_splits > 0 they are
    // (num_splits, BH, m, D) split workspaces: tile_idx writes slot
    // (tile_idx % num_splits), cutting contention num_splits-fold; the
    // launcher reduces the slots afterwards (same scheme as kernel3_bwd's
    // dQ_tilde split).
    float* __restrict__ dstep2_ptr,
    float* __restrict__ dK_tilde_ptr,
    int num_splits,
    int N, int D, int m
) {
    using Element = typename Traits::Element;
    constexpr int kBlockM = Traits::kBlockM;
    constexpr int kBlockN = Traits::kBlockN;
    constexpr int kHeadDim = Traits::kHeadDim;

    const int tile_idx = blockIdx.x, tidx = threadIdx.x;
    const int64_t bh = blockIdx.y;
    const int row_start = tile_idx * kBlockM;
    if (row_start >= N) return;
    int tile_rows = min(kBlockM, N - row_start);
    bool full = (tile_rows == kBlockM);

    // SMEM: slot (Q) + sKt (K_tilde, persistent) + sS2 (step2, then P/dS).
    // Narrow mode time-multiplexes slot between Q and dO, forcing a mid-kernel
    // swap stall plus a second Q load from GMEM. Wide mode adds a dedicated dO
    // buffer: dO's load is issued in the prologue and completes under GEMM1,
    // and both the swap and the Q reload disappear.
    extern __shared__ char smem_[];
    Element* slot_ptr = reinterpret_cast<Element*>(smem_);
    Element* sKt_ptr  = slot_ptr + Traits::kSmemQElems;
    Element* sS2_ptr  = sKt_ptr + Traits::kSmemKVElems;
    Element* sdO_ptr  = kWide ? sS2_ptr + Traits::kSmemKVElems : slot_ptr;

    auto sSlot = make_tensor(make_smem_ptr(slot_ptr), typename Traits::SmemLayoutQ{});
    auto sKt   = make_tensor(make_smem_ptr(sKt_ptr),  typename Traits::SmemLayoutKV{});
    auto sS2   = make_tensor(make_smem_ptr(sS2_ptr),  typename Traits::SmemLayoutKV{});
    auto sdO   = make_tensor(make_smem_ptr(sdO_ptr),  typename Traits::SmemLayoutQ{});

    // Transposed views for GEMM3 B-operand (Kt), GEMM4 B-operand (Q in slot),
    // and GEMM5 B-operand (dO). kBlockM == kBlockN == 64, so SmemLayoutQ ==
    // SmemLayoutKV — reuse KV transposed types. sdO* alias sSlot* when narrow.
    auto sKtt    = make_tensor(sKt.data(),       typename Traits::SmemLayoutKVtransposed{});
    auto sKttNS  = make_tensor(sKt.data().get(), typename Traits::SmemLayoutKVtransposedNoSwizzle{});
    auto sSlotT  = make_tensor(sSlot.data(),       typename Traits::SmemLayoutKVtransposed{});
    auto sSlotTNS = make_tensor(sSlot.data().get(), typename Traits::SmemLayoutKVtransposedNoSwizzle{});
    auto sdOT    = make_tensor(sdO.data(),       typename Traits::SmemLayoutKVtransposed{});
    auto sdOTNS  = make_tensor(sdO.data().get(), typename Traits::SmemLayoutKVtransposedNoSwizzle{});

    // PdS views on sS2 region (SmemLayoutPdS fits: cosize(PdS) = 64*64 <= cosize(KV) = 64*D)
    auto sPdS    = make_tensor(make_smem_ptr(sS2_ptr), typename Traits::SmemLayoutPdS{});
    auto sPdSt   = make_tensor(make_smem_ptr(sS2_ptr), typename Traits::SmemLayoutPdStransposed{});
    auto sPdStNS = make_tensor(make_smem_ptr(sS2_ptr),  typename Traits::SmemLayoutPdStransposedNoSwizzle{});

    // Zero-init all buffers (needed for partial tiles: rows >= tile_rows and cols >= m)
    constexpr int total_elems = Traits::kSmemQElems + Traits::kSmemKVElems*2
                              + (kWide ? Traits::kSmemQElems : 0);
    for (int i = tidx; i < total_elems; i += Traits::kNThreads)
        reinterpret_cast<Element*>(smem_)[i] = Element(0);
    __syncthreads();

    // Load persistent buffers: K_tilde and step2
    typename Traits::GmemTiledCopy gmem_copy;
    auto gmem_thr = gmem_copy.get_thread_slice(tidx);

    if (m == kBlockN) {
        auto gKt = make_tensor(make_gmem_ptr(k_tilde_ptr+bh*m*D), Shape<Int<kBlockN>,Int<kHeadDim>>{}, Stride<Int<kHeadDim>,_1>{});
        auto gS2 = make_tensor(make_gmem_ptr(step2_ptr+bh*m*D),   Shape<Int<kBlockN>,Int<kHeadDim>>{}, Stride<Int<kHeadDim>,_1>{});
        cute::copy(gmem_copy, gmem_thr.partition_S(gKt), gmem_thr.partition_D(sKt));
        cute::copy(gmem_copy, gmem_thr.partition_S(gS2), gmem_thr.partition_D(sS2));
    } else {
        auto* kt=k_tilde_ptr+bh*m*D; auto* s2=step2_ptr+bh*m*D;
        for(int i=tidx;i<m*kHeadDim;i+=Traits::kNThreads){int r=i/kHeadDim,c=i%kHeadDim;if(c<D){sKt(r,c)=kt[r*D+c];sS2(r,c)=s2[r*D+c];}}
    }

    // Load Q into slot
    if (full) {
        auto gQ = make_tensor(make_gmem_ptr(q_s_ptr+bh*N*D+row_start*D), Shape<Int<kBlockM>,Int<kHeadDim>>{}, Stride<Int<kHeadDim>,_1>{});
        cute::copy(gmem_copy, gmem_thr.partition_S(gQ), gmem_thr.partition_D(sSlot));
    } else {
        auto* q=q_s_ptr+bh*N*D+row_start*D;
        for(int i=tidx;i<tile_rows*kHeadDim;i+=Traits::kNThreads){int r=i/kHeadDim,c=i%kHeadDim;if(c<D) sSlot(r,c)=q[r*D+c];}
    }
    if constexpr (kWide) {
        cp_async_fence();  // group 1: Kt + step2 + Q (GEMM1's operands)
        // Issue dO now; it completes under GEMM1 and is waited before GEMM2.
        if (full) {
            auto gdO = make_tensor(make_gmem_ptr(dO_ptr+bh*N*D+row_start*D), Shape<Int<kBlockM>,Int<kHeadDim>>{}, Stride<Int<kHeadDim>,_1>{});
            cute::copy(gmem_copy, gmem_thr.partition_S(gdO), gmem_thr.partition_D(sdO));
        } else {
            auto* d=dO_ptr+bh*N*D+row_start*D;
            for(int i=tidx;i<tile_rows*kHeadDim;i+=Traits::kNThreads){int r=i/kHeadDim,c=i%kHeadDim;if(c<D) sdO(r,c)=d[r*D+c];}
        }
        cp_async_fence();  // group 2: dO
        cp_async_wait<1>();  // group 1 done; dO may still be in flight
    } else {
        cp_async_fence(); cp_async_wait<0>();
    }
    __syncthreads();

    // ======== TiledMma setup (GEMM1/2/3: kBlockM × kBlockN output) ========
    typename Traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);

    auto smem_copy_A  = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto thr_copy_A   = smem_copy_A.get_thread_slice(tidx);
    auto smem_copy_B  = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto thr_copy_B   = smem_copy_B.get_thread_slice(tidx);
    auto smem_copy_Bt = make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto thr_copy_Bt  = smem_copy_Bt.get_thread_slice(tidx);

    // Identity tensor for physical row/col of (kBlockM, kBlockN) accumulators
    auto cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    auto tScS = thr_mma.partition_C(cS);
    auto tScS_rc = make_tensor(tScS.data(), convert_layout_acc_rowcol(tScS.layout()));

    // ======== TiledMmaDKV setup (GEMM4/5: kBlockN × kHeadDim output) ========
    typename Traits::TiledMmaDKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);

    // A-operand: P^T or dS^T from SmemLayoutPdStransposed (LDSM_T)
    auto smem_copy_PdSt = make_tiled_copy_A(typename Traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto thr_copy_PdSt  = smem_copy_PdSt.get_thread_slice(tidx);

    // B-operand: Q or dO from SmemLayoutKVtransposed (LDSM_T)
    // kBlockM == kBlockN, so SmemLayoutQ == SmemLayoutKV — transposed views are compatible
    auto smem_copy_SlotT = make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto thr_copy_SlotT  = smem_copy_SlotT.get_thread_slice(tidx);

    // Identity tensor for (kBlockN, kHeadDim) accumulators — maps fragments to physical indices
    auto cDKV = make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
    auto tDKVc = thr_mma_dkv.partition_C(cDKV);

    // ======== GEMM1: S = Q @ Kt^T (slot has Q) ========
    auto acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);
    gemm_smem(acc_s, thr_mma.partition_fragment_A(sSlot), thr_mma.partition_fragment_B(sKt),
              thr_copy_A.partition_S(sSlot), thr_copy_B.partition_S(sKt),
              tiled_mma, smem_copy_A, smem_copy_B, thr_copy_A, thr_copy_B);

    // P = exp(S - LSE), mask cols >= m and rows >= tile_rows
    auto scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    constexpr int nrow = decltype(size<0>(scores))::value;

    const float* lse = lse1_ptr + bh*N + row_start;
    #pragma unroll
    for (int mi = 0; mi < nrow; mi++) {
        int pr = get<0>(tScS_rc(mi, 0));
        float lse_val = (pr < tile_rows) ? lse[pr] : 0.0f;
        #pragma unroll
        for (int ni = 0; ni < size<1>(scores); ni++) {
            int pc = get<1>(tScS_rc(mi, ni));
            scores(mi, ni) = (pc < m && pr < tile_rows) ? expf(scores(mi, ni) - lse_val) : 0.0f;
        }
    }
    // Eagerly convert acc_s (P, FP32) to FP16 so the 32 high FP32 regs can
    // die before GEMM2 (which needs B-fragment registers) and softmax_bwd
    // (which reads P alongside acc_dp). The FP32->FP16 round here is the
    // same precision boundary we cross at the sPdS write anyway, just
    // performed sooner. SASS check on RTX 5060 Laptop showed P occupying
    // R140-R171; this refactor frees those 32 high registers.
    Tensor rP = convert_type<Element>(acc_s);
    auto rP_rc = make_tensor(rP.data(), convert_layout_acc_rowcol(acc_s.layout()));

    // ======== dO becomes readable ========
    if constexpr (kWide) {
        // dO was issued in the prologue and has been loading under GEMM1.
        cp_async_wait<0>();
        __syncthreads();
    } else {
        // Narrow: swap slot Q -> dO (Q is reloaded later for GEMM4).
        __syncthreads();
        if (full) {
            auto gdO = make_tensor(make_gmem_ptr(dO_ptr+bh*N*D+row_start*D), Shape<Int<kBlockM>,Int<kHeadDim>>{}, Stride<Int<kHeadDim>,_1>{});
            cute::copy(gmem_copy, gmem_thr.partition_S(gdO), gmem_thr.partition_D(sdO));
        } else {
            auto* d=dO_ptr+bh*N*D+row_start*D;
            for(int i=tidx;i<tile_rows*kHeadDim;i+=Traits::kNThreads){int r=i/kHeadDim,c=i%kHeadDim;if(c<D) sdO(r,c)=d[r*D+c];}
        }
        cp_async_fence(); cp_async_wait<0>(); __syncthreads();
    }

    // ======== GEMM2: dP = dO @ step2^T (sdO=dO, sS2=step2) ========
    auto acc_dp = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_dp);
    gemm_smem(acc_dp, thr_mma.partition_fragment_A(sdO), thr_mma.partition_fragment_B(sS2),
              thr_copy_A.partition_S(sdO), thr_copy_B.partition_S(sS2),
              tiled_mma, smem_copy_A, smem_copy_B, thr_copy_A, thr_copy_B);

    // Softmax backward: dS = P * (dP - D1). P comes from rP (FP16) and is
    // upconverted per element; the FP32 acc_s registers were freed after
    // the eager convert_type above.
    auto dP_rc = make_tensor(acc_dp.data(), convert_layout_acc_rowcol(acc_dp.layout()));
    const float* d1 = D1_ptr + bh*N + row_start;
    #pragma unroll
    for (int mi = 0; mi < nrow; mi++) {
        int pr = get<0>(tScS_rc(mi, 0));
        float d1v = (pr < tile_rows) ? d1[pr] : 0.0f;
        #pragma unroll
        for (int ni = 0; ni < size<1>(dP_rc); ni++) {
            // Use the FP32 P (acc_s/scores), NOT the eagerly-downcast fp16 rP:
            // the softmax-Jacobian cancellation (dP - D) needs full-precision P
            // or a small systematic bias compounds over depth and collapses
            // training. dS is still downcast for the GEMM below (TC unchanged).
            float p_val = scores(mi, ni);
            dP_rc(mi, ni) = p_val * (dP_rc(mi, ni) - d1v);
        }
    }
    // acc_dp = dS in registers. Eagerly convert to FP16 so the 32 FP32 regs
    // can die before GEMM5 (which allocates a 64-FP32-reg acc_ds2). rdS
    // (16 FP16 regs) is what survives across GEMM5 to the sPdS write below.
    Tensor rdS = convert_type<Element>(acc_dp);
    // sS2 held step2 as GEMM2's B-operand. Without a barrier here a warp that
    // finished GEMM2 races ahead and overwrites sS2 with P (the cute::copy
    // below) while a lagging warp is still loading step2 fragments from sS2 in
    // GEMM2's mainloop: a WAR hazard on sS2 that compute-sanitizer racecheck
    // flags and that corrupts dstep2/dK_tilde at large N (multi-CTA occupancy).
    __syncthreads();

    // ======== Write P (rP, already FP16) to sS2 for GEMM5 ========
    {
        auto sc = make_tiled_copy_C(typename Traits::SmemCopyAtomPdS{}, tiled_mma);
        auto tc = sc.get_thread_slice(tidx);
        cute::copy(sc, tc.retile_S(rP), tc.partition_D(sPdS));
    }
    __syncthreads();

    // ======== GEMM5 TC: dstep2 = P^T @ dO (TiledMmaDKV) ========
    // A = P^T: SmemLayoutPdStransposed on sS2_ptr (LDSM_T)
    // B = dO:  SmemLayoutKVtransposed on sdO_ptr (LDSM_T; aliases slot when narrow)
    {
        auto acc_ds2 = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
        clear(acc_ds2);
        gemm_smem(acc_ds2,
                  thr_mma_dkv.partition_fragment_A(sPdStNS),
                  thr_mma_dkv.partition_fragment_B(sdOTNS),
                  thr_copy_PdSt.partition_S(sPdSt),
                  thr_copy_SlotT.partition_S(sdOT),
                  tiled_mma_dkv, smem_copy_PdSt, smem_copy_SlotT,
                  thr_copy_PdSt, thr_copy_SlotT);

        // atomicAdd results to dstep2 (FP32 accumulator; split slot when
        // num_splits > 0, same slot math as kernel3_bwd's dQ_tilde split).
        const int split_idx = (num_splits > 0) ? (tile_idx % num_splits) : 0;
        const long long bh_slot_stride = static_cast<long long>(m) * D;
        const long long split_slot_stride =
            static_cast<long long>(gridDim.y) * bh_slot_stride;
        float* ds2 = dstep2_ptr
                   + static_cast<long long>(split_idx) * split_slot_stride
                   + static_cast<long long>(bh) * bh_slot_stride;
        #pragma unroll
        for (int fi = 0; fi < size(acc_ds2); fi++) {
            int row = get<0>(tDKVc(fi));
            int col = get<1>(tDKVc(fi));
            if (row < m && col < D)
                atomicAdd(&ds2[row * D + col], acc_ds2(fi));
        }
    }

    // ======== Write dS to sS2 (+ reload Q into slot when narrow) ========
    __syncthreads();  // GEMM5 done reading sS2 (P) and sdO (dO)

    // Write dS (rdS, already FP16 from the eager convert above) over P in sS2
    {
        auto sc = make_tiled_copy_C(typename Traits::SmemCopyAtomPdS{}, tiled_mma);
        auto tc = sc.get_thread_slice(tidx);
        cute::copy(sc, tc.retile_S(rdS), tc.partition_D(sPdS));
    }

    if constexpr (!kWide) {
        // Narrow: reload Q into slot (dO evicted it). Wide keeps Q resident.
        if (full) {
            auto gQ = make_tensor(make_gmem_ptr(q_s_ptr+bh*N*D+row_start*D), Shape<Int<kBlockM>,Int<kHeadDim>>{}, Stride<Int<kHeadDim>,_1>{});
            cute::copy(gmem_copy, gmem_thr.partition_S(gQ), gmem_thr.partition_D(sSlot));
        } else {
            auto* q=q_s_ptr+bh*N*D+row_start*D;
            for(int i=tidx;i<tile_rows*kHeadDim;i+=Traits::kNThreads){int r=i/kHeadDim,c=i%kHeadDim;if(c<D) sSlot(r,c)=q[r*D+c];}
        }
        cp_async_fence(); cp_async_wait<0>();
    }
    __syncthreads();  // dS visible to GEMM4/GEMM3 readers (+ Q reloaded when narrow)

    // ======== GEMM4 TC: dKt = dS^T @ Q (TiledMmaDKV) ========
    // A = dS^T: SmemLayoutPdStransposed on sS2_ptr (LDSM_T)
    // B = Q:    SmemLayoutKVtransposed on slot_ptr (LDSM_T)
    {
        auto acc_dkt = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
        clear(acc_dkt);
        gemm_smem(acc_dkt,
                  thr_mma_dkv.partition_fragment_A(sPdStNS),
                  thr_mma_dkv.partition_fragment_B(sSlotTNS),
                  thr_copy_PdSt.partition_S(sPdSt),
                  thr_copy_SlotT.partition_S(sSlotT),
                  tiled_mma_dkv, smem_copy_PdSt, smem_copy_SlotT,
                  thr_copy_PdSt, thr_copy_SlotT);

        // atomicAdd results to dK_tilde (FP32 accumulator; split slot when
        // num_splits > 0, mirroring the dstep2 slot write above).
        const int split_idx = (num_splits > 0) ? (tile_idx % num_splits) : 0;
        const long long bh_slot_stride = static_cast<long long>(m) * D;
        const long long split_slot_stride =
            static_cast<long long>(gridDim.y) * bh_slot_stride;
        float* dk = dK_tilde_ptr
                  + static_cast<long long>(split_idx) * split_slot_stride
                  + static_cast<long long>(bh) * bh_slot_stride;
        #pragma unroll
        for (int fi = 0; fi < size(acc_dkt); fi++) {
            int row = get<0>(tDKVc(fi));
            int col = get<1>(tDKVc(fi));
            if (row < m && col < D)
                atomicAdd(&dk[row * D + col], acc_dkt(fi));
        }
    }

    // ======== GEMM3 TC: dQ = dS @ Kt (TiledMma) ========
    // A = dS: SmemLayoutPdS on sS2_ptr (LDSM_N, non-transposed)
    // B = Kt: SmemLayoutKVtransposed on sKt_ptr (LDSM_T)
    auto acc_dq = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_dq);
    gemm_smem(acc_dq, thr_mma.partition_fragment_A(sPdS), thr_mma.partition_fragment_B(sKttNS),
              thr_copy_A.partition_S(sPdS), thr_copy_Bt.partition_S(sKtt),
              tiled_mma, smem_copy_A, smem_copy_Bt, thr_copy_A, thr_copy_Bt);

    // ======== Write dQ to GMEM via slot ========
    // GEMM4 above reads Q from slot; the rdQ store below overwrites slot. With
    // GEMM3 (which does not touch slot) the only thing in between, a warp that
    // finishes GEMM4+GEMM3 races ahead and clobbers slot with dQ while a
    // lagging warp is still loading Q in GEMM4's mainloop: a WAR on slot that
    // compute-sanitizer racecheck flags (large-N multi-CTA occupancy).
    __syncthreads();
    {
        auto rdQ = convert_type<Element>(acc_dq);
        auto sc = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{}, tiled_mma);
        auto tc = sc.get_thread_slice(tidx);
        cute::copy(sc, tc.retile_S(rdQ), tc.partition_D(sSlot));
        __syncthreads();
        Element* dq = dQ_s_ptr + bh*N*D + row_start*D;
        if (full) {
            auto gdQ = make_tensor(make_gmem_ptr(dq), Shape<Int<kBlockM>,Int<kHeadDim>>{}, Stride<Int<kHeadDim>,_1>{});
            auto gc = make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                typename Traits::GmemLayoutAtom{}, Layout<Shape<_1, Int<Traits::kGmemElemsPerLoad>>>{});
            auto gt = gc.get_thread_slice(tidx);
            cute::copy(gc, gt.partition_S(sSlot), gt.partition_D(gdQ));
        } else {
            for (int i=tidx; i<tile_rows*D; i+=Traits::kNThreads) { int r=i/D,c=i%D; dq[i]=sSlot(r,c); }
        }
    }
}

// -- launch wrapper --


template <typename scalar_t>
void launch_kernel1_bwd(
    const scalar_t* q_s, const scalar_t* k_tilde, const scalar_t* step2,
    const float* lse1, const float* D1, const scalar_t* dO,
    scalar_t* dQ_s, float* dstep2, float* dK_tilde,
    // Split-K workspaces (k1_num_splits, BH, m, D) FP32 for the two m*D
    // accumulators, or nullptr / 0 for the legacy direct atomicAdd. Only the
    // TC path uses them (reduced into dstep2 / dK_tilde before returning);
    // the FP32 scalar path keeps direct atomics.
    float* dstep2_split, float* dK_tilde_split, int k1_num_splits,
    int BH, int N, int D, int m, cudaStream_t stream
) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        // FP32: scalar
        constexpr int Br = 64;
        dim3 grid((N+Br-1)/Br, BH); dim3 block(256);
        size_t smem = m*D*4*2 + Br*D*4*2 + Br*m*4 + 32;
        if (smem > 48*1024) {
            FN_CHECK(smem <= get_max_smem_per_block(), "kernel1_bwd: smem");
            FN_CUDA_CHECK(cudaFuncSetAttribute(kernel1_bwd_scalar_kernel<float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        }
        kernel1_bwd_scalar_kernel<float><<<grid, block, smem, stream>>>(
            q_s, k_tilde, step2, lse1, D1, dO, dQ_s, dstep2, dK_tilde, N, D, m);
    } else {
        // FP16/BF16: tensor cores
        FN_CHECK(D == 64 || D == 128, "kernel1_bwd: D must be 64 or 128");
        const bool use_split =
            (dstep2_split != nullptr && dK_tilde_split != nullptr && k1_num_splits > 0);
        float* ds2_target = use_split ? dstep2_split : dstep2;
        float* dkt_target = use_split ? dK_tilde_split : dK_tilde;
        const int eff_splits = use_split ? k1_num_splits : 0;
        auto launch = [&](auto HeadDimTag) {
            constexpr int kHeadDim = decltype(HeadDimTag)::value;
            using Traits = K1Traits<kHeadDim, scalar_t>;
            dim3 grid((N+Traits::kBlockM-1)/Traits::kBlockM, BH);
            dim3 block(Traits::kNThreads);
            // Narrow: slot (Q/dO time-multiplexed) + sKt + sS2. Wide adds a
            // dedicated dO buffer (no swap stall, no second Q load); use it
            // only where it keeps at least 2 CTAs resident per SM.
            constexpr size_t kNarrowBytes =
                (Traits::kSmemQElems + Traits::kSmemKVElems*2) * sizeof(scalar_t);
            constexpr size_t kWideBytes =
                (Traits::kSmemQElems*2 + Traits::kSmemKVElems*2) * sizeof(scalar_t);
            // Auto rule: wide only if it costs no resident CTAs vs narrow (the
            // runtime occupancy query accounts for SMEM and register limits;
            // see kernel3_bwd for the measurement basis). Cached per Traits.
            static const bool wide_auto = [] {
                // Recomputed here (not captured): MSVC rejects implicit
                // capture of the enclosing constexpr locals (C3493).
                constexpr size_t wide_b =
                    (Traits::kSmemQElems*2 + Traits::kSmemKVElems*2) * sizeof(scalar_t);
                constexpr size_t narrow_b =
                    (Traits::kSmemQElems + Traits::kSmemKVElems*2) * sizeof(scalar_t);
                if (wide_b > 48 * 1024) {
                    cudaFuncSetAttribute((const void*)kernel1_bwd_tc<Traits, true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(wide_b));
                }
                int bw = 0, bn = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bw,
                    (const void*)kernel1_bwd_tc<Traits, true>,
                    Traits::kNThreads, wide_b);
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bn,
                    (const void*)kernel1_bwd_tc<Traits, false>,
                    Traits::kNThreads, narrow_b);
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
                size_t smem = kWide ? kWideBytes : kNarrowBytes;
                if (smem > 48*1024) {
                    FN_CHECK(smem <= get_max_smem_per_block(), "kernel1_bwd: smem");
                    FN_CUDA_CHECK(cudaFuncSetAttribute((kernel1_bwd_tc<Traits, kWide>),
                        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
                }
                kernel1_bwd_tc<Traits, kWide><<<grid, block, smem, stream>>>(
                    q_s, k_tilde, step2, lse1, D1, dO, dQ_s,
                    ds2_target, dkt_target, eff_splits, N, D, m);
            });
        };
        if (D == 64) launch(Int<64>{}); else launch(Int<128>{});
        FN_CUDA_KERNEL_CHECK();
        if (use_split) {
            // Sum the split slots into the real accumulators (the reducer is
            // layout-generic over (num_splits, BH, m*D) -> (BH, m*D)).
            launch_reduce_dQ_tilde_split(dstep2_split, dstep2,
                                         eff_splits, BH, m, D, stream);
            launch_reduce_dQ_tilde_split(dK_tilde_split, dK_tilde,
                                         eff_splits, BH, m, D, stream);
        }
    }
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
