/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once
#include "utils.h"
#include "nystrom_utils.h"
#include "kernels/kernel1_output_fused.cuh"

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
    const int tile_idx = blockIdx.x, bh = blockIdx.y, tid = threadIdx.x, nt = blockDim.x;
    constexpr int Br = 64;
    const int rs = tile_idx*Br, re = min(rs+Br, N), tr = re-rs;
    if (tr <= 0) return;
    extern __shared__ char sr[];
    scalar_t* sKt = reinterpret_cast<scalar_t*>(sr);
    scalar_t* sS2 = sKt+m*D; scalar_t* sQ = sS2+m*D; scalar_t* sdO = sQ+Br*D;
    float* sP = reinterpret_cast<float*>(sdO+Br*D);
    auto* kt=k_tilde+bh*m*D; auto* s2=step2+bh*m*D;
    for(int i=tid;i<m*D;i+=nt){sKt[i]=kt[i];sS2[i]=s2[i];}
    auto* q=q_s+bh*N*D+rs*D; auto* d=dO+bh*N*D+rs*D;
    for(int i=tid;i<tr*D;i+=nt){sQ[i]=q[i];sdO[i]=d[i];}
    __syncthreads();
    auto* lse=lse1+bh*N+rs;
    for(int i=tid;i<tr*m;i+=nt){int r=i/m,c=i%m;float dot=0;for(int dd=0;dd<D;dd++)dot+=to_float(sQ[r*D+dd])*to_float(sKt[c*D+dd]);sP[i]=expf(dot-lse[r]);}
    __syncthreads();
    float* ds2=dstep2+bh*m*D;
    for(int i=tid;i<m*D;i+=nt){int j=i/D,dd=i%D;float s=0;for(int ii=0;ii<tr;ii++)s+=sP[ii*m+j]*to_float(sdO[ii*D+dd]);atomicAdd(&ds2[i],s);}
    __syncthreads();
    auto* d1=D1+bh*N+rs;
    for(int i=tid;i<tr*m;i+=nt){int r=i/m,c=i%m;float dp=0;for(int dd=0;dd<D;dd++)dp+=to_float(sdO[r*D+dd])*to_float(sS2[c*D+dd]);sP[i]=sP[i]*(dp-d1[r]);}
    __syncthreads();
    auto* dq=dQ_s+bh*N*D+rs*D;
    for(int i=tid;i<tr*D;i+=nt){int r=i/D,dd=i%D;float s=0;for(int j=0;j<m;j++)s+=sP[r*m+j]*to_float(sKt[j*D+dd]);dq[i]=from_float<scalar_t>(s);}
    float* dkt=dK_tilde+bh*m*D;
    for(int i=tid;i<m*D;i+=nt){int j=i/D,dd=i%D;float s=0;for(int ii=0;ii<tr;ii++)s+=sP[ii*m+j]*to_float(sQ[ii*D+dd]);atomicAdd(&dkt[i],s);}
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


template <typename Traits>
__global__ void __launch_bounds__(Traits::kNThreads)
kernel1_bwd_tc(
    const typename Traits::Element* __restrict__ q_s_ptr,
    const typename Traits::Element* __restrict__ k_tilde_ptr,
    const typename Traits::Element* __restrict__ step2_ptr,
    const float* __restrict__ lse1_ptr,
    const float* __restrict__ D1_ptr,
    const typename Traits::Element* __restrict__ dO_ptr,
    typename Traits::Element* __restrict__ dQ_s_ptr,
    float* __restrict__ dstep2_ptr,
    float* __restrict__ dK_tilde_ptr,
    int N, int D, int m
) {
    using Element = typename Traits::Element;
    constexpr int kBlockM = Traits::kBlockM;
    constexpr int kBlockN = Traits::kBlockN;
    constexpr int kHeadDim = Traits::kHeadDim;

    const int tile_idx = blockIdx.x, bh = blockIdx.y, tidx = threadIdx.x;
    const int row_start = tile_idx * kBlockM;
    if (row_start >= N) return;
    int tile_rows = min(kBlockM, N - row_start);
    bool full = (tile_rows == kBlockM);

    // 3-buffer SMEM: slot (Q or dO) + sKt (K_tilde, persistent) + sS2 (step2, then P/dS)
    extern __shared__ char smem_[];
    Element* slot_ptr = reinterpret_cast<Element*>(smem_);
    Element* sKt_ptr  = slot_ptr + Traits::kSmemQElems;
    Element* sS2_ptr  = sKt_ptr + Traits::kSmemKVElems;

    auto sSlot = make_tensor(make_smem_ptr(slot_ptr), typename Traits::SmemLayoutQ{});
    auto sKt   = make_tensor(make_smem_ptr(sKt_ptr),  typename Traits::SmemLayoutKV{});
    auto sS2   = make_tensor(make_smem_ptr(sS2_ptr),  typename Traits::SmemLayoutKV{});

    // Transposed views for GEMM3 B-operand (Kt) and GEMM4/5 B-operand (slot)
    // kBlockM == kBlockN == 64, so SmemLayoutQ == SmemLayoutKV — reuse KV transposed types
    auto sKtt    = make_tensor(sKt.data(),       typename Traits::SmemLayoutKVtransposed{});
    auto sKttNS  = make_tensor(sKt.data().get(), typename Traits::SmemLayoutKVtransposedNoSwizzle{});
    auto sSlotT  = make_tensor(sSlot.data(),       typename Traits::SmemLayoutKVtransposed{});
    auto sSlotTNS = make_tensor(sSlot.data().get(), typename Traits::SmemLayoutKVtransposedNoSwizzle{});

    // PdS views on sS2 region (SmemLayoutPdS fits: cosize(PdS) = 64*64 <= cosize(KV) = 64*D)
    auto sPdS    = make_tensor(make_smem_ptr(sS2_ptr), typename Traits::SmemLayoutPdS{});
    auto sPdSt   = make_tensor(make_smem_ptr(sS2_ptr), typename Traits::SmemLayoutPdStransposed{});
    auto sPdStNS = make_tensor(make_smem_ptr(sS2_ptr),  typename Traits::SmemLayoutPdStransposedNoSwizzle{});

    // Zero-init all 3 buffers (needed for partial tiles: rows >= tile_rows and cols >= m)
    constexpr int total_elems = Traits::kSmemQElems + Traits::kSmemKVElems*2;
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
    cp_async_fence(); cp_async_wait<0>(); __syncthreads();

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

    // ======== Swap slot: Q → dO ========
    __syncthreads();
    if (full) {
        auto gdO = make_tensor(make_gmem_ptr(dO_ptr+bh*N*D+row_start*D), Shape<Int<kBlockM>,Int<kHeadDim>>{}, Stride<Int<kHeadDim>,_1>{});
        cute::copy(gmem_copy, gmem_thr.partition_S(gdO), gmem_thr.partition_D(sSlot));
    } else {
        auto* d=dO_ptr+bh*N*D+row_start*D;
        for(int i=tidx;i<tile_rows*kHeadDim;i+=Traits::kNThreads){int r=i/kHeadDim,c=i%kHeadDim;if(c<D) sSlot(r,c)=d[r*D+c];}
    }
    cp_async_fence(); cp_async_wait<0>(); __syncthreads();

    // ======== GEMM2: dP = dO @ step2^T (slot=dO, sS2=step2) ========
    auto acc_dp = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_dp);
    gemm_smem(acc_dp, thr_mma.partition_fragment_A(sSlot), thr_mma.partition_fragment_B(sS2),
              thr_copy_A.partition_S(sSlot), thr_copy_B.partition_S(sS2),
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
            float p_val = static_cast<float>(rP_rc(mi, ni));
            dP_rc(mi, ni) = p_val * (dP_rc(mi, ni) - d1v);
        }
    }
    // acc_dp = dS in registers. Eagerly convert to FP16 so the 32 FP32 regs
    // can die before GEMM5 (which allocates a 64-FP32-reg acc_ds2). rdS
    // (16 FP16 regs) is what survives across GEMM5 to the sPdS write below.
    Tensor rdS = convert_type<Element>(acc_dp);
    // sS2 is FREE (step2 no longer needed after GEMM2)

    // ======== Write P (rP, already FP16) to sS2 for GEMM5 ========
    {
        auto sc = make_tiled_copy_C(typename Traits::SmemCopyAtomPdS{}, tiled_mma);
        auto tc = sc.get_thread_slice(tidx);
        cute::copy(sc, tc.retile_S(rP), tc.partition_D(sPdS));
    }
    __syncthreads();

    // ======== GEMM5 TC: dstep2 = P^T @ dO (TiledMmaDKV) ========
    // A = P^T: SmemLayoutPdStransposed on sS2_ptr (LDSM_T)
    // B = dO:  SmemLayoutKVtransposed on slot_ptr (LDSM_T)
    {
        auto acc_ds2 = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
        clear(acc_ds2);
        gemm_smem(acc_ds2,
                  thr_mma_dkv.partition_fragment_A(sPdStNS),
                  thr_mma_dkv.partition_fragment_B(sSlotTNS),
                  thr_copy_PdSt.partition_S(sPdSt),
                  thr_copy_SlotT.partition_S(sSlotT),
                  tiled_mma_dkv, smem_copy_PdSt, smem_copy_SlotT,
                  thr_copy_PdSt, thr_copy_SlotT);

        // atomicAdd results to dstep2 (FP32 global accumulator)
        float* ds2 = dstep2_ptr + bh*m*D;
        #pragma unroll
        for (int fi = 0; fi < size(acc_ds2); fi++) {
            int row = get<0>(tDKVc(fi));
            int col = get<1>(tDKVc(fi));
            if (row < m && col < D)
                atomicAdd(&ds2[row * D + col], acc_ds2(fi));
        }
    }

    // ======== Write dS to sS2 + reload Q into slot (parallel: different SMEM regions) ========
    __syncthreads();  // GEMM5 done reading sS2 (P) and slot (dO)

    // Write dS (rdS, already FP16 from the eager convert above) over P in sS2
    {
        auto sc = make_tiled_copy_C(typename Traits::SmemCopyAtomPdS{}, tiled_mma);
        auto tc = sc.get_thread_slice(tidx);
        cute::copy(sc, tc.retile_S(rdS), tc.partition_D(sPdS));
    }

    // Reload Q into slot (GMEM → SMEM, async or scalar)
    if (full) {
        auto gQ = make_tensor(make_gmem_ptr(q_s_ptr+bh*N*D+row_start*D), Shape<Int<kBlockM>,Int<kHeadDim>>{}, Stride<Int<kHeadDim>,_1>{});
        cute::copy(gmem_copy, gmem_thr.partition_S(gQ), gmem_thr.partition_D(sSlot));
    } else {
        auto* q=q_s_ptr+bh*N*D+row_start*D;
        for(int i=tidx;i<tile_rows*kHeadDim;i+=Traits::kNThreads){int r=i/kHeadDim,c=i%kHeadDim;if(c<D) sSlot(r,c)=q[r*D+c];}
    }
    cp_async_fence(); cp_async_wait<0>(); __syncthreads();

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

        // atomicAdd results to dK_tilde (FP32 global accumulator)
        float* dk = dK_tilde_ptr + bh*m*D;
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
        auto launch = [&](auto HeadDimTag) {
            constexpr int kHeadDim = decltype(HeadDimTag)::value;
            using Traits = K1Traits<kHeadDim, scalar_t>;
            dim3 grid((N+Traits::kBlockM-1)/Traits::kBlockM, BH);
            dim3 block(Traits::kNThreads);
            // 3-buffer layout: slot (Q/dO) + sKt + sS2 (reused for sdS)
            size_t smem = (Traits::kSmemQElems + Traits::kSmemKVElems*2) * sizeof(scalar_t);
            if (smem > 48*1024) {
                FN_CHECK(smem <= get_max_smem_per_block(), "kernel1_bwd: smem");
                FN_CUDA_CHECK(cudaFuncSetAttribute(kernel1_bwd_tc<Traits>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
            }
            kernel1_bwd_tc<Traits><<<grid, block, smem, stream>>>(
                q_s, k_tilde, step2, lse1, D1, dO, dQ_s, dstep2, dK_tilde, N, D, m);
        };
        if (D == 64) launch(Int<64>{}); else launch(Int<128>{});
    }
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
