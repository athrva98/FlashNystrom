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

    extern __shared__ char smem_[];
    Element* sQ_ptr  = reinterpret_cast<Element*>(smem_);
    Element* sKt_ptr = sQ_ptr  + Traits::kSmemQElems;
    Element* sdO_ptr = sKt_ptr + Traits::kSmemKVElems;
    Element* sS2_ptr = sdO_ptr + Traits::kSmemQElems;
    // sP and sdS overlay sS2's space (sS2 is no longer needed after GEMM2)
    // sS2 has kSmemKVElems elems = 64*D. sP+sdS need 2*cosize(SmemLayoutPdS) = 2*64*64.
    // For D>=64: 64*D >= 2*64*64 iff D >= 128. For D=64: 64*64 < 2*64*64, need separate.
    int kPdSElems = static_cast<int>(cosize(typename Traits::SmemLayoutPdS{}));
    Element* sP_ptr;
    Element* sdS_ptr;
    if constexpr (Traits::kHeadDim >= 128) {
        // overlay on sS2 (saves 2*kPdSElems elems = 16KB)
        sP_ptr  = sS2_ptr;
        sdS_ptr = sS2_ptr + kPdSElems;
    } else {
        // D=64: sS2 too small, allocate after
        sP_ptr  = sS2_ptr + Traits::kSmemKVElems;
        sdS_ptr = sP_ptr  + kPdSElems;
    }

    auto sQ  = make_tensor(make_smem_ptr(sQ_ptr),  typename Traits::SmemLayoutQ{});
    auto sKt = make_tensor(make_smem_ptr(sKt_ptr), typename Traits::SmemLayoutKV{});
    auto sdO = make_tensor(make_smem_ptr(sdO_ptr), typename Traits::SmemLayoutQ{});
    auto sS2 = make_tensor(make_smem_ptr(sS2_ptr), typename Traits::SmemLayoutKV{});
    auto sP  = make_tensor(make_smem_ptr(sP_ptr),  typename Traits::SmemLayoutPdS{});
    auto sdS = make_tensor(make_smem_ptr(sdS_ptr), typename Traits::SmemLayoutPdS{});

    // Transposed views
    auto sKtt  = make_tensor(sKt.data(),  typename Traits::SmemLayoutKVtransposed{});
    auto sKttNS = make_tensor(sKt.data().get(), typename Traits::SmemLayoutKVtransposedNoSwizzle{});
    auto sQt   = make_tensor(sQ.data(),   typename Traits::SmemLayoutKVtransposed{});
    auto sQtNS = make_tensor(sQ.data().get(), typename Traits::SmemLayoutKVtransposedNoSwizzle{});
    auto sdOt  = make_tensor(sdO.data(),  typename Traits::SmemLayoutKVtransposed{});
    auto sdOtNS = make_tensor(sdO.data().get(), typename Traits::SmemLayoutKVtransposedNoSwizzle{});
    auto sPt   = make_tensor(sP.data(),   typename Traits::SmemLayoutPdStransposed{});
    auto sPtNS = make_tensor(sP.data().get(), typename Traits::SmemLayoutPdStransposedNoSwizzle{});
    auto sdSt  = make_tensor(sdS.data(),  typename Traits::SmemLayoutPdStransposed{});
    auto sdStNS = make_tensor(sdS.data().get(), typename Traits::SmemLayoutPdStransposedNoSwizzle{});

    // Zero-init SMEM
    int total_elems;
    if constexpr (Traits::kHeadDim >= 128) {
        // sP+sdS overlay sS2, so total = sQ + sKt + sdO + sS2 (sP/sdS inside sS2)
        total_elems = Traits::kSmemQElems*2 + Traits::kSmemKVElems*2;
    } else {
        total_elems = Traits::kSmemQElems*2 + Traits::kSmemKVElems*2 + kPdSElems*2;
    }
    for (int i = tidx; i < total_elems; i += Traits::kNThreads)
        reinterpret_cast<Element*>(smem_)[i] = Element(0);
    __syncthreads();

    // Load data to SMEM
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
    if (full) {
        auto gQ  = make_tensor(make_gmem_ptr(q_s_ptr+bh*N*D+row_start*D), Shape<Int<kBlockM>,Int<kHeadDim>>{}, Stride<Int<kHeadDim>,_1>{});
        auto gdO = make_tensor(make_gmem_ptr(dO_ptr+bh*N*D+row_start*D),  Shape<Int<kBlockM>,Int<kHeadDim>>{}, Stride<Int<kHeadDim>,_1>{});
        cute::copy(gmem_copy, gmem_thr.partition_S(gQ),  gmem_thr.partition_D(sQ));
        cute::copy(gmem_copy, gmem_thr.partition_S(gdO), gmem_thr.partition_D(sdO));
    } else {
        auto* q=q_s_ptr+bh*N*D+row_start*D; auto* d=dO_ptr+bh*N*D+row_start*D;
        for(int i=tidx;i<tile_rows*kHeadDim;i+=Traits::kNThreads){int r=i/kHeadDim,c=i%kHeadDim;if(c<D){sQ(r,c)=q[r*D+c];sdO(r,c)=d[r*D+c];}}
    }
    cp_async_fence(); cp_async_wait<0>(); __syncthreads();

    // MMA setup — primary TiledMma for S/dP (kBlockM × kBlockN)
    typename Traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    auto smem_copy_A  = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto thr_copy_A   = smem_copy_A.get_thread_slice(tidx);
    auto smem_copy_B  = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto thr_copy_B   = smem_copy_B.get_thread_slice(tidx);
    // For GEMM3: B uses transposed view of Kt (kHeadDim, kBlockN)
    auto smem_copy_Bt = make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto thr_copy_Bt  = smem_copy_Bt.get_thread_slice(tidx);

    // Identity tensor for physical row extraction
    auto cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    auto tScS = thr_mma.partition_C(cS);
    auto tScS_rc = make_tensor(tScS.data(), convert_layout_acc_rowcol(tScS.layout()));

    // GEMM1: S = Q @ Kt^T
    auto acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);
    gemm_smem(acc_s, thr_mma.partition_fragment_A(sQ), thr_mma.partition_fragment_B(sKt),
              thr_copy_A.partition_S(sQ), thr_copy_B.partition_S(sKt),
              tiled_mma, smem_copy_A, smem_copy_B, thr_copy_A, thr_copy_B);

    // P = exp(S - LSE), mask cols >= m
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

    // Write P to SMEM
    {
        auto rP = convert_type<Element>(acc_s);
        auto sc = make_tiled_copy_C(typename Traits::SmemCopyAtomPdS{}, tiled_mma);
        auto tc = sc.get_thread_slice(tidx);
        cute::copy(sc, tc.retile_S(rP), tc.partition_D(sP));
    }

    // GEMM2: dP = dO @ step2^T
    auto acc_dp = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_dp);
    gemm_smem(acc_dp, thr_mma.partition_fragment_A(sdO), thr_mma.partition_fragment_B(sS2),
              thr_copy_A.partition_S(sdO), thr_copy_B.partition_S(sS2),
              tiled_mma, smem_copy_A, smem_copy_B, thr_copy_A, thr_copy_B);

    // Softmax backward: dS = P * (dP - D1)
    auto dP = make_tensor(acc_dp.data(), convert_layout_acc_rowcol(acc_dp.layout()));
    const float* d1 = D1_ptr + bh*N + row_start;
    #pragma unroll
    for (int mi = 0; mi < nrow; mi++) {
        int pr = get<0>(tScS_rc(mi, 0));
        float d1v = (pr < tile_rows) ? d1[pr] : 0.0f;
        #pragma unroll
        for (int ni = 0; ni < size<1>(dP); ni++)
            dP(mi, ni) = scores(mi, ni) * (dP(mi, ni) - d1v);
    }

    // Write dS to SMEM
    {
        auto rdS = convert_type<Element>(acc_dp);
        auto sc = make_tiled_copy_C(typename Traits::SmemCopyAtomPdS{}, tiled_mma);
        auto tc = sc.get_thread_slice(tidx);
        cute::copy(sc, tc.retile_S(rdS), tc.partition_D(sdS));
    }
    __syncthreads();

    // GEMM3: dQ_s = dS @ Kt^T (Br×m @ m×D -> Br×D)
    auto acc_dq = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_dq);
    gemm_smem(acc_dq, thr_mma.partition_fragment_A(sdS), thr_mma.partition_fragment_B(sKttNS),
              thr_copy_A.partition_S(sdS), thr_copy_Bt.partition_S(sKtt),
              tiled_mma, smem_copy_A, smem_copy_Bt, thr_copy_A, thr_copy_Bt);

    // GEMM4/5 MUST run BEFORE we write dQ to sQ (which overwrites Q data).
    // GEMM4: dKt[j,d] += sum_i dS[i,j] * Q[i,d]
    // GEMM5: dstep2[j,d] += sum_i P[i,j] * dO[i,d]
    // Read dS/P from registers, Q/dO from sQ/sdO (still contains original data).
    {
        float* dk = dK_tilde_ptr + bh*m*D;
        float* ds2 = dstep2_ptr + bh*m*D;

        #pragma unroll
        for (int fi = 0; fi < size(acc_dp); fi++) {
            int row = get<0>(tScS(fi));
            int col = get<1>(tScS(fi));
            if (row >= tile_rows || col >= m) continue;

            float dS_val = acc_dp(fi);
            float P_val  = acc_s(fi);

            for (int d = 0; d < D; d++) {
                float q_val  = to_float(sQ(row, d));
                float dO_val = to_float(sdO(row, d));
                atomicAdd(&dk[col * D + d],  dS_val * q_val);
                atomicAdd(&ds2[col * D + d], P_val  * dO_val);
            }
        }
    }

    // NOW write dQ_s to GMEM (overwrites sQ with dQ values)
    {
        auto rdQ = convert_type<Element>(acc_dq);
        auto sc = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{}, tiled_mma);
        auto tc = sc.get_thread_slice(tidx);
        cute::copy(sc, tc.retile_S(rdQ), tc.partition_D(sQ));
        __syncthreads();
        Element* dq = dQ_s_ptr + bh*N*D + row_start*D;
        if (full) {
            auto gdQ = make_tensor(make_gmem_ptr(dq), Shape<Int<kBlockM>,Int<kHeadDim>>{}, Stride<Int<kHeadDim>,_1>{});
            auto gc = make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                typename Traits::GmemLayoutAtom{}, Layout<Shape<_1, Int<Traits::kGmemElemsPerLoad>>>{});
            auto gt = gc.get_thread_slice(tidx);
            cute::copy(gc, gt.partition_S(sQ), gt.partition_D(gdQ));
        } else {
            for (int i=tidx; i<tile_rows*D; i+=Traits::kNThreads) { int r=i/D,c=i%D; dq[i]=sQ(r,c); }
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
            int kPdS = static_cast<int>(cosize(typename Traits::SmemLayoutPdS{}));
            size_t smem;
            if constexpr (kHeadDim >= 128) {
                smem = (Traits::kSmemQElems*2 + Traits::kSmemKVElems*2) * sizeof(scalar_t);
            } else {
                smem = (Traits::kSmemQElems*2 + Traits::kSmemKVElems*2 + kPdS*2) * sizeof(scalar_t);
            }
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
