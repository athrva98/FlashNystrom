/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
// Batched tensor-core GEMM for the kernel2_inv Newton-Schulz chain (m x m x m,
// m <= 64). tf32 operands, fp32 accumulate. This is the building block for the
// TC-fied forward pinv; CP1 ships the plain C = A @ B primitive, verified
// against torch.bmm. Operand transposes and the affine epilogue are layered on
// in later checkpoints.
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include "nystrom_utils.h"   // convert_type, get_max_smem_per_block, FN_* checks

namespace flash_nystrom {

using namespace cute;

template <int M_, int N_, int K_, int NWarps_>
struct K2GemmTraits {
    static constexpr int kM = M_, kN = N_, kK = K_;
    static constexpr int kNWarps = NWarps_, kNThreads = NWarps_ * 32;

    // tf32 MMA atom: m16 n8 k8, TN, fp32 accumulate.
    using TiledMma = decltype(make_tiled_mma(
        MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>{},
        Layout<Shape<Int<NWarps_>, _1, _1>>{},
        Tile<Int<kM>, Int<kN>, _8>{}));

    // Plain row-major smem (m <= 64 -> 64x64x4 = 16 KB per operand, fits easily;
    // no swizzle needed at this size, correctness-first).
    // The TN MMA atom contracts the INNER dim of BOTH operands (C = A @ Bop^T),
    // so A is (M,K) [inner K] and the B operand must be (N,K) [inner K] = B^T.
    using SmemLayoutA = Layout<Shape<Int<kM>, Int<kK>>, Stride<Int<kK>, _1>>;  // (M,K) row-major
    using SmemLayoutB = Layout<Shape<Int<kN>, Int<kK>>, Stride<Int<kK>, _1>>;  // (N,K) = B^T

    static constexpr int kSmemElems = kM * kK + kK * kN;
    static constexpr int kSmemBytes = kSmemElems * static_cast<int>(sizeof(cutlass::tfloat32_t));
};

// C[bh] = A[bh] @ B[bh], A:(M,K) B:(K,N) C:(M,N), all FP32 row-major, one CTA per bh.
template <typename Traits>
__global__ void __launch_bounds__(Traits::kNThreads)
k2inv_gemm_nn_kernel(
    const float* __restrict__ A,   // (BH, M, K)
    const float* __restrict__ B,   // (BH, K, N)
    float* __restrict__ C          // (BH, M, N)
) {
    using TF = cutlass::tfloat32_t;
    constexpr int kM = Traits::kM, kN = Traits::kN, kK = Traits::kK;
    const int bh = blockIdx.x, tid = threadIdx.x;

    extern __shared__ TF smem_tf[];
    TF* sA_ = smem_tf;
    TF* sB_ = smem_tf + kM * kK;

    const float* Abh = A + bh * kM * kK;
    const float* Bbh = B + bh * kK * kN;       // B stored (K,N) row-major in GMEM
    for (int i = tid; i < kM * kK; i += Traits::kNThreads) sA_[i] = TF(Abh[i]);
    // Transpose B into smem: sB(n,k) = B(k,n).
    for (int i = tid; i < kN * kK; i += Traits::kNThreads) {
        int n = i / kK, k = i % kK;
        sB_[i] = TF(Bbh[k * kN + n]);
    }
    __syncthreads();

    Tensor sA = make_tensor(make_smem_ptr(sA_), typename Traits::SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(sB_), typename Traits::SmemLayoutB{});

    typename Traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    // Layout-aware smem -> register copy (AutoVectorizing, not ldmatrix: correct
    // for the tf32 fragment layout without ldmatrix's 16-bit assumptions).
    auto copy_A = make_tiled_copy_A(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, TF>{}, tiled_mma);
    auto copy_B = make_tiled_copy_B(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, TF>{}, tiled_mma);
    auto thr_copy_A = copy_A.get_thread_slice(tid);
    auto thr_copy_B = copy_B.get_thread_slice(tid);

    Tensor tCsA = thr_copy_A.partition_S(sA);
    Tensor tCsB = thr_copy_B.partition_S(sB);
    Tensor tCrA = thr_mma.partition_fragment_A(sA);   // (MMA, MMA_M, MMA_K) tf32
    Tensor tCrB = thr_mma.partition_fragment_B(sB);   // (MMA, MMA_N, MMA_K) tf32
    Tensor acc  = partition_fragment_C(tiled_mma, Shape<Int<kM>, Int<kN>>{});
    clear(acc);

    cute::copy(copy_A, tCsA, thr_copy_A.retile_D(tCrA));
    cute::copy(copy_B, tCsB, thr_copy_B.retile_D(tCrB));
    cute::gemm(tiled_mma, tCrA, tCrB, acc);   // full K contraction

    Tensor mC = make_tensor(make_gmem_ptr(C + bh * kM * kN),
                            make_layout(Shape<Int<kM>, Int<kN>>{}, GenRowMajor{}));
    Tensor tCgC = thr_mma.partition_C(mC);
    cute::copy(acc, tCgC);
}

// Launcher for the square m x m x m case used by the NS chain.
inline void launch_k2inv_gemm_nn(
    const float* A, const float* B, float* C, int BH, int m, cudaStream_t stream
) {
    FN_CHECK(m > 0 && m <= 64, "k2inv_gemm_nn: m must be in (0,64]");
    auto run = [&](auto MTag) {
        constexpr int kM = decltype(MTag)::value;
        using Traits = K2GemmTraits<kM, kM, kM, 4>;
        size_t smem = Traits::kSmemBytes;
        if (smem > 48 * 1024) {
            FN_CHECK(smem <= get_max_smem_per_block(), "k2inv_gemm_nn: insufficient SMEM");
            FN_CUDA_CHECK(cudaFuncSetAttribute(k2inv_gemm_nn_kernel<Traits>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        }
        k2inv_gemm_nn_kernel<Traits><<<dim3(BH), dim3(Traits::kNThreads), smem, stream>>>(A, B, C);
    };
    // Only m == 64 is exercised by the production pinv (landmarks fixed at 64);
    // pad/handle smaller m later. Assert for now.
    FN_CHECK(m == 64, "k2inv_gemm_nn: CP1 supports m == 64");
    run(Int<64>{});
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
