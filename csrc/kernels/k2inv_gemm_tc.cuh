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
#include <cstdlib>
#include "nystrom_utils.h"   // get_sm_count, FN_* checks

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

// C[bh] = A[bh] @ B[bh] for the square m x m case (m = kM = kK = 64). Tiled over
// output columns: grid is (BH, m/kN), CTA (bh, ct) computes the kN-wide column
// slab C[bh][:, ct*kN : ct*kN+kN]. Column tiling multiplies the CTA count by m/kN
// to fill the GPU at low BH while keeping the 4-warps-along-M MMA layout. Loads
// the full A and the kN-column B slab. Per-operand bh strides let callers index
// iterate slices of an (BH, J+1, m, m) buffer in place.
// kTransB selects what the kN-column slab of the B operand is:
//   false (NN): C = A @ B   -> Bop(n,k) = B(k, col_off+n)   (transpose-on-load)
//   true  (NT): C = A @ B^T -> Bop(n,k) = B(col_off+n, k)   (direct row slab)
// The TN atom contracts the inner dim of both, so in NT mode it computes A @ B^T
// with no transpose (used for the Tikhonov final multiply Z_J @ K2^T).
template <typename Traits, bool kTransB = false>
__global__ void __launch_bounds__(Traits::kNThreads)
k2inv_gemm_nn_kernel(
    const float* __restrict__ A,   // base of (.., M, K)
    const float* __restrict__ B,   // base of (.., K, M) NN  /  (.., M, K) NT
    float* __restrict__ C,         // base of (.., M, M)
    long long strideA, long long strideB, long long strideC
) {
    using TF = cutlass::tfloat32_t;
    constexpr int kM = Traits::kM, kN = Traits::kN, kK = Traits::kK;  // kN = column tile
    const int bh = blockIdx.x, ct = blockIdx.y, tid = threadIdx.x;
    const int col_off = ct * kN;     // this CTA writes C[:, col_off : col_off+kN]

    extern __shared__ TF smem_tf[];
    TF* sA_ = smem_tf;             // (kM, kK) full A
    TF* sB_ = smem_tf + kM * kK;   // (kN, kK) Bop column slab

    const float* Abh = A + bh * strideA;
    const float* Bbh = B + bh * strideB;
    for (int i = tid; i < kM * kK; i += Traits::kNThreads) sA_[i] = TF(Abh[i]);
    for (int i = tid; i < kN * kK; i += Traits::kNThreads) {
        int n = i / kK, k = i % kK;
        if constexpr (kTransB) sB_[i] = TF(Bbh[(col_off + n) * kK + k]);  // B(col_off+n, k)
        else                   sB_[i] = TF(Bbh[k * kM + col_off + n]);    // B(k, col_off+n)
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

    // Write the (kM, kN) slab into C[:, col_off:col_off+kN] (row stride = full m = kM).
    Tensor mC = make_tensor(make_gmem_ptr(C + bh * strideC + col_off),
                            make_layout(Shape<Int<kM>, Int<kN>>{}, Stride<Int<kM>, _1>{}));
    Tensor tCgC = thr_mma.partition_C(mC);
    cute::copy(acc, tCgC);
}

// Column-tile count: fill ~2 waves of CTAs (BH * tiles ~ 2*#SMs) at low BH, 1 at
// high BH. tileN = 64/tiles in {64,32,16}. FN_K2INV_SPLITS (1/2/4) forces it.
inline int k2inv_choose_col_tiles(int BH) {
    const char* env = std::getenv("FN_K2INV_SPLITS");
    if (env && env[0]) { int f = std::atoi(env); if (f == 1 || f == 2 || f == 4) return f; }
    int raw = (2 * get_sm_count() + BH - 1) / BH;
    return raw <= 1 ? 1 : (raw <= 2 ? 2 : 4);
}

// Launcher for the square m x m x m case. strideX default to the contiguous m*m.
// kTransB=false: C = A @ B.  kTransB=true: C = A @ B^T.
template <bool kTransB>
inline void launch_k2inv_gemm_impl(
    const float* A, const float* B, float* C, int BH, int m, cudaStream_t stream,
    long long strideA, long long strideB, long long strideC
) {
    FN_CHECK(m == 64, "k2inv_gemm: supports m == 64 (landmarks fixed at 64)");
    const long long mm = (long long)m * m;
    if (strideA < 0) strideA = mm;
    if (strideB < 0) strideB = mm;
    if (strideC < 0) strideC = mm;
    const int tiles = k2inv_choose_col_tiles(BH);
    auto run = [&](auto TN) {
        constexpr int tileN = decltype(TN)::value;
        using Traits = K2GemmTraits<64, tileN, 64, 4>;   // smem <= 32KB, no opt-in
        k2inv_gemm_nn_kernel<Traits, kTransB><<<dim3(BH, 64 / tileN), dim3(Traits::kNThreads),
            Traits::kSmemBytes, stream>>>(A, B, C, strideA, strideB, strideC);
    };
    if (tiles == 1)      run(cute::Int<64>{});
    else if (tiles == 2) run(cute::Int<32>{});
    else                 run(cute::Int<16>{});
    FN_CUDA_KERNEL_CHECK();
}

inline void launch_k2inv_gemm_nn(const float* A, const float* B, float* C, int BH, int m,
    cudaStream_t stream, long long sA = -1, long long sB = -1, long long sC = -1) {
    launch_k2inv_gemm_impl<false>(A, B, C, BH, m, stream, sA, sB, sC);   // C = A @ B
}
inline void launch_k2inv_gemm_nt(const float* A, const float* B, float* C, int BH, int m,
    cudaStream_t stream, long long sA = -1, long long sB = -1, long long sC = -1) {
    launch_k2inv_gemm_impl<true>(A, B, C, BH, m, stream, sA, sB, sC);    // C = A @ B^T
}

} // namespace flash_nystrom
