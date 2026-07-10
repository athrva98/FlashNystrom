/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once

#include "utils.h"
#include "nystrom_utils.h"

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/swizzle.hpp>
#include <cutlass/numeric_types.h>

namespace flash_nystrom {

using namespace cute;

// Kernel1 Traits
//
// GEMM1: S[kBlockM, kBlockN] = Q[kBlockM, kHeadDim] @ K_tilde[kBlockN, kHeadDim]^T
// Softmax on S in register fragments
// GEMM2: O[kBlockM, kHeadDim] = P[kBlockM, kBlockN] @ step2[kBlockN, kHeadDim]
//        P stays in registers; step2 from SMEM

template <int kHeadDim_, typename elem_type>
struct K1Traits {
    using Element = elem_type;
    using ElementAccum = float;

    static constexpr int kBlockM   = 64;       // Q tile rows
    static constexpr int kBlockN   = 64;       // landmarks (must be >= m)
    static constexpr int kHeadDim  = kHeadDim_;
    static constexpr int kNWarps   = 4;
    static constexpr int kNThreads = kNWarps * 32;  // 128

    // MMA atom: m16n8k16 with TN layout
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

    // TiledMma for forward GEMM1 (S=Q@K^T) and GEMM2 (O=P@step2)
    // and backward GEMM1-2 (score recompute + dP)
    // 4 warps along M, 1 along N — ensures softmax rows are within a single warp-column
    using TiledMma = decltype(make_tiled_mma(
        MMA_Atom_Arch{},
        Layout<Shape<_4, _1, _1>>{},
        Tile<Int<kBlockM>, Int<kBlockN>, _16>{}));

    // TiledMma for backward GEMM4-5 (dK, dstep2, dV: kBlockN × kHeadDim outputs)
    // 2 warps along M, 2 along N — following FlashAttention's TiledMmadKV pattern
    using TiledMmaDKV = decltype(make_tiled_mma(
        MMA_Atom_Arch{},
        Layout<Shape<_2, _2, _1>>{},
        Tile<Int<kBlockN>, Int<kHeadDim>, _16>{}));

    // SMEM layouts with swizzle for bank-conflict-free ldmatrix
    // CRITICAL: atom inner dim must be kBlockKSmem (64 for D=128, 32 for D=64),
    // NOT kHeadDim. tile_to_shape extends to full kHeadDim.
    // FlashAttention comment: "This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128"
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using SmemLayoutAtom = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutQ     = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutKV    = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    // Transposed view of KV for GEMM2's B-operand: (kHeadDim, kBlockN)
    // Composition with GenRowMajor maps (kHeadDim, kBlockN) indices to the
    // physical (kBlockN, kHeadDim) storage, preserving the swizzle.
    using SmemLayoutKVtransposed = decltype(
        composition(SmemLayoutKV{},
                    make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutKVtransposedNoSwizzle = decltype(
        get_nonswizzle_portion(SmemLayoutKVtransposed{}));

    // SMEM layout for P and dS matrices: (kBlockM, kBlockN)
    // Following FA exactly: atom covers the full (kBlockM, kPBlockN) block.
    static constexpr int kPBlockN = kBlockN >= 64 ? 64 : 32;
    static constexpr int kSwizzlePdS = 3;
    using SmemLayoutAtomPdS = decltype(
        composition(Swizzle<kSwizzlePdS, 3, 3>{},
                    Layout<Shape<Int<kBlockM>, Int<kPBlockN>>,
                           Stride<Int<kPBlockN>, _1>>{}));
    using SmemLayoutPdS = decltype(tile_to_shape(SmemLayoutAtomPdS{},
        Shape<Int<kBlockM>, Int<kBlockN>>{}));
    using SmemLayoutPdStransposed = decltype(
        composition(SmemLayoutPdS{},
                    make_layout(Shape<Int<kBlockN>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutPdStransposedNoSwizzle = decltype(
        get_nonswizzle_portion(SmemLayoutPdStransposed{}));

    // SmemCopyAtom for writing MMA accumulators to SMEM (P, dS)
    using SmemCopyAtomPdS = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, elem_type>;

    // SMEM -> Register copy atoms (following FlashAttention's exact pattern)
    // GEMM1: both A (Q) and B (K_tilde) use non-transposed LDSM
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    // GEMM2: B (step2^T) uses transposed LDSM to read the transposed layout
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;

    // Global -> SMEM async copy (128-bit = 8 fp16 elements per thread)
    static constexpr int kGmemElemsPerLoad = 128 / cutlass::sizeof_bits<elem_type>::value;
    // Use kBlockKSmem for threads-per-row to match the smem tiling pattern
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;

    using GmemLayoutAtom = Layout<
        Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;

    using GmemTiledCopy = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, elem_type>{},
        GmemLayoutAtom{},
        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));

    // SMEM sizes (elems, not bytes)
    static constexpr int kSmemQElems  = static_cast<int>(cosize(SmemLayoutQ{}));
    static constexpr int kSmemKVElems = static_cast<int>(cosize(SmemLayoutKV{}));
    static constexpr int kSmemBytes   = (kSmemQElems + kSmemKVElems) * sizeof(Element);
};

// -- the actual kernel --


template <typename Traits>
__global__ void __launch_bounds__(Traits::kNThreads)
kernel1_fused_tc(
    const typename Traits::Element* __restrict__ q_ptr,
    const typename Traits::Element* __restrict__ k_tilde_ptr,
    const typename Traits::Element* __restrict__ step2_ptr,
    typename Traits::Element* __restrict__ o_ptr,
    float* __restrict__ lse_ptr,
    int N, int D, int m
) {
    using Element = typename Traits::Element;
    constexpr int kBlockM  = Traits::kBlockM;
    constexpr int kBlockN  = Traits::kBlockN;
    constexpr int kHeadDim = Traits::kHeadDim;

    const int tile_idx = blockIdx.x;
    const int64_t bh = blockIdx.y;
    const int tidx     = threadIdx.x;

    const int row_start = tile_idx * kBlockM;
    if (row_start >= N) return;

    // SMEM allocation
    // sQ: persistent across GEMM1. sKV: K_tilde for GEMM1, then step2 for GEMM2.
    extern __shared__ char smem_[];
    Element* sQ_ptr  = reinterpret_cast<Element*>(smem_);
    Element* sKV_ptr = sQ_ptr + Traits::kSmemQElems;

    Tensor sQ  = make_tensor(make_smem_ptr(sQ_ptr),  typename Traits::SmemLayoutQ{});
    Tensor sKV = make_tensor(make_smem_ptr(sKV_ptr), typename Traits::SmemLayoutKV{});

    // Zero-init SMEM (handles m < kBlockN and N-row_start < kBlockM)
    for (int idx = tidx; idx < Traits::kSmemQElems; idx += Traits::kNThreads)
        sQ_ptr[idx] = Element(0);
    for (int idx = tidx; idx < Traits::kSmemKVElems; idx += Traits::kNThreads)
        sKV_ptr[idx] = Element(0);
    __syncthreads();

    // GMEM tensors (use actual sizes for safety)
    // Q tile: might be partial at the end of the sequence
    const int q_tile_rows = min(kBlockM, N - row_start);
    // K_tilde: m rows, possibly < kBlockN
    // We must NOT create a gmem tensor larger than the actual allocation.
    // Use runtime-sized gmem tensors and manual copy.

    typename Traits::GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr = gmem_tiled_copy.get_thread_slice(tidx);

    // If m == kBlockN and tile is full, use fast CuTe copy. Otherwise manual.
    if (m == kBlockN && q_tile_rows == kBlockM) {
        Tensor gQ = make_tensor(
            make_gmem_ptr(q_ptr + bh * N * D + row_start * D),
            Shape<Int<kBlockM>, Int<kHeadDim>>{},
            Stride<Int<kHeadDim>, _1>{});
        Tensor gK = make_tensor(
            make_gmem_ptr(k_tilde_ptr + bh * m * D),
            Shape<Int<kBlockN>, Int<kHeadDim>>{},
            Stride<Int<kHeadDim>, _1>{});

        cute::copy(gmem_tiled_copy, gmem_thr.partition_S(gQ), gmem_thr.partition_D(sQ));
        cute::copy(gmem_tiled_copy, gmem_thr.partition_S(gK), gmem_thr.partition_D(sKV));

        cp_async_fence();
        cp_async_wait<0>();
    } else {
        // Manual copy with bounds checking, writing through swizzled SMEM tensors
        const Element* q_base = q_ptr + bh * N * D + row_start * D;
        for (int idx = tidx; idx < q_tile_rows * kHeadDim; idx += Traits::kNThreads) {
            int r = idx / kHeadDim, c = idx % kHeadDim;
            if (c < D) sQ(r, c) = q_base[r * D + c];
        }
        const Element* kt_base = k_tilde_ptr + bh * m * D;
        for (int idx = tidx; idx < m * kHeadDim; idx += Traits::kNThreads) {
            int r = idx / kHeadDim, c = idx % kHeadDim;
            if (c < D) sKV(r, c) = kt_base[r * D + c];
        }
    }
    __syncthreads();

    // GEMM1: S = Q @ K_tilde^T  (tensor cores)
    typename Traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);

    // Fragment allocation
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);    // (MMA, MMA_M, MMA_K)
    Tensor tSrK = thr_mma.partition_fragment_B(sKV);   // (MMA, MMA_N, MMA_K)
    Tensor acc_s = partition_fragment_C(tiled_mma,
                       Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);

    // SMEM -> Reg copy setup (both A and B use non-transposed LDSM for GEMM1)
    auto smem_copy_A = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto thr_copy_A  = smem_copy_A.get_thread_slice(tidx);
    auto smem_copy_B = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto thr_copy_B  = smem_copy_B.get_thread_slice(tidx);

    Tensor tCsQ = thr_copy_A.partition_S(sQ);
    Tensor tCsK = thr_copy_B.partition_S(sKV);

    // K-loop: kHeadDim / 16 steps
    gemm_smem(acc_s, tSrQ, tSrK, tCsQ, tCsK,
              tiled_mma, smem_copy_A, smem_copy_B, thr_copy_A, thr_copy_B);

    // Mask invalid columns (m < kBlockN) and apply softmax
    // Create identity tensor to find each element's (row, col) coordinate
    Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thr_mma.partition_C(cS);  // Same partitioning as acc_s

    // Mask columns >= m to -inf
    #pragma unroll
    for (int i = 0; i < size(acc_s); i++) {
        if (get<1>(tScS(i)) >= m) {
            acc_s(i) = fp32_neg_inf();
        }
    }

    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));

    constexpr int nrow = decltype(size<0>(scores))::value;
    Tensor row_max = make_tensor<float>(Shape<Int<nrow>>{});
    Tensor row_sum = make_tensor<float>(Shape<Int<nrow>>{});

    frag_reduce_max<true>(scores, row_max);
    frag_exp_sum(scores, row_max, row_sum, kLog2e);  // log2(e)
    frag_normalize(scores, row_sum);

    // Write LSE to global memory (needed for backward)
    // Map fragment rows to physical rows using the identity tensor's rowcol view
    if (lse_ptr != nullptr) {
        Tensor tScS_rc = make_tensor(tScS.data(), convert_layout_acc_rowcol(tScS.layout()));
        float* lse_base = lse_ptr + bh * N + row_start;
        #pragma unroll
        for (int mi = 0; mi < nrow; mi++) {
            // tScS_rc(mi, 0) gives (row, col) for the first column of fragment row mi
            int phys_row = get<0>(tScS_rc(mi, 0));
            int phys_col = get<1>(tScS_rc(mi, 0));
            // Only one thread per physical row should write (the one owning col 0)
            if (phys_col == 0 && phys_row < (N - row_start)) {
                // exp2-based softmax: row_max and row_sum are in exp2 space
                // LSE = row_max / log2(e) + log(row_sum)
                // Because exp2(x * log2e - max * log2e) = exp(x - max)
                // So row_sum = sum(exp(x - max/log2e * log2e)) = sum(exp(x - max_orig))
                // where max_orig = row_max / log2e
                // LSE = max_orig + log(row_sum) = row_max / log2e + log(row_sum)
                lse_base[phys_row] = row_max(mi) + logf(row_sum(mi) + 1e-12f);
            }
        }
    }

    // Convert P from FP32 to elem_type
    Tensor rP = convert_type<Element>(acc_s);

    // Reshape P's fragment from C-layout to A-layout for GEMM2
    // TiledMma inherits Shape_MNK from MMA_Atom — needed by convert_layout_acc_Aregs
    Tensor tOrP = make_tensor(rP.data(),
        convert_layout_acc_Aregs<typename Traits::TiledMma>(rP.layout()));

    // Load step2 into sKV (reusing K_tilde's SMEM space)
    __syncthreads();

    // Zero-init and reload (step2 also has m rows, not kBlockN)
    for (int idx = tidx; idx < Traits::kSmemKVElems; idx += Traits::kNThreads)
        sKV_ptr[idx] = Element(0);
    __syncthreads();

    if (m == kBlockN) {
        Tensor gS2 = make_tensor(
            make_gmem_ptr(step2_ptr + bh * m * D),
            Shape<Int<kBlockN>, Int<kHeadDim>>{},
            Stride<Int<kHeadDim>, _1>{});
        cute::copy(gmem_tiled_copy, gmem_thr.partition_S(gS2), gmem_thr.partition_D(sKV));
        cp_async_fence();
        cp_async_wait<0>();
    } else {
        const Element* s2_base = step2_ptr + bh * m * D;
        for (int idx = tidx; idx < m * kHeadDim; idx += Traits::kNThreads) {
            int r = idx / kHeadDim, c = idx % kHeadDim;
            if (c < D) sKV(r, c) = s2_base[r * D + c];
        }
    }
    __syncthreads();

    // GEMM2: O = P @ step2  (P in regs, step2^T from SMEM)
    // step2 stored as (kBlockN, kHeadDim) in sKV.
    // For GEMM2: A=P(kBlockM, kBlockN), B=step2 viewed as (kHeadDim, kBlockN).
    // Create transposed SMEM views (following FlashAttention's sVt pattern):
    Tensor sKVt = make_tensor(sKV.data(), typename Traits::SmemLayoutKVtransposed{});
    Tensor sKVtNoSwizzle = make_tensor(sKV.data().get(), typename Traits::SmemLayoutKVtransposedNoSwizzle{});

    Tensor acc_o = partition_fragment_C(tiled_mma,
                       Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_o);

    // B fragments partitioned from the non-swizzle transposed view
    Tensor tOrVt = thr_mma.partition_fragment_B(sKVtNoSwizzle);

    // SMEM copy for GEMM2 B uses TRANSPOSED LDSM (reads transposed layout)
    auto smem_copy_V = make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto thr_copy_V  = smem_copy_V.get_thread_slice(tidx);
    Tensor tCsVt = thr_copy_V.partition_S(sKVt);

    gemm_rs(acc_o, tOrP, tOrVt, tCsVt,
            tiled_mma, smem_copy_V, thr_copy_V);

    // Write output to GMEM
    // Convert acc_o FP32 -> elem_type, write through SMEM for coalesced stores
    Tensor rO = convert_type<Element>(acc_o);

    // Write rO to sQ (reuse SMEM)
    auto smem_copy_O = make_tiled_copy_C(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{}, tiled_mma);
    auto thr_copy_O = smem_copy_O.get_thread_slice(tidx);

    Tensor taccOsQ = thr_copy_O.partition_D(sQ);
    Tensor taccOrO = thr_copy_O.retile_S(rO);

    cute::copy(smem_copy_O, taccOrO, taccOsQ);
    __syncthreads();

    // SMEM -> GMEM (bounds-checked for last tile where row_start + kBlockM > N)
    const int valid_rows = min(kBlockM, N - row_start);
    Element* o_base = o_ptr + bh * N * D + row_start * D;

    if (valid_rows == kBlockM) {
        // Full tile: use fast vectorized copy
        Tensor gO = make_tensor(make_gmem_ptr(o_base),
            Shape<Int<kBlockM>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
        auto gmem_copy_O = make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
            typename Traits::GmemLayoutAtom{},
            Layout<Shape<_1, Int<Traits::kGmemElemsPerLoad>>>{});
        auto gmem_thr_O = gmem_copy_O.get_thread_slice(tidx);
        cute::copy(gmem_copy_O, gmem_thr_O.partition_S(sQ), gmem_thr_O.partition_D(gO));
    } else {
        // Partial tile: manual bounds-checked write through swizzled SMEM
        for (int idx = tidx; idx < valid_rows * D; idx += Traits::kNThreads) {
            int r = idx / D, c = idx % D;
            o_base[idx] = sQ(r, c);
        }
    }
}

// -- launch wrapper --


template <typename scalar_t>
void launch_kernel1_output_fused(
    const scalar_t* q, const scalar_t* k_tilde, const scalar_t* step2,
    scalar_t* output, float* softmax1_lse,
    int BH, int N, int D, int m,
    cudaStream_t stream
) {
    FN_CHECK(m > 0 && m <= 64, "kernel1: m must be <= 64");
    FN_CHECK(D == 64 || D == 128, "kernel1: head_dim must be 64 or 128");

    auto launch = [&](auto HeadDimTag) {
        constexpr int kHeadDim = decltype(HeadDimTag)::value;
        using Traits = K1Traits<kHeadDim, scalar_t>;

        int num_tiles = (N + Traits::kBlockM - 1) / Traits::kBlockM;
        dim3 grid(num_tiles, BH);
        dim3 block(Traits::kNThreads);
        size_t smem = Traits::kSmemBytes;

        if (smem > 48 * 1024) {
            FN_CHECK(smem <= get_max_smem_per_block(), "kernel1: insufficient smem");
            FN_CUDA_CHECK(cudaFuncSetAttribute(
                kernel1_fused_tc<Traits>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem)));
        }

        kernel1_fused_tc<Traits><<<grid, block, smem, stream>>>(
            q, k_tilde, step2, output, softmax1_lse, N, D, m);
    };

    if (D == 64) { launch(Int<64>{}); }
    else         { launch(Int<128>{}); }
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
