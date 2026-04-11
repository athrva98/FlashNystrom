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

// kernel3 traits — online softmax tiled over K/V, same structure as FlashAttn
//
// Online-softmax tiled GEMM:
//   For each tile j of K/V:
//     S_tile[m, Bc] = Q_tilde[m, D] @ K_tile[Bc, D]^T    (GEMM-SS)
//     Online softmax update on S_tile
//     O_acc[m, D] += P_tile[m, Bc] @ V_tile[Bc, D]        (GEMM-RS)
//   Then: step2[m, D] = K2_inv[m, m] @ (O_acc / row_sum)  (scalar, m small)
//
// m plays the role of kBlockM (query rows). Bc plays kBlockN (key columns).
// D plays kHeadDim. Exact same structure as FlashAttention.

template <int kHeadDim_, typename elem_type>
struct K3Traits {
    using Element = elem_type;
    using ElementAccum = float;

    static constexpr int kBlockM   = 64;  // m (landmarks, padded to 64)
    static constexpr int kBlockN   = 64;  // Bc (K/V tile columns)
    static constexpr int kHeadDim  = kHeadDim_;
    static constexpr int kNWarps   = 4;
    static constexpr int kNThreads = kNWarps * 32;  // 128

    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

    // 4 warps along M, 1 along N — same as kernel1
    using TiledMma = decltype(make_tiled_mma(
        MMA_Atom_Arch{},
        Layout<Shape<_4, _1, _1>>{},
        Tile<Int<kBlockM>, Int<kBlockN>, _16>{}));

    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using SmemLayoutAtom = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));

    // Q_tilde: (kBlockM, kHeadDim) — persistent
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    // K_tile: (kBlockN, kHeadDim) — reloaded per tile
    using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    // Transposed KV for GEMM2 B-operand (V_tile)
    using SmemLayoutKVtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutKVtransposedNoSwizzle = decltype(
        get_nonswizzle_portion(SmemLayoutKVtransposed{}));

    // PdS layouts for backward (P and dS: kBlockM × kBlockN)
    // Following FA: atom covers full (kBlockM, kPBlockN) for correct transposed reads
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
    using SmemCopyAtomPdS = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, elem_type>;

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;

    static constexpr int kGmemElemsPerLoad = 128 / cutlass::sizeof_bits<elem_type>::value;
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;

    using GmemLayoutAtom = Layout<
        Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, elem_type>{},
        GmemLayoutAtom{}, Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));

    // SMEM: Q_tilde (persistent) + KV (double-buffered not yet, single for now)
    static constexpr int kSmemQElems  = static_cast<int>(cosize(SmemLayoutQ{}));
    static constexpr int kSmemKVElems = static_cast<int>(cosize(SmemLayoutKV{}));
    // K and V tiles share one SMEM slot, loaded sequentially per tile iteration.
    static constexpr int kSmemBytes = (kSmemQElems + kSmemKVElems) * sizeof(Element);
};

// -- the actual kernel --


template <typename Traits>
__global__ void __launch_bounds__(Traits::kNThreads, 3)
kernel3_fused_tc(
    const typename Traits::Element* __restrict__ q_tilde_ptr,  // (B*H, m, D)
    const typename Traits::Element* __restrict__ k_ptr,        // (B*H, N, D), pre-scaled
    const typename Traits::Element* __restrict__ v_ptr,        // (B*H, N, D)
    const float*                    __restrict__ kernel2_inv_ptr, // (B*H, m, m)
    typename Traits::Element*       __restrict__ step2_ptr,    // (B*H, m, D)
    float*                          __restrict__ lse_ptr,      // (B*H, m)
    int N, int D, int m
) {
    using Element = typename Traits::Element;
    constexpr int kBlockM  = Traits::kBlockM;
    constexpr int kBlockN  = Traits::kBlockN;
    constexpr int kHeadDim = Traits::kHeadDim;

    const int bh   = blockIdx.x;  // One CTA per (batch, head)
    const int tidx = threadIdx.x;

    extern __shared__ char smem_[];
    Element* sQ_ptr  = reinterpret_cast<Element*>(smem_);
    Element* sKV_ptr = sQ_ptr + Traits::kSmemQElems;

    Tensor sQ  = make_tensor(make_smem_ptr(sQ_ptr),  typename Traits::SmemLayoutQ{});
    Tensor sKV = make_tensor(make_smem_ptr(sKV_ptr), typename Traits::SmemLayoutKV{});

    // Load Q_tilde into SMEM (persistent)
    for (int idx = tidx; idx < Traits::kSmemQElems; idx += Traits::kNThreads)
        sQ_ptr[idx] = Element(0);
    __syncthreads();

    // Manual load (m may be < kBlockM)
    const Element* qt_base = q_tilde_ptr + bh * m * D;
    for (int idx = tidx; idx < m * kHeadDim; idx += Traits::kNThreads) {
        int r = idx / kHeadDim, c = idx % kHeadDim;
        if (c < D) sQ(r, c) = qt_base[r * D + c];
    }
    __syncthreads();

    // Setup MMA
    typename Traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);

    typename Traits::GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr = gmem_tiled_copy.get_thread_slice(tidx);

    // Q fragments (persistent — loaded once)
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    auto smem_copy_Q = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto thr_copy_Q  = smem_copy_Q.get_thread_slice(tidx);
    Tensor tCsQ = thr_copy_Q.partition_S(sQ);
    // Pre-load Q into registers (it's reused every tile)
    Tensor tCrQ_view = thr_copy_Q.retile_D(tSrQ);
    #pragma unroll
    for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
        cute::copy(smem_copy_Q, tCsQ(_, _, ki), tCrQ_view(_, _, ki));
    }

    // Accumulator for O = softmax(Q_tilde @ K^T) @ V
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_o);

    // Identity tensor for column masking
    Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thr_mma.partition_C(cS);

    // Online softmax state (per fragment row)
    // Determine nrow from the accumulator's rowcol layout
    auto acc_s_dummy = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    auto scores_dummy = make_tensor(acc_s_dummy.data(), convert_layout_acc_rowcol(acc_s_dummy.layout()));
    constexpr int nrow = decltype(size<0>(scores_dummy))::value;

    Tensor row_max = make_tensor<float>(Shape<Int<nrow>>{});
    Tensor row_sum = make_tensor<float>(Shape<Int<nrow>>{});
    #pragma unroll
    for (int i = 0; i < nrow; i++) { row_max(i) = -INFINITY; row_sum(i) = 0.0f; }

    // Main tile loop over K/V
    const int num_tiles = (N + kBlockN - 1) / kBlockN;

    for (int tile = 0; tile < num_tiles; tile++) {
        const int tile_start = tile * kBlockN;
        const int tile_end = min(tile_start + kBlockN, N);
        const bool is_full_tile = (tile_end - tile_start) == kBlockN;

        // Load K_tile into sKV
        for (int idx = tidx; idx < Traits::kSmemKVElems; idx += Traits::kNThreads)
            sKV_ptr[idx] = Element(0);
        __syncthreads();

        if (is_full_tile) {
            Tensor gK = make_tensor(
                make_gmem_ptr(k_ptr + bh * N * D + tile_start * D),
                Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            cute::copy(gmem_tiled_copy, gmem_thr.partition_S(gK), gmem_thr.partition_D(sKV));
            cp_async_fence();
            cp_async_wait<0>();
        } else {
            const Element* k_base = k_ptr + bh * N * D + tile_start * D;
            int tile_len = tile_end - tile_start;
            for (int idx = tidx; idx < tile_len * kHeadDim; idx += Traits::kNThreads) {
                int r = idx / kHeadDim, c = idx % kHeadDim;
                if (c < D) sKV(r, c) = k_base[r * D + c];
            }
        }
        __syncthreads();

        // GEMM1: S_tile = Q_tilde @ K_tile^T (Q in regs, K from SMEM)
        Tensor tSrK = thr_mma.partition_fragment_B(sKV);
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);

        auto smem_copy_K = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
        auto thr_copy_K  = smem_copy_K.get_thread_slice(tidx);
        Tensor tCsK = thr_copy_K.partition_S(sKV);

        // Q is already in registers (tSrQ loaded before the loop)
        // K-loop
        {
            auto tCrK_view = thr_copy_K.retile_D(tSrK);
            #pragma unroll
            for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
                cute::copy(smem_copy_K, tCsK(_, _, ki), tCrK_view(_, _, ki));
                cute::gemm(tiled_mma, tSrQ(_, _, ki), tSrK(_, _, ki), acc_s);
            }
        }

        // Mask invalid columns (tile padding + m padding)
        int valid_cols = is_full_tile ? kBlockN : (tile_end - tile_start);
        #pragma unroll
        for (int i = 0; i < size(acc_s); i++) {
            int col = get<1>(tScS(i));
            if (col >= valid_cols) acc_s(i) = -INFINITY;
        }
        // Also mask rows >= m
        #pragma unroll
        for (int i = 0; i < size(acc_s); i++) {
            int row = get<0>(tScS(i));
            if (row >= m) acc_s(i) = -INFINITY;
        }

        // Online softmax update
        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));

        // New tile max
        Tensor tile_max = make_tensor<float>(Shape<Int<nrow>>{});
        frag_reduce_max<true>(scores, tile_max);

        // Compute new global max and correction factor
        Tensor new_max = make_tensor<float>(Shape<Int<nrow>>{});
        Tensor alpha   = make_tensor<float>(Shape<Int<nrow>>{});
        #pragma unroll
        for (int i = 0; i < nrow; i++) {
            new_max(i) = fmaxf(row_max(i), tile_max(i));
            alpha(i) = expf(row_max(i) - new_max(i));
        }

        // Apply exp and compute tile sum
        // kLog2e defined in utils.h
        Tensor tile_sum = make_tensor<float>(Shape<Int<nrow>>{});
        #pragma unroll
        for (int mi = 0; mi < size<0>(scores); mi++) {
            float max_scaled = new_max(mi) * kLog2e;
            float sum = 0.0f;
            #pragma unroll
            for (int ni = 0; ni < size<1>(scores); ni++) {
                scores(mi, ni) = exp2f(scores(mi, ni) * kLog2e - max_scaled);
                sum += scores(mi, ni);
            }
            tile_sum(mi) = sum;
        }
        // 4-thread allreduce for tile_sum
        #pragma unroll
        for (int i = 0; i < nrow; i++) {
            tile_sum(i) += __shfl_xor_sync(0xffffffff, tile_sum(i), 1);
            tile_sum(i) += __shfl_xor_sync(0xffffffff, tile_sum(i), 2);
        }

        // Rescale accumulator: acc_o[row,:] *= alpha[row]
        {
            Tensor acc_o_rc = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rc); mi++) {
            float a = alpha(mi);
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rc); ni++) {
                acc_o_rc(mi, ni) *= a;
            }
        }
        } // end scope for acc_o_rc rescaling

        // Update running state
        #pragma unroll
        for (int i = 0; i < nrow; i++) {
            row_sum(i) = alpha(i) * row_sum(i) + tile_sum(i);
            row_max(i) = new_max(i);
        }

        // GEMM2: O_acc += P_tile @ V_tile
        // Convert P from FP32 to elem_type
        Tensor rP = convert_type<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(),
            convert_layout_acc_Aregs<typename Traits::TiledMma>(rP.layout()));

        // Load V_tile into sKV (reusing K_tile's space)
        __syncthreads();
        for (int idx = tidx; idx < Traits::kSmemKVElems; idx += Traits::kNThreads)
            sKV_ptr[idx] = Element(0);
        __syncthreads();

        if (is_full_tile) {
            Tensor gV = make_tensor(
                make_gmem_ptr(v_ptr + bh * N * D + tile_start * D),
                Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            cute::copy(gmem_tiled_copy, gmem_thr.partition_S(gV), gmem_thr.partition_D(sKV));
            cp_async_fence();
            cp_async_wait<0>();
        } else {
            const Element* v_base = v_ptr + bh * N * D + tile_start * D;
            int tile_len = tile_end - tile_start;
            for (int idx = tidx; idx < tile_len * kHeadDim; idx += Traits::kNThreads) {
                int r = idx / kHeadDim, c = idx % kHeadDim;
                if (c < D) sKV(r, c) = v_base[r * D + c];
            }
        }
        __syncthreads();

        // V transposed views for GEMM2
        Tensor sKVt = make_tensor(sKV.data(), typename Traits::SmemLayoutKVtransposed{});
        Tensor sKVtNoSwizzle = make_tensor(sKV.data().get(), typename Traits::SmemLayoutKVtransposedNoSwizzle{});

        Tensor tOrVt = thr_mma.partition_fragment_B(sKVtNoSwizzle);
        auto smem_copy_V = make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{}, tiled_mma);
        auto thr_copy_V  = smem_copy_V.get_thread_slice(tidx);
        Tensor tCsVt = thr_copy_V.partition_S(sKVt);

        gemm_rs(acc_o, tOrP, tOrVt, tCsVt, tiled_mma, smem_copy_V, thr_copy_V);
    }

    // Final normalization: O_acc /= row_sum
    Tensor acc_o_rc = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rc); mi++) {
        float inv_sum = (row_sum(mi) > 0.0f) ? (1.0f / row_sum(mi)) : 0.0f;
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rc); ni++) {
            acc_o_rc(mi, ni) *= inv_sum;
        }
    }

    // Write LSE3 to global memory
    if (lse_ptr != nullptr) {
        Tensor tScS_rc = make_tensor(tScS.data(), convert_layout_acc_rowcol(tScS.layout()));
        float* lse_base = lse_ptr + bh * m;
        #pragma unroll
        for (int mi = 0; mi < nrow; mi++) {
            int phys_row = get<0>(tScS_rc(mi, 0));
            int phys_col = get<1>(tScS_rc(mi, 0));
            if (phys_col == 0 && phys_row < m) {
                lse_base[phys_row] = row_max(mi) + logf(row_sum(mi) + 1e-12f);
            }
        }
    }

    // Write O_acc (m, D) to SMEM via sQ, then scalar multiply by K2_inv
    // O_acc is now kernel_3 @ V. We need step2 = K2_inv @ O_acc.
    // This is a small (m, m) × (m, D) matmul. Do it scalar since m is small.

    // Write O_acc to SMEM (reuse sQ space)
    // Zero sQ first, then write through swizzled tensor
    for (int idx = tidx; idx < Traits::kSmemQElems; idx += Traits::kNThreads)
        sQ_ptr[idx] = Element(0);
    __syncthreads();

    // Write fragments to sQ using the MMA's C-partition writeback
    auto smem_copy_O = make_tiled_copy_C(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{}, tiled_mma);
    auto thr_copy_O = smem_copy_O.get_thread_slice(tidx);

    Tensor rO = convert_type<Element>(acc_o);
    Tensor taccOsQ = thr_copy_O.partition_D(sQ);
    Tensor taccOrO = thr_copy_O.retile_S(rO);
    cute::copy(smem_copy_O, taccOrO, taccOsQ);
    __syncthreads();

    // Scalar K2_inv @ O_acc -> step2
    // sQ now holds O_acc as (kBlockM, kHeadDim) in SMEM
    // K2_inv is (m, m) in GMEM (FP32)
    const float* k2inv = kernel2_inv_ptr + bh * m * m;
    Element* step2 = step2_ptr + bh * m * D;

    for (int idx = tidx; idx < m * D; idx += Traits::kNThreads) {
        int row = idx / D, d = idx % D;
        float acc = 0.0f;
        for (int j = 0; j < m; j++) {
            acc += k2inv[row * m + j] * to_float(sQ(j, d));
        }
        step2[idx] = from_float<Element>(acc);
    }
}

// -- launch wrapper --


template <typename scalar_t>
void launch_kernel3_output_fused(
    const scalar_t* q_tilde, const scalar_t* k, const scalar_t* v,
    const float* kernel2_inv, scalar_t* step2, float* softmax3_lse,
    int BH, int N, int D, int m,
    cudaStream_t stream
) {
    FN_CHECK(m > 0 && m <= 64, "kernel3: m must be <= 64");
    FN_CHECK(D == 64 || D == 128, "kernel3: head_dim must be 64 or 128");

    auto launch = [&](auto HeadDimTag) {
        constexpr int kHeadDim = decltype(HeadDimTag)::value;
        using Traits = K3Traits<kHeadDim, scalar_t>;

        dim3 grid(BH);  // One CTA per (batch, head)
        dim3 block(Traits::kNThreads);
        size_t smem = Traits::kSmemBytes;

        if (smem > 48 * 1024) {
            FN_CHECK(smem <= get_max_smem_per_block(), "kernel3: insufficient smem");
            FN_CUDA_CHECK(cudaFuncSetAttribute(
                kernel3_fused_tc<Traits>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem)));
        }

        kernel3_fused_tc<Traits><<<grid, block, smem, stream>>>(
            q_tilde, k, v, kernel2_inv, step2, softmax3_lse, N, D, m);
    };

    if (D == 64) { launch(Int<64>{}); }
    else         { launch(Int<128>{}); }
    FN_CUDA_KERNEL_CHECK();
}

// NOTE: FP32 path uses kernel3_scalar.cuh (scalar fallback).
// The tensor-core path is only instantiated for cutlass::half_t and cutlass::bfloat16_t.

} // namespace flash_nystrom
