/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once

#include "utils.h"
#include "nystrom_utils.h"
#include "../static_switch.h"

#include <algorithm>  // std::min / std::max in host launch dispatch
#include <cstdlib>    // std::getenv / std::atoi for the splits override

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
    // Used for forward GEMM1/GEMM2 and backward GEMM1/GEMM_dP/GEMM_dQt
    using TiledMma = decltype(make_tiled_mma(
        MMA_Atom_Arch{},
        Layout<Shape<_4, _1, _1>>{},
        Tile<Int<kBlockM>, Int<kBlockN>, _16>{}));

    // 2 warps along M, 2 along N — for backward GEMM_dK and GEMM_dV
    // Output shape: (kBlockN × kHeadDim) = (Bc × D)
    using TiledMmaDKV = decltype(make_tiled_mma(
        MMA_Atom_Arch{},
        Layout<Shape<_2, _2, _1>>{},
        Tile<Int<kBlockN>, Int<kHeadDim>, _16>{}));

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

    // SMEM sizes (element counts, not bytes)
    static constexpr int kSmemQElems   = static_cast<int>(cosize(SmemLayoutQ{}));
    static constexpr int kSmemKVElems  = static_cast<int>(cosize(SmemLayoutKV{}));
    static constexpr int kSmemPdSElems = static_cast<int>(cosize(SmemLayoutPdS{}));

    // Fallback forward (kPipelined=false): sQt (persistent) + one shared K/V
    // buffer, reloaded twice per tile. Used when the pipelined footprint would
    // cost a resident CTA (100KB/SM parts at D=128).
    static constexpr int kSmemBytes = (kSmemQElems + kSmemKVElems) * sizeof(Element);

    // Pipelined forward: sQt (persistent) + separate sK + sV. Single buffers
    // suffice for a FlashAttention-2 style one-stage lookahead (at most one
    // cp.async group in flight): V(t)'s load hides under GEMM1 and K(t+1)'s
    // load hides under softmax+GEMM2, with no double buffering and therefore
    // no occupancy cost beyond the one extra KV buffer.
    static constexpr int kSmemFwdBytes =
        (kSmemQElems + 2 * kSmemKVElems) * sizeof(Element);

    // Backward: sQB (shared Qt/dO3, time-multiplexed) + sKV + sPdS (3 buffers).
    // sQt and sdO3 never coexist in time, so we fold them into one buffer and
    // reload Qt from GMEM after Phase 7. Saves 16KB. SmemLayoutQ and
    // SmemLayoutKV are identical when kBlockM == kBlockN, so views over the
    // shared buffer compose correctly.
    static constexpr int kSmemBwdElems = kSmemQElems + kSmemKVElems + kSmemPdSElems;
    static constexpr int kSmemBwdBytes = kSmemBwdElems * sizeof(Element);

    // Wide backward (kWide=true in kernel3_bwd_tc): dedicated buffers for Qt,
    // dO3, K, V (+ sPdS), so nothing is swapped or reloaded mid-kernel and the
    // dO3/V loads complete under GEMM1. Used only where the footprint keeps at
    // least 2 CTAs resident per SM.
    static constexpr int kSmemBwdWideBytes =
        (2 * kSmemQElems + 2 * kSmemKVElems + kSmemPdSElems) * sizeof(Element);
};

// -- the actual kernel --


template <typename Traits, bool kPipelined>
__global__ void __launch_bounds__(Traits::kNThreads, 3)
kernel3_fused_tc(
    const typename Traits::Element* __restrict__ q_tilde_ptr,  // (B*H, m, D)
    const typename Traits::Element* __restrict__ k_ptr,        // (B*H, N, D), pre-scaled
    const typename Traits::Element* __restrict__ v_ptr,        // (B*H, N, D)
    const float*                    __restrict__ kernel2_inv_ptr, // (B*H, m, m)
    typename Traits::Element*       __restrict__ step2_ptr,    // (B*H, m, D)
    typename Traits::Element*       __restrict__ b_ptr,        // (B*H, m, D) or nullptr; B = softmax(Qt@K^T)@V
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
    Element* sQ_ptr = reinterpret_cast<Element*>(smem_);
    Element* sK_ptr = sQ_ptr + Traits::kSmemQElems;
    // Fallback mode time-multiplexes one buffer between K and V (the legacy
    // layout); pipelined mode gives V its own buffer so loads overlap compute.
    Element* sV_ptr = kPipelined ? sK_ptr + Traits::kSmemKVElems : sK_ptr;

    Tensor sQ = make_tensor(make_smem_ptr(sQ_ptr), typename Traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(sK_ptr), typename Traits::SmemLayoutKV{});
    Tensor sV = make_tensor(make_smem_ptr(sV_ptr), typename Traits::SmemLayoutKV{});

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
    for (int i = 0; i < nrow; i++) { row_max(i) = fp32_neg_inf(); row_sum(i) = 0.0f; }

    // K/V tile loaders. A full tile goes through the 128-bit cp.async tiled
    // copy; the (at most one) partial tail tile takes a bounds-checked scalar
    // path that also zero-fills the padding rows so GEMM2 never multiplies
    // P (exactly 0 in masked columns) by garbage that could be NaN/Inf.
    // Both fence one commit group so the consumer's cp_async_wait<0>() is
    // uniform. Callers guarantee no other cp.async group is in flight and all
    // warps are past the buffer's last reader when a loader is invoked.
    auto load_k_tile = [&](int tile) {
        const int ts = tile * kBlockN;
        const Element* base = k_ptr + bh * N * D + ts * D;
        if (N - ts >= kBlockN) {
            Tensor gK = make_tensor(make_gmem_ptr(base),
                Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            cute::copy(gmem_tiled_copy, gmem_thr.partition_S(gK), gmem_thr.partition_D(sK));
        } else {
            const int len = N - ts;
            for (int idx = tidx; idx < kBlockN * kHeadDim; idx += Traits::kNThreads) {
                int r = idx / kHeadDim, c = idx % kHeadDim;
                sK(r, c) = (r < len) ? base[r * D + c] : Element(0);
            }
        }
        cp_async_fence();
    };
    auto load_v_tile = [&](int tile) {
        const int ts = tile * kBlockN;
        const Element* base = v_ptr + bh * N * D + ts * D;
        if (N - ts >= kBlockN) {
            Tensor gV = make_tensor(make_gmem_ptr(base),
                Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            cute::copy(gmem_tiled_copy, gmem_thr.partition_S(gV), gmem_thr.partition_D(sV));
        } else {
            const int len = N - ts;
            for (int idx = tidx; idx < kBlockN * kHeadDim; idx += Traits::kNThreads) {
                int r = idx / kHeadDim, c = idx % kHeadDim;
                sV(r, c) = (r < len) ? base[r * D + c] : Element(0);
            }
        }
        cp_async_fence();
    };

    // Loop-invariant MMA/copy descriptors over the dedicated sK/sV buffers.
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    auto smem_copy_K = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto thr_copy_K  = smem_copy_K.get_thread_slice(tidx);
    Tensor tCsK = thr_copy_K.partition_S(sK);
    auto tCrK_view = thr_copy_K.retile_D(tSrK);

    Tensor sVt = make_tensor(sV.data(), typename Traits::SmemLayoutKVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Traits::SmemLayoutKVtransposedNoSwizzle{});
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    auto smem_copy_V = make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto thr_copy_V  = smem_copy_V.get_thread_slice(tidx);
    Tensor tCsVt = thr_copy_V.partition_S(sVt);

    // Main tile loop over K/V. Pipelined mode is a FlashAttention-2 style
    // single-stage software pipeline with at most one cp.async group in
    // flight: V(tile)'s load overlaps GEMM1 and K(tile+1)'s load overlaps
    // softmax + GEMM2. sK is rewritten only after the mid-loop barrier (all
    // warps done reading it in GEMM1); sV only after the top-of-loop barrier
    // (all warps past the previous tile's gemm_rs). Fallback mode (sV aliases
    // sK) loads K and V synchronously in-loop, as before the pipeline.
    const int num_tiles = (N + kBlockN - 1) / kBlockN;
    if constexpr (kPipelined) {
        load_k_tile(0);  // prologue: K(0) in flight before the first iteration
    }

    for (int tile = 0; tile < num_tiles; tile++) {
        const int tile_start = tile * kBlockN;
        const int tile_end = min(tile_start + kBlockN, N);

        if constexpr (kPipelined) {
            cp_async_wait<0>();
            __syncthreads();      // K(tile) visible; sV free to overwrite
            load_v_tile(tile);    // in flight underneath GEMM1
        } else {
            __syncthreads();      // all warps past the previous tile's gemm_rs
            load_k_tile(tile);
            cp_async_wait<0>();
            __syncthreads();      // K(tile) visible
        }

        // GEMM1: S_tile = Q_tilde @ K_tile^T (Q in regs, K from SMEM)
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        #pragma unroll
        for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
            cute::copy(smem_copy_K, tCsK(_, _, ki), tCrK_view(_, _, ki));
            cute::gemm(tiled_mma, tSrQ(_, _, ki), tSrK(_, _, ki), acc_s);
        }

        // Mask invalid columns (tile padding + m padding)
        int valid_cols = tile_end - tile_start;
        #pragma unroll
        for (int i = 0; i < size(acc_s); i++) {
            int col = get<1>(tScS(i));
            if (col >= valid_cols) acc_s(i) = fp32_neg_inf();
        }
        // Also mask rows >= m
        #pragma unroll
        for (int i = 0; i < size(acc_s); i++) {
            int row = get<0>(tScS(i));
            if (row >= m) acc_s(i) = fp32_neg_inf();
        }

        if constexpr (kPipelined) {
            cp_async_wait<0>();
            __syncthreads();      // V(tile) visible; all warps done reading sK
            if (tile + 1 < num_tiles) load_k_tile(tile + 1);  // hides under softmax + GEMM2
        } else {
            __syncthreads();      // all warps done reading sK in GEMM1
            load_v_tile(tile);
            cp_async_wait<0>();
            __syncthreads();      // V(tile) visible
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

        gemm_rs(acc_o, tOrP, tOrVt, tCsVt, tiled_mma, smem_copy_V, thr_copy_V);

        // No trailing barrier: the next iteration's cp_async_wait<0>() +
        // __syncthreads() orders every warp past this gemm_rs before sV is
        // rewritten, and the epilogue below has its own barrier before it
        // reuses SMEM.
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

    // Side output: write B = O_acc to GMEM if requested. This is the same
    // intermediate the backward's compute_dk2inv would otherwise N-walk to
    // recompute; saving it once here turns that bwd kernel into a pair of
    // small m-bounded matmuls. sQ holds B at this point (kBlockM x kHeadDim
    // in the swizzled SMEM layout); rows >= m and cols >= D are zero.
    if (b_ptr != nullptr) {
        Element* b_out = b_ptr + bh * m * D;
        for (int idx = tidx; idx < m * D; idx += Traits::kNThreads) {
            int row = idx / D, d = idx % D;
            b_out[idx] = sQ(row, d);
        }
    }

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

// ===========================================================================
// Multi-CTA (split-N) variant.
//
// kernel3_fused_tc above is grid(BH): one CTA per batch-head, each walking the
// whole N dimension serially. That starves a large GPU when BH is small — at
// B=1,H=4 only 4 CTAs run, leaving 100+ SMs idle on an A100. The two kernels
// below split the N-walk across `num_splits` CTAs per batch-head (flash-
// decoding / split-KV), then combine the partial softmax states.
//
//   Phase A  kernel3_partial_tc  grid(num_splits, BH)
//       Each CTA processes a contiguous slice of N tiles, producing per-row
//       partial online-softmax state: max m_s, sum l_s, and the weighted-V
//       accumulator NORMALIZED by l_s before storing (so the stored partial
//       sits at softmax-output magnitude ~|V|, well-scaled for the FP32
//       round-trip rather than growing with tiles-per-split). Written to
//       scratch GMEM in FP32.
//   Phase B  kernel3_combine     grid(BH)
//       Reads all splits' partials, does the flash cross-split merge. Because
//       O_s is the NORMALIZED partial (O_s_unnorm = l_s * O_s), each split is
//       re-weighted by w_s = e^{m_s-M} * l_s:
//       (M = max_s m_s; L = sum_s e^{m_s-M} l_s; O = sum_s w_s O_s / L),
//       writes B and LSE, then the K2_inv @ O -> step2 tail.
// ===========================================================================

// Phase A: per-split partial. Same inner loop as kernel3_fused_tc, restricted
// to this split's tile range, writing partial state instead of final outputs.
template <typename Traits, bool kPipelined>
__global__ void __launch_bounds__(Traits::kNThreads, 3)
kernel3_partial_tc(
    const typename Traits::Element* __restrict__ q_tilde_ptr,  // (BH, m, D)
    const typename Traits::Element* __restrict__ k_ptr,        // (BH, N, D) pre-scaled
    const typename Traits::Element* __restrict__ v_ptr,        // (BH, N, D)
    float* __restrict__ partial_o_ptr,    // (num_splits, BH, m, D) FP32
    float* __restrict__ partial_max_ptr,  // (num_splits, BH, m)    FP32
    float* __restrict__ partial_sum_ptr,  // (num_splits, BH, m)    FP32
    int N, int D, int m, int num_splits
) {
    using Element = typename Traits::Element;
    constexpr int kBlockM  = Traits::kBlockM;
    constexpr int kBlockN  = Traits::kBlockN;
    constexpr int kHeadDim = Traits::kHeadDim;

    const int split = blockIdx.x;
    const int bh    = blockIdx.y;
    const int tidx  = threadIdx.x;

    const int num_tiles = (N + kBlockN - 1) / kBlockN;
    const int tiles_per_split = (num_tiles + num_splits - 1) / num_splits;
    const int tile_begin = split * tiles_per_split;
    const int tile_stop  = min(tile_begin + tiles_per_split, num_tiles);

    extern __shared__ char smem_[];
    Element* sQ_ptr = reinterpret_cast<Element*>(smem_);
    Element* sK_ptr = sQ_ptr + Traits::kSmemQElems;
    // Fallback mode time-multiplexes one buffer between K and V (the legacy
    // layout); pipelined mode gives V its own buffer so loads overlap compute.
    Element* sV_ptr = kPipelined ? sK_ptr + Traits::kSmemKVElems : sK_ptr;
    Tensor sQ = make_tensor(make_smem_ptr(sQ_ptr), typename Traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(sK_ptr), typename Traits::SmemLayoutKV{});
    Tensor sV = make_tensor(make_smem_ptr(sV_ptr), typename Traits::SmemLayoutKV{});

    // Determine nrow from the accumulator's rowcol layout.
    typename Traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    auto acc_s_dummy = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    auto scores_dummy = make_tensor(acc_s_dummy.data(), convert_layout_acc_rowcol(acc_s_dummy.layout()));
    constexpr int nrow = decltype(size<0>(scores_dummy))::value;

    Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thr_mma.partition_C(cS);

    Tensor row_max = make_tensor<float>(Shape<Int<nrow>>{});
    Tensor row_sum = make_tensor<float>(Shape<Int<nrow>>{});
    #pragma unroll
    for (int i = 0; i < nrow; i++) { row_max(i) = fp32_neg_inf(); row_sum(i) = 0.0f; }

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_o);

    // Empty split (more splits than tiles): emit identity partial and bail.
    if (tile_begin >= tile_stop) {
        Tensor tScS_rc = make_tensor(tScS.data(), convert_layout_acc_rowcol(tScS.layout()));
        float* pmax = partial_max_ptr + (size_t)(split * gridDim.y + bh) * m;
        float* psum = partial_sum_ptr + (size_t)(split * gridDim.y + bh) * m;
        #pragma unroll
        for (int mi = 0; mi < nrow; mi++) {
            int phys_row = get<0>(tScS_rc(mi, 0));
            int phys_col = get<1>(tScS_rc(mi, 0));
            if (phys_col == 0 && phys_row < m) {
                pmax[phys_row] = fp32_neg_inf();
                psum[phys_row] = 0.0f;
            }
        }
        // partial_o for an empty split is zero; the combine multiplies it by
        // exp(-inf - M) = 0 anyway, so we can skip writing it. But other CTAs
        // for non-empty splits write their own slices; this split's O slice
        // must still be zeroed so the combine's read is well-defined.
        float* po = partial_o_ptr + (size_t)(split * gridDim.y + bh) * m * D;
        for (int idx = tidx; idx < m * D; idx += Traits::kNThreads) po[idx] = 0.0f;
        return;
    }

    // Load Q_tilde into SMEM (persistent), zero-padded.
    for (int idx = tidx; idx < Traits::kSmemQElems; idx += Traits::kNThreads)
        sQ_ptr[idx] = Element(0);
    __syncthreads();
    const Element* qt_base = q_tilde_ptr + bh * m * D;
    for (int idx = tidx; idx < m * kHeadDim; idx += Traits::kNThreads) {
        int r = idx / kHeadDim, c = idx % kHeadDim;
        if (c < D) sQ(r, c) = qt_base[r * D + c];
    }
    __syncthreads();

    typename Traits::GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr = gmem_tiled_copy.get_thread_slice(tidx);

    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    auto smem_copy_Q = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto thr_copy_Q  = smem_copy_Q.get_thread_slice(tidx);
    Tensor tCsQ = thr_copy_Q.partition_S(sQ);
    Tensor tCrQ_view = thr_copy_Q.retile_D(tSrQ);
    #pragma unroll
    for (int ki = 0; ki < size<2>(tSrQ); ++ki)
        cute::copy(smem_copy_Q, tCsQ(_, _, ki), tCrQ_view(_, _, ki));

    // K/V tile loaders: identical contract to kernel3_fused_tc's (full tiles
    // via 128-bit cp.async, the at-most-one tail tile via a bounds-checked
    // scalar path that zero-fills padding; one commit group fenced either way).
    auto load_k_tile = [&](int tile) {
        const int ts = tile * kBlockN;
        const Element* base = k_ptr + bh * N * D + ts * D;
        if (N - ts >= kBlockN) {
            Tensor gK = make_tensor(make_gmem_ptr(base),
                Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            cute::copy(gmem_tiled_copy, gmem_thr.partition_S(gK), gmem_thr.partition_D(sK));
        } else {
            const int len = N - ts;
            for (int idx = tidx; idx < kBlockN * kHeadDim; idx += Traits::kNThreads) {
                int r = idx / kHeadDim, c = idx % kHeadDim;
                sK(r, c) = (r < len) ? base[r * D + c] : Element(0);
            }
        }
        cp_async_fence();
    };
    auto load_v_tile = [&](int tile) {
        const int ts = tile * kBlockN;
        const Element* base = v_ptr + bh * N * D + ts * D;
        if (N - ts >= kBlockN) {
            Tensor gV = make_tensor(make_gmem_ptr(base),
                Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            cute::copy(gmem_tiled_copy, gmem_thr.partition_S(gV), gmem_thr.partition_D(sV));
        } else {
            const int len = N - ts;
            for (int idx = tidx; idx < kBlockN * kHeadDim; idx += Traits::kNThreads) {
                int r = idx / kHeadDim, c = idx % kHeadDim;
                sV(r, c) = (r < len) ? base[r * D + c] : Element(0);
            }
        }
        cp_async_fence();
    };

    // Loop-invariant MMA/copy descriptors over the dedicated sK/sV buffers.
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    auto smem_copy_K = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto thr_copy_K  = smem_copy_K.get_thread_slice(tidx);
    Tensor tCsK = thr_copy_K.partition_S(sK);
    auto tCrK_view = thr_copy_K.retile_D(tSrK);

    Tensor sVt = make_tensor(sV.data(), typename Traits::SmemLayoutKVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Traits::SmemLayoutKVtransposedNoSwizzle{});
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    auto smem_copy_V = make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto thr_copy_V  = smem_copy_V.get_thread_slice(tidx);
    Tensor tCsVt = thr_copy_V.partition_S(sVt);

    // Tile loop over this split's slice; same pipelined/fallback scheme as
    // kernel3_fused_tc (V load overlaps GEMM1, K(tile+1) load overlaps
    // softmax + GEMM2 when pipelined; synchronous in-loop loads otherwise).
    if constexpr (kPipelined) {
        load_k_tile(tile_begin);  // prologue
    }

    for (int tile = tile_begin; tile < tile_stop; tile++) {
        const int tile_start = tile * kBlockN;
        const int tile_end = min(tile_start + kBlockN, N);

        if constexpr (kPipelined) {
            cp_async_wait<0>();
            __syncthreads();      // K(tile) visible; sV free to overwrite
            load_v_tile(tile);    // in flight underneath GEMM1
        } else {
            __syncthreads();      // all warps past the previous tile's gemm_rs
            load_k_tile(tile);
            cp_async_wait<0>();
            __syncthreads();      // K(tile) visible
        }

        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        #pragma unroll
        for (int ki = 0; ki < size<2>(tSrQ); ++ki) {
            cute::copy(smem_copy_K, tCsK(_, _, ki), tCrK_view(_, _, ki));
            cute::gemm(tiled_mma, tSrQ(_, _, ki), tSrK(_, _, ki), acc_s);
        }

        int valid_cols = tile_end - tile_start;
        #pragma unroll
        for (int i = 0; i < size(acc_s); i++)
            if (get<1>(tScS(i)) >= valid_cols) acc_s(i) = fp32_neg_inf();
        #pragma unroll
        for (int i = 0; i < size(acc_s); i++)
            if (get<0>(tScS(i)) >= m) acc_s(i) = fp32_neg_inf();

        if constexpr (kPipelined) {
            cp_async_wait<0>();
            __syncthreads();      // V(tile) visible; all warps done reading sK
            if (tile + 1 < tile_stop) load_k_tile(tile + 1);  // hides under softmax + GEMM2
        } else {
            __syncthreads();      // all warps done reading sK in GEMM1
            load_v_tile(tile);
            cp_async_wait<0>();
            __syncthreads();      // V(tile) visible
        }

        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
        Tensor tile_max = make_tensor<float>(Shape<Int<nrow>>{});
        frag_reduce_max<true>(scores, tile_max);
        Tensor new_max = make_tensor<float>(Shape<Int<nrow>>{});
        Tensor alpha   = make_tensor<float>(Shape<Int<nrow>>{});
        #pragma unroll
        for (int i = 0; i < nrow; i++) {
            new_max(i) = fmaxf(row_max(i), tile_max(i));
            alpha(i) = expf(row_max(i) - new_max(i));
        }
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
        #pragma unroll
        for (int i = 0; i < nrow; i++) {
            tile_sum(i) += __shfl_xor_sync(0xffffffff, tile_sum(i), 1);
            tile_sum(i) += __shfl_xor_sync(0xffffffff, tile_sum(i), 2);
        }
        {
            Tensor acc_o_rc = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
            #pragma unroll
            for (int mi = 0; mi < size<0>(acc_o_rc); mi++) {
                float a = alpha(mi);
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rc); ni++) acc_o_rc(mi, ni) *= a;
            }
        }
        #pragma unroll
        for (int i = 0; i < nrow; i++) {
            row_sum(i) = alpha(i) * row_sum(i) + tile_sum(i);
            row_max(i) = new_max(i);
        }

        Tensor rP = convert_type<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(),
            convert_layout_acc_Aregs<typename Traits::TiledMma>(rP.layout()));

        gemm_rs(acc_o, tOrP, tOrVt, tCsVt, tiled_mma, smem_copy_V, thr_copy_V);

        // No trailing barrier: the next iteration's cp_async_wait<0>() +
        // __syncthreads() orders every warp past this gemm_rs before sV is
        // rewritten, and the epilogue below re-syncs before reusing SMEM.
    }

    // Normalize acc_o by this split's row_sum BEFORE storing. The stored
    // partial is then at softmax-output magnitude (~|V|, FP16-safe) rather
    // than the unnormalized accumulator, which grows with tiles-per-split and
    // loses relative precision through the FP16 round-trip. The combine
    // re-weights each split by w_s = exp(m_s - M) * l_s to undo the per-split
    // normalization. (O_s_unnorm = l_s * O_s_norm, so the global
    //  O = sum_s exp(m_s - M) O_s_unnorm / L = sum_s w_s O_s_norm / L.)
    {
        Tensor acc_o_rc = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rc); mi++) {
            float inv = (row_sum(mi) > 0.0f) ? (1.0f / row_sum(mi)) : 0.0f;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rc); ni++) acc_o_rc(mi, ni) *= inv;
        }
    }

    // Stage acc_o (FP32) into SMEM in canonical (row, d) order via an identity
    // scatter, then store to partial_o in FP32. We must NOT round through the
    // Element (FP16) sQ buffer here: the single-CTA kernel rounds O to FP16
    // exactly once at the very end, but a split round-trips each split's
    // partial through FP16 before the cross-split combine, and the combine's
    // weighting (w_s spanning a large dynamic range) amplifies that FP16 error
    // far beyond the single rounding. Keeping partials in FP32 is what makes
    // the split path match the single-CTA path.
    //
    // Reuse the whole dynamic SMEM block as float scratch (sQ, sK, and sV are
    // dead after the loop). kSmemFwdBytes == (kSmemQElems + 2*kSmemKVElems) *
    // sizeof(Element) >= m*D floats at the max m=64, D=128, so it fits.
    //
    // CRITICAL: barrier before touching this memory. The last loop iteration's
    // gemm_rs reads sV (which aliases this float scratch). Without the sync,
    // threads still finishing that GEMM race against threads zeroing sOf,
    // corrupting the partial output in a timing/shape-dependent way.
    __syncthreads();
    float* sOf = reinterpret_cast<float*>(smem_);
    for (int idx = tidx; idx < m * D; idx += Traits::kNThreads) sOf[idx] = 0.0f;
    __syncthreads();
    Tensor cO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tOcO = thr_mma.partition_C(cO);
    #pragma unroll
    for (int e = 0; e < size(acc_o); e++) {
        int row = get<0>(tOcO(e));
        int col = get<1>(tOcO(e));
        if (row < m && col < D) sOf[row * D + col] = acc_o(e);
    }
    __syncthreads();

    float* po = partial_o_ptr + (size_t)(split * gridDim.y + bh) * m * D;
    for (int idx = tidx; idx < m * D; idx += Traits::kNThreads) po[idx] = sOf[idx];

    Tensor tScS_rc = make_tensor(tScS.data(), convert_layout_acc_rowcol(tScS.layout()));
    float* pmax = partial_max_ptr + (size_t)(split * gridDim.y + bh) * m;
    float* psum = partial_sum_ptr + (size_t)(split * gridDim.y + bh) * m;
    #pragma unroll
    for (int mi = 0; mi < nrow; mi++) {
        int phys_row = get<0>(tScS_rc(mi, 0));
        int phys_col = get<1>(tScS_rc(mi, 0));
        if (phys_col == 0 && phys_row < m) {
            pmax[phys_row] = row_max(mi);
            psum[phys_row] = row_sum(mi);
        }
    }
}

// Phase B: combine partials across splits, write B + LSE, do K2_inv @ O.
template <typename Element>
__global__ void kernel3_combine_kernel(
    const float* __restrict__ partial_o_ptr,   // (num_splits, BH, m, D) FP32
    const float* __restrict__ partial_max_ptr, // (num_splits, BH, m)    FP32
    const float* __restrict__ partial_sum_ptr, // (num_splits, BH, m)    FP32
    const float* __restrict__ kernel2_inv_ptr, // (BH, m, m) FP32
    Element*     __restrict__ step2_ptr,        // (BH, m, D)
    Element*     __restrict__ b_ptr,            // (BH, m, D) or nullptr
    float*       __restrict__ lse_ptr,          // (BH, m) or nullptr
    int BH, int D, int m, int num_splits
) {
    const int bh   = blockIdx.x;
    const int tidx = threadIdx.x;
    const int nthreads = blockDim.x;

    extern __shared__ float scomb_[];
    float* sO = scomb_;            // m * D
    float* sM = sO + m * D;        // m  (combined max)
    float* sL = sM + m;            // m  (combined denom)

    // Step 1: per-row combined max and denominator across splits.
    for (int i = tidx; i < m; i += nthreads) {
        float M = fp32_neg_inf();
        for (int s = 0; s < num_splits; s++)
            M = fmaxf(M, partial_max_ptr[(size_t)(s * BH + bh) * m + i]);
        float L = 0.0f;
        if (M > fp32_neg_inf()) {
            for (int s = 0; s < num_splits; s++) {
                float pm = partial_max_ptr[(size_t)(s * BH + bh) * m + i];
                float ps = partial_sum_ptr[(size_t)(s * BH + bh) * m + i];
                L += expf(pm - M) * ps;
            }
        }
        sM[i] = M;
        sL[i] = L;
    }
    __syncthreads();

    // Step 2: combined output. partial_o stores the per-split NORMALIZED
    // output O_s_norm, so each split is weighted by w_s = e^{m_s - M} * l_s
    // (the same weights that sum to L), then divided by L:
    //     O[i,d] = sum_s w_s * O_s_norm[s,i,d] / L
    for (int idx = tidx; idx < m * D; idx += nthreads) {
        int i = idx / D, d = idx % D;
        float M = sM[i];
        float acc = 0.0f;
        if (M > fp32_neg_inf()) {
            for (int s = 0; s < num_splits; s++) {
                float pm = partial_max_ptr[(size_t)(s * BH + bh) * m + i];
                float ps = partial_sum_ptr[(size_t)(s * BH + bh) * m + i];
                float w  = expf(pm - M) * ps;
                acc += w * partial_o_ptr[((size_t)(s * BH + bh) * m + i) * D + d];
            }
        }
        float Li = sL[i];
        float o = (Li > 0.0f) ? (acc / Li) : 0.0f;
        sO[idx] = o;
        if (b_ptr != nullptr) b_ptr[(size_t)bh * m * D + idx] = from_float<Element>(o);
    }
    __syncthreads();

    // Step 3: LSE = M + log(L).
    if (lse_ptr != nullptr) {
        for (int i = tidx; i < m; i += nthreads) {
            float M = sM[i], L = sL[i];
            lse_ptr[(size_t)bh * m + i] =
                (M > fp32_neg_inf()) ? (M + logf(L + 1e-12f)) : 0.0f;
        }
    }

    // Step 4: step2 = K2_inv @ O.
    const float* k2inv = kernel2_inv_ptr + (size_t)bh * m * m;
    Element* step2 = step2_ptr + (size_t)bh * m * D;
    for (int idx = tidx; idx < m * D; idx += nthreads) {
        int row = idx / D, d = idx % D;
        float acc = 0.0f;
        for (int j = 0; j < m; j++) acc += k2inv[row * m + j] * sO[j * D + d];
        step2[idx] = from_float<Element>(acc);
    }
}

// -- launch wrapper --

// Per-(device,stream) scratch for the split-N path. Grown on demand; freed by
// reset_kernel3_scratch (wired into _C.reset_caches). Sized in floats.
struct Kernel3Scratch {
    float* partial_o   = nullptr;
    float* partial_max = nullptr;
    float* partial_sum = nullptr;
    size_t o_elems = 0;    // capacity, floats
    size_t ms_elems = 0;   // capacity, floats
};

inline Kernel3Scratch& kernel3_scratch() {
    static thread_local Kernel3Scratch s;
    return s;
}

inline void reset_kernel3_scratch() {
    Kernel3Scratch& s = kernel3_scratch();
    if (s.partial_o)   cudaFree(s.partial_o);
    if (s.partial_max) cudaFree(s.partial_max);
    if (s.partial_sum) cudaFree(s.partial_sum);
    s = Kernel3Scratch{};
}

// Choose the number of N-splits. Goal: num_splits * BH CTAs is enough to fill
// the GPU, capped by the number of N-tiles and a hard ceiling on scratch.
// FLASH_NYSTROM_KERNEL3_SPLITS overrides: 1 forces the single-CTA path, N>1
// forces N splits (clamped to num_tiles).
inline int kernel3_choose_splits(int BH, int num_tiles) {
    const char* env = std::getenv("FLASH_NYSTROM_KERNEL3_SPLITS");
    if (env != nullptr && env[0] != '\0') {
        int forced = std::atoi(env);
        if (forced >= 1) return std::min(forced, num_tiles);
    }
    if (num_tiles <= 1) return 1;
    constexpr int kMaxSplits = 64;        // bounds scratch
    int sm = get_sm_count();
    // Aim for ~2 waves of CTAs across the SMs.
    int target_ctas = 2 * sm;
    int splits = (target_ctas + BH - 1) / BH;   // ceil(target / BH)
    splits = std::max(1, std::min(splits, std::min(num_tiles, kMaxSplits)));
    return splits;
}

template <typename scalar_t>
void launch_kernel3_output_fused(
    const scalar_t* q_tilde, const scalar_t* k, const scalar_t* v,
    const float* kernel2_inv, scalar_t* step2,
    scalar_t* b_out,                         // (BH, m, D) saved B, or nullptr
    float* softmax3_lse,
    int BH, int N, int D, int m,
    cudaStream_t stream
) {
    FN_CHECK(m > 0 && m <= 64, "kernel3: m must be <= 64");
    FN_CHECK(D == 64 || D == 128, "kernel3: head_dim must be 64 or 128");

    const int kBlockN = 64;
    const int num_tiles = (N + kBlockN - 1) / kBlockN;
    const int num_splits = kernel3_choose_splits(BH, num_tiles);

    // Pipeline the forward only where the extra V buffer does not cost a
    // resident CTA: __launch_bounds__ targets 3 CTAs/SM, so the pipelined
    // footprint must fit three times into the SM's shared memory. True on
    // every part at D=64 (3x24KB) and on A100/H100/B200 at D=128 (3x48KB
    // <= 164/228KB); false on 100KB/SM parts (sm_86/89, consumer Blackwell)
    // at D=128, where losing a CTA costs more than the overlap buys.
    auto pipeline_fits = [&](size_t fwd_bytes) {
        return 3 * fwd_bytes <= get_smem_per_multiprocessor();
    };

    auto launch_single = [&](auto HeadDimTag) {
        constexpr int kHeadDim = decltype(HeadDimTag)::value;
        using Traits = K3Traits<kHeadDim, scalar_t>;
        dim3 grid(BH);
        dim3 block(Traits::kNThreads);
        BOOL_SWITCH(pipeline_fits(Traits::kSmemFwdBytes), kPipelined, [&] {
            size_t smem = kPipelined ? Traits::kSmemFwdBytes : Traits::kSmemBytes;
            if (smem > 48 * 1024) {
                FN_CHECK(smem <= get_max_smem_per_block(), "kernel3: insufficient smem");
                FN_CUDA_CHECK(cudaFuncSetAttribute((kernel3_fused_tc<Traits, kPipelined>),
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
            }
            kernel3_fused_tc<Traits, kPipelined><<<grid, block, smem, stream>>>(
                q_tilde, k, v, kernel2_inv, step2, b_out, softmax3_lse, N, D, m);
        });
    };

    auto launch_split = [&](auto HeadDimTag) {
        constexpr int kHeadDim = decltype(HeadDimTag)::value;
        using Traits = K3Traits<kHeadDim, scalar_t>;

        // Scratch: partial_o (num_splits, BH, m, D), partial_max/sum
        // (num_splits, BH, m). Grow the cached buffers if needed.
        Kernel3Scratch& sc = kernel3_scratch();
        size_t need_o  = (size_t)num_splits * BH * m * D;
        size_t need_ms = (size_t)num_splits * BH * m;
        if (need_o > sc.o_elems) {
            if (sc.partial_o) cudaFree(sc.partial_o);
            FN_CUDA_CHECK(cudaMalloc(&sc.partial_o, need_o * sizeof(float)));
            sc.o_elems = need_o;
        }
        if (need_ms > sc.ms_elems) {
            if (sc.partial_max) cudaFree(sc.partial_max);
            if (sc.partial_sum) cudaFree(sc.partial_sum);
            FN_CUDA_CHECK(cudaMalloc(&sc.partial_max, need_ms * sizeof(float)));
            FN_CUDA_CHECK(cudaMalloc(&sc.partial_sum, need_ms * sizeof(float)));
            sc.ms_elems = need_ms;
        }

        // Phase A: partials.
        dim3 gridA(num_splits, BH);
        dim3 blockA(Traits::kNThreads);
        BOOL_SWITCH(pipeline_fits(Traits::kSmemFwdBytes), kPipelined, [&] {
            size_t smemA = kPipelined ? Traits::kSmemFwdBytes : Traits::kSmemBytes;
            if (smemA > 48 * 1024) {
                FN_CHECK(smemA <= get_max_smem_per_block(), "kernel3 partial: insufficient smem");
                FN_CUDA_CHECK(cudaFuncSetAttribute((kernel3_partial_tc<Traits, kPipelined>),
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemA)));
            }
            kernel3_partial_tc<Traits, kPipelined><<<gridA, blockA, smemA, stream>>>(
                q_tilde, k, v, sc.partial_o, sc.partial_max, sc.partial_sum,
                N, D, m, num_splits);
        });
        FN_CUDA_KERNEL_CHECK();

        // Phase B: combine.
        dim3 gridB(BH);
        dim3 blockB(128);
        size_t smemB = ((size_t)m * D + 2 * m) * sizeof(float);
        if (smemB > 48 * 1024) {
            FN_CHECK(smemB <= get_max_smem_per_block(), "kernel3 combine: insufficient smem");
            FN_CUDA_CHECK(cudaFuncSetAttribute(kernel3_combine_kernel<scalar_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemB)));
        }
        kernel3_combine_kernel<scalar_t><<<gridB, blockB, smemB, stream>>>(
            sc.partial_o, sc.partial_max, sc.partial_sum, kernel2_inv,
            step2, b_out, softmax3_lse, BH, D, m, num_splits);
    };

    if (num_splits <= 1) {
        if (D == 64) launch_single(Int<64>{}); else launch_single(Int<128>{});
    } else {
        if (D == 64) launch_split(Int<64>{}); else launch_split(Int<128>{});
    }
    FN_CUDA_KERNEL_CHECK();
}

// NOTE: FP32 path uses kernel3_scalar.cuh (scalar fallback).
// The tensor-core path is only instantiated for cutlass::half_t and cutlass::bfloat16_t.

} // namespace flash_nystrom
