/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once

// cute utilities ripped from FlashAttention (thanks Tri Dao)
// handles the tensor core gemms and in-register softmax stuff

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>

namespace flash_nystrom {

using namespace cute;

// gemm with both operands from shared memory. prefetches next k-slice
// while current one is being crunched by the tensor cores

template <bool A_in_regs=false, bool B_in_regs=false,
          typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename Tensor4,
          typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm_smem(
    Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB,
    Tensor3 const& tCsA, Tensor4 const& tCsB,
    TiledMma tiled_mma,
    TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
    ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B
) {
    auto tCrA_view = smem_thr_copy_A.retile_D(tCrA);
    auto tCrB_view = smem_thr_copy_B.retile_D(tCrB);
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

// gemm_rs: A lives in registers (like the softmax output P), B from smem
// this is the key trick — P never touches HBM between the two gemms

template <typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(
    Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB,
    Tensor3 const& tCsB,
    TiledMma tiled_mma, TiledCopy smem_tiled_copy_B, ThrCopy smem_thr_copy_B
) {
    auto tCrB_view = smem_thr_copy_B.retile_D(tCrB);
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

// layout conversions for MMA accumultor fragments
// the (MMA=4, MMA_M, MMA_N) -> (nrow, ncol) conversion is needed
// so we can do per-row softmax operations on the fragments

// (MMA=4, MMA_M, MMA_N) -> (nrow=(2, MMA_M), ncol=(2, MMA_N))
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                       make_layout(get<0, 0>(l), get<2>(l)));
}

// (MMA=4, MMA_M, MMA_N) -> ((4, 2), MMA_M, MMA_N/2) for m16n8k16
// or (4, MMA_M, MMA_N) for m16n8k8
template <typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
}

// type conversion: fp32 accumulator -> fp16/bf16 for the next gemm
// uses cutlass NumericArrayConverter for vectorised conversion

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

// in-register softmax primitives — no smem roundtrip needed
// the 4-thread shuffle handles the cross-thread reduction within each warp

// Per-row max across fragment columns + 4-thread allreduce
template <bool zero_init=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1>
__device__ __forceinline__ void frag_reduce_max(
    Tensor<Engine0, Layout0> const& tensor,
    Tensor<Engine1, Layout1>& max_vec
) {
    static_assert(Layout0::rank == 2);
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        float m = zero_init ? tensor(mi, 0) : fmaxf(max_vec(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) m = fmaxf(m, tensor(mi, ni));
        max_vec(mi) = m;
    }
    // 4-thread allreduce (threads sharing the same row in SM80_16x8x16)
    #pragma unroll
    for (int mi = 0; mi < size(max_vec); mi++) {
        max_vec(mi) = fmaxf(max_vec(mi), __shfl_xor_sync(0xffffffff, max_vec(mi), 1));
        max_vec(mi) = fmaxf(max_vec(mi), __shfl_xor_sync(0xffffffff, max_vec(mi), 2));
    }
}

// Apply exp and compute row sum (using exp2 for speed)
template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void frag_exp_sum(
    Tensor<Engine0, Layout0>& tensor,
    Tensor<Engine1, Layout1> const& max_vec,
    Tensor<Engine1, Layout1>& sum_vec,
    const float scale  // log2(e) = 1.4426950408889634
) {
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        float max_scaled = max_vec(mi) * scale;
        float row_sum = 0.0f;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ni++) {
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            row_sum += tensor(mi, ni);
        }
        sum_vec(mi) = row_sum;
    }
    // 4-thread allreduce for sum
    #pragma unroll
    for (int mi = 0; mi < size(sum_vec); mi++) {
        sum_vec(mi) += __shfl_xor_sync(0xffffffff, sum_vec(mi), 1);
        sum_vec(mi) += __shfl_xor_sync(0xffffffff, sum_vec(mi), 2);
    }
}

// Normalize rows by dividing by sum
template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void frag_normalize(
    Tensor<Engine0, Layout0>& tensor,
    Tensor<Engine1, Layout1> const& sum_vec
) {
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        float inv_sum = 1.0f / (sum_vec(mi) + 1e-12f);
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ni++) {
            tensor(mi, ni) *= inv_sum;
        }
    }
}

} // namespace flash_nystrom
