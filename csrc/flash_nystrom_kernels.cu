/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
// kernel orchestration — this is where the actual forward/backward pipelines
// get assembled from individual kernel launches. nothing fancy, just sequencing.

#include "flash_nystrom.h"
#include "utils.h"
#include "static_switch.h"
#include "kernels/landmark.cuh"
#include "kernels/kernel2_inv.cuh"
#include "kernels/kernel3_output_fused.cuh"  // Tensor-core version (FP16/BF16)
#include "kernels/kernel3_scalar.cuh"        // Scalar fallback (FP32)
#include "kernels/kernel1_output_fused.cuh"  // Tensor-core version (FP16/BF16)
#include "kernels/kernel1_scalar.cuh"        // Scalar fallback (FP32)
#include "kernels/dconv_residual.cuh"

// Backward kernels
#include "kernels/backward/precompute_di.cuh"
#include "kernels/backward/kernel1_bwd.cuh"
#include "kernels/backward/kernel3_bwd.cuh"
#include "kernels/backward/compute_dk2inv.cuh"
#include "kernels/backward/kernel2_inv_bwd.cuh"
#include "kernels/backward/landmark_bwd.cuh"
#include "kernels/backward/dconv_residual_bwd.cuh"

namespace flash_nystrom {

// FP16/BF16 path: uses tensor-core kernel1
template <typename elem_type>
static void run_nystrom_fwd_half(NystromParams &p) {
    auto* q   = static_cast<const elem_type*>(p.q_ptr);
    auto* k   = static_cast<const elem_type*>(p.k_ptr);
    auto* v   = static_cast<const elem_type*>(p.v_ptr);
    auto* o   = static_cast<elem_type*>(p.o_ptr);
    auto* qt  = static_cast<elem_type*>(p.q_tilde_ptr);
    auto* kt  = static_cast<elem_type*>(p.k_tilde_ptr);
    auto* s2  = static_cast<elem_type*>(p.step2_ptr);
    auto* q_m = static_cast<elem_type*>(p.q_ptr);
    auto* k_m = static_cast<elem_type*>(p.k_ptr);

    launch_landmarks<elem_type>(q, k, qt, kt,
        p.BH, p.seq_len, p.head_dim, p.num_landmarks, p.scale, p.stream);

    int total = p.BH * p.seq_len * p.head_dim;
    launch_scale_inplace<elem_type>(q_m, total, p.scale, p.stream);
    launch_scale_inplace<elem_type>(k_m, total, p.scale, p.stream);

    launch_kernel2_inv<elem_type>(qt, kt,
        p.kernel2_inv_ptr, p.softmax2_lse_ptr, p.ns_iterates_ptr,
        p.BH, p.head_dim, p.num_landmarks, p.newton_iter, p.stream);

    launch_kernel3_output_fused<elem_type>(qt, k_m, v,
        p.kernel2_inv_ptr, s2, p.softmax3_lse_ptr,
        p.BH, p.seq_len, p.head_dim, p.num_landmarks, p.stream);

    // Tensor-core kernel1 (the main performance kernel)
    launch_kernel1_output_fused<elem_type>(q_m, kt, s2,
        o, p.softmax1_lse_ptr,
        p.BH, p.seq_len, p.head_dim, p.num_landmarks, p.stream);

    if (p.conv_weight_ptr != nullptr && p.conv_kernel_size > 0) {
        auto* cw = static_cast<const elem_type*>(p.conv_weight_ptr);
        launch_dconv_residual<elem_type>(v, cw, o,
            p.BH, p.seq_len, p.head_dim, p.num_heads, p.conv_kernel_size, p.stream);
    }
}

// FP32 path: uses scalar kernel1 (LDSM doesn't support 32-bit elements)
static void run_nystrom_fwd_fp32_impl(NystromParams &p) {
    using T = float;
    auto* q   = static_cast<const T*>(p.q_ptr);
    auto* k   = static_cast<const T*>(p.k_ptr);
    auto* v   = static_cast<const T*>(p.v_ptr);
    auto* o   = static_cast<T*>(p.o_ptr);
    auto* qt  = static_cast<T*>(p.q_tilde_ptr);
    auto* kt  = static_cast<T*>(p.k_tilde_ptr);
    auto* s2  = static_cast<T*>(p.step2_ptr);
    auto* q_m = static_cast<T*>(p.q_ptr);
    auto* k_m = static_cast<T*>(p.k_ptr);

    launch_landmarks<T>(q, k, qt, kt,
        p.BH, p.seq_len, p.head_dim, p.num_landmarks, p.scale, p.stream);

    int total = p.BH * p.seq_len * p.head_dim;
    launch_scale_inplace<T>(q_m, total, p.scale, p.stream);
    launch_scale_inplace<T>(k_m, total, p.scale, p.stream);

    launch_kernel2_inv<T>(qt, kt,
        p.kernel2_inv_ptr, p.softmax2_lse_ptr, p.ns_iterates_ptr,
        p.BH, p.head_dim, p.num_landmarks, p.newton_iter, p.stream);

    launch_kernel3_scalar<T>(qt, k_m, v,
        p.kernel2_inv_ptr, s2, p.softmax3_lse_ptr,
        p.BH, p.seq_len, p.head_dim, p.num_landmarks, p.stream);

    launch_kernel1_scalar<T>(q_m, kt, s2,
        o, p.softmax1_lse_ptr,
        p.BH, p.seq_len, p.head_dim, p.num_landmarks, p.stream);

    if (p.conv_weight_ptr != nullptr && p.conv_kernel_size > 0) {
        auto* cw = static_cast<const T*>(p.conv_weight_ptr);
        launch_dconv_residual<T>(v, cw, o,
            p.BH, p.seq_len, p.head_dim, p.num_heads, p.conv_kernel_size, p.stream);
    }
}

void run_nystrom_fwd(NystromParams &params) {
    params.set_derived();
    FP16_SWITCH(!params.is_bf16, [&] {
        run_nystrom_fwd_half<elem_type>(params);
    });
}

void run_nystrom_fwd_fp32(NystromParams &params) {
    params.set_derived();
    run_nystrom_fwd_fp32_impl(params);
}

// -- backward --


template <typename elem_type>
static void run_nystrom_bwd_impl(NystromBwdParams &p) {
    auto* q_s     = static_cast<const elem_type*>(p.q_s_ptr);
    auto* k_s     = static_cast<const elem_type*>(p.k_s_ptr);
    auto* v       = static_cast<const elem_type*>(p.v_ptr);
    auto* q_tilde = static_cast<const elem_type*>(p.q_tilde_ptr);
    auto* k_tilde = static_cast<const elem_type*>(p.k_tilde_ptr);
    auto* step2   = static_cast<const elem_type*>(p.step2_ptr);
    auto* output  = static_cast<const elem_type*>(p.o_ptr);
    auto* dO      = static_cast<const elem_type*>(p.dO_ptr);
    auto* dQ      = static_cast<elem_type*>(p.dQ_ptr);
    auto* dK      = static_cast<elem_type*>(p.dK_ptr);
    auto* dV      = static_cast<elem_type*>(p.dV_ptr);

    int BH = p.BH, N = p.seq_len, D = p.head_dim, m = p.num_landmarks;

    if (p.conv_weight_ptr != nullptr && p.conv_kernel_size > 0) {
        launch_dconv_bwd<elem_type>(dO, v,
            static_cast<const elem_type*>(p.conv_weight_ptr),
            dV, static_cast<float*>(p.dconv_weight_ptr),
            BH, N, D, p.num_heads, p.batch_size, p.conv_kernel_size, p.stream);
    }

    launch_precompute_di<elem_type>(dO, output, p.D1_ptr, BH, N, D, p.stream);

    launch_kernel1_bwd<elem_type>(q_s, k_tilde, step2, p.lse1_ptr, p.D1_ptr, dO,
        dQ, p.dstep2_ptr, p.dK_tilde_ptr, BH, N, D, m, p.stream);

    launch_kernel3_bwd<elem_type>(q_tilde, k_s, v, p.k2_inv_ptr, p.lse3_ptr,
        p.dstep2_ptr, dV, dK, p.dQ_tilde_ptr, p.dK2_inv_ptr, BH, N, D, m, p.stream);

    // Compute dK2_inv separately in FP32 to avoid amplification from IFT backward.
    // kernel3_bwd no longer accumulates dK2_inv (was removed for numerical stability).
    launch_compute_dk2inv<elem_type>(q_tilde, k_tilde, step2, p.lse2_ptr,
        p.dstep2_ptr, p.dK2_inv_ptr, BH, D, m, p.stream);

    launch_kernel2_inv_bwd<elem_type>(q_tilde, k_tilde, p.lse2_ptr,
        p.k2_inv_ptr, p.dK2_inv_ptr, p.dQ_tilde_ptr, p.dK_tilde_ptr,
        BH, D, m, p.newton_iter, p.stream);

    launch_landmark_bwd<elem_type>(p.dQ_tilde_ptr, p.dK_tilde_ptr, dQ, dK,
        BH, N, D, m, p.stream);

    int total = BH * N * D;
    launch_scale_inplace<elem_type>(dQ, total, p.scale, p.stream);
    launch_scale_inplace<elem_type>(dK, total, p.scale, p.stream);
}

void run_nystrom_bwd(NystromBwdParams &params) {
    params.set_derived();
    FP16_SWITCH(!params.is_bf16, [&] {
        run_nystrom_bwd_impl<elem_type>(params);
    });
}

void run_nystrom_bwd_fp32(NystromBwdParams &params) {
    params.set_derived();
    run_nystrom_bwd_impl<float>(params);
}

} // namespace flash_nystrom
