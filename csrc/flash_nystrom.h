/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 *
 * FlashNystrom parameter structs for forward and backward passes.
 * All pointers are void* following FlashAttention's pattern — the actual
 * type dispatch happens in the kernel launch layer via FP16_SWITCH.
 ******************************************************************************/
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstddef>
#include <cmath>

namespace flash_nystrom {

// -- forward params --


struct NystromParams {
    int batch_size, num_heads, seq_len, head_dim, num_landmarks;
    int newton_iter, conv_kernel_size;
    bool is_bf16;

    void* __restrict__ q_ptr;             // (B, H, N, D)
    void* __restrict__ k_ptr;             // (B, H, N, D)
    void* __restrict__ v_ptr;             // (B, H, N, D)
    void* __restrict__ conv_weight_ptr;   // (H, ks) or nullptr
    void* __restrict__ o_ptr;             // (B, H, N, D)

    void* __restrict__ q_tilde_ptr;       // (B, H, m, D)
    void* __restrict__ k_tilde_ptr;       // (B, H, m, D)
    float* __restrict__ kernel2_inv_ptr;  // (B, H, m, m) FP32
    void* __restrict__ step2_ptr;         // (B, H, m, D)
    float* __restrict__ softmax1_lse_ptr; // (B, H, N)
    float* __restrict__ softmax2_lse_ptr; // (B, H, m)
    float* __restrict__ softmax3_lse_ptr; // (B, H, m)
    float* __restrict__ ns_iterates_ptr;  // (B, H, newton_iter, m, m) FP32, or nullptr

    cudaStream_t stream;

    int BH;
    int seg_len;
    float scale;

    void set_derived() {
        BH = batch_size * num_heads;
        seg_len = (seq_len + num_landmarks - 1) / num_landmarks;
        scale = powf(static_cast<float>(head_dim), -0.25f);
    }
};

// -- backward --


struct NystromBwdParams {
    int batch_size, num_heads, seq_len, head_dim, num_landmarks;
    int newton_iter, conv_kernel_size;
    bool is_bf16;

    // Forward saved tensors (const, read-only)
    const void* __restrict__ q_s_ptr;        // (B,H,N,D) scaled Q
    const void* __restrict__ k_s_ptr;        // (B,H,N,D) scaled K
    const void* __restrict__ v_ptr;          // (B,H,N,D)
    const void* __restrict__ q_tilde_ptr;    // (B,H,m,D)
    const void* __restrict__ k_tilde_ptr;    // (B,H,m,D)
    const float* __restrict__ k2_inv_ptr;    // (B,H,m,m) FP32
    const void* __restrict__ step2_ptr;      // (B,H,m,D)
    const void* __restrict__ o_ptr;          // (B,H,N,D) forward output
    const float* __restrict__ lse1_ptr;      // (B,H,N)
    const float* __restrict__ lse2_ptr;      // (B,H,m)
    const float* __restrict__ lse3_ptr;      // (B,H,m)
    const float* __restrict__ ns_iterates_ptr; // (B,H,newton_iter,m,m) FP32
    const void* __restrict__ conv_weight_ptr;  // (H,ks) or nullptr

    // Gradient input
    const void* __restrict__ dO_ptr;         // (B,H,N,D)

    // Gradient outputs (written)
    void* __restrict__ dQ_ptr;               // (B,H,N,D) zero-initialized
    void* __restrict__ dK_ptr;               // (B,H,N,D) zero-initialized
    void* __restrict__ dV_ptr;               // (B,H,N,D) zero-initialized
    void* __restrict__ dconv_weight_ptr;     // (H,ks) or nullptr

    // Intermediate FP32 accumulators (zero-initialized by caller)
    float* __restrict__ dstep2_ptr;          // (B,H,m,D) FP32
    float* __restrict__ dQ_tilde_ptr;        // (B,H,m,D) FP32
    float* __restrict__ dK_tilde_ptr;        // (B,H,m,D) FP32
    float* __restrict__ dK2_inv_ptr;         // (B,H,m,m) FP32
    float* __restrict__ D1_ptr;              // (B,H,N) FP32 precomputed dot(dO,O)

    cudaStream_t stream;

    int BH;
    float scale;

    void set_derived() {
        BH = batch_size * num_heads;
        scale = powf(static_cast<float>(head_dim), -0.25f);
    }
};

// Forward
void run_nystrom_fwd(NystromParams &params);
void run_nystrom_fwd_fp32(NystromParams &params);

// Backward
void run_nystrom_bwd(NystromBwdParams &params);
void run_nystrom_bwd_fp32(NystromBwdParams &params);

} // namespace flash_nystrom
