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
    int newton_iter;
    bool is_bf16;
    // Tikhonov ridge target condition number: the pinv inverts
    // M = K2^T K2 + lambda*I with lambda = (||K2||_1 ||K2||_inf)/kappa_star,
    // guaranteeing cond(M) <= kappa_star. 0 disables the ridge (raw K2 pinv).
    // Threaded from the Python API (no longer an env var); validated at entry.
    float kappa_star;
    // Route the pinv through the tf32 tensor-core Newton-Schulz chain (faster,
    // verified accurate). Only applies at num_landmarks == 64; the scalar fp32
    // kernel is used otherwise regardless of this flag.
    bool use_tc_pinv;

    // q_ptr/k_ptr hold the SCALED copies (q*scale, k*scale) that the forward
    // kernels read and that are saved for the backward. q_in_ptr/k_in_ptr are
    // the ORIGINAL unscaled user tensors, read by the landmark kernel and used
    // as the source of the scaled copy. Folding the scale into that copy (one
    // scaled_copy pass) removes the redundant clone + scale_inplace double pass
    // over Q and K, which was ~44% of the forward at high batch*head.
    const void* __restrict__ q_in_ptr;    // (B, H, N, D) unscaled
    const void* __restrict__ k_in_ptr;    // (B, H, N, D) unscaled
    void* __restrict__ q_ptr;             // (B, H, N, D) scaled (written by scaled_copy)
    void* __restrict__ k_ptr;             // (B, H, N, D) scaled (written by scaled_copy)
    void* __restrict__ v_ptr;             // (B, H, N, D)
    void* __restrict__ o_ptr;             // (B, H, N, D)

    void* __restrict__ q_tilde_ptr;       // (B, H, m, D)
    void* __restrict__ k_tilde_ptr;       // (B, H, m, D)
    float* __restrict__ kernel2_inv_ptr;  // (B, H, m, m) FP32
    void* __restrict__ step2_ptr;         // (B, H, m, D)
    // B = softmax(Q_tilde @ K^T) @ V — saved from forward so the backward can
    // skip the N-walk in compute_dk2inv. Same dtype as Q/K/V. Nullable: if
    // null, the forward kernel does not emit B (used by FP32 scalar path).
    void* __restrict__ b_ptr;             // (B, H, m, D) or nullptr
    float* __restrict__ softmax1_lse_ptr; // (B, H, N)
    float* __restrict__ softmax2_lse_ptr; // (B, H, m)
    float* __restrict__ softmax3_lse_ptr; // (B, H, m)
    float* __restrict__ ns_iterates_ptr;  // (B, H, newton_iter+1, m, m) FP32 — REQUIRED
    float* __restrict__ k2_softmax_ptr;   // (B, H, m, m) FP32 — REQUIRED for backward

    cudaStream_t stream;

    int BH;
    float scale;

    void set_derived() {
        BH = batch_size * num_heads;
        scale = powf(static_cast<float>(head_dim), -0.25f);
    }
};

// -- backward --


struct NystromBwdParams {
    int batch_size, num_heads, seq_len, head_dim, num_landmarks;
    int newton_iter;
    bool is_bf16;
    // Opt-in tensor-core path for compute_dk2inv. Default false → FP32 scalar.
    // True only for FP16/BF16 input dtype; trades a small precision drop for
    // a large bwd latency win at large N. See compute_dk2inv.cuh.
    bool fast_dk2inv;
    // Tikhonov ridge target cond(M); must match the forward's kappa_star so the
    // backward inverts the same M = K2^T K2 + lambda*I. 0 = no ridge. Threaded
    // from the Python API (no longer an env var).
    float kappa_star;

    // Forward saved tensors (const, read-only)
    const void* __restrict__ q_s_ptr;        // (B,H,N,D) scaled Q
    const void* __restrict__ k_s_ptr;        // (B,H,N,D) scaled K
    const void* __restrict__ v_ptr;          // (B,H,N,D)
    const void* __restrict__ q_tilde_ptr;    // (B,H,m,D)
    const void* __restrict__ k_tilde_ptr;    // (B,H,m,D)
    const float* __restrict__ k2_inv_ptr;    // (B,H,m,m) FP32
    const void* __restrict__ step2_ptr;      // (B,H,m,D)
    // B = softmax(Q_tilde @ K^T) @ V saved from the forward, same dtype as
    // Q/K/V. compute_dk2inv reuses this instead of recomputing the N-walk.
    // Nullable for the FP32 scalar path which falls back to N-walking.
    const void* __restrict__ b_ptr;          // (B,H,m,D) or nullptr
    const void* __restrict__ o_ptr;          // (B,H,N,D) forward output
    const float* __restrict__ lse1_ptr;      // (B,H,N)
    const float* __restrict__ lse2_ptr;      // (B,H,m)
    const float* __restrict__ lse3_ptr;      // (B,H,m)
    const float* __restrict__ ns_iterates_ptr; // (B,H,newton_iter+1,m,m) FP32 — Z_0..Z_N
    const float* __restrict__ k2_softmax_ptr;  // (B,H,m,m) FP32 — softmax K2 (saved from fwd)

    // Gradient input
    const void* __restrict__ dO_ptr;         // (B,H,N,D)

    // Gradient outputs (written)
    void* __restrict__ dQ_ptr;               // (B,H,N,D) zero-initialized
    void* __restrict__ dK_ptr;               // (B,H,N,D) zero-initialized
    void* __restrict__ dV_ptr;               // (B,H,N,D) zero-initialized

    // Intermediate FP32 accumulators (zero-initialized by caller)
    float* __restrict__ dstep2_ptr;          // (B,H,m,D) FP32
    float* __restrict__ dQ_tilde_ptr;        // (B,H,m,D) FP32
    float* __restrict__ dK_tilde_ptr;        // (B,H,m,D) FP32
    float* __restrict__ dK2_inv_ptr;         // (B,H,m,m) FP32
    float* __restrict__ D1_ptr;              // (B,H,N) FP32 precomputed dot(dO,O)
    float* __restrict__ D3_ptr;              // (B,H,m) FP32 precomputed sum_n A3[i,n]*dP3[i,n]
    // Split-K workspace for the dQ_tilde accumulator in kernel3_bwd_tc.
    // Shape (num_splits, B, H, m, D) FP32, zero-initialized. The bwd kernel
    // writes each tile's dQ_tilde contribution to its own split slot
    // (tile_idx % num_splits) instead of all tiles atomicAdding to the same
    // m*D cells; a small reduction kernel then sums across the split axis
    // into dQ_tilde_ptr. num_splits = 0 disables split-K (legacy atomicAdd
    // straight to dQ_tilde_ptr).
    float* __restrict__ dQ_tilde_split_ptr;  // (num_splits, B, H, m, D) FP32 or nullptr
    int num_splits;

    // Split-K workspaces for kernel1_bwd_tc's dstep2 / dK_tilde accumulators,
    // same scheme as dQ_tilde_split: row-tile tile_idx atomicAdds into slot
    // (tile_idx % k1_num_splits), then a reduce sums the slots into
    // dstep2_ptr / dK_tilde_ptr. k1_num_splits == 0 (or nullptr) keeps the
    // legacy direct atomicAdd. The FP32 scalar path ignores these.
    float* __restrict__ dstep2_split_ptr;    // (k1_num_splits, B, H, m, D) FP32 or nullptr
    float* __restrict__ dK_tilde_split_ptr;  // (k1_num_splits, B, H, m, D) FP32 or nullptr
    int k1_num_splits;

    // Intermediate for TC kernel3_bwd: dO3 = K2_inv^T @ dstep2, stored as elem_type
    void* __restrict__ dO3_ptr;              // (B,H,m,D) FP16/BF16, or nullptr for FP32

    // No NS-backward workspace pointers here: launch_kernel2_inv_bwd owns
    // its own persistent thread-local NsBwdGraphState cache (workspaces
    // sized by shape, reused across calls).

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

// Free the kernel3 split-N scratch buffers. Defined in flash_nystrom_kernels.cu
// (bridges to the inline reset_kernel3_scratch in kernel3_output_fused.cuh) so
// the pybind reset_caches can free them without flash_nystrom.cu pulling in the
// full CUTLASS header.
void reset_kernel3_caches();
// Free the thread-local TC-pinv forward graph cache (K2InvTcGraph). Defined in
// flash_nystrom_kernels.cu.
void reset_k2inv_tc_caches();

} // namespace flash_nystrom
