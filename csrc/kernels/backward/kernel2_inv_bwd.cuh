/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
// Header for the unrolled Newton-Schulz backward.
// Declarations only — definitions live in kernel2_inv_bwd.cu so we don't
// get multiple symbols when this header is included from both
// flash_nystrom.cu (debug hooks) and flash_nystrom_kernels.cu (orchestration).
#pragma once
#include <cuda_runtime.h>

namespace flash_nystrom {

// -- Production launch wrapper used by run_nystrom_bwd_impl --
//
// Unrolls all NS backward iterations and runs the final softmax-bwd step.
// Inputs:
//   q_tilde, k_tilde     (BH, m, D)               — input/output dtype
//   lse2                 (unused, kept for ABI)
//   k2_inv               (unused — Z_N is in ns_iterates)
//   dK2_inv_in           (BH, m, m)  FP32         — incoming gradient dZ_N
//   ns_iterates          (BH, newton_iter+1, m, m) FP32 — Z_0 .. Z_N
//   K2_softmax           (BH, m, m) FP32          — softmax(QK^T) output
// Outputs (accumulated):
//   dQ_tilde, dK_tilde   (BH, m, D) FP32
// Workspace (caller-allocated):
//   dZ_workspace         (BH, m, m) FP32 — rolling dZ
//   dK2_workspace        (BH, m, m) FP32 — accumulator
template <typename scalar_t>
void launch_kernel2_inv_bwd(
    const scalar_t* q_tilde, const scalar_t* k_tilde,
    const float* lse2,
    const float* k2_inv,
    const float* dK2_inv_in,
    const float* ns_iterates,
    const float* K2_softmax,
    float* dQ_tilde, float* dK_tilde,
    float* dZ_workspace,
    float* dK2_workspace,
    int BH, int D, int m, int newton_iter, cudaStream_t stream);

// -- Test-only standalone launchers (used by debug pybind hooks) --
//
// Single backward step. Caller provides:
//   K2_in   (BH, m, m) FP32 — softmax K2
//   Z_j_in  (BH, m, m) FP32 — Z_j (per-bh stride = m*m)
//   dZ_in   (BH, m, m) FP32 — dZ_{j+1}
//   dZ_out  (BH, m, m) FP32 — receives dZ_j (overwritten)
//   dK2_acc (BH, m, m) FP32 — atomicAdd accumulator (caller zeroes if needed)
void launch_ns_bwd_step_test(
    const float* K2_in, const float* Z_j_in, const float* dZ_in,
    float* dZ_out, float* dK2_acc,
    int BH, int m, cudaStream_t stream);

// Final step. Caller provides:
//   q_tilde, k_tilde (BH, m, D) FP32
//   K2_in           (BH, m, m) FP32 — softmax K2
//   dZ0_in          (BH, m, m) FP32 — dZ_0
//   dK2_inout       (BH, m, m) FP32 — kernel adds dZ0^T/c then uses for softmax bwd
//   dQ_tilde_out    (BH, m, D) FP32 — kernel does plain += (caller zeroes)
//   dK_tilde_out    (BH, m, D) FP32 — kernel does plain += (caller zeroes)
void launch_ns_bwd_final_test(
    const float* q_tilde, const float* k_tilde,
    const float* K2_in, const float* dZ0_in,
    float* dK2_inout, float* dQ_tilde_out, float* dK_tilde_out,
    int BH, int D, int m, cudaStream_t stream);

} // namespace flash_nystrom
