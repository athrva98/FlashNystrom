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

// Forward declarations of the kernel functions. Needed by host code that
// wants to take their addresses (e.g. for occupancy queries). Definitions
// and explicit template instantiations live in kernel2_inv_bwd.cu. The
// __restrict__ qualifiers must match the definition exactly so that the
// declared function type is identical and the linker resolves the symbol.
__global__ void ns_bwd_step_kernel(
    const float* __restrict__ K2_in,
    const float* __restrict__ Z_j_in,
    int Z_j_bh_stride,
    float* __restrict__ dZ_inout,
    float* __restrict__ dK2_acc,
    int m);

template <typename scalar_t>
__global__ void ns_bwd_final_kernel(
    const scalar_t* __restrict__ q_tilde,
    const scalar_t* __restrict__ k_tilde,
    const float* __restrict__ K2_in,
    const float* __restrict__ dZ0_in,
    float* __restrict__ dK2_acc,
    float* __restrict__ dQ_tilde,
    float* __restrict__ dK_tilde,
    int D, int m);

// -- Production launch wrapper used by run_nystrom_bwd_impl --
//
// Unrolls all NS backward iterations and runs the final softmax-bwd step.
// The cuBLAS + CUDA-graph implementation owns its own persistent workspaces
// (thread-local NsBwdGraphState cache keyed by shape). The caller does not
// pass any scratch tensors: inputs are memcpy'd into the workspaces, the
// captured graph is replayed, outputs are memcpy'd back.
//
// Inputs:
//   q_tilde, k_tilde   (BH, m, D)               — input dtype
//   dK2_inv_in         (BH, m, m)  FP32         — incoming gradient dZ_J (J = newton_iter)
//   ns_iterates        (BH, newton_iter+1, m, m) FP32 — Z_0 .. Z_J from forward
//   K2_softmax         (BH, m, m) FP32          — softmax(QK^T) output
// Outputs (accumulated; caller-allocated, copied in then back out):
//   dQ_tilde, dK_tilde (BH, m, D) FP32
template <typename scalar_t>
void launch_kernel2_inv_bwd(
    const scalar_t* q_tilde, const scalar_t* k_tilde,
    const float* dK2_inv_in,
    const float* ns_iterates,
    const float* K2_softmax,
    float* dQ_tilde, float* dK_tilde,
    int BH, int D, int m, int newton_iter, cudaStream_t stream,
    float kappa_star = 0.0f);

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

// Free all thread-local NS-backward graph caches and workspaces. Use to
// reclaim GPU memory after a shape change, or before measuring residual
// memory usage. Per-dtype caches: float, half_t, bfloat16_t.
void reset_ns_bwd_caches();

} // namespace flash_nystrom
