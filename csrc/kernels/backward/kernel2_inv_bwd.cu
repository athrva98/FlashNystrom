/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
// Single-definition .cu for the unrolled Newton-Schulz backward.
// The header (kernel2_inv_bwd.cuh) only declares; everything in this TU.

#include "kernels/backward/kernel2_inv_bwd.cuh"
#include "cublas_helpers.cuh"
#include "utils.h"

#include <cfloat>
#include <cutlass/numeric_types.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>

namespace flash_nystrom {

// =========================================================================
// One backward iteration. Reads K2 (per-bh) and Z_j (per-bh, with explicit
// stride into ns_iterates), reads dZ_{j+1} from dZ_inout, writes dZ_j to
// dZ_inout, atomicAdds dK2 contribution to dK2_acc.
//
// SMEM: 6 * m * m * sizeof(float) = 96 KB at m = 64.  Needs SMEM opt-in.
//
// Forward iteration j:
//   M     = K2 @ Z_j
//   U     = 7*I - M
//   V     = 15*I - M @ U
//   T     = 13*I - M @ V
//   Z_{j+1} = (1/4) * Z_j @ T
//
// Backward at iteration j (given dZ_{j+1}, compute dZ_j and dK2_j_contrib):
//   dT             = (1/4) * Z_j^T @ dZ_{j+1}
//   dZ_outer       = (1/4) * dZ_{j+1} @ T^T
//   dM_T           = - dT @ V^T
//   dV             = - M^T @ dT
//   dM_V           = - dV @ U^T = -7*dV + dV @ M^T
//   dU             = - M^T @ dV
//   dM_U           = - dU
//   dM             = dM_T + dM_V + dM_U
//   dK2_j_contrib  = dM @ Z_j^T          (atomicAdd to dK2)
//   dZ_inner       = K2^T @ dM
//   dZ_j           = dZ_outer + dZ_inner
// =========================================================================

__global__ void ns_bwd_step_kernel(
    const float* __restrict__ K2_in,         // (BH, m, m)
    const float* __restrict__ Z_j_in,        // base of iterate j
    int   Z_j_bh_stride,                     // floats per bh
    float*       __restrict__ dZ_inout,      // (BH, m, m)
    float*       __restrict__ dK2_acc,       // (BH, m, m)
    int m
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int mm = m * m;

    extern __shared__ float smem[];
    float* sK2  = smem;            // K2
    float* sZ   = smem + mm;       // Z_j
    float* sM   = smem + 2 * mm;   // M = K2 @ Z
    float* sV   = smem + 3 * mm;   // V (then dV)
    float* sdZ  = smem + 4 * mm;   // dZ_in (then dM accumulator)
    float* sTmp = smem + 5 * mm;   // workspace (T, dT, dU)

    const float* K2_bh = K2_in + bh * mm;
    const float* Z_bh  = Z_j_in + bh * Z_j_bh_stride;
    float* dZ_bh       = dZ_inout + bh * mm;
    float* dK2_bh      = dK2_acc + bh * mm;

    // Load K2, Z_j, dZ_in
    for (int idx = tid; idx < mm; idx += nthreads) {
        sK2[idx] = K2_bh[idx];
        sZ[idx]  = Z_bh[idx];
        sdZ[idx] = dZ_bh[idx];
    }
    __syncthreads();

    // M = K2 @ Z
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        float acc = 0.0f;
        for (int k = 0; k < m; k++) acc += sK2[r * m + k] * sZ[k * m + c];
        sM[idx] = acc;
    }
    __syncthreads();

    // V = 15I - M @ (7I - M).  First sV = M @ U = 7M - M^2.
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        float acc = 0.0f;
        for (int k = 0; k < m; k++) {
            float U_kc = ((k == c) ? 7.0f : 0.0f) - sM[k * m + c];
            acc += sM[r * m + k] * U_kc;
        }
        sV[idx] = acc;
    }
    __syncthreads();
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        sV[idx] = ((r == c) ? 15.0f : 0.0f) - sV[idx];
    }
    __syncthreads();

    // T = 13I - M @ V into sTmp
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        float acc = 0.0f;
        for (int k = 0; k < m; k++) acc += sM[r * m + k] * sV[k * m + c];
        sTmp[idx] = ((r == c) ? 13.0f : 0.0f) - acc;
    }
    __syncthreads();

    // dZ_outer = (1/4) dZ_in @ T^T  -->  write directly to dZ_bh
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        float acc = 0.0f;
        for (int k = 0; k < m; k++) acc += sdZ[r * m + k] * sTmp[c * m + k];
        dZ_bh[idx] = 0.25f * acc;
    }
    __syncthreads();
    // sTmp (T) no longer needed.

    // dT = (1/4) Z^T @ dZ_in into sTmp
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        float acc = 0.0f;
        for (int k = 0; k < m; k++) acc += sZ[k * m + r] * sdZ[k * m + c];
        sTmp[idx] = 0.25f * acc;
    }
    __syncthreads();
    // sdZ (dZ_in) no longer needed; reused as dM accumulator.

    // dM_from_T = -dT @ V^T  -->  start dM in sdZ
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        float acc = 0.0f;
        for (int k = 0; k < m; k++) acc += sTmp[r * m + k] * sV[c * m + k];
        sdZ[idx] = -acc;
    }
    __syncthreads();

    // dV = -M^T @ dT  -->  overwrite sV
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        float acc = 0.0f;
        for (int k = 0; k < m; k++) acc += sM[k * m + r] * sTmp[k * m + c];
        sV[idx] = -acc;
    }
    __syncthreads();
    // sTmp (dT) no longer needed.

    // dM_from_V = -7*dV + dV @ M^T  -->  accumulate into sdZ
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        float acc_dvMT = 0.0f;
        for (int k = 0; k < m; k++) acc_dvMT += sV[r * m + k] * sM[c * m + k];
        float dM_from_V = -7.0f * sV[idx] + acc_dvMT;
        sdZ[idx] += dM_from_V;
    }
    __syncthreads();

    // dU = -M^T @ dV  -->  into sTmp
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        float acc = 0.0f;
        for (int k = 0; k < m; k++) acc += sM[k * m + r] * sV[k * m + c];
        sTmp[idx] = -acc;
    }
    __syncthreads();
    // sV (dV) no longer needed.

    // dM_from_U = -dU  -->  accumulate into sdZ
    for (int idx = tid; idx < mm; idx += nthreads) {
        sdZ[idx] += -sTmp[idx];
    }
    __syncthreads();
    // sdZ now holds full dM.

    // dK2_contrib = dM @ Z^T  -->  atomicAdd to dK2_acc
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        float acc = 0.0f;
        for (int k = 0; k < m; k++) acc += sdZ[r * m + k] * sZ[c * m + k];
        atomicAdd(&dK2_bh[idx], acc);
    }
    __syncthreads();

    // dZ_inner = K2^T @ dM  -->  add to dZ_bh (currently holds dZ_outer)
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        float acc = 0.0f;
        for (int k = 0; k < m; k++) acc += sK2[k * m + r] * sdZ[k * m + c];
        float prev = dZ_bh[idx];
        dZ_bh[idx] = prev + acc;
    }
    __syncthreads();
}

// =========================================================================
// Final-step kernel: applied AFTER all NS iterations are unrolled.
//
// Forward Z_0 init:  Z_0 = K2^T / c,  c = ||K2||_1 * ||K2||_inf.
//   ||K2||_1   = max_j sum_i K2[i,j]   (max column sum,  argmax col = jc1)
//   ||K2||_inf = max_i sum_j K2[i,j]   (max row sum,     argmax row = ir_inf)
// Both norms are differentiable w.r.t. K2 via the max() argmax positions.
//
// Backward dK2 contributions from Z_0 = K2^T / c:
//   (a) Direct:  dK2[r, c] += dZ_0[c, r] / c                       (all r, c)
//   (b) via dnorm_1:    dK2[r, jc1]   += -S * norm_inf / c^2       (all r)
//   (c) via dnorm_inf:  dK2[ir_inf, c] += -S * norm_1   / c^2      (all c)
// where S = trace(dZ_0 @ K2) = sum_{a,b} dZ_0[a, b] * K2[b, a].
// (Sign of K2 entries: K2 >= 0 since it's a softmax output, so |K2| = K2.)
//
// After the dK2 init step:
//   D_i        = sum_j dK2[i,j] * K2[i,j]
//   dS2[i,j]   = K2[i,j] * (dK2[i,j] - D_i)
//   dQ_tilde  += dS2 @ k_tilde
//   dK_tilde  += dS2^T @ q_tilde
// =========================================================================

template <typename scalar_t>
__global__ void ns_bwd_final_kernel(
    const scalar_t* __restrict__ q_tilde,    // (BH, m, D)
    const scalar_t* __restrict__ k_tilde,    // (BH, m, D)
    const float*    __restrict__ K2_in,       // (BH, m, m)
    const float*    __restrict__ dZ0_in,      // (BH, m, m)
    float*          __restrict__ dK2_acc,     // (BH, m, m)
    float*          __restrict__ dQ_tilde,    // (BH, m, D)
    float*          __restrict__ dK_tilde,    // (BH, m, D)
    int D, int m
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int mm = m * m;

    extern __shared__ float smem[];
    float* sK2     = smem;
    float* sdK2    = smem + mm;
    float* sdS2    = smem + 2 * mm;
    float* scratch = smem + 3 * mm;

    const float* K2_bh  = K2_in + bh * mm;
    const float* dZ0_bh = dZ0_in + bh * mm;
    float* dK2_bh = dK2_acc + bh * mm;

    for (int idx = tid; idx < mm; idx += nthreads) sK2[idx]  = K2_bh[idx];
    for (int idx = tid; idx < mm; idx += nthreads) sdK2[idx] = dK2_bh[idx];
    __syncthreads();

    // ----- Norms and argmax positions -----
    // Compute column sums and find max (=norm1) + argmax column (jc1).
    // Use the dS2 region as scratch for the per-column / per-row sum vectors;
    // dS2 isn't touched until later.
    float* col_sums = sdS2;            // size m
    float* row_sums = sdS2 + m;        // size m
    for (int j = tid; j < m; j += nthreads) {
        float s = 0.0f;
        for (int i = 0; i < m; i++) s += sK2[i * m + j];   // K2 >= 0 (softmax)
        col_sums[j] = s;
    }
    for (int i = tid; i < m; i += nthreads) {
        float s = 0.0f;
        for (int j = 0; j < m; j++) s += sK2[i * m + j];
        row_sums[i] = s;
    }
    __syncthreads();

    // Single-thread reduction for argmax (m <= 64, trivial).
    __shared__ float s_norm1;
    __shared__ float s_norm_inf;
    __shared__ int   s_jc1;
    __shared__ int   s_ir_inf;
    if (tid == 0) {
        float maxv = -FLT_MAX; int maxi = 0;
        for (int j = 0; j < m; j++) {
            // Strict > matches torch.max's first-occurrence-on-ties convention.
            if (col_sums[j] > maxv) { maxv = col_sums[j]; maxi = j; }
        }
        s_norm1 = maxv;
        s_jc1   = maxi;
        maxv = -FLT_MAX; maxi = 0;
        for (int i = 0; i < m; i++) {
            if (row_sums[i] > maxv) { maxv = row_sums[i]; maxi = i; }
        }
        s_norm_inf = maxv;
        s_ir_inf   = maxi;
    }
    __syncthreads();

    const float norm1   = s_norm1;
    const float norm_inf = s_norm_inf;
    const int   jc1     = s_jc1;
    const int   ir_inf  = s_ir_inf;
    const float c_val   = fmaxf(norm1 * norm_inf, 1e-12f);
    const float inv_c   = 1.0f / c_val;
    const float inv_c2  = inv_c * inv_c;

    // ----- S = sum_{a,b} dZ_0[a, b] * K2[b, a]   (= trace(dZ_0 @ K2)) -----
    float local_S = 0.0f;
    for (int idx = tid; idx < mm; idx += nthreads) {
        int a = idx / m, b = idx % m;
        local_S += dZ0_bh[a * m + b] * sK2[b * m + a];
    }
    float S = block_reduce_sum(local_S, scratch);
    __syncthreads();

    // (a) Direct: dK2[r, c] += dZ_0[c, r] / c_val
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        sdK2[idx] += dZ0_bh[c * m + r] * inv_c;
    }

    // (b) Column term from d(norm_1): dK2[*, jc1] += -S * norm_inf / c^2
    {
        float col_term = -S * norm_inf * inv_c2;
        for (int r = tid; r < m; r += nthreads) {
            sdK2[r * m + jc1] += col_term;
        }
    }

    // (c) Row term from d(norm_inf): dK2[ir_inf, *] += -S * norm_1 / c^2
    {
        float row_term = -S * norm1 * inv_c2;
        for (int j = tid; j < m; j += nthreads) {
            sdK2[ir_inf * m + j] += row_term;
        }
    }
    __syncthreads();

    // dS2 = K2 * (dK2 - rowsum(dK2 * K2))
    for (int i = tid; i < m; i += nthreads) {
        float D_i = 0.0f;
        for (int j = 0; j < m; j++) D_i += sdK2[i * m + j] * sK2[i * m + j];
        for (int j = 0; j < m; j++) {
            sdS2[i * m + j] = sK2[i * m + j] * (sdK2[i * m + j] - D_i);
        }
    }
    __syncthreads();

    // Persist updated dK2 back to GMEM (so caller sees dK2_after_init)
    for (int idx = tid; idx < mm; idx += nthreads) dK2_bh[idx] = sdK2[idx];

    const scalar_t* qt = q_tilde + bh * m * D;
    const scalar_t* kt = k_tilde + bh * m * D;
    float* dQt_bh = dQ_tilde + bh * m * D;
    float* dKt_bh = dK_tilde + bh * m * D;

    // dQ_tilde += dS2 @ k_tilde
    for (int idx = tid; idx < m * D; idx += nthreads) {
        int i = idx / D, d = idx % D;
        float sum = 0.0f;
        for (int j = 0; j < m; j++) sum += sdS2[i * m + j] * to_float(kt[j * D + d]);
        dQt_bh[idx] += sum;
    }

    // dK_tilde += dS2^T @ q_tilde
    for (int idx = tid; idx < m * D; idx += nthreads) {
        int j = idx / D, d = idx % D;
        float sum = 0.0f;
        for (int i = 0; i < m; i++) sum += sdS2[i * m + j] * to_float(qt[i * D + d]);
        dKt_bh[idx] += sum;
    }
}

// =========================================================================
// cuBLAS-friendly variant of ns_bwd_final: does everything except the two
// trailing matmuls. Persists dK2 to GMEM and writes dS2 in scalar_t to a
// GMEM workspace; the caller then runs cublasGemmEx for
//   dQ_tilde += dS2 @ k_tilde
//   dK_tilde += dS2^T @ q_tilde
// with elem_type A and B and FP32 C, dispatching to TC for FP16/BF16.
// =========================================================================

template <typename scalar_t>
__global__ void ns_bwd_final_pre_kernel(
    const float*    __restrict__ K2_in,    // (BH, m, m)
    const float*    __restrict__ dZ0_in,   // (BH, m, m)
    float*          __restrict__ dK2_acc,  // (BH, m, m) FP32, updated in place
    scalar_t*       __restrict__ dS2_out,  // (BH, m, m) elem_type, written
    int m,
    float ridge_lambda                     // lambda*I added to K2 post-softmax in the forward
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int mm = m * m;

    extern __shared__ float smem[];
    float* sK2     = smem;
    float* sdK2    = smem + mm;
    float* sdS2    = smem + 2 * mm;
    float* scratch = smem + 3 * mm;

    const float* K2_bh  = K2_in + bh * mm;
    const float* dZ0_bh = dZ0_in + bh * mm;
    float* dK2_bh = dK2_acc + bh * mm;

    for (int idx = tid; idx < mm; idx += nthreads) sK2[idx]  = K2_bh[idx];
    for (int idx = tid; idx < mm; idx += nthreads) sdK2[idx] = dK2_bh[idx];
    __syncthreads();

    // Column and row sums of K2.
    float* col_sums = sdS2;
    float* row_sums = sdS2 + m;
    for (int j = tid; j < m; j += nthreads) {
        float s = 0.0f;
        for (int i = 0; i < m; i++) s += sK2[i * m + j];
        col_sums[j] = s;
    }
    for (int i = tid; i < m; i += nthreads) {
        float s = 0.0f;
        for (int j = 0; j < m; j++) s += sK2[i * m + j];
        row_sums[i] = s;
    }
    __syncthreads();

    __shared__ float s_norm1;
    __shared__ float s_norm_inf;
    __shared__ int   s_jc1;
    __shared__ int   s_ir_inf;
    if (tid == 0) {
        float maxv = -FLT_MAX; int maxi = 0;
        for (int j = 0; j < m; j++) {
            if (col_sums[j] > maxv) { maxv = col_sums[j]; maxi = j; }
        }
        s_norm1 = maxv;
        s_jc1   = maxi;
        maxv = -FLT_MAX; maxi = 0;
        for (int i = 0; i < m; i++) {
            if (row_sums[i] > maxv) { maxv = row_sums[i]; maxi = i; }
        }
        s_norm_inf = maxv;
        s_ir_inf   = maxi;
    }
    __syncthreads();

    const float norm1    = s_norm1;
    const float norm_inf = s_norm_inf;
    const int   jc1      = s_jc1;
    const int   ir_inf   = s_ir_inf;
    const float c_val    = fmaxf(norm1 * norm_inf, 1e-12f);
    const float inv_c    = 1.0f / c_val;
    const float inv_c2   = inv_c * inv_c;

    // S = trace(dZ_0 @ K2)
    float local_S = 0.0f;
    for (int idx = tid; idx < mm; idx += nthreads) {
        int a = idx / m, b = idx % m;
        local_S += dZ0_bh[a * m + b] * sK2[b * m + a];
    }
    float S = block_reduce_sum(local_S, scratch);
    __syncthreads();

    // dK2 += dZ_0^T / c (direct)
    for (int idx = tid; idx < mm; idx += nthreads) {
        int r = idx / m, c = idx % m;
        sdK2[idx] += dZ0_bh[c * m + r] * inv_c;
    }
    // The column/row corrections below read-modify-write cells this loop also
    // wrote (column jc1, row ir_inf), under a different thread->index mapping.
    // The three += passes must be serialized or they race on the shared cells.
    __syncthreads();

    // dK2[*, jc1] += -S * norm_inf / c^2
    {
        float col_term = -S * norm_inf * inv_c2;
        for (int r = tid; r < m; r += nthreads) {
            sdK2[r * m + jc1] += col_term;
        }
    }
    __syncthreads();  // cell (ir_inf, jc1) is RMW by this pass and the next

    // dK2[ir_inf, *] += -S * norm_1 / c^2
    {
        float row_term = -S * norm1 * inv_c2;
        for (int j = tid; j < m; j += nthreads) {
            sdK2[ir_inf * m + j] += row_term;
        }
    }
    __syncthreads();

    // dS2 = K2_sm * (dK2 - rowsum(dK2 * K2_sm)), the softmax-Jacobian VJP.
    // K2_sm is the UN-ridged softmax K2: the forward added lambda*I AFTER the
    // softmax (for pinv conditioning) and saved the ridged K2, so subtract
    // lambda back off the diagonal here. The NS chain-rule and norm-gradient
    // pieces above intentionally keep the ridged K2 (the NS inverted that).
    for (int i = tid; i < m; i += nthreads) {
        float D_i = 0.0f;
        for (int j = 0; j < m; j++) {
            float k2 = sK2[i * m + j] - ((i == j) ? ridge_lambda : 0.0f);
            D_i += sdK2[i * m + j] * k2;
        }
        for (int j = 0; j < m; j++) {
            float k2 = sK2[i * m + j] - ((i == j) ? ridge_lambda : 0.0f);
            sdS2[i * m + j] = k2 * (sdK2[i * m + j] - D_i);
        }
    }
    __syncthreads();

    // Persist dK2 to GMEM and write dS2 (cast to scalar_t) to GMEM.
    for (int idx = tid; idx < mm; idx += nthreads) dK2_bh[idx] = sdK2[idx];
    scalar_t* dS2_bh = dS2_out + bh * mm;
    for (int idx = tid; idx < mm; idx += nthreads) {
        dS2_bh[idx] = from_float<scalar_t>(sdS2[idx]);
    }
}

// =========================================================================
// Tikhonov-ridge backward helpers (kappa_star > 0). The forward inverts the
// symmetric PSD M = K2^T K2 + lambda*I and returns K2^+ = Z_J K2^T (Z_J =
// M^-1). The NS chain-rule machinery above is reused unchanged on M; these
// three kernels handle the M <-> K2 wrapping at the two ends.
// =========================================================================

// Re-add the Tikhonov ridge to M's diagonal at reconstruct time:
// lambda = (||K2||_1 * ||K2||_inf)/kappa_star per (b,h), identical to the
// forward kernel's lambda. M arrives as K2^T K2 (from a cuBLAS gemm).
__global__ void add_ridge_diag_kernel(
    float* __restrict__ M,             // (BH, m, m), ridge added in place
    const float* __restrict__ K2_in,   // (BH, m, m) un-ridged softmax K2 (>=0)
    int m, float kappa_star
) {
    const int bh = blockIdx.x, tid = threadIdx.x, nthreads = blockDim.x;
    const int mm = m * m;
    const float* K2 = K2_in + bh * mm;
    extern __shared__ float smem[];
    float* col = smem;          // m
    float* row = smem + m;      // m
    float* sc  = smem + 2 * m;  // reduction scratch
    for (int j = tid; j < m; j += nthreads) {
        float s = 0.0f; for (int i = 0; i < m; i++) s += K2[i * m + j];
        col[j] = s;
    }
    for (int i = tid; i < m; i += nthreads) {
        float s = 0.0f; for (int j = 0; j < m; j++) s += K2[i * m + j];
        row[i] = s;
    }
    __syncthreads();
    float lc = -FLT_MAX; for (int j = tid; j < m; j += nthreads) lc = fmaxf(lc, col[j]);
    float n1 = block_reduce_max(lc, sc);
    float lr = -FLT_MAX; for (int i = tid; i < m; i += nthreads) lr = fmaxf(lr, row[i]);
    float ninf = block_reduce_max(lr, sc);
    float lam = (n1 * ninf) / kappa_star;
    float* Mbh = M + bh * mm;
    for (int i = tid; i < m; i += nthreads) Mbh[i * m + i] += lam;
}

// Z_0-init gradient, on M (symmetric, >= 0). Forward Z_0 = M^T/(||M||_1 ||M||_inf).
// Same three dK2 contributions as ns_bwd_final_pre's first half, but the matrix
// is M, the output accumulates into dM, and there is no softmax-Jacobian here
// (that is applied later, on K2). dM_acc arrives holding the NS-loop's dM.
__global__ void ns_bwd_z0grad_kernel(
    const float* __restrict__ M_in,    // (BH, m, m) ridged M (>= 0)
    const float* __restrict__ dZ0_in,  // (BH, m, m) dZ_0
    float*       __restrict__ dM_acc,  // (BH, m, m) += Z_0-init grad
    int m
) {
    const int bh = blockIdx.x, tid = threadIdx.x, nthreads = blockDim.x;
    const int mm = m * m;
    extern __shared__ float smem[];
    float* sM      = smem;
    float* sdM     = smem + mm;
    float* tmp     = smem + 2 * mm;   // col/row sum vectors
    float* scratch = smem + 3 * mm;
    const float* M_bh   = M_in + bh * mm;
    const float* dZ0_bh = dZ0_in + bh * mm;
    float* dM_bh        = dM_acc + bh * mm;
    for (int idx = tid; idx < mm; idx += nthreads) { sM[idx] = M_bh[idx]; sdM[idx] = dM_bh[idx]; }
    __syncthreads();
    float* col = tmp; float* rowv = tmp + m;
    for (int j = tid; j < m; j += nthreads) { float s = 0.0f; for (int i = 0; i < m; i++) s += sM[i * m + j]; col[j] = s; }
    for (int i = tid; i < m; i += nthreads) { float s = 0.0f; for (int j = 0; j < m; j++) s += sM[i * m + j]; rowv[i] = s; }
    __syncthreads();
    __shared__ float s_n1, s_ninf; __shared__ int s_jc1, s_ir;
    if (tid == 0) {
        float mv = -FLT_MAX; int mi = 0;
        for (int j = 0; j < m; j++) if (col[j] > mv) { mv = col[j]; mi = j; }
        s_n1 = mv; s_jc1 = mi;
        mv = -FLT_MAX; mi = 0;
        for (int i = 0; i < m; i++) if (rowv[i] > mv) { mv = rowv[i]; mi = i; }
        s_ninf = mv; s_ir = mi;
    }
    __syncthreads();
    const float n1 = s_n1, ninf = s_ninf; const int jc1 = s_jc1, ir = s_ir;
    const float c = fmaxf(n1 * ninf, 1e-12f); const float ic = 1.0f / c; const float ic2 = ic * ic;
    float lS = 0.0f;
    for (int idx = tid; idx < mm; idx += nthreads) { int a = idx / m, b = idx % m; lS += dZ0_bh[a * m + b] * sM[b * m + a]; }
    float S = block_reduce_sum(lS, scratch);
    __syncthreads();
    for (int idx = tid; idx < mm; idx += nthreads) { int r = idx / m, c2 = idx % m; sdM[idx] += dZ0_bh[c2 * m + r] * ic; }
    __syncthreads();   // serialize the three RMW passes (cell (ir,jc1) is shared)
    { float ct = -S * ninf * ic2; for (int r = tid; r < m; r += nthreads) sdM[r * m + jc1] += ct; }
    __syncthreads();
    { float rt = -S * n1 * ic2;  for (int j = tid; j < m; j += nthreads) sdM[ir * m + j] += rt; }
    __syncthreads();
    for (int idx = tid; idx < mm; idx += nthreads) dM_bh[idx] = sdM[idx];
}

// Softmax-Jacobian VJP, on the un-ridged K2: dS2 = K2 * (dK2 - rowsum(dK2*K2)).
// dK2 is the assembled K2-gradient (dK2_a + K2(dM+dM^T)). Writes dS2 in elem_type
// for the trailing cuBLAS GemmEx (dQ_tilde, dK_tilde), exactly as ns_bwd_final_pre.
template <typename scalar_t>
__global__ void ns_bwd_softmax_jac_kernel(
    const float* __restrict__ K2_in,    // (BH, m, m) un-ridged softmax K2
    const float* __restrict__ dK2_in,   // (BH, m, m) dK2_final
    scalar_t*    __restrict__ dS2_out,  // (BH, m, m) elem_type
    int m
) {
    const int bh = blockIdx.x, tid = threadIdx.x, nthreads = blockDim.x;
    const int mm = m * m;
    extern __shared__ float smem[];
    float* sK2 = smem; float* sdK2 = smem + mm; float* sdS2 = smem + 2 * mm;
    const float* K2_bh = K2_in + bh * mm; const float* dK2_bh = dK2_in + bh * mm;
    for (int idx = tid; idx < mm; idx += nthreads) { sK2[idx] = K2_bh[idx]; sdK2[idx] = dK2_bh[idx]; }
    __syncthreads();
    for (int i = tid; i < m; i += nthreads) {
        float Di = 0.0f;
        for (int j = 0; j < m; j++) Di += sdK2[i * m + j] * sK2[i * m + j];
        for (int j = 0; j < m; j++) sdS2[i * m + j] = sK2[i * m + j] * (sdK2[i * m + j] - Di);
    }
    __syncthreads();
    scalar_t* dS2_bh = dS2_out + bh * mm;
    for (int idx = tid; idx < mm; idx += nthreads) dS2_bh[idx] = from_float<scalar_t>(sdS2[idx]);
}

// Row-major batched GemmEx wrapper. Supports FP16/BF16/FP32 A, B with FP32
// C. cuBLAS dispatches to tensor cores for FP16/BF16. Same row-major <-> col
// -major transposition trick as rm_sgemm_strided_batched in cublas_helpers.cuh.
template <typename scalar_t>
inline void rm_gemm_ex_strided_batched(
    cublasHandle_t handle,
    char opA, char opB,
    int M_rm, int N_rm, int K_rm,
    float alpha,
    const void* A, long long strideA,
    const void* B, long long strideB,
    float beta,
    float* C, long long strideC,
    int batch
) {
    cublasOperation_t cb_opA = (opB == 'T' || opB == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cb_opB = (opA == 'T' || opA == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda_rm = (opA == 'N' || opA == 'n') ? K_rm : M_rm;
    int ldb_rm = (opB == 'N' || opB == 'n') ? N_rm : K_rm;
    int ldc_rm = N_rm;
    cudaDataType_t ab_type;
    if constexpr (std::is_same_v<scalar_t, float>)                 ab_type = CUDA_R_32F;
    else if constexpr (std::is_same_v<scalar_t, cutlass::half_t>)  ab_type = CUDA_R_16F;
    else                                                            ab_type = CUDA_R_16BF;
    auto status = cublasGemmStridedBatchedEx(handle,
        cb_opA, cb_opB,
        N_rm, M_rm, K_rm,
        &alpha,
        B, ab_type, ldb_rm, strideB,
        A, ab_type, lda_rm, strideA,
        &beta,
        C, CUDA_R_32F, ldc_rm, strideC,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "[FlashNystrom] cublasGemmStridedBatchedEx failed with status "
            << static_cast<int>(status)
            << " (shape M_rm=" << M_rm << ", N_rm=" << N_rm
            << ", K_rm=" << K_rm << ", batch=" << batch << ")";
        throw std::runtime_error(oss.str());
    }
}

// =========================================================================
// Production launch wrapper (called from run_nystrom_bwd_impl).
// =========================================================================

// -- per-thread NS bwd graph cache ----------------------------------------
//
// The NS backward fires ~80 small cuBLAS Sgemm calls + a handful of small
// elementwise kernels + one ns_bwd_final per backward. Each launch is
// ~5 us of host-side overhead; the GPU compute per call is microseconds.
// Without graphs, host-side launch dominates: the per-iter NS cost was
// ~480 us legacy, ~160 us cuBLAS-without-graphs at m=64 BH=8.
//
// The graph capture records all the launches once per (m, BH, niter, D)
// shape and replays them with one cudaGraphLaunch on subsequent calls. The
// pointers must stay stable across replays, so we hold persistent workspace
// tensors here and memcpy the caller's inputs in / outputs out around the
// graph launch. Memcpys are ~50 us total (a few MB on a 400 GB/s bus); the
// graph launch is single-digit microseconds.

template <typename T>
struct ElemScalarType;
template <> struct ElemScalarType<float>              { static constexpr c10::ScalarType value = at::kFloat;    };
template <> struct ElemScalarType<cutlass::half_t>    { static constexpr c10::ScalarType value = at::kHalf;     };
template <> struct ElemScalarType<cutlass::bfloat16_t>{ static constexpr c10::ScalarType value = at::kBFloat16; };

template <typename scalar_t>
struct NsBwdGraphState {
    int m = -1, BH = -1, niter = -1, D = -1;
    float captured_kappa = -1.0f;  // kappa_star baked into the captured graph
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    // Persistent workspaces with stable addresses across calls.
    at::Tensor K2_buf;       // (BH, m, m) FP32           — copied in
    at::Tensor ns_iter_buf;  // (BH, niter+1, m, m) FP32 — copied in
    at::Tensor q_tilde_buf;  // (BH, m, D) elem_type     — copied in
    at::Tensor k_tilde_buf;  // (BH, m, D) elem_type     — copied in
    at::Tensor dZ_buf;       // (BH, m, m) FP32          — rolling state
    at::Tensor dK2_buf;      // (BH, m, m) FP32          — accumulator (zeroed each call)
    at::Tensor dQ_tilde_buf; // (BH, m, D) FP32          — accumulator (copied in/out)
    at::Tensor dK_tilde_buf; // (BH, m, D) FP32          — accumulator (copied in/out)
    at::Tensor scratch_buf;  // (11, BH, m, m) FP32       — NS step intermediates
    at::Tensor dS2_buf;      // (BH, m, m) elem_type      — ns_bwd_final pre-kernel output

    bool matches(int m_, int BH_, int n_, int D_) const {
        return m == m_ && BH == BH_ && niter == n_ && D == D_;
    }

    void invalidate_graph() {
        if (exec)  { cudaGraphExecDestroy(exec); exec  = nullptr; }
        if (graph) { cudaGraphDestroy(graph);    graph = nullptr; }
    }

    // Free workspaces and reset cached graph. Safe to call any time outside
    // of an active graph capture. Restores the state to "uninitialized";
    // the next call to allocate() will re-allocate fresh.
    void reset() {
        invalidate_graph();
        K2_buf       = at::Tensor();
        ns_iter_buf  = at::Tensor();
        q_tilde_buf  = at::Tensor();
        k_tilde_buf  = at::Tensor();
        dZ_buf       = at::Tensor();
        dK2_buf      = at::Tensor();
        dQ_tilde_buf = at::Tensor();
        dK_tilde_buf = at::Tensor();
        scratch_buf  = at::Tensor();
        dS2_buf      = at::Tensor();
        m = -1; BH = -1; niter = -1; D = -1;
    }

    void allocate(int m_, int BH_, int n_, int D_) {
        m = m_; BH = BH_; niter = n_; D = D_;
        invalidate_graph();
        auto opts_f32  = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
        auto opts_elem = at::TensorOptions()
            .dtype(ElemScalarType<scalar_t>::value).device(at::kCUDA);
        K2_buf       = at::empty({BH, m, m},          opts_f32);
        ns_iter_buf  = at::empty({BH, n_ + 1, m, m},  opts_f32);
        q_tilde_buf  = at::empty({BH, m, D},          opts_elem);
        k_tilde_buf  = at::empty({BH, m, D},          opts_elem);
        dZ_buf       = at::empty({BH, m, m},          opts_f32);
        dK2_buf      = at::empty({BH, m, m},          opts_f32);
        dQ_tilde_buf = at::empty({BH, m, D},          opts_f32);
        dK_tilde_buf = at::empty({BH, m, D},          opts_f32);
        scratch_buf  = at::empty({13, BH, m, m},      opts_f32);
        dS2_buf      = at::empty({BH, m, m},          opts_elem);
    }

    ~NsBwdGraphState() {
        // Best-effort cleanup on thread exit. CUDA context may already be
        // gone at process exit; ignore errors silently. Note: a thrown
        // exception here would terminate during stack unwinding, so the
        // wrapping if-checks deliberately swallow any error.
        if (exec) {
            (void)cudaGraphExecDestroy(exec);
            exec = nullptr;
        }
        if (graph) {
            (void)cudaGraphDestroy(graph);
            graph = nullptr;
        }
    }
};

template <typename scalar_t>
static NsBwdGraphState<scalar_t>& get_ns_bwd_graph_state() {
    static thread_local NsBwdGraphState<scalar_t> s;
    return s;
}

// Record the entire NS backward (per-iter cuBLAS step loop + final softmax
// bwd) onto whatever stream is active. Called inside stream capture to build
// the graph; pointers are the persistent workspace pointers in `s`.
template <typename scalar_t>
static void record_ns_bwd_on_workspace(
    NsBwdGraphState<scalar_t>& s, cudaStream_t stream, float kappa_star
) {
    const int m = s.m, BH = s.BH, niter = s.niter, D = s.D;
    const int mm = m * m;
    const int Z_bh_stride = (niter + 1) * mm;
    const bool ridge = (kappa_star > 0.0f);

    // Inside a function template, member templates on dependent objects
    // need the explicit `template` disambiguator. Each .template data_ptr<T>()
    // here is just .data_ptr<T>() with the disambiguation.
    float*    K2       = s.K2_buf.template data_ptr<float>();
    float*    ns_iter  = s.ns_iter_buf.template data_ptr<float>();
    float*    dZ       = s.dZ_buf.template data_ptr<float>();
    float*    dK2      = s.dK2_buf.template data_ptr<float>();
    float*    scratch  = s.scratch_buf.template data_ptr<float>();
    // ATen's data_ptr<T>() has no specialization for cutlass scalar types;
    // go through the void-pointer form and reinterpret.
    scalar_t* q_tilde  = reinterpret_cast<scalar_t*>(s.q_tilde_buf.data_ptr());
    scalar_t* k_tilde  = reinterpret_cast<scalar_t*>(s.k_tilde_buf.data_ptr());
    float*    dQ_tilde = s.dQ_tilde_buf.template data_ptr<float>();
    float*    dK_tilde = s.dK_tilde_buf.template data_ptr<float>();

    float* ws_M       = scratch + 0  * BH * mm;
    float* ws_M2      = scratch + 1  * BH * mm;
    float* ws_V       = scratch + 2  * BH * mm;
    float* ws_T       = scratch + 3  * BH * mm;
    float* ws_dT      = scratch + 4  * BH * mm;
    float* ws_dV      = scratch + 5  * BH * mm;
    float* ws_dM_T    = scratch + 6  * BH * mm;
    float* ws_dV_MT   = scratch + 7  * BH * mm;
    float* ws_dU      = scratch + 8  * BH * mm;
    float* ws_dM      = scratch + 9  * BH * mm;
    float* ws_dZ_out  = scratch + 10 * BH * mm;
    float* ws_Mmat    = scratch + 11 * BH * mm;   // reconstructed M = K2^T K2 + lam*I (ridge)
    float* ws_dK2a    = scratch + 12 * BH * mm;   // dK2 accumulator: dK2_a then dK2_final (ridge)

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, stream);

    // --- Tikhonov pre-loop wrap: NS runs on M, not K2 (see header note) ---
    // dZ currently holds G = dL/dK2^+ where K2^+ = Z_J K2^T, Z_J = M^-1.
    //   dK2_a = G^T @ Z_J   (gradient through the trailing K2^T)
    //   dZ    = G  @ K2     (gradient into Z_J; becomes the NS-loop seed dZ_J)
    //   M     = K2^T K2 + lambda*I  (the matrix the saved iterates invert)
    const float* Z_J = ns_iter + niter * mm;
    if (ridge) {
        rm_sgemm_strided_batched(handle, 'T', 'N', m, m, m,
            1.0f, dZ, mm, Z_J, Z_bh_stride,
            0.0f, ws_dK2a, mm, BH);                 // ws_dK2a = G^T Z_J
        rm_sgemm_strided_batched(handle, 'N', 'N', m, m, m,
            1.0f, dZ, mm, K2, mm,
            0.0f, ws_M, mm, BH);                    // ws_M = G K2  (temp)
        FN_CUDA_CHECK(cudaMemcpyAsync(dZ, ws_M, (size_t)BH * mm * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));     // dZ = dZ_J
        rm_sgemm_strided_batched(handle, 'T', 'N', m, m, m,
            1.0f, K2, mm, K2, mm,
            0.0f, ws_Mmat, mm, BH);                 // ws_Mmat = K2^T K2
        size_t smem_rd = (2 * m + 32) * sizeof(float);
        add_ridge_diag_kernel<<<dim3(BH), dim3(256), smem_rd, stream>>>(
            ws_Mmat, K2, m, kappa_star);            // ws_Mmat += lambda*I
    }

    // The matrix being inverted by NS: M (ridge) or K2 (no ridge). The loop
    // accumulates dK2 (= dM under ridge) and rolls dZ from dZ_J down to dZ_0.
    const float* mat = ridge ? ws_Mmat : K2;

    for (int j = niter - 1; j >= 0; j--) {
        const float* Z_j = ns_iter + j * mm;

        rm_sgemm_strided_batched(handle, 'N', 'N', m, m, m,
            1.0f, mat, mm, Z_j, Z_bh_stride,
            0.0f, ws_M, mm, BH);

        rm_sgemm_strided_batched(handle, 'N', 'N', m, m, m,
            1.0f, ws_M, mm, ws_M, mm,
            0.0f, ws_M2, mm, BH);

        launch_affine_with_identity(ws_V, ws_M, ws_M2,
            15.0f, -7.0f, 1.0f, BH, m, stream);

        rm_sgemm_strided_batched(handle, 'N', 'N', m, m, m,
            1.0f, ws_M, mm, ws_V, mm,
            0.0f, ws_T, mm, BH);
        launch_affine_with_identity(ws_T, ws_T, nullptr,
            13.0f, -1.0f, 0.0f, BH, m, stream);

        rm_sgemm_strided_batched(handle, 'N', 'T', m, m, m,
            0.25f, dZ, mm, ws_T, mm,
            0.0f, ws_dZ_out, mm, BH);

        rm_sgemm_strided_batched(handle, 'T', 'N', m, m, m,
            0.25f, Z_j, Z_bh_stride, dZ, mm,
            0.0f, ws_dT, mm, BH);

        rm_sgemm_strided_batched(handle, 'N', 'T', m, m, m,
            -1.0f, ws_dT, mm, ws_V, mm,
            0.0f, ws_dM_T, mm, BH);

        rm_sgemm_strided_batched(handle, 'T', 'N', m, m, m,
            -1.0f, ws_M, mm, ws_dT, mm,
            0.0f, ws_dV, mm, BH);

        rm_sgemm_strided_batched(handle, 'N', 'T', m, m, m,
            1.0f, ws_dV, mm, ws_M, mm,
            0.0f, ws_dV_MT, mm, BH);

        rm_sgemm_strided_batched(handle, 'T', 'N', m, m, m,
            -1.0f, ws_M, mm, ws_dV, mm,
            0.0f, ws_dU, mm, BH);

        launch_affine_with_identity(ws_dM, ws_dM_T, nullptr,
            0.0f, 1.0f, 0.0f, BH, m, stream);
        launch_add_scaled_inplace(ws_dM, ws_dV,    -7.0f, BH, m, stream);
        launch_add_scaled_inplace(ws_dM, ws_dV_MT,  1.0f, BH, m, stream);
        launch_add_scaled_inplace(ws_dM, ws_dU,    -1.0f, BH, m, stream);

        rm_sgemm_strided_batched(handle, 'N', 'T', m, m, m,
            1.0f, ws_dM, mm, Z_j, Z_bh_stride,
            1.0f, dK2, mm, BH);

        rm_sgemm_strided_batched(handle, 'T', 'N', m, m, m,
            1.0f, mat, mm, ws_dM, mm,
            0.0f, dZ, mm, BH);

        launch_add_scaled_inplace(dZ, ws_dZ_out, 1.0f, BH, m, stream);
    }

    scalar_t* dS2 = reinterpret_cast<scalar_t*>(s.dS2_buf.data_ptr());
    dim3 grid(BH), block(256);

    if (ridge) {
        // dK2 holds the NS-loop dM. Add the Z_0-init grad (on M) -> full dM.
        size_t smem_z0 = (3 * mm + 8) * sizeof(float);
        if (smem_z0 > 48 * 1024) {
            FN_CHECK(smem_z0 <= get_max_smem_per_block(), "ns_bwd_z0grad: insufficient SMEM");
            FN_CUDA_CHECK(cudaFuncSetAttribute(ns_bwd_z0grad_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_z0)));
        }
        ns_bwd_z0grad_kernel<<<grid, block, smem_z0, stream>>>(ws_Mmat, dZ, dK2, m);

        // dK2_final = dK2_a + K2(dM + dM^T)  (accumulate into ws_dK2a).
        rm_sgemm_strided_batched(handle, 'N', 'N', m, m, m,
            1.0f, K2, mm, dK2, mm, 1.0f, ws_dK2a, mm, BH);   // += K2 dM
        rm_sgemm_strided_batched(handle, 'N', 'T', m, m, m,
            1.0f, K2, mm, dK2, mm, 1.0f, ws_dK2a, mm, BH);   // += K2 dM^T

        // Softmax-Jacobian on the un-ridged K2 -> dS2.
        size_t smem_sj = (3 * mm) * sizeof(float);
        if (smem_sj > 48 * 1024) {
            FN_CHECK(smem_sj <= get_max_smem_per_block(), "ns_bwd_softmax_jac: insufficient SMEM");
            FN_CUDA_CHECK(cudaFuncSetAttribute(ns_bwd_softmax_jac_kernel<scalar_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_sj)));
        }
        ns_bwd_softmax_jac_kernel<scalar_t><<<grid, block, smem_sj, stream>>>(
            K2, ws_dK2a, dS2, m);
    } else {
        // No-ridge: NS inverted K2 directly; the fused final pre-kernel does the
        // Z_0-init grad and softmax-Jacobian together (ridge_lambda = 0).
        size_t smem = (3 * mm + 8) * sizeof(float);
        if (smem > 48 * 1024) {
            FN_CHECK(smem <= get_max_smem_per_block(), "ns_bwd_final_pre: insufficient SMEM");
            FN_CUDA_CHECK(cudaFuncSetAttribute(ns_bwd_final_pre_kernel<scalar_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        }
        ns_bwd_final_pre_kernel<scalar_t><<<grid, block, smem, stream>>>(
            K2, dZ, dK2, dS2, m, 0.0f);
    }

    // Trailing m x m x D matmuls (TC for FP16/BF16), common to both paths.
    rm_gemm_ex_strided_batched<scalar_t>(handle, 'N', 'N', m, D, m,
        1.0f, dS2, mm, k_tilde, (long long)m * D,
        1.0f, dQ_tilde, (long long)m * D, BH);     // dQ_tilde += dS2 @ k_tilde
    rm_gemm_ex_strided_batched<scalar_t>(handle, 'T', 'N', m, D, m,
        1.0f, dS2, mm, q_tilde, (long long)m * D,
        1.0f, dK_tilde, (long long)m * D, BH);     // dK_tilde += dS2^T @ q_tilde
}

// =========================================================================
// Production launch wrapper (called from run_nystrom_bwd_impl).
// =========================================================================

template <typename scalar_t>
void launch_kernel2_inv_bwd(
    const scalar_t* q_tilde, const scalar_t* k_tilde,
    const float* dK2_inv_in,
    const float* ns_iterates,
    const float* K2_softmax,
    float* dQ_tilde, float* dK_tilde,
    int BH, int D, int m, int newton_iter, cudaStream_t stream,
    float kappa_star
) {
    const int mm = m * m;
    const int mD = m * D;

    auto& s = get_ns_bwd_graph_state<scalar_t>();
    if (!s.matches(m, BH, newton_iter, D)) {
        s.allocate(m, BH, newton_iter, D);
    }

    // Memcpy inputs from caller pointers to persistent workspaces.
    FN_CUDA_CHECK(cudaMemcpyAsync(s.K2_buf.template data_ptr<float>(),
        K2_softmax, BH * mm * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    FN_CUDA_CHECK(cudaMemcpyAsync(s.ns_iter_buf.template data_ptr<float>(),
        ns_iterates, BH * (newton_iter + 1) * mm * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    FN_CUDA_CHECK(cudaMemcpyAsync(s.dZ_buf.template data_ptr<float>(),
        dK2_inv_in, BH * mm * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    FN_CUDA_CHECK(cudaMemcpyAsync(s.q_tilde_buf.data_ptr(),
        q_tilde, BH * mD * sizeof(scalar_t),
        cudaMemcpyDeviceToDevice, stream));
    FN_CUDA_CHECK(cudaMemcpyAsync(s.k_tilde_buf.data_ptr(),
        k_tilde, BH * mD * sizeof(scalar_t),
        cudaMemcpyDeviceToDevice, stream));
    FN_CUDA_CHECK(cudaMemcpyAsync(s.dQ_tilde_buf.template data_ptr<float>(),
        dQ_tilde, BH * mD * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    FN_CUDA_CHECK(cudaMemcpyAsync(s.dK_tilde_buf.template data_ptr<float>(),
        dK_tilde, BH * mD * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    FN_CUDA_CHECK(cudaMemsetAsync(s.dK2_buf.template data_ptr<float>(),
        0, BH * mm * sizeof(float), stream));

    // Capture graph on first call for this shape, replay on subsequent calls.
    // PyTorch's current stream is often the legacy default (stream 0), which
    // is not capturable. We do the capture on a side stream from the pool
    // (no kernels actually run during capture; the graph just records the
    // launch sequence) and then replay on the caller's stream, which both
    // executes the work and preserves the caller's stream ordering.
    // kappa_star is baked into the captured graph (the ridge path's lambda is
    // computed per-bh inside add_ridge_diag/z0grad from it). If it changed
    // since capture, invalidate so we recapture with the new value.
    if (s.exec != nullptr && s.captured_kappa != kappa_star) {
        cudaGraphExecDestroy(s.exec); s.exec = nullptr;
        cudaGraphDestroy(s.graph);    s.graph = nullptr;
    }
    if (s.exec == nullptr) {
        // Pre-warm the cuBLAS handle outside capture. PyTorch creates the
        // handle lazily on first access; cublasCreate is not allowed during
        // stream capture, so we trigger creation here.
        (void)at::cuda::getCurrentCUDABlasHandle();

        auto side = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
        cudaStream_t side_stream = side.stream();
        FN_CUDA_CHECK(cudaStreamBeginCapture(side_stream,
            cudaStreamCaptureModeThreadLocal));
        record_ns_bwd_on_workspace<scalar_t>(s, side_stream, kappa_star);
        FN_CUDA_CHECK(cudaStreamEndCapture(side_stream, &s.graph));
        FN_CUDA_CHECK(cudaGraphInstantiate(&s.exec, s.graph, nullptr, nullptr, 0));
        s.captured_kappa = kappa_star;
    }
    FN_CUDA_CHECK(cudaGraphLaunch(s.exec, stream));

    // Memcpy outputs from workspaces to caller pointers.
    FN_CUDA_CHECK(cudaMemcpyAsync(dQ_tilde,
        s.dQ_tilde_buf.template data_ptr<float>(), BH * mD * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    FN_CUDA_CHECK(cudaMemcpyAsync(dK_tilde,
        s.dK_tilde_buf.template data_ptr<float>(), BH * mD * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    FN_CUDA_KERNEL_CHECK();
}

// Explicit template instantiations. Required so other TUs (e.g. the
// occupancy probe in flash_nystrom.cu) can take the kernel function
// pointer without seeing the kernel definition.
template void launch_kernel2_inv_bwd<float>(
    const float*, const float*,
    const float*, const float*, const float*,
    float*, float*,
    int, int, int, int, cudaStream_t, float);
template void launch_kernel2_inv_bwd<cutlass::half_t>(
    const cutlass::half_t*, const cutlass::half_t*,
    const float*, const float*, const float*,
    float*, float*,
    int, int, int, int, cudaStream_t, float);
template void launch_kernel2_inv_bwd<cutlass::bfloat16_t>(
    const cutlass::bfloat16_t*, const cutlass::bfloat16_t*,
    const float*, const float*, const float*,
    float*, float*,
    int, int, int, int, cudaStream_t, float);

template __global__ void ns_bwd_final_kernel<float>(
    const float* __restrict__, const float* __restrict__,
    const float* __restrict__, const float* __restrict__,
    float* __restrict__, float* __restrict__, float* __restrict__,
    int, int);
template __global__ void ns_bwd_final_kernel<cutlass::half_t>(
    const cutlass::half_t* __restrict__, const cutlass::half_t* __restrict__,
    const float* __restrict__, const float* __restrict__,
    float* __restrict__, float* __restrict__, float* __restrict__,
    int, int);
template __global__ void ns_bwd_final_kernel<cutlass::bfloat16_t>(
    const cutlass::bfloat16_t* __restrict__, const cutlass::bfloat16_t* __restrict__,
    const float* __restrict__, const float* __restrict__,
    float* __restrict__, float* __restrict__, float* __restrict__,
    int, int);

// =========================================================================
// Test-only launchers (debug pybind hooks).
// =========================================================================

void launch_ns_bwd_step_test(
    const float* K2_in, const float* Z_j_in, const float* dZ_in,
    float* dZ_out, float* dK2_acc,
    int BH, int m, cudaStream_t stream
) {
    const int mm = m * m;
    // dZ slot must be initialized to dZ_in (kernel reads dZ_inout).
    FN_CUDA_CHECK(cudaMemcpyAsync(dZ_out, dZ_in, BH * mm * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));
    size_t smem = 6 * mm * sizeof(float);
    dim3 grid(BH);
    dim3 block(256);
    if (smem > 48 * 1024) {
        FN_CHECK(smem <= get_max_smem_per_block(),
                 "ns_bwd_step_test: insufficient SMEM");
        FN_CUDA_CHECK(cudaFuncSetAttribute(ns_bwd_step_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    }
    ns_bwd_step_kernel<<<grid, block, smem, stream>>>(
        K2_in, Z_j_in, /*Z_bh_stride=*/mm, dZ_out, dK2_acc, m);
    FN_CUDA_KERNEL_CHECK();
}

void launch_ns_bwd_final_test(
    const float* q_tilde, const float* k_tilde,
    const float* K2_in, const float* dZ0_in,
    float* dK2_inout, float* dQ_tilde_out, float* dK_tilde_out,
    int BH, int D, int m, cudaStream_t stream
) {
    const int mm = m * m;
    size_t smem = (3 * mm + 8) * sizeof(float);
    dim3 grid(BH);
    dim3 block(256);
    if (smem > 48 * 1024) {
        FN_CHECK(smem <= get_max_smem_per_block(),
                 "ns_bwd_final_test: insufficient SMEM");
        FN_CUDA_CHECK(cudaFuncSetAttribute(ns_bwd_final_kernel<float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    }
    ns_bwd_final_kernel<float><<<grid, block, smem, stream>>>(
        q_tilde, k_tilde, K2_in, dZ0_in, dK2_inout, dQ_tilde_out, dK_tilde_out, D, m);
    FN_CUDA_KERNEL_CHECK();
}

// =========================================================================
// Public cache-reset hook. Frees the thread-local NsBwdGraphState
// workspaces and destroys the cached cudaGraphExec across all dtypes.
// Called from Python via _C.reset_caches() when the user wants to reclaim
// GPU memory after a shape change or before measuring memory usage.
// Safe to call concurrently with nystrom_bwd on a different thread (the
// state is thread-local). Not safe to call from a CUDA-graph capture
// context, but no user code should be doing that.
// =========================================================================
void reset_ns_bwd_caches() {
    get_ns_bwd_graph_state<float>().reset();
    get_ns_bwd_graph_state<cutlass::half_t>().reset();
    get_ns_bwd_graph_state<cutlass::bfloat16_t>().reset();
}

} // namespace flash_nystrom
