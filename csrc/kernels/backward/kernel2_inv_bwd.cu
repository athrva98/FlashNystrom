/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
// Single-definition .cu for the unrolled Newton-Schulz backward.
// The header (kernel2_inv_bwd.cuh) only declares; everything in this TU.

#include "kernels/backward/kernel2_inv_bwd.cuh"
#include "utils.h"

#include <cfloat>
#include <cutlass/numeric_types.h>

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
// Production launch wrapper (called from run_nystrom_bwd_impl).
// =========================================================================

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
    int BH, int D, int m, int newton_iter, cudaStream_t stream
) {
    (void)k2_inv;
    (void)lse2;
    const int mm = m * m;
    const int Z_bh_stride = (newton_iter + 1) * mm;

    // dZ_workspace = dK2_inv_in (gradient w.r.t. Z_N)
    FN_CUDA_CHECK(cudaMemcpyAsync(dZ_workspace, dK2_inv_in, BH * mm * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));
    // dK2_workspace = 0
    FN_CUDA_CHECK(cudaMemsetAsync(dK2_workspace, 0, BH * mm * sizeof(float), stream));

    // Per-iteration kernel
    {
        size_t smem = 6 * mm * sizeof(float);
        dim3 grid(BH);
        dim3 block(256);
        if (smem > 48 * 1024) {
            FN_CHECK(smem <= get_max_smem_per_block(),
                     "ns_bwd_step: insufficient SMEM");
            FN_CUDA_CHECK(cudaFuncSetAttribute(ns_bwd_step_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        }
        for (int j = newton_iter - 1; j >= 0; j--) {
            const float* Z_j_base = ns_iterates + j * mm;  // Z_j of bh=0
            ns_bwd_step_kernel<<<grid, block, smem, stream>>>(
                K2_softmax, Z_j_base, Z_bh_stride,
                dZ_workspace, dK2_workspace, m);
            FN_CUDA_KERNEL_CHECK();
        }
    }

    // Final-step kernel
    {
        size_t smem = (3 * mm + 8) * sizeof(float);
        dim3 grid(BH);
        dim3 block(256);
        if (smem > 48 * 1024) {
            FN_CHECK(smem <= get_max_smem_per_block(),
                     "ns_bwd_final: insufficient SMEM");
            FN_CUDA_CHECK(cudaFuncSetAttribute(ns_bwd_final_kernel<scalar_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        }
        ns_bwd_final_kernel<scalar_t><<<grid, block, smem, stream>>>(
            q_tilde, k_tilde, K2_softmax, dZ_workspace, dK2_workspace,
            dQ_tilde, dK_tilde, D, m);
        FN_CUDA_KERNEL_CHECK();
    }
}

// Explicit template instantiations.
template void launch_kernel2_inv_bwd<float>(
    const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, float*, float*, float*, float*,
    int, int, int, int, cudaStream_t);
template void launch_kernel2_inv_bwd<cutlass::half_t>(
    const cutlass::half_t*, const cutlass::half_t*, const float*, const float*, const float*,
    const float*, const float*, float*, float*, float*, float*,
    int, int, int, int, cudaStream_t);
template void launch_kernel2_inv_bwd<cutlass::bfloat16_t>(
    const cutlass::bfloat16_t*, const cutlass::bfloat16_t*, const float*, const float*, const float*,
    const float*, const float*, float*, float*, float*, float*,
    int, int, int, int, cudaStream_t);

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

} // namespace flash_nystrom
