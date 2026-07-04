/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once

#include "utils.h"

namespace flash_nystrom {

// kernel2_inv: K2 = softmax(Q_tilde @ K_tilde^T), then Newton-Schulz pseudoinverse.
//
// Forward Newton-Schulz iteration (third-order Higham polynomial):
//
//     M_j = K2 @ Z_j
//     Z_{j+1} = (1/4) * Z_j * (13*I - M_j*(15*I - M_j*(7*I - M_j)))
//             = (1/4) * Z_j * (13*I - 15*M_j + 7*M_j^2 - M_j^3)
//
// Initialization: Z_0 = K2^T / (||K2||_1 * ||K2||_inf).
//
// To enable the unrolled backward, ALL iterates Z_0, Z_1, ..., Z_J are
// written to ns_iterates_out (BH, J+1, m, m), where J = newton_iter. The
// forward output K2_inv = Z_J is also written to kernel2_inv_out separately
// for kernel3.
//
// Iteration count is `newton_iter` (default 6, capped at 20 by binding).
// For row-stochastic K2 with spread eigenvalues, 6 iters has residual O(1).
// The backward consistency does NOT require convergence — autograd-through-NS
// gives correct gradients of the actual approximation regardless.

// One SMEM m x m matmul phase of the NS loop: dst = scale * A @ f(B), where
// f is an optional on-the-fly diagonal transform (diag_c > 0 selects
// B'[k,j] = (k==j ? diag_c : 0) - B[k,j], fusing the "cI - X" elementwise
// phases into the consuming matmul and saving their barriers). Each thread
// owns kCells consecutive cells of one row: A[row,k] is loaded once per k
// and broadcast across the cells, and the kCells independent accumulators
// give the 64-long dependent FMA chains 4-way ILP. The per-cell summation
// order over k is unchanged from the naive form, so each output cell is
// bitwise identical to the one-cell-per-thread version.
__device__ __forceinline__ void ns_matmul_phase(
    float* __restrict__ dst, const float* __restrict__ A,
    const float* __restrict__ B, int m, int tid, int nthreads,
    float diag_c, float scale
) {
    constexpr int kCells = 4;
    const int mm = m * m;
    for (int base = tid * kCells; base < mm; base += nthreads * kCells) {
        const int row = base / m;
        const int col0 = base - row * m;
        if (col0 + kCells <= m) {   // always taken when kCells divides m
            float acc[kCells] = {0.f, 0.f, 0.f, 0.f};
            const float* arow = A + row * m;
            for (int k = 0; k < m; k++) {
                const float a = arow[k];
                // one 16-byte load for the 4 consecutive B cells (base is a
                // multiple of kCells and m % kCells == 0, so brow is 16B
                // aligned); the LSU instruction count is what binds this
                // kernel, not bandwidth.
                const float4 b4 =
                    *reinterpret_cast<const float4*>(B + k * m + col0);
                float bv[kCells] = {b4.x, b4.y, b4.z, b4.w};
                #pragma unroll
                for (int c = 0; c < kCells; c++) {
                    float b = bv[c];
                    if (diag_c > 0.0f) b = ((k == col0 + c) ? diag_c : 0.0f) - b;
                    acc[c] += a * b;
                }
            }
            #pragma unroll
            for (int c = 0; c < kCells; c++) dst[base + c] = scale * acc[c];
        } else {                    // row-straddling tail for m % kCells != 0
            for (int cell = base; cell < base + kCells && cell < mm; cell++) {
                const int r = cell / m, col = cell - r * m;
                float acc = 0.0f;
                for (int k = 0; k < m; k++) {
                    float b = B[k * m + col];
                    if (diag_c > 0.0f) b = ((k == col) ? diag_c : 0.0f) - b;
                    acc += A[r * m + k] * b;
                }
                dst[cell] = scale * acc;
            }
        }
    }
}

template <typename scalar_t, bool kSetupOnly = false>
__global__ void __launch_bounds__(1024, 1) kernel2_inv_kernel(
    const scalar_t* __restrict__ q_tilde,    // (B*H, m, D)
    const scalar_t* __restrict__ k_tilde,    // (B*H, m, D)
    float* __restrict__ kernel2_inv_out,      // (B*H, m, m) FP32 — final K2_inv = Z_J
    float* __restrict__ softmax_lse_out,      // (B*H, m)
    float* __restrict__ ns_iterates_out,      // (B*H, newton_iter+1, m, m) FP32 — Z_0..Z_J
    float* __restrict__ k2_softmax_out,       // (B*H, m, m) FP32 — the un-ridged softmax K2 (for backward)
    int D, int m, int newton_iter,
    float kappa_star                          // Tikhonov target condition number (0 = no ridge)
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int mm = m * m;

    extern __shared__ float smem[];
    float* K2      = smem;
    float* Z       = smem + mm;
    float* Zold    = smem + 2 * mm;
    float* T1      = smem + 3 * mm;
    float* T2      = smem + 4 * mm;
    float* scratch = smem + 5 * mm;
    // Zold/T2 are only touched by the in-kernel NS loop, which is compiled out in
    // setup-only mode; reference them so -Werror does not flag them as unused.
    (void)Zold; (void)T2;

    const scalar_t* qt = q_tilde + bh * m * D;
    const scalar_t* kt = k_tilde + bh * m * D;

    // Step 1: K2_logits = Q_tilde @ K_tilde^T (FP32 accumulation).
    // Row-blocked (4 consecutive cells share the Q row) with 16-byte input
    // loads: the element-wise form issued ~1K scalar 2-byte loads per
    // thread and saturated the LSU pipe (this kernel is load-instruction
    // bound, 98.8% L1 hit). Per-cell accumulation order over d unchanged.
    {
        constexpr int kEpt = 16 / (int)sizeof(scalar_t);  // elems per 16B
        for (int base = tid * 4; base < mm; base += nthreads * 4) {
            const int i = base / m;
            const int col0 = base - i * m;
            if (col0 + 4 <= m && (D % kEpt) == 0) {
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                for (int d0 = 0; d0 < D; d0 += kEpt) {
                    scalar_t qv[kEpt], kv[4][kEpt];
                    *reinterpret_cast<uint4*>(qv) =
                        *reinterpret_cast<const uint4*>(qt + i * D + d0);
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        *reinterpret_cast<uint4*>(kv[c]) =
                            *reinterpret_cast<const uint4*>(
                                kt + (col0 + c) * D + d0);
                    }
                    #pragma unroll
                    for (int e = 0; e < kEpt; e++) {
                        const float qf = to_float(qv[e]);
                        #pragma unroll
                        for (int c = 0; c < 4; c++) {
                            acc[c] += qf * to_float(kv[c][e]);
                        }
                    }
                }
                #pragma unroll
                for (int c = 0; c < 4; c++) K2[base + c] = acc[c];
            } else {
                for (int cell = base; cell < base + 4 && cell < mm; cell++) {
                    const int r = cell / m, j = cell - r * m;
                    float acc = 0.0f;
                    for (int d = 0; d < D; d++) {
                        acc += to_float(qt[r * D + d]) * to_float(kt[j * D + d]);
                    }
                    K2[cell] = acc;
                }
            }
        }
    }
    __syncthreads();

    // Step 2: Row-wise softmax on K2, save LSE. One WARP per row: the rows
    // are independent, so the reductions run as warp shuffles with no block
    // barriers. The previous block-per-row form serialized m rows behind
    // ~500 block-wide barriers (2 block_reduces x ~4 __syncthreads each per
    // row) and was the dominant flat cost of the whole kernel on a B200.
    float* lse_out = softmax_lse_out + bh * m;
    {
        const int lane = tid % 32;
        const int warp = tid / 32;
        const int nwarps = nthreads / 32;
        for (int row = warp; row < m; row += nwarps) {
            float local_max = -FLT_MAX;
            for (int j = lane; j < m; j += 32) {
                local_max = fmaxf(local_max, K2[row * m + j]);
            }
            float row_max = warp_reduce_max(local_max);

            float local_sum = 0.0f;
            for (int j = lane; j < m; j += 32) {
                float val = expf(K2[row * m + j] - row_max);
                K2[row * m + j] = val;
                local_sum += val;
            }
            float row_sum = warp_reduce_sum(local_sum);

            float inv_sum = 1.0f / (row_sum + 1e-12f);
            for (int j = lane; j < m; j += 32) {
                K2[row * m + j] *= inv_sum;
            }
            if (lane == 0) {
                lse_out[row] = row_max + logf(row_sum + 1e-12f);
            }
        }
    }
    __syncthreads();

    // Step 2.5: save the UN-ridged softmax K2 to GMEM (the backward needs it,
    // and the Tikhonov path reloads it for the final M^-1 K2^T multiply).
    if (k2_softmax_out != nullptr) {
        float* k2sm_bh = k2_softmax_out + bh * mm;
        for (int idx = tid; idx < mm; idx += nthreads) k2sm_bh[idx] = K2[idx];
        __syncthreads();
    }

    // Tikhonov ridge (non-normality-proof). K2 is non-normal in general, so
    // K2 + lambda*I does NOT shift its singular values / bound cond(K2). Instead
    // invert the SYMMETRIC PSD normal-equations matrix M = K2^T K2 + lambda*I and
    // return K2^+ = M^-1 K2^T. With lambda = (||K2||_1 ||K2||_inf)/kappa_star
    // (>= sigma_max^2/kappa_star) we get cond(M) <= kappa_star regardless of
    // non-normality. We overwrite the K2 SMEM slot with M so the unchanged NS
    // loop below inverts M (its Z_0 init then uses M's own norms); the un-ridged
    // K2 is reloaded from GMEM for the final multiply (Step 5).
    if (kappa_star > 0.0f) {
        float lcs = -FLT_MAX;
        for (int j = tid; j < m; j += nthreads) {
            float cs = 0.0f; for (int i = 0; i < m; i++) cs += K2[i * m + j];
            lcs = fmaxf(lcs, cs);
        }
        float n1 = block_reduce_max(lcs, scratch);          // ||K2||_1 (max col sum)
        float lrs = -FLT_MAX;
        for (int i = tid; i < m; i += nthreads) {
            float rs = 0.0f; for (int j = 0; j < m; j++) rs += K2[i * m + j];
            lrs = fmaxf(lrs, rs);
        }
        float ninf = block_reduce_max(lrs, scratch);        // ||K2||_inf (max row sum)
        float lam = (n1 * ninf) / kappa_star;
        // M = K2^T K2 + lambda*I  (build in Z, then move into the K2 slot)
        for (int idx = tid; idx < mm; idx += nthreads) {
            int i = idx / m, j = idx % m;
            float acc = 0.0f;
            for (int k = 0; k < m; k++) acc += K2[k * m + i] * K2[k * m + j];
            Z[idx] = acc + ((i == j) ? lam : 0.0f);
        }
        __syncthreads();
        for (int idx = tid; idx < mm; idx += nthreads) K2[idx] = Z[idx];
        __syncthreads();
    }

    // Step 3: Z_0 = K2^T / (||K2||_1 * ||K2||_inf).
    // Both norms are computed and used as a piecewise-constant scalar — the
    // gradient through their max() ops is handled in the backward final kernel.
    for (int j = tid; j < m; j += nthreads) {
        float col_sum = 0.0f;
        for (int i = 0; i < m; i++) col_sum += K2[i * m + j];   // K2 >= 0 (softmax)
        T1[j] = col_sum;
    }
    __syncthreads();

    float local_col_max = -FLT_MAX;
    for (int j = tid; j < m; j += nthreads) {
        local_col_max = fmaxf(local_col_max, T1[j]);
    }
    float norm1 = block_reduce_max(local_col_max, scratch);

    // ||K2||_inf = max row sum. For a row-stochastic K2 this is ~1 but we
    // compute it explicitly so the scaling matches the reference exactly.
    float local_row_max = -FLT_MAX;
    for (int i = tid; i < m; i += nthreads) {
        float row_sum = 0.0f;
        for (int j = 0; j < m; j++) row_sum += K2[i * m + j];
        local_row_max = fmaxf(local_row_max, row_sum);
    }
    float norm_inf = block_reduce_max(local_row_max, scratch);

    float inv_c = 1.0f / fmaxf(norm1 * norm_inf, 1e-12f);

    for (int idx = tid; idx < mm; idx += nthreads) {
        int row = idx / m;
        int col = idx % m;
        Z[idx] = K2[col * m + row] * inv_c;  // transpose
    }
    __syncthreads();

    // Save Z_0 as the first iterate (index 0 in ns_iterates_out)
    if (ns_iterates_out != nullptr) {
        float* iter_out = ns_iterates_out + bh * (newton_iter + 1) * mm;
        for (int idx = tid; idx < mm; idx += nthreads) iter_out[idx] = Z[idx];
        __syncthreads();
    }

    // Setup-only mode (TC pinv path): K2 (k2_softmax) and Z_0 (ns_iterates[0]) are
    // now in GMEM; the Newton-Schulz iterations run as external tensor-core GEMMs.
    // For the Tikhonov ridge the matrix inverted by NS is M = K2^T K2 + lambda*I,
    // which the ridge block above built into the K2 SMEM slot. Export it so the
    // external NS can read it (no-ridge reads K2 directly from k2_softmax).
    if constexpr (kSetupOnly) {
        if (kappa_star > 0.0f && kernel2_inv_out != nullptr) {
            float* mout = kernel2_inv_out + bh * mm;
            for (int idx = tid; idx < mm; idx += nthreads) mout[idx] = K2[idx];
        }
    }

    // Compile the in-kernel scalar NS + final write out entirely (an early return
    // would make the loop statically unreachable, which -Werror rejects).
    if constexpr (!kSetupOnly) {

    // Step 4: Newton-Schulz iterations (third order Higham)
    // Z_{j+1} = 0.25 * Z_j * (13I - K2*Z_j * (15I - K2*Z_j * (7I - K2*Z_j)))
    //
    // Each iteration is four ns_matmul_phase calls with the "cI - X"
    // elementwise transforms fused into the consuming matmul's B operand
    // (same fp32 values, three fewer barriers) and Z advanced by pointer
    // swap instead of a Zold copy. Buffers rotate through the same four
    // SMEM slots the naive form used. Per-cell sums are bitwise identical
    // to the naive form; only phase count and thread-to-cell mapping differ.
    //
    // Save each iterate Z_{j+1} to GMEM (index j+1).
    {
    float* W = Zold;
    for (int iter = 0; iter < newton_iter; iter++) {
        // T1 = M = K2 @ Z_j
        ns_matmul_phase(T1, K2, Z, m, tid, nthreads, 0.0f, 1.0f);
        __syncthreads();
        // T2 = M @ (7I - M)
        ns_matmul_phase(T2, T1, T1, m, tid, nthreads, 7.0f, 1.0f);
        __syncthreads();
        // W = M @ (15I - T2)
        ns_matmul_phase(W, T1, T2, m, tid, nthreads, 15.0f, 1.0f);
        __syncthreads();
        // T2 = 0.25 * Z_j @ (13I - W)   (Z_j still intact; no Zold copy)
        ns_matmul_phase(T2, Z, W, m, tid, nthreads, 13.0f, 0.25f);
        __syncthreads();
        // advance the iterate by pointer swap (uniform across the block)
        float* t = Z; Z = T2; T2 = t;

        // Save Z_{iter+1} to GMEM (index iter+1)
        if (ns_iterates_out != nullptr) {
            float* iter_out = ns_iterates_out + bh * (newton_iter + 1) * mm + (iter + 1) * mm;
            for (int idx = tid; idx < mm; idx += nthreads) iter_out[idx] = Z[idx];
            __syncthreads();
        }
    }
    }

    // Step 5: Write final K2_inv for kernel3.
    //  - no ridge:  K2_inv = Z_J            (Z_J = K2^-1 from NS on K2)
    //  - Tikhonov:  K2_inv = M^-1 K2^T = Z_J K2^T  (Z_J = M^-1; reload un-ridged
    //               K2 from GMEM since the K2 SMEM slot held M during NS)
    float* out = kernel2_inv_out + bh * mm;
    if (kappa_star > 0.0f && k2_softmax_out != nullptr) {
        const float* k2sm_bh = k2_softmax_out + bh * mm;
        for (int idx = tid; idx < mm; idx += nthreads) K2[idx] = k2sm_bh[idx];  // K2 slot <- un-ridged K2
        __syncthreads();
        for (int idx = tid; idx < mm; idx += nthreads) {
            int i = idx / m, j = idx % m;
            float acc = 0.0f;
            for (int k = 0; k < m; k++) acc += Z[i * m + k] * K2[j * m + k];  // (Z_J K2^T)[i,j]
            out[idx] = acc;
        }
    } else {
        for (int idx = tid; idx < mm; idx += nthreads) out[idx] = Z[idx];
    }
    }  // if constexpr (!kSetupOnly)
}

template <typename scalar_t>
void launch_kernel2_inv(
    const scalar_t* q_tilde, const scalar_t* k_tilde,
    float* kernel2_inv, float* softmax_lse,
    float* ns_iterates,        // (BH, newton_iter+1, m, m) — REQUIRED for backward
    float* k2_softmax,         // (BH, m, m) — the (ridged) K2 (REQUIRED for backward)
    int BH, int D, int m, int newton_iter,
    cudaStream_t stream,
    float kappa_star = 0.0f
) {
    FN_CHECK(m > 0 && m <= kMaxLandmarks, "launch_kernel2_inv: m out of range");

    dim3 grid(BH);
    // 1024 threads: the NS matmuls give each thread mm/blockDim serial
    // 64-long dot products; at 256 threads the kernel ran 8 warps/SM (96KB
    // smem -> 1 CTA/SM) and was latency-bound at a flat ~0.86 ms on a B200,
    // 62% of the high-BH forward at N=4096. 1024 threads quarter the
    // per-thread chain and quadruple the warps available to hide smem
    // latency. Per-cell dot-product order is unchanged (same fp32 results
    // cell-by-cell); only the block-reduce tree shape differs (~1 ulp on
    // the softmax row sums and norm scalars).
    dim3 block(1024);

    // scratch: blockDim/32 floats for the block reductions
    size_t smem_bytes = (5 * m * m + 32) * sizeof(float);

    size_t max_smem = get_max_smem_per_block();
    if (smem_bytes > 48 * 1024) {
        FN_CHECK(smem_bytes <= max_smem,
                 "kernel2_inv: m too large for available shared memory");
        FN_CUDA_CHECK(cudaFuncSetAttribute(
            kernel2_inv_kernel<scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
    }

    kernel2_inv_kernel<scalar_t><<<grid, block, smem_bytes, stream>>>(
        q_tilde, k_tilde, kernel2_inv, softmax_lse, ns_iterates, k2_softmax,
        D, m, newton_iter, kappa_star);
    FN_CUDA_KERNEL_CHECK();
}

// Setup-only launch for the TC pinv path: computes softmax K2 (-> k2_softmax),
// the Z_0 init (-> ns_iterates[0]), and the LSE, then returns. The Newton-Schulz
// iterations run as external tensor-core GEMMs (see run_kernel2_inv_tc). Does not
// touch kernel2_inv_out.
// m_out receives M = K2^T K2 + lambda*I when kappa_star > 0 (the matrix the
// external NS inverts); pass nullptr for the no-ridge path (NS reads k2_softmax).
template <typename scalar_t>
void launch_kernel2_inv_setup(
    const scalar_t* q_tilde, const scalar_t* k_tilde,
    float* softmax_lse, float* ns_iterates, float* k2_softmax, float* m_out,
    int BH, int D, int m, int newton_iter, cudaStream_t stream, float kappa_star
) {
    FN_CHECK(m > 0 && m <= kMaxLandmarks, "launch_kernel2_inv_setup: m out of range");
    dim3 grid(BH), block(1024);   // see launch_kernel2_inv
    size_t smem_bytes = (5 * m * m + 32) * sizeof(float);
    if (smem_bytes > 48 * 1024) {
        FN_CHECK(smem_bytes <= get_max_smem_per_block(),
                 "kernel2_inv_setup: m too large for available shared memory");
        FN_CUDA_CHECK(cudaFuncSetAttribute(
            kernel2_inv_kernel<scalar_t, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    kernel2_inv_kernel<scalar_t, true><<<grid, block, smem_bytes, stream>>>(
        q_tilde, k_tilde, m_out, softmax_lse, ns_iterates, k2_softmax,
        D, m, newton_iter, kappa_star);
    FN_CUDA_KERNEL_CHECK();
}

} // namespace flash_nystrom
