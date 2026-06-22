/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
// kernel orchestration — this is where the actual forward/backward pipelines
// get assembled from individual kernel launches. nothing fancy, just sequencing.

#include "flash_nystrom.h"
#include "utils.h"
#include "profile.h"
#include "static_switch.h"
#include "kernels/landmark.cuh"
#include "kernels/kernel2_inv.cuh"
#include "kernels/k2inv_gemm_tc.cuh"          // tf32 TC GEMM primitive (forward NS)
#include "cublas_helpers.cuh"                 // launch_affine_with_identity
#include <ATen/ops/empty.h>
#include <ATen/Tensor.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAStream.h>
#include "kernels/kernel3_output_fused.cuh"  // Tensor-core version (FP16/BF16)
#include "kernels/kernel3_scalar.cuh"        // Scalar fallback (FP32)
#include "kernels/kernel1_output_fused.cuh"  // Tensor-core version (FP16/BF16)
#include "kernels/kernel1_scalar.cuh"        // Scalar fallback (FP32)

// Backward kernels
#include "kernels/backward/precompute_di.cuh"
#include "kernels/backward/kernel1_bwd.cuh"
#include "kernels/backward/kernel3_bwd.cuh"
#include "kernels/backward/compute_dO3.cuh"
#include "kernels/backward/compute_dk2inv.cuh"
#include "kernels/backward/kernel2_inv_bwd.cuh"
#include "kernels/backward/landmark_bwd.cuh"

namespace flash_nystrom {

// Persistent workspace + cached CUDA graph for the tf32 TC forward NS chain. The
// ~9 GEMM/affine/copy launches per iteration are captured once per shape and
// replayed with a single cudaGraphLaunch; setup (reads q,k) stays outside.
struct K2InvTcGraph {
    int BH = -1, m = -1, niter = -1;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    at::Tensor K2, iter, k2inv, wKZ, wA, wB, wZ, wZn;

    bool matches(int bh, int m_, int n) const { return BH == bh && m == m_ && niter == n; }
    void invalidate() {
        if (exec)  { cudaGraphExecDestroy(exec); exec = nullptr; }
        if (graph) { cudaGraphDestroy(graph);    graph = nullptr; }
    }
    void allocate(int bh, int m_, int n) {
        BH = bh; m = m_; niter = n; invalidate();
        auto o = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
        K2 = at::empty({bh, m_, m_}, o);  iter = at::empty({bh, n + 1, m_, m_}, o);
        k2inv = at::empty({bh, m_, m_}, o);
        wKZ = at::empty({bh, m_, m_}, o); wA = at::empty({bh, m_, m_}, o);
        wB = at::empty({bh, m_, m_}, o);  wZ = at::empty({bh, m_, m_}, o);
        wZn = at::empty({bh, m_, m_}, o);
    }
    ~K2InvTcGraph() {
        if (exec)  (void)cudaGraphExecDestroy(exec);
        if (graph) (void)cudaGraphDestroy(graph);
    }
};

static K2InvTcGraph& k2inv_tc_graph() { static thread_local K2InvTcGraph s; return s; }

// Record the NS chain on persistent buffers (K2 and iter[0]=Z_0 must be populated
// before launch). Fills iter[1..J] and k2inv (=Z_J). Capturable: only kernel
// launches + memcpys, no synchronizing calls. Mirrors iterative_pinverse:
//   KZ = K2 Z;  Z' = 0.25 Z (13I - KZ(15I - KZ(7I - KZ))).
static void record_k2inv_tc_ns(K2InvTcGraph &s, cudaStream_t stream) {
    const int BH = s.BH, m = s.m, J = s.niter;
    const long long mm = (long long)m * m, zstr = (long long)(J + 1) * mm;
    const size_t rowB = mm * sizeof(float);
    float* K2 = s.K2.data_ptr<float>(); float* IT = s.iter.data_ptr<float>();
    float* KZ = s.wKZ.data_ptr<float>(); float* A = s.wA.data_ptr<float>();
    float* B = s.wB.data_ptr<float>();  float* Z = s.wZ.data_ptr<float>(); float* Zn = s.wZn.data_ptr<float>();
    FN_CUDA_CHECK(cudaMemcpy2DAsync(Z, rowB, IT, zstr * sizeof(float), rowB, BH,
        cudaMemcpyDeviceToDevice, stream));                                       // Z_0
    for (int j = 0; j < J; j++) {
        launch_k2inv_gemm_nn(K2, Z, KZ, BH, m, stream);                              // KZ = K2 Z
        launch_affine_with_identity(A, KZ, nullptr, 7.f, -1.f, 0.f, BH, m, stream);  // 7I-KZ
        launch_k2inv_gemm_nn(KZ, A, B, BH, m, stream);
        launch_affine_with_identity(A, B, nullptr, 15.f, -1.f, 0.f, BH, m, stream);  // 15I-KZ(.)
        launch_k2inv_gemm_nn(KZ, A, B, BH, m, stream);
        launch_affine_with_identity(A, B, nullptr, 13.f, -1.f, 0.f, BH, m, stream);  // 13I-KZ(.)
        launch_k2inv_gemm_nn(Z, A, B, BH, m, stream);
        launch_affine_with_identity(Zn, B, nullptr, 0.f, 0.25f, 0.f, BH, m, stream); // 0.25 Z(.)
        FN_CUDA_CHECK(cudaMemcpy2DAsync(IT + (long long)(j + 1) * mm, zstr * sizeof(float),
            Zn, rowB, rowB, BH, cudaMemcpyDeviceToDevice, stream));                  // -> iter[j+1]
        float* tmp = Z; Z = Zn; Zn = tmp;
    }
    FN_CUDA_CHECK(cudaMemcpyAsync(s.k2inv.data_ptr<float>(), Z,
        (size_t)BH * mm * sizeof(float), cudaMemcpyDeviceToDevice, stream));         // k2inv = Z_J
}

// Tensor-core forward pseudoinverse (no-ridge), graph-captured per shape.
template <typename elem_type>
static void run_kernel2_inv_tc(const elem_type* qt, const elem_type* kt, NystromParams &p) {
    const int BH = p.BH, m = p.num_landmarks, D = p.head_dim, J = p.newton_iter;
    const long long mm = (long long)m * m, zstr = (long long)(J + 1) * mm;
    cudaStream_t stream = p.stream;

    launch_kernel2_inv_setup<elem_type>(qt, kt, p.softmax2_lse_ptr, p.ns_iterates_ptr,
        p.k2_softmax_ptr, BH, D, m, J, stream, 0.0f);

    auto &s = k2inv_tc_graph();
    if (!s.matches(BH, m, J)) s.allocate(BH, m, J);

    // Setup outputs -> persistent buffers: K2 (contiguous) and Z_0 (iter[*,0]).
    FN_CUDA_CHECK(cudaMemcpyAsync(s.K2.data_ptr<float>(), p.k2_softmax_ptr,
        (size_t)BH * mm * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    FN_CUDA_CHECK(cudaMemcpy2DAsync(s.iter.data_ptr<float>(), zstr * sizeof(float),
        p.ns_iterates_ptr, zstr * sizeof(float), mm * sizeof(float), BH,
        cudaMemcpyDeviceToDevice, stream));

    if (s.exec == nullptr) {
        auto side = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
        cudaStream_t cap = side.stream();
        FN_CUDA_CHECK(cudaStreamBeginCapture(cap, cudaStreamCaptureModeThreadLocal));
        record_k2inv_tc_ns(s, cap);
        FN_CUDA_CHECK(cudaStreamEndCapture(cap, &s.graph));
        FN_CUDA_CHECK(cudaGraphInstantiate(&s.exec, s.graph, nullptr, nullptr, 0));
    }
    FN_CUDA_CHECK(cudaGraphLaunch(s.exec, stream));

    // Persistent results -> caller buffers.
    FN_CUDA_CHECK(cudaMemcpyAsync(p.ns_iterates_ptr, s.iter.data_ptr<float>(),
        (size_t)BH * (J + 1) * mm * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    FN_CUDA_CHECK(cudaMemcpyAsync(p.kernel2_inv_ptr, s.k2inv.data_ptr<float>(),
        (size_t)BH * mm * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    FN_CUDA_KERNEL_CHECK();
}

// Pinv dispatch: tensor-core NS chain (UseTC) or the scalar single-CTA kernel.
template <typename elem_type, bool UseTC>
static void run_kernel2_inv(const elem_type* qt, const elem_type* kt,
                            NystromParams &p, float kappa_star) {
    if constexpr (UseTC) {
        run_kernel2_inv_tc<elem_type>(qt, kt, p);
    } else {
        launch_kernel2_inv<elem_type>(qt, kt,
            p.kernel2_inv_ptr, p.softmax2_lse_ptr, p.ns_iterates_ptr, p.k2_softmax_ptr,
            p.BH, p.head_dim, p.num_landmarks, p.newton_iter, p.stream, kappa_star);
    }
}

// FP16/BF16 path: uses tensor-core kernel1
template <typename elem_type>
static void run_nystrom_fwd_half(NystromParams &p) {
    auto* q_in = static_cast<const elem_type*>(p.q_in_ptr);  // unscaled user Q
    auto* k_in = static_cast<const elem_type*>(p.k_in_ptr);  // unscaled user K
    auto* v   = static_cast<const elem_type*>(p.v_ptr);
    auto* o   = static_cast<elem_type*>(p.o_ptr);
    auto* qt  = static_cast<elem_type*>(p.q_tilde_ptr);
    auto* kt  = static_cast<elem_type*>(p.k_tilde_ptr);
    auto* s2  = static_cast<elem_type*>(p.step2_ptr);
    auto* b   = static_cast<elem_type*>(p.b_ptr);
    auto* q_m = static_cast<elem_type*>(p.q_ptr);            // scaled Q (dst)
    auto* k_m = static_cast<elem_type*>(p.k_ptr);            // scaled K (dst)

    KernelProfiler prof(p.stream);
    int total = p.BH * p.seq_len * p.head_dim;

    prof.run("landmarks", [&] {
        launch_landmarks<elem_type>(q_in, k_in, qt, kt,
            p.BH, p.seq_len, p.head_dim, p.num_landmarks, p.scale, p.stream);
    });
    // Scaled copy q_in -> q_m, k_in -> k_m (folds the softmax scale into the
    // clone the backward needs anyway, replacing a separate scale_inplace pass).
    prof.run("scaled_copy(q,k)", [&] {
        launch_scaled_copy<elem_type>(q_in, q_m, total, p.scale, p.stream);
        launch_scaled_copy<elem_type>(k_in, k_m, total, p.scale, p.stream);
    });
    // Tikhonov ridge target condition number; the kernel computes lambda =
    // (||K2||_1 ||K2||_inf)/kappa_star internally and inverts M = K2^T K2 +
    // lambda*I (non-normality-proof). FN_KAPPA_STAR is the knob; unset/0 = off.
    const char* ks_env = std::getenv("FN_KAPPA_STAR");
    float kappa_star = ks_env ? static_cast<float>(atof(ks_env)) : 0.0f;
    // FN_K2INV_TC=1 routes the pinv through the tf32 tensor-core NS chain. CP2
    // covers the no-ridge path only; with the ridge on we still use the scalar
    // kernel until the TC ridge lands (CP5).
    const char* tc_env = std::getenv("FN_K2INV_TC");
    bool use_tc = (tc_env && atoi(tc_env) != 0) && (kappa_star == 0.0f);
    prof.run("kernel2_inv", [&] {
        BOOL_SWITCH(use_tc, kUseTC, [&] {
            run_kernel2_inv<elem_type, kUseTC>(qt, kt, p, kappa_star);
        });
    });
    prof.run("kernel3_output_fused", [&] {
        launch_kernel3_output_fused<elem_type>(qt, k_m, v,
            p.kernel2_inv_ptr, s2, b, p.softmax3_lse_ptr,
            p.BH, p.seq_len, p.head_dim, p.num_landmarks, p.stream);
    });
    prof.run("kernel1_output_fused", [&] {
        // Tensor-core kernel1 (the main performance kernel)
        launch_kernel1_output_fused<elem_type>(q_m, kt, s2,
            o, p.softmax1_lse_ptr,
            p.BH, p.seq_len, p.head_dim, p.num_landmarks, p.stream);
    });
    prof.report("forward (FP16/BF16)");
}

// FP32 path: uses scalar kernel1 (LDSM doesn't support 32-bit elements)
static void run_nystrom_fwd_fp32_impl(NystromParams &p) {
    using T = float;
    auto* q_in = static_cast<const T*>(p.q_in_ptr);  // unscaled user Q
    auto* k_in = static_cast<const T*>(p.k_in_ptr);  // unscaled user K
    auto* v   = static_cast<const T*>(p.v_ptr);
    auto* o   = static_cast<T*>(p.o_ptr);
    auto* qt  = static_cast<T*>(p.q_tilde_ptr);
    auto* kt  = static_cast<T*>(p.k_tilde_ptr);
    auto* s2  = static_cast<T*>(p.step2_ptr);
    auto* b   = static_cast<T*>(p.b_ptr);
    auto* q_m = static_cast<T*>(p.q_ptr);            // scaled Q (dst)
    auto* k_m = static_cast<T*>(p.k_ptr);            // scaled K (dst)

    launch_landmarks<T>(q_in, k_in, qt, kt,
        p.BH, p.seq_len, p.head_dim, p.num_landmarks, p.scale, p.stream);

    int total = p.BH * p.seq_len * p.head_dim;
    launch_scaled_copy<T>(q_in, q_m, total, p.scale, p.stream);
    launch_scaled_copy<T>(k_in, k_m, total, p.scale, p.stream);

    const char* ks_env = std::getenv("FN_KAPPA_STAR");
    float kappa_star = ks_env ? static_cast<float>(atof(ks_env)) : 0.0f;
    launch_kernel2_inv<T>(qt, kt,
        p.kernel2_inv_ptr, p.softmax2_lse_ptr, p.ns_iterates_ptr, p.k2_softmax_ptr,
        p.BH, p.head_dim, p.num_landmarks, p.newton_iter, p.stream, kappa_star);

    launch_kernel3_scalar<T>(qt, k_m, v,
        p.kernel2_inv_ptr, s2, b, p.softmax3_lse_ptr,
        p.BH, p.seq_len, p.head_dim, p.num_landmarks, p.stream);

    launch_kernel1_scalar<T>(q_m, kt, s2,
        o, p.softmax1_lse_ptr,
        p.BH, p.seq_len, p.head_dim, p.num_landmarks, p.stream);
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

    KernelProfiler prof(p.stream);

    prof.run("precompute_di", [&] {
        launch_precompute_di<elem_type>(dO, output, p.D1_ptr, BH, N, D, p.stream);
    });
    prof.run("kernel1_bwd", [&] {
        launch_kernel1_bwd<elem_type>(q_s, k_tilde, step2, p.lse1_ptr, p.D1_ptr, dO,
            dQ, p.dstep2_ptr, p.dK_tilde_ptr, BH, N, D, m, p.stream);
    });

    // dO3 = K2_inv^T @ dstep2 — precomputed in GMEM for both the FP32 scalar
    // and FP16/BF16 TC paths. Used by the kernel3_bwd softmax-bwd stage and
    // by compute_dk2inv (for the D3 byproduct).
    auto* dO3 = static_cast<elem_type*>(p.dO3_ptr);
    prof.run("compute_dO3", [&] {
        launch_compute_dO3<elem_type>(p.k2_inv_ptr, p.dstep2_ptr, dO3,
            BH, D, m, p.stream);
    });

    // B = softmax(Q_tilde @ K_s^T) @ V is saved from the forward kernel3.
    // When B is provided, compute_dk2inv collapses to two small m-bounded
    // matmuls (no N-walk). The launch dispatches to compute_dk2inv_from_b.
    // If B is null (FP32 input without a saved B, for example), the path
    // falls back to the prior N-walking compute_dk2inv variants.
    auto* b_saved = static_cast<const elem_type*>(p.b_ptr);
    prof.run("compute_dk2inv", [&] {
        launch_compute_dk2inv<elem_type>(q_tilde, k_s, v, b_saved, dO3,
            p.lse3_ptr, p.dstep2_ptr,
            p.dK2_inv_ptr, p.D3_ptr,
            BH, N, D, m, p.fast_dk2inv, p.stream);
    });
    prof.run("kernel3_bwd", [&] {
        launch_kernel3_bwd<elem_type>(q_tilde, k_s, v, p.k2_inv_ptr, p.lse3_ptr,
            p.D3_ptr, p.dstep2_ptr, dV, dK, p.dQ_tilde_ptr, p.dK2_inv_ptr,
            static_cast<const elem_type*>(dO3),
            p.dQ_tilde_split_ptr, p.num_splits,
            BH, N, D, m, p.stream);
    });
    // Tikhonov ridge: pass kappa_star through; the backward computes the
    // per-bh lambda and the M = K2^T K2 + lambda*I wrap internally, matching
    // the forward. FN_KAPPA_STAR is the knob; unset/0 = no ridge.
    const char* ks_env_b = std::getenv("FN_KAPPA_STAR");
    float kappa_star_b = ks_env_b ? static_cast<float>(atof(ks_env_b)) : 0.0f;
    prof.run("kernel2_inv_bwd", [&] {
        launch_kernel2_inv_bwd<elem_type>(q_tilde, k_tilde,
            p.dK2_inv_ptr,
            p.ns_iterates_ptr, p.k2_softmax_ptr,
            p.dQ_tilde_ptr, p.dK_tilde_ptr,
            BH, D, m, p.newton_iter, p.stream, kappa_star_b);
    });
    prof.run("landmark_bwd", [&] {
        launch_landmark_bwd<elem_type>(p.dQ_tilde_ptr, p.dK_tilde_ptr, dQ, dK,
            BH, N, D, m, p.stream);
    });

    int total = BH * N * D;
    prof.run("scale_inplace(dQ,dK)", [&] {
        launch_scale_inplace<elem_type>(dQ, total, p.scale, p.stream);
        launch_scale_inplace<elem_type>(dK, total, p.scale, p.stream);
    });
    prof.report("backward (FP16/BF16)");
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

// Bridge so the pybind reset_caches can free the kernel3 split-N scratch.
void reset_kernel3_caches() { reset_kernel3_scratch(); }

} // namespace flash_nystrom
