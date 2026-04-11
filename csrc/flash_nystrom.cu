/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
// pytorch binding — converts torch tensors to void* pointers in param structs,
// dispatch based on dtype, and wraps everthing up for pybind11.
// the kernel code never touches pytorch types directly (learned from FlashAttention)

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "flash_nystrom.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
    #x " shape mismatch, got " + std::to_string(x.dim()) + "D")

namespace flash_nystrom {

std::vector<torch::Tensor> nystrom_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int64_t num_landmarks,
    int64_t newton_iter,
    int64_t conv_kernel_size,
    c10::optional<torch::Tensor> conv_weight
) {
    // Input validation
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_CONTIGUOUS(q); CHECK_CONTIGUOUS(k); CHECK_CONTIGUOUS(v);
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, v must be 4D (B, H, N, D)");
    TORCH_CHECK(q.dtype() == k.dtype() && k.dtype() == v.dtype(),
                "q, k, v dtypes must match");

    const auto dtype = q.scalar_type();
    TORCH_CHECK(dtype == at::ScalarType::Float ||
                dtype == at::ScalarType::Half ||
                dtype == at::ScalarType::BFloat16,
                "Supported dtypes: float32, float16, bfloat16");

    const int64_t B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);
    const int64_t m = num_landmarks;

    // Overflow check: B*H*N*D must fit in int32 for CUDA kernel indexing
    TORCH_CHECK(B * H * N * D <= INT32_MAX,
                "Total tensor elements (", B*H*N*D, ") exceeds int32 range");
    TORCH_CHECK(B * H <= INT32_MAX / (N * D),
                "B*H*N*D overflow");

    CHECK_SHAPE(k, B, H, N, D);
    CHECK_SHAPE(v, B, H, N, D);
    TORCH_CHECK(N >= m, "seq_len (", N, ") must be >= num_landmarks (", m, ")");
    TORCH_CHECK(m > 0 && m <= 64,
                "num_landmarks must be in [1, 64] (kernel tile size limit)");
    TORCH_CHECK(D == 64 || D == 128,
                "head_dim must be 64 or 128 (other values not yet supported)");
    TORCH_CHECK(newton_iter >= 1 && newton_iter <= 20, "newton_iter must be in [1, 20]");
    TORCH_CHECK(conv_kernel_size >= 0, "conv_kernel_size must be non-negative");
    if (conv_kernel_size > 0) {
        TORCH_CHECK(conv_kernel_size % 2 == 1, "conv_kernel_size must be odd");
    }

    if (conv_weight.has_value()) {
        auto cw = conv_weight.value();
        CHECK_DEVICE(cw); CHECK_CONTIGUOUS(cw);
        TORCH_CHECK(cw.dtype() == dtype, "conv_weight dtype must match q/k/v");
        TORCH_CHECK(cw.dim() == 2, "conv_weight must be 2D (H, kernel_size)");
        TORCH_CHECK(cw.size(0) == H && cw.size(1) == conv_kernel_size,
                     "conv_weight shape must be (", H, ", ", conv_kernel_size, ")");
    } else {
        TORCH_CHECK(conv_kernel_size == 0,
                     "conv_kernel_size > 0 but conv_weight not provided");
    }

    // FP32 with D=128 may exceed SMEM limits on some GPUs.
    // The kernel launcher will check and abort if SMEM is insufficient.
    if (dtype == at::ScalarType::Float && D == 128) {
        TORCH_WARN_ONCE("FlashNystrom FP32 with D=128 uses scalar kernels with large SMEM. "
                         "Consider FP16/BF16 for better performance.");
    }

    const at::cuda::CUDAGuard device_guard(q.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto opts = q.options();
    auto opts_f32 = opts.dtype(torch::kFloat32);

    // Clone q, k for in-place scaling (originals are not modified)
    auto q_s = q.clone();
    auto k_s = k.clone();

    auto output       = torch::empty({B, H, N, D}, opts);
    auto q_tilde      = torch::empty({B, H, m, D}, opts);
    auto k_tilde      = torch::empty({B, H, m, D}, opts);
    auto kernel2_inv  = torch::empty({B, H, m, m}, opts_f32);
    auto step2        = torch::empty({B, H, m, D}, opts);
    auto softmax1_lse = torch::empty({B, H, N}, opts_f32);
    auto softmax2_lse = torch::empty({B, H, m}, opts_f32);
    auto softmax3_lse = torch::empty({B, H, m}, opts_f32);
    NystromParams params = {};
    params.batch_size = static_cast<int>(B);
    params.num_heads = static_cast<int>(H);
    params.seq_len = static_cast<int>(N);
    params.head_dim = static_cast<int>(D);
    params.num_landmarks = static_cast<int>(m);
    params.newton_iter = static_cast<int>(newton_iter);
    params.conv_kernel_size = static_cast<int>(conv_kernel_size);
    params.is_bf16 = (dtype == at::ScalarType::BFloat16);

    params.q_ptr = q_s.data_ptr();
    params.k_ptr = k_s.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = output.data_ptr();
    params.q_tilde_ptr = q_tilde.data_ptr();
    params.k_tilde_ptr = k_tilde.data_ptr();
    params.kernel2_inv_ptr = kernel2_inv.data_ptr<float>();
    params.step2_ptr = step2.data_ptr();
    params.softmax1_lse_ptr = softmax1_lse.data_ptr<float>();
    params.softmax2_lse_ptr = softmax2_lse.data_ptr<float>();
    params.softmax3_lse_ptr = softmax3_lse.data_ptr<float>();
    params.ns_iterates_ptr = nullptr;  // IFT backward doesn't need iterates
    params.conv_weight_ptr = conv_weight.has_value() ? conv_weight.value().data_ptr() : nullptr;
    params.stream = stream;

    if (dtype == at::ScalarType::Float) {
        run_nystrom_fwd_fp32(params);
    } else {
        run_nystrom_fwd(params);
    }

    return {output, q_s, k_s, q_tilde, k_tilde, kernel2_inv, step2,
            softmax1_lse, softmax2_lse, softmax3_lse};
}

// -- backward --


std::vector<torch::Tensor> nystrom_bwd(
    torch::Tensor dO,
    torch::Tensor q_s, torch::Tensor k_s,
    torch::Tensor q_tilde, torch::Tensor k_tilde,
    torch::Tensor kernel2_inv, torch::Tensor step2,
    torch::Tensor softmax1_lse, torch::Tensor softmax2_lse, torch::Tensor softmax3_lse,
    torch::Tensor v, torch::Tensor output,
    int64_t num_landmarks, int64_t newton_iter, int64_t conv_kernel_size,
    c10::optional<torch::Tensor> conv_weight
) {
    const auto dtype = dO.scalar_type();
    const int64_t B = dO.size(0), H = dO.size(1), N = dO.size(2), D = dO.size(3);
    const int64_t m = num_landmarks;

    const at::cuda::CUDAGuard device_guard(dO.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto opts = dO.options();
    auto opts_f32 = opts.dtype(torch::kFloat32);

    // Gradient outputs (zero-initialized)
    auto dQ = torch::zeros({B, H, N, D}, opts);
    auto dK = torch::zeros({B, H, N, D}, opts);
    auto dV = torch::zeros({B, H, N, D}, opts);

    // FP32 intermediate accumulators (zero-initialized)
    auto dstep2    = torch::zeros({B, H, m, D}, opts_f32);
    auto dQ_tilde  = torch::zeros({B, H, m, D}, opts_f32);
    auto dK_tilde  = torch::zeros({B, H, m, D}, opts_f32);
    auto dK2_inv   = torch::zeros({B, H, m, m}, opts_f32);
    auto D1        = torch::empty({B, H, N}, opts_f32);

    // dconv_weight (FP32 accumulator, converted back to dtype at the end)
    torch::Tensor dconv_weight;
    if (conv_weight.has_value() && conv_kernel_size > 0) {
        dconv_weight = torch::zeros({H, conv_kernel_size}, opts_f32);
    }

    NystromBwdParams params = {};
    params.batch_size = static_cast<int>(B);
    params.num_heads = static_cast<int>(H);
    params.seq_len = static_cast<int>(N);
    params.head_dim = static_cast<int>(D);
    params.num_landmarks = static_cast<int>(m);
    params.newton_iter = static_cast<int>(newton_iter);
    params.conv_kernel_size = static_cast<int>(conv_kernel_size);
    params.is_bf16 = (dtype == at::ScalarType::BFloat16);

    params.q_s_ptr = q_s.data_ptr();
    params.k_s_ptr = k_s.data_ptr();
    params.v_ptr = v.data_ptr();
    params.q_tilde_ptr = q_tilde.data_ptr();
    params.k_tilde_ptr = k_tilde.data_ptr();
    params.k2_inv_ptr = kernel2_inv.data_ptr<float>();
    params.step2_ptr = step2.data_ptr();
    params.o_ptr = output.data_ptr();
    params.lse1_ptr = softmax1_lse.data_ptr<float>();
    params.lse2_ptr = softmax2_lse.data_ptr<float>();
    params.lse3_ptr = softmax3_lse.data_ptr<float>();
    params.ns_iterates_ptr = nullptr;  // IFT backward doesn't need iterates
    params.conv_weight_ptr = conv_weight.has_value() ? conv_weight.value().data_ptr() : nullptr;

    params.dO_ptr = dO.data_ptr();
    params.dQ_ptr = dQ.data_ptr();
    params.dK_ptr = dK.data_ptr();
    params.dV_ptr = dV.data_ptr();
    params.dconv_weight_ptr = (conv_weight.has_value() && conv_kernel_size > 0) ?
        dconv_weight.data_ptr<float>() : nullptr;

    params.dstep2_ptr = dstep2.data_ptr<float>();
    params.dQ_tilde_ptr = dQ_tilde.data_ptr<float>();
    params.dK_tilde_ptr = dK_tilde.data_ptr<float>();
    params.dK2_inv_ptr = dK2_inv.data_ptr<float>();
    params.D1_ptr = D1.data_ptr<float>();
    params.stream = stream;

    if (dtype == at::ScalarType::Float) {
        run_nystrom_bwd_fp32(params);
    } else {
        run_nystrom_bwd(params);
    }

    // Convert dconv_weight from FP32 to output dtype
    torch::Tensor dconv_out;
    if (conv_weight.has_value() && conv_kernel_size > 0) {
        dconv_out = dconv_weight.to(dtype);
    }

    return {dQ, dK, dV, dconv_out};
}

} // namespace flash_nystrom

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_nystrom::nystrom_fwd,
          "FlashNystrom forward (CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("num_landmarks") = 64,
          py::arg("newton_iter") = 6,
          py::arg("conv_kernel_size") = 0,
          py::arg("conv_weight") = c10::nullopt);
    m.def("backward", &flash_nystrom::nystrom_bwd,
          "FlashNystrom backward (CUDA)");
}
