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

#include <algorithm>  // std::min / std::max for the k1 split heuristic
#include <cstdlib>    // std::getenv / std::atoi for FLASH_NYSTROM_K1_SPLITS
#include <type_traits> // std::remove_pointer_t for the landmark dtype dispatch

#include "flash_nystrom.h"
#include "kernels/backward/kernel2_inv_bwd.cuh"  // for debug hooks
#include "kernels/backward/compute_dk2inv.cuh"   // for debug hooks
#include "kernels/k2inv_gemm_tc.cuh"             // for the TC GEMM debug hook
#include "kernels/leverage_landmarks.cuh"        // leverage-seeded Voronoi-mean landmarks
#include "occupancy_probe.h"                      // for occupancy reporting

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
    double kappa_star,
    bool use_tc_pinv,
    int64_t landmark_mode,
    int64_t landmark_seed,
    int64_t landmark_subsample,
    double landmark_gumbel_scale,
    int64_t landmark_force_first
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

    // The per-(batch,head) slice base offset is computed in int64 (the slice
    // index bh is int64_t in every kernel), so B*H*N*D may exceed int32.
    // Tile-local indexing within one slice stays int32, so N*D must fit int32.
    TORCH_CHECK(N * D <= INT32_MAX,
                "per-slice N*D (", N * D, ") exceeds int32 range");

    CHECK_SHAPE(k, B, H, N, D);
    CHECK_SHAPE(v, B, H, N, D);
    TORCH_CHECK(N >= m, "seq_len (", N, ") must be >= num_landmarks (", m, ")");
    TORCH_CHECK(m > 0 && m <= 64,
                "num_landmarks must be in [1, 64] (kernel tile size limit)");
    TORCH_CHECK(D == 64 || D == 128,
                "head_dim must be 64 or 128 (other values not yet supported)");
    TORCH_CHECK(newton_iter >= 1 && newton_iter <= 20, "newton_iter must be in [1, 20]");
    // Tikhonov ridge target cond(M). Must be finite and >= 0; 0 disables the
    // ridge. Reject inf/NaN/negative (would yield a meaningless lambda).
    TORCH_CHECK(std::isfinite(kappa_star) && kappa_star >= 0.0,
                "kappa_star must be finite and >= 0 (got ", kappa_star,
                "); 0 disables the Tikhonov ridge");

    // FP32 D=128 needs large opt-in dynamic SMEM: the scalar kernels use ~4
    // bytes/elem, so kernel3_scalar/kernel1_bwd peak near ~145-150KB at D=128,
    // m=64. That fits on datacenter parts (A100 164KB, H100/B200 228KB) but NOT
    // on consumer cards (RTX 40/50: ~100KB opt-in). We no longer hard-reject it
    // here: each scalar kernel opts into the size it needs via
    // cudaFuncSetAttribute and FN_CHECKs the request against
    // get_max_smem_per_block(), so an undersized GPU gets a clear per-kernel
    // "insufficient smem" error and a capable GPU just runs. FP32 D=128 is a
    // verification path (gradcheck), not a perf path — the scalar kernels are
    // slow by design.

    const at::cuda::CUDAGuard device_guard(q.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto opts = q.options();
    auto opts_f32 = opts.dtype(torch::kFloat32);

    // Scaled copies of q, k (q_s = q * scale). Allocated empty and filled by a
    // single scaled_copy pass in the kernel pipeline -- this replaces the old
    // "clone then scale_inplace" double pass (the clone existed only to avoid
    // mutating the user's inputs; folding the scale into it removes a full
    // redundant read+write of Q and K). The user's q, k are read-only. q_s, k_s
    // are saved for the backward, which still consumes the SCALED values.
    auto q_s = torch::empty_like(q);
    auto k_s = torch::empty_like(k);

    auto output       = torch::empty({B, H, N, D}, opts);
    auto q_tilde      = torch::empty({B, H, m, D}, opts);
    auto k_tilde      = torch::empty({B, H, m, D}, opts);
    auto kernel2_inv  = torch::empty({B, H, m, m}, opts_f32);
    auto step2        = torch::empty({B, H, m, D}, opts);
    auto softmax1_lse = torch::empty({B, H, N}, opts_f32);
    auto softmax2_lse = torch::empty({B, H, m}, opts_f32);
    auto softmax3_lse = torch::empty({B, H, m}, opts_f32);
    // Saved-for-backward tensors (always allocated; the unrolled NS backward needs them).
    auto ns_iterates  = torch::empty({B, H, static_cast<int64_t>(newton_iter + 1), m, m}, opts_f32);
    auto k2_softmax   = torch::empty({B, H, m, m}, opts_f32);
    // B = softmax(Q_tilde @ K^T) @ V. Saved so the backward's compute_dk2inv
    // does not need to N-walk to recompute it. m*D per batch-head, so total
    // size B*H*m*D bytes in dtype (typically a few hundred KB).
    auto b_saved      = torch::empty({B, H, m, D}, opts);

    // Landmark selection mode. 0 = segment mean (default). 1 = leverage-seeded
    // Voronoi means, which needs a scratch workspace and (for the
    // straight-through backward) exports the per-row Voronoi assignment and
    // per-cell counts for Q and K. Mode 0 leaves these as 0-element tensors.
    TORCH_CHECK(landmark_mode == 0 || landmark_mode == 1,
                "landmark_mode must be 0 (segment mean) or 1 (leverage Voronoi)");
    TORCH_CHECK(landmark_subsample >= 1, "landmark_subsample must be >= 1");
    const int BH = static_cast<int>(B * H);
    auto opts_i32 = opts.dtype(torch::kInt32);
    auto opts_u8  = opts.dtype(torch::kUInt8);
    torch::Tensor lm_workspace, q_assign, k_assign, q_cnt, k_cnt;
    if (landmark_mode == 1) {
        const size_t ws = flash_nystrom::lm_workspace_bytes(BH, (int)N, (int)D, (int)m);
        lm_workspace = torch::empty({(int64_t)ws}, opts_u8);
        q_assign = torch::empty({B, H, N}, opts_i32);
        k_assign = torch::empty({B, H, N}, opts_i32);
        q_cnt    = torch::empty({B, H, m}, opts_i32);
        k_cnt    = torch::empty({B, H, m}, opts_i32);
    } else {
        lm_workspace = torch::empty({0}, opts_u8);
        q_assign = torch::empty({0}, opts_i32);
        k_assign = torch::empty({0}, opts_i32);
        q_cnt    = torch::empty({0}, opts_i32);
        k_cnt    = torch::empty({0}, opts_i32);
    }

    NystromParams params = {};
    params.batch_size = static_cast<int>(B);
    params.num_heads = static_cast<int>(H);
    params.seq_len = static_cast<int>(N);
    params.head_dim = static_cast<int>(D);
    params.num_landmarks = static_cast<int>(m);
    params.newton_iter = static_cast<int>(newton_iter);
    params.is_bf16 = (dtype == at::ScalarType::BFloat16);
    params.kappa_star = static_cast<float>(kappa_star);
    params.use_tc_pinv = use_tc_pinv;
    params.landmark_mode = static_cast<int>(landmark_mode);
    params.landmark_seed = static_cast<uint64_t>(landmark_seed);
    params.landmark_subsample = static_cast<int>(landmark_subsample);
    params.landmark_gumbel_scale = static_cast<float>(landmark_gumbel_scale);
    params.landmark_force_first = static_cast<int>(landmark_force_first);
    if (landmark_mode == 1) {
        params.lm_workspace_ptr = lm_workspace.data_ptr();
        params.lm_workspace_bytes = static_cast<size_t>(lm_workspace.numel());
        params.q_assign_ptr = q_assign.data_ptr<int>();
        params.k_assign_ptr = k_assign.data_ptr<int>();
        params.q_cnt_ptr = q_cnt.data_ptr<int>();
        params.k_cnt_ptr = k_cnt.data_ptr<int>();
    }

    params.q_in_ptr = q.data_ptr();   // unscaled user Q (landmark + scaled_copy source)
    params.k_in_ptr = k.data_ptr();   // unscaled user K
    params.q_ptr = q_s.data_ptr();    // scaled Q (filled by scaled_copy, saved for bwd)
    params.k_ptr = k_s.data_ptr();    // scaled K
    params.v_ptr = v.data_ptr();
    params.o_ptr = output.data_ptr();
    params.q_tilde_ptr = q_tilde.data_ptr();
    params.k_tilde_ptr = k_tilde.data_ptr();
    params.kernel2_inv_ptr = kernel2_inv.data_ptr<float>();
    params.step2_ptr = step2.data_ptr();
    params.b_ptr = b_saved.data_ptr();
    params.softmax1_lse_ptr = softmax1_lse.data_ptr<float>();
    params.softmax2_lse_ptr = softmax2_lse.data_ptr<float>();
    params.softmax3_lse_ptr = softmax3_lse.data_ptr<float>();
    params.ns_iterates_ptr = ns_iterates.data_ptr<float>();
    params.k2_softmax_ptr  = k2_softmax.data_ptr<float>();
    params.stream = stream;

    if (dtype == at::ScalarType::Float) {
        run_nystrom_fwd_fp32(params);
    } else {
        run_nystrom_fwd(params);
    }

    return {output, q_s, k_s, q_tilde, k_tilde, kernel2_inv, step2,
            softmax1_lse, softmax2_lse, softmax3_lse, ns_iterates, k2_softmax,
            b_saved, q_assign, k_assign, q_cnt, k_cnt};
}

// -- backward --


std::vector<torch::Tensor> nystrom_bwd(
    torch::Tensor dO,
    torch::Tensor q_s, torch::Tensor k_s,
    torch::Tensor q_tilde, torch::Tensor k_tilde,
    torch::Tensor kernel2_inv, torch::Tensor step2,
    torch::Tensor softmax1_lse, torch::Tensor softmax2_lse, torch::Tensor softmax3_lse,
    torch::Tensor ns_iterates, torch::Tensor k2_softmax,
    torch::Tensor b_saved,
    torch::Tensor v, torch::Tensor output,
    int64_t num_landmarks, int64_t newton_iter,
    bool fast_dk2inv,
    double kappa_star,
    int64_t landmark_mode,
    torch::Tensor q_assign, torch::Tensor k_assign,
    torch::Tensor q_cnt, torch::Tensor k_cnt
) {
    // ------------------------------------------------------------------
    // Input validation. Backward gets called from autograd with saved
    // tensors that the forward produced; in normal use these are well
    // shaped. But this entry point is also reachable directly through
    // the pybind binding, and any drift between forward/backward
    // signatures or any user error would otherwise produce a kernel
    // crash with a CUDA error code rather than a clear Python message.
    // The checks below are O(1) (no tensor reads), so they cost nothing
    // on the hot path.
    // ------------------------------------------------------------------
    CHECK_DEVICE(dO); CHECK_CONTIGUOUS(dO);
    TORCH_CHECK(dO.dim() == 4, "dO must be 4D (B, H, N, D), got ", dO.dim(), "D");

    const auto dtype = dO.scalar_type();
    const int64_t B = dO.size(0), H = dO.size(1), N = dO.size(2), D = dO.size(3);
    const int64_t m = num_landmarks;

    TORCH_CHECK(dtype == at::ScalarType::Float ||
                dtype == at::ScalarType::Half ||
                dtype == at::ScalarType::BFloat16,
                "dO dtype must be float32, float16, or bfloat16; got ", dtype);
    // FP32 D=128 backward: large opt-in SMEM, gated per-kernel (see the matching
    // note in nystrom_fwd). No blanket rejection — capable GPUs run, small ones
    // get a clear per-kernel "insufficient smem" error.
    TORCH_CHECK(D == 64 || D == 128,
                "head_dim must be 64 or 128, got ", D);
    TORCH_CHECK(m > 0 && m <= 64,
                "num_landmarks must be in [1, 64], got ", m);
    TORCH_CHECK(N >= m,
                "seq_len (", N, ") must be >= num_landmarks (", m, ")");
    TORCH_CHECK(newton_iter >= 1 && newton_iter <= 20,
                "newton_iter must be in [1, 20], got ", newton_iter);
    TORCH_CHECK(std::isfinite(kappa_star) && kappa_star >= 0.0,
                "kappa_star must be finite and >= 0 (got ", kappa_star,
                "); must match the forward's kappa_star");
    TORCH_CHECK(B > 0 && H > 0,
                "batch_size and num_heads must be positive (got B=", B, ", H=", H, ")");
    // Per-slice N*D must fit int32 (tile-local indexing); the global
    // B*H*N*D offset is computed in int64 via the int64_t slice index.
    TORCH_CHECK(N * D <= INT32_MAX,
                "per-slice N*D (", N * D, ") exceeds int32 range");

    auto _ck = [&](const torch::Tensor& t, const char* name,
                   torch::IntArrayRef expected_shape, at::ScalarType expected_dtype) {
        CHECK_DEVICE(t);
        CHECK_CONTIGUOUS(t);
        TORCH_CHECK(t.scalar_type() == expected_dtype,
                    name, " must be ", expected_dtype, ", got ", t.scalar_type());
        TORCH_CHECK(t.sizes() == expected_shape,
                    name, " shape mismatch: expected ", expected_shape,
                    ", got ", t.sizes());
    };

    _ck(q_s,           "q_s",           {B, H, N, D},                    dtype);
    _ck(k_s,           "k_s",           {B, H, N, D},                    dtype);
    _ck(v,             "v",             {B, H, N, D},                    dtype);
    _ck(output,        "output",        {B, H, N, D},                    dtype);
    _ck(q_tilde,       "q_tilde",       {B, H, m, D},                    dtype);
    _ck(k_tilde,       "k_tilde",       {B, H, m, D},                    dtype);
    _ck(step2,         "step2",         {B, H, m, D},                    dtype);
    _ck(b_saved,       "b_saved",       {B, H, m, D},                    dtype);
    _ck(kernel2_inv,   "kernel2_inv",   {B, H, m, m},                    at::ScalarType::Float);
    _ck(k2_softmax,    "k2_softmax",    {B, H, m, m},                    at::ScalarType::Float);
    _ck(softmax1_lse,  "softmax1_lse",  {B, H, N},                       at::ScalarType::Float);
    _ck(softmax2_lse,  "softmax2_lse",  {B, H, m},                       at::ScalarType::Float);
    _ck(softmax3_lse,  "softmax3_lse",  {B, H, m},                       at::ScalarType::Float);
    _ck(ns_iterates,   "ns_iterates",   {B, H, newton_iter + 1, m, m},   at::ScalarType::Float);

    const at::cuda::CUDAGuard device_guard(dO.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // The backward runs in the INPUT dtype (fp16 stays fp16, bf16 stays bf16)
    // — see run_nystrom_bwd's FP16_SWITCH. There is no bf16->fp16 cast. The
    // precision-sensitive softmax Jacobians are computed in FP32 inside the
    // kernels (the fix for the large-N STL collapse), so bf16's 7-bit mantissa
    // does not bias the gradient: an FP32 backward matches the pure-PyTorch
    // reference to ~1e-5, and fp16/bf16 track it at their mantissa floor with
    // zero systematic bias (verified by the grad-bias probe and by paired
    // multi-seed training, where FN and the reference are statistically tied).
    //
    // DIAGNOSTIC ONLY (FN_FP32_BWD=1): route the whole 16-bit backward through
    // FP32 to separate precision from the nondeterministic atomicAdd
    // accumulation. Off by default and never a shipping path (the fp32 scalar
    // backward is slow); kept as a debugging escape hatch. D=64 only (the fp32
    // scalar backward overflows SMEM at D=128). 16-bit -> fp32 is lossless.
    const bool to_fp32 = std::getenv("FN_FP32_BWD") && (dtype != at::ScalarType::Float) && (D == 64);
    if (to_fp32) {
        dO      = dO.to(at::kFloat);
        q_s     = q_s.to(at::kFloat);
        k_s     = k_s.to(at::kFloat);
        v       = v.to(at::kFloat);
        q_tilde = q_tilde.to(at::kFloat);
        k_tilde = k_tilde.to(at::kFloat);
        step2   = step2.to(at::kFloat);
        b_saved = b_saved.to(at::kFloat);
        output  = output.to(at::kFloat);
    }

    auto opts = dO.options();  // FP32 when to_fp32, else original dtype
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
    auto D3        = torch::empty({B, H, m}, opts_f32);
    // Split-K workspace for dQ_tilde (kernel3_bwd_tc). num_splits is capped
    // at 64; at the worst-case N=16384 BH=8 m=64 D=128 this is 16 MB. We
    // zero-init since the kernel atomicAdds into split slots (multiple
    // tiles per slot when num_tiles > num_splits).
    constexpr int kNumSplits = 64;
    auto dQ_tilde_split = torch::zeros({kNumSplits, B, H, m, D}, opts_f32);

    // Split-K workspaces for kernel1_bwd's dstep2/dK_tilde accumulators (same
    // scheme as dQ_tilde_split). Without them every row-tile CTA in a (b,h)
    // column atomicAdds the same m*D cells: N/64-way contention per cell.
    // 16 slots cut that 16x while bounding the workspace + zero-init + reduce
    // cost at 2*16*BH*m*D floats. FLASH_NYSTROM_K1_SPLITS overrides the slot
    // count (0 = legacy direct atomicAdd; capped at the row-tile count).
    const int64_t k1_row_tiles = (N + 63) / 64;
    int64_t k1_splits = std::min<int64_t>(16, k1_row_tiles);
    if (const char* env = std::getenv("FLASH_NYSTROM_K1_SPLITS");
        env != nullptr && env[0] != '\0') {
        k1_splits = std::min<int64_t>(std::max(0, std::atoi(env)), k1_row_tiles);
    }
    torch::Tensor k1_split_ws;  // (2, k1_splits, B, H, m, D): dstep2 then dK_tilde
    if (k1_splits > 0) {
        k1_split_ws = torch::zeros({2, k1_splits, B, H, m, D}, opts_f32);
    }

    // No NS backward workspaces here. launch_kernel2_inv_bwd owns its own
    // thread-local persistent workspaces (NsBwdGraphState).

    // dO3 intermediate (always allocated — used by precompute_d3 and the TC
    // kernel3_bwd). Allocated in input dtype: FP16/BF16 for the TC path,
    // FP32 for the FP32 scalar path.
    auto dO3 = torch::empty({B, H, m, D}, opts);

    NystromBwdParams params = {};
    params.batch_size = static_cast<int>(B);
    params.num_heads = static_cast<int>(H);
    params.seq_len = static_cast<int>(N);
    params.head_dim = static_cast<int>(D);
    params.num_landmarks = static_cast<int>(m);
    params.newton_iter = static_cast<int>(newton_iter);
    params.is_bf16 = (dtype == at::ScalarType::BFloat16);
    params.fast_dk2inv = fast_dk2inv;
    params.kappa_star = static_cast<float>(kappa_star);

    params.q_s_ptr = q_s.data_ptr();
    params.k_s_ptr = k_s.data_ptr();
    params.v_ptr = v.data_ptr();
    params.q_tilde_ptr = q_tilde.data_ptr();
    params.k_tilde_ptr = k_tilde.data_ptr();
    params.k2_inv_ptr = kernel2_inv.data_ptr<float>();
    params.step2_ptr = step2.data_ptr();
    params.b_ptr = b_saved.data_ptr();
    params.o_ptr = output.data_ptr();
    params.lse1_ptr = softmax1_lse.data_ptr<float>();
    params.lse2_ptr = softmax2_lse.data_ptr<float>();
    params.lse3_ptr = softmax3_lse.data_ptr<float>();
    params.ns_iterates_ptr = ns_iterates.data_ptr<float>();
    params.k2_softmax_ptr  = k2_softmax.data_ptr<float>();

    params.dO_ptr = dO.data_ptr();
    params.dQ_ptr = dQ.data_ptr();
    params.dK_ptr = dK.data_ptr();
    params.dV_ptr = dV.data_ptr();

    params.dstep2_ptr = dstep2.data_ptr<float>();
    params.dQ_tilde_ptr = dQ_tilde.data_ptr<float>();
    params.dK_tilde_ptr = dK_tilde.data_ptr<float>();
    params.dK2_inv_ptr = dK2_inv.data_ptr<float>();
    params.dQ_tilde_split_ptr = dQ_tilde_split.data_ptr<float>();
    params.num_splits = kNumSplits;
    if (k1_splits > 0) {
        const int64_t k1_slot_elems = k1_splits * B * H * m * D;
        params.dstep2_split_ptr   = k1_split_ws.data_ptr<float>();
        params.dK_tilde_split_ptr = k1_split_ws.data_ptr<float>() + k1_slot_elems;
    } else {
        params.dstep2_split_ptr   = nullptr;
        params.dK_tilde_split_ptr = nullptr;
    }
    params.k1_num_splits = static_cast<int>(k1_splits);
    params.D1_ptr = D1.data_ptr<float>();
    params.D3_ptr = D3.data_ptr<float>();
    params.dO3_ptr = dO3.data_ptr();
    params.landmark_mode = static_cast<int>(landmark_mode);
    if (landmark_mode == 1) {
        TORCH_CHECK(q_assign.numel() > 0 && k_assign.numel() > 0 &&
                    q_cnt.numel() > 0 && k_cnt.numel() > 0,
                    "landmark_mode 1 backward requires the forward's assignment/"
                    "count tensors (q_assign, k_assign, q_cnt, k_cnt)");
        params.q_assign_ptr = q_assign.data_ptr<int>();
        params.k_assign_ptr = k_assign.data_ptr<int>();
        params.q_cnt_ptr = q_cnt.data_ptr<int>();
        params.k_cnt_ptr = k_cnt.data_ptr<int>();
    }
    params.stream = stream;

    if (dtype == at::ScalarType::Float || to_fp32) {
        run_nystrom_bwd_fp32(params);
    } else {
        run_nystrom_bwd(params);
    }

    // Env-gated kernel-boundary instrumentation (set FN_BWD_DEBUG=1). Logs the
    // dynamic grad-scale, max|dO|, and finiteness/abs-max of the kernel's own
    // outputs and intermediates -- visibility Python hooks cannot get. The
    // .item() calls force syncs, so this is only for diagnosis, never the hot
    // path. Use to locate where (which quantity) a collapse first goes bad.
    if (std::getenv("FN_BWD_DEBUG")) {
        auto fin = [](const torch::Tensor& t) { return torch::isfinite(t).all().item<bool>(); };
        auto amx = [](const torch::Tensor& t) { return t.abs().max().item<double>(); };
        fprintf(stderr,
            "[fn_bwd] N=%lld m=%lld to_fp32=%d max|dO|=%.3e | "
            "dQ{fin=%d max=%.3e} dV{fin=%d max=%.3e} dO3{fin=%d max=%.3e} "
            "dstep2{fin=%d max=%.3e} dQt{fin=%d max=%.3e} dK2inv{fin=%d max=%.3e}\n",
            (long long)N, (long long)m, (int)to_fp32, amx(dO),
            (int)fin(dQ), amx(dQ), (int)fin(dV), amx(dV),
            (int)fin(dO3), amx(dO3), (int)fin(dstep2), amx(dstep2),
            (int)fin(dQ_tilde), amx(dQ_tilde), (int)fin(dK2_inv), amx(dK2_inv));
    }

    if (to_fp32) {
        // Hand back grads in the model's original dtype (fp16/bf16).
        return {dQ.to(dtype), dK.to(dtype), dV.to(dtype)};
    }
    return {dQ, dK, dV};
}

// ===== Debug entry points for kernel2_inv backward isolation tests =====
// These call the NS bwd kernels directly with FP32 inputs, no autograd path,
// no parameter struct. Used by tests/test_ns_bwd_kernel.py to verify the
// kernels match PyTorch element-wise.

std::vector<torch::Tensor> debug_ns_bwd_step(
    torch::Tensor K2,        // (BH, m, m) FP32 — softmax K2
    torch::Tensor Z_j,       // (BH, m, m) FP32 — Z_j iterate
    torch::Tensor dZ_in      // (BH, m, m) FP32 — dZ_{j+1}
) {
    CHECK_DEVICE(K2); CHECK_DEVICE(Z_j); CHECK_DEVICE(dZ_in);
    CHECK_CONTIGUOUS(K2); CHECK_CONTIGUOUS(Z_j); CHECK_CONTIGUOUS(dZ_in);
    TORCH_CHECK(K2.dtype() == torch::kFloat32, "K2 must be FP32");
    TORCH_CHECK(Z_j.dtype() == torch::kFloat32, "Z_j must be FP32");
    TORCH_CHECK(dZ_in.dtype() == torch::kFloat32, "dZ_in must be FP32");
    TORCH_CHECK(K2.dim() == 3, "K2 must be (BH, m, m)");
    TORCH_CHECK(K2.sizes() == Z_j.sizes() && K2.sizes() == dZ_in.sizes(),
                "K2, Z_j, dZ_in must all have shape (BH, m, m)");
    TORCH_CHECK(K2.size(1) == K2.size(2), "K2 must be square in last two dims");

    const int BH = static_cast<int>(K2.size(0));
    const int m  = static_cast<int>(K2.size(1));

    const at::cuda::CUDAGuard device_guard(K2.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto opts_f32 = K2.options();
    auto dZ_out  = torch::empty({BH, m, m}, opts_f32);
    auto dK2_acc = torch::zeros({BH, m, m}, opts_f32);

    flash_nystrom::launch_ns_bwd_step_test(
        K2.data_ptr<float>(),
        Z_j.data_ptr<float>(),
        dZ_in.data_ptr<float>(),
        dZ_out.data_ptr<float>(),
        dK2_acc.data_ptr<float>(),
        BH, m, stream);

    return {dZ_out, dK2_acc};
}

// CP1 debug hook: batched tf32 tensor-core GEMM C = A @ B (m x m x m).
torch::Tensor debug_k2inv_gemm_nn(torch::Tensor A, torch::Tensor B) {
    CHECK_DEVICE(A); CHECK_DEVICE(B);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B);
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "A,B must be FP32");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "A,B must be (BH, m, m)");
    TORCH_CHECK(A.sizes() == B.sizes() && A.size(1) == A.size(2), "A,B must be (BH,m,m) square");
    const int BH = static_cast<int>(A.size(0)), m = static_cast<int>(A.size(1));
    const at::cuda::CUDAGuard device_guard(A.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    auto C = torch::empty({BH, m, m}, A.options());
    flash_nystrom::launch_k2inv_gemm_nn(A.data_ptr<float>(), B.data_ptr<float>(),
                                        C.data_ptr<float>(), BH, m, stream);
    return C;
}

std::vector<torch::Tensor> debug_ns_bwd_final(
    torch::Tensor q_tilde,   // (BH, m, D) FP32
    torch::Tensor k_tilde,   // (BH, m, D) FP32
    torch::Tensor K2,        // (BH, m, m) FP32 — softmax K2
    torch::Tensor dZ0,       // (BH, m, m) FP32 — dZ_0 from NS unroll
    torch::Tensor dK2_in     // (BH, m, m) FP32 — dK2 accumulator from NS unroll
) {
    CHECK_DEVICE(q_tilde); CHECK_DEVICE(k_tilde);
    CHECK_DEVICE(K2); CHECK_DEVICE(dZ0); CHECK_DEVICE(dK2_in);
    CHECK_CONTIGUOUS(q_tilde); CHECK_CONTIGUOUS(k_tilde);
    CHECK_CONTIGUOUS(K2); CHECK_CONTIGUOUS(dZ0); CHECK_CONTIGUOUS(dK2_in);
    TORCH_CHECK(q_tilde.dtype() == torch::kFloat32, "q_tilde must be FP32");
    TORCH_CHECK(k_tilde.dtype() == torch::kFloat32, "k_tilde must be FP32");
    TORCH_CHECK(K2.dtype() == torch::kFloat32, "K2 must be FP32");
    TORCH_CHECK(dZ0.dtype() == torch::kFloat32, "dZ0 must be FP32");
    TORCH_CHECK(dK2_in.dtype() == torch::kFloat32, "dK2_in must be FP32");

    const int BH = static_cast<int>(K2.size(0));
    const int m  = static_cast<int>(K2.size(1));
    const int D  = static_cast<int>(q_tilde.size(2));

    const at::cuda::CUDAGuard device_guard(K2.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto opts_f32 = K2.options();
    auto dQ_tilde_out = torch::zeros({BH, m, D}, opts_f32);
    auto dK_tilde_out = torch::zeros({BH, m, D}, opts_f32);
    auto dK2_inout    = dK2_in.clone();  // kernel modifies in place (adds dZ0^T/c)

    flash_nystrom::launch_ns_bwd_final_test(
        q_tilde.data_ptr<float>(),
        k_tilde.data_ptr<float>(),
        K2.data_ptr<float>(),
        dZ0.data_ptr<float>(),
        dK2_inout.data_ptr<float>(),
        dQ_tilde_out.data_ptr<float>(),
        dK_tilde_out.data_ptr<float>(),
        BH, D, m, stream);

    return {dQ_tilde_out, dK_tilde_out, dK2_inout};
}

// Debug hook for the FULL launch_kernel2_inv_bwd. Drives the production
// orchestration (per-iter loop + final softmax-bwd step) with explicit FP32
// inputs. Compares against the PyTorch autograd-through-Newton-Schulz reference
// in tests/test_ns_bwd_kernel.py.
std::vector<torch::Tensor> debug_kernel2_inv_bwd_full(
    torch::Tensor q_tilde,    // (BH, m, D) FP32
    torch::Tensor k_tilde,    // (BH, m, D) FP32
    torch::Tensor K2_softmax, // (BH, m, m) FP32 — softmax(QK^T) output
    torch::Tensor ns_iterates,// (BH, niter+1, m, m) FP32 — Z_0 .. Z_N from forward
    torch::Tensor dK2_inv_in, // (BH, m, m) FP32 — gradient w.r.t. Z_N
    int64_t newton_iter,
    double kappa_star
) {
    CHECK_DEVICE(q_tilde); CHECK_DEVICE(k_tilde); CHECK_DEVICE(K2_softmax);
    CHECK_DEVICE(ns_iterates); CHECK_DEVICE(dK2_inv_in);
    CHECK_CONTIGUOUS(q_tilde); CHECK_CONTIGUOUS(k_tilde); CHECK_CONTIGUOUS(K2_softmax);
    CHECK_CONTIGUOUS(ns_iterates); CHECK_CONTIGUOUS(dK2_inv_in);
    TORCH_CHECK(q_tilde.dtype() == torch::kFloat32, "q_tilde must be FP32");
    TORCH_CHECK(k_tilde.dtype() == torch::kFloat32, "k_tilde must be FP32");
    TORCH_CHECK(K2_softmax.dtype() == torch::kFloat32, "K2_softmax must be FP32");
    TORCH_CHECK(ns_iterates.dtype() == torch::kFloat32, "ns_iterates must be FP32");
    TORCH_CHECK(dK2_inv_in.dtype() == torch::kFloat32, "dK2_inv_in must be FP32");

    const int BH = static_cast<int>(q_tilde.size(0));
    const int m  = static_cast<int>(q_tilde.size(1));
    const int D  = static_cast<int>(q_tilde.size(2));
    TORCH_CHECK(K2_softmax.size(0) == BH && K2_softmax.size(1) == m && K2_softmax.size(2) == m,
                "K2_softmax shape mismatch");
    TORCH_CHECK(ns_iterates.dim() == 4 && ns_iterates.size(0) == BH &&
                ns_iterates.size(1) == newton_iter + 1 &&
                ns_iterates.size(2) == m && ns_iterates.size(3) == m,
                "ns_iterates must have shape (BH, newton_iter+1, m, m)");
    TORCH_CHECK(dK2_inv_in.size(0) == BH && dK2_inv_in.size(1) == m && dK2_inv_in.size(2) == m,
                "dK2_inv_in shape mismatch");

    const at::cuda::CUDAGuard device_guard(q_tilde.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto opts_f32 = q_tilde.options();
    auto dQ_tilde = torch::zeros({BH, m, D}, opts_f32);
    auto dK_tilde = torch::zeros({BH, m, D}, opts_f32);

    flash_nystrom::launch_kernel2_inv_bwd<float>(
        q_tilde.data_ptr<float>(), k_tilde.data_ptr<float>(),
        dK2_inv_in.data_ptr<float>(),
        ns_iterates.data_ptr<float>(),
        K2_softmax.data_ptr<float>(),
        dQ_tilde.data_ptr<float>(), dK_tilde.data_ptr<float>(),
        BH, D, m, static_cast<int>(newton_iter), stream,
        static_cast<float>(kappa_star));

    return {dQ_tilde, dK_tilde};
}

// Debug hook for compute_dk2inv: drives the kernel directly with FP32 inputs
// (BH, m, D)/(BH, N, D). Returns (dK2_inv, D3) FP32.
std::vector<torch::Tensor> debug_compute_dk2inv(
    torch::Tensor q_tilde, // (BH, m, D) FP32
    torch::Tensor k_s,     // (BH, N, D) FP32
    torch::Tensor v,       // (BH, N, D) FP32
    torch::Tensor dO3,     // (BH, m, D) FP32
    torch::Tensor lse3,    // (BH, m)   FP32
    torch::Tensor dstep2   // (BH, m, D) FP32
) {
    CHECK_DEVICE(q_tilde); CHECK_DEVICE(k_s); CHECK_DEVICE(v);
    CHECK_DEVICE(dO3); CHECK_DEVICE(lse3); CHECK_DEVICE(dstep2);
    CHECK_CONTIGUOUS(q_tilde); CHECK_CONTIGUOUS(k_s); CHECK_CONTIGUOUS(v);
    CHECK_CONTIGUOUS(dO3); CHECK_CONTIGUOUS(lse3); CHECK_CONTIGUOUS(dstep2);
    TORCH_CHECK(q_tilde.dtype() == torch::kFloat32, "q_tilde must be FP32");
    TORCH_CHECK(k_s.dtype() == torch::kFloat32, "k_s must be FP32");
    TORCH_CHECK(v.dtype() == torch::kFloat32, "v must be FP32");
    TORCH_CHECK(dO3.dtype() == torch::kFloat32, "dO3 must be FP32");
    TORCH_CHECK(lse3.dtype() == torch::kFloat32, "lse3 must be FP32");
    TORCH_CHECK(dstep2.dtype() == torch::kFloat32, "dstep2 must be FP32");
    TORCH_CHECK(q_tilde.dim() == 3, "q_tilde must be (BH, m, D)");
    TORCH_CHECK(k_s.dim() == 3 && v.dim() == 3, "k_s, v must be (BH, N, D)");

    const int BH = static_cast<int>(q_tilde.size(0));
    const int m  = static_cast<int>(q_tilde.size(1));
    const int D  = static_cast<int>(q_tilde.size(2));
    const int N  = static_cast<int>(k_s.size(1));
    TORCH_CHECK(k_s.size(0) == BH && v.size(0) == BH, "BH mismatch");
    TORCH_CHECK(k_s.size(2) == D && v.size(2) == D, "D mismatch");
    TORCH_CHECK(dO3.sizes() == q_tilde.sizes(), "dO3 shape mismatch");
    TORCH_CHECK(lse3.size(0) == BH && lse3.size(1) == m, "lse3 shape mismatch");
    TORCH_CHECK(dstep2.sizes() == q_tilde.sizes(), "dstep2 shape mismatch");

    const at::cuda::CUDAGuard device_guard(q_tilde.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto opts_f32 = q_tilde.options();
    auto dK2_inv = torch::empty({BH, m, m}, opts_f32);
    auto D3      = torch::empty({BH, m},    opts_f32);

    // Debug hook always exercises the scalar path (FP32 input dtype + the
    // TC kernel only runs for FP16/BF16 anyway).
    flash_nystrom::launch_compute_dk2inv<float>(
        q_tilde.data_ptr<float>(),
        k_s.data_ptr<float>(),
        v.data_ptr<float>(),
        /*b=*/nullptr,                 // debug hook always re-walks N (legacy path)
        dO3.data_ptr<float>(),
        lse3.data_ptr<float>(),
        dstep2.data_ptr<float>(),
        dK2_inv.data_ptr<float>(),
        D3.data_ptr<float>(),
        BH, N, D, m, /*fast_dk2inv=*/false, stream);

    return {dK2_inv, D3};
}

// Debug/standalone entry point for the leverage-seeded Voronoi-mean landmark
// selector. Takes x = (BH, N, D) and returns x_tilde = (BH, m, D). Not wired
// into the fused forward yet; used by tests/test_leverage_landmarks.py to
// validate against a CPU ridge-leverage reference before it replaces the
// segment-mean landmark path.
torch::Tensor debug_leverage_landmarks(
    torch::Tensor x, int64_t m, int64_t seed, int64_t subsample, double scale) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be (BH, N, D)");
    const int BH = (int)x.size(0), N = (int)x.size(1), D = (int)x.size(2);
    TORCH_CHECK(D == 64 || D == 128, "head_dim must be 64 or 128");
    TORCH_CHECK(m >= 1 && m <= LM_TOPM_MAX, "m out of range [1, ", LM_TOPM_MAX, "]");

    const at::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_tilde = torch::empty({(int64_t)BH, (int64_t)m, (int64_t)D}, x.options());
    const size_t ws_bytes = flash_nystrom::lm_workspace_bytes(BH, N, D, (int)m);
    auto ws = torch::empty({(int64_t)ws_bytes}, x.options().dtype(torch::kUInt8));

    auto launch = [&](auto* tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const T* xp = static_cast<const T*>(x.data_ptr());
        T* op = static_cast<T*>(x_tilde.data_ptr());
        if (D == 64)
            flash_nystrom::launch_rls_vmean_landmarks<T, 64>(
                xp, op, BH, N, (int)m, (float)scale, ws.data_ptr(), ws_bytes,
                (uint64_t)seed, (int)subsample, stream);
        else
            flash_nystrom::launch_rls_vmean_landmarks<T, 128>(
                xp, op, BH, N, (int)m, (float)scale, ws.data_ptr(), ws_bytes,
                (uint64_t)seed, (int)subsample, stream);
    };

    switch (x.scalar_type()) {
        case torch::kFloat16:  launch((cutlass::half_t*)nullptr); break;
        case torch::kBFloat16: launch((cutlass::bfloat16_t*)nullptr); break;
        case torch::kFloat32:  launch((float*)nullptr); break;
        default: TORCH_CHECK(false, "unsupported dtype for leverage landmarks");
    }
    return x_tilde;
}

} // namespace flash_nystrom

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_nystrom::nystrom_fwd,
          "FlashNystrom forward (CUDA). The depthwise-conv residual is computed "
          "at the Python level via cuDNN (F.conv1d) and is not part of this entry "
          "point. See flash_nystrom.flash_nystrom_attention.",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("num_landmarks") = 64,
          py::arg("newton_iter") = 6,
          py::arg("kappa_star") = 0.0,
          py::arg("use_tc_pinv") = false,
          py::arg("landmark_mode") = 0,
          py::arg("landmark_seed") = 0,
          py::arg("landmark_subsample") = 1,
          py::arg("landmark_gumbel_scale") = 1.0,
          py::arg("landmark_force_first") = 0);
    m.def("backward", &flash_nystrom::nystrom_bwd,
          "FlashNystrom backward (CUDA). Pass b_saved = softmax(Q_tilde @ K^T) "
          "@ V from the forward to skip the N-walk in compute_dk2inv. "
          "fast_dk2inv only matters when b_saved is empty.",
          py::arg("dO"),
          py::arg("q_s"), py::arg("k_s"),
          py::arg("q_tilde"), py::arg("k_tilde"),
          py::arg("kernel2_inv"), py::arg("step2"),
          py::arg("softmax1_lse"), py::arg("softmax2_lse"), py::arg("softmax3_lse"),
          py::arg("ns_iterates"), py::arg("k2_softmax"),
          py::arg("b_saved"),
          py::arg("v"), py::arg("output"),
          py::arg("num_landmarks"), py::arg("newton_iter"),
          py::arg("fast_dk2inv") = true,
          py::arg("kappa_star") = 0.0,
          py::arg("landmark_mode") = 0,
          py::arg("q_assign") = torch::Tensor(),
          py::arg("k_assign") = torch::Tensor(),
          py::arg("q_cnt") = torch::Tensor(),
          py::arg("k_cnt") = torch::Tensor());
    m.def("debug_k2inv_gemm_nn", &flash_nystrom::debug_k2inv_gemm_nn,
          "Debug: batched tf32 tensor-core GEMM C = A @ B (m x m x m).",
          py::arg("A"), py::arg("B"));
    m.def("debug_ns_bwd_step", &flash_nystrom::debug_ns_bwd_step,
          "Debug: single NS backward iteration (returns dZ_j, dK2_contrib).",
          py::arg("K2"), py::arg("Z_j"), py::arg("dZ_in"));
    m.def("debug_ns_bwd_final", &flash_nystrom::debug_ns_bwd_final,
          "Debug: NS backward final step (Z_0 init grad + softmax bwd). "
          "Returns (dQ_tilde, dK_tilde, dK2_after_init).",
          py::arg("q_tilde"), py::arg("k_tilde"), py::arg("K2"),
          py::arg("dZ0"), py::arg("dK2_in"));
    m.def("debug_compute_dk2inv", &flash_nystrom::debug_compute_dk2inv,
          "Debug: fused compute_dk2inv. Walks N once and returns "
          "(dK2_inv, D3) where dK2_inv = dstep2 @ B^T and D3 = diag(B @ dO3^T), "
          "B = softmax(Qt @ Ks^T) @ V.",
          py::arg("q_tilde"), py::arg("k_s"), py::arg("v"), py::arg("dO3"),
          py::arg("lse3"), py::arg("dstep2"));
    m.def("debug_kernel2_inv_bwd_full", &flash_nystrom::debug_kernel2_inv_bwd_full,
          "Debug: full launch_kernel2_inv_bwd (per-iter loop + final softmax bwd). "
          "Returns (dQ_tilde, dK_tilde) FP32.",
          py::arg("q_tilde"), py::arg("k_tilde"), py::arg("K2_softmax"),
          py::arg("ns_iterates"), py::arg("dK2_inv_in"), py::arg("newton_iter"),
          py::arg("kappa_star") = 0.0);
    m.def("debug_leverage_landmarks", &flash_nystrom::debug_leverage_landmarks,
          "Leverage-seeded Voronoi-mean landmark selection. x=(BH,N,D) -> "
          "x_tilde=(BH,m,D). Deterministic for a fixed seed. subsample>1 thins "
          "the assign pass (systematic tile sampling). Standalone; not yet in "
          "the fused forward.",
          py::arg("x"), py::arg("m") = 64, py::arg("seed") = 0,
          py::arg("subsample") = 1, py::arg("scale") = 1.0);
    m.def("reset_caches", []() {
              flash_nystrom::reset_ns_bwd_caches();
              flash_nystrom::reset_kernel3_caches();
              flash_nystrom::reset_k2inv_tc_caches();
          },
          "Free ALL three thread-local GPU caches held by FlashNystrom on the "
          "calling thread: the NS-backward graph/workspaces, the kernel3 split-N "
          "scratch, and the TC-pinv forward graph. Each cache holds one shape's "
          "worth of CUDA graphs + FP32 workspaces per thread; they reallocate on "
          "shape change (no per-shape growth) but persist for the thread's "
          "lifetime. In a long-lived multi-threaded server, call this per worker "
          "between requests (or on shape change) to reclaim the memory. Safe "
          "between calls; must NOT be called during a graph capture.");

    // ----- Occupancy probe -----
    py::class_<flash_nystrom::OccupancyRow>(m, "OccupancyRow")
        .def_readonly("kernel_name",        &flash_nystrom::OccupancyRow::kernel_name)
        .def_readonly("threads_per_block",  &flash_nystrom::OccupancyRow::threads_per_block)
        .def_readonly("dynamic_smem_bytes", &flash_nystrom::OccupancyRow::dynamic_smem_bytes)
        .def_readonly("regs_per_thread",    &flash_nystrom::OccupancyRow::regs_per_thread)
        .def_readonly("static_smem_bytes",  &flash_nystrom::OccupancyRow::static_smem_bytes)
        .def_readonly("max_blocks_per_sm",  &flash_nystrom::OccupancyRow::max_blocks_per_sm)
        .def_readonly("max_warps_per_sm",   &flash_nystrom::OccupancyRow::max_warps_per_sm)
        .def_readonly("regs_per_block",     &flash_nystrom::OccupancyRow::regs_per_block)
        .def_readonly("total_smem_per_block", &flash_nystrom::OccupancyRow::total_smem_per_block)
        .def_readonly("blocks_by_threads",  &flash_nystrom::OccupancyRow::blocks_by_threads)
        .def_readonly("blocks_by_regs",     &flash_nystrom::OccupancyRow::blocks_by_regs)
        .def_readonly("blocks_by_smem",     &flash_nystrom::OccupancyRow::blocks_by_smem)
        .def_readonly("blocks_by_hardware", &flash_nystrom::OccupancyRow::blocks_by_hardware)
        .def_readonly("binding_constraint", &flash_nystrom::OccupancyRow::binding_constraint);

    m.def("probe_occupancy",
          &flash_nystrom::probe_all,
          "Run cudaOccupancyMaxActiveBlocksPerMultiprocessor on every kernel "
          "this build ships, at the given (m, D, newton_iter) launch config.",
          py::arg("m") = 64, py::arg("D") = 128, py::arg("newton_iter") = 6,
          py::arg("dtype") = "half");

    py::class_<flash_nystrom::SmLimits>(m, "SmLimits")
        .def_readonly("max_threads_per_sm",     &flash_nystrom::SmLimits::max_threads_per_sm)
        .def_readonly("max_blocks_per_sm",      &flash_nystrom::SmLimits::max_blocks_per_sm)
        .def_readonly("max_regs_per_sm",        &flash_nystrom::SmLimits::max_regs_per_sm)
        .def_readonly("max_smem_per_sm_bytes",  &flash_nystrom::SmLimits::max_smem_per_sm_bytes)
        .def_readonly("reg_alloc_unit",         &flash_nystrom::SmLimits::reg_alloc_unit)
        .def_readonly("warp_alloc_unit",        &flash_nystrom::SmLimits::warp_alloc_unit);

    m.def("query_sm_limits", &flash_nystrom::query_sm_limits,
          "Return the per-SM hardware limits for the current device.");
}
