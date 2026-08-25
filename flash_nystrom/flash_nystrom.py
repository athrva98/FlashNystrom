# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_nystrom.nystrom_config import NystromConfig

try:
    import flash_nystrom._C as _C

    HAS_CUDA = True
except ImportError:
    _C = None
    HAS_CUDA = False


def _depthwise_conv_residual(v, weight):
    """Depthwise 1D conv along the sequence dim, dispatched to cuDNN via F.conv1d.

    For each head h, applies the same 1D kernel weight[h, :] independently to
    every D channel. Equivalent to the original Nystromformer's conv residual.

    v:       (B, H, N, D), float dtype (FP16/BF16/FP32)
    weight:  (H, kernel_size), any float dtype (cast to match v)
    returns: (B, H, N, D), same dtype as v

    Replaces the hand-written `dconv_residual.cuh` kernel. Using cuDNN gives:
      - battle-tested numerical stability (custom kernel had FP16 NaN issues)
      - optimal performance for depthwise 1D conv
      - autograd handled by PyTorch (no manual backward)
    """
    B, H, N, D = v.shape
    ks = weight.shape[1]
    pad = ks // 2

    # Match dtypes (weight may be FP32 nn.Parameter, v may be FP16 under autocast)
    if weight.dtype != v.dtype:
        weight = weight.to(v.dtype)

    # (B, H, N, D) -> (B, H, D, N) -> (B, H*D, N)
    # Each (head, dim) pair becomes its own depthwise channel.
    v_in = v.permute(0, 1, 3, 2).contiguous().view(B, H * D, N)

    # weight (H, ks) -> (H, D, ks) -> (H*D, 1, ks)
    # Same weight applied to all D channels within a head.
    w = weight.unsqueeze(1).expand(H, D, ks).contiguous().view(H * D, 1, ks)

    # groups = H*D makes this a true depthwise conv (each output channel reads
    # only its corresponding input channel).
    out = F.conv1d(v_in, w, padding=pad, groups=H * D)  # (B, H*D, N)

    # (B, H*D, N) -> (B, H, D, N) -> (B, H, N, D)
    return out.view(B, H, D, N).permute(0, 1, 3, 2).contiguous()


class FlashNystromFunction(torch.autograd.Function):
    """Autograd wrapper around the pure Nystrom-attention C extension.

    The depthwise-conv residual is NOT part of this Function; it is added at
    the Python level via cuDNN (F.conv1d) by `flash_nystrom_attention` so its
    backward is handled by PyTorch automatically. Keeping conv out of the
    autograd Function lets the C extension boundary stay narrow and lets us
    drop the custom dconv CUDA kernels.
    """

    @staticmethod
    def forward(ctx, q, k, v, num_landmarks, newton_iter, fast_dk2inv,
                kappa_star, use_tc_pinv):
        assert _C is not None, "CUDA extension not available"

        # The scaled Q/K copies exist only for the backward; producing them
        # costs a read+write of both tensors, 25% of the forward at N=1M. Grad
        # mode is disabled inside Function.forward, so this cannot be inferred
        # in C++ -- but the inputs keep their requires_grad flags here.
        need_scaled_qk = bool(q.requires_grad or k.requires_grad
                              or v.requires_grad)
        ctx.need_scaled_qk = need_scaled_qk
        results = _C.forward(q, k, v, num_landmarks, newton_iter,
                             kappa_star, use_tc_pinv, need_scaled_qk)
        # results: [output, q_s, k_s, q_tilde, k_tilde, k2inv, step2,
        #           lse1, lse2, lse3, ns_iterates, k2_softmax, b_saved]
        # b_saved = softmax(Q_tilde @ K_s^T) @ V (K_s = scaled K) is reused in
        # the backward so compute_dk2inv skips an O(m*N*D) recomputation pass.
        output = results[0]

        saved = list(results[1:])  # q_s, k_s, qt, kt, k2inv, step2, lse1, lse2, lse3, ns_iter, k2sm, b_saved
        saved.append(v)
        saved.append(output)
        ctx.save_for_backward(*saved)
        ctx.num_landmarks = num_landmarks
        ctx.newton_iter = newton_iter
        ctx.fast_dk2inv = fast_dk2inv
        ctx.kappa_star = kappa_star

        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        q_s, k_s, q_tilde, k_tilde, k2_inv, step2 = saved[0:6]
        lse1, lse2, lse3 = saved[6:9]
        ns_iterates = saved[9]
        k2_softmax = saved[10]
        b_saved = saved[11]
        v = saved[12]
        output = saved[13]

        dQ, dK, dV = _C.backward(
            grad_output.contiguous(),
            q_s,
            k_s,
            q_tilde,
            k_tilde,
            k2_inv,
            step2,
            lse1,
            lse2,
            lse3,
            ns_iterates,
            k2_softmax,
            b_saved,
            v,
            output,
            ctx.num_landmarks,
            ctx.newton_iter,
            ctx.fast_dk2inv,
            ctx.kappa_star,
        )

        # apply() args were (q, k, v, num_landmarks, newton_iter, fast_dk2inv,
        # kappa_star, use_tc_pinv) → eight grad outputs; only q/k/v differentiable.
        return dQ, dK, dV, None, None, None, None, None


# Landmark range gating constants. The custom CUDA kernels only handle m <= 64
# (kernel tile-size limit). For m > 64 we currently dispatch to the pure-PyTorch
# reference, which materializes the (N, m) softmax intermediates — see the
# comment in flash_nystrom_attention below for the trade-off and the memory
# guard. The plan is to replace this reference dispatch with custom kernels,
# kernel-by-kernel, as each m-agnostic implementation lands and is verified.
_M_CUSTOM_KERNEL_MAX = 64

# Hard ceiling on the bytes the reference path is allowed to materialize for
# its two (N, m) softmax intermediates in FP16/BF16. 8 GiB is a defensible
# default that fits comfortably on consumer GPUs. Tunable via the env var
# FLASH_NYSTROM_REFERENCE_MAX_BYTES (in bytes).
_DEFAULT_REFERENCE_BYTE_BUDGET = 8 * (1024 ** 3)


def _reference_softmax_bytes(q_shape, num_landmarks):
    # Two (B, H, N, m) FP16/BF16 matrices: softmax(Q@Kt^T) for kernel1 and
    # softmax(Qt@K^T) for kernel3. Both live concurrently in the autograd
    # graph for the duration of the backward pass.
    B, H, N, _ = q_shape
    return 2 * 2 * B * H * N * int(num_landmarks)


def _check_reference_budget(q, num_landmarks):
    import os
    budget = int(os.environ.get("FLASH_NYSTROM_REFERENCE_MAX_BYTES",
                                _DEFAULT_REFERENCE_BYTE_BUDGET))
    need = _reference_softmax_bytes(q.shape, num_landmarks)
    if need > budget:
        raise RuntimeError(
            f"flash_nystrom_attention at num_landmarks={num_landmarks} routes to "
            f"the pure-PyTorch reference (custom kernels currently support "
            f"num_landmarks <= {_M_CUSTOM_KERNEL_MAX} only). The reference "
            f"materializes ~{need / (1024**3):.2f} GiB of softmax intermediates "
            f"for shape B={q.shape[0]} H={q.shape[1]} N={q.shape[2]} "
            f"m={num_landmarks}, which exceeds the "
            f"FLASH_NYSTROM_REFERENCE_MAX_BYTES budget of "
            f"~{budget / (1024**3):.2f} GiB. Drop num_landmarks <= "
            f"{_M_CUSTOM_KERNEL_MAX} for the custom path, drop N or BH, or "
            f"raise the budget via the env var if you have the memory."
        )


def flash_nystrom_attention(
    q, k, v, num_landmarks=64, newton_iter=6, conv_weight=None, conv_kernel_size=0,
    fast_dk2inv=True, kappa_star=0.0, use_tc_pinv=False,
):
    """main entry point — uses CUDA kernels if available, falls back to pytorch.

    The conv residual (when enabled) is computed via cuDNN through F.conv1d,
    OUTSIDE the FlashNystromFunction autograd boundary. PyTorch handles its
    backward automatically. The previous custom CUDA conv kernels
    (dconv_residual.cuh / dconv_residual_bwd.cuh) have been removed; only
    the cuDNN path exists now.

    `num_landmarks` (m) range:
      * m <= 64: custom CUDA forward + backward. Memory is O(B*H*(N+m)*D) —
        no (N, m) softmax materialization. This is the FlashNystrom fast
        path the library exists for.
      * m > 64:  dispatched to the pure-PyTorch reference
        (nystrom_attention_reference). Each matmul lowers to cuBLAS via the
        `@` operator and PyTorch autograd handles the backward. The reference
        materializes (N, m) softmax matrices — see _check_reference_budget
        for the OOM guard. The intent is to gradually replace this dispatch
        with custom kernels (kernel-by-kernel, with tests per kernel) as each
        m-agnostic implementation lands.

    `fast_dk2inv` controls the `compute_dk2inv` path in the backward (FP16/BF16
    only). Default True uses the tensor-core kernel (4-6x faster on the full
    bwd). Set False to use the FP32 scalar fallback, which is bit-for-bit
    consistent with the autograd reference. The TC path converts the softmax
    output P from FP32 to FP16/BF16 before GEMM2, trimming P to a 10-bit
    mantissa — small accuracy cost, typically below FP16 training noise.
    `fast_dk2inv` is ignored when num_landmarks > 64 (the reference path
    doesn't use the custom dk2inv kernel).

    `kappa_star` is the Tikhonov ridge target condition number, threaded
    identically to the kernel and the reference dispatch so both compute the
    same regularized pseudoinverse (0 = no ridge). `use_tc_pinv` routes the
    pinv through the tf32 tensor-core path (m == 64 only).
    """
    if HAS_CUDA and q.is_cuda:
        if num_landmarks > _M_CUSTOM_KERNEL_MAX:
            # m > 64 -> reference. Memory check first so the user sees a
            # clear Python error before any allocation, not a CUDA OOM.
            # kappa_star is passed through so the reference matches whatever the
            # custom path would have used (consistency, not a separate default).
            _check_reference_budget(q, num_landmarks)
            from flash_nystrom.reference import nystrom_attention_reference
            return nystrom_attention_reference(
                q, k, v, num_landmarks, newton_iter,
                conv_weight, conv_kernel_size, kappa_star=kappa_star,
            )

        # m <= 64 -> custom CUDA path.
        # Pure attention via the fused kernels. The C extension knows nothing
        # about conv — that is added below at the Python level via cuDNN.
        out = FlashNystromFunction.apply(
            q, k, v, num_landmarks, newton_iter, fast_dk2inv,
            kappa_star, use_tc_pinv,
        )
        # Add depthwise-conv residual via cuDNN if requested.
        if conv_weight is not None and conv_kernel_size > 0:
            out = out + _depthwise_conv_residual(v, conv_weight)
        return out
    else:
        from flash_nystrom.reference import nystrom_attention_reference

        return nystrom_attention_reference(
            q, k, v, num_landmarks, newton_iter, conv_weight, conv_kernel_size,
            kappa_star=kappa_star,
        )


class FlashNystromAttention(nn.Module):
    def __init__(self, dim, heads=8, config=None):
        super().__init__()
        if config is None:
            config = NystromConfig()
        assert dim % heads == 0
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.config = config
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        if config.use_conv_residual and config.conv_kernel_size > 0:
            self.conv_weight = nn.Parameter(
                torch.randn(heads, config.conv_kernel_size) * 0.02
            )
        else:
            self.conv_weight = None

    def forward(self, x):
        B, N, _ = x.shape
        H, D = self.heads, self.head_dim
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        ks = self.config.conv_kernel_size if self.conv_weight is not None else 0
        # conv_weight is FP32 (nn.Parameter); F.conv1d under autocast handles
        # the dtype matching internally — no manual cast needed.
        out = flash_nystrom_attention(
            q,
            k,
            v,
            num_landmarks=self.config.num_landmarks,
            newton_iter=self.config.newton_iter,
            conv_weight=self.conv_weight,
            conv_kernel_size=ks,
            fast_dk2inv=self.config.fast_dk2inv,
            kappa_star=self.config.kappa_star,
            use_tc_pinv=self.config.use_tc_pinv,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, N, self.dim))
