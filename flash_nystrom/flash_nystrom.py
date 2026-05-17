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
    def forward(ctx, q, k, v, num_landmarks, newton_iter, fast_dk2inv):
        assert _C is not None, "CUDA extension not available"

        results = _C.forward(q, k, v, num_landmarks, newton_iter)
        # results: [output, q_s, k_s, q_tilde, k_tilde, k2inv, step2,
        #           lse1, lse2, lse3, ns_iterates, k2_softmax, b_saved]
        # b_saved = softmax(Q_tilde @ K^T) @ V is reused in the backward so
        # compute_dk2inv skips an O(m*N*D) recomputation pass.
        output = results[0]

        saved = list(results[1:])  # q_s, k_s, qt, kt, k2inv, step2, lse1, lse2, lse3, ns_iter, k2sm, b_saved
        saved.append(v)
        saved.append(output)
        ctx.save_for_backward(*saved)
        ctx.num_landmarks = num_landmarks
        ctx.newton_iter = newton_iter
        ctx.fast_dk2inv = fast_dk2inv

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
        )

        # apply() args were (q, k, v, num_landmarks, newton_iter, fast_dk2inv)
        # → six grad outputs; only q/k/v are differentiable.
        return dQ, dK, dV, None, None, None


def flash_nystrom_attention(
    q, k, v, num_landmarks=64, newton_iter=6, conv_weight=None, conv_kernel_size=0,
    fast_dk2inv=True,
):
    """main entry point — uses CUDA kernels if available, falls back to pytorch.

    The conv residual (when enabled) is computed via cuDNN through F.conv1d,
    OUTSIDE the FlashNystromFunction autograd boundary. PyTorch handles its
    backward automatically. The previous custom CUDA conv kernels
    (dconv_residual.cuh / dconv_residual_bwd.cuh) have been removed; only
    the cuDNN path exists now.

    `fast_dk2inv` controls the `compute_dk2inv` path in the backward (FP16/BF16
    only). Default True uses the tensor-core kernel (4-6x faster on the full
    bwd). Set False to use the FP32 scalar fallback, which is bit-for-bit
    consistent with the autograd reference. The TC path converts the softmax
    output P from FP32 to FP16/BF16 before GEMM2, trimming P to a 10-bit
    mantissa — small accuracy cost, typically below FP16 training noise.
    """
    if HAS_CUDA and q.is_cuda:
        # Pure attention via the fused kernels. The C extension knows nothing
        # about conv — that is added below at the Python level via cuDNN.
        out = FlashNystromFunction.apply(
            q, k, v, num_landmarks, newton_iter, fast_dk2inv,
        )
        # Add depthwise-conv residual via cuDNN if requested.
        if conv_weight is not None and conv_kernel_size > 0:
            out = out + _depthwise_conv_residual(v, conv_weight)
        return out
    else:
        from flash_nystrom.reference import nystrom_attention_reference

        return nystrom_attention_reference(
            q, k, v, num_landmarks, newton_iter, conv_weight, conv_kernel_size
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
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, N, self.dim))
