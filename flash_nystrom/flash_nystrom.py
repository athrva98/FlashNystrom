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
    @staticmethod
    def forward(
        ctx, q, k, v, conv_weight, num_landmarks, newton_iter, conv_kernel_size,
        fast_dk2inv,
    ):
        assert _C is not None, "CUDA extension not available"

        results = _C.forward(
            q, k, v, num_landmarks, newton_iter, conv_kernel_size, conv_weight
        )
        # results: [output, q_s, k_s, q_tilde, k_tilde, k2inv, step2,
        #           lse1, lse2, lse3, ns_iterates, k2_softmax]
        output = results[0]

        saved = list(results[1:])  # q_s, k_s, qt, kt, k2inv, step2, lse1, lse2, lse3, ns_iter, k2sm
        saved.append(v)
        saved.append(output)
        if conv_weight is not None:
            saved.append(conv_weight)
        ctx.save_for_backward(*saved)
        ctx.num_landmarks = num_landmarks
        ctx.newton_iter = newton_iter
        ctx.conv_kernel_size = conv_kernel_size
        ctx.has_conv = conv_weight is not None
        ctx.fast_dk2inv = fast_dk2inv

        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        q_s, k_s, q_tilde, k_tilde, k2_inv, step2 = saved[0:6]
        lse1, lse2, lse3 = saved[6:9]
        ns_iterates = saved[9]
        k2_softmax = saved[10]
        v = saved[11]
        output = saved[12]
        conv_weight = saved[13] if ctx.has_conv else None

        results = _C.backward(
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
            v,
            output,
            ctx.num_landmarks,
            ctx.newton_iter,
            ctx.conv_kernel_size,
            conv_weight,
            ctx.fast_dk2inv,
        )
        dQ, dK, dV = results[0], results[1], results[2]
        dconv = results[3] if ctx.has_conv and results[3] is not None else None

        return dQ, dK, dV, dconv, None, None, None, None


def flash_nystrom_attention(
    q, k, v, num_landmarks=64, newton_iter=6, conv_weight=None, conv_kernel_size=0,
    fast_dk2inv=True,
):
    """main entry point — uses CUDA kernels if available, falls back to pytorch.

    The conv residual (when enabled) is computed via cuDNN through F.conv1d,
    OUTSIDE the FlashNystromFunction autograd boundary. PyTorch handles its
    backward automatically. The custom CUDA conv kernels (`dconv_residual.cuh`
    and `dconv_residual_bwd.cuh`) are no longer used by this path — they remain
    in the codebase only for the reference implementation's compatibility.

    `fast_dk2inv` controls the `compute_dk2inv` path in the backward (FP16/BF16
    only). Default True uses the tensor-core kernel (4-6x faster on the full
    bwd). Set False to use the FP32 scalar fallback, which is bit-for-bit
    consistent with the autograd reference. The TC path converts the softmax
    output P from FP32 to FP16/BF16 before GEMM2, trimming P to a 10-bit
    mantissa — small accuracy cost, typically below FP16 training noise.
    """
    if HAS_CUDA and q.is_cuda:
        # Pure attention via the fused kernels — pass conv_weight=None so the
        # custom CUDA conv kernels never run.
        out = FlashNystromFunction.apply(
            q, k, v, None, num_landmarks, newton_iter, 0, fast_dk2inv,
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
