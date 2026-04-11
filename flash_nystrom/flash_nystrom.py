# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn

from flash_nystrom.nystrom_config import NystromConfig

try:
    import flash_nystrom._C as _C

    HAS_CUDA = True
except ImportError:
    _C = None
    HAS_CUDA = False


class FlashNystromFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, conv_weight, num_landmarks, newton_iter, conv_kernel_size
    ):
        assert _C is not None, "CUDA extension not available"

        results = _C.forward(
            q, k, v, num_landmarks, newton_iter, conv_kernel_size, conv_weight
        )
        # results: [output, q_s, k_s, q_tilde, k_tilde, k2inv, step2,
        #           lse1, lse2, lse3]
        output = results[0]

        saved = list(results[1:])  # q_s, k_s, qt, kt, k2inv, step2, lse1, lse2, lse3
        saved.append(v)
        saved.append(output)
        if conv_weight is not None:
            saved.append(conv_weight)
        ctx.save_for_backward(*saved)
        ctx.num_landmarks = num_landmarks
        ctx.newton_iter = newton_iter
        ctx.conv_kernel_size = conv_kernel_size
        ctx.has_conv = conv_weight is not None

        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        q_s, k_s, q_tilde, k_tilde, k2_inv, step2 = saved[0:6]
        lse1, lse2, lse3 = saved[6:9]
        v = saved[9]
        output = saved[10]
        conv_weight = saved[11] if ctx.has_conv else None

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
            v,
            output,
            ctx.num_landmarks,
            ctx.newton_iter,
            ctx.conv_kernel_size,
            conv_weight,
        )
        dQ, dK, dV = results[0], results[1], results[2]
        dconv = results[3] if ctx.has_conv and results[3] is not None else None

        return dQ, dK, dV, dconv, None, None, None


def flash_nystrom_attention(
    q, k, v, num_landmarks=64, newton_iter=6, conv_weight=None, conv_kernel_size=0
):
    """main entry point — uses CUDA kernels if available, falls back to pytorch."""
    if HAS_CUDA and q.is_cuda:
        return FlashNystromFunction.apply(
            q, k, v, conv_weight, num_landmarks, newton_iter, conv_kernel_size
        )
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
        out = flash_nystrom_attention(
            q,
            k,
            v,
            num_landmarks=self.config.num_landmarks,
            newton_iter=self.config.newton_iter,
            conv_weight=self.conv_weight,
            conv_kernel_size=ks,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, N, self.dim))
