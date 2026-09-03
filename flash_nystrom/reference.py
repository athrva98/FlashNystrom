# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

"""
Pure PyTorch reference implementation of NystromFormer attention.
This is the ground truth the CUDA kernels are tested against.
Deliberately kept simple and readable; it is not optimized.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def iterative_pinverse(matrix: torch.Tensor, n_iter: int = 6) -> torch.Tensor:
    """Newton-Schulz pseudoinverse (third-order convergence).

    Must be float32 — fp16 will diverge. 6 iterations is plenty for m<=64.
    The formula is dense but it is matrix multiplication in a loop.

    Z_0 init: Z_0 = A^T / (||A||_1 * ||A||_inf). This is the standard
    Cauchy-interpolation bound that guarantees ||Z_0 A - I|| < 1 in operator
    norm, which is what NS needs for convergence. Both norms are autograd-
    differentiable so the chain rule includes their max() contributions.
    """
    assert matrix.dtype == torch.float32, "Newton-Schulz requires float32"

    abs_mat = matrix.abs()
    # ||A||_1 = max column sum, ||A||_inf = max row sum
    norm_1 = abs_mat.sum(dim=-2).max(dim=-1, keepdim=True).values.unsqueeze(-1)
    norm_inf = abs_mat.sum(dim=-1).max(dim=-1, keepdim=True).values.unsqueeze(-1)

    # Z_0 = A^T / (||A||_1 * ||A||_inf)
    Z = matrix.transpose(-2, -1) / (norm_1 * norm_inf).clamp(min=1e-12)

    I = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    I = I.expand_as(matrix)

    for _ in range(n_iter):
        KZ = matrix @ Z
        inner = 7.0 * I - KZ
        mid = 15.0 * I - KZ @ inner
        outer = 13.0 * I - KZ @ mid
        Z = 0.25 * Z @ outer

    return Z


def nystrom_attention_reference(  # noqa: C901
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_landmarks: int = 64,
    newton_iter: int = 6,
    conv_weight: Optional[torch.Tensor] = None,
    conv_kernel_size: int = 0,
    kappa_star: float = 0.0,
) -> torch.Tensor:
    """Reference nystrom attention — the one that actually does math correctly.

    Compute right-to-left: kernel_1 @ kernel_2_inv @ kernel_3 @ V
    so an (N, N) matrix is never materialized, which is the point of the
    factorization.
    """
    B, H, N, D = q.shape
    m = num_landmarks
    assert N >= m, f"seq_len ({N}) must be >= num_landmarks ({m})"

    # Scale by d^(-1/4)
    scale = D ** (-0.25)
    q_s = q * scale
    k_s = k * scale

    # Segment mean landmarks using floor division (matches CUDA kernel)
    # First (m-1) segments have floor(N/m) elements.
    # Last segment absorbs the remainder: [floor(N/m)*(m-1), N)
    seg_len = N // m

    # First (m-1) landmarks: reshape and mean
    truncated_N = seg_len * (m - 1)
    if m > 1:
        q_first = (
            q_s[:, :, :truncated_N, :].reshape(B, H, m - 1, seg_len, D).mean(dim=3)
        )
        k_first = (
            k_s[:, :, :truncated_N, :].reshape(B, H, m - 1, seg_len, D).mean(dim=3)
        )
    else:
        q_first = q_s[:, :, :0, :].reshape(B, H, 0, D)  # empty
        k_first = k_s[:, :, :0, :].reshape(B, H, 0, D)

    # Last landmark: mean of remaining elements
    q_last = q_s[:, :, truncated_N:N, :].mean(dim=2, keepdim=True)  # (B, H, 1, D)
    k_last = k_s[:, :, truncated_N:N, :].mean(dim=2, keepdim=True)

    q_tilde = torch.cat([q_first, q_last], dim=2)  # (B, H, m, D)
    k_tilde = torch.cat([k_first, k_last], dim=2)

    # Three kernel matrices
    kernel_1 = F.softmax(q_s @ k_tilde.transpose(-2, -1), dim=-1)  # (B, H, N, m)
    kernel_2 = F.softmax(q_tilde @ k_tilde.transpose(-2, -1), dim=-1)  # (B, H, m, m)
    kernel_3 = F.softmax(q_tilde @ k_s.transpose(-2, -1), dim=-1)  # (B, H, m, N)

    # Pseudoinverse via Newton-Schulz iteration (always FP32).
    kernel_2_f32 = kernel_2.float()
    # Rigorous (non-normality-proof) Tikhonov ridge via the normal equations.
    # K2 is non-normal in general (real trained K2 has ||[K,K^T]||/||K||^2 up to
    # ~0.8 in early layers), so K2 + lambda*I does NOT shift its singular values
    # and cannot reliably bound cond(K2). Instead regularize the SYMMETRIC PSD
    # M = K2^T K2 + lambda*I: its singular values equal its eigenvalues, so
    # +lambda*I shifts them exactly, and
    #   cond(M) = (sigma_max^2 + lambda)/(sigma_min^2 + lambda) ~ sigma_max^2/lambda,
    # bounded by kappa_star with  lambda = sigma_max^2 / kappa_star,  regardless of
    # how non-normal K2 is. The regularized pseudoinverse is the Tikhonov
    # least-squares solution
    #   K2^+_reg = (K2^T K2 + lambda*I)^{-1} K2^T.
    # cond(K2) = sigma_max/sigma_min grows ~linearly with N (landmarks regress to
    # the global mean as each pools N/m tokens), so the ridge is what keeps the
    # Newton-Schulz pinv and its unrolled backward convergent/stable. kappa_star
    # is the single knob (target cond of M): ~5 for J=6 iterations; larger ridges
    # less (preserves more directions) at the cost of NS convergence.
    if kappa_star > 0.0:
        m = kernel_2_f32.shape[-1]
        eye = torch.eye(m, device=kernel_2_f32.device, dtype=kernel_2_f32.dtype)
        # lambda is a detached regularization SCALE (a data-derived
        # hyperparameter, not differentiated). sigma_max^2 is estimated by the
        # cheap spectral-norm bound ||K2||_1 * ||K2||_inf >= sigma_max^2, so
        # cond(M) <= kappa_star is still guaranteed. The kernel computes the
        # SAME bound (the NS init already needs both norms), so reference and
        # kernel use an identical lambda with no power-iteration drift.
        Kd = kernel_2_f32.detach()
        norm1 = Kd.abs().sum(-2).amax(-1)      # max column sum  = ||K2||_1
        norm_inf = Kd.abs().sum(-1).amax(-1)   # max row sum     = ||K2||_inf (=1 for row-stochastic)
        lam = (norm1 * norm_inf / kappa_star)[..., None, None]
        M = kernel_2_f32.transpose(-2, -1) @ kernel_2_f32 + lam * eye  # symmetric PSD
        M_inv = iterative_pinverse(M, n_iter=newton_iter)
        kernel_2_inv = (M_inv @ kernel_2_f32.transpose(-2, -1)).to(q.dtype)
    else:
        kernel_2_inv = iterative_pinverse(kernel_2_f32, n_iter=newton_iter).to(q.dtype)

    # Output (right-to-left for O(n*m) complexity)
    step1 = kernel_3 @ v  # (B, H, m, D)
    step2 = kernel_2_inv @ step1  # (B, H, m, D)
    output = kernel_1 @ step2  # (B, H, N, D)

    # Depthwise conv residual
    if conv_weight is not None and conv_kernel_size > 0:
        assert conv_weight.shape == (H, conv_kernel_size), (
            f"conv_weight shape {conv_weight.shape} != expected ({H}, {conv_kernel_size})"
        )

        # Use F.conv1d: reshape v to (B*H, D, N) and weight to (D, 1, ks) grouped
        # Actually, per-head conv means groups=H. Reshape to (B, H*D, N).
        # Simpler: loop over heads, each with F.conv1d on (B, D, N) with (D, 1, ks).
        # But that's still a loop. Better: use groups.
        #
        # v: (B, H, N, D) -> (B, H, D, N) -> (B, H*D, N)
        # weight: (H, ks) -> (H*D, 1, ks) where each group of D channels shares the same weight
        # groups = H*D would mean each channel is independent, but we want each HEAD to share.
        # That requires groups = H, with D channels per group.
        #
        # weight shape for groups=H: (H*D, D, ks)? No, for depthwise with groups=H:
        # in_channels = H*D, out_channels = H*D, groups = H
        # -> weight: (H*D, D, ks) which is huge.
        #
        # Easiest correct approach: loop over heads with F.conv1d.
        # H is typically 8-16, so this is fine for a reference.

        pad = conv_kernel_size // 2
        conv_out = torch.zeros_like(v)
        for h in range(H):
            # v[:, h]: (B, N, D) -> (B, D, N) for conv1d
            v_h = v[:, h].transpose(1, 2)  # (B, D, N)
            # weight: (D, 1, ks) — same weight for all D channels (depthwise)
            w_h = (
                conv_weight[h].unsqueeze(0).unsqueeze(0).expand(D, 1, conv_kernel_size)
            )
            # groups=D for depthwise
            conv_h = F.conv1d(v_h, w_h, padding=pad, groups=D)  # (B, D, N)
            conv_out[:, h] = conv_h.transpose(1, 2)  # (B, N, D)

        output = output + conv_out

    return output


def nystrom_attention_reference_simple(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_landmarks: int = 64,
    newton_iter: int = 6,
) -> torch.Tensor:
    """Simplified reference without conv residual."""
    return nystrom_attention_reference(q, k, v, num_landmarks, newton_iter, None, 0)
