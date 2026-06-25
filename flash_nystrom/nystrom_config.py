# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

from dataclasses import dataclass


@dataclass
class NystromConfig:
    """Configuration for FlashNystrom attention."""

    num_landmarks: int = 64
    """Number of landmark/inducing points for Nystrom approximation."""

    newton_iter: int = 6
    """Number of Newton-Schulz iterations for pseudoinverse.

    6 is the default from the original Nyströmformer paper. The backward is
    an exact unrolled chain rule through every NS iteration (not IFT, not
    truncated), so the gradient is correct regardless of whether NS has fully
    converged to K2^+. Lower values (e.g. 3) are cheaper but make K2_inv less
    accurate as a pseudoinverse approximation; higher values (e.g. 10-15)
    give a tighter approximation but cost more flops in both forward and
    backward."""

    conv_kernel_size: int = 3
    """Kernel size for depthwise conv1d residual connection. Set to 0 to disable."""

    use_conv_residual: bool = True
    """Whether to add depthwise conv1d residual."""

    fast_dk2inv: bool = True
    """Tensor-core path for `compute_dk2inv` in the backward pass.

    Default True uses the FP16/BF16 tensor-core kernel. Set False to use
    the FP32 scalar fallback (bit-for-bit consistent with the autograd
    reference modulo FP32 noise).

    The TC path is 4-6x faster on the full backward at N=4096+ (since
    `compute_dk2inv` accounts for ~75-85% of bwd time when the scalar
    kernel is used). The tradeoff is one FP32->FP16/BF16 conversion of
    the softmax output P right before the second GEMM, trimming P to a
    10-bit mantissa. Single-seed CIFAR-10 runs show this loss within
    FP16 stochastic variance; it can cost a fraction of a percentage
    point on tight accuracy comparisons. FP32 inputs always use the
    scalar fallback regardless of this flag (the TC atom requires
    16-bit operands)."""

    kappa_star: float = 5.0
    """Tikhonov ridge target condition number for the pseudoinverse.

    The pinv inverts M = K2^T K2 + lambda*I with
    lambda = (||K2||_1 ||K2||_inf) / kappa_star, guaranteeing cond(M) <=
    kappa_star and keeping the Newton-Schulz iteration well-conditioned even
    when the landmark Gram matrix K2 is near-singular (which it becomes as N
    grows, since segment-mean landmarks regress toward the global mean).

    Default 5.0 (the value used in all experiments). Set 0.0 to disable the
    ridge and invert the raw K2 (the original Nystromformer formulation; only
    safe when K2 is well-conditioned). Threaded identically to the kernel and
    the reference so both compute the same regularized pseudoinverse — this
    replaces the old FN_KAPPA_STAR environment variable."""

    use_tc_pinv: bool = True
    """Route the pseudoinverse through the tf32 tensor-core Newton-Schulz chain.

    Default True (faster, verified accurate; the tf32 pinv floor ~6e-4 is
    actually tighter than the fp16-reference's ~1.2e-3). Set False to force the
    fp32 scalar kernel. Only applies at num_landmarks == 64; the scalar kernel
    is used otherwise regardless. Replaces the old FN_K2INV_TC environment var."""

    def __post_init__(self):
        assert self.num_landmarks > 0, "num_landmarks must be positive"
        assert self.newton_iter > 0, "newton_iter must be positive"
        assert self.conv_kernel_size >= 0, "conv_kernel_size must be non-negative"
        if self.conv_kernel_size > 0:
            assert self.conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
        import math
        assert math.isfinite(self.kappa_star) and self.kappa_star >= 0.0, \
            "kappa_star must be finite and >= 0 (0 disables the Tikhonov ridge)"
