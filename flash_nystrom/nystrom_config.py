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

    def __post_init__(self):
        assert self.num_landmarks > 0, "num_landmarks must be positive"
        assert self.newton_iter > 0, "newton_iter must be positive"
        assert self.conv_kernel_size >= 0, "conv_kernel_size must be non-negative"
        if self.conv_kernel_size > 0:
            assert self.conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
