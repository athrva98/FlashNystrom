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

    def __post_init__(self):
        assert self.num_landmarks > 0, "num_landmarks must be positive"
        assert self.newton_iter > 0, "newton_iter must be positive"
        assert self.conv_kernel_size >= 0, "conv_kernel_size must be non-negative"
        if self.conv_kernel_size > 0:
            assert self.conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
