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

    kappa_star: float = 0.0
    """Tikhonov ridge target condition number for the pseudoinverse.

    When > 0, the pinv inverts M = K2^T K2 + lambda*I with
    lambda = (||K2||_1 ||K2||_inf) / kappa_star, guaranteeing cond(M) <=
    kappa_star regardless of how ill-conditioned the landmark Gram matrix
    K2 becomes (cond(K2) grows with N, since segment-mean landmarks regress
    toward the global mean).

    Default 0.0 (no ridge, the original Nystromformer formulation). The
    2026-07 three-seed sweeps found end-task accuracy insensitive to (STL-10 at
    N=9216, cond(K2) ~1e13) or hurt by (STL-10 at N=2304: -2.6 points; MQAR:
    -11 points recall) a kappa_star=1e3 ridge, and never helped by it. This
    is expected from the exact unrolled Newton-Schulz backward: a
    non-converged pinv is still a deterministic operator with exact
    gradients, so trainability does not depend on conditioning. Set a
    finite kappa_star (e.g. 1e3) only when the application needs a
    well-conditioned operator per se (e.g. operator fidelity to exact
    attention at large N). Threaded identically to the kernel and the
    reference so both compute the same (un)regularized pseudoinverse."""

    use_tc_pinv: bool = False
    """Route the pseudoinverse through the tf32 tensor-core Newton-Schulz chain.

    Default False (the faithful path): the fp32 scalar Newton-Schulz matches the
    pure-torch reference to ~3e-4 on the output at every N, and the full fwd+bwd
    is still ~3x faster than the reference. Set True to opt into the tf32 tensor-
    core chain, which is ~4x faster than the reference but carries an N-
    independent ~1-3% error in the pseudoinverse (tf32 truncation when forming
    M = K2^T K2). That error is invisible when the Nystrom approximation is near-
    exact (small N, landmarks ~ tokens) but costs ~5% accuracy where the
    approximation actually works: 3-seed STL-10 at N=2304 measured scalar 36.0
    vs tf32 31.2 (matched kappa=1e3, J=16). Use True only as a speed/accuracy
    trade when that cost is acceptable. Only applies at num_landmarks == 64; the
    scalar kernel is used otherwise regardless. Replaces the old FN_K2INV_TC
    environment var."""

    landmark_mode: int = 0
    """Landmark selection. 0 = segment mean (default). 1 = leverage-seeded
    Voronoi means (leverage_landmarks.cuh): ridge-leverage scores -> Gumbel-top-m
    seeds -> Euclidean Voronoi partition -> cell means. Backward is
    straight-through (membership held fixed). Only the custom CUDA path
    (num_landmarks <= 64, head_dim in {64,128}) supports mode 1."""

    landmark_seed: int = 0
    """Base RNG seed for mode 1 (Q uses seed, K uses seed+1). Deterministic."""

    landmark_subsample: int = 1
    """Assign-pass thinning for mode 1 (1 = exact means). >1 systematically
    subsamples row tiles; use only at very large N."""

    landmark_gumbel_scale: float = 1.0
    """Mode 1 selection exploration. 1.0 = Plackett-Luce sampling (diverse, good
    for clustered data / MQAR). 0.0 = deterministic top-m leverage (stable
    selection, no step-to-step landmark jitter)."""

    landmark_force_first: int = 0
    """Mode 1: pin rows [0, force_first) as landmarks regardless of leverage.
    Set to 1 for a CLS-token ViT so the classifier's token is always a landmark
    (segment means always cover it; leverage may otherwise drop it)."""

    def __post_init__(self):
        assert self.num_landmarks > 0, "num_landmarks must be positive"
        assert self.landmark_mode in (0, 1), "landmark_mode must be 0 or 1"
        assert self.landmark_subsample >= 1, "landmark_subsample must be >= 1"
        assert self.newton_iter > 0, "newton_iter must be positive"
        assert self.conv_kernel_size >= 0, "conv_kernel_size must be non-negative"
        if self.conv_kernel_size > 0:
            assert self.conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
        import math
        assert math.isfinite(self.kappa_star) and self.kappa_star >= 0.0, \
            "kappa_star must be finite and >= 0 (0 disables the Tikhonov ridge)"
