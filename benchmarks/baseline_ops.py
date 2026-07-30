# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Sub-quadratic attention operators, for the latency comparison.

These are OPERATORS, not blocks: every one maps (q, k, v) of shape
(B, H, N, D) to an output of the same shape, with no projections, residual,
norm, or gating. That is the only way a latency number is attributable to the
attention math rather than to how much surrounding work a particular block
happens to do. Mamba and DeltaNet do not fit this interface (they are whole
sequence mixers with their own projections and convolutions) and are compared
separately at block level.

Each operator is written to its own paper's formulation, not tuned by us:

  * ``linear_attention`` -- Katharopoulos et al. 2020, feature map
    phi(x) = elu(x) + 1, computed in the associative order
    phi(Q) (phi(K)^T V), which is O(N D^2) and never forms the N x N matrix.
  * ``linformer`` -- Wang et al. 2020: fixed low-rank projections E, F map the
    key/value length axis N -> r before ordinary softmax attention, giving
    O(N r D). Note the projections are (r, N), so unlike Nystrom's landmark
    count the parameter count GROWS with the sequence length; the caller
    allocates them per length, which is itself part of the method's cost.
  * ``sdpa`` -- exact attention through PyTorch's fused kernel, the ceiling.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sdpa_op(q, k, v):
    """Exact attention (PyTorch's fused SDPA). O(N^2 D)."""
    return F.scaled_dot_product_attention(q, k, v)


def linear_attention_op(q, k, v, eps: float = 1e-6):
    """Linear attention, phi(x) = elu(x) + 1 (Katharopoulos et al. 2020).

    Associative order: (phi(K)^T V) is (D, D), so cost is O(N D^2) and the
    N x N matrix is never formed. Non-causal, matching how every operator in
    this comparison is timed.

    The normalizer accumulates in fp32. It is a sum over N terms, so in fp16 it
    passes 65504 around N=2304 and the operator returns zeros and then NaN --
    at lengths well inside this paper's range. sum(dtype=) uses an fp32
    accumulator without materializing an fp32 copy of kf, so this is the same
    memory traffic as the fp16 reduction and the timing is unchanged."""
    from benchmarks.baseline_attn import _fp16_safe_scale
    qf = F.elu(q) + 1.0
    kf = F.elu(k) + 1.0
    # BOTH accumulators grow like N. The numerator gets a common scale s on v
    # (undone on the denominator, so out = num/den is unchanged) which keeps it
    # in fp16 range without an fp32 O(ND^2) matmul that would change the timing.
    s = _fp16_safe_scale(q.shape[-2], v.dtype)
    kv = kf.transpose(-2, -1) @ (v * s if s != 1.0 else v)   # (B, H, D, D)
    z = kf.sum(dim=-2, dtype=torch.float32)                  # (B, H, D)
    num = qf @ kv                                            # (B, H, N, D)
    den = (qf.float() @ z.unsqueeze(-1)) * s + eps
    return (num / den).to(v.dtype)


def linear_attention_fused_op(q, k, v, eps: float = 1e-6):
    """Linear attention through flash_bla's fused Triton kernel.

    The COMPLETE operator, not just the kernel call. flash_bla's ``simple_la``
    implements the unnormalized core, so the feature map and the normalization
    belong to the caller and must be timed with it: an operator timed without
    them is not the operator, and comparing it against arms that ARE timed end
    to end understates its cost. On top of the kernel this adds two elementwise
    passes for phi, one reduction for z, and one division.

    Raises ImportError when flash_bla is absent rather than silently falling
    back, so a latency table can never mix fused and unfused rows.
    """
    try:
        from flash_bla.ops.simple_la.fused import simple_la
    except ImportError as e:                                  # pragma: no cover
        raise ImportError(
            "linear_attention_fused_op needs flash_bla (the fused Triton "
            "kernel): pip install -e git+https://github.com/fla-org/flash-"
            "bidirectional-linear-attention.git#egg=flash_bla") from e
    qf = F.elu(q) + 1.0
    kf = F.elu(k) + 1.0
    num = simple_la(qf, kf, v, 1.0)
    z = kf.sum(dim=-2, dtype=torch.float32)
    den = (qf.float() @ z.unsqueeze(-1)) + eps
    return (num / den).to(v.dtype)


def linformer_op(q, k, v, E, F_):
    """Linformer (Wang et al. 2020): project the key/value length axis N -> r,
    then ordinary softmax attention against the r projected positions.

    E, F_ are (r, N) and supplied by the caller because they are parameters
    whose size depends on N -- see ``make_linformer_projections``. Cost is
    O(N r D) for the projections plus O(N r D) for the attention."""
    k_proj = E @ k                            # (B, H, r, D)
    v_proj = F_ @ v                           # (B, H, r, D)
    scale = q.shape[-1] ** -0.5
    scores = (q @ k_proj.transpose(-2, -1)) * scale   # (B, H, N, r)
    return torch.softmax(scores, dim=-1) @ v_proj     # (B, H, N, D)


def make_linformer_projections(B, H, N, r, device, dtype, seed=0,
                               requires_grad: bool = True):
    """The (r, N) projections Linformer needs at this length.

    Broadcast over batch/head (shape (1, 1, r, N)), which is Linformer's
    shared-projection variant and the cheapest form of the method. Returned
    separately from the op so their allocation cost -- which grows linearly
    with N, unlike a landmark count -- is visible to the caller.

    ``requires_grad`` defaults True because E and F are LEARNED parameters:
    a training-step comparison must pay for dE and dF, each an (r, N) matrix
    computed with O(N r D) work. Timing them as frozen buffers understates
    Linformer's training cost by roughly the projection backward, which is
    the same order as its forward -- so the default here is the honest one,
    and False is only for inference-style measurements."""
    g = torch.Generator(device=device).manual_seed(seed)
    scale = N ** -0.5
    E = (torch.randn(1, 1, r, N, generator=g, device=device, dtype=dtype) * scale
         ).requires_grad_(requires_grad)
    F_ = (torch.randn(1, 1, r, N, generator=g, device=device, dtype=dtype) * scale
          ).requires_grad_(requires_grad)
    return E, F_


def linformer_projection_bytes(N, r, dtype_size=2):
    """Bytes held by the two (r, N) projections, the memory Linformer needs
    that a landmark method does not."""
    return 2 * r * N * dtype_size


def sliding_window_op(q, k, v, window: int):
    """Bidirectional sliding-window attention: each query attends exactly to
    the `window` nearest keys, half on each side (Longformer / BigBird /
    Mistral-style local attention, without the global tokens).

    Routed through FlashAttention-2's fused windowed kernel, which is the
    fastest available implementation, so the comparison is against the
    baseline at its best rather than against a masked O(N^2) stand-in.

    The receptive field is the point of contrast, not just the cost: a window
    of w gives each query an EXACT view of w local keys and no global mixing
    (information travels only w/2 positions per layer), whereas m landmarks
    give an APPROXIMATE view of the entire sequence in one layer. Matching the
    per-query key budget (w = m) matches FLOPs but not what the operator can
    represent, which is why both are reported.
    """
    try:
        from flash_attn import flash_attn_func
    except ImportError as e:  # pragma: no cover - depends on the bench image
        raise RuntimeError(
            "sliding_window_op needs flash-attn (the fused windowed kernel); "
            "a masked SDPA stand-in would be O(N^2) and not a fair baseline"
        ) from e
    half = window // 2
    # flash_attn wants (B, N, H, D); ours are (B, H, N, D).
    qt, kt, vt = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    o = flash_attn_func(qt, kt, vt, causal=False, window_size=(half, half))
    return o.transpose(1, 2)


def sliding_window_receptive_field(window: int, layers: int = 1):
    """Positions reachable after `layers` sliding-window layers: each layer
    moves information at most window//2 in each direction. Contrast with a
    landmark method, whose receptive field is the whole sequence at depth 1."""
    return window // 2 * layers
