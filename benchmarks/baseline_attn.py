# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Bidirectional-native baseline attention MODULES for the training harnesses.

``baseline_ops.py`` holds the bare operators for latency measurement. These are
the ``nn.Module`` versions with q/k/v/out projections, so they drop into
``train_three_way.py``'s factory table and the genomics model on the same
footing as ``SDPAAttention`` / ``FlashNystromAttention``: only the attention
math differs, every arm carries the same projection cost.

Each baseline routes to an OPTIMIZED implementation when one exists, because
comparing our fused kernel against unfused framework code would measure
implementation effort rather than the operator:

  * linear attention -> fla-org/flash-bidirectional-linear-attention, Triton
    kernels for non-causal linear attention with a fused forward AND backward.
    Their benchmark reports ~2.1x fwd / ~2.3x bwd over a torch baseline, so the
    unfused version is not a fair opponent. Falls back to the torch path with a
    loud warning if the package is absent.
  * sliding window -> FlashAttention-2's fused windowed kernel.
  * Linformer -> two dense projections plus one softmax attention; the work is
    already three cuBLAS-class GEMMs with no fusible softmax chain, so a
    hand-written module IS the optimized form. xformers exposes the same
    computation; we keep it local to avoid a dependency that changes nothing.

Install for full fidelity:
    pip install flash-attn --no-build-isolation
    pip install -e git+https://github.com/fla-org/flash-bidirectional-linear-attention.git#egg=flash_bla
"""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

_FUSED_LA = None
_FUSED_LA_WARNED = False


def _get_fused_la():
    """flash_bla's fused non-causal linear attention, or None."""
    global _FUSED_LA
    if _FUSED_LA is None:
        try:
            from flash_bla.ops.simple_la.fused import simple_la
            _FUSED_LA = simple_la
        except Exception:
            _FUSED_LA = False
    return _FUSED_LA or None


class _QKVOut(nn.Module):
    """Shared projection shell so every baseline carries identical wrapper cost."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _qkv(self, x):
        B, N, _ = x.shape
        H, D = self.heads, self.head_dim
        shape = lambda t: t.view(B, N, H, D).transpose(1, 2)   # (B,H,N,D)
        return shape(self.q_proj(x)), shape(self.k_proj(x)), shape(self.v_proj(x))

    def _merge(self, o):
        B, H, N, D = o.shape
        return self.out_proj(o.transpose(1, 2).reshape(B, N, H * D))


class LinearAttention(_QKVOut):
    """Linear attention, phi(x)=elu(x)+1 (Katharopoulos et al. 2020).

    Uses flash_bla's fused Triton kernel when available. That kernel implements
    the plain (unnormalized) form, so the feature map and normalization are
    applied around it to keep the math identical to the torch path."""

    def forward(self, x):
        q, k, v = self._qkv(x)
        qf, kf = F.elu(q) + 1.0, F.elu(k) + 1.0
        fused = _get_fused_la()
        if fused is not None:
            global _FUSED_LA_WARNED
            num = fused(qf, kf, v, 1.0)                     # (B,H,N,D)
            den = (qf @ kf.sum(dim=-2).unsqueeze(-1)) + 1e-6
            return self._merge(num / den)
        if not _FUSED_LA_WARNED:
            warnings.warn("flash_bla not installed: linear attention is running "
                          "the UNFUSED torch path, which is ~2x slower than the "
                          "available Triton kernel and is not a fair baseline.")
            _FUSED_LA_WARNED = True
        kv = kf.transpose(-2, -1) @ v
        z = kf.sum(dim=-2)
        num = qf @ kv
        den = (qf @ z.unsqueeze(-1)) + 1e-6
        return self._merge(num / den)


class LinformerAttention(_QKVOut):
    """Linformer (Wang et al. 2020): project the key/value length axis N -> r.

    The (r, N) projections are learned parameters, so their size grows with the
    sequence length and a trained model is tied to the length it was trained
    at. That is the structural cost of the method and is reported as such."""

    def __init__(self, dim: int, heads: int, seq_len: int, rank: int = 64):
        super().__init__(dim, heads)
        self.seq_len, self.rank = seq_len, rank
        self.E = nn.Parameter(torch.randn(rank, seq_len) * seq_len ** -0.5)
        self.F = nn.Parameter(torch.randn(rank, seq_len) * seq_len ** -0.5)

    def forward(self, x):
        q, k, v = self._qkv(x)
        N = q.shape[-2]
        E, Fp = self.E[:, :N], self.F[:, :N]              # tolerate shorter N
        k_p, v_p = E @ k, Fp @ v                          # (B,H,r,D)
        scores = (q @ k_p.transpose(-2, -1)) * self.head_dim ** -0.5
        return self._merge(torch.softmax(scores, dim=-1) @ v_p)


class SlidingWindowAttention(_QKVOut):
    """Bidirectional local attention through FlashAttention-2's windowed kernel.

    Each query attends to `window` nearest keys, half per side. Information
    travels window/2 positions per layer, so the receptive field is local by
    construction -- the contrast with a landmark method is reach, not cost."""

    def __init__(self, dim: int, heads: int, window: int = 256):
        super().__init__(dim, heads)
        self.window = window

    def forward(self, x):
        from flash_attn import flash_attn_func      # hard requirement: fused kernel
        q, k, v = self._qkv(x)
        half = self.window // 2
        qt, kt, vt = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        o = flash_attn_func(qt, kt, vt, causal=False, window_size=(half, half))
        return self._merge(o.transpose(1, 2))


def build_baseline(name: str, dim: int, heads: int, seq_len: int,
                   rank: int = 64, window: int = 256):
    """Factory used by the training harnesses."""
    if name == "linear_attention":
        return LinearAttention(dim, heads)
    if name == "linformer":
        return LinformerAttention(dim, heads, seq_len, rank)
    if name == "sliding_window":
        return SlidingWindowAttention(dim, heads, window)
    raise ValueError(f"unknown baseline {name!r}")
