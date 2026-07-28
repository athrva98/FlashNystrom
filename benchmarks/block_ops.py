# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Sequence-mixer BLOCKS, for the block-level latency comparison.

Mamba and DeltaNet cannot enter the operator table: they are not attention
operators taking (q, k, v) -> out, they are whole sequence mixers that own
their input projections, short convolutions and gating. Timing them against a
bare attention kernel would compare different amounts of work and flatter
whichever side carries less.

So they are compared at BLOCK level instead: every entry here maps
(B, N, d_model) -> (B, N, d_model) and includes whatever projections its own
formulation requires. The attention-family entries are wrapped in the same
q/k/v/out projection structure so the comparison is like-for-like -- that
wrapper is exactly ``paper.mqar.model.build_attention``, the module the MQAR
experiments use, so the two experiments measure the same objects.

Read the two tables together: the operator table isolates the attention math,
this one shows what a practitioner actually pays per layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_block(name: str, d_model: int, heads: int, seq_len: int,
                num_landmarks: int = 64, device="cuda", dtype=torch.float16):
    """One sequence-mixer block: (B, N, d_model) -> (B, N, d_model).

    Names route to the same implementations the MQAR experiments use, so a
    latency number here refers to the same object as a recall number there.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paper.mqar.model import build_attention

    if name in ("mamba", "hyena"):
        # Vendored baselines; they own their projections and convolutions.
        mod = build_attention(name, d_model, heads, seq_len=seq_len)
    elif name == "delta_net":
        mod = DeltaNetBlock(d_model, heads)
    else:
        # Attention family (sdpa / flash_nystrom / flash_nystrom_tc /
        # nystrom_reference / linear_attention): build_attention supplies the
        # q/k/v/out projections, so every entry carries the same wrapper cost.
        mod = build_attention(name, d_model, heads, seq_len=seq_len,
                              num_landmarks=num_landmarks, kappa_star=0.0)
    return mod.to(device=device, dtype=dtype)


class DeltaNetBlock(nn.Module):
    """DeltaNet (Yang et al. 2024, arXiv:2406.06484), recurrent form.

    The delta rule updates a fast-weight memory with an error-corrected write:

        S_t = S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T,   o_t = S_t q_t

    which is the paper's Eq. for the state update, with beta_t = sigmoid(W_b x)
    the write strength. Queries and keys are SiLU-activated and L2-normalised
    per the paper ("k_t = SiLU(W_K x_t)/||.||_2"); values are SiLU-activated.

    This is the SEQUENTIAL reference form, written from the paper's equations.
    It is correct but O(N) sequential steps, so it is far slower than the
    authors' chunkwise-parallel Triton kernel in the `fla` library. Any timing
    taken from this class measures the reference implementation, not DeltaNet's
    achievable speed, and must be reported that way or not at all.
    """

    def __init__(self, d_model: int, heads: int):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by heads {heads}")
        self.heads, self.head_dim = heads, d_model // heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.b_proj = nn.Linear(d_model, heads, bias=False)   # write strength
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, _ = x.shape
        H, D = self.heads, self.head_dim
        shape = lambda t: t.view(B, N, H, D).transpose(1, 2)   # (B, H, N, D)
        q = shape(torch.nn.functional.silu(self.q_proj(x)))
        k = shape(torch.nn.functional.silu(self.k_proj(x)))
        v = shape(torch.nn.functional.silu(self.v_proj(x)))
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)    # (B, H, N)

        S = torch.zeros(B, H, D, D, device=x.device, dtype=torch.float32)
        outs = []
        for t in range(N):
            k_t = k[:, :, t].float().unsqueeze(-1)              # (B, H, D, 1)
            v_t = v[:, :, t].float().unsqueeze(-1)
            b_t = beta[:, :, t].float().unsqueeze(-1).unsqueeze(-1)
            # S (I - b k k^T) + b v k^T, grouped to avoid forming (I - b k k^T)
            S = S - b_t * (S @ k_t) @ k_t.transpose(-2, -1) + b_t * v_t @ k_t.transpose(-2, -1)
            outs.append((S @ q[:, :, t].float().unsqueeze(-1)).squeeze(-1))
        o = torch.stack(outs, dim=2).to(x.dtype)                # (B, H, N, D)
        return self.out_proj(o.transpose(1, 2).reshape(B, N, H * D))
