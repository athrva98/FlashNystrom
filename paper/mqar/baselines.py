# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Faithful, hardware-portable sub-quadratic baselines for the MQAR comparison.

The operators are vendored from the reference implementations, NOT reconstructed
from memory:

  * Hyena           -- HazyResearch/zoology, zoology/mixers/hyena.py (Apache-2.0),
                       which follows Poli et al. 2023 (arXiv:2302.10866). Verbatim
                       except the pydantic @validate_call decorator is dropped.
  * Mamba           -- HazyResearch/zoology, zoology/mixers/mamba.py (Gu & Dao,
                       Apache-2.0). The module and its S4D/dt initialization are
                       verbatim; the forward is switched to the pure-PyTorch path
                       (no causal_conv1d / selective_scan CUDA kernels, whose
                       prebuilt binaries do not target every GPU) using the
                       official reference scan below.
  * selective_scan_ref -- state-spaces/mamba, mamba_ssm/ops/selective_scan_interface.py
                       (Apache-2.0). This is the reference the CUDA selective-scan
                       kernel is checked against, so the math is identical.

Each is wrapped in a mixer with the harness's (dim, heads) -> forward(x) contract
so build_attention can swap it into the attention slot for an apples-to-apples
comparison against sdpa / flash_nystrom / linear_attention.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Real Mamba CUDA kernels when installed (fast, the way Mamba is actually run).
# Without them, Mamba falls back to the pure-PyTorch reference scan below, which
# is correct but ~1000x slower (a Python loop over the sequence). The local
# sm_120 card cannot build these; supported archs (Colab A100/L4/T4>=sm80) can,
# via `pip install mamba-ssm causal-conv1d` (see the Colab notebook).
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_cuda
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_cuda
    _HAS_MAMBA_CUDA = True
except Exception:
    _selective_scan_cuda = None
    _causal_conv1d_cuda = None
    _HAS_MAMBA_CUDA = False


# =========================================================================== #
# Mamba selective-scan reference (state-spaces/mamba, verbatim)
# =========================================================================== #

def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False):
    """u:(B D L) delta:(B D L) A:(D N) B,C:(B N L) D:(D) z:(B D L). Returns (B D L)."""
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


# =========================================================================== #
# Hyena (HazyResearch/zoology, verbatim; pydantic decorator removed)
# =========================================================================== #

class OptimModule(nn.Module):
    def register(self, name, tensor, lr=None, wd=0.0):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {}
            if lr is not None:
                optim["lr"] = lr
            if wd is not None:
                optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


def fftconv_ref(u, k, D, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    out = y + u * D.unsqueeze(-1)
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    return out.to(dtype=u.dtype)


class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = (nn.Parameter(w * torch.ones(1, dim)) if train_freq
                     else w * torch.ones(1, dim))

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]
        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    def __init__(self, d_model, fast_decay_pct=0.3, slow_decay_pct=1.5, target=1e-2,
                 modulation_lr=0.0, shift: float = 0.0, **kwargs):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        decay = torch.exp(-t * self.deltas.abs())
        x = x * (decay + self.shift)
        return x


class Filter(OptimModule):
    def __init__(self, d_model, emb_dim=3, order=16, seq_len=1024, lr=1e-3,
                 lr_pos_emb=1e-5, dropout=0.0, w=1, wd=0, bias=True,
                 num_inner_mlps=2, linear_mixer=False, modulate: bool = True,
                 normalized=False, num_heads: int = 1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.modulate = modulate
        self.num_heads = num_heads
        self.use_bias = bias
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)
        act = Sin(dim=order, w=w)
        assert emb_dim % 2 != 0 and emb_dim >= 3, \
            "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)
        if linear_mixer is False:
            self.implicit_filter = [nn.Linear(emb_dim, order), act]
            for i in range(num_inner_mlps):
                self.implicit_filter.append(nn.Linear(order, order))
                self.implicit_filter.append(act)
            self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
            self.implicit_filter = nn.Sequential(*self.implicit_filter)
        else:
            self.implicit_filter = nn.Sequential(nn.Linear(emb_dim, d_model, bias=False))
        self.modulation = ExponentialModulation(d_model, **kwargs)
        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        if self.modulate:
            h = self.modulation(t, h)
        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None:
            k = self.filter(L)
        k = k[0] if type(k) is tuple else k
        if bias is None:
            bias = self.bias
        bias = bias if self.use_bias else 0 * bias
        y = fftconv_ref(x, k, bias, dropout_mask=None, gelu=False)
        return y.to(dtype=x.dtype)


class Hyena(nn.Module):
    NUM_PROJECTIONS = 3

    def __init__(self, d_model: int, l_max: int, filter_order: int = 64,
                 num_heads: int = 1, num_blocks: int = 1, outer_mixing: bool = False,
                 dropout: float = 0.0, filter_dropout: float = 0.0,
                 short_filter_order: int = 3, return_state: bool = False,
                 bidirectional: bool = False, layer_idx: int = None, **filter_args):
        super().__init__()
        assert d_model % num_heads == 0
        assert l_max % num_blocks == 0
        block_dim = l_max // num_blocks
        head_dim = d_model // num_heads
        self.d_model = d_model
        self.l_max = l_max
        self.num_heads = num_heads
        self.block_dim = block_dim
        self.head_dim = head_dim
        self.filter_order = filter_order
        self.short_filter_order = short_filter_order
        self.num_blocks = num_blocks
        self.filter_dropout = filter_dropout
        self.outer_mixing = outer_mixing
        self.return_state = return_state
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(self.d_model, self.NUM_PROJECTIONS * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.bidirectional = bidirectional
        total_width = self.d_model * self.NUM_PROJECTIONS
        self.short_filter = nn.Conv1d(in_channels=total_width, out_channels=total_width,
                                      kernel_size=self.short_filter_order, groups=total_width,
                                      padding=self.short_filter_order - 1)
        if "channels" not in filter_args:
            filter_args["channels"] = 1
        self.filter_fn = Filter(self.head_dim, order=self.filter_order, seq_len=self.l_max,
                                dropout=self.filter_dropout, bidirectional=self.bidirectional,
                                l_max=self.l_max, **filter_args)

    def forward(self, u, *args, **kwargs) -> torch.Tensor:
        l = u.size(1)
        assert l <= self.l_max, f"Input length {l} exceeds maximum length {self.l_max}"
        u = self.in_proj(u)
        u = rearrange(u, "b l d -> b d l")
        uc = self.short_filter(u)[..., :l]
        uc = rearrange(uc, "b (ho v) (z l) -> b ho v z l", z=self.num_blocks,
                       ho=self.num_heads, v=self.head_dim * self.NUM_PROJECTIONS)
        x1, x2, v = uc.split(self.d_model, dim=2)
        v = v * x1
        v = self.dropout(v)
        k = self.filter_fn.filter(l)
        k = rearrange(k, "c l d -> c d l")[0]
        v = self.filter_fn(v, l, k=k, bias=self.filter_fn.bias[None, :, None])
        v = v * x2
        y = rearrange(v, "b h v z l -> b (z l) (h v)", z=self.num_blocks, h=self.num_heads)
        y = self.out_proj(y)
        if self.return_state:
            return y, None
        return y


# =========================================================================== #
# Mamba (HazyResearch/zoology __init__ verbatim; pure-torch forward)
# =========================================================================== #

class Mamba(nn.Module):
    def __init__(self, d_model, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_rank: str = "auto", dt_min: float = 0.001, dt_max: float = 0.1,
                 dt_init: str = "random", dt_scale: float = 1.0, dt_init_floor: float = 1e-4,
                 conv_bias: bool = True, bias: bool = False, layer_idx=None,
                 device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                bias=conv_bias, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, **factory_kwargs)
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
                                bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                   "n -> d n", d=self.d_inner).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states):
        """hidden_states: (B, L, D) -> (B, L, D). Uses the Mamba CUDA kernels when
        installed, else the pure-PyTorch reference scan."""
        batch, seqlen, dim = hidden_states.shape
        xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                       "d (b l) -> b d l", l=seqlen)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)
        if _HAS_MAMBA_CUDA:
            x = _causal_conv1d_cuda(x, rearrange(self.conv1d.weight, "d 1 w -> d w"),
                                    self.conv1d.bias, None, self.activation)
        else:
            x = self.act(self.conv1d(x)[..., :seqlen])  # causal short conv
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        scan = _selective_scan_cuda if _HAS_MAMBA_CUDA else selective_scan_ref
        y = scan(x, dt, A, B, C, self.D.float(), z=z,
                 delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)


# =========================================================================== #
# Harness wrappers: (dim, heads) -> forward(x)
# =========================================================================== #

class HyenaMixer(nn.Module):
    """Hyena order-2, causal, num_heads=1 (Zoology defaults). heads is ignored
    (Hyena mixes channels globally via the implicit long filter)."""

    def __init__(self, dim: int, heads: int, seq_len: int):
        super().__init__()
        self.hyena = Hyena(d_model=dim, l_max=seq_len, filter_order=64, short_filter_order=3)

    def forward(self, x):
        return self.hyena(x)


class MambaMixer(nn.Module):
    """Mamba (Gu & Dao) with the official pure-PyTorch selective-scan reference.
    heads is ignored (Mamba is not multi-head)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.mamba = Mamba(d_model=dim)

    def forward(self, x):
        return self.mamba(x)
