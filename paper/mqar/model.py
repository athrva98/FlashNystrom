# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""The MQAR model, a port of HazyResearch/zoology's figure-2 MQAR model with the
sequence mixer made swappable across backends.

Structure follows zoology/experiments/paper_configs/iclr24_zoology_figure2/
configs.py, the config behind the published MQAR comparison (Arora et al.,
ICLR 2024):

  * ``n_layers=2``, and every layer is the SAME sequence mixer (zoology/model.py:243
    builds each block from ``config.sequence_mixer``), i.e. "hyena" is two Hyena
    layers. Selected by ``layer_layout="uniform"`` (the default here);
  * ``state_mixer = torch.nn.Identity`` -- no MLP (configs.py:145);
  * token embeddings tied to the output head; embedding dropout 0.1;
  * position embeddings for attention only (configs.py:142), generalized here to
    all permutation-equivariant backends -- see ``_POS_EMB_BACKENDS``.

``layer_layout="hybrid"`` (even layers BaseConv, odd the mixer) reproduces the
structure of Zoology's Based/Hybrid entry instead, and is what this file used
before being aligned to figure 2.

Swapping only the mixer is what isolates the operator:

  * ``sdpa``              : exact full attention via F.scaled_dot_product_attention.
  * ``flash_nystrom``     : the FlashNystrom Nystrom-attention CUDA kernels.
  * ``nystrom_reference`` : the same Nystrom math in pure PyTorch (no kernels).
  * ``linear_attention``, ``hyena``, ``mamba`` : the sub-quadratic baselines.

Two deliberate deviations from official MQAR, both forced and both applied
uniformly to every backend so none is advantaged:

  1. The attention layer is *bidirectional* (Zoology's MHA is causal).
     FlashNystrom has no causal kernel. MQAR stays solvable bidirectionally:
     every query token's bound value lies earlier in the sequence, so dropping
     the causal mask does not leak the answer. Paired with the predict-in-place
     labels in data.py (Zoology shifts by one for next-token prediction).
  2. Training runs in bf16, not Zoology's fp32: the FlashNystrom kernel exhausts
     shared memory in fp32 ("kernel3_scalar: insufficient smem"), so fp32 is not
     available to our own method. Every baseline runs in the same bf16 to keep
     the comparison controlled.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Attention backends (the swappable layer)
# --------------------------------------------------------------------------- #
class SdpaAttention(nn.Module):
    """Standard multi-head attention via scaled_dot_product_attention."""

    def __init__(self, dim: int, heads: int, causal: bool = False):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.causal = causal
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.heads, self.head_dim
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        o = o.transpose(1, 2).contiguous().view(B, N, self.dim)
        return self.out_proj(o)


class LinearAttention(nn.Module):
    """Bidirectional linear attention (Katharopoulos et al., 2020), feature map
    phi(x) = elu(x) + 1.

    Computes O = phi(Q) (phi(K)^T V) / (phi(Q) sum_j phi(K_j)) in O(N d^2), the
    canonical sub-quadratic attention approximation. Same projection structure as
    the other backends, so swapping it in is a controlled operator change. This is
    the linear-attention point in Zoology's MQAR comparison (Arora et al., 2023)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.heads, self.head_dim
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)
        qf = F.elu(q) + 1.0
        kf = F.elu(k) + 1.0
        kv = kf.transpose(-2, -1) @ v            # (B, H, D, D)
        z = kf.sum(dim=2)                        # (B, H, D)
        num = qf @ kv                            # (B, H, N, D)
        den = (qf @ z.unsqueeze(-1)) + 1e-6      # (B, H, N, 1)
        o = (num / den).transpose(1, 2).contiguous().view(B, N, self.dim)
        return self.out_proj(o)


class NystromReferenceAttention(nn.Module):
    """Pure-PyTorch Nystrom attention: the exact same Nystrom math as the
    FlashNystrom kernels (segment-mean landmarks, three softmaxes, FP32
    Newton-Schulz pseudoinverse, right-to-left contraction), but built from
    cuBLAS matmuls, torch.softmax, and autograd, with no custom kernels.

    Same projection structure as the other backends. Running this alongside
    the ``flash_nystrom`` backend separates two questions: a gap that appears
    here too is the Nystrom approximation (the math); a gap only in
    ``flash_nystrom`` is the kernels."""

    def __init__(
        self,
        dim: int,
        heads: int,
        num_landmarks: int = 64,
        newton_iter: int = 6,
        use_conv_residual: bool = False,
        conv_kernel_size: int = 3,
        kappa_star: float = 0.0,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.num_landmarks = num_landmarks
        self.newton_iter = newton_iter
        self.kappa_star = kappa_star
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        if use_conv_residual and conv_kernel_size > 0:
            self.conv_kernel_size = conv_kernel_size
            self.conv_weight = nn.Parameter(torch.randn(heads, conv_kernel_size) * 0.02)
        else:
            self.conv_kernel_size = 0
            self.conv_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from flash_nystrom.reference import nystrom_attention_reference

        B, N, _ = x.shape
        H, D = self.heads, self.head_dim
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        out = nystrom_attention_reference(
            q, k, v, self.num_landmarks, self.newton_iter,
            self.conv_weight, self.conv_kernel_size, kappa_star=self.kappa_star,
        )
        out = out.transpose(1, 2).contiguous().view(B, N, self.dim)
        return self.out_proj(out)


def build_attention(
    backend: str,
    dim: int,
    heads: int,
    causal: bool = False,
    num_landmarks: int = 64,
    newton_iter: int = 6,
    use_conv_residual: bool = False,
    kappa_star: float = 1.0e3,
    seq_len: int = None,
) -> nn.Module:
    if backend == "sdpa":
        return SdpaAttention(dim, heads, causal=causal)
    if backend in ("nystrom_reference", "nystrom_reference_compile"):
        if causal:
            raise ValueError("Nystrom attention does not support causal masking")
        mod = NystromReferenceAttention(
            dim, heads,
            num_landmarks=num_landmarks,
            newton_iter=newton_iter,
            use_conv_residual=use_conv_residual,
            kappa_star=kappa_star,
        )
        # nystrom_reference_compile: the same eager reference wrapped in
        # torch.compile (Inductor), the first optimization a practitioner
        # reaches for. It isolates the value of hand-fusion beyond what the
        # compiler gives for free. Inductor fuses the pointwise/softmax
        # epilogues around the cuBLAS matmuls but does not tile the
        # matmul-softmax-matmul chain into a register-resident kernel, so the
        # (N, m) intermediates still reach HBM.
        if backend == "nystrom_reference_compile":
            # dynamic=True compiles once and handles the autobatch batch-size
            # sweep without a recompile per shape.
            return torch.compile(mod, dynamic=True)
        return mod
    if backend in ("flash_nystrom", "flash_nystrom_tc"):
        if causal:
            raise ValueError("FlashNystrom does not support causal masking")
        from flash_nystrom import FlashNystromAttention
        from flash_nystrom.nystrom_config import NystromConfig

        # flash_nystrom    -> faithful scalar fp32 Newton-Schulz pinv (default).
        # flash_nystrom_tc -> opt-in tf32 tensor-core pinv (faster, small accuracy cost).
        cfg = NystromConfig(
            num_landmarks=num_landmarks,
            newton_iter=newton_iter,
            kappa_star=kappa_star,
            use_tc_pinv=(backend == "flash_nystrom_tc"),
            conv_kernel_size=3 if use_conv_residual else 0,
            use_conv_residual=use_conv_residual,
        )
        # FlashNystromAttention owns its own q/k/v/out projections, matching
        # SdpaAttention's structure.
        return FlashNystromAttention(dim, heads=heads, config=cfg)
    if backend == "linear_attention":
        if causal:
            raise ValueError("linear attention is bidirectional here; causal not implemented")
        return LinearAttention(dim, heads)
    if backend == "hyena":
        from .baselines import HyenaMixer
        if seq_len is None:
            raise ValueError("hyena requires seq_len (max sequence length for l_max)")
        return HyenaMixer(dim, heads, seq_len=seq_len)
    if backend == "mamba":
        from .baselines import MambaMixer
        return MambaMixer(dim, heads)
    raise ValueError(f"unknown backend {backend!r}")


# --------------------------------------------------------------------------- #
# Zoology's causal short-conv sequence mixer (the even layers)
# --------------------------------------------------------------------------- #
class ShortConvolution(nn.Module):
    """Causal depthwise short convolution (Zoology). Maps (B, L, D) -> (B, L, D).

    A grouped Conv1d left-padded by ``kernel_size - 1`` and truncated back to L,
    so output position t sees only inputs t-(k-1) .. t. This local causal mixing
    is what lets the model fold each key into its neighbouring value's
    representation, the mechanism that replaces learned position embeddings."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            dim, dim, kernel_size, groups=dim, padding=kernel_size - 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = x.size(1)
        return self.conv(x.transpose(1, 2))[..., :l].transpose(1, 2)


class BaseConv(nn.Module):
    """Zoology's gated short-conv mixer: ``conv(u) * proj(u) + u``."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.projection = nn.Linear(dim, dim)
        self.conv = ShortConvolution(dim, kernel_size)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.conv(u) * self.projection(u) + u


# Zoology figure 2 gives position embeddings to attention only
# (``max_position_embeddings=input_seq_len if sequence_mixer == "attention" else 0``,
# configs.py:142). Generalized here by the property that motivates it: a
# permutation-equivariant operator cannot recover order on its own, so every
# attention-family backend needs them. Hyena and Mamba carry order in their
# convolution / recurrence and get none, exactly as in figure 2.
_POS_EMB_BACKENDS = frozenset({
    "sdpa", "linear_attention", "nystrom_reference", "nystrom_reference_compile",
    "flash_nystrom", "flash_nystrom_tc",
})


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ResidualSublayer(nn.Module):
    """Pre-norm residual sublayer ``x + mixer(LayerNorm(x))``.

    Equivalent to Zoology's deferred-residual TransformerBlock with
    ``state_mixer = Identity`` (no MLP): each layer is a single normed mixer
    added back to the residual stream."""

    def __init__(self, dim: int, mixer: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mixer = mixer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class MQARModel(nn.Module):
    """Zoology's MQAR attention model with a swappable attention backend.

    Layers alternate conv / attention by index (depth 2 => conv, attention).
    Outputs per-position vocab logits; the MQAR loss is taken only at query
    positions (labels != -100)."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        dim: int = 128,
        depth: int = 2,
        heads: int = 2,
        backend: str = "sdpa",
        init: str = "normal",
        conv_kernel_size: int = 3,
        embed_dropout: float = 0.1,
        use_pos_emb: bool | None = None,
        layer_layout: str = "uniform",
        **attn_kw,
    ):
        super().__init__()
        if init not in ("normal", "orthogonal"):
            raise ValueError(f"init must be 'normal' or 'orthogonal', got {init!r}")
        if layer_layout not in ("uniform", "hybrid"):
            raise ValueError(f"layer_layout must be 'uniform' or 'hybrid', got {layer_layout!r}")
        self.init = init
        self.max_seq_len = max_seq_len
        self.layer_layout = layer_layout

        self.tok_emb = nn.Embedding(vocab_size, dim)
        # None => follow figure 2 (attention-family yes, Hyena/Mamba no).
        if use_pos_emb is None:
            use_pos_emb = backend in _POS_EMB_BACKENDS
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.embed_drop = nn.Dropout(embed_dropout)

        # uniform (Zoology figure 2): EVERY layer is the sequence mixer. model.py:243
        #   builds `block_cls(config, layer_idx=i) for i in range(n_layers)`, and each
        #   block instantiates the same `config.sequence_mixer`, so "hyena" means two
        #   Hyena layers, "attention" two MHA layers.
        # hybrid: even layers BaseConv, odd the mixer. This is the structure of
        #   Zoology's Based/Hybrid entry, not of the figure-2 pure-mixer runs.
        layers = []
        for i in range(depth):
            use_mixer = layer_layout == "uniform" or i % 2 == 1
            mixer = (
                build_attention(backend, dim, heads, seq_len=max_seq_len, **attn_kw)
                if use_mixer else BaseConv(dim, conv_kernel_size)
            )
            layers.append(ResidualSublayer(dim, mixer))
        self.layers = nn.ModuleList(layers)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)
        # Tie the output head to the token embeddings (Zoology default).
        self.head.weight = self.tok_emb.weight

    def _init_weights(self, module: nn.Module):
        # Linear weights use the chosen scheme; embeddings stay normal(std=0.02),
        # matching Zoology's default init. Conv1d keeps PyTorch defaults (Zoology
        # only re-inits Linear and Embedding). Orthogonal init makes each
        # projection orthonormal (the head-independence ablation).
        if isinstance(module, nn.Linear):
            if self.init == "orthogonal":
                nn.init.orthogonal_(module.weight)
            else:
                nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def encode(self, idx: torch.Tensor) -> torch.Tensor:
        """Run the body and final norm, returning hidden states (B, N, dim).

        Kept separate from the head so the caller can apply the (vocab-sized)
        head to only the query positions: the MQAR loss/eval touch ~16 of 256
        positions, so this avoids materializing a (B, N, vocab) logits tensor."""
        B, N = idx.shape
        if self.use_pos_emb and N > self.max_seq_len:
            raise ValueError(f"sequence length {N} exceeds max_seq_len {self.max_seq_len}")
        h = self.tok_emb(idx)
        if self.use_pos_emb:
            pos = torch.arange(N, device=idx.device)
            h = h + self.pos_emb(pos)[None, :, :]
        h = self.embed_drop(h)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(idx))  # (B, N, vocab_size)
