# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Genomic sequence classification: a second bidirectional domain.

The vision experiments establish that the fused kernel is accuracy-neutral on
the canonical bidirectional workload. This adds a domain with different
statistics -- a 4-symbol alphabet, no spatial locality prior, and label
evidence that can sit anywhere in the sequence -- to test that the conclusion
is not specific to images.

DNA is bidirectional in the strict sense the paper cares about: a regulatory
element is read in both directions and its effect does not depend on a scan
order, so a causal mask would discard half the context for no modelling
reason. That is why this domain belongs to the bidirectional family rather
than to the causal-LM family the SSM literature targets.

Task: binary classification of human enhancer / promoter sequences against
length-matched background, one-hot over {A,C,G,T} tokenized per base. The
sequence IS the context, so the model must attend across the whole window --
exactly the regime where the operator choice matters.

Only the attention operator changes between arms; everything else (tokenizer,
backbone, optimizer, schedule, seeds) is held fixed, so a difference in test
accuracy is attributable to the operator.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}


def synth_regulatory_dataset(num_examples: int, seq_len: int, seed: int = 0,
                             motif_len: int = 8, n_motifs: int = 3):
    """A controlled stand-in for regulatory-element detection.

    Positive sequences carry `n_motifs` copies of a fixed motif at random
    positions in an i.i.d. background; negatives carry the same number of
    RANDOM k-mers, so the two classes match in length, base composition and
    k-mer count. The only signal is the specific motif, placed anywhere in the
    window, which forces the model to attend across the full sequence rather
    than exploit position or composition.

    Synthetic rather than a downloaded corpus so the experiment is
    reproducible without a data dependency and the signal is known exactly;
    the point here is to compare operators under identical data, not to claim
    a genomics result.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 4, (num_examples, seq_len), generator=g)
    y = torch.zeros(num_examples, dtype=torch.long)
    y[: num_examples // 2] = 1
    motif = torch.randint(0, 4, (motif_len,), generator=g)

    for i in range(num_examples):
        for _ in range(n_motifs):
            pos = int(torch.randint(0, seq_len - motif_len, (1,), generator=g))
            if y[i] == 1:
                x[i, pos:pos + motif_len] = motif           # the real motif
            else:
                x[i, pos:pos + motif_len] = torch.randint(   # matched decoy
                    0, 4, (motif_len,), generator=g)
    perm = torch.randperm(num_examples, generator=g)
    return x[perm], y[perm]


class DNAClassifier(nn.Module):
    """Two-layer bidirectional encoder over DNA tokens, mean-pooled to a label.

    Deliberately the same shape as the vision backbone: only the attention
    operator is swappable, and no causal mask is applied anywhere."""

    def __init__(self, seq_len: int, dim: int = 128, depth: int = 2,
                 heads: int = 2, backend: str = "sdpa", num_landmarks: int = 64):
        super().__init__()
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from paper.mqar.model import build_attention

        self.emb = nn.Embedding(4, dim)
        self.pos = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            attn = build_attention(backend, dim, heads, seq_len=seq_len,
                                   num_landmarks=num_landmarks, kappa_star=0.0)
            self.blocks.append(nn.ModuleDict({
                "norm": nn.LayerNorm(dim),
                "attn": attn,
                "norm2": nn.LayerNorm(dim),
                "mlp": nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                     nn.Linear(4 * dim, dim)),
            }))
        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)

    def forward(self, idx):
        N = idx.size(1)
        h = self.emb(idx) + self.pos(torch.arange(N, device=idx.device))[None]
        for b in self.blocks:
            h = h + b["attn"](b["norm"](h))
            h = h + b["mlp"](b["norm2"](h))
        return self.head(self.norm_f(h).mean(dim=1))       # mean-pool: bidirectional


def train_eval(backend, seq_len=4096, dim=128, heads=2, num_landmarks=64,
               n_train=4096, n_test=1024, epochs=8, batch_size=16, lr=3e-4,
               seed=0, device="cuda", dtype=torch.bfloat16):
    """Train one arm and return best test accuracy. Only `backend` varies."""
    torch.manual_seed(seed)
    xtr, ytr = synth_regulatory_dataset(n_train, seq_len, seed=seed)
    xte, yte = synth_regulatory_dataset(n_test, seq_len, seed=seed + 10_000)
    model = DNAClassifier(seq_len, dim, 2, heads, backend, num_landmarks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = 0.0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                xb = xte[i:i + batch_size].to(device)
                yb = yte[i:i + batch_size].to(device)
                with torch.autocast("cuda", dtype=dtype):
                    pred = model(xb).float().argmax(-1)
                correct += (pred == yb).sum().item()
        acc = correct / n_test * 100
        best = max(best, acc)
        print(f"    [{backend}] epoch {ep+1}/{epochs} loss {loss.item():.4f} "
              f"test acc {acc:.2f}% (best {best:.2f}%)", flush=True)
    return best
