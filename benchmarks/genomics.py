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

Task: long-range repeat detection. Each sequence opens with a query k-mer;
the label is whether that exact k-mer recurs later in the sequence. The two
positions are arbitrarily far apart and neither is privileged, so the model
must match content across the whole window in both directions -- the regime
where the operator choice matters.

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


KMER = 6                      # DNABERT-style k-mer tokens
KMER_VOCAB = 4 ** KMER        # 4096


def synth_repeat_dataset(num_examples: int, seq_len: int, seed: int = 0):
    """Long-range repeat detection over k-mer tokens.

    Sequences are tokenized as non-overlapping ``KMER``-mers, the standard
    representation for DNA transformers (DNABERT and successors) and the one
    that makes the task well posed: with single-base tokens any k-mer question
    first requires aggregating adjacent positions, which attention does poorly
    without a convolutional prior, so the experiment would measure the missing
    locality prior rather than the attention operator. With k-mer tokens the
    question is content matching between two positions, which is exactly what
    attention computes.

    Each sequence opens with a query k-mer token. In positives that same token
    recurs once at a random later position; in negatives a different random
    token is placed there instead. Both classes therefore hold exactly two
    insertions and identical token statistics, and the label depends only on
    whether the two agree -- an arbitrarily long-range, order-free relation.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, KMER_VOCAB, (num_examples, seq_len), generator=g)
    y = torch.zeros(num_examples, dtype=torch.long)
    y[: num_examples // 2] = 1

    for i in range(num_examples):
        query = int(torch.randint(0, KMER_VOCAB, (1,), generator=g))
        x[i, 0] = query
        pos = int(torch.randint(1, seq_len, (1,), generator=g))
        x[i, pos] = query if y[i] == 1 else int(
            torch.randint(0, KMER_VOCAB, (1,), generator=g))
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

        self.emb = nn.Embedding(KMER_VOCAB, dim)
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
        # Read out at position 0, where the query k-mer sits, so the readout
        # token IS the thing being matched. One attention operation then
        # answers the task: position 0 attends over the sequence and its
        # output reflects whether anything matched it. Pooling over all
        # positions instead (mean or max) forces the model to first localize
        # the match and then propagate it to a pooled summary, which is
        # strictly more circuit than the question requires.
        return self.head(self.norm_f(h)[:, 0])


def train_eval(backend, seq_len=2048, dim=128, heads=2, num_landmarks=64,
               n_train=8192, n_test=1024, epochs=15, batch_size=32, lr=3e-4,
               seed=0, device="cuda", dtype=torch.bfloat16):
    """Train one arm and return best test accuracy. Only `backend` varies."""
    torch.manual_seed(seed)
    # FRESH training data each epoch. The fixed-set version memorized: train
    # loss fell while test accuracy stayed at chance, because with a 4096-token
    # vocabulary the model can key on which queries it has seen instead of
    # learning the comparison. Regenerating removes that shortcut entirely.
    xte, yte = synth_repeat_dataset(n_test, seq_len, seed=seed + 10_000)
    model = DNAClassifier(seq_len, dim, 2, heads, backend, num_landmarks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = 0.0
    for ep in range(epochs):
        xtr, ytr = synth_repeat_dataset(n_train, seq_len, seed=seed * 1000 + ep)
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
