# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Genomic sequence modeling: the second bidirectional domain after vision.

One backbone, one optimizer, one schedule, one seed stream. The ONLY thing that
differs between arms is the attention operator, so a difference in accuracy is
attributable to the operator and nothing else. No arm is causally masked: DNA
has no scan order, and masking some arms but not others would measure the
masking regime instead of the operator.

Tasks live in genomics_data.py with their protocol sources; this file is the
model and the training loop.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.genomics_data import (                              # noqa: E402
    DNA_VOCAB_SIZE, KMER_VOCAB, SpeciesDataset, load_genomic_benchmark,
    synth_repeat_dataset,
)


class DNABackbone(nn.Module):
    """Conv stem, then a stack of bidirectional attention blocks.

    The stem is a depthwise convolution with a CENTERED (non-causal) receptive
    field. Base-resolution DNA models universally carry a convolutional front
    end (Enformer, Basenji, DNABERT-2's tokenizer, and HyenaDNA is convolutional
    throughout) because a k-mer motif is a local pattern that attention over
    single nucleotides has no cheap way to assemble. It is identical across
    arms, so it does not confound the comparison; it just stops the experiment
    from measuring a missing locality prior instead of the attention operator.
    """

    def __init__(self, vocab_size: int, seq_len: int, dim: int = 128,
                 depth: int = 2, heads: int = 2, backend: str = "sdpa",
                 num_landmarks: int = 64, conv_kernel: int = 9,
                 use_pos_emb: bool = True):
        super().__init__()
        from paper.mqar.model import build_attention

        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(seq_len, dim) if use_pos_emb else None
        self.stem = nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel // 2,
                              groups=dim) if conv_kernel else None
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "norm": nn.LayerNorm(dim),
                "attn": build_attention(backend, dim, heads, seq_len=seq_len,
                                        num_landmarks=num_landmarks,
                                        kappa_star=0.0),
                "norm2": nn.LayerNorm(dim),
                "mlp": nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                     nn.Linear(4 * dim, dim)),
            }))
        self.norm_f = nn.LayerNorm(dim)

    def forward(self, idx):
        h = self.emb(idx)
        if self.pos is not None:
            h = h + self.pos(torch.arange(idx.size(1), device=idx.device))[None]
        if self.stem is not None:
            h = h + self.stem(h.transpose(1, 2)).transpose(1, 2)
        for b in self.blocks:
            h = h + b["attn"](b["norm"](h))
            h = h + b["mlp"](b["norm2"](h))
        return self.norm_f(h)


class DNAClassifier(nn.Module):
    """Sequence classification by masked mean pooling, the standard readout for
    bidirectional encoders (and what Caduceus uses for its downstream heads)."""

    def __init__(self, num_classes: int, **kw):
        super().__init__()
        self.backbone = DNABackbone(**kw)
        self.head = nn.Linear(kw.get("dim", 128), num_classes)

    def forward(self, idx, mask=None):
        h = self.backbone(idx)
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            m = mask[..., None].to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled)


class DNAPointer(nn.Module):
    """Needle retrieval: score every position, softmax over positions.

    A pointer readout, not a pooled one. The task is "where is the match", so
    the output space is the sequence itself and the supervision is log2(L) bits
    per example rather than one."""

    def __init__(self, **kw):
        super().__init__()
        self.backbone = DNABackbone(**kw)
        self.score = nn.Linear(kw.get("dim", 128), 1)

    def forward(self, idx):
        logits = self.score(self.backbone(idx)).squeeze(-1)   # (B, L)
        # Position 0 holds the query itself and is never the answer. masked_fill
        # is out-of-place and uses the dtype floor rather than -inf, so the
        # softmax backward cannot produce a NaN from inf * 0.
        block = torch.zeros(1, logits.size(1), dtype=torch.bool,
                            device=logits.device)
        block[0, 0] = True
        return logits.masked_fill(block, torch.finfo(logits.dtype).min)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def _unpack(batch):
    """(x, y) from a DataLoader (a list) or (x, y, mask) from _batches."""
    if len(batch) == 2:
        return batch[0], batch[1], None
    return batch[0], batch[1], batch[2]


def _loaders_species(seq_len, n_train, n_test, seed, species_dir, species,
                     chroms_per_split, batch_size, num_workers=4):
    tr = SpeciesDataset(species_dir, "train", seq_len, n_train, species,
                        chroms_per_split, seed=seed)
    te = SpeciesDataset(species_dir, "test", seq_len, n_test, species,
                        chroms_per_split, seed=seed + 10_000)
    mk = lambda d, sh: torch.utils.data.DataLoader(
        d, batch_size=batch_size, shuffle=sh, num_workers=num_workers,
        pin_memory=True, drop_last=False)
    return mk(tr, True), mk(te, False), len(tr.species)


def train_eval(backend, task="species", seq_len=1024, dim=128, heads=2,
               num_landmarks=64, epochs=20, batch_size=32, lr=3e-4, seed=0,
               device="cuda", dtype=torch.bfloat16, depth=2,
               n_train=32768, n_test=4096, species_dir="data/genomes",
               species=None, chroms_per_split=4, gb_dataset=None,
               variant="pointer", weight_decay=0.1, log_every=1):
    """Train one arm on one task and return its best test metric (percent).

    task="species"            HyenaDNA species classification, real genomes
    task="genomic_benchmarks" one Grešová et al. dataset (``gb_dataset``)
    task="repeat"             the synthetic diagnostic (``variant``)
    """
    torch.manual_seed(seed)
    # Device-aware so the pipeline can be smoke-tested on CPU; bf16 autocast
    # is a no-op there, which keeps the GPU numerics path unchanged.
    dev_type = torch.device(device).type
    amp = (dev_type == "cuda")
    is_pointer = (task == "repeat" and variant == "pointer")

    # ---- data -------------------------------------------------------------
    train_loader = test_loader = None
    if task == "species":
        train_loader, test_loader, n_cls = _loaders_species(
            seq_len, n_train, n_test, seed, species_dir, species,
            chroms_per_split, batch_size)
        vocab, use_pos = DNA_VOCAB_SIZE, True
    elif task == "genomic_benchmarks":
        if not gb_dataset:
            raise ValueError("task='genomic_benchmarks' needs gb_dataset=<name>")
        xtr, ytr, mtr = load_genomic_benchmark(gb_dataset, "train")
        xte, yte, mte = load_genomic_benchmark(gb_dataset, "test",
                                               max_len=xtr.size(1))
        seq_len = xtr.size(1)
        n_cls = int(max(ytr.max(), yte.max())) + 1
        vocab, use_pos = DNA_VOCAB_SIZE, True
    elif task == "repeat":
        xte, yte = synth_repeat_dataset(n_test, seq_len, seed=seed + 10_000,
                                        variant=variant)
        mtr = mte = None
        n_cls = seq_len if is_pointer else 2
        vocab, use_pos = KMER_VOCAB, True
    else:
        raise ValueError(f"unknown task {task!r}")

    # ---- model ------------------------------------------------------------
    kw = dict(vocab_size=vocab, seq_len=seq_len, dim=dim, depth=depth,
              heads=heads, backend=backend, num_landmarks=num_landmarks,
              use_pos_emb=use_pos)
    model = (DNAPointer(**kw) if is_pointer
             else DNAClassifier(num_classes=n_cls, **kw)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    chance = 100.0 / (seq_len - 1) if is_pointer else 100.0 / n_cls

    def _batches(x, y, m, bs, shuffle, gen=None):
        order = torch.randperm(x.size(0), generator=gen) if shuffle \
            else torch.arange(x.size(0))
        for i in range(0, x.size(0), bs):
            j = order[i:i + bs]
            yield x[j], y[j], (m[j] if m is not None else None)

    best = 0.0
    for ep in range(epochs):
        model.train()
        if task == "repeat":
            # Fresh draws per epoch. Safe here (unlike the old fixed-set
            # version) because the task is now retrieval with a unique answer:
            # there is no query-identity shortcut left to memorize.
            xtr, ytr = synth_repeat_dataset(n_train, seq_len,
                                            seed=seed * 1000 + ep, variant=variant)
        gen = torch.Generator().manual_seed(seed * 7919 + ep)

        src = (train_loader if task == "species"
               else _batches(xtr, ytr, mtr, batch_size, True, gen))
        last = math.nan
        for batch in src:
            xb, yb, mb = _unpack(batch)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            mb = mb.to(device) if mb is not None else None
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev_type, enabled=amp, dtype=dtype):
                out = model(xb) if is_pointer else model(xb, mb)
                loss = F.cross_entropy(out.float(), yb)
            loss.backward()
            opt.step()
            last = loss.item()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            ev = (test_loader if task == "species"
                  else _batches(xte, yte, mte, batch_size, False))
            for batch in ev:
                xb, yb, mb = _unpack(batch)
                xb, yb = xb.to(device), yb.to(device)
                mb = mb.to(device) if mb is not None else None
                with torch.autocast(dev_type, enabled=amp, dtype=dtype):
                    out = model(xb) if is_pointer else model(xb, mb)
                correct += (out.float().argmax(-1) == yb).sum().item()
                total += yb.numel()
        acc = correct / max(total, 1) * 100
        best = max(best, acc)
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f"    [{backend}] epoch {ep+1}/{epochs} loss {last:.4f} "
                  f"test {acc:.2f}% (best {best:.2f}%, chance {chance:.2f}%)",
                  flush=True)
    return best
