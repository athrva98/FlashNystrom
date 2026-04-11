# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

"""
Train a small transformer language model using FlashNystrom attention.
Demonstrates end-to-end training with CUDA backward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
import time
import sys

sys.path.insert(0, ".")

from flash_nystrom import FlashNystromAttention, NystromConfig


class NystromTransformerBlock(nn.Module):
    def __init__(self, dim, heads, nystrom_config):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashNystromAttention(dim, heads, nystrom_config)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class NystromLM(nn.Module):
    def __init__(self, vocab_size, dim, heads, n_layers, max_seq_len, nystrom_config):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList(
            [
                NystromTransformerBlock(dim, heads, nystrom_config)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


class CopyDataset(Dataset):
    """Task: given a sequence, predict the next token (shifted by 1)."""

    def __init__(self, num_samples, seq_len, vocab_size):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]  # input, target


def train():
    # Hyperparameters
    vocab_size = 256
    dim = 128  # head_dim = 128/2 = 64
    heads = 2
    n_layers = 2
    seq_len = 512
    batch_size = 4
    num_samples = 200
    num_epochs = 10
    lr = 3e-4

    device = torch.device("cuda")

    nystrom_config = NystromConfig(
        num_landmarks=32,
        newton_iter=6,
        conv_kernel_size=0,
        use_conv_residual=False,
    )

    model = NystromLM(vocab_size, dim, heads, n_layers, seq_len, nystrom_config).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    dataset = CopyDataset(num_samples, seq_len, vocab_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_layers} layers, dim={dim}, heads={heads}, D={dim // heads}")
    print(f"Parameters: {num_params:,}")
    print(f"Nystrom: m={nystrom_config.num_landmarks}, seq_len={seq_len}")
    print(f"Dataset: {num_samples} samples, vocab={vocab_size}")
    print()

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)  # (B, T, vocab)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            epoch_tokens += inputs.numel()
            global_step += 1

        dt = time.time() - t0
        avg_loss = epoch_loss / len(dataset)
        tok_per_sec = epoch_tokens / dt

        print(
            f"Epoch {epoch + 1:2d}/{num_epochs} | loss={avg_loss:.4f} | "
            f"grad_norm={grad_norm:.2f} | "
            f"{tok_per_sec:.0f} tok/s | {dt:.2f}s"
        )

    # Final evaluation: check if model learned anything
    model.eval()
    with torch.no_grad():
        sample_input = dataset.data[:1, :-1].to(device)
        logits = model(sample_input)
        pred = logits.argmax(dim=-1)
        target = dataset.data[:1, 1:].to(device)
        accuracy = (pred == target).float().mean().item()
        print(f"\nFinal accuracy on first sample: {accuracy:.2%}")
        print(f"Random baseline: {1 / vocab_size:.2%}")

    if avg_loss < 5.0:
        print(
            "\nSUCCESS: Model trained and loss decreased from ~5.5 (random) to {:.2f}".format(
                avg_loss
            )
        )
    else:
        print("\nModel did not converge significantly (loss={:.2f})".format(avg_loss))


if __name__ == "__main__":
    train()
