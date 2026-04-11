# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

"""
Train a ~6M parameter Vision Transformer on CIFAR-10 using FlashNystrom attention.

Architecture:
  - Patch embedding: 4x4 patches -> 32x32 image = 8x8 = 64 patches
  - CLS token prepended -> 65 tokens
  - Transformer: 8 layers, dim=256, 4 heads (head_dim=64), Nystrom m=32
  - Classification head on CLS token

Target: ~6M parameters, >85% accuracy in reasonable training time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import time
import math
import sys
import os

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(
    sys.stdout, "reconfigure"
) else None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from flash_nystrom import FlashNystromAttention, NystromConfig


class PatchEmbed(nn.Module):
    """Split image into patches and project to embedding dimension."""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # 64
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, 3, 32, 32) -> (B, embed_dim, 8, 8) -> (B, 64, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class NystromBlock(nn.Module):
    def __init__(self, dim, heads, nystrom_config, drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashNystromAttention(dim, heads, nystrom_config)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(dim * 4, dim),
            nn.Dropout(drop_rate),
        )
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class NystromViT(nn.Module):
    """
    Vision Transformer with FlashNystrom attention for CIFAR-10.

    ~6M params with dim=256, depth=8, heads=4, patch_size=4.
    """

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=8,
        heads=4,
        num_landmarks=32,
        drop_rate=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches  # 64

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(drop_rate)

        nystrom_config = NystromConfig(
            num_landmarks=num_landmarks,
            newton_iter=6,
            conv_kernel_size=0,
            use_conv_residual=False,
        )

        self.blocks = nn.ModuleList(
            [
                NystromBlock(embed_dim, heads, nystrom_config, drop_rate)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, 64, dim)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 65, dim)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # CLS token
        return self.head(cls_out)


# ============================================================================
# Training utilities
# ============================================================================


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [max(self.min_lr, base_lr * factor) for base_lr in self.base_lrs]


def get_cifar10_loaders(batch_size=128, num_workers=0, data_dir="./data"):
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total, total_loss / total


# ============================================================================
# Main
# ============================================================================


def main():
    # Config
    device = torch.device("cuda")
    epochs = 30
    batch_size = 128
    lr = 1e-3
    weight_decay = 0.05
    warmup_epochs = 5
    embed_dim = 256
    depth = 8
    heads = 4
    num_landmarks = 32

    print("=" * 70)
    print("FlashNystrom ViT — CIFAR-10 Training")
    print("=" * 70)

    # Data
    train_loader, test_loader = get_cifar10_loaders(batch_size)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    # Model
    model = NystromViT(
        embed_dim=embed_dim,
        depth=depth,
        heads=heads,
        num_landmarks=num_landmarks,
        drop_rate=0.1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"Architecture: depth={depth}, dim={embed_dim}, heads={heads}, D={embed_dim // heads}"
    )
    print(f"Nystrom: m={num_landmarks}, seq_len=65 (64 patches + CLS)")
    print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
    print(f"Training: {epochs} epochs, batch={batch_size}, lr={lr}")
    print()

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

    # Training loop
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = (
                images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * labels.size(0)
            epoch_correct += (logits.argmax(1) == labels).sum().item()
            epoch_total += labels.size(0)

        dt = time.time() - t0
        train_acc = epoch_correct / epoch_total
        train_loss = epoch_loss / epoch_total

        # Evaluate every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            test_acc, test_loss = evaluate(model, test_loader, device)
            best_acc = max(best_acc, test_acc)
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.1%} | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.1%} | "
                f"best={best_acc:.1%} | {dt:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.1%} | "
                f"{dt:.1f}s"
            )

    print()
    print(f"{'=' * 70}")
    print(f"Final test accuracy: {best_acc:.1%}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
