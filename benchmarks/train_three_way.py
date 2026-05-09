# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

# Three-way training comparison:
#   1. SDPA (PyTorch FA-2 backend) — exact attention, baseline
#   2. Nystrom reference (pure PyTorch) — Nyström algorithm, no custom kernels
#   3. FlashNystrom CUDA — Nyström algorithm via our fused kernels
#
# This isolates: (a) is the gap caused by Nyström approximation itself, or
# (b) by something specific to our kernels (e.g. FP16 backward noise)?
import sys, time, json
sys.path.insert(0, "C:/Users/athrv/Documents/FlashNystrom/benchmarks")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from flash_nystrom import FlashNystromAttention, NystromConfig
from flash_nystrom.reference import nystrom_attention_reference


class SDPAAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, _ = x.shape
        H, D = self.heads, self.head_dim
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, N, -1))


class NystromRefAttention(nn.Module):
    """Pure-PyTorch Nyström attention — same algorithm as FlashNystrom but no custom kernels."""
    def __init__(self, dim, heads, num_landmarks=32, newton_iter=20, conv_kernel_size=0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.m = num_landmarks
        self.newton_iter = newton_iter
        self.conv_kernel_size = conv_kernel_size
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        if conv_kernel_size > 0:
            self.conv_weight = nn.Parameter(
                torch.randn(heads, conv_kernel_size) * 0.02)
        else:
            self.conv_weight = None

    def forward(self, x):
        B, N, _ = x.shape
        H, D = self.heads, self.head_dim
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        cw = self.conv_weight.to(q.dtype) if self.conv_weight is not None else None
        out = nystrom_attention_reference(q, k, v, self.m, self.newton_iter, cw, self.conv_kernel_size)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, N, -1))


class TinyViT(nn.Module):
    def __init__(self, attn_module_factory, dim=256, depth=4, heads=4,
                 patch_size=4, num_classes=10):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        n_patches = (32 // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "attn": attn_module_factory(dim, heads),
                "norm1": nn.LayerNorm(dim),
                "ff": nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                "norm2": nn.LayerNorm(dim),
            }) for _ in range(depth)
        ])
        self.head = nn.Linear(dim, num_classes)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = x + blk["attn"](blk["norm1"](x))
            x = x + blk["ff"](blk["norm2"](x))
        return self.head(self.norm(x[:, 0]))


def train_one(label, attn_factory, epochs=20, batch_size=128, lr=1e-3,
              patch_size=4, dim=256, heads=4):
    transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
                           T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    transform_test = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    trainset = torchvision.datasets.CIFAR10(
        root="C:/Users/athrv/Documents/FlashNystrom/data",
        train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root="C:/Users/athrv/Documents/FlashNystrom/data",
        train=False, download=False, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=200, shuffle=False, num_workers=0,
        pin_memory=True, drop_last=True)

    torch.manual_seed(42)
    model = TinyViT(attn_factory, dim=dim, depth=4, heads=heads, patch_size=patch_size).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    nparams = sum(p.numel() for p in model.parameters())
    n_tokens = (32 // patch_size) ** 2 + 1
    print(f"\n{label}: {nparams/1e6:.2f}M params, N={n_tokens} tokens")

    times = []
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in trainloader:
            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(imgs)
                loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        scheduler.step()
        times.append(time.time() - t0)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Ep {epoch+1:>2}/{epochs}: loss={total_loss/total:.4f} "
                  f"train_acc={100*correct/total:.1f}% time={times[-1]:.1f}s")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    test_acc = 100 * correct / total
    avg_time = sum(times) / len(times)
    print(f"  Final test_acc={test_acc:.1f}%  avg_epoch={avg_time:.1f}s")
    return {"label": label, "test_acc": test_acc, "avg_epoch": avg_time, "params_M": nparams/1e6}


def main():
    print("="*70)
    print("Three-way training comparison on CIFAR-10")
    print("="*70)
    results = []

    print("\n--- 1. SDPA (PyTorch FA-2 backend, exact attention) ---")
    results.append(train_one("SDPA", lambda d, h: SDPAAttention(d, h)))

    print("\n--- 2. Nystrom reference (pure PyTorch, no custom kernels) ---")
    results.append(train_one("Nystrom-Ref",
        lambda d, h: NystromRefAttention(d, h, num_landmarks=32, newton_iter=6, conv_kernel_size=0)))

    print("\n--- 3. FlashNystrom CUDA (our fused kernels) ---")
    cfg = NystromConfig(num_landmarks=32, conv_kernel_size=0, use_conv_residual=False)
    results.append(train_one("FlashNystrom",
        lambda d, h: FlashNystromAttention(d, h, cfg)))

    print("\n" + "="*70)
    print("Summary:")
    for r in results:
        print(f"  {r['label']:>14}: test_acc={r['test_acc']:.1f}%  "
              f"avg_epoch={r['avg_epoch']:.1f}s  params={r['params_M']:.2f}M")

    with open("three_way_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to three_way_results.json")


if __name__ == "__main__":
    main()
