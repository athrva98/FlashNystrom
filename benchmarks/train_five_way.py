# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

# Five-way training comparison to isolate forward vs backward bugs:
#   1. SDPA                              (PyTorch FA-2 backend, exact attention)
#   2. Nystrom-Ref                       (full PyTorch Nystrom: torch fwd + torch bwd)
#   3. FlashNystrom                      (full FN: CUDA fwd + CUDA bwd)
#   4. FN-fwd + torch-bwd                (CUDA forward, PyTorch reference backward)
#   5. torch-fwd + FN-bwd                (PyTorch reference forward, CUDA backward)
#
# If config 4 matches Nystrom-Ref → CUDA forward is correct.
# If config 5 matches Nystrom-Ref → CUDA backward is correct.
# If config 4 differs → bug in FN forward.
# If config 5 differs → bug in FN backward.
import sys, time, json
sys.path.insert(0, "C:/Users/athrv/Documents/FlashNystrom/benchmarks")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import flash_nystrom._C as _C
from flash_nystrom import FlashNystromAttention, NystromConfig
from flash_nystrom.flash_nystrom import FlashNystromFunction
from flash_nystrom.reference import nystrom_attention_reference


# -------------------------------------------------------------------------
# Config 4: FlashNystrom forward, PyTorch reference backward.
# Forward value comes from FN CUDA kernels.
# Gradient is computed by PyTorch autograd through the reference Nystrom code,
# evaluated at the same (Q, K, V).
# -------------------------------------------------------------------------
class FNFwdTorchBwdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, m, niter):
        ctx.save_for_backward(q, k, v)
        ctx.m, ctx.niter = m, niter
        # FN forward (no autograd graph attached; we'll handle backward manually)
        with torch.no_grad():
            results = _C.forward(q, k, v, m, niter, 0, None)
        return results[0]  # output

    @staticmethod
    def backward(ctx, dout):
        q, k, v = ctx.saved_tensors
        # Recompute the reference forward under autograd tracking.
        with torch.enable_grad():
            qg = q.detach().requires_grad_(True)
            kg = k.detach().requires_grad_(True)
            vg = v.detach().requires_grad_(True)
            ref_out = nystrom_attention_reference(
                qg, kg, vg, num_landmarks=ctx.m, newton_iter=ctx.niter,
                conv_weight=None, conv_kernel_size=0)
            dq, dk, dv = torch.autograd.grad(ref_out, (qg, kg, vg), grad_outputs=dout)
        return dq, dk, dv, None, None


# -------------------------------------------------------------------------
# Config 5: PyTorch reference forward, FlashNystrom backward.
# Forward value comes from the PyTorch reference.
# Gradient is computed by FN's CUDA backward kernels using the FN-saved
# tensors (we have to run FN forward under-the-hood to obtain those, even
# though we discard its output value).
# -------------------------------------------------------------------------
class TorchFwdFNBwdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, m, niter):
        # 1. Reference forward (no autograd) for the value we return.
        with torch.no_grad():
            ref_out = nystrom_attention_reference(
                q, k, v, num_landmarks=m, newton_iter=niter,
                conv_weight=None, conv_kernel_size=0)

        # 2. FN forward to obtain the saved tensors needed by FN backward.
        with torch.no_grad():
            results = _C.forward(q, k, v, m, niter, 0, None)
            # [output, q_s, k_s, q_tilde, k_tilde, k2inv, step2,
            #  lse1, lse2, lse3, ns_iterates, k2_softmax]
            fn_out = results[0]

        ctx.save_for_backward(*results[1:], v, fn_out)
        ctx.m, ctx.niter = m, niter
        return ref_out

    @staticmethod
    def backward(ctx, dout):
        saved = ctx.saved_tensors
        q_s, k_s, q_tilde, k_tilde, k2_inv, step2 = saved[0:6]
        lse1, lse2, lse3 = saved[6:9]
        ns_iterates, k2_softmax = saved[9], saved[10]
        v, fn_out = saved[11], saved[12]
        results = _C.backward(
            dout.contiguous(),
            q_s, k_s, q_tilde, k_tilde, k2_inv, step2,
            lse1, lse2, lse3, ns_iterates, k2_softmax,
            v, fn_out,
            ctx.m, ctx.niter, 0, None,
        )
        dQ, dK, dV = results[0], results[1], results[2]
        return dQ, dK, dV, None, None


# -------------------------------------------------------------------------
# Attention modules (one per config) — same projection structure, different
# attention routine.
# -------------------------------------------------------------------------
class SDPAAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
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


class _NystromBase(nn.Module):
    """Shared projection layout for the four Nystrom variants."""
    def __init__(self, dim, heads, num_landmarks, newton_iter):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.m, self.niter = num_landmarks, newton_iter
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _qkv(self, x):
        B, N, _ = x.shape
        H, D = self.heads, self.head_dim
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        return q, k, v, B, N

    def _project_out(self, out, B, N):
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, N, -1))


class NystromRefAttention(_NystromBase):
    """Full PyTorch Nystrom (torch fwd + torch bwd via autograd)."""
    def forward(self, x):
        q, k, v, B, N = self._qkv(x)
        out = nystrom_attention_reference(
            q, k, v, self.m, self.niter, conv_weight=None, conv_kernel_size=0)
        return self._project_out(out, B, N)


class FullFNAttention(_NystromBase):
    """Full FlashNystrom (CUDA fwd + CUDA bwd)."""
    def forward(self, x):
        q, k, v, B, N = self._qkv(x)
        out = FlashNystromFunction.apply(q, k, v, None, self.m, self.niter, 0)
        return self._project_out(out, B, N)


class FNFwdTorchBwdAttention(_NystromBase):
    """FN CUDA forward + PyTorch reference backward."""
    def forward(self, x):
        q, k, v, B, N = self._qkv(x)
        out = FNFwdTorchBwdFunction.apply(q, k, v, self.m, self.niter)
        return self._project_out(out, B, N)


class TorchFwdFNBwdAttention(_NystromBase):
    """PyTorch reference forward + FN CUDA backward."""
    def forward(self, x):
        q, k, v, B, N = self._qkv(x)
        out = TorchFwdFNBwdFunction.apply(q, k, v, self.m, self.niter)
        return self._project_out(out, B, N)


# -------------------------------------------------------------------------
# TinyViT: same backbone, parameterized by attention factory.
# -------------------------------------------------------------------------
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
                "attn":  attn_module_factory(dim, heads),
                "norm1": nn.LayerNorm(dim),
                "ff":    nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
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
    print(f"\n{label}: {nparams/1e6:.2f}M params, N={n_tokens} tokens", flush=True)

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
                  f"train_acc={100*correct/total:.1f}% time={times[-1]:.1f}s", flush=True)

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
    print(f"  Final test_acc={test_acc:.1f}%  avg_epoch={avg_time:.1f}s", flush=True)
    return {"label": label, "test_acc": test_acc, "avg_epoch": avg_time, "params_M": nparams/1e6}


def main():
    print("="*70, flush=True)
    print("Five-way training comparison on CIFAR-10", flush=True)
    print("="*70, flush=True)

    M, NITER = 32, 6
    results = []

    print("\n--- 1. SDPA (PyTorch FA-2 backend, exact attention) ---", flush=True)
    results.append(train_one("SDPA", lambda d, h: SDPAAttention(d, h)))

    print("\n--- 2. Nystrom-Ref (torch fwd + torch bwd) ---", flush=True)
    results.append(train_one("Nystrom-Ref",
        lambda d, h: NystromRefAttention(d, h, M, NITER)))

    print("\n--- 3. FlashNystrom (CUDA fwd + CUDA bwd) ---", flush=True)
    results.append(train_one("FlashNystrom",
        lambda d, h: FullFNAttention(d, h, M, NITER)))

    print("\n--- 4. FN fwd + torch bwd ---", flush=True)
    results.append(train_one("FN-fwd+torch-bwd",
        lambda d, h: FNFwdTorchBwdAttention(d, h, M, NITER)))

    print("\n--- 5. torch fwd + FN bwd ---", flush=True)
    results.append(train_one("torch-fwd+FN-bwd",
        lambda d, h: TorchFwdFNBwdAttention(d, h, M, NITER)))

    print("\n" + "="*70, flush=True)
    print("Summary:", flush=True)
    for r in results:
        print(f"  {r['label']:>20}: test_acc={r['test_acc']:.1f}%  "
              f"avg_epoch={r['avg_epoch']:.1f}s  params={r['params_M']:.2f}M", flush=True)

    with open("five_way_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to five_way_results.json", flush=True)


if __name__ == "__main__":
    main()
