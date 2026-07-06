# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

# Three-way training comparison:
#   1. SDPA (PyTorch FA-2 backend) — exact attention, baseline
#   2. Nystrom reference (pure PyTorch) — Nyström algorithm, no custom kernels
#   3. FlashNystrom CUDA — Nyström algorithm via our fused kernels
#
# This isolates: (a) is the gap caused by Nyström approximation itself, or
# (b) by something specific to our kernels (e.g. FP16 backward noise)?
import os, sys, time, json
_HERE = os.path.dirname(os.path.abspath(__file__))   # .../benchmarks
_REPO = os.path.dirname(_HERE)                        # repo root
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)
_DATA = os.environ.get("FN_DATA_DIR", os.path.join(_REPO, "data"))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# cs.toronto.edu throttles CIFAR-10 downloads to ~10 kB/s (hours for 170 MB).
# Probe faster mirrors of the identical tarball (verified md5
# c58f30108f718f92721af3b95e74349a) and use the first one that responds;
# fall back to torchvision's default host if none do. A pre-placed or cached
# tarball always skips the download entirely (torchvision checks the md5).
def _pin_cifar_mirror():
    import urllib.request
    mirrors = [
        "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
    ]
    for url in mirrors:
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    torchvision.datasets.CIFAR10.url = url
                    return
        except Exception:
            continue
_pin_cifar_mirror()

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
    def __init__(self, dim, heads, num_landmarks=32, newton_iter=20, conv_kernel_size=0,
                 kappa_star=0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.m = num_landmarks
        self.newton_iter = newton_iter
        self.conv_kernel_size = conv_kernel_size
        self.kappa_star = kappa_star
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
        out = nystrom_attention_reference(q, k, v, self.m, self.newton_iter, cw,
                                          self.conv_kernel_size, kappa_star=self.kappa_star)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, N, -1))


class TinyViT(nn.Module):
    def __init__(self, attn_module_factory, dim=256, depth=4, heads=4,
                 patch_size=4, num_classes=10, img_size=32):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        n_patches = (img_size // patch_size) ** 2
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
              patch_size=4, dim=256, heads=4, grad_clip=1.0,
              autobatch=False, autobatch_cap=2048, dataset="cifar10", img_size=32,
              instrument=False, seed=42, amp=True):
    if autobatch and torch.cuda.is_available():
        from autobatch import search_and_profile

        def _trial(bs):
            m = TinyViT(attn_factory, dim=dim, depth=4, heads=heads,
                        patch_size=patch_size, img_size=img_size).cuda()
            o = torch.optim.AdamW(m.parameters(), lr=lr)
            x = torch.randn(bs, 3, img_size, img_size, device="cuda")
            yb = torch.randint(0, 10, (bs,), device="cuda")

            def run():
                o.zero_grad(set_to_none=True)
                with _amp_ctx():
                    loss = F.cross_entropy(m(x), yb)
                loss.backward()
                o.step()

            return run

        res = search_and_profile(_trial, lo=64, cap=autobatch_cap, warmup=2, iters=3)
        if res:
            batch_size = res["batch"]
        print(f"  autobatch: batch_size={batch_size}")

    norm = T.Normalize((0.5,) * 3, (0.5,) * 3)
    # Native dataset resolution; when img_size exceeds it (e.g. STL-10
    # upscaled to 180px for the N=32K conditioning experiment), resize
    # BEFORE the random crop (RandomCrop cannot exceed image + padding).
    native = 96 if dataset == "stl10" else 32
    pre = [T.Resize(img_size)] if img_size != native else []
    transform = T.Compose(pre + [T.RandomHorizontalFlip(),
                                 T.RandomCrop(img_size, padding=img_size // 8),
                                 T.ToTensor(), norm])
    transform_test = T.Compose(pre + [T.ToTensor(), norm])
    if dataset == "stl10":
        # 96x96 native real images, 10 classes, 5000 labeled train -> cheap, large N.
        trainset = torchvision.datasets.STL10(
            root=_DATA, split="train", download=True, transform=transform)
        testset = torchvision.datasets.STL10(
            root=_DATA, split="test", download=True, transform=transform_test)
    else:
        trainset = torchvision.datasets.CIFAR10(
            root=_DATA, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root=_DATA, train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=200, shuffle=False, num_workers=0,
        pin_memory=True, drop_last=True)

    torch.manual_seed(seed)
    model = TinyViT(attn_factory, dim=dim, depth=4, heads=heads,
                    patch_size=patch_size, img_size=img_size).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Per-step / per-attention-layer instrumentation on the SAME run that
    # produces the result (enable with --instrument). Captures dO (grad wrt
    # attention output) max + fraction under the fp16 floor + non-finite count,
    # the first non-finite parameter grad (NaN origin), and a collapse detector.
    _ins = {"bw": [], "ema": None, "first_nf": None, "collapse": None, "gstep": 0}
    if instrument:
        _layers = [blk["attn"] for blk in model.blocks]
        _ins["bw"] = [None] * len(_layers)
        def _mk_hook(i):
            def h(mod, gin, gout):
                g = gout[0]
                if g is None:
                    _ins["bw"][i] = None; return
                af = g.detach().float().abs()
                _ins["bw"][i] = (af.max().item(),
                                 (af < 6.104e-5).float().mean().item(),
                                 int((~torch.isfinite(g)).sum().item()))
            return h
        for _i, _L in enumerate(_layers):
            _L.register_full_backward_hook(_mk_hook(_i))
        print(f"  [instrument] hooks on {len(_layers)} attention layers", file=sys.stderr, flush=True)

    nparams = sum(p.numel() for p in model.parameters())
    n_tokens = (img_size // patch_size) ** 2 + 1
    print(f"\n{label}: {nparams/1e6:.2f}M params, N={n_tokens} tokens")

    # amp=False runs the whole step in fp32 (no autocast, GradScaler is a no-op).
    # Used by the nystrom_reference_fp32 arm to confirm that the reference's
    # large-N accuracy collapse is fp16 precision, not the algorithm: the length-N
    # softmax probabilities ~1/N fall below the fp16 normal floor (6.1e-5) at
    # N > ~16K, so its autograd softmax-Jacobian underflows. FlashNystrom keeps
    # that Jacobian in fp32 by construction and does not collapse.
    import contextlib
    _amp_ctx = (lambda: torch.amp.autocast("cuda", dtype=torch.float16)) if amp         else contextlib.nullcontext

    # FP16 loss scaling. Without it, the backward gradients (dO ~ 1e-3 and
    # everything downstream) sit below the FP16 normal floor (6.1e-5) and the
    # FP16 gradient stores flush to zero; see measure_bwd_ranges.py. The
    # dynamic scaler grows S until a gradient nears the FP16 overflow ceiling,
    # i.e. it uses the largest representable range. Applies to all three models.
    scaler = torch.amp.GradScaler("cuda")

    torch.cuda.reset_peak_memory_stats()
    times = []
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in trainloader:
            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            with _amp_ctx():
                logits = model(imgs)
                loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # Unscale before clip/instrumentation so grads are at their true
            # magnitude. (A dynamic scaler may occasionally produce inf here and
            # skip the step on backoff; that is expected, not a collapse.)
            scaler.unscale_(optimizer)
            if instrument and _ins["first_nf"] is None:
                for _pn, _p in model.named_parameters():
                    if _p.grad is not None and not torch.isfinite(_p.grad).all():
                        _ins["first_nf"] = (_ins["gstep"], _pn)
                        print(f"  !! first non-finite grad step {_ins['gstep']}: {_pn}",
                              file=sys.stderr, flush=True)
                        break
            gn = None
            if grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if instrument:
                _l = loss.item()
                _ins["ema"] = _l if _ins["ema"] is None else 0.98 * _ins["ema"] + 0.02 * _l
                if _l > _ins["ema"] * 1.30 and _ins["gstep"] > 30 and _ins["collapse"] is None:
                    _ins["collapse"] = _ins["gstep"]
                    print(f"  ** COLLAPSE onset step {_ins['gstep']}: loss {_l:.3f} "
                          f"(ema {_ins['ema']:.3f})", file=sys.stderr, flush=True)
                if _ins["gstep"] % 50 == 0:
                    _bw = [b for b in _ins["bw"] if b]
                    _dO = max((b[0] for b in _bw), default=0.0)
                    _uf = max((b[1] for b in _bw), default=0.0)
                    _nfc = sum((b[2] for b in _bw), 0)
                    _gnv = (gn.item() if gn is not None and torch.isfinite(gn) else -1.0)
                    print(f"  [instr] step {_ins['gstep']} ep{epoch} loss={_l:.3f} "
                          f"grad_norm={_gnv:.2e} dO_max={_dO:.2e} max_uflow_frac={_uf:.3f} "
                          f"dO_nonfinite={_nfc}", file=sys.stderr, flush=True)
                _ins["gstep"] += 1
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
            with _amp_ctx():
                logits = model(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    test_acc = 100 * correct / total
    timed = times[1:] if len(times) > 1 else times  # drop warmup epoch if possible
    avg_time = sum(timed) / max(len(timed), 1)
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    n_train = len(trainloader) * batch_size
    samp_s = n_train / avg_time if avg_time > 0 else 0.0
    print(f"  Final test_acc={test_acc:.1f}%  avg_epoch={avg_time:.1f}s  "
          f"batch={batch_size}  samp/s={samp_s:.0f}  peak_GiB={peak_gib:.2f}")
    return {"label": label, "test_acc": test_acc, "avg_epoch": avg_time,
            "params_M": nparams / 1e6, "batch": batch_size,
            "samples_per_s": samp_s, "peak_gib": peak_gib, "N_tokens": n_tokens}


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Three-way vision training: sdpa vs nystrom-ref vs flash_nystrom")
    ap.add_argument("--dataset", choices=["cifar10", "stl10"], default="cifar10",
                    help="cifar10 (32x32) or stl10 (96x96 native -> larger N, cheap)")
    ap.add_argument("--patch_size", type=int, default=4,
                    help="tokens = (img_size/patch_size)^2 + 1; patch_size=1 = pixel tokens")
    ap.add_argument("--img_size", type=int, default=0,
                    help="input resolution; 0 = dataset native (32 cifar / 96 stl10). "
                         "Larger values upscale (e.g. 180 -> N=32401 pixel tokens on stl10)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--num_landmarks", type=int, default=64)
    ap.add_argument("--newton_iter", type=int, default=6)
    ap.add_argument("--kappa_star", type=float, default=0.0,
                    help="Tikhonov ridge target cond(M) for the pinv, threaded "
                         "identically to FN and the reference. 0 = no ridge "
                         "(vanilla Nystromformer). Use 0 at small N (CIFAR), "
                         "~1e3 at large N where cond(K2) explodes.")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--no-fast-dk2inv", dest="fast_dk2inv", action="store_false",
                    help="use the FP32 reference-consistent dk2inv backward "
                         "(default: FP16 tensor-core fast path)")
    ap.add_argument("--batch_size", type=int, default=128,
                    help="fixed batch size (ignored when --autobatch is set). "
                         "Use an explicit value at very large N where autobatch's "
                         "single-shot probe overshoots and OOMs.")
    ap.add_argument("--autobatch", action="store_true")
    ap.add_argument("--autobatch_cap", type=int, default=2048)
    ap.add_argument("--backends", nargs="+",
                    default=["sdpa", "nystrom_reference", "flash_nystrom"])
    ap.add_argument("--no-instrument", dest="instrument", action="store_false",
                    help="disable the per-step collapse diagnostics (ON by default): "
                         "per-layer dO underflow, NaN origin, collapse onset")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for weight init + data order (for multi-seed runs)")
    ap.add_argument("--out_json", type=str, default="three_way_results.json",
                    help="where to write the structured per-backend results")
    a = ap.parse_args()
    m, ni, fdk = a.num_landmarks, a.newton_iter, a.fast_dk2inv
    img_size = a.img_size or (96 if a.dataset == "stl10" else 32)
    kw = dict(epochs=a.epochs, batch_size=a.batch_size, patch_size=a.patch_size,
              grad_clip=a.grad_clip, autobatch=a.autobatch,
              autobatch_cap=a.autobatch_cap, dataset=a.dataset, img_size=img_size,
              instrument=a.instrument, seed=a.seed)

    ks = a.kappa_star
    factories = {
        "sdpa": ("SDPA", lambda d, h: SDPAAttention(d, h)),
        "nystrom_reference": ("Nystrom-Ref",
            lambda d, h: NystromRefAttention(d, h, num_landmarks=m, newton_iter=ni,
                                             conv_kernel_size=0, kappa_star=ks)),
        # Same reference, but trained in full fp32 (amp forced off in the loop
        # below). Confirms the large-N reference collapse is fp16 precision, not
        # the algorithm: this arm should recover to ~FlashNystrom accuracy at 32K.
        "nystrom_reference_fp32": ("Nystrom-Ref-FP32",
            lambda d, h: NystromRefAttention(d, h, num_landmarks=m, newton_iter=ni,
                                             conv_kernel_size=0, kappa_star=ks)),
        # *_vanilla arms: identical backend with the ridge forced OFF (kappa=0),
        # regardless of --kappa_star. Paired with the ridged arm they isolate
        # the Tikhonov ridge's effect per backend (the 2026-07 sweeps found the
        # ridge never helps training and sometimes hurts).
        "nystrom_vanilla": ("Nystrom-Vanilla",
            lambda d, h: NystromRefAttention(d, h, num_landmarks=m, newton_iter=ni,
                                             conv_kernel_size=0, kappa_star=0.0)),
        "flash_nystrom_vanilla": ("FlashNystrom-V",
            lambda d, h: FlashNystromAttention(
                d, h, NystromConfig(num_landmarks=m, newton_iter=ni, fast_dk2inv=fdk,
                                    kappa_star=0.0, use_tc_pinv=False,
                                    conv_kernel_size=0, use_conv_residual=False))),
        "flash_nystrom_tc_vanilla": ("FlashNystrom-TC-V",
            lambda d, h: FlashNystromAttention(
                d, h, NystromConfig(num_landmarks=m, newton_iter=ni, fast_dk2inv=fdk,
                                    kappa_star=0.0, use_tc_pinv=True,
                                    conv_kernel_size=0, use_conv_residual=False))),
        # flash_nystrom = the faithful scalar fp32 Newton-Schulz pinv (default path).
        "flash_nystrom": ("FlashNystrom",
            lambda d, h: FlashNystromAttention(
                d, h, NystromConfig(num_landmarks=m, newton_iter=ni, fast_dk2inv=fdk,
                                    kappa_star=ks, use_tc_pinv=False,
                                    conv_kernel_size=0, use_conv_residual=False))),
        # flash_nystrom_tc = the opt-in tf32 tensor-core pinv (faster, ~1-5% accuracy cost).
        "flash_nystrom_tc": ("FlashNystrom-TC",
            lambda d, h: FlashNystromAttention(
                d, h, NystromConfig(num_landmarks=m, newton_iter=ni, fast_dk2inv=fdk,
                                    kappa_star=ks, use_tc_pinv=True,
                                    conv_kernel_size=0, use_conv_residual=False))),
    }
    n_tokens = (img_size // a.patch_size) ** 2 + 1
    print("=" * 70)
    print(f"{a.dataset}  img_size={img_size}  patch_size={a.patch_size}  "
          f"N={n_tokens} tokens  m={m}  grad_clip={a.grad_clip}  "
          f"fast_dk2inv={fdk}  kappa_star={a.kappa_star:g}  autobatch={a.autobatch}")
    print("=" * 70)
    results = []
    for backend in a.backends:
        label, fac = factories[backend]
        print(f"\n--- {label} ---")
        results.append(train_one(label, fac,
                                  amp=(backend != "nystrom_reference_fp32"), **kw))

    print("\n" + "=" * 70)
    print("Summary:")
    for r in results:
        print(f"  {r['label']:>14}: test_acc={r['test_acc']:.1f}%  N={r.get('N_tokens','?')}  "
              f"batch={r.get('batch','?')}  samp/s={r.get('samples_per_s',0):.0f}  "
              f"peak_GiB={r.get('peak_gib',0):.2f}")
    record = dict(dataset=a.dataset, patch_size=a.patch_size, seed=a.seed,
                  num_landmarks=m, newton_iter=ni, kappa_star=a.kappa_star,
                  n_tokens=n_tokens, results=results)
    with open(a.out_json, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved to {a.out_json}")


if __name__ == "__main__":
    main()
