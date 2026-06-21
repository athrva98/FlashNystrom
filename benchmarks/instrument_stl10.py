# Heavily-instrumented STL-10 trainer to diagnose the FlashNystrom training
# collapse that appears at full scale (dim 256, depth 4, batch 128) on A100/Colab
# but NOT at the small scales that fit a laptop. Run this on Colab.
#
# It captures, per step and per attention layer:
#   - dO (grad wrt attention output): max, abs-mean, fraction below the FP16
#     normal floor (~6e-5), count of non-finite entries, and the implied
#     dynamic grad-scale the kernel uses (64 / max|dO|).
#   - attention output abs-max (forward blow-up detector).
#   - total grad norm, loss, lr.
#   - FIRST non-finite parameter gradient (name + step) -> NaN origin.
#   - K2 landmark condition number per layer (periodic).
#   - a collapse detector (loss EMA jump / train-acc drop) that switches on
#     dense per-step logging around the event.
#
# Output: a CSV (per step) + a stderr summary that names the first anomaly,
# the collapse step, and the per-layer state at collapse. Runs FN and the
# pure-PyTorch reference so their trajectories can be diffed.
#
# Usage on Colab:
#   python benchmarks/instrument_stl10.py --backends fn ref \
#       --epochs 50 --batch 128 --dim 256 --depth 4 --heads 4 --csv /content/instr.csv
import os, sys, argparse, csv, math, statistics
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO); sys.path.insert(0, _HERE)
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from train_three_way import TinyViT, NystromRefAttention
from flash_nystrom import FlashNystromAttention, NystromConfig

_DATA = os.environ.get("FN_DATA_DIR", os.path.join(_REPO, "data"))
FP16_MIN_NORMAL = 6.104e-5  # smallest fp16 normal; below this -> underflow risk

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--subset", type=int, default=0, help="0 = full train split")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--grad_clip", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--kappa_every", type=int, default=200, help="steps between K2 cond probes")
ap.add_argument("--csv", type=str, default="instr_stl10.csv")
ap.add_argument("--backends", nargs="+", default=["fn", "ref"])
a = ap.parse_args()

FACT = {
    "ref": lambda d, h: NystromRefAttention(d, h, num_landmarks=64, newton_iter=6, conv_kernel_size=0),
    "fn":  lambda d, h: FlashNystromAttention(d, h, NystromConfig(
                num_landmarks=64, newton_iter=6, conv_kernel_size=0, use_conv_residual=False)),
}


def log(*x): print(*x, file=sys.stderr, flush=True)


def data_loader():
    norm = T.Normalize((0.5,) * 3, (0.5,) * 3)
    tf = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(96, padding=12), T.ToTensor(), norm])
    full = torchvision.datasets.STL10(root=_DATA, split="train", download=True, transform=tf)
    ds = full if a.subset <= 0 else torch.utils.data.Subset(full, list(range(min(a.subset, len(full)))))
    return torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=2)


def attn_layers(model):
    return [blk["attn"] for blk in model.blocks]


def run(name, loader, writer):
    torch.manual_seed(a.seed)
    model = TinyViT(FACT[name], dim=a.dim, depth=a.depth, heads=a.heads,
                    patch_size=2, img_size=96).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    layers = attn_layers(model)

    # Per-layer capture buffers, refreshed each step by hooks.
    fwd_absmax = [0.0] * len(layers)
    bwd = [None] * len(layers)

    def mk_fwd(i):
        def h(mod, inp, out):
            fwd_absmax[i] = out.detach().float().abs().max().item()
        return h

    def mk_bwd(i):
        def h(mod, gin, gout):
            g = gout[0]
            if g is None:
                bwd[i] = None; return
            gf = g.detach().float()
            af = gf.abs()
            amax = af.max().item()
            bwd[i] = dict(
                dO_max=amax,
                dO_absmean=af.mean().item(),
                dO_uflow_frac=(af < FP16_MIN_NORMAL).float().mean().item(),
                dO_nonfinite=int((~torch.isfinite(gf)).sum().item()),
                grad_scale=(64.0 / amax) if amax > 0 else float("inf"),
            )
        return h

    for i, L in enumerate(layers):
        L.register_forward_hook(mk_fwd(i))
        L.register_full_backward_hook(mk_bwd(i))

    log(f"\n===== backend={name}  dim={a.dim} depth={a.depth} heads={a.heads} "
        f"batch={a.batch} N=2305 m=64 =====")
    gstep = 0
    loss_ema = None
    best_acc = 0.0
    first_nonfinite = None
    collapse_step = None
    for ep in range(a.epochs):
        model.train()
        tot = cor = n = 0
        for imgs, labels in loader:
            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()

            # Scan param grads for the first non-finite (NaN origin).
            step_nonfinite = None
            for pn, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    step_nonfinite = pn
                    break
            if step_nonfinite and first_nonfinite is None:
                first_nonfinite = (gstep, step_nonfinite)
                log(f"  !! FIRST non-finite grad at step {gstep}: {step_nonfinite}")

            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()

            lval = loss.item()
            logits_nonfinite = int((~torch.isfinite(logits.detach())).sum().item())
            # Per-layer dO summary across layers (worst case)
            uflow = max((b["dO_uflow_frac"] for b in bwd if b), default=0.0)
            dO_max = max((b["dO_max"] for b in bwd if b), default=0.0)
            dO_nf = sum((b["dO_nonfinite"] for b in bwd if b), 0)
            gscale_min = min((b["grad_scale"] for b in bwd if b), default=0.0)

            # collapse detector
            if loss_ema is None:
                loss_ema = lval
            jump = lval > loss_ema * 1.30 and gstep > 30
            if jump and collapse_step is None:
                collapse_step = gstep
                log(f"  ** COLLAPSE onset step {gstep}: loss {lval:.3f} (ema {loss_ema:.3f})")
            loss_ema = 0.98 * loss_ema + 0.02 * lval

            # Per-layer detail row(s) to CSV every step
            for i, b in enumerate(bwd):
                if b is None:
                    continue
                writer.writerow(dict(
                    backend=name, gstep=gstep, epoch=ep, layer=i, loss=f"{lval:.5f}",
                    grad_norm=f"{(gn.item() if torch.isfinite(gn) else float('inf')):.4e}",
                    logits_nonfinite=logits_nonfinite,
                    attn_out_absmax=f"{fwd_absmax[i]:.4e}",
                    dO_max=f"{b['dO_max']:.4e}", dO_absmean=f"{b['dO_absmean']:.4e}",
                    dO_uflow_frac=f"{b['dO_uflow_frac']:.4f}",
                    dO_nonfinite=b["dO_nonfinite"],
                    grad_scale=f"{b['grad_scale']:.4e}",
                ))

            # K2 condition probe (periodic, one current batch, no_grad)
            if gstep % a.kappa_every == 0:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    _ = model(imgs)  # refill fwd hooks (cheap)
                kap = _probe_kappa(model, imgs)
                log(f"  step {gstep} ep{ep}: loss={lval:.3f} dO_max={dO_max:.2e} "
                    f"min_grad_scale={gscale_min:.2e} max_uflow_frac={uflow:.3f} "
                    f"dO_nonfinite={dO_nf} logits_nonfinite={logits_nonfinite} "
                    f"kappa(K2)~{kap}")

            tot += lval * len(labels); cor += (logits.argmax(1) == labels).sum().item(); n += len(labels)
            gstep += 1
        sched.step()
        acc = 100 * cor / n; best_acc = max(best_acc, acc)
        log(f"  ep{ep:2d} loss={tot/n:.3f} train_acc={acc:4.1f}% best={best_acc:4.1f}%"
            + ("  <-- well below best" if acc < best_acc - 8 else ""))

    log(f"  SUMMARY backend={name}: best_train_acc={best_acc:.1f}%  "
        f"first_nonfinite={first_nonfinite}  collapse_step={collapse_step}")
    return best_acc


def _probe_kappa(model, imgs):
    """Cheap-ish: grab one head's K2 landmark cond from the first attn layer."""
    try:
        blk = model.blocks[0]["attn"]
        x = model.patch_embed(imgs).flatten(2).transpose(1, 2)
        cls = model.cls_token.expand(imgs.shape[0], -1, -1)
        x = torch.cat([cls, x], 1) + model.pos_embed
        x = model.blocks[0]["norm1"](x)
        H, D = blk.heads, getattr(blk, "head_dim", a.dim // a.heads)
        q = blk.q_proj(x).view(x.shape[0], x.shape[1], H, D).transpose(1, 2).float()
        k = blk.k_proj(x).view(x.shape[0], x.shape[1], H, D).transpose(1, 2).float()
        N = q.shape[2]; m = 64; seg = N // m; tn = seg * (m - 1)
        s = D ** -0.25
        qf = (q * s)[:, :, :tn].reshape(q.shape[0], H, m - 1, seg, D).mean(3)
        kf = (k * s)[:, :, :tn].reshape(q.shape[0], H, m - 1, seg, D).mean(3)
        ql = (q * s)[:, :, tn:].mean(2, keepdim=True); kl = (k * s)[:, :, tn:].mean(2, keepdim=True)
        qt = torch.cat([qf, ql], 2); kt = torch.cat([kf, kl], 2)
        K2 = torch.softmax(qt @ kt.transpose(-2, -1), -1)
        return f"{torch.linalg.cond(K2).mean().item():.2e}"
    except Exception as e:
        return f"err:{type(e).__name__}"


def main():
    loader = data_loader()
    fields = ["backend", "gstep", "epoch", "layer", "loss", "grad_norm",
              "logits_nonfinite", "attn_out_absmax", "dO_max", "dO_absmean",
              "dO_uflow_frac", "dO_nonfinite", "grad_scale"]
    with open(a.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        res = {}
        for b in a.backends:
            res[b] = run(b, loader, writer)
    log(f"\nCSV written to {a.csv}")
    log("FINAL: " + "  ".join(f"{k}={v:.1f}%" for k, v in res.items()))


if __name__ == "__main__":
    main()
