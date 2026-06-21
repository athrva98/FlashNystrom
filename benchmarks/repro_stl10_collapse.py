# Cheap, faithful reproduction of the FlashNystrom STL-10 training collapse.
#
# Same model/recipe as train_three_way.py (TinyViT dim=256 depth=4 heads=4,
# patch_size=2 -> N=2305, m=64, newton_iter=6, AdamW lr=1e-3 wd=0.05, cosine,
# fp16 autocast, grad_clip=1.0) but shrunk to run on a laptop GPU: a STL-10
# subset and a small batch. The collapse is a property of the kernel backward
# at N=2305, so it reproduces at small batch/subset.
#
# Usage:  python benchmarks/repro_stl10_collapse.py [--epochs N] [--subset N] [--batch N]
import os, sys, argparse, math
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO); sys.path.insert(0, _HERE)
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from train_three_way import TinyViT, NystromRefAttention
from flash_nystrom import FlashNystromAttention, NystromConfig

_DATA = os.environ.get("FN_DATA_DIR", os.path.join(_REPO, "data"))

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--subset", type=int, default=2000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--backends", nargs="+", default=["ref", "fn"])
a = ap.parse_args()

FACT = {
    "ref": lambda d, h: NystromRefAttention(d, h, num_landmarks=64, newton_iter=6, conv_kernel_size=0),
    "fn":  lambda d, h: FlashNystromAttention(d, h, NystromConfig(
                num_landmarks=64, newton_iter=6, conv_kernel_size=0, use_conv_residual=False)),
}

norm = T.Normalize((0.5,) * 3, (0.5,) * 3)
tf = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(96, padding=12), T.ToTensor(), norm])
full = torchvision.datasets.STL10(root=_DATA, split="train", download=True, transform=tf)
sub = torch.utils.data.Subset(full, list(range(min(a.subset, len(full)))))
loader = torch.utils.data.DataLoader(sub, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=0)
print(f"STL-10 subset={len(sub)} batch={a.batch} epochs={a.epochs}  N=2305 m=64")


def run(name):
    torch.manual_seed(42)
    net = TinyViT(FACT[name], dim=a.dim, depth=a.depth, heads=a.heads, patch_size=2, img_size=96).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    print(f"\n=== {name} ===")
    best = 0.0
    for ep in range(a.epochs):
        net.train(); tl = cor = tot = 0; gmax = 0.0; nbad = 0
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logit = net(x); loss = F.cross_entropy(logit, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if not torch.isfinite(gn):
                nbad += 1
            else:
                gmax = max(gmax, gn.item())
            opt.step()
            tl += loss.item() * len(y); cor += (logit.argmax(1) == y).sum().item(); tot += len(y)
        sched.step()
        acc = 100 * cor / tot; best = max(best, acc)
        flag = "  <-- COLLAPSE" if (ep > 3 and acc < best - 8) else ""
        print(f" ep{ep:2d} loss={tl/tot:6.3f} train_acc={acc:4.1f}% grad_max={gmax:7.2e} nan_steps={nbad}{flag}")
    print(f" {name}: best_train_acc={best:.1f}%")


for b in a.backends:
    run(b)
