# Copyright (c) 2026, Athrva Pandhare
# EMPIRICAL backward-intermediate range measurement (no symbolic bounds).
#
# Captures the REAL q,k,v and the REAL upstream dO at each attention layer
# during an actual fp16-autocast STL-10 training step (no GradScaler, matching
# train_three_way), then recomputes every Nystrom backward intermediate in FP32
# via autograd .grad on retained nodes (== exactly what the CUDA kernels compute)
# and reports measured abs-min(nonzero)/abs-max against the FP16 walls.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from flash_nystrom.reference import iterative_pinverse
from train_three_way import TinyViT, NystromRefAttention

FP16_MAX      = 65504.0
FP16_MIN_NORM = 2.0 ** -14      # 6.1035e-5  smallest normal
FP16_MIN_SUB  = 2.0 ** -24      # 5.96e-8    smallest subnormal (below -> flush to 0)

torch.manual_seed(0)
dev = "cuda"
M, NEWTON, DIM, HEADS, DEPTH = 64, 6, 256, 4, 4
BATCH = int(os.environ.get("FN_BWD_BATCH", "8"))
WARMUP_STEPS = int(os.environ.get("FN_BWD_WARMUP", "40"))

# ---- instrumented attention: capture q,k,v and the real grad of the nystrom output ----
CAP = {}   # layer_idx -> dict(q,k,v,dO)
class CapAttn(NystromRefAttention):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.idx = None
    def forward(self, x):
        from flash_nystrom.reference import nystrom_attention_reference
        B, N, _ = x.shape; H, D = self.heads, self.head_dim
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2).contiguous()
        out = nystrom_attention_reference(q, k, v, self.m, self.newton_iter, None, 0)
        if self._capture:
            d = {"q": q.detach().clone(), "k": k.detach().clone(), "v": v.detach().clone()}
            def hook(g, d=d): d["dO"] = g.detach().clone()
            out.register_hook(hook)
            CAP[self.idx] = d
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, N, -1))

def attn_factory(dim, heads):
    return CapAttn(dim, heads, num_landmarks=M, newton_iter=NEWTON, conv_kernel_size=0)

# tag layers
model = TinyViT(attn_factory, dim=DIM, depth=DEPTH, heads=HEADS, patch_size=1,
                num_classes=10, img_size=96).to(dev)
for i, blk in enumerate(model.blocks):
    blk["attn"].idx = i; blk["attn"]._capture = False

tf = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
DATA_ROOT = os.environ.get("FN_DATA_DIR", "./data")
ds = torchvision.datasets.STL10(root=DATA_ROOT, split="train", download=True, transform=tf)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

print(f"warmup {WARMUP_STEPS} steps  batch={BATCH}  N={(96//1)**2+1}  m={M}  no GradScaler", flush=True)
it = iter(dl)
for step in range(WARMUP_STEPS + 1):
    try: xb, yb = next(it)
    except StopIteration: it = iter(dl); xb, yb = next(it)
    xb, yb = xb.to(dev), yb.to(dev)
    cap = (step == WARMUP_STEPS)
    for blk in model.blocks: blk["attn"]._capture = cap
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = F.cross_entropy(model(xb), yb)
    loss.backward()                # NO scaler, exactly like train_three_way
    if not cap: opt.step()
    if step % 10 == 0: print(f"  step {step} loss={loss.item():.3f}", flush=True)

# ----------------- FP32 recompute with retained grads = kernel quantities -----------------
def report(rows):
    hdr = f"{'quantity':<26}{'kernel store':<14}{'abs_max':>11}{'min_nz':>11}{'%>MAX':>8}{'%subN':>8}{'%flush':>8}"
    print(hdr); print("-"*len(hdr))
    for name, store, t in rows:
        a = t.detach().abs().float().reshape(-1)
        nz = a[a > 0]
        mx = a.max().item()
        mn = nz.min().item() if nz.numel() else 0.0
        over  = 100.0 * (a > FP16_MAX).float().mean().item()
        denom = max(nz.numel(), 1)
        subn  = 100.0 * ((a >= FP16_MIN_SUB) & (a < FP16_MIN_NORM)).sum().item() / denom
        flush = 100.0 * ((a > 0) & (a < FP16_MIN_SUB)).sum().item() / denom
        flag = "  <-- OVERFLOW" if over > 0 else ("  <-- flush>0" if flush > 0 else "")
        print(f"{name:<26}{store:<14}{mx:>11.3e}{mn:>11.3e}{over:>8.2f}{subn:>8.2f}{flush:>8.2f}{flag}")

def recompute(d):
    q = d["q"].float().requires_grad_(True)
    k = d["k"].float().requires_grad_(True)
    v = d["v"].float().requires_grad_(True)
    dO = d["dO"].float()
    B, H, N, D = q.shape; m = M
    scale = D ** -0.25
    q_s = q * scale; k_s = k * scale
    seg = N // m; trunc = seg * (m - 1)
    qf = q_s[:, :, :trunc, :].reshape(B, H, m-1, seg, D).mean(3)
    kf = k_s[:, :, :trunc, :].reshape(B, H, m-1, seg, D).mean(3)
    ql = q_s[:, :, trunc:, :].mean(2, keepdim=True); kl = k_s[:, :, trunc:, :].mean(2, keepdim=True)
    q_t = torch.cat([qf, ql], 2); k_t = torch.cat([kf, kl], 2)
    S1 = q_s @ k_t.transpose(-2, -1); K1 = F.softmax(S1, -1)
    S2 = q_t @ k_t.transpose(-2, -1); K2 = F.softmax(S2, -1)
    S3 = q_t @ k_s.transpose(-2, -1); K3 = F.softmax(S3, -1)
    K2inv = iterative_pinverse(K2.float(), NEWTON)
    step1 = K3 @ v                      # B  (kernel: b_saved, dO3 = step1.grad)
    step2 = K2inv @ step1
    O = K1 @ step2
    for t in (q_s, k_s, q_t, k_t, S1, K1, S2, K2, K2inv, S3, K3, step1, step2, O):
        t.retain_grad()
    O.backward(dO)
    D1 = (K1 * K1.grad).sum(-1, keepdim=True)
    D3 = (K3 * K3.grad).sum(-1, keepdim=True)
    return dict(K1=K1, K3=K3, step1=step1, step2=step2, K2inv=K2inv, q_s=q_s, k_s=k_s,
                q_t=q_t, k_t=k_t, v=v, O=O, dO=dO,
                dP1=K1.grad, dS1=S1.grad, dP3=K3.grad, dS3=S3.grad,
                dstep2=step2.grad, dO3=step1.grad, dK2inv=K2inv.grad,
                D1=D1, D3=D3, dq=q.grad, dk=k.grad, dv=v.grad,
                dq_t=q_t.grad, dk_t=k_t.grad)

for li in sorted(CAP):
    d = CAP[li]
    if "dO" not in d:
        print(f"layer {li}: no dO captured"); continue
    R = recompute(d)
    print(f"\n================ LAYER {li}  (dO abs_max={d['dO'].float().abs().max():.3e}) ================")
    report([
        ("dO (upstream grad)",   "input fp16",  R["dO"]),
        ("P1=K1 softmax",        "rP fp16",     R["K1"]),
        ("P3=K3 softmax",        "rP fp16",     R["K3"]),
        ("B=K3@V (fwd)",         "b_saved fp16",R["step1"]),
        ("step2 (fwd)",          "step2 fp16",  R["step2"]),
        ("K2inv (fwd)",          "fp32",        R["K2inv"]),
        ("dP1=dO@step2^T",       "fp32 reg",    R["dP1"]),
        ("dS1=P1*(dP1-D1)",      "rdS fp16",    R["dS1"]),
        ("D1 rowterm",           "fp32",        R["D1"]),
        ("dstep2=K1^T@dO",       "fp32 acc",    R["dstep2"]),
        ("dO3=K2inv^T@dstep2",   "dO3 fp16",    R["dO3"]),
        ("dK2inv=dstep2@B^T",    "fp32",        R["dK2inv"]),
        ("dP3=dO3@V^T",          "fp32 reg",    R["dP3"]),
        ("dS3=P3*(dP3-D3)",      "rdS fp16",    R["dS3"]),
        ("D3 rowterm",           "fp32",        R["D3"]),
        ("dV=K3^T@dO3",          "dV fp16 RMW", R["dv"]),
        ("dq_tilde",             "fp32 acc",    R["dq_t"]),
        ("dk_tilde",             "fp32 acc",    R["dk_t"]),
        ("dQ (final)",           "fp16 out",    R["dq"]),
        ("dK (final)",           "fp16 out",    R["dk"]),
    ])
print("\nFP16 walls: MAX=6.55e4  MIN_normal=6.10e-5  flush-to-0 below 5.96e-8")
