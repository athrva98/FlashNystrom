# CP0 fixture: capture REAL trained q,k,v,dO from an STL-10 step, then store the
# current kernel2_inv golden outputs (k2inv, ns_iterates) for both ridge off/on.
# All TC-fy checkpoint tests load this single fixture so we never regress against
# torch.randn (which produces degenerate near-rank-1 K2).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from train_three_way import TinyViT, NystromRefAttention

dev = "cuda"; torch.manual_seed(0)
M, NEWTON, DIM, HEADS, DEPTH = 64, 6, 256, 4, 4
BATCH = int(os.environ.get("CAP_BATCH", "4"))
WARMUP = int(os.environ.get("CAP_WARMUP", "30"))
OUT = os.environ.get("CAP_OUT", "C:/tmp/fn_real_fixture.pt")

CAP = {}
class CapAttn(NystromRefAttention):
    def __init__(self, *a, **k): super().__init__(*a, **k); self.idx=None; self._capture=False
    def forward(self, x):
        from flash_nystrom.reference import nystrom_attention_reference
        B,N,_ = x.shape; H,D = self.heads, self.head_dim
        q = self.q_proj(x).view(B,N,H,D).transpose(1,2).contiguous()
        k = self.k_proj(x).view(B,N,H,D).transpose(1,2).contiguous()
        v = self.v_proj(x).view(B,N,H,D).transpose(1,2).contiguous()
        out = nystrom_attention_reference(q,k,v,self.m,self.newton_iter,None,0)
        if self._capture:
            d = {"q":q.detach().clone(),"k":k.detach().clone(),"v":v.detach().clone()}
            out.register_hook(lambda g,d=d: d.__setitem__("dO", g.detach().clone()))
            CAP[self.idx] = d
        return self.out_proj(out.transpose(1,2).contiguous().view(B,N,-1))

def attn_factory(dim, heads):
    return CapAttn(dim, heads, num_landmarks=M, newton_iter=NEWTON, conv_kernel_size=0)

model = TinyViT(attn_factory, dim=DIM, depth=DEPTH, heads=HEADS, patch_size=1,
                num_classes=10, img_size=96).to(dev)
for i,blk in enumerate(model.blocks): blk["attn"].idx=i; blk["attn"]._capture=False

tf = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3,(0.5,)*3)])
ds = torchvision.datasets.STL10(root=os.environ.get("CAP_DATA_ROOT","./data"),
                                split="train", download=True, transform=tf)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
print(f"warmup {WARMUP} steps batch={BATCH} N={96*96+1} m={M}", flush=True)
it = iter(dl)
for step in range(WARMUP+1):
    try: xb,yb = next(it)
    except StopIteration: it=iter(dl); xb,yb=next(it)
    xb,yb = xb.to(dev), yb.to(dev)
    cap = (step==WARMUP)
    for blk in model.blocks: blk["attn"]._capture = cap
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    if not cap: opt.step()
    if step%10==0: print(f"  step {step} loss={loss.item():.3f}", flush=True)

# Stack the per-layer captures into (L*BH, N, D)-ish; keep per-layer for clarity.
layers = sorted(CAP)
fix = {"M":M, "NEWTON":NEWTON, "layers":layers}
for li in layers:
    d = CAP[li]
    fix[f"q{li}"]=d["q"].half().cpu(); fix[f"k{li}"]=d["k"].half().cpu()
    fix[f"v{li}"]=d["v"].half().cpu(); fix[f"dO{li}"]=d.get("dO", torch.zeros_like(d["q"])).half().cpu()
torch.save(fix, OUT)
print(f"saved fixture -> {OUT}  layers={layers}  shape={fix[f'q{layers[0]}'].shape}")

# ---- report K2 conditioning per layer so we know the fixture is 'real' (non-normal) ----
from flash_nystrom.reference import iterative_pinverse
for li in layers:
    q = fix[f"q{li}"].float().to(dev); k = fix[f"k{li}"].float().to(dev)
    B,H,N,D = q.shape; m=M; scale=D**-0.25
    qs=q*scale; ks=k*scale; seg=N//m; tr=seg*(m-1)
    qt=torch.cat([qs[:,:,:tr].reshape(B,H,m-1,seg,D).mean(3), qs[:,:,tr:].mean(2,keepdim=True)],2)
    kt=torch.cat([ks[:,:,:tr].reshape(B,H,m-1,seg,D).mean(3), ks[:,:,tr:].mean(2,keepdim=True)],2)
    K2 = F.softmax(qt@kt.transpose(-2,-1), -1)
    sv = torch.linalg.svdvals(K2.float())
    cond = (sv[...,0]/sv[...,-1].clamp(min=1e-20)).mean().item()
    comm = K2@K2.transpose(-2,-1) - K2.transpose(-2,-1)@K2
    nonnorm = (comm.norm(dim=(-2,-1))/(K2.float().norm(dim=(-2,-1))**2).clamp(min=1e-20)).mean().item()
    print(f"  layer {li}: cond(K2)={cond:.3e}  nonnormality||[K,K^T]||/||K||^2={nonnorm:.3f}")
