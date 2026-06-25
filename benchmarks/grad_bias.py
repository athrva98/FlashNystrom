# Is the FN-vs-reference gradient difference a BIAS or zero-mean noise? Over many
# MNIST batches x layers, in PRODUCTION precision (FN fp16/tf32 vs reference fp16
# autocast), measure for dq/dk:
#   proj  = <g_FN, g_ref>/||g_ref||^2   (effective step scale along the ref direction)
#   nrat  = ||g_FN||/||g_ref||          (magnitude ratio)
#   smean = mean(g_FN - g_ref)/mean|g_ref|  (signed bias, normalized)
# proj systematically <1  -> FN undershoots the reference gradient -> systematic loss.
# proj scattered around 1  -> zero-mean noise (FN should win ~half the time).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# kappa_star is now an explicit param (passed to apply / the reference), not an env var.
import torch, torch.nn as nn, torch.nn.functional as F, torchvision, torchvision.transforms as T
from flash_nystrom.flash_nystrom import FlashNystromFunction
from flash_nystrom.reference import nystrom_attention_reference
dev="cuda"; torch.manual_seed(0)
DIM,HEADS,DEPTH,M,NW=256,4,3,64,6; HD=DIM//HEADS
class Attn(nn.Module):
    def __init__(s):
        super().__init__()
        s.qp=nn.Linear(DIM,DIM,bias=False);s.kp=nn.Linear(DIM,DIM,bias=False);s.vp=nn.Linear(DIM,DIM,bias=False);s.op=nn.Linear(DIM,DIM,bias=False); s.cap=False; s.store={}
    def forward(s,x):
        B,N,_=x.shape
        q=s.qp(x).view(B,N,HEADS,HD).transpose(1,2).contiguous();k=s.kp(x).view(B,N,HEADS,HD).transpose(1,2).contiguous();v=s.vp(x).view(B,N,HEADS,HD).transpose(1,2).contiguous()
        o=FlashNystromFunction.apply(q,k,v,M,NW,True,5.0,True)
        if s.cap: s.store=dict(q=q.detach().clone(),k=k.detach().clone(),v=v.detach().clone()); o.register_hook(lambda g:s.store.__setitem__('dO',g.detach().clone()))
        return s.op(o.transpose(1,2).contiguous().view(B,N,DIM))
class Block(nn.Module):
    def __init__(s): super().__init__(); s.n1=nn.LayerNorm(DIM);s.at=Attn();s.n2=nn.LayerNorm(DIM);s.mlp=nn.Sequential(nn.Linear(DIM,4*DIM),nn.GELU(),nn.Linear(4*DIM,DIM))
    def forward(s,x): x=x+s.at(s.n1(x)); return x+s.mlp(s.n2(x))
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.pe=nn.Conv2d(1,DIM,1,1); n=28*28
        s.pos=nn.Parameter(torch.randn(1,n+1,DIM)*0.02);s.cls=nn.Parameter(torch.randn(1,1,DIM)*0.02)
        s.blocks=nn.ModuleList([Block() for _ in range(DEPTH)]);s.norm=nn.LayerNorm(DIM);s.head=nn.Linear(DIM,10)
    def forward(s,x):
        B=x.shape[0]; x=s.pe(x).flatten(2).transpose(1,2); x=torch.cat([s.cls.expand(B,-1,-1),x],1)+s.pos
        for b in s.blocks: x=b(x)
        return s.head(s.norm(x)[:,0])
model=Net().to(dev)
ds=torchvision.datasets.MNIST(root=os.environ.get("CAP_DATA_ROOT","./data"),train=True,download=True,transform=T.ToTensor())
dl=torch.utils.data.DataLoader(ds,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
opt=torch.optim.AdamW(model.parameters(),lr=1e-3);scaler=torch.amp.GradScaler("cuda"); it=iter(dl)
def step_batch(cap):
    global it
    try: xb,yb=next(it)
    except StopIteration: it=iter(dl);xb,yb=next(it)
    xb,yb=xb.to(dev),yb.to(dev)
    for b in model.blocks: b.at.cap=cap
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda",dtype=torch.float16): loss=F.cross_entropy(model(xb),yb)
    scaler.scale(loss).backward()
    if not cap: scaler.step(opt);scaler.update()
for _ in range(20): step_batch(False)   # warm up

def proj(a,b): return (a.flatten()@b.flatten()/ (b.flatten()@b.flatten()).clamp(min=1e-30)).item()
def nrat(a,b): return (a.norm()/b.norm().clamp(min=1e-30)).item()
def smean(a,b): return ((a-b).mean()/b.abs().mean().clamp(min=1e-30)).item()
agg={k:[] for k in ["q_proj","k_proj","q_nrat","k_nrat","q_smean","k_smean","q_cos","k_cos"]}
NB=24
for bi in range(NB):
    step_batch(True)
    for blk in model.blocks:
        st=blk.at.store
        if 'dO' not in st: continue
        q,k,v,dO=st['q'],st['k'],st['v'],st['dO']
        qg=q.clone().requires_grad_();kg=k.clone().requires_grad_();vg=v.clone().requires_grad_()
        FlashNystromFunction.apply(qg,kg,vg,M,NW,True,5.0,True).backward(dO)        # FN production
        q2=q.clone().requires_grad_();k2=k.clone().requires_grad_();v2=v.clone().requires_grad_()
        with torch.amp.autocast("cuda",dtype=torch.float16):              # reference fp16 (paper)
            oref=nystrom_attention_reference(q2,k2,v2,M,NW,kappa_star=5.0)
        oref.backward(dO)
        agg["q_proj"].append(proj(qg.grad,q2.grad)); agg["k_proj"].append(proj(kg.grad,k2.grad))
        agg["q_nrat"].append(nrat(qg.grad,q2.grad)); agg["k_nrat"].append(nrat(kg.grad,k2.grad))
        agg["q_smean"].append(smean(qg.grad,q2.grad)); agg["k_smean"].append(smean(kg.grad,k2.grad))
        agg["q_cos"].append(F.cosine_similarity(qg.grad.flatten(),q2.grad.flatten(),0).item())
        agg["k_cos"].append(F.cosine_similarity(kg.grad.flatten(),k2.grad.flatten(),0).item())
import statistics as S
n=len(agg["q_proj"])
print(f"production FN vs fp16 reference, {n} samples ({NB} batches x {DEPTH} layers)")
print(f"{'metric':>10} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}   (proj/nrat<1 => FN undershoots)")
for key in ["q_proj","k_proj","q_nrat","k_nrat","q_cos","k_cos","q_smean","k_smean"]:
    v=agg[key]; print(f"{key:>10} {S.mean(v):>10.4f} {S.pstdev(v):>10.4f} {min(v):>10.4f} {max(v):>10.4f}")
wins_q=sum(1 for p in agg["q_proj"] if p>1.0); wins_k=sum(1 for p in agg["k_proj"] if p>1.0)
print(f"\nproj>1 (FN overshoots): dq {wins_q}/{n}   dk {wins_k}/{n}   <- if ~half, it's noise; if ~0, systematic")
sys.stdout.flush(); os._exit(0)
