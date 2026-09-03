# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
# Paired 3-seed MNIST per-pixel training: FlashNystrom vs reference. For each seed,
# BOTH methods get identical init + identical data order (seed reset before each),
# so the only difference is the attention kernel. Compare test accuracy per seed:
# if the kernel is unbiased, sign(FN - Ref) should flip across seeds.
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# kappa_star is now an explicit param (passed to apply / the reference), not an env var.
import torch, torch.nn as nn, torch.nn.functional as F, torchvision, torchvision.transforms as T
from flash_nystrom.flash_nystrom import FlashNystromFunction
from flash_nystrom.reference import nystrom_attention_reference
dev="cuda"
DIM,HEADS,DEPTH,M,NW=256,4,3,64,6; HD=DIM//HEADS
EPOCHS,BS,LR=4,128,1e-3

class Attn(nn.Module):
    def __init__(s):
        super().__init__()
        s.qp=nn.Linear(DIM,DIM,bias=False);s.kp=nn.Linear(DIM,DIM,bias=False);s.vp=nn.Linear(DIM,DIM,bias=False);s.op=nn.Linear(DIM,DIM,bias=False); s.method="fn"
    def forward(s,x):
        B,N,_=x.shape
        q=s.qp(x).view(B,N,HEADS,HD).transpose(1,2).contiguous();k=s.kp(x).view(B,N,HEADS,HD).transpose(1,2).contiguous();v=s.vp(x).view(B,N,HEADS,HD).transpose(1,2).contiguous()
        if s.method=="fn": o=FlashNystromFunction.apply(q,k,v,M,NW,True,5.0,True)
        else: o=nystrom_attention_reference(q,k,v,M,NW,kappa_star=5.0)
        return s.op(o.transpose(1,2).contiguous().view(B,N,DIM))
class Block(nn.Module):
    def __init__(s): super().__init__(); s.n1=nn.LayerNorm(DIM);s.at=Attn();s.n2=nn.LayerNorm(DIM);s.mlp=nn.Sequential(nn.Linear(DIM,4*DIM),nn.GELU(),nn.Linear(4*DIM,DIM))
    def forward(s,x): x=x+s.at(s.n1(x)); return x+s.mlp(s.n2(x))
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.pe=nn.Conv2d(1,DIM,1,1); n=28*28
        s.pos=nn.Parameter(torch.randn(1,n+1,DIM)*0.02);s.cls=nn.Parameter(torch.randn(1,1,DIM)*0.02)
        s.blocks=nn.ModuleList([Block() for _ in range(DEPTH)]);s.norm=nn.LayerNorm(DIM);s.head=nn.Linear(DIM,10)
    def set_method(s,mth):
        for b in s.blocks: b.at.method=mth
    def forward(s,x):
        B=x.shape[0]; x=s.pe(x).flatten(2).transpose(1,2); x=torch.cat([s.cls.expand(B,-1,-1),x],1)+s.pos
        for b in s.blocks: x=b(x)
        return s.head(s.norm(x)[:,0])

root=os.environ.get("FN_DATA_DIR", "./data")
train=torchvision.datasets.MNIST(root=root,train=True,download=True,transform=T.ToTensor())
test=torchvision.datasets.MNIST(root=root,train=False,download=True,transform=T.ToTensor())
testdl=torch.utils.data.DataLoader(test,batch_size=512,shuffle=False,num_workers=0)

def seed_all(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def evaluate(model):
    model.eval(); correct=0; total=0
    for xb,yb in testdl:
        xb,yb=xb.to(dev),yb.to(dev)
        with torch.amp.autocast("cuda",dtype=torch.float16): out=model(xb)
        correct+=(out.argmax(1)==yb).sum().item(); total+=yb.numel()
    return 100.0*correct/total

def run(method,seed):
    seed_all(seed)
    model=Net().to(dev); model.set_method(method)
    g=torch.Generator(); g.manual_seed(seed)
    dl=torch.utils.data.DataLoader(train,batch_size=BS,shuffle=True,num_workers=0,drop_last=True,generator=g)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.05); scaler=torch.amp.GradScaler("cuda")
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS*len(dl))
    model.train()
    for ep in range(EPOCHS):
        for xb,yb in dl:
            xb,yb=xb.to(dev),yb.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",dtype=torch.float16): loss=F.cross_entropy(model(xb),yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    return evaluate(model)

res={"fn":{},"ref":{}}
print(f"MNIST per-pixel N=785 m={M} dim={DIM} depth={DEPTH} epochs={EPOCHS} bs={BS}",flush=True)
for seed in (0,1,2):
    for method in ("ref","fn"):
        acc=run(method,seed); res[method][seed]=acc
        print(f"  seed {seed} {method:>3}: test_acc={acc:.3f}",flush=True)
print("\n=== paired results ===")
print(f"{'seed':>4} {'FN':>8} {'Ref':>8} {'FN-Ref':>8} {'winner':>8}")
fnwins=0
for seed in (0,1,2):
    fa,ra=res['fn'][seed],res['ref'][seed]; d=fa-ra; w="FN" if d>0 else "Ref"; fnwins+=(d>0)
    print(f"{seed:>4} {fa:>8.3f} {ra:>8.3f} {d:>+8.3f} {w:>8}")
import statistics as S
fnm=[res['fn'][s] for s in (0,1,2)]; rfm=[res['ref'][s] for s in (0,1,2)]
print(f"\nFN  mean={S.mean(fnm):.3f} std={S.pstdev(fnm):.3f}")
print(f"Ref mean={S.mean(rfm):.3f} std={S.pstdev(rfm):.3f}")
print(f"FN won {fnwins}/3 seeds  (mixed => unbiased; 0/3 => systematic loss)")
sys.stdout.flush(); os._exit(0)
