# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
# Controlled cond(K2) sweep of the pinv backward: FN's kernel2_inv_bwd (fp32 debug
# hook) vs an autograd UNROLL of FN's EXACT ridge forward. Same forward function ->
# if FN's hand-derived backward is correct, rel ~ 1e-5 at every cond. Reports rel,
# cos, and norm-ratio for both lambda conventions (detached vs differentiated) to
# pin which one FN uses and whether high-cond blowup is a bug or numerical amplification.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from flash_nystrom.flash_nystrom import _C
dev="cuda"; torch.manual_seed(0); torch.zeros(1,device=dev)
BH,m,d,NW,KS=8,64,64,6,5.0
I=torch.eye(m,device=dev).expand(BH,m,m)
def rel(a,b): return ((a.float()-b.float()).norm()/b.float().norm().clamp(min=1e-30)).item()
def cos(a,b): return F.cosine_similarity(a.float().flatten(),b.float().flatten(),0).item()

def fn_forward_unroll(qt,kt,detach_lam):
    K2=torch.softmax(qt@kt.transpose(-2,-1),-1)
    nrm=(K2.detach() if detach_lam else K2)
    lam=(nrm.abs().sum(-2).amax(-1)*nrm.abs().sum(-1).amax(-1)/KS)[...,None,None]
    M=K2.transpose(-2,-1)@K2+lam*I
    n1=M.abs().sum(-2).amax(-1,keepdim=True).unsqueeze(-1); ninf=M.abs().sum(-1).amax(-1,keepdim=True).unsqueeze(-1)
    Z=M.transpose(-2,-1)/(n1*ninf).clamp(min=1e-12)
    for _ in range(NW): KZ=M@Z; Z=0.25*Z@(13*I-KZ@(15*I-KZ@(7*I-KZ)))
    return Z@K2.transpose(-2,-1), K2

def fn_iterates(qt,kt):  # detached NS iterates of M, for the FN hook
    with torch.no_grad():
        K2=torch.softmax(qt@kt.transpose(-2,-1),-1)
        lam=(K2.abs().sum(-2).amax(-1)*K2.abs().sum(-1).amax(-1)/KS)[...,None,None]
        M=K2.transpose(-2,-1)@K2+lam*I
        n1=M.abs().sum(-2).amax(-1,keepdim=True).unsqueeze(-1); ninf=M.abs().sum(-1).amax(-1,keepdim=True).unsqueeze(-1)
        Z=M.transpose(-2,-1)/(n1*ninf).clamp(min=1e-12); its=[Z.clone()]
        for _ in range(NW): KZ=M@Z; Z=0.25*Z@(13*I-KZ@(15*I-KZ@(7*I-KZ))); its.append(Z.clone())
        return torch.stack(its,1).contiguous(), K2

print(f"{'spread':>8} {'cond(K2)':>10} | {'lam=detach':>26} | {'lam=diff':>26}")
print(f"{'':>8} {'':>10} | {'rel':>8} {'cos':>8} {'|fn|/|rf|':>8} | {'rel':>8} {'cos':>8} {'|fn|/|rf|':>8}")
for spread in [3.0, 1.0, 0.3, 0.1, 0.03, 0.01]:
    base=torch.randn(BH,1,d,device=dev)
    qt=(base+spread*torch.randn(BH,m,d,device=dev)).contiguous()
    kt=(base+spread*torch.randn(BH,m,d,device=dev)).contiguous()
    K2c=torch.softmax(qt@kt.transpose(-2,-1),-1); sv=torch.linalg.svdvals(K2c)
    condK2=(sv[...,0]/sv[...,-1].clamp(min=1e-30)).mean().item()
    G=torch.randn(BH,m,m,device=dev)   # upstream dL/dk2inv
    nsit,K2=fn_iterates(qt,kt)
    dqt_fn,dkt_fn=_C.debug_kernel2_inv_bwd_full(qt,kt,K2.contiguous(),nsit,G.contiguous(),NW,KS)
    out=[]
    for det in (True,False):
        qtr=qt.detach().requires_grad_(); ktr=kt.detach().requires_grad_()
        k2inv,_=fn_forward_unroll(qtr,ktr,det); (k2inv*G).sum().backward()
        r=rel(dqt_fn,qtr.grad); c=cos(dqt_fn,qtr.grad); nr=(dqt_fn.norm()/qtr.grad.norm().clamp(min=1e-30)).item()
        out.append((r,c,nr))
    (r1,c1,n1_),(r2,c2,n2_)=out
    print(f"{spread:>8.3f} {condK2:>10.2e} | {r1:>8.1e} {c1:>8.4f} {n1_:>8.3f} | {r2:>8.1e} {c2:>8.4f} {n2_:>8.3f}")
sys.stdout.flush(); os._exit(0)
