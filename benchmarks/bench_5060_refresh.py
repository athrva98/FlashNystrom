# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
# Refresh the README RTX-5060 latency table with the CURRENT default config
# (kappa_star=5, tf32 TC pinv, fast_dk2inv=True). FN vs SDPA vs cuBLAS reference,
# fwd and fwd+bwd, B=1 H=4 D=64 m=32 newton=6 fp16 — matching the README footnote.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from flash_nystrom.flash_nystrom import FlashNystromFunction
from flash_nystrom.reference import nystrom_attention_reference
dev="cuda"; dtype=torch.float16
B,H,D,m=1,4,64,32
_DEFAULT_NS="128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144"
Ns=[int(x) for x in os.environ.get("FN_BENCH_NS",_DEFAULT_NS).split(",")]
print(f"GPU: {torch.cuda.get_device_name(0)}  B={B} H={H} D={D} m={m} newton=6 {dtype} (kappa=5, TC pinv, fast_dk2inv)")

def reps(N):
    if N<=8192: return 5,30
    if N<=32768: return 5,15
    if N<=131072: return 3,8
    return 2,5

def t(fn,w,r):
    try:
        for _ in range(w): fn()
        torch.cuda.synchronize()
        evs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(r)]
        for s,e in evs: s.record(); fn(); e.record()
        torch.cuda.synchronize()
        return sorted(s.elapsed_time(e) for s,e in evs)[r//2]
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache(); return float("nan")

def g(N): return tuple(torch.randn(B,H,N,D,device=dev,dtype=dtype) for _ in range(4))
def fn_fwd(q,k,v):
    with torch.no_grad(): return FlashNystromFunction.apply(q,k,v,m,6,True,5.0,True)
def ref_fwd(q,k,v):
    with torch.no_grad(): return nystrom_attention_reference(q,k,v,m,6,None,0,5.0)
def fb(impl,q,k,v,dO):
    def run():
        qq=q.detach().requires_grad_();kk=k.detach().requires_grad_();vv=v.detach().requires_grad_()
        if impl=="fn": o=FlashNystromFunction.apply(qq,kk,vv,m,6,True,5.0,True)
        elif impl=="sdpa": o=F.scaled_dot_product_attention(qq,kk,vv)
        else: o=nystrom_attention_reference(qq,kk,vv,m,6,None,0,5.0)
        o.backward(dO)
    return run
def rx(a,b): return f"{b/a:.1f}x" if (a==a and b==b and a>0) else "  -"

hdr=f"{'N':>8} | {'FN fwd':>8} {'FN tot':>8} | {'SDPA fwd':>9} {'SDPA tot':>9} | {'cuB fwd':>8} {'cuB tot':>8} | {'vs SDPA':>8} {'vs cuB':>7}"
print(hdr); print("-"*len(hdr)); sys.stdout.flush()
for N in Ns:
    w,r=reps(N); q,k,v,dO=g(N)
    fnf=t(lambda:fn_fwd(q,k,v),w,r); fnt=t(fb("fn",q,k,v,dO),w,r)
    sdf=t(lambda:F.scaled_dot_product_attention(q,k,v),w,r); sdt=t(fb("sdpa",q,k,v,dO),w,r)
    cbf=t(lambda:ref_fwd(q,k,v),w,r); cbt=t(fb("ref",q,k,v,dO),w,r)
    print(f"{N:>8} | {fnf:8.3f} {fnt:8.3f} | {sdf:9.3f} {sdt:9.3f} | {cbf:8.3f} {cbt:8.3f} | {rx(fnt,sdt):>8} {rx(fnt,cbt):>7}",flush=True)
    del q,k,v,dO; torch.cuda.empty_cache()
print("\nvs SDPA / vs cuB = their_total / FN_total. >1 = FN faster. '-' = OOM (SDPA is O(N^2)).")
sys.stdout.flush(); os._exit(0)
