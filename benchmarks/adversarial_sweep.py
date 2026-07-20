# Adversarial FN-vs-reference sweep. For each (N,B,H,m,d,dtype,kappa,regime):
# run the FN kernels and the fp32 reference on identical inputs, log a PER-STAGE
# error breakdown (landmarks -> K2 -> pinv -> step1 -> step2 -> output), grad
# errors (dq,dk,dv), subnormal/flush fractions, NaN/Inf, and the conditioning
# regime hit (cond(K2), non-normality, NS residual). Output: JSON + a summary
# sorted by worst divergence, so we see WHERE the kernels break, not just that.
import os, sys, json, itertools, math, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from flash_nystrom.flash_nystrom import _C, FlashNystromFunction
from flash_nystrom.reference import iterative_pinverse
dev = "cuda"; torch.zeros(1, device=dev)
SUB = 2.0**-14; FLUSH = 2.0**-24
IDX = dict(O=0, qs=1, ks=2, qt=3, kt=4, k2inv=5, step2=6, nsit=10, k2sm=11, step1=12)

def rel(a, b): return ((a.float()-b.float()).norm()/b.float().norm().clamp(min=1e-30)).item()
def cos(a, b): return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()
def fracs(t):
    a=t.float().abs(); n=a.numel()
    return (100.0*((a>0)&(a<SUB)).sum().item()/max(n,1), 100.0*((a>0)&(a<FLUSH)).sum().item()/max(n,1))

def make_inputs(B,H,N,d,m,regime,dtype):
    g=lambda: torch.randn(B,H,N,d,device=dev)
    if regime=="randn":     q,k,v=g(),g(),g()
    elif regime=="bigmag":  q,k,v=g()*10,g()*10,g()
    elif regime=="wellcond":           # distinct landmarks -> well-conditioned K2
        q=g()*0.1; seg=N//m
        for i in range(m): q[:,:,i*seg:(i+1)*seg, i%d]+=8.0
        k=q.clone(); v=g()
    elif regime=="pathological":       # near-identical landmarks -> cond huge
        bq=torch.randn(B,H,1,d,device=dev); bk=torch.randn(B,H,1,d,device=dev)
        q=bq+0.02*g(); k=bk+0.02*g(); v=g()
    return q.to(dtype),k.to(dtype),v.to(dtype)

def ref_stages(q,k,v,m,nw,kappa):
    q,k,v=q.float(),k.float(),v.float(); B,H,N,d=q.shape; sc=d**-0.25
    qs,ks=q*sc,k*sc; seg=N//m; tr=seg*(m-1)
    qt=torch.cat([qs[:,:,:tr].reshape(B,H,m-1,seg,d).mean(3),qs[:,:,tr:].mean(2,keepdim=True)],2)
    kt=torch.cat([ks[:,:,:tr].reshape(B,H,m-1,seg,d).mean(3),ks[:,:,tr:].mean(2,keepdim=True)],2)
    K1=torch.softmax(qs@kt.transpose(-2,-1),-1); K2=torch.softmax(qt@kt.transpose(-2,-1),-1)
    K3=torch.softmax(qt@ks.transpose(-2,-1),-1)
    if kappa>0:
        n1=K2.abs().sum(-2).amax(-1); ninf=K2.abs().sum(-1).amax(-1); lam=(n1*ninf/kappa)[...,None,None]
        I=torch.eye(m,device=dev).expand_as(K2); M=K2.transpose(-2,-1)@K2+lam*I
        K2inv=iterative_pinverse(M,nw)@K2.transpose(-2,-1)
    else: K2inv=iterative_pinverse(K2,nw)
    step1=K3@v; step2=K2inv@step1; O=K1@step2
    return dict(qt=qt,kt=kt,K1=K1,K2=K2,K3=K3,k2inv=K2inv,step1=step1,step2=step2,O=O)

def run_one(B,H,N,d,m,dtype,kappa,regime,nw=6):
    torch.manual_seed(0)
    q,k,v=make_inputs(B,H,N,d,m,regime,dtype)
    # kappa_star is a forward() parameter; the old FN_KAPPA_STAR env var is gone.
    res=_C.forward(q,k,v,m,nw,float(max(kappa,0.0)),False)
    R=ref_stages(q,k,v,m,nw,kappa)
    # conditioning regime hit
    sv=torch.linalg.svdvals(R["K2"]); condK2=(sv[...,0]/sv[...,-1].clamp(min=1e-30)).mean().item()
    comm=R["K2"]@R["K2"].transpose(-2,-1)-R["K2"].transpose(-2,-1)@R["K2"]
    nn=(comm.norm(dim=(-2,-1))/(R["K2"].norm(dim=(-2,-1))**2).clamp(min=1e-30)).mean().item()
    Iexp=torch.eye(m,device=dev).expand_as(R["K2"]); nsres=((R["K2"]@res[IDX["k2inv"]].float()-Iexp).norm(dim=(-2,-1))/m**0.5).mean().item()
    # per-stage forward errors (FN vs fp32 ref)
    fe={ "qt":rel(res[IDX["qt"]],R["qt"]), "kt":rel(res[IDX["kt"]],R["kt"]),
         "K2":rel(res[IDX["k2sm"]],R["K2"]), "k2inv":rel(res[IDX["k2inv"]],R["k2inv"]),
         "step1":rel(res[IDX["step1"]],R["step1"]), "step2":rel(res[IDX["step2"]],R["step2"]),
         "O":rel(res[IDX["O"]],R["O"]), "O_cos":cos(res[IDX["O"]],R["O"]) }
    # subnormal/flush in the long-axis softmaxes
    k1s,k1f=fracs(R["K1"]); k3s,k3f=fracs(R["K3"])
    # gradients (GradScaler-style scaled dO so subnormals don't confound the accuracy check)
    dO=torch.randn(B,H,N,d,device=dev,dtype=dtype)*1024.0
    qg=q.clone().requires_grad_();kg=k.clone().requires_grad_();vg=v.clone().requires_grad_()
    FlashNystromFunction.apply(qg,kg,vg,m,nw,True,float(max(kappa,0.0)),False).backward(dO)
    q2=q.float().requires_grad_();k2=k.float().requires_grad_();v2=v.float().requires_grad_()
    from flash_nystrom.reference import nystrom_attention_reference
    nystrom_attention_reference(q2,k2,v2,m,nw,kappa_star=kappa).backward(dO.float())
    ge={"dq":rel(qg.grad,q2.grad),"dk":rel(kg.grad,k2.grad),"dv":rel(vg.grad,v2.grad),
        "dq_cos":cos(qg.grad,q2.grad),"dk_cos":cos(kg.grad,k2.grad)}
    gsub={"dq_sub":fracs(qg.grad)[0],"dk_sub":fracs(kg.grad)[0]}
    nonfinite=not all(torch.isfinite(x).all().item() for x in (res[IDX["O"]],qg.grad,kg.grad,vg.grad))
    return dict(B=B,H=H,N=N,d=d,m=m,dtype=str(dtype).split(".")[-1],kappa=kappa,regime=regime,
                condK2=condK2,nonnormal=nn,ns_resid=nsres,k1_sub=k1s,k1_flush=k1f,k3_sub=k3s,k3_flush=k3f,
                nonfinite=nonfinite,**{f"fe_{k}":v for k,v in fe.items()},**{f"ge_{k}":v for k,v in ge.items()},**gsub)

def main():
    base=dict(B=2,H=4,N=2049,d=64,m=64,dtype=torch.float16,kappa=5.0,regime="randn")
    sweeps={
      "N":[129,513,2049,9217,32769], "m":[16,32,64], "d":[64,128], "regime":["randn","wellcond","pathological","bigmag"],
      "dtype":[torch.float16,torch.bfloat16], "kappa":[0.0,5.0], "BH":[(1,1),(2,4),(4,8)],
    }
    configs=[]
    for axis,vals in sweeps.items():
        for val in vals:
            c=dict(base)
            if axis=="BH": c["B"],c["H"]=val
            else: c[axis]=val
            configs.append(c)
    # de-dup
    seen=set(); uniq=[]
    for c in configs:
        key=tuple(sorted((k,str(v)) for k,v in c.items()))
        if key not in seen: seen.add(key); uniq.append(c)
    rows=[]
    for c in uniq:
        try:
            r=run_one(**c); rows.append(r)
            print(f"N={r['N']:>6} BH={r['B']}x{r['H']} m={r['m']} d={r['d']} {r['dtype']} k={r['kappa']} {r['regime']:12s}"
                  f" cond={r['condK2']:.1e} | O rel={r['fe_O']:.1e} k2inv={r['fe_k2inv']:.1e} | dq={r['ge_dq']:.1e} dk={r['ge_dk']:.1e}"
                  f" | k3flush={r['k3_flush']:.1f}% {'NONFINITE' if r['nonfinite'] else ''}", flush=True)
        except Exception as e:
            print(f"FAIL {c}: {str(e).splitlines()[0]}", flush=True)
            rows.append(dict(**{k:(str(v) if k=='dtype' else v) for k,v in c.items()}, error=str(e).splitlines()[0]))
    json.dump(rows, open("C:/tmp/adversarial_sweep.json","w"), indent=1, default=str)
    print("\n==== WORST forward-output rel ====");
    for r in sorted([x for x in rows if 'fe_O' in x], key=lambda x:-x['fe_O'])[:8]:
        print(f"  O rel={r['fe_O']:.2e} cos={r['fe_O_cos']:.5f}  N={r['N']} BH={r['B']}x{r['H']} m={r['m']} d={r['d']} {r['dtype']} k={r['kappa']} {r['regime']}")
    print("==== WORST grad (dk) rel ====")
    for r in sorted([x for x in rows if 'ge_dk' in x], key=lambda x:-x['ge_dk'])[:8]:
        print(f"  dk rel={r['ge_dk']:.2e} cos={r['ge_dk_cos']:.5f}  N={r['N']} BH={r['B']}x{r['H']} m={r['m']} d={r['d']} {r['dtype']} k={r['kappa']} {r['regime']}")
    print(f"\nsaved {len(rows)} rows -> C:/tmp/adversarial_sweep.json")
    sys.stdout.flush(); os._exit(0)

if __name__=="__main__": main()
