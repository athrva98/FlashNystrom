# MNIST per-pixel (N=785 tokens), FlashNystrom vs nystrom_reference: train a small
# model with FN attention, capture real per-layer q,k,v + GradScaler-scaled dO, then
# diff every intermediate (FN vs fp32 reference) per layer to localize where they diverge.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# kappa_star is now an explicit param (passed to apply / _C.forward / the reference), not an env var.
import torch, torch.nn as nn, torch.nn.functional as F, torchvision, torchvision.transforms as T
from flash_nystrom.flash_nystrom import _C, FlashNystromFunction
from flash_nystrom.reference import iterative_pinverse, nystrom_attention_reference
dev="cuda"; torch.manual_seed(0)
DIM,HEADS,DEPTH,M,NW=256,4,3,64,6; HD=DIM//HEADS

class Attn(nn.Module):
    def __init__(s):
        super().__init__()
        s.qp=nn.Linear(DIM,DIM,bias=False);s.kp=nn.Linear(DIM,DIM,bias=False)
        s.vp=nn.Linear(DIM,DIM,bias=False);s.op=nn.Linear(DIM,DIM,bias=False)
        s.cap=False; s.store={}
    def forward(s,x):
        B,N,_=x.shape
        q=s.qp(x).view(B,N,HEADS,HD).transpose(1,2).contiguous()
        k=s.kp(x).view(B,N,HEADS,HD).transpose(1,2).contiguous()
        v=s.vp(x).view(B,N,HEADS,HD).transpose(1,2).contiguous()
        o=FlashNystromFunction.apply(q,k,v,M,NW,True,5.0,True)
        if s.cap:
            s.store=dict(q=q.detach().clone(),k=k.detach().clone(),v=v.detach().clone())
            o.register_hook(lambda g: s.store.__setitem__('dO',g.detach().clone()))
        return s.op(o.transpose(1,2).contiguous().view(B,N,DIM))

class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.n1=nn.LayerNorm(DIM); s.at=Attn(); s.n2=nn.LayerNorm(DIM)
        s.mlp=nn.Sequential(nn.Linear(DIM,4*DIM),nn.GELU(),nn.Linear(4*DIM,DIM))
    def forward(s,x): x=x+s.at(s.n1(x)); return x+s.mlp(s.n2(x))

class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.pe=nn.Conv2d(1,DIM,1,1); n=28*28
        s.pos=nn.Parameter(torch.randn(1,n+1,DIM)*0.02); s.cls=nn.Parameter(torch.randn(1,1,DIM)*0.02)
        s.blocks=nn.ModuleList([Block() for _ in range(DEPTH)]); s.norm=nn.LayerNorm(DIM); s.head=nn.Linear(DIM,10)
    def forward(s,x):
        B=x.shape[0]; x=s.pe(x).flatten(2).transpose(1,2)
        x=torch.cat([s.cls.expand(B,-1,-1),x],1)+s.pos
        for b in s.blocks: x=b(x)
        return s.head(s.norm(x)[:,0])

model=Net().to(dev)
tf=T.Compose([T.ToTensor()])
ds=torchvision.datasets.MNIST(root=os.environ.get("CAP_DATA_ROOT","./data"),train=True,download=True,transform=tf)
dl=torch.utils.data.DataLoader(ds,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.05); scaler=torch.amp.GradScaler("cuda")
print(f"MNIST per-pixel N={28*28+1} m={M} dim={DIM} heads={HEADS} d={HD} depth={DEPTH}",flush=True)
it=iter(dl); WARM=30
for step in range(WARM+1):
    try: xb,yb=next(it)
    except StopIteration: it=iter(dl); xb,yb=next(it)
    xb,yb=xb.to(dev),yb.to(dev); cap=(step==WARM)
    for b in model.blocks: b.at.cap=cap
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda",dtype=torch.float16): loss=F.cross_entropy(model(xb),yb)
    scaler.scale(loss).backward()
    if not cap: scaler.step(opt); scaler.update()
    if step%10==0: print(f"  step {step} loss={loss.item():.3f}",flush=True)

def ref_stages(q,k,v,kappa):
    q,k,v=q.float(),k.float(),v.float(); B,H,N,d=q.shape; sc=d**-0.25
    qs,ks=q*sc,k*sc; seg=N//M; tr=seg*(M-1)
    qt=torch.cat([qs[:,:,:tr].reshape(B,H,M-1,seg,d).mean(3),qs[:,:,tr:].mean(2,keepdim=True)],2)
    kt=torch.cat([ks[:,:,:tr].reshape(B,H,M-1,seg,d).mean(3),ks[:,:,tr:].mean(2,keepdim=True)],2)
    K1=torch.softmax(qs@kt.transpose(-2,-1),-1);K2=torch.softmax(qt@kt.transpose(-2,-1),-1);K3=torch.softmax(qt@ks.transpose(-2,-1),-1)
    n1=K2.abs().sum(-2).amax(-1);ninf=K2.abs().sum(-1).amax(-1);lam=(n1*ninf/kappa)[...,None,None]
    I=torch.eye(M,device=dev).expand_as(K2);Mm=K2.transpose(-2,-1)@K2+lam*I
    k2inv=iterative_pinverse(Mm,NW)@K2.transpose(-2,-1);step1=K3@v;step2=k2inv@step1;O=K1@step2
    sv=torch.linalg.svdvals(K2);cond=(sv[...,0]/sv[...,-1].clamp(min=1e-30)).mean().item()
    return dict(qt=qt,kt=kt,K2=K2,k2inv=k2inv,step1=step1,step2=step2,O=O,cond=cond)

def rel(a,b): return ((a.float()-b.float()).norm()/b.float().norm().clamp(min=1e-30)).item()
def co(a,b): return F.cosine_similarity(a.float().flatten(),b.float().flatten(),0).item()
print("\nper-layer FN vs fp32 reference (real MNIST activations):")
for li,blk in enumerate(model.blocks):
    st=blk.at.store
    if 'dO' not in st: print(f"layer {li}: no dO"); continue
    q,k,v,dO=st['q'],st['k'],st['v'],st['dO']
    res=_C.forward(q,k,v,M,NW,5.0,True); R=ref_stages(q,k,v,5.0)
    qg=q.clone().requires_grad_();kg=k.clone().requires_grad_();vg=v.clone().requires_grad_()
    FlashNystromFunction.apply(qg,kg,vg,M,NW,True,5.0,True).backward(dO)
    q2=q.float().requires_grad_();k2=k.float().requires_grad_();v2=v.float().requires_grad_()
    nystrom_attention_reference(q2,k2,v2,M,NW,kappa_star=5.0).backward(dO.float())   # ref fp32
    q3=q.float().requires_grad_();k3=k.float().requires_grad_();v3=v.float().requires_grad_()
    with torch.amp.autocast("cuda",dtype=torch.float16):                            # ref fp16 (training-equiv)
        o16=nystrom_attention_reference(q3,k3,v3,M,NW,kappa_star=5.0)
    o16.backward(dO.float())
    print(f"layer {li}: cond(K2)={R['cond']:.2e}")
    for nm,a,b in [("qt",res[3],R['qt']),("kt",res[4],R['kt']),("K2",res[11],R['K2']),("k2inv",res[5],R['k2inv']),
                   ("step1",res[12],R['step1']),("step2",res[6],R['step2']),("output",res[0],R['O'])]:
        print(f"    {nm:8s} FN-vs-fp32 rel={rel(a,b):.3e}  cos={co(a,b):.6f}")
    for nm,a,b16,bf in [("dq",qg.grad,q3.grad,q2.grad),("dk",kg.grad,k3.grad,k2.grad),("dv",vg.grad,v3.grad,v2.grad)]:
        print(f"    {nm:8s} FN-vs-fp32 rel={rel(a,bf):.3e}  ref_fp16-vs-fp32 rel={rel(b16,bf):.3e}  FN-vs-ref_fp16 rel={rel(a,b16):.3e}")
sys.stdout.flush(); os._exit(0)
