# Localize the FN backward dq/dk divergence to a specific internal op. Capture a
# high-cond K2 from MNIST per-pixel, build the fp32 forward state, then drive FN's
# backward KERNELS in fp32 (debug hooks) on the SAME fp32 inputs as the fp32
# autograd reference. fp32-vs-fp32 isolates ALGORITHM from fp16 precision:
#   match  -> backward algorithm correct, the MNIST dq/dk gap is fp16
#   differ -> structural backward bug
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# kappa_star is now an explicit param (passed to apply / debug hooks / the reference), not an env var.
import torch, torch.nn as nn, torch.nn.functional as F, torchvision, torchvision.transforms as T
from flash_nystrom.flash_nystrom import _C, FlashNystromFunction
from flash_nystrom.reference import iterative_pinverse
dev="cuda"; torch.manual_seed(0)
DIM,HEADS,DEPTH,M,NW,KS=256,4,3,64,6,5.0; HD=DIM//HEADS

class Attn(nn.Module):
    def __init__(s):
        super().__init__()
        s.qp=nn.Linear(DIM,DIM,bias=False);s.kp=nn.Linear(DIM,DIM,bias=False)
        s.vp=nn.Linear(DIM,DIM,bias=False);s.op=nn.Linear(DIM,DIM,bias=False); s.cap=False; s.store={}
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
for step in range(31):
    try: xb,yb=next(it)
    except StopIteration: it=iter(dl);xb,yb=next(it)
    xb,yb=xb.to(dev),yb.to(dev)
    for b in model.blocks: b.at.cap=(step==30)
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda",dtype=torch.float16): loss=F.cross_entropy(model(xb),yb)
    scaler.scale(loss).backward()
    if step<30: scaler.step(opt);scaler.update()

def rel(a,b): return ((a.float()-b.float()).norm()/b.float().norm().clamp(min=1e-30)).item()
# pick the highest-cond layer
best=None
for li,blk in enumerate(model.blocks):
    q=blk.at.store['q'].float(); k=blk.at.store['k'].float(); B,H,N,d=q.shape; sc=d**-0.25
    qt=(q*sc)[:,:,:(N//M)*(M-1)].reshape(B,H,M-1,N//M,d).mean(3); kt=(k*sc)[:,:,:(N//M)*(M-1)].reshape(B,H,M-1,N//M,d).mean(3)
    qt=torch.cat([qt,(q*sc)[:,:,(N//M)*(M-1):].mean(2,keepdim=True)],2); kt=torch.cat([kt,(k*sc)[:,:,(N//M)*(M-1):].mean(2,keepdim=True)],2)
    K2=torch.softmax(qt@kt.transpose(-2,-1),-1); c=(torch.linalg.svdvals(K2)[...,0]/torch.linalg.svdvals(K2)[...,-1]).mean().item()
    if best is None or c>best[1]: best=(li,c)
li=best[0]; print(f"using layer {li} cond(K2)={best[1]:.2e}")
st=model.blocks[li].at.store; q=st['q'].float();k=st['k'].float();v=st['v'].float();dO=st['dO'].float()
B,H,N,d=q.shape; BH=B*H; sc=d**-0.25
def lm(x): a=x[:,:,:(N//M)*(M-1)].reshape(B,H,M-1,N//M,d).mean(3); return torch.cat([a,x[:,:,(N//M)*(M-1):].mean(2,keepdim=True)],2)
qs=q*sc; ks=k*sc; qt=lm(qs); kt=lm(ks)
# full fp32 reference with retained grads
qtr=qt.detach().requires_grad_(); ktr=kt.detach().requires_grad_(); qsr=qs.detach().requires_grad_(); ksr=ks.detach().requires_grad_(); vr=v.detach().requires_grad_()
K1=torch.softmax(qsr@ktr.transpose(-2,-1),-1); K2=torch.softmax(qtr@ktr.transpose(-2,-1),-1); S3=qtr@ksr.transpose(-2,-1); K3=torch.softmax(S3,-1); lse3=torch.logsumexp(S3,-1)
n1=K2.abs().sum(-2).amax(-1);ninf=K2.abs().sum(-1).amax(-1);lam=(n1*ninf/KS)[...,None,None]; I=torch.eye(M,device=dev).expand_as(K2)
Mm=K2.transpose(-2,-1)@K2+lam*I; k2inv=iterative_pinverse(Mm,NW)@K2.transpose(-2,-1)
step1=K3@vr; step2=k2inv@step1
for t in (step1,step2,k2inv,K2): t.retain_grad()
O=K1@step2; O.backward(dO)   # dO has shape (B,H,N,d) matching O
dstep2=step2.grad; dO3=step1.grad; dK2inv=k2inv.grad
# reference pinv-subgraph: dq~,dk~ from dK2inv alone
qtp=qt.detach().requires_grad_(); ktp=kt.detach().requires_grad_()
K2p=torch.softmax(qtp@ktp.transpose(-2,-1),-1)
lamp=(K2p.detach().abs().sum(-2).amax(-1)*K2p.detach().abs().sum(-1).amax(-1)/KS)[...,None,None]  # lambda detached (matches FN)
Mp=K2p.transpose(-2,-1)@K2p+lamp*I
k2invp=iterative_pinverse(Mp,NW)@K2p.transpose(-2,-1); (k2invp*dK2inv).sum().backward()
ref_pdqt,ref_pdkt=qtp.grad,ktp.grad
# fp32 NS iterates of M for the FN hook
Z=Mm.transpose(-2,-1)/(Mm.abs().sum(-2).amax(-1,keepdim=True).unsqueeze(-1)*Mm.abs().sum(-1).amax(-1,keepdim=True).unsqueeze(-1)); its=[Z.clone()]
for _ in range(NW): KZ=Mm@Z; Z=0.25*Z@(13*I-KZ@(15*I-KZ@(7*I-KZ))); its.append(Z.clone())
nsit=torch.stack(its,1).reshape(BH,NW+1,M,M).contiguous()
def f3(x): return x.reshape(BH,*x.shape[2:]).contiguous()
# FN backward kernels in fp32 on the SAME fp32 inputs:
dK2inv_fn,_=_C.debug_compute_dk2inv(f3(qt),f3(ks),f3(v),f3(dO3),f3(lse3),f3(dstep2))
dqt_fn,dkt_fn=_C.debug_kernel2_inv_bwd_full(f3(qt),f3(kt),f3(K2.detach()),nsit,f3(dK2inv),NW,KS)
print("== FN backward kernels (fp32) vs fp32 autograd, same fp32 inputs ==")
print(f"  dK2inv (compute_dk2inv)     rel={rel(dK2inv_fn, f3(dK2inv)):.3e}")
print(f"  dq_tilde (pinv backward)    rel={rel(dqt_fn,   f3(ref_pdqt)):.3e}")
print(f"  dk_tilde (pinv backward)    rel={rel(dkt_fn,   f3(ref_pdkt)):.3e}")
sys.stdout.flush(); os._exit(0)
