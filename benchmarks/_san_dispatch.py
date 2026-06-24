import os, sys, torch
os.environ["FN_KAPPA_STAR"]="5"
from flash_nystrom.flash_nystrom import _C
dev="cuda"; torch.manual_seed(0); torch.zeros(1,device=dev)
B,H,N,d,nw=1,4,1025,64,6
for mm in (64,32):   # 64 -> TC path, 32 -> scalar fallback (the dispatch fix)
    q=torch.randn(B,H,N,d,device=dev).half();k=torch.randn(B,H,N,d,device=dev).half();v=torch.randn(B,H,N,d,device=dev).half()
    _C.forward(q,k,v,mm,nw); r=_C.forward(q,k,v,mm,nw)
    print(f"m={mm} ok finite={torch.isfinite(r[0]).all().item()}",flush=True)
torch.cuda.synchronize(); sys.stdout.flush(); os._exit(0)
