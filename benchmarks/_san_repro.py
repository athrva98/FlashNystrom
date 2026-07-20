import os, sys, torch
from flash_nystrom.flash_nystrom import FlashNystromFunction
dev="cuda"; torch.manual_seed(0); torch.zeros(1,device=dev)
N   = int(os.environ.get("SAN_N", "9217"))
B,H,D,m = 1, 4, 64, 64
fast = os.environ.get("SAN_FAST","1") == "1"
kappa = float(os.environ.get("SAN_KAPPA","0"))
tc = os.environ.get("SAN_TC","0") == "1"
dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("SAN_DTYPE","fp16")]
q=torch.randn(B,H,N,D,device=dev,dtype=dt,requires_grad=True)
k=torch.randn(B,H,N,D,device=dev,dtype=dt,requires_grad=True)
v=torch.randn(B,H,N,D,device=dev,dtype=dt,requires_grad=True)
dO=(torch.randn(B,H,N,D,device=dev,dtype=dt)*1e-3)
o=FlashNystromFunction.apply(q,k,v,m,6,fast,kappa,tc)
o.backward(dO)
torch.cuda.synchronize()
print(f"SAN done: N={N} fast_dk2inv={fast} FN_FP32_BWD={os.environ.get('FN_FP32_BWD','0')} "
      f"dQ_finite={torch.isfinite(q.grad).all().item()} dK_finite={torch.isfinite(k.grad).all().item()}")
