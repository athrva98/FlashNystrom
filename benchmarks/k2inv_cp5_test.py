# CP5 gate: Tikhonov ridge on the tf32 TC path (FN_K2INV_TC=1, FN_KAPPA_STAR=5)
# vs the reference, on real fixture data (large N, where the ridge makes NS converge).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["FN_K2INV_TC"] = "1"; os.environ["FN_KAPPA_STAR"] = "5"
import torch
from flash_nystrom.flash_nystrom import _C, FlashNystromFunction
from flash_nystrom.reference import nystrom_attention_reference, iterative_pinverse
dev = "cuda"; torch.zeros(1, device=dev)
fix = torch.load("C:/tmp/fn_real_fixture.pt"); m, nw, KS = fix["M"], fix["NEWTON"], 5.0
def rel(a, b): return ((a - b).norm() / b.norm().clamp(min=1e-20)).item()
allok = True

# A) forward k2inv vs reference Tikhonov on the same K2.
q, k, v = fix["q0"].to(dev), fix["k0"].to(dev), fix["v0"].to(dev)
r = _C.forward(q, k, v, m, nw); k2inv = r[5].float(); K2 = r[11].float()
n1 = K2.abs().sum(-2).amax(-1); ninf = K2.abs().sum(-1).amax(-1)
lam = (n1 * ninf / KS)[..., None, None]
I = torch.eye(m, device=dev).expand_as(K2)
ref = iterative_pinverse(K2.transpose(-2, -1) @ K2 + lam * I, nw) @ K2.transpose(-2, -1)
ra = rel(k2inv, ref); print(f"A) TC ridge k2inv vs reference Tikhonov: rel={ra:.2e}")
allok &= ra < 5e-3

# B) full fwd+bwd grads vs reference (TC forward + existing Tikhonov backward).
# Scale dO like GradScaler (the training loop does this): the raw upstream grad is
# ~7e-5, which underflows fp16 in the backward. Real training never sees that;
# testing the algorithm (not fp16 subnormals) requires a realistic grad scale.
dO = fix["dO0"].to(dev) * 1024.0
def fn_grads():
    q = fix["q0"].to(dev).requires_grad_(); k = fix["k0"].to(dev).requires_grad_(); v = fix["v0"].to(dev).requires_grad_()
    o = FlashNystromFunction.apply(q, k, v, m, nw, True); o.backward(dO)
    return o.detach(), q.grad, k.grad, v.grad
oF, gq, gk, gv = fn_grads()
q2 = fix["q0"].to(dev).requires_grad_(); k2 = fix["k0"].to(dev).requires_grad_(); v2 = fix["v0"].to(dev).requires_grad_()
oR = nystrom_attention_reference(q2, k2, v2, m, nw, kappa_star=KS); oR.backward(dO)
print(f"B) grads vs reference: o={rel(oF,oR):.2e} dq={rel(gq,q2.grad):.2e} "
      f"dk={rel(gk,k2.grad):.2e} dv={rel(gv,v2.grad):.2e}")
allok &= rel(oF, oR) < 5e-3 and rel(gq, q2.grad) < 5e-2 and rel(gk, k2.grad) < 5e-2 and rel(gv, v2.grad) < 5e-2
print("RESULT:", "ALL OK" if allok else "FAILURES")
sys.stdout.flush(); os._exit(0 if allok else 1)   # os._exit: skip CUDA teardown (Windows hang)
