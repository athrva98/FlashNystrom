# CP2 gate: tf32 TC forward NS (FN_K2INV_TC=1, no-ridge) vs scalar kernel.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from flash_nystrom.flash_nystrom import _C
IDX_K2INV, IDX_NSITER, IDX_K2SM = 5, 10, 11
dev = "cuda"; torch.manual_seed(0); torch.zeros(1, device=dev)

def fwd(q, k, v, m, nw, tc):
    os.environ.pop("FN_KAPPA_STAR", None); os.environ["FN_K2INV_TC"] = "1" if tc else "0"
    r = _C.forward(q.contiguous(), k.contiguous(), v.contiguous(), m, nw)
    return r[IDX_K2INV].float(), r[IDX_NSITER].float(), r[IDX_K2SM].float()

def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-20)).item()

allok = True
fix = torch.load(os.environ.get("CAP_OUT", "C:/tmp/fn_real_fixture.pt"))
m, nw = fix["M"], fix["NEWTON"]

# A) real (ill-conditioned) K2: each single NS step must match scalar to tf32 prec.
q = fix["q0"][:, :, :1024].to(dev); k = fix["k0"][:, :, :1024].to(dev); v = fix["v0"][:, :, :1024].to(dev)
_, nss, _ = fwd(q, k, v, m, nw, tc=False)
_, nst, _ = fwd(q, k, v, m, nw, tc=True)
print("A) real K2, per-iterate TC-vs-scalar rel (smooth growth = tf32, not a bug):")
for j in range(nw + 1):
    rj = rel(nst[:, :, j], nss[:, :, j]); print(f"   iterate[{j}] rel={rj:.2e}")
    if j == 0: allok &= rj < 1e-5
    if j == 1: allok &= rj < 1e-2

# B) constructed well-conditioned K2: NS converges, TC must reach K2^-1.
B, H, D, seg = 1, 4, 64, 8; Nb = m * seg
qb = torch.randn(B, H, Nb, D, device=dev) * 0.1
for i in range(m): qb[:, :, i*seg:(i+1)*seg, i % D] += 8.0
qb, kb, vb = qb.half(), qb.clone().half(), torch.randn(B, H, Nb, D, device=dev).half()
k2s, _, K2 = fwd(qb, kb, vb, m, nw, tc=False)
k2t, _, _  = fwd(qb, kb, vb, m, nw, tc=True)
I = torch.eye(m, device=dev).expand_as(K2)
resid = ((K2 @ k2t - I).norm(dim=(-2, -1)) / m**0.5).mean().item()
inv = torch.linalg.inv(K2)
print(f"B) constructed K2: residual={resid:.2e}  TC-vs-scalar={rel(k2t,k2s):.2e}  TC-vs-inv={rel(k2t,inv):.2e}")
allok &= resid < 1e-2 and rel(k2t, k2s) < 3e-3 and rel(k2t, inv) < 1e-2
print("RESULT:", "ALL OK" if allok else "FAILURES")
sys.exit(0 if allok else 1)
