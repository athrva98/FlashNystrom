# CP2 gate for the tf32 TC forward NS (FN_K2INV_TC=1, no-ridge). Two checks:
#   A) REAL data, per-iteration: one NS step (4 GEMMs + affines) must match the
#      scalar kernel to tf32 precision. Real K2 is ill-conditioned so the chain
#      does not converge over 6 iters (that is the ridge's job), but each single
#      iteration is a well-defined map -> ns_iterates[1] must match closely. A
#      mis-wired chain (wrong affine const / GEMM operand) would be far off.
#   B) CONSTRUCTED well-conditioned K2 (diagonally dominant) where NS converges,
#      so TC must reach K2^-1 and match the scalar kernel and torch.linalg.inv.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
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
# ---- A) real data, per-iteration ----
fix = torch.load(os.environ.get("CAP_OUT", "C:/tmp/fn_real_fixture.pt"))
m, nw = fix["M"], fix["NEWTON"]
N = 1024
q = fix["q0"][:, :, :N].to(dev); k = fix["k0"][:, :, :N].to(dev); v = fix["v0"][:, :, :N].to(dev)
_, nss, _ = fwd(q, k, v, m, nw, tc=False)
_, nst, _ = fwd(q, k, v, m, nw, tc=True)
print("A) real K2 (ill-conditioned), per-iterate TC-vs-scalar rel:")
for j in range(nw + 1):
    rj = rel(nst[:, :, j], nss[:, :, j])
    tag = "  (Z_0 setup, must be ~0)" if j == 0 else ("  <-- gate: one NS step" if j == 1 else "")
    print(f"    iterate[{j}] rel={rj:.2e}{tag}")
    if j == 0: allok &= rj < 1e-5
    if j == 1: allok &= rj < 1e-2

# ---- B) constructed well-conditioned K2 (NS converges) ----
B, H, D = 1, 4, 64; seg = 8; Nb = m * seg
qb = (torch.randn(B, H, Nb, D, device=dev) * 0.1)
for i in range(m):
    qb[:, :, i*seg:(i+1)*seg, i % D] += 8.0          # segment i -> landmark ~ e_i (distinct)
kb = qb.clone(); vb = torch.randn(B, H, Nb, D, device=dev)
qb, kb, vb = qb.half(), kb.half(), vb.half()
k2s, _, K2 = fwd(qb, kb, vb, m, nw, tc=False)
k2t, _, _  = fwd(qb, kb, vb, m, nw, tc=True)
I = torch.eye(m, device=dev).expand_as(K2)
resid = ((K2 @ k2t - I).norm(dim=(-2, -1)) / m**0.5).mean().item()
cond = (torch.linalg.svdvals(K2)[..., 0] / torch.linalg.svdvals(K2)[..., -1].clamp(min=1e-20)).mean().item()
inv = torch.linalg.inv(K2)
print(f"\nB) constructed K2: cond={cond:.2f}  TC residual ||K2 k2inv - I||={resid:.2e}")
print(f"    converged          : {'OK' if resid < 1e-2 else 'FAIL <--'}")
print(f"    TC vs scalar       : rel={rel(k2t, k2s):.2e} {'OK' if rel(k2t,k2s)<3e-3 else 'FAIL <--'}")
print(f"    TC vs torch.inv    : rel={rel(k2t, inv):.2e} {'OK' if rel(k2t,inv)<1e-2 else 'FAIL <--'}")
allok &= resid < 1e-2 and rel(k2t, k2s) < 3e-3 and rel(k2t, inv) < 1e-2
print("\nRESULT:", "ALL OK" if allok else "FAILURES")
sys.exit(0 if allok else 1)
