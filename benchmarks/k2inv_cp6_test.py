# CP6 gate: TC pinv is the DEFAULT. The promotion must not REGRESS vs the previous
# default (fp32-scalar pinv), so we gate on TC-vs-scalar grads at the SAME dtype
# (a tf32-vs-fp32 NS difference, dtype-independent of the K1/K3 path). TC-vs-fp32-
# reference is reported for context (dtype-bounded, inherent to fp16/bf16, not TC).
# No-ridge uses a constructed well-conditioned K2; ridge uses the real fixture.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from flash_nystrom.flash_nystrom import FlashNystromFunction
from flash_nystrom.reference import nystrom_attention_reference
dev = "cuda"; torch.manual_seed(0); torch.zeros(1, device=dev)
fix = torch.load("C:/tmp/fn_real_fixture.pt"); m, nw = fix["M"], fix["NEWTON"]
def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-20)).item()

def inputs(dtype, kappa):
    if kappa == 0:   # constructed well-conditioned (NS converges without ridge)
        B, H, D, seg = 1, 4, 64, 16; N = m * seg
        q = torch.randn(B, H, N, D, device=dev) * 0.1
        for i in range(m): q[:, :, i*seg:(i+1)*seg, i % D] += 8.0
        return q.to(dtype), q.clone().to(dtype), torch.randn(B, H, N, D, device=dev).to(dtype), torch.randn(B, H, N, D, device=dev).to(dtype) * 10.0
    return (fix["q0"].to(dtype).to(dev), fix["k0"].to(dtype).to(dev), fix["v0"].to(dtype).to(dev), fix["dO0"].to(dtype).to(dev) * 1024.0)

def grads(q0, k0, v0, dO, tc, kappa):
    # use_tc_pinv / kappa_star are apply() parameters now (the FN_K2INV_TC and
    # FN_KAPPA_STAR env vars they replaced are no longer read anywhere).
    q = q0.clone().requires_grad_(); k = k0.clone().requires_grad_(); v = v0.clone().requires_grad_()
    FlashNystromFunction.apply(q, k, v, m, nw, True, kappa, tc).backward(dO)
    return q.grad, k.grad, v.grad

allok = True
for dtype in (torch.float16, torch.bfloat16):
    for kappa in (0.0, 5.0):
        q0, k0, v0, dO = inputs(dtype, kappa)
        gt = grads(q0, k0, v0, dO, True, kappa)    # TC (new default)
        gs = grads(q0, k0, v0, dO, False, kappa)   # fp32-scalar (old default)
        q2 = q0.clone().requires_grad_(); k2 = k0.clone().requires_grad_(); v2 = v0.clone().requires_grad_()
        nystrom_attention_reference(q2, k2, v2, m, nw, kappa_star=kappa).backward(dO)
        refg = (q2.grad, k2.grad, v2.grad)
        rr = [rel(gt[i], gs[i]) for i in range(3)]          # tf32-pinv delta (TC vs scalar)
        rs = [rel(gs[i], refg[i]) for i in range(3)]        # dtype quantization floor (scalar vs fp32)
        # Gate: the tf32-pinv perturbation stays within the dtype's own error floor
        # (+ a small abs margin), i.e. switching the default to tf32 is within noise.
        ok = all(rr[i] <= rs[i] + 5e-3 for i in range(3))
        allok &= ok
        dt = str(dtype).split('.')[-1]
        print(f"  {dt:8s} kappa={kappa}: tf32-pinv-delta={rr[0]:.1e}/{rr[1]:.1e}/{rr[2]:.1e}  "
              f"dtype-floor={rs[0]:.1e}/{rs[1]:.1e}/{rs[2]:.1e}  {'OK' if ok else 'REGRESSION <--'}")
print("RESULT:", "ALL OK" if allok else "FAILURES")
sys.stdout.flush(); os._exit(0 if allok else 1)
