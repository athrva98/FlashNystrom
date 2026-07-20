# Reusable kernel2_inv checkpoint harness. Loads the real-data fixture and either
#   --save : record current-kernel golden (k2inv, ns_iterates) per layer/kappa
#   --check: compare current build vs golden AND vs the reference Tikhonov pinv
# Usage: python benchmarks/k2inv_test.py [--save|--check] [--kappa 0,5]
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from flash_nystrom.flash_nystrom import _C
from flash_nystrom.reference import iterative_pinverse

FIX = os.environ.get("CAP_OUT", "C:/tmp/fn_real_fixture.pt")
GOLD = os.environ.get("K2INV_GOLD", "C:/tmp/fn_k2inv_golden.pt")
IDX_K2INV, IDX_NSITER, IDX_K2SM = 5, 10, 11

def fwd_layer(fix, li, kappa):
    dev = "cuda"
    q = fix[f"q{li}"].to(dev); k = fix[f"k{li}"].to(dev); v = fix[f"v{li}"].to(dev)
    m, nw = fix["M"], fix["NEWTON"]
    # kappa_star is a forward() parameter; the old FN_KAPPA_STAR env var is gone.
    res = _C.forward(q, k, v, m, nw, float(max(kappa, 0.0)), False)
    return (res[IDX_K2INV].float().cpu(), res[IDX_NSITER].float().cpu(), res[IDX_K2SM].float().cpu())

def ref_tikhonov(k2sm, kappa, nw):
    K2 = k2sm.cuda()
    if kappa <= 0:
        return iterative_pinverse(K2, nw).cpu()
    n1 = K2.abs().sum(-2).amax(-1); ninf = K2.abs().sum(-1).amax(-1)
    lam = (n1*ninf/kappa)[...,None,None]
    I = torch.eye(K2.shape[-1], device=K2.device).expand_as(K2)
    M = K2.transpose(-2,-1)@K2 + lam*I
    return (iterative_pinverse(M, nw) @ K2.transpose(-2,-1)).cpu()

def cmp(a, b, tag, tol=2e-3):
    c = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    r = ((a-b).norm()/b.norm().clamp(min=1e-20)).item()
    ok = (c > 0.9995 and r < tol)
    print(f"    {tag:28s} cos={c:.6f} rel={r:.2e} {'OK' if ok else 'FAIL <----'}")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--kappa", default="0,5")
    a = ap.parse_args()
    fix = torch.load(FIX)
    kappas = [float(x) for x in a.kappa.split(",")]
    torch.zeros(1, device="cuda")
    if a.save:
        gold = {}
        for li in fix["layers"]:
            for kp in kappas:
                k2inv, nsit, _ = fwd_layer(fix, li, kp)
                gold[(li, kp)] = (k2inv, nsit)
                print(f"layer {li} kappa={kp}: saved k2inv{tuple(k2inv.shape)} nsiter{tuple(nsit.shape)}")
        torch.save(gold, GOLD); print(f"golden -> {GOLD}")
        return
    # --check
    gold = torch.load(GOLD) if os.path.exists(GOLD) else {}
    allok = True
    for li in fix["layers"]:
        for kp in kappas:
            print(f"layer {li} kappa={kp}:")
            k2inv, nsit, k2sm = fwd_layer(fix, li, kp)
            ref = ref_tikhonov(k2sm, kp, fix["NEWTON"])
            allok &= cmp(k2inv, ref, "k2inv vs reference")
            if (li, kp) in gold:
                allok &= cmp(k2inv, gold[(li, kp)][0], "k2inv vs golden")
                allok &= cmp(nsit, gold[(li, kp)][1], "ns_iterates vs golden", tol=5e-3)
    print("\nRESULT:", "ALL OK" if allok else "FAILURES PRESENT")
    sys.exit(0 if allok else 1)

if __name__ == "__main__":
    main()
