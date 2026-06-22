# Hardened sanitizer driver for the CP2 tf32 TC forward NS path (FN_K2INV_TC=1).
# os._exit avoids the Windows CUDA-teardown hang under compute-sanitizer.
import os, sys, torch
os.environ["FN_K2INV_TC"] = "1"; os.environ.pop("FN_KAPPA_STAR", None)
from flash_nystrom.flash_nystrom import _C
dev = "cuda"; torch.manual_seed(0); torch.zeros(1, device=dev)
fix = torch.load("C:/tmp/fn_real_fixture.pt"); m, nw = fix["M"], fix["NEWTON"]
N = 1025
q = fix["q0"][:, :, :N].contiguous().to(dev)
k = fix["k0"][:, :, :N].contiguous().to(dev)
v = fix["v0"][:, :, :N].contiguous().to(dev)
r = _C.forward(q, k, v, m, nw)
torch.cuda.synchronize()
print(f"SAN_K2INV_TC done k2inv_finite={torch.isfinite(r[5]).all().item()}", flush=True)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
