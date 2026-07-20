# Hardened sanitizer driver for the CP2 tf32 TC forward NS path (use_tc_pinv=True).
# os._exit avoids the Windows CUDA-teardown hang under compute-sanitizer.
import os, sys, torch
from flash_nystrom.flash_nystrom import _C
# use_tc_pinv / kappa_star are forward() parameters now (FN_K2INV_TC / FN_KAPPA_STAR are gone).
KAPPA = float(os.environ.get("SAN_KAPPA", "0"))
dev = "cuda"; torch.manual_seed(0); torch.zeros(1, device=dev)
fix = torch.load("C:/tmp/fn_real_fixture.pt"); m, nw = fix["M"], fix["NEWTON"]
dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[os.environ.get("SAN_DTYPE", "fp16")]
N = 1025
q = fix["q0"][:, :, :N].contiguous().to(dev).to(dt)
k = fix["k0"][:, :, :N].contiguous().to(dev).to(dt)
v = fix["v0"][:, :, :N].contiguous().to(dev).to(dt)
_C.forward(q, k, v, m, nw, KAPPA, True)        # graph capture
r = _C.forward(q, k, v, m, nw, KAPPA, True)    # graph replay
torch.cuda.synchronize()
print(f"SAN_K2INV_TC done k2inv_finite={torch.isfinite(r[5]).all().item()}", flush=True)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
