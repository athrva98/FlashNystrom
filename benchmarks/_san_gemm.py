# Hardened sanitizer driver for the CP1 tf32 GEMM primitive. Exits via os._exit
# right after the kernel work so compute-sanitizer does not stall on PyTorch's
# CUDA-context teardown (a known Windows hang). Errors/races are reported by the
# sanitizer as kernels execute, before this exit; the summary prints on child exit.
import os, sys, torch
from flash_nystrom.flash_nystrom import _C
dev = "cuda"; torch.manual_seed(0); torch.zeros(1, device=dev)
m = 64; ok = True
for BH in (8, 16):
    A = torch.randn(BH, m, m, device=dev); B = torch.randn(BH, m, m, device=dev)
    C = _C.debug_k2inv_gemm_nn(A.contiguous(), B.contiguous())
    rel = ((C - torch.bmm(A, B)).norm() / torch.bmm(A, B).norm()).item()
    ok &= rel < 3e-3
    print(f"  BH={BH} rel={rel:.2e}", flush=True)
torch.cuda.synchronize()
print(f"SAN_GEMM done ok={ok}", flush=True)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
