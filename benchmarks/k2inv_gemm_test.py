# CP1 test: the standalone tf32 tensor-core batched GEMM primitive vs torch.bmm.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from flash_nystrom.flash_nystrom import _C
dev = "cuda"; torch.manual_seed(0); torch.zeros(1, device=dev)

def check(A, B, tag):
    C = _C.debug_k2inv_gemm_nn(A.contiguous(), B.contiguous())
    ref = torch.bmm(A.reshape(-1, *A.shape[-2:]), B.reshape(-1, *B.shape[-2:])).reshape_as(C)
    cos = F.cosine_similarity(C.flatten(), ref.flatten(), dim=0).item()
    rel = ((C - ref).norm() / ref.norm()).item()
    ok = cos > 0.99999 and rel < 3e-3   # tf32: ~10-bit mantissa
    print(f"  {tag:34s} cos={cos:.7f} rel={rel:.2e} {'OK' if ok else 'FAIL <----'}")
    return ok

allok = True
m = 64
# random matrices, several BH
for BH in (1, 8, 16, 64):
    A = torch.randn(BH, m, m, device=dev)
    B = torch.randn(BH, m, m, device=dev)
    allok &= check(A, B, f"randn BH={BH}")
# realistic magnitudes: a row-stochastic K2 @ a pinv-scale matrix
K2 = torch.softmax(torch.randn(16, m, m, device=dev), dim=-1)
allok &= check(K2, K2.transpose(-2, -1).contiguous(), "softmax K2 @ K2^T")
allok &= check(K2, torch.randn(16, m, m, device=dev) * 5.0, "K2 @ (scale 5)")
print("\nRESULT:", "ALL OK" if allok else "FAILURES")
sys.exit(0 if allok else 1)
