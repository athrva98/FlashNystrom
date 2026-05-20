# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Nsight Compute workload: FlashNystrom kernel3 vs the equivalent cuBLAS GEMMs.

kernel3 computes, per batch-head:
    S = softmax(Q_tilde(m,D) @ K(N,D)^T)        -> (m, N)
    O = S @ V(N,D)                              -> (m, D)
and FN fuses that into one custom kernel. The pure-PyTorch reference does the
same math as two cuBLAS batched GEMMs plus a torch softmax. This script runs
both, each wrapped in an NVTX range, so `ncu --nvtx` can profile just the
steady-state calls (warmup excluded). See tools/profile_ncu.{ps1,sh}.

Run directly (no profiler) for a quick wall-clock sanity check, or under ncu
for the hardware counters.

    python tools/ncu_workload.py [B H N D m]
    default: 1 8 4096 128 64
"""
import os
import sys

import torch

B, H, N, D, m = 1, 8, 4096, 128, 64
if len(sys.argv) >= 6:
    B, H, N, D, m = (int(x) for x in sys.argv[1:6])

assert torch.cuda.is_available(), "CUDA GPU required"
torch.manual_seed(0)
dtype = torch.float16
dev = "cuda"

print(f"workload: B={B} H={H} N={N} D={D} m={m} dtype={dtype} "
      f"GPU={torch.cuda.get_device_name(0)}")

BH = B * H
q = torch.randn(B, H, N, D, dtype=dtype, device=dev)
k = torch.randn(B, H, N, D, dtype=dtype, device=dev)
v = torch.randn(B, H, N, D, dtype=dtype, device=dev)

# Landmark-shaped operands for the cuBLAS equivalent of kernel3's GEMMs.
qt = torch.randn(BH, m, D, dtype=dtype, device=dev)
kf = k.reshape(BH, N, D)
vf = v.reshape(BH, N, D)

import flash_nystrom._C as _C  # noqa: E402

# Auto-dispatch (split-N when the GPU is underfilled). Override externally with
# FLASH_NYSTROM_KERNEL3_SPLITS=1 to profile the single-CTA path instead.
os.environ.setdefault("FLASH_NYSTROM_KERNEL3_SPLITS", "0")


def cublas_equiv():
    """kernel3's math via cuBLAS batched GEMM + torch softmax."""
    S = torch.bmm(qt, kf.transpose(1, 2))   # (BH, m, N) = Qt @ K^T  -> cuBLAS
    P = torch.softmax(S, dim=-1)            # torch fused softmax
    return torch.bmm(P, vf)                 # (BH, m, D) = P @ V      -> cuBLAS


def fn_forward():
    """Full FN forward; kernel3 is the kernel of interest in the report."""
    return _C.forward(q, k, v, m, 6)


with torch.no_grad():
    # Warm up both paths (kernel load, graph capture, allocator) OUTSIDE the
    # NVTX ranges so the profiled launches are steady-state.
    for _ in range(5):
        cublas_equiv()
        fn_forward()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("prof_cublas")
    cublas_equiv()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("prof_fn")
    fn_forward()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

print("workload done")
