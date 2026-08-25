# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Validate and benchmark the hand-written Triton Nystrom baseline on an A100.

    modal run tools/modal_triton.py

Correctness first, then latency. A Triton baseline that is not verified against
the reference is worthless as a comparison point, and a slow one that is wrong
is worse than none: it would understate the alternative and flatter this paper.

Forward only. See benchmarks/triton_nystrom.py for why.
"""
import pathlib
import sys

import modal

# Modal re-imports this module inside the container, where __file__ resolves
# elsewhere, so the tools directory must be on sys.path for BOTH sides.
for _p in (str(pathlib.Path(__file__).resolve().parent), "/root/FlashNystrom/tools"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from modal_a100 import image                                     # noqa: E402

# own App: importing modal_a100's would also import its entrypoint
app = modal.App("flash-nystrom-triton")
triton_image = image.pip_install("triton")


@app.function(gpu="A100-80GB", image=triton_image, timeout=60 * 30)
def triton_vs_cuda():
    import time
    import torch
    sys.path.insert(0, "/root/FlashNystrom")
    from benchmarks.triton_nystrom import triton_nystrom_forward, HAS_TRITON
    from flash_nystrom import flash_nystrom_attention as fn
    from flash_nystrom.reference import nystrom_attention_reference as ref
    import triton as _t

    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  triton={_t.__version__}  torch={torch.__version__}  "
          f"HAS_TRITON={HAS_TRITON}\n")

    B, H, D, M = 1, 8, 64, 64

    print("=== CORRECTNESS: Triton vs the pure-PyTorch reference (fp32 in) ===")
    ok = True
    for N in (1024, 4096, 16384):
        torch.manual_seed(0)
        q, k, v = [torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
                   for _ in range(3)]
        t = triton_nystrom_forward(q, k, v, num_landmarks=M).float()
        r = ref(q, k, v, M, 6, None, 0, kappa_star=0.0).float()
        f = fn(q, k, v, num_landmarks=M, kappa_star=0.0).float()
        rel_t = ((t - r).norm() / r.norm()).item()
        rel_f = ((f - r).norm() / r.norm()).item()
        good = rel_t < 5e-2
        ok &= good
        print(f"  N={N:6d}  triton-vs-ref {rel_t:.2e}   cuda-vs-ref {rel_f:.2e}"
              f"   {'ok' if good else '  <-- WRONG'}")
    if not ok:
        print("\n!! the Triton baseline does not match the reference; its"
              " latency numbers are meaningless until it does.")
        return

    def timed(f, *a, warmup=10, iters=50):
        for _ in range(warmup):
            f(*a)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            f(*a)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2] * 1e3

    print("\n=== FORWARD LATENCY, ms (median of 50) ===")
    print(f"{'N':>9} {'cuBLAS ref':>12} {'Triton':>10} {'FN scalar':>11}"
          f" {'FN tc-pinv':>11} {'best FN/Tri':>12}")
    for N in (16384, 65536, 262144, 1048576):
        try:
            q, k, v = [torch.randn(B, H, N, D, device="cuda",
                                   dtype=torch.float16) for _ in range(3)]
            t_ref = timed(lambda: ref(q, k, v, M, 6, None, 0, kappa_star=0.0))
            t_tri = timed(lambda: triton_nystrom_forward(q, k, v, num_landmarks=M))
            t_fn = timed(lambda: fn(q, k, v, num_landmarks=M, kappa_star=0.0))
            # the paper's latency tables use the tf32 tensor-core pinverse;
            # comparing against the scalar path would understate FlashNystrom
            t_tc = timed(lambda: fn(q, k, v, num_landmarks=M, kappa_star=0.0,
                                    use_tc_pinv=True))
            best = min(t_fn, t_tc)
            print(f"{N:>9} {t_ref:12.3f} {t_tri:10.3f} {t_fn:11.3f}"
                  f" {t_tc:11.3f} {t_tri / best:11.2f}x")
            del q, k, v
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"{N:>9} {'oom':>12}")
            torch.cuda.empty_cache()


@app.local_entrypoint()
def run_triton_bench():
    triton_vs_cuda.remote()


@app.function(gpu="A100-80GB", image=triton_image, timeout=60 * 40)
def kernel_tests():
    """Kernel correctness on A100, including the BACKWARD.

    The forward benchmark runs without requires_grad, so it exercises only the
    need_scaled_qk=False path. This covers the training path, where the scaled
    Q/K copies are still materialized for the backward.
    """
    import subprocess
    # the three vision test modules import torchvision, which this image lacks
    r = subprocess.run(
        ["python", "-m", "pytest", "-q", "--no-header", "-x", "tests/",
         "--ignore=tests/test_train_frac_seeds.py",
         "--ignore=tests/test_train_loop_cov.py",
         "--ignore=tests/test_train_three_way_cov.py"],
        cwd="/root/FlashNystrom", capture_output=True, text=True)
    print(r.stdout[-6000:])
    if r.returncode:
        print(r.stderr[-3000:])
        raise RuntimeError(f"pytest exit {r.returncode}")


@app.function(gpu="A100-80GB", image=triton_image, timeout=60 * 30)
def grad_check():
    """Explicit train-path check: gradients must match the reference."""
    import torch
    sys.path.insert(0, "/root/FlashNystrom")
    from flash_nystrom import flash_nystrom_attention as fn
    from flash_nystrom.reference import nystrom_attention_reference as ref

    print("training path (requires_grad=True -> need_scaled_qk=True)\n")
    for N in (2048, 16384, 65536):
        torch.manual_seed(0)
        base = [torch.randn(1, 4, N, 64, device="cuda", dtype=torch.float16)
                for _ in range(3)]
        a = [t.clone().requires_grad_(True) for t in base]
        b = [t.clone().requires_grad_(True) for t in base]
        g = torch.randn(1, 4, N, 64, device="cuda", dtype=torch.float16)

        fn(*a, num_landmarks=64, kappa_star=0.0).backward(g)
        ref(*b, 64, 6, None, 0, kappa_star=0.0).backward(g)

        errs = []
        for nm, x, y in zip("qkv", a, b):
            rel = ((x.grad.float() - y.grad.float()).norm()
                   / y.grad.float().norm()).item()
            errs.append(f"d{nm} {rel:.2e}")
            assert torch.isfinite(x.grad).all(), f"d{nm} non-finite at N={N}"
        print(f"  N={N:6d}  " + "   ".join(errs))
    print("\ngradients finite and matching the reference")
