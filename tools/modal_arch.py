# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""FlashNystrom vs hand-written Triton across GPU generations.

    modal run tools/modal_arch.py::a100
    modal run tools/modal_arch.py::h100
    modal run tools/modal_arch.py::b200

What this measures is the cost of the single-binary SM80 contract. On an A100
the contract is free: sm_80 is native, and both sides compile to the same ISA.
On Hopper and Blackwell it may not be. Our kernels are written with SM80 idioms
(cp.async, SM80 MMA atoms) so that ONE binary serves sm_80/90a/100, while Triton
JIT-compiles for the actual target and is free to emit WGMMA, TMA, and whatever
else the architecture offers.

So: if the FlashNystrom-to-Triton ratio degrades as the architecture advances,
that degradation IS the price of the contract, and it is measured rather than
argued about. The Triton PTX is scanned for arch-specific instructions on each
target, which says directly what Triton is using that we are not.

Forward only; see benchmarks/triton_nystrom.py.
"""
import pathlib
import sys

import modal

for _p in (str(pathlib.Path(__file__).resolve().parent), "/root/FlashNystrom/tools"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from modal_a100 import image                                     # noqa: E402

app = modal.App("flash-nystrom-arch")
arch_image = image.pip_install("triton")


def _compare(lens):
    """Correctness, then forward latency, then what Triton's PTX actually uses."""
    import os
    import re
    import time
    os.environ["TRITON_CACHE_DIR"] = "/tmp/tcache"
    import glob
    import torch
    sys.path.insert(0, "/root/FlashNystrom")
    from benchmarks.triton_nystrom import triton_nystrom_forward
    from flash_nystrom import flash_nystrom_attention as fn
    from flash_nystrom.reference import nystrom_attention_reference as ref
    import triton as _t

    p = torch.cuda.get_device_properties(0)
    cc = torch.cuda.get_device_capability()
    print(f"{p.name}   sm_{cc[0]}{cc[1]}   triton {_t.__version__}   "
          f"torch {torch.__version__}\n")

    B, H, D, M = 1, 8, 64, 64

    print("=== correctness vs the pure-PyTorch reference ===")
    ok = True
    for N in (4096, 16384):
        torch.manual_seed(0)
        q, k, v = [torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
                   for _ in range(3)]
        r = ref(q, k, v, M, 6, None, 0, kappa_star=0.0).float()
        t = triton_nystrom_forward(q, k, v, num_landmarks=M).float()
        f = fn(q, k, v, num_landmarks=M, kappa_star=0.0).float()
        rt = ((t - r).norm() / r.norm()).item()
        rf = ((f - r).norm() / r.norm()).item()
        ok &= rt < 5e-2 and rf < 5e-2
        print(f"  N={N:6d}  triton {rt:.2e}   flashnystrom {rf:.2e}")
    if not ok:
        print("!! correctness failed on this architecture; timings withheld")
        return

    def timed(f, warmup=10, iters=50):
        for _ in range(warmup):
            f()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            f()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2] * 1e3

    print("\n=== forward latency, ms (median of 50) ===")
    print(f"{'N':>9} {'cuBLAS ref':>11} {'Triton':>9} {'FlashNystrom':>13}"
          f" {'FN/Triton':>10}")
    for N in lens:
        try:
            q, k, v = [torch.randn(B, H, N, D, device="cuda",
                                   dtype=torch.float16) for _ in range(3)]
            t_ref = timed(lambda: ref(q, k, v, M, 6, None, 0, kappa_star=0.0))
            t_tri = timed(lambda: triton_nystrom_forward(q, k, v, num_landmarks=M))
            t_fn = timed(lambda: fn(q, k, v, num_landmarks=M, kappa_star=0.0,
                                    use_tc_pinv=True))
            print(f"{N:>9} {t_ref:11.3f} {t_tri:9.3f} {t_fn:13.3f}"
                  f" {t_tri / t_fn:9.2f}x")
            del q, k, v
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"{N:>9} {'oom':>11}")
            torch.cuda.empty_cache()

    # What is Triton emitting here that our SM80-idiom kernels cannot?
    print("\n=== Triton PTX: architecture-specific instructions ===")
    ARCH = {"wgmma": "Hopper async warpgroup MMA (sm_90+)",
            "cp.async.bulk": "TMA bulk copy (sm_90+)",
            "tcgen05": "Blackwell 5th-gen tensor core (sm_100+)",
            "mma.sync": "classic synchronous MMA (sm_70+)",
            "cp.async": "Ampere async copy (sm_80+)",
            "ldmatrix": "shared-memory matrix load"}
    for pf in sorted(glob.glob("/tmp/tcache/**/*.ptx", recursive=True)):
        ptx = open(pf).read()
        m = re.search(r"\.target\s+(\S+)", ptx)
        nm = re.search(r"\.visible \.entry (\w+)", ptx)
        if not nm:
            continue
        hits = {k: len(re.findall(re.escape(k), ptx)) for k in ARCH}
        hits = {k: c for k, c in hits.items() if c}
        print(f"  {nm.group(1):24s} target={m.group(1) if m else '?':10s} "
              + "  ".join(f"{k}x{c}" for k, c in hits.items()))
        shapes = set(re.findall(r"mma\.sync\.aligned\.(m\d+n\d+k\d+)", ptx))
        if shapes:
            print(f"    {'':24s} mma shapes: {sorted(shapes)}")


@app.function(gpu="A100-80GB", image=arch_image, timeout=60 * 40)
def _a100():
    _compare([16384, 65536, 262144, 1048576])


@app.function(gpu="H100", image=arch_image, timeout=60 * 40)
def _h100():
    _compare([16384, 65536, 262144, 1048576])


@app.function(gpu="B200", image=arch_image, timeout=60 * 40)
def _b200():
    _compare([16384, 65536, 262144, 1048576])


@app.local_entrypoint()
def a100():
    _a100.remote()


@app.local_entrypoint()
def h100():
    _h100.remote()


@app.local_entrypoint()
def b200():
    _b200.remote()


def _profile(lens):
    """Per-kernel forward time. At short N the N-independent kernels dominate,
    which is a different bottleneck from the one visible at N=1M."""
    import torch
    from torch.profiler import profile, ProfilerActivity
    sys.path.insert(0, "/root/FlashNystrom")
    from flash_nystrom import flash_nystrom_attention as fn

    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  sm_{torch.cuda.get_device_capability()[0]}"
          f"{torch.cuda.get_device_capability()[1]}\n")
    B, H, D, M = 1, 8, 64, 64
    for N in lens:
        q, k, v = [torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
                   for _ in range(3)]
        f = lambda: fn(q, k, v, num_landmarks=M, kappa_star=0.0, use_tc_pinv=True)
        for _ in range(10):
            f()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                f()
            torch.cuda.synchronize()
        evs = [e for e in prof.key_averages() if e.self_device_time_total > 0]
        evs.sort(key=lambda e: -e.self_device_time_total)
        tot = sum(e.self_device_time_total for e in evs)
        print(f"  N={N}   total {tot/20/1000:.3f} ms")
        for e in evs[:7]:
            nm = e.key.replace("void flash_nystrom::", "")[:44]
            print(f"    {nm:46s} {e.self_device_time_total/20/1000:7.3f} ms"
                  f" {100*e.self_device_time_total/tot:6.1f}%")
        print()
        del q, k, v
        torch.cuda.empty_cache()


@app.function(gpu="B200", image=arch_image, timeout=60 * 30)
def _prof_b200():
    _profile([16384, 65536, 1048576])


@app.function(gpu="A100-80GB", image=arch_image, timeout=60 * 30)
def _prof_a100():
    _profile([16384, 65536, 1048576])


@app.local_entrypoint()
def prof_b200():
    _prof_b200.remote()


@app.local_entrypoint()
def prof_a100():
    _prof_a100.remote()
