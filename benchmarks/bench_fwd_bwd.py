# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

"""Forward + backward latency comparison: FlashNystrom vs Nystrom-Ref vs SDPA.

CUDA-event timing, 10 warmup + 50 timed runs, median reported.
Config: B=1, H=4, D=64 (CIFAR-10 ViT setup), FP16, newton_iter=6, num_landmarks=32.
"""
import sys
sys.path.insert(0, "C:/Users/athrv/Documents/FlashNystrom/benchmarks")
import torch
import torch.nn.functional as F
import flash_nystrom._C as _C
from flash_nystrom import flash_nystrom_attention
from flash_nystrom.reference import nystrom_attention_reference


def benchmark_cuda(fn, warmup=10, repeat=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in events:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in events)
    return times[len(times) // 2]   # median


def _bench_safe(label, fn, warmup, repeat):
    """Run benchmark_cuda(fn, ...), returning float or 'OOM' on CUDA OOM.

    Each measurement is wrapped independently so that a method running out
    of memory at large N does not abort the whole row. After an OOM we
    empty_cache and reset_peak_memory_stats to give later measurements a
    clean baseline.
    """
    try:
        return benchmark_cuda(fn, warmup=warmup, repeat=repeat)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return "OOM"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            return "OOM"
        raise


def bench_one(N, B=1, H=4, D=64, m=32, niter=6, dtype=torch.float16):
    try:
        q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
        k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
        v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
        dout = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        return {"N": N, "input_oom": True}

    # ---- forward ----
    def fwd_fn():
        with torch.no_grad():
            return flash_nystrom_attention(q, k, v, num_landmarks=m, newton_iter=niter)
    def fwd_ref():
        with torch.no_grad():
            return nystrom_attention_reference(q, k, v, num_landmarks=m, newton_iter=niter)
    def fwd_sdpa():
        with torch.no_grad():
            return F.scaled_dot_product_attention(q, k, v)

    # Reduce repeat count at large N to keep the wall-clock manageable.
    fwd_repeat = 50 if N <= 8192 else 20 if N <= 32768 else 10 if N <= 131072 else 5

    fwd_fn_t   = _bench_safe("FN fwd",   fwd_fn,   warmup=10, repeat=fwd_repeat)
    fwd_ref_t  = _bench_safe("Ref fwd",  fwd_ref,  warmup=10, repeat=fwd_repeat)
    fwd_sdpa_t = _bench_safe("SDPA fwd", fwd_sdpa, warmup=10, repeat=fwd_repeat)

    # ---- forward + backward ----
    def fwdbwd_fn():
        qq = q.detach().requires_grad_(True)
        kk = k.detach().requires_grad_(True)
        vv = v.detach().requires_grad_(True)
        out = flash_nystrom_attention(qq, kk, vv, num_landmarks=m, newton_iter=niter)
        out.backward(dout)
    def fwdbwd_ref():
        qq = q.detach().requires_grad_(True)
        kk = k.detach().requires_grad_(True)
        vv = v.detach().requires_grad_(True)
        out = nystrom_attention_reference(qq, kk, vv, num_landmarks=m, newton_iter=niter)
        out.backward(dout)
    def fwdbwd_sdpa():
        qq = q.detach().requires_grad_(True)
        kk = k.detach().requires_grad_(True)
        vv = v.detach().requires_grad_(True)
        out = F.scaled_dot_product_attention(qq, kk, vv)
        out.backward(dout)

    fb_repeat = 30 if N <= 8192 else 15 if N <= 32768 else 8 if N <= 131072 else 3

    fb_fn_t   = _bench_safe("FN tot",   fwdbwd_fn,   warmup=5, repeat=fb_repeat)
    fb_ref_t  = _bench_safe("Ref tot",  fwdbwd_ref,  warmup=5, repeat=fb_repeat)
    fb_sdpa_t = _bench_safe("SDPA tot", fwdbwd_sdpa, warmup=5, repeat=fb_repeat)

    def _diff(tot, fwd):
        if tot == "OOM" or fwd == "OOM":
            return "OOM"
        return tot - fwd

    return {
        "N": N,
        "input_oom": False,
        "fwd_fn_ms":   fwd_fn_t,   "bwd_fn_ms":   _diff(fb_fn_t,   fwd_fn_t),   "tot_fn_ms":   fb_fn_t,
        "fwd_ref_ms":  fwd_ref_t,  "bwd_ref_ms":  _diff(fb_ref_t,  fwd_ref_t),  "tot_ref_ms":  fb_ref_t,
        "fwd_sdpa_ms": fwd_sdpa_t, "bwd_sdpa_ms": _diff(fb_sdpa_t, fwd_sdpa_t), "tot_sdpa_ms": fb_sdpa_t,
    }


def _fmt(v, width, prec=2):
    if v == "OOM":
        return f"{'OOM':>{width}}"
    return f"{v:>{width}.{prec}f}"


def main():
    print("Latency: FlashNystrom vs Nystrom-Ref vs SDPA")
    print("Config: B=1, H=4, D=64, m=32, niter=6, FP16")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Stop at 262144 (256K). Neither method OOMs at this size on an 8 GB
    # consumer card; SDPA hits a practical wall via wall-clock long before
    # memory does (~20 seconds per fwd+bwd at N=262144 on a 5060). If you
    # want OOM-finding, extend this list and accept the time cost.
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192,
                    16384, 32768, 65536, 131072, 262144]

    print(f"{'N':>7} | {'FN fwd':>8} {'FN bwd':>8} {'FN tot':>8} | "
          f"{'Ref fwd':>8} {'Ref bwd':>8} {'Ref tot':>8} | "
          f"{'SDPA fwd':>9} {'SDPA bwd':>9} {'SDPA tot':>9}")
    print("-" * 122)
    for N in seq_lengths:
        r = bench_one(N)
        if r.get("input_oom"):
            print(f"{N:>7} | OOM allocating Q/K/V inputs; stopping.")
            break
        print(f"{r['N']:>7} | "
              f"{_fmt(r['fwd_fn_ms'],   8)} {_fmt(r['bwd_fn_ms'],   8)} {_fmt(r['tot_fn_ms'],   8)} | "
              f"{_fmt(r['fwd_ref_ms'],  8)} {_fmt(r['bwd_ref_ms'],  8)} {_fmt(r['tot_ref_ms'],  8)} | "
              f"{_fmt(r['fwd_sdpa_ms'], 9)} {_fmt(r['bwd_sdpa_ms'], 9)} {_fmt(r['tot_sdpa_ms'], 9)}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
