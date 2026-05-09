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


def bench_one(N, B=1, H=4, D=64, m=32, niter=6, dtype=torch.float16):
    q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    dout = torch.randn(B, H, N, D, dtype=dtype, device="cuda")

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

    fwd_fn_t = benchmark_cuda(fwd_fn)
    fwd_ref_t = benchmark_cuda(fwd_ref)
    fwd_sdpa_t = benchmark_cuda(fwd_sdpa)

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

    fb_fn_t = benchmark_cuda(fwdbwd_fn, warmup=5, repeat=30)
    fb_ref_t = benchmark_cuda(fwdbwd_ref, warmup=5, repeat=30)
    fb_sdpa_t = benchmark_cuda(fwdbwd_sdpa, warmup=5, repeat=30)

    bwd_fn_t = fb_fn_t - fwd_fn_t
    bwd_ref_t = fb_ref_t - fwd_ref_t
    bwd_sdpa_t = fb_sdpa_t - fwd_sdpa_t

    return {
        "N": N,
        "fwd_fn_ms":   fwd_fn_t,   "bwd_fn_ms":   bwd_fn_t,   "tot_fn_ms":   fb_fn_t,
        "fwd_ref_ms":  fwd_ref_t,  "bwd_ref_ms":  bwd_ref_t,  "tot_ref_ms":  fb_ref_t,
        "fwd_sdpa_ms": fwd_sdpa_t, "bwd_sdpa_ms": bwd_sdpa_t, "tot_sdpa_ms": fb_sdpa_t,
    }


def main():
    print("Latency: FlashNystrom vs Nystrom-Ref vs SDPA")
    print("Config: B=1, H=4, D=64, m=32, niter=6, FP16")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

    print(f"{'N':>6} | {'FN fwd':>8} {'FN bwd':>8} {'FN tot':>8} | "
          f"{'Ref fwd':>8} {'Ref bwd':>8} {'Ref tot':>8} | "
          f"{'SDPA fwd':>9} {'SDPA bwd':>9} {'SDPA tot':>9} | "
          f"{'FN/Ref':>7} {'FN/SDPA':>8}")
    print("-" * 130)
    for N in seq_lengths:
        try:
            r = bench_one(N)
        except RuntimeError as e:
            print(f"{N:>6} | OOM or error: {e}")
            continue
        fn_ref_ratio = r["tot_fn_ms"] / r["tot_ref_ms"]
        fn_sdpa_ratio = r["tot_fn_ms"] / r["tot_sdpa_ms"]
        print(f"{r['N']:>6} | "
              f"{r['fwd_fn_ms']:>8.2f} {r['bwd_fn_ms']:>8.2f} {r['tot_fn_ms']:>8.2f} | "
              f"{r['fwd_ref_ms']:>8.2f} {r['bwd_ref_ms']:>8.2f} {r['tot_ref_ms']:>8.2f} | "
              f"{r['fwd_sdpa_ms']:>9.2f} {r['bwd_sdpa_ms']:>9.2f} {r['tot_sdpa_ms']:>9.2f} | "
              f"{fn_ref_ratio:>6.2f}x {fn_sdpa_ratio:>7.2f}x")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
