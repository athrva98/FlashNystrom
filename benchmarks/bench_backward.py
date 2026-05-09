# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

"""Benchmarks for FlashNystrom backward pass."""

import torch


def benchmark_fn(fn, warmup=10, repeat=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    trim = max(1, len(times) // 5)
    trimmed = times[trim:-trim] if trim < len(times) // 2 else times
    return {
        "median_ms": times[len(times) // 2],
        "min_ms": times[0],
        "mean_ms": sum(trimmed) / len(trimmed),
    }


def bench_nystrom_bwd(B, H, N, D, m, dtype):
    try:
        from flash_nystrom.flash_nystrom import FlashNystromFunction
    except ImportError:
        return None, None

    q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)

    # benchmark forward
    def fwd():
        return FlashNystromFunction.apply(q, k, v, None, m, 6, 0)

    fwd_result = benchmark_fn(fwd)

    # benchmark forward+backward together
    def fwd_bwd():
        out = FlashNystromFunction.apply(
            q.detach().requires_grad_(True),
            k.detach().requires_grad_(True),
            v.detach().requires_grad_(True),
            None, m, 6, 0
        )
        out.sum().backward()

    bwd_result = benchmark_fn(fwd_bwd, warmup=5, repeat=20)
    return fwd_result, bwd_result


def bench_sdpa_bwd(B, H, N, D, dtype):
    q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)

    def fwd():
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    fwd_result = benchmark_fn(fwd)

    def fwd_bwd():
        qq = q.detach().requires_grad_(True)
        kk = k.detach().requires_grad_(True)
        vv = v.detach().requires_grad_(True)
        out = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv)
        out.sum().backward()

    bwd_result = benchmark_fn(fwd_bwd, warmup=5, repeat=20)
    return fwd_result, bwd_result


def bench_ref_bwd(B, H, N, D, m, dtype):
    from flash_nystrom.reference import nystrom_attention_reference_simple

    q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)

    def fwd_bwd():
        qq = q.detach().requires_grad_(True)
        kk = k.detach().requires_grad_(True)
        vv = v.detach().requires_grad_(True)
        out = nystrom_attention_reference_simple(qq, kk, vv, m)
        out.sum().backward()

    result = benchmark_fn(fwd_bwd, warmup=3, repeat=10)
    return result


def fmt(result, key="median_ms"):
    return f"{result[key]:.2f}" if result else "N/A"


def main():
    B, H, D, m = 1, 8, 128, 64
    dtype = torch.float16

    print(f"{'=' * 100}")
    print(f"FlashNystrom Backward Benchmark: B={B} H={H} D={D} m={m} dtype=fp16")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 100}")
    print(f"{'N':>8} | {'FN fwd':>8} | {'FN fwd+bwd':>10} | {'FN bwd':>8} | {'Ref fwd+bwd':>11} | {'SDPA fwd+bwd':>12} | {'vs SDPA':>8}")
    print(f"{'-' * 100}")

    for N in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        fn_fwd, fn_fb = bench_nystrom_bwd(B, H, N, D, m, dtype)

        try:
            sdpa_fwd, sdpa_fb = bench_sdpa_bwd(B, H, N, D, dtype)
        except Exception:
            sdpa_fwd, sdpa_fb = None, None

        try:
            ref_fb = bench_ref_bwd(B, H, N, D, m, dtype)
        except Exception:
            ref_fb = None

        fn_bwd_ms = ""
        if fn_fwd and fn_fb:
            bwd_only = fn_fb["median_ms"] - fn_fwd["median_ms"]
            fn_bwd_ms = f"{bwd_only:.2f}"

        speedup = ""
        if fn_fb and sdpa_fb:
            speedup = f"{sdpa_fb['median_ms'] / fn_fb['median_ms']:.1f}x"

        print(f"{N:>8} | {fmt(fn_fwd):>8} | {fmt(fn_fb):>10} | {fn_bwd_ms:>8} | {fmt(ref_fb):>11} | {fmt(sdpa_fb):>12} | {speedup:>8}")


if __name__ == "__main__":
    main()
