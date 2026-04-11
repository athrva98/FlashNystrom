# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

"""Benchmarks for FlashNystrom forward pass vs FlashAttention and PyTorch SDPA."""

import argparse
import torch


def benchmark_fn(fn, warmup=10, repeat=50):
    """CUDA-accurate benchmarking with events."""
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
    return {
        "median_ms": times[len(times) // 2],
        "min_ms": times[0],
        "mean_ms": sum(times) / len(times),
    }


def bench_nystrom_cuda(B, H, N, D, m, dtype):
    try:
        import flash_nystrom._C as _C
    except ImportError:
        return None
    q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    # Warmup
    _C.forward(q, k, v, m, 6, 0, None)
    return benchmark_fn(lambda: _C.forward(q, k, v, m, 6, 0, None))


def bench_nystrom_ref(B, H, N, D, m, dtype):
    from flash_nystrom.reference import nystrom_attention_reference_simple

    q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    return benchmark_fn(
        lambda: nystrom_attention_reference_simple(q, k, v, m), warmup=3, repeat=10
    )


def bench_sdpa(B, H, N, D, dtype):
    q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    return benchmark_fn(
        lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
    )


def bench_flash_attn(B, H, N, D, dtype):
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        return None
    # flash_attn expects (B, N, H, D) layout
    q = torch.randn(B, N, H, D, dtype=dtype, device="cuda")
    k = torch.randn(B, N, H, D, dtype=dtype, device="cuda")
    v = torch.randn(B, N, H, D, dtype=dtype, device="cuda")
    return benchmark_fn(lambda: flash_attn_func(q, k, v))


def fmt(result, key="median_ms"):
    return f"{result[key]:.2f}" if result else "N/A"


def main():
    parser = argparse.ArgumentParser(description="FlashNystrom benchmarks")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--landmarks", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    B, H, D, m = args.batch, args.heads, args.head_dim, args.landmarks

    print(f"{'=' * 90}")
    print(f"FlashNystrom Benchmark: B={B} H={H} D={D} m={m} dtype={args.dtype}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 90}")
    print(
        f"{'N':>8} | {'Nystrom CUDA':>12} | {'Nystrom Ref':>12} | {'SDPA':>12} | {'FlashAttn':>12} | {'vs SDPA':>8}"
    )
    print(f"{'-' * 90}")

    for N in [512, 1024, 2048, 4096, 8192]:
        nystrom = bench_nystrom_cuda(B, H, N, D, m, dtype)
        ref = bench_nystrom_ref(B, H, N, D, m, dtype)
        sdpa = bench_sdpa(B, H, N, D, dtype)
        fa = bench_flash_attn(B, H, N, D, dtype)

        speedup = ""
        if nystrom and sdpa:
            speedup = f"{sdpa['median_ms'] / nystrom['median_ms']:.1f}x"

        print(
            f"{N:>8} | {fmt(nystrom):>12} | {fmt(ref):>12} | {fmt(sdpa):>12} | {fmt(fa):>12} | {speedup:>8}"
        )


if __name__ == "__main__":
    main()
