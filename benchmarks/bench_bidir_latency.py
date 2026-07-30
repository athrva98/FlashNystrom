# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Regenerate the paper's bidirectional-operator latency table on this GPU.

    python benchmarks/bench_bidir_latency.py
    python benchmarks/bench_bidir_latency.py --lens 262144 1048576 --out tab.txt

Every arm is timed as the COMPLETE operator, forward plus backward. That
distinction is the point: flash_bla's kernel implements only the unnormalized
core, so timing the bare call omits two feature-map passes, the normalizer
reduction and the division, and flatters linear attention against arms that are
timed end to end. An earlier version of this measurement made exactly that
mistake, which is why the numbers it produced are not comparable to these.

Requires flash_attn (sliding window) and flash_bla (fused linear attention).
Arms whose kernel is missing are reported as such rather than silently
substituted with an unfused stand-in, so a row can never mix the two.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.baseline_ops import (                          # noqa: E402
    linear_attention_op, linformer_op, sdpa_op, sliding_window_op,
)


def timed(fn, *args, warmup=5, iters=20):
    """Median fwd+bwd milliseconds. Median, not mean: one stray context switch
    should not set the number that goes in a table."""
    for t in args:
        if torch.is_tensor(t) and t.requires_grad and t.grad is not None:
            t.grad = None
    try:
        for _ in range(warmup):
            fn(*args).sum().backward()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            for t in args:
                if torch.is_tensor(t) and t.requires_grad:
                    t.grad = None
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record()
            fn(*args).sum().backward()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        return times[len(times) // 2]
    except torch.cuda.OutOfMemoryError:
        return "OOM"
    except Exception as ex:
        return f"ERR:{type(ex).__name__}"


def build_arms(m, r, window, dev, dt):
    from flash_nystrom import flash_nystrom_attention as fn

    arms = [
        ("FlashNystrom", lambda q, k, v: fn(q, k, v, num_landmarks=m,
                                            kappa_star=0.0)),
        ("linear attn (torch)", linear_attention_op),
    ]
    try:
        from benchmarks.baseline_ops import linear_attention_fused_op
        linear_attention_fused_op(
            *[torch.randn(1, 1, 128, 64, device=dev, dtype=dt) for _ in range(3)])
        arms.append(("linear attn (fused)", linear_attention_fused_op))
    except Exception as e:
        arms.append(("linear attn (fused)", None))
        print(f"  !! fused linear attention unavailable ({type(e).__name__}); "
              f"that column is NOT a fair baseline and reports n/a\n")
    arms.append(("Linformer", None))          # needs per-N projections
    for w in (64, 256):
        try:
            import flash_attn                                   # noqa: F401
            arms.append((f"sliding window w={w}",
                         lambda q, k, v, _w=w: sliding_window_op(q, k, v, _w)))
        except ImportError:
            arms.append((f"sliding window w={w}", None))
    arms.append(("exact attention", sdpa_op))
    return arms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lens", nargs="+", type=int,
                    default=[131072, 262144, 524288, 1048576])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head_dim", type=int, default=64)
    ap.add_argument("--landmarks", type=int, default=64)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--exact_max_n", type=int, default=262144,
                    help="skip exact attention past this; it is already ~200x slower")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("needs a GPU"); return 1
    dev, dt = "cuda", torch.float16
    B, H, D = a.batch, a.heads, a.head_dim
    print(f"{torch.cuda.get_device_name(0)}  fp16  B={B} H={H} D={D} "
          f"m=r={a.landmarks}\nfwd+bwd ms, median of 20\n")

    arms = build_arms(a.landmarks, a.rank, 64, dev, dt)
    names = [n for n, _ in arms]
    rows = []
    print(f"{'N':>9} | " + " | ".join(f"{n:>19}" for n in names))
    for N in a.lens:
        q, k, v = [torch.randn(B, H, N, D, device=dev, dtype=dt,
                               requires_grad=True) for _ in range(3)]
        cells = []
        for name, fn_ in arms:
            if fn_ is None and "Linformer" not in name:
                cells.append("n/a")
            elif "Linformer" in name:
                # leaf tensors: scaling a requires_grad tensor makes a non-leaf,
                # whose .grad is never populated and whose backward then errors
                E = (torch.randn(a.rank, N, device=dev, dtype=dt)
                     * N ** -0.5).requires_grad_(True)
                Fp = (torch.randn(a.rank, N, device=dev, dtype=dt)
                      * N ** -0.5).requires_grad_(True)
                cells.append(timed(lambda x, y, z: linformer_op(x, y, z, E, Fp),
                                   q, k, v))
            elif "exact" in name and N > a.exact_max_n:
                cells.append("---")
            else:
                cells.append(timed(fn_, q, k, v))
        fmt = lambda c: f"{c:19.1f}" if isinstance(c, float) else f"{str(c):>19}"
        print(f"{N:>9} | " + " | ".join(fmt(c) for c in cells), flush=True)
        rows.append((N, cells))
        del q, k, v
        torch.cuda.empty_cache()

    if a.out:
        with open(a.out, "w") as fh:
            fh.write("N," + ",".join(names) + "\n")
            for N, cells in rows:
                fh.write(f"{N}," + ",".join(
                    f"{c:.1f}" if isinstance(c, float) else str(c)
                    for c in cells) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
