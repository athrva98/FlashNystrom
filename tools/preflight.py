# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Measure what the sweep will cost on THIS GPU, before committing to it.

    python tools/preflight.py                  # ~5 min, the whole projection
    python tools/preflight.py --preset minimal

For every (tier, arm) in the real plan this times an actual forward+backward at
the batch size the sweep uses, then multiplies by the exact step count from the
job list. The output is a projected wall-clock built from measurements on the
hardware you are about to rent, not from a FLOPs model and a spec-sheet ratio
between two cards.

Also reports peak memory per configuration, so an OOM shows up here in minutes
rather than twenty hours into a run.

Run this first. If the projection does not fit your budget, change the preset
before spending the money, not after.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS = ["sdpa", "linear_attention", "linformer", "sliding_window",
        "nystrom_reference", "flash_nystrom", "flash_nystrom_tc"]

# (label, N, dim, heads, depth, batch, epochs, samples_per_epoch, n_jobs)
# mirrors build_jobs; n_jobs counts seeds x LRs x datasets for that tier.
def plan(preset):
    p12 = preset in ("paper12", "minimal")
    ep = 30 if p12 else 50
    tiers = [
        ("vision cifar10 N=65",   65,    256, 4, 4, 128, 20, 50000, 3),
        ("vision stl10 N=2305",   2305,  256, 4, 4, 128, ep,  5000, 3),
        ("vision stl10 N=9217",   9217,  256, 4, 4,  48, ep,  5000, 3),
        ("vision stl10 N=32401",  32401, 256, 4, 4,  16, ep,  2500, 1 if p12 else 3),
        ("mqar N=512 d=512",      512,   512, 8, 2, 128, 45, 100000, 3 if p12 else 4),
        ("genomics N=1024",       1024,  128, 2, 2,  32, 20, 32768, 12),
        ("genomics N=32768",      32768, 128, 2, 2,   4, 10,  8192, 3 if p12 else 12),
        ("genomics GB N=4707",    4707,  128, 2, 2,  32, 40,   847, 12),
        ("genomics repeat N=2048", 2048, 128, 2, 2,  32, 40, 32768, 3),
    ]
    if preset == "minimal":
        tiers = [t for t in tiers if "32401" not in t[0]]
    return tiers


def time_step(arm, n, dim, heads, depth, batch, iters=3):
    """Median seconds for one real training step, plus peak GiB."""
    from paper.mqar.model import build_attention
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    blocks = [build_attention(arm, dim, heads, seq_len=n, num_landmarks=64,
                              kappa_star=0.0).cuda() for _ in range(depth)]
    params = [p for b in blocks for p in b.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-4)
    x = torch.randn(batch, n, dim, device="cuda")

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            h = x
            for b in blocks:
                h = h + b(h)
            loss = h.float().pow(2).mean() * 65536.0
        loss.backward()
        opt.step()

    step()                                   # warmup: allocator + autotune
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.time()
        step()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    del blocks, params, opt, x
    torch.cuda.empty_cache()
    ts.sort()
    return ts[len(ts) // 2], peak


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="paper12",
                    choices=["full", "paper12", "minimal"])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--iters", type=int, default=3)
    a = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("needs a GPU")
        return 1
    p = torch.cuda.get_device_properties(0)
    total_gb = p.total_memory / 2 ** 30
    print(f"{p.name}  {total_gb:.0f} GiB  sm_{p.major}{p.minor}  torch {torch.__version__}")
    print(f"preset {a.preset}, {len(a.arms)} arms, median of {a.iters} steps\n")

    grand, spill, oom = 0.0, [], []
    print(f"{'tier':24s} {'steps':>7s} {'s/step':>8s} {'peak GiB':>9s} {'hours':>8s}")
    print("-" * 62)
    for label, n, dim, heads, depth, batch, epochs, samples, njobs in plan(a.preset):
        steps = epochs * math.ceil(samples / batch) * njobs
        tier_s, tier_peak = 0.0, 0.0
        for arm in a.arms:
            try:
                sec, peak = time_step(arm, n, dim, heads, depth, batch, a.iters)
                tier_s += sec
                tier_peak = max(tier_peak, peak)
                # past ~85% the allocator starts spilling into host RAM. Measured:
                # 7.3 GiB on an 8 GiB card made FlashNystrom 61x SLOWER than
                # exact attention, inverting the real ordering entirely.
                if peak > total_gb * 0.85:
                    spill.append((label, arm, peak))
            except torch.cuda.OutOfMemoryError:
                oom.append((label, arm))
                torch.cuda.empty_cache()
            except Exception as e:
                oom.append((label, f"{arm} ({type(e).__name__})"))
                torch.cuda.empty_cache()
        hours = steps * tier_s / 3600
        grand += hours
        print(f"{label:24s} {steps:7d} {tier_s:8.3f} {tier_peak:9.1f} {hours:8.2f}",
              flush=True)

    print("-" * 62)
    print(f"{'PROJECTED TOTAL':24s} {'':7s} {'':8s} {'':9s} {grand:8.2f} h")
    print(f"\n  s/step is summed over all {len(a.arms)} arms, so hours is the whole tier.")
    if oom:
        print(f"\n  {len(oom)} configuration(s) failed:")
        for label, arm in oom:
            print(f"    {label:24s} {arm}")
    if spill:
        print(f"\n  !! {len(spill)} configuration(s) allocated past device memory and")
        print("     spilled to host RAM. They ran, but those timings are far slower")
        print("     than the same work on a card that holds them, so the projection")
        print("     above is an OVERESTIMATE. Re-run on the target GPU.")
        for label, arm, peak in spill[:6]:
            print(f"    {label:24s} {arm:20s} {peak:.1f} GiB > {total_gb:.0f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
