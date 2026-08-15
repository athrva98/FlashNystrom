# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Peak GPU memory for every arm at every tier's REAL batch size.

    python tools/memcheck.py

The smoke test runs tiny batches, so it proves the code works but says nothing
about whether the sweep fits in this GPU's memory. This runs one real
forward+backward per (tier, arm) at the batch size the sweep actually uses, and
reports peak allocation against the device limit.

Minutes, not hours. Run it before committing a multi-day sweep to a card
smaller than the one the grid was costed for: an OOM twenty hours in costs far
more than this does.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS = ["sdpa", "linear_attention", "linformer", "sliding_window",
        "nystrom_reference", "flash_nystrom", "flash_nystrom_tc"]

# (label, N tokens, dim, heads, depth, batch) matching the real job configs
TIERS = [
    ("vision cifar10 N=65",    65,    256, 4, 4, 128),
    ("vision stl10  N=2305",   2305,  256, 4, 4, 128),
    ("vision stl10  N=9217",   9217,  256, 4, 4, 48),
    ("vision stl10  N=32401",  32401, 256, 4, 4, 16),
    ("genomics      N=1024",   1024,  128, 2, 2, 32),
    ("genomics      N=32768",  32768, 128, 2, 2, 4),
    ("mqar          N=512",    512,   512, 8, 2, 128),
]


def one_step(arm, n, dim, heads, depth, batch, landmarks=64):
    """One fwd+bwd through `depth` attention blocks at this configuration."""
    from paper.mqar.model import build_attention
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    blocks = [build_attention(arm, dim, heads, seq_len=n, num_landmarks=landmarks,
                              kappa_star=0.0).cuda() for _ in range(depth)]
    params = [p for b in blocks for p in b.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-4)
    x = torch.randn(batch, n, dim, device="cuda")
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        h = x
        for b in blocks:
            h = h + b(h)
        loss = h.float().pow(2).mean() * 65536.0
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    del blocks, params, opt, x, h, loss
    torch.cuda.empty_cache()
    return peak


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--headroom", type=float, default=0.80,
                    help="flag anything above this fraction of device memory")
    a = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("needs a GPU")
        return 1
    p = torch.cuda.get_device_properties(0)
    total = p.total_memory / 2 ** 30
    limit = total * a.headroom
    print(f"{p.name}  {total:.0f} GiB  sm_{p.major}{p.minor}")
    print(f"flagging anything above {limit:.1f} GiB "
          f"({a.headroom:.0%} of device)\n")
    print(f"{'tier':26s} {'batch':>6s} " + " ".join(f"{x[:9]:>9s}" for x in a.arms))

    worst, failures = 0.0, []
    for label, n, dim, heads, depth, batch in TIERS:
        cells = []
        for arm in a.arms:
            try:
                gb = one_step(arm, n, dim, heads, depth, batch)
                worst = max(worst, gb)
                cells.append(f"{gb:9.1f}" if gb < limit else f"{gb:8.1f}!")
                if gb >= limit:
                    failures.append((label, arm, f"{gb:.1f} GiB"))
            except torch.cuda.OutOfMemoryError:
                cells.append(f"{'OOM':>9s}")
                failures.append((label, arm, "OOM"))
                torch.cuda.empty_cache()
            except Exception as e:
                cells.append(f"{type(e).__name__[:9]:>9s}")
                failures.append((label, arm, type(e).__name__))
                torch.cuda.empty_cache()
        print(f"{label:26s} {batch:6d} " + " ".join(cells), flush=True)

    print(f"\npeak across the whole grid: {worst:.1f} GiB of {total:.0f} GiB")
    if failures:
        print(f"\n{len(failures)} configuration(s) at or over the limit:")
        for label, arm, why in failures:
            print(f"  {label:26s} {arm:20s} {why}")
        print("\nReduce the batch for those tiers before starting the sweep.")
        return 1
    print("every configuration fits with headroom")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
