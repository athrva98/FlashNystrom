# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Sweep MQAR over backend x heads x init x seed, with a per-config LR sweep.

MQAR's phase transition is sharp and very LR-sensitive, and the optimal LR
shifts with model width, so a single fixed LR gives flaky, width-confounded
numbers. Following the Zoology protocol, for each (backend, heads, init, seed)
we sweep LR (--lrs) and keep the BEST run, then aggregate mean +/- std over
seeds. The reported number is therefore each config's capability at its best
LR, not LR luck.

head_dim is held fixed (dim = heads * head_dim) because flash_nystrom requires
head_dim in {64, 128}; heads and width scale together.

Each run is an isolated subprocess (clean GPU state). Example:
    python -m paper.mqar.sweep --backends sdpa flash_nystrom nystrom_reference \\
        --heads 2 4 8 --inits normal orthogonal --seeds 0 1 2 \\
        --lrs 1e-4 3e-4 1e-3 3e-3

Any unrecognized flags (e.g. --epochs, --num_landmarks) pass through to train.
"""
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from itertools import product

_BEST = re.compile(r"best test recall:\s*([\d.]+)%")


def run_one(backend, heads, dim, init, seed, lr, passthrough):
    cmd = [
        sys.executable, "-m", "paper.mqar.train",
        "--backend", backend, "--heads", str(heads), "--dim", str(dim),
        "--init", init, "--seed", str(seed), "--lr", str(lr),
    ] + passthrough
    proc = subprocess.run(cmd, capture_output=True, text=True)
    m = _BEST.search(proc.stdout)
    if m is None:
        print(f"    !! FAILED lr={lr}: {' '.join(cmd)}")
        print("    " + (proc.stdout[-300:] or proc.stderr[-300:]).replace("\n", "\n    "))
        return None
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser(description="MQAR sweep with per-config LR tuning")
    ap.add_argument("--backends", nargs="+",
                    default=["sdpa", "flash_nystrom", "nystrom_reference"])
    ap.add_argument("--heads", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--inits", nargs="+", default=["normal"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 3.16e-3, 1e-2, 3.16e-2],
                    help="per-config LR sweep; best run is kept (Zoology's np.logspace(-3, -1.5, 4))")
    ap.add_argument("--head_dim", type=int, default=64,
                    help="fixed per-head dimension; dim = heads * head_dim")
    args, passthrough = ap.parse_known_args()

    # results[(backend, heads, init)] = list of (best_recall, best_lr) per seed
    results: dict[tuple, list[tuple[float, float]]] = {}
    for backend, heads, init in product(args.backends, args.heads, args.inits):
        dim = heads * args.head_dim
        per_seed = []
        for seed in args.seeds:
            best_acc, best_lr = -1.0, None
            for lr in args.lrs:
                acc = run_one(backend, heads, dim, init, seed, lr, passthrough)
                mark = ""
                if acc is not None and acc > best_acc:
                    best_acc, best_lr, mark = acc, lr, "  <- best"
                print(f"  {backend:18s} heads={heads:<2d} dim={dim:<4d} init={init:<10s} "
                      f"seed={seed} lr={lr:<7g}: {'FAIL' if acc is None else f'{acc:5.2f}%'}{mark}")
            if best_acc >= 0:
                per_seed.append((best_acc, best_lr))
        results[(backend, heads, init)] = per_seed

    print("\n=== summary: best-LR test recall, mean +/- std over seeds ===")
    print(f"{'backend':18s} {'heads':>5s} {'dim':>5s} {'init':>11s} {'recall (%)':>20s} {'best LRs':>22s}")
    for (backend, heads, init), per_seed in results.items():
        dim = heads * args.head_dim
        if not per_seed:
            cell, lrs = "n/a", ""
        else:
            accs = [a for a, _ in per_seed]
            mu = statistics.mean(accs)
            sd = statistics.pstdev(accs) if len(accs) > 1 else 0.0
            cell = f"{mu:6.2f} +/- {sd:5.2f} (n={len(accs)})"
            lrs = ",".join(f"{lr:g}" for _, lr in per_seed)
        print(f"{backend:18s} {heads:5d} {dim:5d} {init:>11s} {cell:>20s} {lrs:>22s}")


if __name__ == "__main__":
    main()
