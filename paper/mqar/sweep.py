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
import statistics
from itertools import product


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
    ap.add_argument("--max_parallel", type=int, default=1,
                    help="concurrent runs on one GPU (recall-only, so parallel-safe). "
                         "Defaults to 1 -- raise it on an A100 (e.g. 6); keep it 1 on "
                         "a small local card. This sweep measures recall, not timing, "
                         "so concurrency does not corrupt anything.")
    args, passthrough = ap.parse_known_args()

    from collections import defaultdict
    from .runner import run_many

    # Every (backend, heads, init, seed, lr) is an independent job. best-over-LR
    # and the per-seed aggregation happen after, so parallel vs sequential give
    # identical summaries -- only wall-clock differs.
    jobs = []
    for backend, heads, init in product(args.backends, args.heads, args.inits):
        dim = heads * args.head_dim
        for seed in args.seeds:
            for lr in args.lrs:
                jobs.append(dict(backend=backend, heads=heads, dim=dim, init=init,
                                 seed=seed, lr=lr, extra=passthrough))

    # (backend, heads, init, seed) -> list of (lr, recall)
    collected: dict[tuple, list[tuple[float, float]]] = defaultdict(list)

    def on_done(job, res, n, total):
        acc = res.get("recall")
        print(f"  [{n}/{total}] {job['backend']:18s} heads={job['heads']:<2d} "
              f"dim={job['dim']:<4d} init={job['init']:<10s} seed={job['seed']} "
              f"lr={job['lr']:<7g}: {'FAIL' if acc is None else f'{acc:5.2f}%'}", flush=True)
        if acc is None and res.get("output"):
            print("      " + res["output"][-200:].replace("\n", "\n      "))
        if acc is not None:
            collected[(job["backend"], job["heads"], job["init"], job["seed"])].append(
                (job["lr"], acc))

    run_many(jobs, max_parallel=args.max_parallel, on_done=on_done)

    # results[(backend, heads, init)] = list of (best_recall, best_lr) per seed
    results: dict[tuple, list[tuple[float, float]]] = {}
    for backend, heads, init in product(args.backends, args.heads, args.inits):
        per_seed = []
        for seed in args.seeds:
            runs = collected.get((backend, heads, init, seed), [])
            if runs:
                best_lr, best_acc = max(runs, key=lambda x: x[1])
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
