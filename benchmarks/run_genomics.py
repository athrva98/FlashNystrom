# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Genomics driver: all bidirectional arms x seeds, with an LR sweep.

The task (long-range repeat detection over k-mer tokens, see genomics.py) is
VERIFIED solvable: an oracle that simply asks whether token[0] recurs scores
94.0% (100% of positives, 12% spurious negatives). But a two-layer model only
finds the matching circuit after an MQAR-style phase transition, and short
schedules at a single learning rate leave every arm at chance with the loss
pinned at ln 2. So this driver does what the certified MQAR sweep does: sweep
the learning rate per arm, train long, and report the best.

    python benchmarks/run_genomics.py --arms sdpa flash_nystrom --seeds 0 1 2

VALIDITY GATE: a result is only meaningful if the exact-attention (sdpa) arm
clears --min-sdpa-acc. Below that the arms sit at chance and the table says
nothing about the operator, so the driver says so loudly rather than emitting
numbers that look like findings. Resumable: finished cells skip via their JSON.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.genomics import train_eval


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", nargs="+",
                    default=["sdpa", "linear_attention", "linformer",
                             "sliding_window", "nystrom_reference",
                             "flash_nystrom", "flash_nystrom_tc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lrs", nargs="+", type=float,
                    default=[1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n_train", type=int, default=32768)
    ap.add_argument("--n_test", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--num_landmarks", type=int, default=64)
    ap.add_argument("--out", default="runs/genomics")
    ap.add_argument("--min-sdpa-acc", dest="min_sdpa", type=float, default=85.0,
                    help="validity gate: exact attention must clear this, else "
                         "every arm is at chance and the comparison is void")
    ap.add_argument("--collect_only", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    if not a.collect_only:
        total = len(a.arms) * len(a.seeds) * len(a.lrs)
        print(f"genomics: {total} runs = {len(a.arms)} arms x {len(a.seeds)} "
              f"seeds x {len(a.lrs)} LRs, N={a.seq_len}, {a.epochs} epochs")
        n = 0
        for arm in a.arms:
            for seed in a.seeds:
                for lr in a.lrs:
                    n += 1
                    stem = f"{a.out}/{arm}_seed{seed}_lr{lr:.0e}"
                    if os.path.exists(stem + ".json"):
                        print(f"[{n}/{total}] skip {os.path.basename(stem)}")
                        continue
                    print(f"[{n}/{total}] {arm} seed={seed} lr={lr:g}", flush=True)
                    acc = train_eval(arm, seq_len=a.seq_len, dim=a.dim,
                                     heads=a.heads, num_landmarks=a.num_landmarks,
                                     n_train=a.n_train, n_test=a.n_test,
                                     epochs=a.epochs, batch_size=a.batch_size,
                                     lr=lr, seed=seed)
                    with open(stem + ".json", "w") as f:
                        json.dump({"arm": arm, "seed": seed, "lr": lr,
                                   "acc": acc, "seq_len": a.seq_len,
                                   "epochs": a.epochs, "n_train": a.n_train}, f,
                                  indent=2)

    # aggregate: best over LR per (arm, seed), then mean +/- sd over seeds
    runs = {}
    for f in glob.glob(f"{a.out}/*.json"):
        r = json.load(open(f))
        runs.setdefault((r["arm"], r["seed"]), []).append((r["lr"], r["acc"]))
    print(f"\n=== genomics: repeat detection, N={a.seq_len} "
          f"(best over LR, mean +/- sd over seeds) ===")
    table = {}
    for arm in a.arms:
        per_seed = [max(v, key=lambda x: x[1])[1]
                    for (aa, _), v in sorted(runs.items()) if aa == arm]
        if not per_seed:
            continue
        mu = statistics.mean(per_seed)
        sd = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        table[arm] = mu
        print(f"  {arm:22s} {mu:6.2f} +/- {sd:5.2f}  "
              f"{[round(x, 1) for x in per_seed]}")

    sdpa = table.get("sdpa")
    if sdpa is not None and sdpa < a.min_sdpa:
        print(f"\n  !! INVALID: exact attention reached only {sdpa:.1f}%, below the "
              f"{a.min_sdpa:.0f}% gate. Chance is 50% and the task oracle is 94%, so "
              f"every arm is near chance and these numbers say nothing about the "
              f"operator. Do NOT put this table in the paper: raise --epochs / "
              f"--n_train or widen --lrs until sdpa clears the gate.")
    elif sdpa is not None:
        print(f"\n  valid: exact attention at {sdpa:.1f}% (gate {a.min_sdpa:.0f}%, "
              f"oracle ceiling 94.0%)")


if __name__ == "__main__":
    main()
