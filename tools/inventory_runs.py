# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""What is actually finished, read from the results directory.

    python tools/inventory_runs.py --out /content/drive/MyDrive/flashnystrom_paper/runs

Needs no GPU and no runtime state: it reads the JSONs on disk. Use it after a
disconnect to find out what survived before deciding what to re-run, and to
print any tables that are already complete enough to use.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics


def load(pattern):
    for f in sorted(glob.glob(pattern)):
        if os.path.basename(f).endswith("summary.json"):
            continue
        try:
            yield f, json.load(open(f))
        except Exception:
            continue


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="runs")
    a = ap.parse_args(argv)
    root = a.out
    if not os.path.isdir(root):
        print(f"!! {root} does not exist. Check the path.")
        return 1

    print("=" * 66)
    print(f"INVENTORY  {root}")
    print("=" * 66)

    # ---- vision ----------------------------------------------------------- #
    vis = list(load(os.path.join(root, "vision_*.json")))
    print(f"\nVISION: {len(vis)} completed run(s)")
    by_tier = {}
    for f, r in vis:
        tier = os.path.basename(f)[7:-5].rsplit("_seed", 1)[0]
        by_tier.setdefault(tier, []).append(r)
    for tier, rs in sorted(by_tier.items()):
        n = rs[0].get("n_tokens", "?")
        print(f"  {tier:24s} N={n:>6}  {len(rs)} seed(s)")
        per_arm = {}
        for r in rs:
            for x in r["results"]:
                per_arm.setdefault(x["label"], []).append(x["test_acc"])
        for arm, accs in per_arm.items():
            sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
            print(f"      {arm:18s} {statistics.mean(accs):6.2f} +/- {sd:4.2f}"
                  f"  (n={len(accs)})")

    # ---- mqar ------------------------------------------------------------- #
    mq = list(load(os.path.join(root, "mqar", "*.json")))
    print(f"\nMQAR: {len(mq)} completed cell(s)")
    grid = {}
    for _, r in mq:
        if r.get("best_recall") is not None:
            grid.setdefault((r["backend"], r["dim"]), []).append(r["best_recall"])
    if grid:
        dims = sorted({d for _, d in grid})
        print(f"  {'method':<20}" + "".join(f"{'d=' + str(d):>10}" for d in dims))
        for m in sorted({m for m, _ in grid}):
            row = "".join(f"{max(grid[(m, d)]):10.2f}" if (m, d) in grid
                          else f"{'--':>10}" for d in dims)
            print(f"  {m:<20}{row}")

    # ---- genomics --------------------------------------------------------- #
    gen = list(load(os.path.join(root, "genomics", "*.json")))
    errs = glob.glob(os.path.join(root, "genomics", "*.error.txt"))
    print(f"\nGENOMICS: {len(gen)} completed cell(s), {len(errs)} recorded failure(s)")
    tasks = {}
    for _, r in gen:
        tasks.setdefault((r["task"], r.get("subset")), []).append(r)
    for (task, sub), rs in sorted(tasks.items(), key=lambda kv: str(kv[0])):
        print(f"  {task}{'/' + sub if sub else ''}: {len(rs)} cell(s)")
        per_arm = {}
        for r in rs:
            per_arm.setdefault(r["arm"], []).append(r["acc"])
        for arm, accs in sorted(per_arm.items()):
            print(f"      {arm:20s} best {max(accs):6.2f}  (n={len(accs)})")
    for e in errs[:5]:
        first = open(e).readline().strip()
        print(f"  !! {os.path.basename(e)}: {first}")

    total = len(vis) + len(mq) + len(gen)
    print(f"\n{'=' * 66}\n  {total} completed run(s) on disk")
    if not total:
        print("  Nothing finished. If the sweep appeared to run, check that --out\n"
              "  pointed at this directory and not at the runtime's local disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
