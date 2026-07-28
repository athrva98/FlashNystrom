# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""The certified MQAR experiment for the paper: one protocol, one command.

    python -m paper.mqar.paper_sweep                  # run everything (resumable)
    python -m paper.mqar.paper_sweep --collect_only   # re-print the table
    python -m paper.mqar.paper_sweep --dry_run        # list the runs, touch nothing

Every MQAR number in the paper comes from this driver and nowhere else. The
implementation underneath is the single (data.py, model.py, train.py) stack;
this file only fixes the protocol and fans out the runs.

Protocol (the ``PROTOCOL`` dict below, passed verbatim to train.py):

  task    seq_len 512, 64 key-value pairs, vocab 8192, blank non-query slots
  data    100,000 train / 3,000 test, fixed train set, seed-disjoint test set
  model   depth 2, uniform layout (every layer is the mixer, no BaseConv),
          head_dim held at 64 (heads = dim/64), position embeddings for the
          attention-family mixers only (Hyena/Mamba carry order in their own
          conv/recurrence), bf16; every maskable method runs BIDIRECTIONAL --
          one direction convention for the whole table, since the Nystrom
          family has no causal form and masking only the baselines would
          confound the operator comparison. Hyena and Mamba are causal by
          construction (an operator property, not a protocol knob).
  optim   AdamW wd 0.1, cosine anneal over 64 epochs, no warmup, no gradient
          clipping, batch 128, early stop once test recall clears 99%
  sweep   dim in {64,128,256,512} x lr in logspace(-4,-2,4); the reported
          number per (method, dim) is the best over the LR grid, mean over
          seeds (1 seed by default; --seeds 0 1 2 for error bars)

Why exactly this protocol: it is Zoology's figure-2 recipe (Arora et al.,
ICLR 2024) at the (512, 64) setting, and this harness REPRODUCES the published
DeltaNet Figure 4 under it (Yang et al., arXiv:2406.06484): our Hyena reaches
18.6% at d=512 vs their ~20%, and our Mamba 99.1% at d=256 vs their ~100%.
Those two external anchors certify the harness; the remaining methods are then
measured under identical conditions.

Cost: 112 runs (7 methods x 4 dims x 4 LRs x 1 seed). Solvers early-stop in
~30 epochs; the methods that never reach 99% (Hyena everywhere, others at low
dim) run all 64. Budget on the order of a day on one A100 at the default
--max_parallel 4. Fully resumable: finished runs are skipped by out_json, so
re-running the same command after a session reset continues where it stopped.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics

from .runner import run_many

# The bidirectional-native set. Hyena and Mamba are excluded: they are causal
# by construction, so comparing them here would measure the masking regime
# rather than the operator (see the paper's experiments preamble).
METHODS = ["sdpa", "linear_attention", "linformer", "sliding_window",
           "nystrom_reference", "flash_nystrom", "flash_nystrom_tc"]
DIMS = [64, 128, 256, 512]
LRS = [1e-4, 4.641589e-4, 2.154435e-3, 1e-2]   # np.logspace(-4, -2, 4)

# Direction convention: ONE convention for the whole table -- every method that
# admits a masking choice runs BIDIRECTIONAL (no causal mask). The Nystrom
# family has no causal form, and mixing masked baselines with an unmasked
# subject would confound the operator comparison, so none of the maskable
# methods is masked. Bidirectionality leaks nothing: each bound value lies
# earlier in the sequence than its query. Hyena and Mamba are causal BY
# CONSTRUCTION (causal conv / recurrent scan) -- an operator property, not a
# protocol knob, and the same form behind their published MQAR results, so the
# DeltaNet-Fig4 validation anchors are unaffected.

# Every train.py knob that is not swept, pinned so the protocol cannot drift
# with a default change. Booleans use runner's --no- handling where needed.
PROTOCOL = dict(
    layer_layout="uniform",
    random_non_queries=False,
    fresh_data=False,
    kappa_star=0,
    seq_len=512,
    num_kv_pairs=64,
    vocab_size=8192,
    power_a=0.01,
    depth=2,
    num_landmarks=64,
    newton_iter=6,
    batch_size=128,
    epochs=64,
    early_stop_acc=0.99,
    weight_decay=0.1,
    dtype="bf16",
    num_train=100_000,
    num_test=3_000,
)


def heads_for(dim: int) -> int:
    """head_dim held at 64 across the dim sweep (heads = dim/64): valid for the
    flash_nystrom kernels at every dim and matching Zoology's num_heads=2 at
    d=128. Hyena and Mamba ignore the head count entirely."""
    return max(1, dim // 64)


def build_jobs(methods, dims, lrs, seeds, out_dir):
    """One run_many job per (method, dim, lr, seed). out_json doubles as the
    resume key; log_path keeps each run's full epoch log next to its record."""
    jobs = []
    for m in methods:
        for d in dims:
            for lr in lrs:
                for s in seeds:
                    stem = os.path.join(out_dir, f"{m}_d{d}_lr{lr:.2e}_seed{s}")
                    jobs.append(dict(
                        backend=m, seed=s, dim=d, heads=heads_for(d),
                        lr=f"{lr:.6e}",
                        out_json=stem + ".json", log_path=stem + ".log",
                        **PROTOCOL,
                    ))
    return jobs


def aggregate(out_dir, methods, dims, lrs):
    """Best-over-LR per (method, dim, seed), mean +/- sd over seeds.

    Returns (table, edge_flags) and writes ``summary.json``. edge_flags lists
    unsolved cells whose winning LR sits on a grid boundary: those numbers are
    lower bounds, not located optima, and the grid should be extended there.
    """
    runs: dict[tuple, list[tuple[float, float]]] = {}
    for f in glob.glob(os.path.join(out_dir, "*.json")):
        if os.path.basename(f) == "summary.json":
            continue
        r = json.load(open(f))
        if r.get("best_recall") is None:
            continue
        runs.setdefault((r["backend"], r["dim"], r["seed"]), []).append(
            (r["lr"], r["best_recall"]))

    lo, hi = min(lrs), max(lrs)
    table: dict[tuple, dict] = {}
    edge_flags: list[tuple] = []
    for m in methods:
        for d in dims:
            per_seed = []
            for (mm, dd, s), lr_recs in sorted(runs.items()):
                if mm != m or dd != d:
                    continue
                lr_b, r_b = max(lr_recs, key=lambda x: x[1])
                per_seed.append((s, lr_b, r_b))
                # A best at the grid edge only matters when the run is NOT
                # solved: a 99%+ cell is capped by the task, not the grid.
                at_edge = (math.isclose(lr_b, lo, rel_tol=1e-3)
                           or math.isclose(lr_b, hi, rel_tol=1e-3))
                if at_edge and r_b < 99.0:
                    edge_flags.append((m, d, s, lr_b, r_b))
            if per_seed:
                vals = [r for _, _, r in per_seed]
                table[(m, d)] = dict(
                    mean=statistics.mean(vals),
                    sd=statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    n=len(vals),
                    best_lrs=sorted({lr for _, lr, _ in per_seed}),
                )

    print(f"\nMQAR recall vs model dimension "
          f"(seq_len {PROTOCOL['seq_len']}, {PROTOCOL['num_kv_pairs']} kv pairs, "
          f"best over LR grid, mean over seeds):")
    print(f"  {'method':<20} " + "  ".join(f"{'d='+str(d):>9}" for d in dims))
    for m in methods:
        row = []
        for d in dims:
            c = table.get((m, d))
            row.append(f"{c['mean']:9.2f}" if c else f"{'--':>9}")
        print(f"  {m:<20} " + "  ".join(row))
    if edge_flags:
        print("\n  ! best-over-LR at a grid boundary (unsolved cell -> the value "
              "is a lower bound; extend --lrs there):")
        for m, d, s, lr_b, r_b in edge_flags:
            print(f"    {m} d={d} seed={s}: {r_b:.2f}% at lr={lr_b:g}")

    done = len(runs and [1 for v in runs.values() for _ in v] or [])
    total = len(methods) * len(dims) * len(lrs)  # per seed
    print(f"\n  runs on disk: {done} (grid is {total} per seed)")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "protocol": PROTOCOL,
            "lr_grid": lrs,
            "table": {f"{m}|d{d}": c for (m, d), c in table.items()},
            "edge_flags": [list(e) for e in edge_flags],
        }, f, indent=2)
    return table, edge_flags


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="The certified MQAR sweep for the paper (see module docstring).")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--dims", nargs="+", type=int, default=DIMS)
    ap.add_argument("--lrs", nargs="+", type=float, default=LRS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--max_parallel", type=int, default=4)
    ap.add_argument("--out", default="runs/mqar_paper")
    ap.add_argument("--collect_only", action="store_true",
                    help="skip running; aggregate whatever is on disk")
    ap.add_argument("--dry_run", action="store_true",
                    help="print the run list and exit without training")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    jobs = build_jobs(args.methods, args.dims, args.lrs, args.seeds, args.out)
    print(f"certified MQAR sweep: {len(jobs)} runs = {len(args.methods)} methods "
          f"x {len(args.dims)} dims x {len(args.lrs)} LRs x {len(args.seeds)} seed(s)")
    print("protocol: " + " ".join(f"{k}={v}" for k, v in PROTOCOL.items()))

    if args.dry_run:
        for j in jobs:
            print("  " + os.path.basename(j["out_json"]))
        return

    if not args.collect_only:
        run_many(jobs, max_parallel=args.max_parallel)
    aggregate(args.out, args.methods, args.dims, args.lrs)


if __name__ == "__main__":
    main()
