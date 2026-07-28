# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Genomics driver: all bidirectional arms x LRs x seeds, one task at a time.

    python benchmarks/run_genomics.py --task species --seq_len 1024
    python benchmarks/run_genomics.py --task species --seq_len 32768
    python benchmarks/run_genomics.py --task genomic_benchmarks
    python benchmarks/run_genomics.py --task repeat            # diagnostic

Tasks and their provenance are documented in genomics_data.py. Reported number
per arm is best-over-LR, then mean +/- sd over seeds, matching how the MQAR
sweep reports.

VALIDITY GATE. Each task carries its own gate, because chance and ceiling
differ by an order of magnitude between them and a single threshold was wrong
for all of them. A result counts only if the exact-attention (sdpa) arm clears
its gate; below that every arm sits near chance and the table says nothing
about the operator. The gate, chance level and ceiling are printed together so
the margin is visible rather than asserted.

Resumable: finished cells skip via their JSON.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.genomics import train_eval                          # noqa: E402
from benchmarks.genomics_data import (                              # noqa: E402
    DEFAULT_GB, DEFAULT_SPECIES, GB_DATASETS,
)

ARMS = ["sdpa", "linear_attention", "linformer", "sliding_window",
        "nystrom_reference", "flash_nystrom", "flash_nystrom_tc"]

# gate, and how chance/ceiling are computed for the header line.
TASK_GATES = {
    # 5-way over real genomes: chance 20%. HyenaDNA reports well above this at
    # every length, so a gate at 2x chance is generous and still meaningful.
    "species": 40.0,
    # Binary/3-way regulatory element tasks. Published top-1 for comparable
    # small models sits in the 65-85% band (HyenaDNA Table 4.1), so 60% is
    # "learned real signal" without demanding a specific published number.
    "genomic_benchmarks": 60.0,
    # Pointer retrieval: chance is 1/(L-1), about 0.05% at L=2048. Anything
    # above a few percent is unambiguous signal; 50% demands a working circuit.
    "repeat": 50.0,
}


def cell_path(out, task, arm, seed, lr, extra):
    tag = f"{arm}_seed{seed}_lr{lr:.0e}" + (f"_{extra}" if extra else "")
    return os.path.join(out, f"{task}_{tag}.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="species",
                    choices=["species", "genomic_benchmarks", "repeat"])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lrs", nargs="+", type=float,
                    default=[1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n_train", type=int, default=32768)
    ap.add_argument("--n_test", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--num_landmarks", type=int, default=64)
    # species
    ap.add_argument("--species", nargs="+", default=DEFAULT_SPECIES)
    ap.add_argument("--species_dir", default="data/genomes")
    ap.add_argument("--chroms_per_split", type=int, default=4)
    # genomic benchmarks
    ap.add_argument("--gb_datasets", nargs="+", default=DEFAULT_GB,
                    choices=sorted(GB_DATASETS))
    # synthetic diagnostic
    ap.add_argument("--variant", default="pointer", choices=["pointer", "detect"])
    ap.add_argument("--out", default="runs/genomics")
    ap.add_argument("--gate", type=float, default=None,
                    help="override the task's default sdpa validity gate")
    ap.add_argument("--collect_only", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    gate = a.gate if a.gate is not None else TASK_GATES[a.task]
    # sub-jobs: one per GB dataset, or a single unnamed job for the others
    subjobs = a.gb_datasets if a.task == "genomic_benchmarks" else [None]

    if a.task == "species":
        chance = 100.0 / len(a.species)
        ceiling = 100.0
        label = f"species x{len(a.species)} @ N={a.seq_len}"
    elif a.task == "repeat":
        chance = 100.0 / (a.seq_len - 1) if a.variant == "pointer" else 50.0
        ceiling = 100.0     # exact by construction now; see genomics_data.py
        label = f"repeat/{a.variant} @ N={a.seq_len}"
    else:
        chance, ceiling = 50.0, 100.0
        label = "genomic_benchmarks"

    if not a.collect_only:
        total = len(a.arms) * len(a.seeds) * len(a.lrs) * len(subjobs)
        print(f"genomics [{label}]: {total} runs = {len(a.arms)} arms x "
              f"{len(a.seeds)} seeds x {len(a.lrs)} LRs x {len(subjobs)} set(s)")
        print(f"chance {chance:.2f}%, ceiling {ceiling:.1f}%, sdpa gate {gate:.1f}%")
        n = 0
        for sub in subjobs:
            for arm in a.arms:
                for seed in a.seeds:
                    for lr in a.lrs:
                        n += 1
                        path = cell_path(a.out, a.task, arm, seed, lr, sub)
                        if os.path.exists(path):
                            print(f"[{n}/{total}] skip {os.path.basename(path)}")
                            continue
                        print(f"[{n}/{total}] {a.task} {sub or ''} {arm} "
                              f"seed={seed} lr={lr:g}", flush=True)
                        acc = train_eval(
                            arm, task=a.task, seq_len=a.seq_len, dim=a.dim,
                            heads=a.heads, depth=a.depth,
                            num_landmarks=a.num_landmarks, epochs=a.epochs,
                            batch_size=a.batch_size, lr=lr, seed=seed,
                            n_train=a.n_train, n_test=a.n_test,
                            species_dir=a.species_dir, species=a.species,
                            chroms_per_split=a.chroms_per_split,
                            gb_dataset=sub, variant=a.variant)
                        with open(path, "w") as f:
                            json.dump({"task": a.task, "subset": sub, "arm": arm,
                                       "seed": seed, "lr": lr, "acc": acc,
                                       "seq_len": a.seq_len, "epochs": a.epochs,
                                       "n_train": a.n_train}, f, indent=2)

    # ---- aggregate: best over LR per (arm, seed, subset), mean over seeds ---
    runs = {}
    for f in glob.glob(os.path.join(a.out, f"{a.task}_*.json")):
        if os.path.basename(f).endswith("summary.json"):
            continue
        r = json.load(open(f))
        runs.setdefault((r.get("subset"), r["arm"], r["seed"]), []).append(
            (r["lr"], r["acc"]))

    summary = {}
    for sub in subjobs:
        print(f"\n=== {a.task}{'/' + sub if sub else ''} "
              f"(best over LR, mean +/- sd over seeds) ===")
        print(f"  chance {chance:.2f}%   ceiling {ceiling:.1f}%   "
              f"gate {gate:.1f}%")
        table = {}
        for arm in a.arms:
            per_seed = [max(v, key=lambda t: t[1])[1]
                        for (ss, aa, _), v in sorted(runs.items())
                        if aa == arm and ss == sub]
            if not per_seed:
                continue
            mu = statistics.mean(per_seed)
            sd = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
            table[arm] = mu
            print(f"  {arm:22s} {mu:6.2f} +/- {sd:5.2f}  "
                  f"{[round(x, 1) for x in per_seed]}")
        summary[sub or a.task] = table

        sdpa = table.get("sdpa")
        if sdpa is None:
            print("  (no sdpa cell yet: cannot judge validity)")
        elif sdpa < gate:
            print(f"\n  !! INVALID: exact attention reached only {sdpa:.1f}%, "
                  f"below the {gate:.1f}% gate (chance {chance:.2f}%). Every arm "
                  f"is near chance, so these numbers say nothing about the "
                  f"operator. Do NOT put this in the paper: raise --epochs / "
                  f"--n_train or widen --lrs until sdpa clears the gate.")
        else:
            print(f"\n  valid: exact attention at {sdpa:.1f}% "
                  f"(gate {gate:.1f}%, chance {chance:.2f}%)")

    with open(os.path.join(a.out, f"{a.task}_summary.json"), "w") as f:
        json.dump({"task": a.task, "seq_len": a.seq_len, "chance": chance,
                   "ceiling": ceiling, "gate": gate, "lrs": a.lrs,
                   "seeds": a.seeds, "table": summary}, f, indent=2)


if __name__ == "__main__":
    main()
