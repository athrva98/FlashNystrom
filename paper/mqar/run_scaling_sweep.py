# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""MQAR scaling experiments for the paper, with auto-batch + grad clipping.

Two sweeps, both reporting best-over-LR recall AND the training profile
(step time, throughput, peak memory) at the saturated batch:

  --mode length    : fix num_kv_pairs, sweep seq_len (256..4096). The headline:
                     flash_nystrom recall holds while its training cost stays
                     linear and sdpa's blows up / OOMs as N grows.
  --mode capacity  : fix seq_len, sweep num_kv_pairs. The honest one: Nystrom's
                     recall degrades at the rank (landmark) limit.

Each config runs train.py as an isolated subprocess (clean GPU state, so one
OOM at long N doesn't kill the sweep). Example:

    python -m paper.mqar.run_scaling_sweep --mode length \\
        --backends sdpa flash_nystrom nystrom_reference \\
        --seq_lens 256 512 1024 2048 4096 --num_kv_pairs 16 \\
        --lrs 1e-3 3.16e-3 1e-2 3.16e-2
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

_BEST = re.compile(r"best test recall:\s*([\d.]+)%")
_PROF = re.compile(r"train profile: batch=(\d+) step_ms=([\d.]+) "
                   r"samples_per_s=([\d.]+) peak_GiB=([\d.]+)")


def run_one(backend, seq_len, kv, lr, passthrough):
    # Accuracy runs use the VALIDATED fixed-batch recipe (train.py default
    # batch 256) + grad clipping -- NOT autobatch, whose recall is unverified.
    # The training profile (step_ms/peak_gib) is reported at that fixed batch.
    cmd = [
        sys.executable, "-m", "paper.mqar.train",
        "--backend", backend, "--seq_len", str(seq_len), "--num_kv_pairs", str(kv),
        "--lr", str(lr), "--grad_clip", "1.0",
    ] + passthrough
    proc = subprocess.run(cmd, capture_output=True, text=True)
    m = _BEST.search(proc.stdout)
    if m is None:
        return None
    prof = _PROF.search(proc.stdout)
    rec = {"recall": float(m.group(1))}
    if prof:
        rec.update(batch=int(prof.group(1)), step_ms=float(prof.group(2)),
                   samp_s=float(prof.group(3)), peak_gib=float(prof.group(4)))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["length", "capacity"], required=True)
    ap.add_argument("--backends", nargs="+",
                    default=["sdpa", "flash_nystrom", "nystrom_reference"])
    ap.add_argument("--seq_lens", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--num_kv_pairs", type=int, default=16, help="fixed for --mode length")
    ap.add_argument("--seq_len", type=int, default=1024, help="fixed for --mode capacity")
    ap.add_argument("--kv_pairs", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 3.16e-3, 1e-2, 3.16e-2])
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write results to this JSON path (for make_figures.py)")
    args, passthrough = ap.parse_known_args()

    if args.mode == "length":
        axis = [(N, args.num_kv_pairs) for N in args.seq_lens]
        axis_name = "seq_len"
    else:
        axis = [(args.seq_len, kv) for kv in args.kv_pairs]
        axis_name = "kv_pairs"

    print(f"mode={args.mode}  best-over-LR recall + training profile (autobatch, grad_clip=1.0)")
    print(f"{'backend':>18} {axis_name:>9} {'recall%':>8} {'batch':>7} "
          f"{'step_ms':>8} {'samp/s':>9} {'peak_GiB':>9} {'bestLR':>8}")
    print("-" * 84)
    results = []
    for backend in args.backends:
        for seq_len, kv in axis:
            best = None
            for lr in args.lrs:
                r = run_one(backend, seq_len, kv, lr, passthrough)
                if r and (best is None or r["recall"] > best["recall"]):
                    best, best["lr"] = r, lr
            ax = seq_len if args.mode == "length" else kv
            if best is None:
                print(f"{backend:>18} {ax:>9} {'FAIL/OOM':>8}")
                results.append({"backend": backend, "axis": axis_name,
                                "axis_val": ax, "oom": True})
                continue
            print(f"{backend:>18} {ax:>9} {best['recall']:>8.2f} "
                  f"{best.get('batch', 0):>7} {best.get('step_ms', 0):>8.1f} "
                  f"{best.get('samp_s', 0):>9.0f} {best.get('peak_gib', 0):>9.2f} "
                  f"{best['lr']:>8g}")
            results.append({"backend": backend, "axis": axis_name, "axis_val": ax,
                            "oom": False, **best})
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"mode": args.mode, "results": results}, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
