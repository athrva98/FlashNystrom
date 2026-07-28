# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Every training run behind the paper, from one file.

    python run_all_paper_experiments.py --smoke     # FIRST: minutes, finds bugs
    python run_all_paper_experiments.py             # the real thing, GPU-days
    python run_all_paper_experiments.py --stages genomics
    python run_all_paper_experiments.py --list      # print the plan, run nothing

ALWAYS run --smoke before the real thing. It executes every stage, every arm
and every code path with the budgets cut to near zero: one epoch, a few hundred
examples, one learning rate, tiny sequences. It proves the plumbing works and
surfaces import errors, shape errors, missing dependencies and unregistered
arms in minutes instead of after a day of GPU time. It says nothing about
accuracy and its results are written to a separate directory so they can never
be mistaken for real ones.

Each stage is a subprocess. One arm crashing does not kill the run: the failure
is recorded and the driver continues, then every failure is reprinted at the
end with the exact command needed to reproduce it. Everything is resumable
(finished cells skip via their result JSON), so a disconnect costs only the run
in flight.

Run the real sweep under tmux or nohup.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# The bidirectional-native arms. Only the attention operator differs between
# them; backbone, optimizer, schedule, seeds and data are identical.
ARMS = ["sdpa", "linear_attention", "linformer", "sliding_window",
        "nystrom_reference", "flash_nystrom", "flash_nystrom_tc"]

STAGES = ["vision", "mqar", "genomics"]


# --------------------------------------------------------------------------- #
# dependency preflight: report everything missing up front rather than one
# failure at a time, hours apart
# --------------------------------------------------------------------------- #

DEPS = [
    ("torch", "core", "required"),
    ("flash_attn", "sliding_window arm (fused windowed kernel)",
     "pip install flash-attn --no-build-isolation"),
    ("flash_bla", "linear_attention arm (fused Triton kernel); without it the "
     "arm silently falls back to an UNFUSED torch path and is not a fair baseline",
     "pip install -e git+https://github.com/fla-org/flash-bidirectional-linear-"
     "attention.git#egg=flash_bla"),
    ("pyfaidx", "genomics species task", "pip install pyfaidx"),
    ("genomic_benchmarks", "genomics regulatory task",
     "pip install genomic-benchmarks"),
]


def preflight(verbose=True):
    missing = []
    if verbose:
        print("=== dependencies ===")
    for mod, what, how in DEPS:
        ok = subprocess.run([PY, "-c", f"import {mod}"],
                            capture_output=True).returncode == 0
        if verbose:
            print(f"  {mod:20s} {'OK' if ok else 'MISSING':8s} {what}")
        if not ok:
            missing.append((mod, how))
    if missing and verbose:
        print("\n  install:")
        for mod, how in missing:
            print(f"    {mod}: {how}")
    if verbose:
        gpu = subprocess.run(
            [PY, "-c", "import torch;print(torch.cuda.get_device_name(0) "
                       "if torch.cuda.is_available() else 'NO CUDA')"],
            capture_output=True, text=True)
        print(f"\n  device: {gpu.stdout.strip() or gpu.stderr.strip()[:80]}")
    return [m for m, _ in missing]


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #

def build_jobs(stages, arms, seeds, out, smoke):
    """Every job is (stage, name, argv). Smoke shrinks budgets, never coverage:
    the same stages and the same arms run, just briefly."""
    jobs = []
    s = smoke

    if "vision" in stages:
        # dataset, patch, img, epochs, batch, train_frac
        tiers = [("cifar10", 4, 32, 20, 128, 1.0),
                 ("stl10", 2, 96, 50, 128, 1.0),
                 ("stl10", 1, 96, 50, 48, 1.0),
                 ("stl10", 1, 180, 50, 16, 0.5)]
        if s:
            tiers = [("cifar10", 4, 32, 1, 16, 0.02),
                     ("stl10", 2, 96, 1, 8, 0.02)]
        for seed in seeds:
            for ds, ps, img, ep, bs, frac in tiers:
                tag = f"{ds}_p{ps}_i{img}_seed{seed}"
                jobs.append(("vision", tag, [
                    PY, "-u", "benchmarks/train_three_way.py",
                    "--dataset", ds, "--patch_size", str(ps),
                    "--img_size", str(img), "--epochs", str(ep),
                    "--batch_size", str(bs), "--train_frac", str(frac),
                    "--backends", *arms, "--seed", str(seed),
                    "--kappa_star", "0", "--no-instrument",
                    "--out_json", f"{out}/vision_{tag}.json"]))

    if "mqar" in stages:
        argv = [PY, "-u", "-m", "paper.mqar.paper_sweep",
                "--methods", *arms, "--out", f"{out}/mqar",
                "--seeds", *[str(x) for x in seeds]]
        if s:
            # one dim, one LR, one seed: still every arm, still the real code
            argv += ["--dims", "64", "--lrs", "1e-3", "--max_parallel", "1"]
        else:
            argv += ["--max_parallel", "4"]
        jobs.append(("mqar", "sweep", argv))

    if "genomics" in stages:
        gcommon = [PY, "-u", "benchmarks/run_genomics.py",
                   "--arms", *arms, "--seeds", *[str(x) for x in seeds],
                   "--out", f"{out}/genomics"]
        lrs = ["1e-3"] if s else ["1e-4", "3e-4", "1e-3", "3e-3"]

        for N, bs in ([(256, 4)] if s else [(1024, 32), (32768, 4)]):
            jobs.append(("genomics", f"species_N{N}", gcommon + [
                "--task", "species", "--seq_len", str(N),
                "--batch_size", str(bs), "--lrs", *lrs,
                "--epochs", "1" if s else "20",
                "--n_train", "64" if s else "32768",
                "--n_test", "32" if s else "4096",
                "--chroms_per_split", "2" if s else "4",
                "--species_dir", "data/genomes"]))

        jobs.append(("genomics", "genomic_benchmarks", gcommon + [
            "--task", "genomic_benchmarks", "--lrs", *lrs,
            "--epochs", "1" if s else "40", "--batch_size", "8" if s else "32"]
            + (["--gb_datasets", "dummy_mouse_enhancers_ensembl"] if s else [])))

        jobs.append(("genomics", "repeat_diagnostic", gcommon + [
            "--task", "repeat", "--variant", "pointer", "--lrs", *lrs,
            "--seq_len", "256" if s else "2048",
            "--epochs", "1" if s else "40",
            "--batch_size", "8" if s else "32",
            "--n_train", "256" if s else "32768",
            "--n_test", "128" if s else "4096"]))

    return jobs


# --------------------------------------------------------------------------- #

def run(jobs, log_dir, keep_going=True):
    os.makedirs(log_dir, exist_ok=True)
    results, t_all = [], time.time()
    for i, (stage, name, argv) in enumerate(jobs, 1):
        log = os.path.join(log_dir, f"{stage}_{name}.log")
        print(f"\n[{i}/{len(jobs)}] {stage}/{name}\n  {' '.join(argv)}",
              flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            p = subprocess.run(argv, cwd=HERE, stdout=fh,
                               stderr=subprocess.STDOUT)
        dt = time.time() - t0
        ok = p.returncode == 0
        results.append((stage, name, ok, dt, log, argv))
        print(f"  {'ok' if ok else 'FAILED (rc=%d)' % p.returncode} "
              f"in {dt:.0f}s -> {log}", flush=True)
        if not ok:
            # show the tail immediately: waiting until the end to learn a stage
            # died is the thing --smoke exists to prevent
            with open(log, encoding="utf-8", errors="replace") as fh:
                tail = fh.read()[-1500:]
            print("  --- tail ---")
            for line in tail.splitlines()[-15:]:
                print(f"  | {line}")
            if not keep_going:
                break

    print(f"\n{'=' * 70}\nSUMMARY  ({time.time() - t_all:.0f}s total)")
    for stage, name, ok, dt, log, _ in results:
        print(f"  {'ok  ' if ok else 'FAIL'}  {stage:9s} {name:22s} {dt:7.0f}s")
    bad = [r for r in results if not r[2]]
    if bad:
        print(f"\n{len(bad)} FAILED. Reproduce individually:")
        for stage, name, _, _, log, argv in bad:
            print(f"  # {stage}/{name}  (full log: {log})")
            print(f"  {' '.join(argv)}")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="default: [0] under --smoke, [0,1,2] otherwise")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny budgets, full coverage: run this FIRST")
    ap.add_argument("--out", default=None,
                    help="default: runs/ (or runs_smoke/ under --smoke)")
    ap.add_argument("--list", action="store_true",
                    help="print the plan and exit")
    ap.add_argument("--stop_on_fail", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the output dir first (smoke only)")
    a = ap.parse_args(argv)

    seeds = a.seeds if a.seeds is not None else ([0] if a.smoke else [0, 1, 2])
    out = a.out or ("runs_smoke" if a.smoke else "runs")
    log_dir = "logs_smoke" if a.smoke else "logs"

    if a.fresh and a.smoke and os.path.isdir(out):
        shutil.rmtree(out)          # smoke results are disposable by design
    os.makedirs(out, exist_ok=True)

    jobs = build_jobs(a.stages, a.arms, seeds, out, a.smoke)
    mode = "SMOKE (tiny budgets, results are NOT results)" if a.smoke else "FULL"
    print(f"=== {mode} ===")
    print(f"stages {a.stages} | {len(a.arms)} arms | seeds {seeds} | out {out}/")

    if a.list:
        for stage, name, argv_ in jobs:
            print(f"  {stage:9s} {name:22s} {' '.join(argv_)}")
        return 0

    missing = preflight()
    if missing:
        blocking = {"pyfaidx", "genomic_benchmarks"} & set(missing)
        if blocking and "genomics" in a.stages:
            print(f"\n!! genomics needs {sorted(blocking)}; those sub-stages "
                  f"will fail. Install them or drop the stage.")
        if "flash_bla" in missing:
            print("\n!! flash_bla missing: the linear_attention arm would run "
                  "UNFUSED, which is not a fair baseline. Install it before "
                  "any run whose numbers go in the paper.")
    if "genomics" in a.stages and not os.path.isdir("data/genomes"):
        print("\n!! data/genomes not found; the species sub-stage will fail. "
              "Fetch it first:\n     python benchmarks/download_genomes.py "
              "--out data/genomes")

    rc = run(jobs, log_dir, keep_going=not a.stop_on_fail)
    if a.smoke:
        print("\nSmoke complete. Accuracies above are meaningless by "
              "construction. If everything says 'ok', run the full sweep:\n"
              "    python run_all_paper_experiments.py")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
