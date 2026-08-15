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

# Estimated A100-80GB hours per job under --preset paper12, used only to order
# jobs and to warn before a stage that cannot finish in the remaining budget.
# Derived from the measured operator latencies; good to about a factor of two.
EST_HOURS = {
    "cifar10_p4_i32": 0.02, "stl10_p2_i96": 0.22, "stl10_p1_i96": 0.76,
    "stl10_p1_i180": 2.78, "sweep": 5.41, "species_N1024": 0.57,
    "species_N32768": 1.07, "genomic_benchmarks": 0.76, "repeat_diagnostic": 0.21,
}

# Scientific value per job, highest first. Ordering matters more than the
# estimate: if the budget runs out, what is missing should be the least
# load-bearing thing, not whatever happened to be last in the list.
PRIORITY = [
    "genomic_benchmarks",   # real data, cheap, and the only external anchor
    "cifar10_p4_i32",       # cheapest accuracy-neutrality check
    "stl10_p2_i96",         # first tier where sub-quadratic matters
    "species_N1024",        # genomics LR sweep; feeds the 32768 tier
    "stl10_p1_i96",
    "sweep",                # MQAR: the adversarial probe, the headline figure
    "species_N32768",       # long-context genomics
    "stl10_p1_i180",        # long-context vision
    "repeat_diagnostic",    # a diagnostic, not evidence
]


def build_jobs(stages, arms, seeds, out, smoke, species_dir="data/genomes",
               preset="full"):
    """Every job is (stage, name, argv). Smoke shrinks budgets, never coverage:
    the same stages and the same arms run, just briefly.

    preset="paper12" is the reduced grid costed for a ~12h A100 budget: the
    learning rate is swept once per task at its CHEAPEST length and the winner
    carried up, 3 seeds are kept wherever a claim needs error bars and dropped
    to 1 only at the most expensive tier of each stage, and the genomics species
    budget is cut from 655k windows to 82k. Every arm still appears at every
    context tier, and exact attention is still present as the reference.
    """
    jobs = []
    s = smoke
    mini = preset == "minimal"
    p12 = preset in ("paper12", "minimal")

    if "vision" in stages:
        # dataset, patch, img, epochs, batch, train_frac
        tiers = [("cifar10", 4, 32, 20, 128, 1.0),
                 ("stl10", 2, 96, 50, 128, 1.0),
                 ("stl10", 1, 96, 50, 48, 1.0),
                 ("stl10", 1, 180, 50, 16, 0.5)]
        if p12:
            # 30 epochs, not 50: STL-10 has 5000 labelled images and a small ViT
            # has converged well before then. The largest tier runs 1 seed; its
            # variance is taken from the three cheaper tiers and reported as such.
            tiers = [("cifar10", 4, 32, 20, 128, 1.0),
                     ("stl10", 2, 96, 30, 128, 1.0),
                     ("stl10", 1, 96, 30, 48, 1.0),
                     ("stl10", 1, 180, 30, 16, 0.5)]
        if s:
            tiers = [("cifar10", 4, 32, 1, 16, 0.02),
                     ("stl10", 2, 96, 1, 8, 0.02)]
        for seed in seeds:
            for ds, ps, img, ep, bs, frac in tiers:
                if mini and img == 180:
                    continue          # the 32K vision tier costs ~4.6h alone
                if p12 and img == 180 and seed != seeds[0]:
                    continue          # largest tier: one seed only
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
            # one dim, one LR, one seed: still every arm, still the real code.
            # --smoke also overrides the pinned 64-epoch/100k-example protocol,
            # which would otherwise cost ~4 min per arm.
            argv += ["--dims", "64", "--lrs", "1e-3", "--max_parallel", "1",
                     "--smoke"]
        elif p12:
            # 3 LR points still bracket Zoology's logspace(-4,-2,4) optimum, and
            # the driver already flags any winner sitting on a grid boundary.
            # 1 seed is paper_sweep's own default.
            argv = [a for a in argv if a not in ("--seeds",)]
            argv = argv[:argv.index("--out")] + ["--out", f"{out}/mqar"]
            argv += ["--seeds", "0", "--lrs", "1e-4", "1e-3", "1e-2",
                     "--max_parallel", "4"]
        else:
            argv += ["--max_parallel", "4"]
        jobs.append(("mqar", "sweep", argv))

    if "genomics" in stages:
        gcommon = [PY, "-u", "benchmarks/run_genomics.py",
                   "--arms", *arms, "--seeds", *[str(x) for x in seeds],
                   "--out", f"{out}/genomics"]
        lrs = ["1e-3"] if s else ["1e-4", "3e-4", "1e-3", "3e-3"]

        for N, bs in ([(256, 4)] if s else [(1024, 32), (32768, 4)]):
            # The LR is swept at N=1024 (cheap) and the winner carried to
            # N=32768, where a full grid costs 70h on its own. --lr_from points
            # the long-context run at the short one's summary.
            jlrs = lrs
            extra = []
            if p12 and N == 32768:
                jlrs = ["1e-3"]
                extra = ["--lr_from", f"{out}/genomics"]
            jobs.append(("genomics", f"species_N{N}", gcommon + [
                "--task", "species", "--seq_len", str(N),
                "--batch_size", str(bs), "--lrs", *jlrs,
                "--epochs", "1" if s else ("10" if (p12 and N == 32768) else "20"),
                "--n_train", "64" if s else ("8192" if (p12 and N == 32768) else "32768"),
                "--n_test", "32" if s else ("2048" if (p12 and N == 32768) else "4096"),
                "--chroms_per_split", "2" if s else "4",
                "--species_dir", species_dir] + extra))

        jobs.append(("genomics", "genomic_benchmarks", gcommon + [
            "--task", "genomic_benchmarks", "--lrs", *lrs,
            "--epochs", "1" if s else "40", "--batch_size", "8" if s else "32"]
            + (["--gb_datasets", "dummy_mouse_enhancers_ensembl"] if s else [])))

        jobs.append(("genomics", "repeat_diagnostic", gcommon + [
            "--task", "repeat", "--variant", "pointer",
            "--lrs", *(["1e-3"] if p12 else lrs),
            "--seq_len", "256" if s else "2048",
            "--epochs", "1" if s else "40",
            "--batch_size", "8" if s else "32",
            "--n_train", "256" if s else "32768",
            "--n_test", "128" if s else "4096"]))

    return jobs


# --------------------------------------------------------------------------- #

def failure_reason(log_path):
    """One line explaining a failure, so the summary distinguishes 'one optional
    kernel is missing' from 'the whole stage died'."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:                                          # pragma: no cover
        return ""
    import re
    cells = re.search(r"!! (\d+) cell\(s\) FAILED", text)
    mods = sorted(set(re.findall(r"No module named '([\w.]+)'", text)))
    bits = []
    if cells:
        bits.append(f"{cells.group(1)} cell(s)")
    if mods:
        bits.append("missing " + ", ".join(mods))
    if bits:
        return "; ".join(bits)
    for line in reversed(text.strip().splitlines()):
        if line.strip():
            return line.strip()[:70]
    return ""


def order_by_value(jobs):
    """Highest scientific value first, so a budget overrun drops the least
    load-bearing experiment rather than whichever happened to be last."""
    rank = {n: i for i, n in enumerate(PRIORITY)}
    return sorted(jobs, key=lambda j: rank.get(j[1].rsplit("_seed", 1)[0], 99))


def completion_marker(argv):
    """Path a job writes only on success, or None if it resumes internally.

    The MQAR and genomics drivers skip finished cells themselves, so their jobs
    have no marker here. train_three_way.py has no such check: without this it
    retrains every vision arm from scratch after a runtime reset, which on Colab
    is the difference between losing one run and losing six hours.
    """
    if "--out_json" in argv:
        return argv[argv.index("--out_json") + 1]
    return None


def run(jobs, log_dir, keep_going=True, budget_h=None):
    os.makedirs(log_dir, exist_ok=True)
    results, t_all = [], time.time()
    skipped = []
    for i, (stage, name, argv) in enumerate(jobs, 1):
        marker = completion_marker(argv)
        if marker and os.path.exists(marker):
            print(f"[{i}/{len(jobs)}] done {stage}/{name} -> {marker}", flush=True)
            continue
        if budget_h is not None:
            spent = (time.time() - t_all) / 3600
            est = EST_HOURS.get(name.rsplit("_seed", 1)[0], 0.0)
            if spent + est > budget_h and spent > 0:
                print(f"\n[{i}/{len(jobs)}] SKIP {stage}/{name}: needs ~{est:.1f}h,"
                      f" only {budget_h - spent:.1f}h of the {budget_h:.0f}h"
                      f" budget left. Resume with the same command.", flush=True)
                skipped.append((stage, name, est))
                continue
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
        why = "" if ok else "  <- " + failure_reason(log)
        print(f"  {'ok  ' if ok else 'FAIL'}  {stage:9s} {name:22s} {dt:7.0f}s{why}")
    if skipped:
        tot_sk = sum(e for _, _, e in skipped)
        print(f"\n  {len(skipped)} job(s) SKIPPED for budget (~{tot_sk:.1f}h)."
              f" Everything is resumable: re-run the same command to continue.")
        for stage, name, est in skipped:
            print(f"    {stage}/{name}  (~{est:.1f}h)")
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
    ap.add_argument("--species_dir", default="data/genomes",
                    help="reference genomes for the species task")
    ap.add_argument("--preset", default="full",
                    choices=["full", "paper12", "minimal"],
                    help="paper12: reduced grid for a ~12h A100 budget. "
                         "minimal: paper12 without the 32K vision tier. "
                         "Genomics is kept: it is what makes the paper's "
                         "bidirectional-domains argument. ~11h.")
    ap.add_argument("--budget_hours", type=float, default=None,
                    help="stop launching jobs once the budget is spent; jobs run "
                         "highest-value-first and everything is resumable")
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

    jobs = build_jobs(a.stages, a.arms, seeds, out, a.smoke, a.species_dir,
                      a.preset)
    if a.budget_hours:
        jobs = order_by_value(jobs)
        est = sum(EST_HOURS.get(n.rsplit("_seed", 1)[0], 0.0) for _, n, _ in jobs)
        print(f"budget {a.budget_hours:.0f}h | estimated {est:.1f}h | "
              f"highest-value-first ordering")
    mode = ("SMOKE (tiny budgets, results are NOT results)" if a.smoke
            else f"{a.preset.upper()} grid")
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
    if "genomics" in a.stages and not os.path.isdir(a.species_dir):
        print("\n!! data/genomes not found; the species sub-stage will fail. "
              "Fetch it first:\n     python benchmarks/download_genomes.py "
              "--out data/genomes")

    rc = run(jobs, log_dir, keep_going=not a.stop_on_fail,
             budget_h=a.budget_hours)
    if a.smoke:
        print("\nSmoke complete. Accuracies above are meaningless by "
              "construction. If everything says 'ok', run the full sweep:\n"
              "    python run_all_paper_experiments.py")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
