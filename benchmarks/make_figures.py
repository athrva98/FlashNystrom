# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Turn the experiment JSON dumps into paper-ready PDF figures.

Reads (whichever exist):
  scaling.json        from profile_scaling.py --json
  mqar_length.json    from run_scaling_sweep.py --mode length --json
  mqar_capacity.json  from run_scaling_sweep.py --mode capacity --json
  three_way_results.json   from train_three_way.py (CIFAR)

Writes vector PDFs into --outdir (default figures/). Missing inputs are skipped
so a partial run still produces what it can.

    python benchmarks/make_figures.py
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"sdpa": "Full attn (SDPA)", "flash_nystrom": "FlashNystrom",
         "nystrom_reference": "Nystrom ref", "SDPA": "Full attn (SDPA)",
         "FlashNystrom": "FlashNystrom", "Nystrom-Ref": "Nystrom ref"}
COLOR = {"sdpa": "tab:blue", "flash_nystrom": "tab:orange",
         "nystrom_reference": "tab:green", "SDPA": "tab:blue",
         "FlashNystrom": "tab:orange", "Nystrom-Ref": "tab:green"}


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _backends(rows, key="backend"):
    return list(dict.fromkeys(r[key] for r in rows))


def fig_scaling(data, outdir):
    rows = [r for r in data["rows"] if not r.get("oom")]
    if not rows:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for be in _backends(rows):
        pts = sorted((r for r in rows if r["backend"] == be), key=lambda r: r["N"])
        Ns = [r["N"] for r in pts]
        ax1.plot(Ns, [r["samples_per_s"] for r in pts], "o-",
                 label=LABEL.get(be, be), color=COLOR.get(be))
        ax2.plot(Ns, [r["peak_gib"] for r in pts], "o-",
                 label=LABEL.get(be, be), color=COLOR.get(be))
    for ax, title, ylab in [(ax1, "Training throughput", "samples / s"),
                            (ax2, "Peak memory", "GiB")]:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("sequence length $N$")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax1.set_yscale("log")
    fig.suptitle(f"Scaling on {data.get('device', 'GPU')} "
                 f"(d={data.get('dim')}, m={data.get('m')})", fontsize=10)
    fig.tight_layout()
    p = os.path.join(outdir, "scaling.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_mqar(data, outdir, fname, xlabel, title):
    rows = [r for r in data["results"] if not r.get("oom")]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for be in _backends(rows):
        pts = sorted((r for r in rows if r["backend"] == be), key=lambda r: r["axis_val"])
        ax.plot([r["axis_val"] for r in pts], [r["recall"] for r in pts], "o-",
                label=LABEL.get(be, be), color=COLOR.get(be))
    ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MQAR recall (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(outdir, fname)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_cifar(rows, outdir):
    if not rows:
        return None
    labels = [r["label"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    x = range(len(labels))
    cols = [COLOR.get(l, "gray") for l in labels]
    ax1.bar(x, [r["test_acc"] for r in rows], color=cols)
    ax1.set_xticks(list(x)); ax1.set_xticklabels([LABEL.get(l, l) for l in labels], rotation=15)
    ax1.set_ylabel("CIFAR-10 test acc (%)"); ax1.set_title("Accuracy")
    ax2.bar(x, [r.get("samples_per_s", 0) for r in rows], color=cols)
    ax2.set_xticks(list(x)); ax2.set_xticklabels([LABEL.get(l, l) for l in labels], rotation=15)
    ax2.set_ylabel("samples / s"); ax2.set_title(f"Throughput (N={rows[0].get('N_tokens','?')} tokens)")
    for ax in (ax1, ax2):
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, "cifar.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaling", default="scaling.json")
    ap.add_argument("--mqar_length", default="mqar_length.json")
    ap.add_argument("--mqar_capacity", default="mqar_capacity.json")
    ap.add_argument("--cifar", default="three_way_results.json")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    made = []
    if (d := _load(a.scaling)):
        made.append(fig_scaling(d, a.outdir))
    if (d := _load(a.mqar_length)):
        made.append(fig_mqar(d, a.outdir, "mqar_length.pdf", "sequence length $N$",
                             "MQAR recall vs context length"))
    if (d := _load(a.mqar_capacity)):
        made.append(fig_mqar(d, a.outdir, "mqar_capacity.pdf", "key-value pairs",
                             "MQAR recall vs capacity (rank limit)"))
    if (d := _load(a.cifar)):
        made.append(fig_cifar(d, a.outdir))

    made = [m for m in made if m]
    if made:
        print("wrote:")
        for m in made:
            print("  " + m)
    else:
        print("no input JSON found — run the experiments with --json first")


if __name__ == "__main__":
    main()
