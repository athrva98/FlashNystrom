# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Render the README benchmark tables as plots.

The numbers below are the measured values shown in the README tables (RTX 5060
via benchmarks/bench_fwd_bwd.py; A100/H100 via tools/modal_a100.py). They are
hardcoded here on purpose so the figures are reproducible without a GPU and stay
in lockstep with the tables. Re-run after updating the tables:

    python benchmarks/plot_benchmarks.py

Writes PNGs to assets/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS, exist_ok=True)

FN_C, REF_C, SDPA_C = "#1f77b4", "#7f7f7f", "#d62728"
FA2_C, FA3_C = "#ff7f0e", "#9467bd"
A100_C, H100_C = "#2ca02c", "#1f77b4"


def _style(ax, title, xlabel="sequence length N", ylabel="fwd+bwd latency (ms)"):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)


def _save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 1. RTX 5060: FlashNystrom vs cuBLAS-Nystrom (Ref) vs SDPA (exact attention).
# ---------------------------------------------------------------------------
def plot_5060():
    N = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    fn = [1.27, 1.18, 0.71, 0.70, 0.74, 0.81, 0.99, 1.83, 2.86, 6.16, 11.76, 20.59]
    ref = [5.54, 5.59, 5.29, 5.01, 5.18, 5.25, 6.03, 5.63, 7.09, 11.12, 21.46, 56.81]
    sdpa = [0.52, 0.47, 0.25, 0.41, 1.24, 4.56, 17.59, 73.62, 287.52, 1195.12, 4886.12, 19882.17]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(N, sdpa, "-o", color=SDPA_C, lw=2, ms=5, label="SDPA (exact, O(N$^2$))")
    ax.plot(N, ref, "-s", color=REF_C, lw=2, ms=5, label="cuBLAS Nystrom (Ref)")
    ax.plot(N, fn, "-^", color=FN_C, lw=2.2, ms=6, label="FlashNystrom")
    _style(ax, "RTX 5060 Laptop, FP16, B=1 H=4 D=64, m=32  (fwd+bwd)")
    fig.text(0.5, -0.02,
             "SDPA explodes as O(N$^2$); FlashNystrom scales ~O(N). Crossover ~N=1-2K.",
             ha="center", fontsize=8.5, color="#444")
    _save(fig, "latency_5060.png")


# ---------------------------------------------------------------------------
# 2. A100 + H100: FlashNystrom vs the same algorithm in cuBLAS.
# ---------------------------------------------------------------------------
def plot_datacenter_cublas():
    hb = {
        "N": [4096, 16384, 65536, 131072],
        "a100_fn": [6.97, 16.63, 57.58, 108.96], "a100_cub": [7.36, 22.01, 82.42, 193.02],
        "h100_fn": [3.59, 8.56, 27.82, 53.56],   "h100_cub": [5.60, 13.04, 49.61, 101.88],
    }
    lc = {
        "N": [65536, 131072, 262144, 524288, 1048576, 2097152],
        "a100_fn": [4.71, 8.06, 14.52, 27.47, 52.95, 105.34],
        "a100_cub": [6.60, 8.22, 17.90, 40.86, 81.48, 162.05],
        "h100_fn": [3.34, 5.62, 10.38, 19.91, 38.92, 77.01],
        "h100_cub": [5.55, 5.65, 8.77, 21.40, 43.80, 87.02],
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, d, title in (
        (axes[0], hb, "High batch x head (B=4 H=16 D=128 m=64)"),
        (axes[1], lc, "Long context (B=1 H=4 D=64 m=32)"),
    ):
        ax.plot(d["N"], d["a100_cub"], "--s", color=A100_C, lw=1.6, ms=5, alpha=0.7, label="cuBLAS A100")
        ax.plot(d["N"], d["a100_fn"], "-^", color=A100_C, lw=2.2, ms=6, label="FN A100")
        ax.plot(d["N"], d["h100_cub"], "--s", color=H100_C, lw=1.6, ms=5, alpha=0.7, label="cuBLAS H100")
        ax.plot(d["N"], d["h100_fn"], "-^", color=H100_C, lw=2.2, ms=6, label="FN H100")
        _style(ax, title)
    fig.suptitle("FlashNystrom vs cuBLAS Nystrom (same algorithm) - lower is better",
                 fontsize=12)
    _save(fig, "latency_datacenter_cublas.png")


# ---------------------------------------------------------------------------
# 3. H100: FlashNystrom (approx O(mN)) vs FlashAttention-2/3 (exact O(N^2)).
# ---------------------------------------------------------------------------
def plot_flashattention():
    hb = {
        "N": [4096, 16384, 65536, 131072],
        "fn": [3.61, 8.62, 28.0, 53.9],
        "fa2": [5.90, 91.5, 1469, 5865], "fa3": [3.47, 50.4, 835, 3395],
    }
    lc = {
        "N": [16384, 65536, 131072, 262144, 524288, 1048576],
        "fn": [1.46, 3.22, 5.89, 10.71, 20.57, 39.90],
        "fa2": [3.08, 48.9, 202, 806, 3295, 13338],
        "fa3": [1.74, 32.2, 123, 478, 1958, 7865],
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, d, title in (
        (axes[0], hb, "High batch x head (B=4 H=16 D=128 m=64)"),
        (axes[1], lc, "Long context (B=1 H=4 D=64 m=32)"),
    ):
        ax.plot(d["N"], d["fa2"], "-o", color=FA2_C, lw=2, ms=5, label="FlashAttention-2 (exact)")
        ax.plot(d["N"], d["fa3"], "-o", color=FA3_C, lw=2, ms=5, label="FlashAttention-3 (exact)")
        ax.plot(d["N"], d["fn"], "-^", color=FN_C, lw=2.4, ms=6, label="FlashNystrom (approx)")
        _style(ax, title)
    fig.suptitle("FlashNystrom O(m.N) vs FlashAttention O(N^2), H100  (fwd+bwd, lower is better)",
                 fontsize=12)
    _save(fig, "latency_flashattention_h100.png")


# ---------------------------------------------------------------------------
# 4. Per-kernel SMEM / occupancy (RTX 5060, m=64 D=128 FP16).
# ---------------------------------------------------------------------------
def plot_smem():
    # (kernel, dyn SMEM KB, blocks/SM, binding)
    rows = [
        ("landmark_kernel", 8, 1, "threads"),
        ("kernel1_fused_tc", 32, 3, "SMEM"),
        ("kernel3_fused_tc", 32, 3, "registers"),
        ("kernel1_bwd_tc", 48, 2, "SMEM"),
        ("kernel3_bwd_tc", 40, 2, "registers"),
        ("compute_dk2inv_tc", 64, 1, "SMEM"),
        ("kernel2_inv", 96, 1, "SMEM"),
        ("ns_bwd_step", 96, 1, "SMEM"),
    ]
    color = {"SMEM": "#1f77b4", "registers": "#d62728", "threads": "#2ca02c"}
    names = [r[0] for r in rows][::-1]
    smem = [r[1] for r in rows][::-1]
    blks = [r[2] for r in rows][::-1]
    cols = [color[r[3]] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, smem, color=cols)
    ax.axvline(100, color="#888", ls="--", lw=1.2)
    ax.text(100, len(names) - 0.4, " 100 KB/SM (consumer)", color="#666", fontsize=8, va="top")
    for b, n in zip(bars, blks):
        ax.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2,
                f"{n} blk/SM", va="center", fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color.values()]
    ax.legend(handles, [f"bound by {k}" for k in color], fontsize=9, loc="lower right")
    ax.set_xlabel("dynamic SMEM per block (KB)", fontsize=10)
    ax.set_title("Per-kernel SMEM and occupancy (RTX 5060, 100 KB/SM, m=64 D=128 FP16)",
                 fontsize=11)
    ax.set_xlim(0, 115)
    _save(fig, "occupancy_smem.png")


if __name__ == "__main__":
    plot_5060()
    plot_datacenter_cublas()
    plot_flashattention()
    plot_smem()
    print("done")
