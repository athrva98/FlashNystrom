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
    fn = [1.47, 0.76, 0.71, 0.66, 0.73, 0.77, 0.98, 1.67, 3.23, 6.26, 11.96, 23.40]
    ref = [5.79, 5.88, 5.38, 5.46, 6.19, 4.99, 5.89, 6.49, 6.09, 11.06, 21.45, 48.59]
    sdpa = [0.31, 0.33, 0.23, 0.41, 1.24, 4.56, 17.73, 74.08, 290.56, 1198.71, 4893.45, 19694.80]
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
        "a100_fn": [6.79, 17.66, 60.94, 116.78], "a100_cub": [7.33, 22.45, 84.66, 198.26],
        "h100_fn": [3.67, 8.66, 27.90, 53.67],   "h100_cub": [6.32, 12.93, 49.56, 101.81],
    }
    lc = {
        "N": [65536, 131072, 262144, 524288, 1048576, 2097152],
        "a100_fn": [4.73, 8.25, 15.05, 28.72, 55.62, 110.75],
        "a100_cub": [6.65, 8.25, 18.21, 41.93, 83.71, 166.69],
        "h100_fn": [3.25, 5.71, 10.58, 20.39, 40.00, 79.15],
        "h100_cub": [6.30, 6.37, 8.72, 21.33, 43.70, 86.95],
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
        "fn": [3.69, 8.75, 28.1, 54.2],
        "fa2": [5.89, 93.1, 1477, 5901], "fa3": [3.38, 51.6, 837, 3380],
    }
    lc = {
        "N": [16384, 65536, 131072, 262144, 524288, 1048576],
        "fn": [1.39, 3.27, 5.95, 10.5, 21.4, 39.5],
        "fa2": [3.09, 49.3, 203, 808, 3319, 13408],
        "fa3": [1.78, 32.2, 123, 483, 1966, 7904],
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
        ("kernel3_bwd_tc", 40, 2, "SMEM"),
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
