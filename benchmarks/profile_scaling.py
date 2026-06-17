# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Training-step scaling sweep: backend x sequence length, at the max batch the
GPU holds. Reports throughput (samples/s, tokens/s) and peak memory, so the
flash_nystrom-vs-full-attention crossover (and where sdpa OOMs) is measured.

Each (backend, N) runs in its OWN subprocess: a hard OOM poisons the CUDA context
for that process, so isolating configs keeps one OOM from killing the sweep. The
parent orchestrates; the ``--worker`` mode profiles a single config and prints a
``RESULT {...}`` line.

    python benchmarks/profile_scaling.py
    python benchmarks/profile_scaling.py --backends sdpa flash_nystrom --Ns 1024 4096 16384
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))      # .../benchmarks
_REPO = os.path.dirname(_HERE)                          # repo root
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import torch
import torch.nn as nn

from paper.mqar.model import build_attention
from autobatch import search_and_profile


class _Block(nn.Module):
    def __init__(self, dim, heads, backend, m):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = build_attention(backend, dim, heads, num_landmarks=m)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, backend, m):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim, heads, backend, m) for _ in range(depth)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def trial_factory(backend, N, dim, depth, heads, m, dtype):
    def make_trial(bs):
        model = Encoder(dim, depth, heads, backend, m).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(bs, N, dim, device="cuda", dtype=dtype)

        def run():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                out = model(x)
                loss = out.float().pow(2).mean()
            loss.backward()
            opt.step()

        return run
    return make_trial


def run_worker(a):
    mt = trial_factory(a.backend, a.N, a.dim, a.depth, a.heads, a.m, torch.bfloat16)
    res = search_and_profile(mt, lo=a.lo, cap=a.cap, warmup=5, iters=a.iters,
                             verbose=a.verbose)
    print("RESULT " + json.dumps(res if res else {}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+",
                    default=["sdpa", "flash_nystrom", "nystrom_reference"])
    ap.add_argument("--Ns", nargs="+", type=int,
                    default=[256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)  # head_dim = 64
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--lo", type=int, default=2)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--verbose", action="store_true")
    # worker mode
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--backend")
    ap.add_argument("--N", type=int)
    a = ap.parse_args()

    if a.worker:
        run_worker(a)
        return

    name = torch.cuda.get_device_name()
    print(f"GPU: {name} | dim={a.dim} depth={a.depth} heads={a.heads} "
          f"head_dim={a.dim // a.heads} m={a.m} dtype=bf16")
    print(f"{'N':>6} {'backend':>18} {'maxbatch':>9} {'step_ms':>9} "
          f"{'samp/s':>10} {'ktok/s':>10} {'peak_GiB':>9}")
    print("-" * 80)
    for N in a.Ns:
        for backend in a.backends:
            cmd = [sys.executable, __file__, "--worker", "--backend", backend,
                   "--N", str(N), "--dim", str(a.dim), "--depth", str(a.depth),
                   "--heads", str(a.heads), "--m", str(a.m), "--lo", str(a.lo),
                   "--cap", str(a.cap), "--iters", str(a.iters)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            res = None
            for line in proc.stdout.splitlines():
                if line.startswith("RESULT "):
                    res = json.loads(line[len("RESULT "):])
            if not res:
                tail = (proc.stderr.strip().splitlines() or ["(no batch fit)"])[-1][:60]
                print(f"{N:>6} {backend:>18} {'OOM':>9}   {tail}")
                continue
            ktok = res["samples_per_s"] * N / 1000.0
            print(f"{N:>6} {backend:>18} {res['batch']:>9} {res['step_ms']:>9.2f} "
                  f"{res['samples_per_s']:>10.1f} {ktok:>10.1f} {res['peak_gib']:>9.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
