# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Regenerate the bidirectional-operator latency table on an A100.

    modal run tools/modal_bidir.py

The table in the paper was assembled from two campaigns, and its fused
linear-attention column was timed on the BARE flash_bla kernel: no feature map,
no normalizer, no division, while every other arm was timed as a complete
operator. That understates linear attention and the abstract's ratio rests on
it. benchmarks/bench_bidir_latency.py times every arm end to end; this runs it
on one machine so every column comes from a single campaign, which also picks
up the four forward optimizations that landed after the original numbers.

Both baseline kernels are installed here: flash_bla (Triton, fused
bidirectional linear attention) and flash_attn (FlashAttention-2's fused
windowed kernel for sliding window). An arm whose kernel is missing reports
n/a rather than silently falling back to an unfused stand-in, so a row can
never mix a fused and an unfused measurement.
"""
import pathlib
import sys

import modal

for _p in (str(pathlib.Path(__file__).resolve().parent), "/root/FlashNystrom/tools"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from modal_a100 import image                                     # noqa: E402

app = modal.App("flash-nystrom-bidir")

# flash_bla is pure Triton, so it is a clone plus an editable install. flash-attn
# ships prebuilt wheels matched to (torch, cuda, python, abi) and its setup.py
# fetches one before falling back to a source build; MAX_JOBS caps that fallback
# so a cache miss degrades to slow rather than to an OOM-killed builder.
bidir_image = (
    image.pip_install("triton", "einops", "packaging")
    .run_commands(
        "git clone --depth 1 "
        "https://github.com/fla-org/flash-bidirectional-linear-attention.git /tmp/fbla",
        "pip install -e /tmp/fbla/. || echo FBLA_INSTALL_FAILED",
        "MAX_JOBS=4 pip install flash-attn --no-build-isolation "
        "|| echo FLASH_ATTN_INSTALL_FAILED",
    )
)


@app.function(gpu="A100-80GB", image=bidir_image, timeout=60 * 60)
def bench():
    import subprocess
    sys.path.insert(0, "/root/FlashNystrom")

    import torch
    print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    for mod in ("flash_bla", "flash_attn"):
        try:
            __import__(mod)
            print(f"  {mod:12s} present")
        except Exception as e:
            print(f"  {mod:12s} MISSING ({type(e).__name__}: {str(e)[:60]})")
    print()

    cmd = [sys.executable, "benchmarks/bench_bidir_latency.py",
           "--lens", "131072", "262144", "524288", "1048576",
           "--out", "/tmp/bidir.csv"]
    subprocess.run(cmd, cwd="/root/FlashNystrom", check=False)

    print("\n=== CSV ===")
    try:
        print(open("/tmp/bidir.csv").read())
    except FileNotFoundError:
        print("no csv written")


@app.local_entrypoint()
def main():
    bench.remote()
