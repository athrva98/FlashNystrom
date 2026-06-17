# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Adaptive batch sizing + training-step profiling, to saturate the GPU.

The other harnesses hardcode a batch size (128 / 256), which under-utilizes a
large GPU at short sequence lengths and OOMs at long ones. ``search_and_profile``
doubles the batch until a fwd+bwd+opt step OOMs, timing each batch that fits, and
returns the largest fitting batch's metrics (median step time, throughput, peak
memory).

Two robustness rules learned the hard way: (1) all timing happens *before* the
terminal OOM, so no CUDA op runs on a possibly-poisoned context; (2) callers
should still run each (config) in its own **subprocess** (see
``profile_scaling.py``), because a hard OOM can poison the CUDA context for the
rest of the process, which no in-process empty_cache reliably recovers.
"""
from __future__ import annotations

import torch


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
    )


def _time_trial(make_trial, bs: int, warmup: int, iters: int) -> dict:
    """Build at batch bs, warm up, and time iters steps. Raises CUDA OOM if it
    doesn't fit. Returns metrics for this batch."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run = make_trial(bs)
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    times = []
    for _ in range(iters):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    med = times[len(times) // 2]
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    del run
    return {"batch": bs, "step_ms": med, "samples_per_s": bs * 1000.0 / med,
            "peak_gib": peak}


def search_and_profile(make_trial, lo: int = 2, cap: int = 8192,
                       warmup: int = 5, iters: int = 20, verbose: bool = False):
    """Double the batch from ``lo`` to ``cap`` until OOM, timing each fit.

    make_trial(batch) -> run(): builds a fresh model+optimizer+inputs and returns
    a zero-arg closure running one fwd+bwd+opt step. Returns the metrics dict for
    the largest batch that fit, or None if even ``lo`` OOMs. All CUDA work occurs
    before the terminal OOM (the failing batch is never used downstream)."""
    best = None
    bs = lo
    while bs <= cap:
        try:
            best = _time_trial(make_trial, bs, warmup, iters)
            if verbose:
                print(f"      batch={bs:<6d} {best['step_ms']:.2f} ms  "
                      f"{best['peak_gib']:.2f} GiB", flush=True)
            bs *= 2
        except BaseException as e:  # noqa: BLE001 - OOM may be RuntimeError
            if _is_oom(e):
                if verbose:
                    print(f"      batch={bs:<6d} OOM (stop)", flush=True)
                break
            raise
        finally:
            torch.cuda.empty_cache()
    return best
