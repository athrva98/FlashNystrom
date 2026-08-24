# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Peak memory: FlashNystrom vs the same algorithm unfused, vs exact attention.

    python benchmarks/bench_memory.py --out mem.csv

The paper's premise is that a framework-level Nystrom attention MATERIALIZES its
intermediates: the two (N, m) probability matrices and the (N, m) and (m, N)
partial products all reach HBM, and the backward keeps them alive. Fusing the
chain means they never exist outside registers and shared memory. That is a
memory claim, and until now the paper asserted it without measuring it.

Reports peak allocation for forward-only and for forward+backward, since the
backward is where the unfused path pays most: autograd holds every intermediate
until it is consumed.

Exact attention is included as the ceiling. Its O(N^2) score matrix is what both
Nystrom paths avoid, which is a different and much larger saving.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def peak_gib(fn, *args, backward: bool):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        out = fn(*args)
        if backward:
            out.sum().backward()
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    except torch.cuda.OutOfMemoryError:
        peak = float("inf")
    for a in args:
        if torch.is_tensor(a):
            a.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    return peak


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lens", nargs="+", type=int,
                    default=[8192, 16384, 32768, 65536, 131072])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head_dim", type=int, default=64)
    ap.add_argument("--landmarks", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("needs a GPU")
        return 1
    from flash_nystrom import flash_nystrom_attention as fn
    from flash_nystrom.reference import nystrom_attention_reference as ref
    import torch.nn.functional as F

    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  {p.total_memory/2**30:.0f} GiB  "
          f"B={a.batch} H={a.heads} D={a.head_dim} m={a.landmarks} fp16")
    print("\npeak GiB above the q/k/v baseline; 'oom' = did not fit\n")
    hdr = ("N", "FN fwd", "ref fwd", "exact fwd", "FN f+b", "ref f+b", "exact f+b",
           "ref/FN")
    print("".join(f"{h:>11s}" for h in hdr))

    rows = []
    for N in a.lens:
        q, k, v = [torch.randn(a.batch, a.heads, N, a.head_dim, device="cuda",
                               dtype=torch.float16, requires_grad=True)
                   for _ in range(3)]
        cells = []
        for backward in (False, True):
            cells.append(peak_gib(lambda x, y, z: fn(x, y, z,
                                                     num_landmarks=a.landmarks,
                                                     kappa_star=0.0),
                                  q, k, v, backward=backward))
            cells.append(peak_gib(lambda x, y, z: ref(x, y, z, a.landmarks, 6,
                                                      None, 0, kappa_star=0.0),
                                  q, k, v, backward=backward))
            cells.append(peak_gib(F.scaled_dot_product_attention, q, k, v,
                                  backward=backward))
        fnf, reff, exf, fnb, refb, exb = cells
        ratio = refb / fnb if fnb and fnb != float("inf") else float("nan")
        f = lambda x: "oom" if x == float("inf") else f"{x:.2f}"
        print(f"{N:>11d}" + "".join(f"{f(c):>11s}" for c in
                                    (fnf, reff, exf, fnb, refb, exb))
              + f"{ratio:>10.1f}x")
        rows.append((N, fnf, reff, exf, fnb, refb, exb, ratio))
        del q, k, v
        gc.collect()
        torch.cuda.empty_cache()

    if a.out:
        with open(a.out, "w") as fh:
            fh.write("N,fn_fwd,ref_fwd,exact_fwd,fn_fwdbwd,ref_fwdbwd,"
                     "exact_fwdbwd,ref_over_fn\n")
            for r in rows:
                fh.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x)
                                  for x in r) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
