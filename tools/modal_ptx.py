# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Compare what Triton emits against what our CUDA kernels emit, on an A100.

    modal run tools/modal_ptx.py

Triton overtakes the hand-written forward past N~256K. This dumps both sides'
generated code so the gap can be attributed rather than guessed at:

  * Triton: compiled metadata (registers, spills, shared memory, warps,
    pipeline stages) and the PTX instruction mix for each kernel.
  * FlashNystrom: SASS from the built extension, with per-kernel register and
    shared-memory usage and an instruction histogram.

The hypothesis under test is that the single-binary SM80 contract costs us
instruction selection that Triton, JIT-compiling for the actual target, is free
to use. Note the contract cannot cost ISA on an A100 specifically, where sm_80
IS native for both; what it can cost is SPECIALIZATION, since one binary must
pick tile shapes and pipeline depths that fit every target it serves.
"""
import pathlib
import sys

import modal

for _p in (str(pathlib.Path(__file__).resolve().parent), "/root/FlashNystrom/tools"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from modal_a100 import image                                     # noqa: E402

app = modal.App("flash-nystrom-ptx")
ptx_image = image.pip_install("triton")


@app.function(gpu="A100-80GB", image=ptx_image, timeout=60 * 40)
def dump_and_compare():
    import collections
    import glob
    import json
    import os
    import re
    import subprocess

    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    import torch
    sys.path.insert(0, "/root/FlashNystrom")
    from benchmarks import triton_nystrom as TN

    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  SMs={p.multi_processor_count}")

    B, H, N, D, M = 1, 8, 1048576, 64, 64
    q, k, v = [torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
               for _ in range(3)]
    TN.triton_nystrom_forward(q, k, v, num_landmarks=M)
    torch.cuda.synchronize()

    # ---- Triton, read from the on-disk cache (stable across releases) ------
    print("\n" + "=" * 72)
    print("TRITON compiled metadata (N=1048576)")
    print("=" * 72)
    ptx_dump = {}
    for jf in sorted(glob.glob("/tmp/triton_cache/**/*.json", recursive=True)):
        try:
            md = json.load(open(jf))
        except Exception:
            continue
        nm = md.get("name", "")
        if "kernel" not in nm:
            continue
        print(f"\n  {nm}")
        for key in ("num_warps", "num_stages", "num_ctas", "shared",
                    "n_regs", "n_spills", "maxntid"):
            if key in md:
                print(f"    {key:12s} {md[key]}")
        cand = glob.glob(os.path.join(os.path.dirname(jf), "*.ptx"))
        if cand:
            ptx_dump[nm] = open(cand[0]).read()

    print("\n" + "=" * 72)
    print("TRITON PTX instruction mix")
    print("=" * 72)
    KEYS = ("mma.sync", "wgmma", "ldmatrix", "cp.async", "ld.global.v4",
            "ld.global.v2", "ld.global.nc", "st.global.v4", "bar.sync",
            "ex2.approx", "fma.rn", "shfl.sync", "red.global", "atom.")
    for name, ptx in ptx_dump.items():
        print(f"\n  {name}  ({ptx.count(chr(10))} PTX lines)")
        for kk in KEYS:
            c = len(re.findall(re.escape(kk), ptx))
            if c:
                print(f"    {kk:16s} {c:5d}")
        shapes = collections.Counter(
            re.findall(r"mma\.sync\.aligned\.(m\d+n\d+k\d+)", ptx))
        if shapes:
            print(f"    mma shapes       {dict(shapes)}")

    # ---- our SASS ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("FLASHNYSTROM SASS")
    print("=" * 72)
    so = (glob.glob("/root/FlashNystrom/**/_C*.so", recursive=True)
          or glob.glob("/usr/local/lib/python3.11/site-packages/**/_C*.so",
                       recursive=True))
    if not so:
        print("  extension .so not found")
        return
    print(f"  {so[0]}")
    try:
        res = subprocess.run(["cuobjdump", "-res-usage", so[0]],
                             capture_output=True, text=True, timeout=900).stdout
        sass = subprocess.run(["cuobjdump", "-sass", so[0]],
                              capture_output=True, text=True, timeout=1200).stdout
    except Exception as e:
        print(f"  cuobjdump failed: {e}")
        return

    names = sorted(set(re.findall(r"Function : (\S+)", sass)))
    fwd = [n for n in names
           if not re.search(r"bwd|backward|_dq|_dk|_dv|_di|grad", n)]
    print(f"  {len(names)} kernels total, {len(fwd)} without a bwd marker")
    print("\n  forward-side kernel names:")
    for n in fwd[:14]:
        print(f"    {n[:76]}")

    reg_of, cur = {}, None
    for line in res.splitlines():
        m = re.search(r"Function (\S+)", line)
        if m:
            cur = m.group(1)
        m = re.search(r"REG:(\d+)", line)
        if m and cur:
            sh = re.search(r"SHARED:(\d+)", line)
            st = re.search(r"STACK:(\d+)", line)
            reg_of[cur] = (int(m.group(1)),
                           int(st.group(1)) if st else 0,
                           int(sh.group(1)) if sh else 0)

    per, cur = collections.defaultdict(collections.Counter), None
    for line in sass.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            continue
        if cur:
            m = re.search(r"/\*[0-9a-f]{4}\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9]*)",
                          line)
            if m:
                per[cur][m.group(1)] += 1

    print("\n  busiest forward kernels:")
    for kname in sorted(fwd, key=lambda n: -sum(per[n].values()))[:5]:
        c = per[kname]
        r = reg_of.get(kname)
        print(f"\n  {kname[:72]}")
        line = f"    instructions {sum(c.values())}"
        if r:
            line += f"   REG {r[0]}  STACK {r[1]}  SHARED {r[2]}"
        print(line)
        for op, cnt in c.most_common(12):
            print(f"      {op:10s} {cnt:5d}")


@app.local_entrypoint()
def run_ptx_dump():
    dump_and_compare.remote()


@app.function(gpu="A100-80GB", image=ptx_image, timeout=60 * 40)
def profile_forward():
    """Where the forward actually spends its time, per kernel, at N=1M.

    Static instruction counts say nothing about runtime: the Newton-Schulz
    pinverse is the largest kernel in the binary and does 64x64 of work. This
    is the dynamic measurement, for FlashNystrom and for Triton side by side.
    """
    import torch
    from torch.profiler import profile, ProfilerActivity
    sys.path.insert(0, "/root/FlashNystrom")
    from benchmarks.triton_nystrom import triton_nystrom_forward
    from flash_nystrom import flash_nystrom_attention as fn

    B, H, N, D, M = 1, 8, 1048576, 64, 64
    q, k, v = [torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
               for _ in range(3)]
    print(f"N={N}  bytes moved by one pass over q: "
          f"{B*H*N*D*2/1e9:.2f} GB\n")

    for label, f in (("FlashNystrom (tc-pinv)",
                      lambda: fn(q, k, v, num_landmarks=M, kappa_star=0.0,
                                 use_tc_pinv=True)),
                     ("Triton", lambda: triton_nystrom_forward(q, k, v,
                                                               num_landmarks=M))):
        for _ in range(5):
            f()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                f()
            torch.cuda.synchronize()
        evs = [e for e in prof.key_averages() if e.self_device_time_total > 0]
        evs.sort(key=lambda e: -e.self_device_time_total)
        total = sum(e.self_device_time_total for e in evs)
        print("=" * 76)
        print(f"{label}   total {total/10/1000:.3f} ms/iter")
        print("=" * 76)
        print(f"  {'kernel':<50}{'ms/iter':>10}{'share':>8}")
        for e in evs[:10]:
            ms = e.self_device_time_total / 10 / 1000
            print(f"  {e.key[:48]:<50}{ms:10.3f}{100*e.self_device_time_total/total:7.1f}%")
        print()


@app.function(gpu="A100-80GB", image=ptx_image, timeout=60 * 40)
def anatomy():
    """Side by side: what each implementation actually does, kernel by kernel.

    Pairs the comparable stages, counts global-memory passes over the (N, D)
    tensors, and prints the inner loop of each side's generated code. The
    question this answers is not "who is faster" but "what is different".
    """
    import os, re, glob
    os.environ["TRITON_CACHE_DIR"] = "/tmp/anat"
    import torch
    from torch.profiler import profile, ProfilerActivity
    sys.path.insert(0, "/root/FlashNystrom")
    from benchmarks.triton_nystrom import triton_nystrom_forward
    from flash_nystrom import flash_nystrom_attention as fn

    B, H, N, D, M = 1, 8, 1048576, 64, 64
    one_pass_gb = B * H * N * D * 2 / 1e9
    q, k, v = [torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
               for _ in range(3)]
    print(f"N={N}  one pass over one (N,D) tensor = {one_pass_gb:.2f} GB")
    print(f"A100 achievable ~1550 GB/s -> {one_pass_gb/1.55*1000:.2f} ms per pass\n")

    def prof(f, label):
        for _ in range(5):
            f()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(10):
                f()
            torch.cuda.synchronize()
        evs = [e for e in pr.key_averages() if e.self_device_time_total > 0]
        evs.sort(key=lambda e: -e.self_device_time_total)
        tot = sum(e.self_device_time_total for e in evs) / 10 / 1000
        print(f"{label}: {tot:.3f} ms")
        for e in evs[:9]:
            nm = e.key.replace("void flash_nystrom::", "").replace("void ", "")[:46]
            ms = e.self_device_time_total / 10 / 1000
            print(f"   {nm:48s} {ms:7.3f} ms   {ms/(one_pass_gb/1.55):5.2f} passes")
        print()
        return tot

    prof(lambda: fn(q, k, v, num_landmarks=M, kappa_star=0.0, use_tc_pinv=True),
         "FLASHNYSTROM")
    prof(lambda: triton_nystrom_forward(q, k, v, num_landmarks=M), "TRITON")

    # ---- the generated inner loops ----------------------------------------
    print("=" * 74)
    print("TRITON PTX: the main loop of _kernel_p1_z (our kernel1_fused_tc twin)")
    print("=" * 74)
    for pf in sorted(glob.glob("/tmp/anat/**/*.ptx", recursive=True)):
        ptx = open(pf).read()
        if "p1_z" not in ptx:
            continue
        lines = ptx.splitlines()
        # the hot block: from the first mma to the last
        idx = [i for i, l in enumerate(lines) if "mma.sync" in l]
        if not idx:
            continue
        lo, hi = max(0, idx[0] - 12), min(len(lines), idx[0] + 26)
        for l in lines[lo:hi]:
            print("   " + l.strip()[:100])
        print(f"\n   ... {len(idx)} mma.sync total in this kernel")
        break
