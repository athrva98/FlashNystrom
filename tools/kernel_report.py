# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

"""Per-kernel resource and occupancy report.

Run after a build to see, for every kernel this extension ships:
    threads/CTA, regs/thread, dynamic SMEM, achieved blocks/SM,
    and which of {threads, registers, SMEM, hardware blocks/SM}
    is the binding constraint at the chosen launch configuration.

The numbers come from cudaOccupancyMaxActiveBlocksPerMultiprocessor and
cudaFuncGetAttributes via the pybind helper exposed by flash_nystrom._C.
This is the actual answer the runtime would give the scheduler, not a
manual estimate.

Usage:
    python tools/kernel_report.py                    # m=64, D=128, niter=6, FP16
    python tools/kernel_report.py --m 32 --D 64      # smaller config
    python tools/kernel_report.py --dtype bfloat16
    python tools/kernel_report.py --dtype fp32       # FP32 scalar paths only
"""
import argparse
import sys

try:
    import torch
    import flash_nystrom._C as _C
except ImportError as e:
    sys.exit(f"flash_nystrom._C not built. pip install -e . --no-build-isolation\n{e}")


def fmt_bytes(b: int) -> str:
    if b >= 1024 * 1024:
        return f"{b / (1024*1024):.1f} MB"
    if b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=64, help="num_landmarks")
    p.add_argument("--D", type=int, default=128, help="head_dim")
    p.add_argument("--niter", type=int, default=6, help="newton_iter")
    p.add_argument("--dtype", default="half", choices=["half", "bfloat16", "fp32"])
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    dev = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(dev)
    lim = _C.query_sm_limits()

    print(f"Device: {prop.name}  (SM {prop.major}.{prop.minor},  "
          f"{prop.multi_processor_count} SMs)")
    print(f"  Per-SM limits: max_threads={lim.max_threads_per_sm}, "
          f"max_blocks={lim.max_blocks_per_sm}, "
          f"max_regs={lim.max_regs_per_sm}, "
          f"max_smem={fmt_bytes(lim.max_smem_per_sm_bytes)}")
    print(f"  Register allocation granularity: {lim.reg_alloc_unit} "
          f"32-bit registers per chunk")
    print(f"  Probe config: m={args.m}, D={args.D}, "
          f"newton_iter={args.niter}, dtype={args.dtype}")
    print()

    rows = _C.probe_occupancy(args.m, args.D, args.niter, args.dtype)

    # Column widths
    name_w = max(len(r.kernel_name) for r in rows)
    name_w = max(name_w, 30)

    hdr = (
        f"{'kernel':<{name_w}} | "
        f"{'tpb':>4} {'regs':>5} {'dyn SMEM':>9} | "
        f"{'thr':>3} {'reg':>3} {'smem':>4} {'hw':>3} | "
        f"{'blk/SM':>6} {'warp/SM':>7} | "
        f"binding"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        smem_str = fmt_bytes(r.dynamic_smem_bytes) if r.dynamic_smem_bytes else "0"
        # Cap each axis at hardware block count for display
        hw = lim.max_blocks_per_sm
        b_thr = min(r.blocks_by_threads, hw)
        b_reg = min(r.blocks_by_regs, hw)
        b_smm = min(r.blocks_by_smem, hw)
        b_hw  = r.blocks_by_hardware
        # Highlight binding cell with brackets
        def cell(v, name):
            return f"[{v}]" if name == r.binding_constraint else f" {v} "
        print(
            f"{r.kernel_name:<{name_w}} | "
            f"{r.threads_per_block:>4} {r.regs_per_thread:>5} {smem_str:>9} | "
            f"{cell(b_thr, 'threads/SM')} "
            f"{cell(b_reg, 'registers')} "
            f"{cell(b_smm, 'SMEM')} "
            f"{cell(b_hw,  'hw blocks/SM')} | "
            f"{r.max_blocks_per_sm:>6} "
            f"{r.max_warps_per_sm:>7} | "
            f"{r.binding_constraint}"
        )

    # Summary: kernels at lowest occupancy first
    print()
    print("Lowest-occupancy kernels (most likely worth optimizing):")
    sorted_rows = sorted(rows, key=lambda r: (r.max_blocks_per_sm, r.kernel_name))
    for r in sorted_rows[:5]:
        # Compute warp utilization as fraction of theoretical max
        max_warps = lim.max_threads_per_sm // 32
        ach_warps = r.max_warps_per_sm
        pct = 100.0 * ach_warps / max_warps if max_warps else 0.0
        print(f"  {r.kernel_name}: "
              f"{r.max_blocks_per_sm} block/SM, {ach_warps}/{max_warps} warps ({pct:.0f}%), "
              f"binding={r.binding_constraint}, "
              f"regs={r.regs_per_thread}, smem={fmt_bytes(r.dynamic_smem_bytes)}")


if __name__ == "__main__":
    main()
