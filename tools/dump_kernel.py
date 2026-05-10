# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

"""Dump PTX and SASS for a specific kernel from the built .pyd.

Usage:
    python tools/dump_kernel.py kernel3_bwd_tc            # dumps both PTX and SASS
    python tools/dump_kernel.py kernel3_bwd_tc --ptx-only
    python tools/dump_kernel.py kernel3_bwd_tc --sass-only
    python tools/dump_kernel.py --list                    # list all kernels
    python tools/dump_kernel.py kernel3_bwd_tc --out-dir build/disasm

The tool calls cuobjdump on the .pyd file to extract per-kernel disassembly.
PTX is the high-level intermediate. SASS is the actual hardware instruction
stream the GPU executes. Read SASS to see register allocation, spill code,
and instruction-level scheduling. Read PTX when SASS is too dense and you
want a higher-level view of the algorithm structure.

Filters: pass a substring of the kernel name to match. Templates produce
multiple instantiations (FP16, BF16, D=64, D=128, etc.); the substring
match catches all of them by default. Add --dtype half / --dtype bfloat16
or --headdim 64 / --headdim 128 to narrow.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


CUOBJDUMP_CANDIDATES = [
    Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/cuobjdump.exe"),
    Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/cuobjdump.exe"),
    Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/cuobjdump.exe"),
    Path("/usr/local/cuda/bin/cuobjdump"),
]


def find_cuobjdump():
    on_path = shutil.which("cuobjdump")
    if on_path:
        return Path(on_path)
    for c in CUOBJDUMP_CANDIDATES:
        if c.exists():
            return c
    sys.exit("cuobjdump not found. Install CUDA toolkit and add it to PATH.")


def find_pyd():
    candidates = list(Path("flash_nystrom").glob("_C*.pyd")) + \
                 list(Path("flash_nystrom").glob("_C*.so")) + \
                 list(Path("build").rglob("flash_nystrom/_C*.pyd")) + \
                 list(Path("build").rglob("flash_nystrom/_C*.so"))
    if not candidates:
        sys.exit("Built extension not found. Run pip install -e . --no-build-isolation")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# CUDA mangles names with leading _ZN13flash_nystrom + length-prefixed pieces.
# The kernel name itself appears verbatim. We just substring-match.
def list_kernels(cuobjdump: Path, pyd: Path):
    out = subprocess.run(
        [str(cuobjdump), "--dump-resource-usage", str(pyd)],
        check=True, capture_output=True, text=True
    ).stdout
    names = []
    for line in out.splitlines():
        # Lines look like: " Function _ZN13flash_nystrom14kernel3_bwd_tc...:"
        m = re.match(r"\s*Function\s+(_ZN13flash_nystrom\S+):", line)
        if m:
            names.append(m.group(1))
    return names


def demangle(mangled: str) -> str:
    """Return a short human-readable form of the kernel name."""
    # Strip leading _ZN13flash_nystrom and trailing argument signature.
    s = mangled
    s = re.sub(r"^_ZN\d+flash_nystrom", "", s)
    # Pull out the readable kernel name (length-prefixed identifier)
    m = re.match(r"(\d+)(.*)", s)
    if m:
        n = int(m.group(1))
        s = m.group(2)
        name = s[:n]
    else:
        name = s
    return name


def filter_kernels(kernels, pattern, dtype, headdim):
    out = []
    for k in kernels:
        if pattern not in demangle(k) and pattern not in k:
            continue
        if dtype:
            if dtype == "half" and "half_t" not in k:
                continue
            if dtype == "bfloat16" and "bfloat16_t" not in k:
                continue
        if headdim:
            # Look for "Li64" or "Li128" inside the mangled name
            if f"Li{headdim}" not in k:
                continue
        out.append(k)
    return out


def dump_one(cuobjdump: Path, pyd: Path, kernel: str, mode: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    short = demangle(kernel)
    safe = re.sub(r"[^A-Za-z0-9_]", "_", short)[:80]

    if mode in ("ptx", "both"):
        ptx_path = out_dir / f"{safe}.ptx"
        result = subprocess.run(
            [str(cuobjdump), "--function", kernel, "--dump-ptx", str(pyd)],
            capture_output=True, text=True
        )
        ptx_path.write_text(result.stdout)
        ptx_lines = result.stdout.count("\n")
        print(f"  PTX:  {ptx_path}  ({ptx_lines} lines)")

    if mode in ("sass", "both"):
        sass_path = out_dir / f"{safe}.sass"
        result = subprocess.run(
            [str(cuobjdump), "--function", kernel, "--dump-sass", str(pyd)],
            capture_output=True, text=True
        )
        sass_path.write_text(result.stdout)
        sass_lines = result.stdout.count("\n")
        # Quick stats: count SASS instructions, register spill markers
        body = "\n".join(line for line in result.stdout.splitlines()
                         if line.startswith("        /*") or
                         (line.strip().startswith("/*") and "*/" in line))
        # Count instructions roughly. Each /* offset */ at line start is an op.
        ninstr = sum(1 for line in result.stdout.splitlines()
                     if re.match(r"\s*/\*[0-9a-fA-F]+\*/", line))
        nspill_st = result.stdout.count("STL")  # store local
        nspill_ld = result.stdout.count("LDL")  # load local
        print(f"  SASS: {sass_path}  ({sass_lines} lines, ~{ninstr} instructions, "
              f"{nspill_st} STL spill stores, {nspill_ld} LDL spill loads)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kernel", nargs="?", help="kernel name (substring match)")
    p.add_argument("--list", action="store_true", help="list all kernels and exit")
    p.add_argument("--ptx-only",  action="store_true")
    p.add_argument("--sass-only", action="store_true")
    p.add_argument("--dtype",   choices=["half", "bfloat16"], default=None)
    p.add_argument("--headdim", type=int, choices=[64, 128], default=None)
    p.add_argument("--out-dir", default="build/disasm",
                   help="where to write the .ptx and .sass files")
    args = p.parse_args()

    cuobjdump = find_cuobjdump()
    pyd = find_pyd()
    print(f"cuobjdump: {cuobjdump}")
    print(f"binary:    {pyd}")
    print()

    all_kernels = list_kernels(cuobjdump, pyd)
    if args.list:
        for k in sorted(all_kernels, key=demangle):
            print(f"  {demangle(k)}")
            print(f"    ({k})")
        return

    if not args.kernel:
        p.error("specify a kernel name (substring), or --list to see all")
    matched = filter_kernels(all_kernels, args.kernel, args.dtype, args.headdim)
    if not matched:
        sys.exit(f"no kernel matched '{args.kernel}'. Run with --list to see options.")

    mode = "both"
    if args.ptx_only:  mode = "ptx"
    if args.sass_only: mode = "sass"

    out_dir = Path(args.out_dir)
    print(f"matched {len(matched)} kernel(s):")
    for k in matched:
        print(f"\n{demangle(k)}")
        print(f"  ({k})")
        dump_one(cuobjdump, pyd, k, mode, out_dir)


if __name__ == "__main__":
    main()
