# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
#
# setup.py is intentionally cautious about its imports. Three command paths
# matter:
#
#   1. `python -m build --sdist` builds a source tarball. No compilation. No
#      CUDA needed. setup.py must import cleanly even when CUDA is absent.
#
#   2. `python -m build --wheel` or `pip install .` builds the extension.
#      Requires CUDA toolkit, nvcc on PATH, and a torch install whose
#      torch.utils.cpp_extension can find CUDA_HOME.
#
#   3. `pip install <sdist tarball>` on a user's machine. Same requirements
#      as (2). This is what PyPI users hit.
#
# We achieve (1) by deferring the import of torch.utils.cpp_extension and the
# instantiation of CUDAExtension until setup() actually needs an ext_modules
# list. When the command is `sdist` we skip all of that.

import os
import sys
import subprocess
from setuptools import setup, find_packages

# ---------------------------------------------------------------------------
# Compute capability detection. nvidia-smi may not exist (sdist build on a
# machine without GPU, CI runner with CPU only, docker image during sdist).
# In that case we fall back to a multi-arch wheel that covers Ampere, Ada,
# Hopper, and Blackwell consumer (SM 8.0 / 8.6 / 8.9 / 9.0 / 12.0). Users
# building from source on a specific machine can override via
# `FLASH_NYSTROM_CUDA_ARCH_LIST=90` (NVCC arch flag list, semicolon or
# space separated). Matches PyTorch's TORCH_CUDA_ARCH_LIST convention.
# ---------------------------------------------------------------------------

def _detect_cuda_arches():
    """Return a sorted list of compute capabilities (strings like '80', '90')."""
    env = os.environ.get("FLASH_NYSTROM_CUDA_ARCH_LIST") \
        or os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env:
        # PyTorch accepts forms like "8.0 8.6+PTX" or "8.0;8.6". Normalize to
        # bare two-digit strings.
        toks = env.replace(";", " ").replace(",", " ").split()
        out = set()
        for t in toks:
            t = t.replace("+PTX", "").replace(".", "").strip()
            # Accept bare caps ("80", "90", "100") and the architecture-specific
            # variants ("90a", "100a") that gate Hopper WGMMA/TMA and Blackwell
            # tcgen05. nvcc emits -gencode arch=compute_90a,code=sm_90a for "90a".
            if t.isdigit() or (len(t) > 1 and t[-1] == "a" and t[:-1].isdigit()):
                out.add(t)
        if out:
            return sorted(out)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        result = None

    if result is not None and result.returncode == 0:
        caps = set()
        for line in result.stdout.strip().split("\n"):
            cap = line.strip().replace(".", "")
            if cap and cap.isdigit():
                caps.add(cap)
        if caps:
            return sorted(caps)

    # Fallback: ship a multi-arch wheel covering the supported range.
    # SM 7.5 and earlier are explicitly unsupported (no SM80 mma atom).
    return ["80", "86", "89", "90"]


def _build_ext_modules():
    """Build CUDAExtension list. Imports torch lazily so sdist does not need CUDA."""
    # Deferred import: only required when we actually build an extension.
    from torch.utils.cpp_extension import CUDAExtension

    this_dir = os.path.dirname(os.path.abspath(__file__))
    cutlass_include = os.path.join(this_dir, "third_party", "cutlass", "include")

    if not os.path.isdir(cutlass_include):
        # The CUTLASS submodule has not been initialized. Fail with a clear
        # message rather than letting nvcc complain about missing headers.
        raise RuntimeError(
            f"CUTLASS submodule not found at {cutlass_include!r}.\n"
            "Initialize it with:\n"
            "  git submodule update --init --recursive\n"
            "or clone the repository with --recursive."
        )

    nvcc_flags = [
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        "-std=c++17",
        # Pin the current cross-TU template instantiation behaviour. nvcc 12.8+
        # warns (#20281-D) that the default will flip to true in a future
        # release, which would break our pattern of declaring __global__
        # template kernels in headers and explicitly instantiating them in a
        # separate .cu (see kernel2_inv_bwd.cuh / kernel2_inv_bwd.cu). Setting
        # this to false locks in the behaviour we depend on.
        "-static-global-template-stub=false",
        # Resource visibility. Off by default in CI / PyPI builds (verbose,
        # slows compile). Set FLASH_NYSTROM_VERBOSE=1 to re-enable for tuning.
        *(["-Xptxas=-v", "--resource-usage"]
          if os.environ.get("FLASH_NYSTROM_VERBOSE") else []),
    ]
    for cap in _detect_cuda_arches():
        nvcc_flags.append(f"-gencode=arch=compute_{cap},code=sm_{cap}")

    # ---------------------- Strict compilation -----------------------------
    # Default ON. The local build was producing standards-noncompliant code
    # that MSVC silently accepted via its rvalue-to-lvalue-ref extension and
    # gcc rejected on Linux (Colab). To prevent that class of bug from
    # recurring, force the MSVC host compiler into standard-conformance mode
    # (/permissive-) and elevate warnings to errors on both host compilers
    # and nvcc itself.
    #
    # Disable via FLASH_NYSTROM_LAX_BUILD=1 if a transient nvcc/CUTLASS
    # warning is blocking work. The lax path should be used sparingly; the
    # default exists because of a real bug it would have caught.
    strict = not os.environ.get("FLASH_NYSTROM_LAX_BUILD")

    # Warnings we cannot fix because they originate in headers we do not own
    # (CUDA SDK, CUTLASS, PyTorch). Suppressed so /WX and -Werror only fire
    # on warnings in *our* code.
    #
    #   MSVC C4996: deprecated declarations. Fired by cusparse.h in CUDA 12.9.
    #   MSVC C4505/C4100: unused static / unused param, fired by CUTLASS.
    #   MSVC C4127: conditional expression constant, fired by CUTLASS unroll
    #   macros.
    #   MSVC C4172: "returning address of local or temporary", a false
    #   positive in cute/numeric/arithmetic_tuple.hpp (TMA ArithTuple).
    third_party_suppressions_msvc = ["/wd4996", "/wd4505", "/wd4100", "/wd4127",
                                     "/wd4172"]
    third_party_suppressions_gcc = [
        "-Wno-deprecated-declarations",
        "-Wno-unused-function",
        "-Wno-unused-parameter",
        # CUTLASS spams these on Hopper-target builds. Not our bugs.
        "-Wno-strict-aliasing",
        "-Wno-sign-compare",
    ]

    if os.name == "nt":
        # /Zc:__cplusplus makes MSVC report the real __cplusplus value
        # (199711L otherwise, even under /std:c++17). CUTLASS >= 3.8 guards
        # its C++17 platform aliases (is_unsigned_v etc., used by
        # exmy_base.h) with `201703L <= __cplusplus`, so without this flag
        # the MSVC host pass of every nvcc TU fails to find them.
        cxx_flags = ["/O2", "/std:c++17", "/Zc:__cplusplus"]
        nvcc_flags += ["-Xcompiler", "/Zc:__cplusplus"]
        if strict:
            # /permissive- disables MSVC's non-conforming extensions. This
            # is the flag that makes the build refuse the rvalue-to-non-
            # const-lvalue-ref binding that previously broke gcc/Linux.
            # /WX promotes warnings to errors. /W3 sets a reasonable warning
            # level (W4 is too noisy on CUTLASS/PyTorch headers).
            cxx_flags += ["/permissive-", "/W3", "/WX"]
            cxx_flags += third_party_suppressions_msvc
            # Mirror the host-compiler strictness through nvcc's -Xcompiler
            # so device-side TUs the host frontend sees also enforce it.
            nvcc_flags += [
                "-Xcompiler", "/permissive-",
                "-Xcompiler", "/W3",
                "-Xcompiler", "/WX",
            ]
            for f in third_party_suppressions_msvc:
                nvcc_flags += ["-Xcompiler", f]
    else:
        cxx_flags = ["-O3", "-std=c++17"]
        if strict:
            cxx_flags += ["-Wall", "-Wextra", "-Werror"]
            cxx_flags += third_party_suppressions_gcc
            nvcc_flags += [
                "-Xcompiler", "-Wall",
                "-Xcompiler", "-Wextra",
                "-Xcompiler", "-Werror",
            ]
            for f in third_party_suppressions_gcc:
                nvcc_flags += ["-Xcompiler", f]

    if strict:
        # nvcc-specific diagnostics. cross-execution-space-call catches host
        # functions called from device code (silent UB otherwise). reorder
        # catches member-init-order bugs. all-warnings promotes every other
        # nvcc-level warning (narrowing, undefined behaviour, etc.) to an
        # error. We deliberately do NOT promote deprecated-declarations
        # because CUDA 12.9's cusparse.h trips it on its own typedefs;
        # that's NVIDIA's bug, not ours.
        nvcc_flags += [
            "-Werror", "cross-execution-space-call",
            "-Werror", "reorder",
            "-Werror", "all-warnings",
        ]
        # Suppress specific nvcc warnings that come from CUTLASS headers
        # which we cannot modify. 550 = "variable set but never used"
        # (cute/layout.hpp:1443 has this pattern intentionally).
        nvcc_flags += ["-Xcudafe", "--diag_suppress=550"]

    modules = [
        CUDAExtension(
            name="flash_nystrom._C",
            sources=[
                "csrc/flash_nystrom.cu",
                "csrc/flash_nystrom_kernels.cu",
                "csrc/kernels/backward/kernel2_inv_bwd.cu",
            ],
            include_dirs=[
                os.path.join(this_dir, "csrc"),
                cutlass_include,
            ],
            # cuBLAS is used for the m-bounded dense matmuls in the
            # Newton-Schulz backward. Tensor-core flash kernels stay custom.
            libraries=["cublas"],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        ),
    ]

    # ------------------- Blackwell-native module (sm_100a) -----------------
    # Separate extension so the tcgen05/TMEM code never touches the
    # multi-arch main module: it is compiled ONLY for sm_100a (arch-specific
    # features are not portable across compute capabilities), imported lazily,
    # and dispatched to at runtime only on Blackwell datacenter GPUs. The
    # module cross-compiles on any host with CUDA >= 12.6; it simply cannot
    # LAUNCH on non-sm_100 devices. Disable with FLASH_NYSTROM_BUILD_SM100=0.
    if os.environ.get("FLASH_NYSTROM_BUILD_SM100", "1") != "0":
        # Strip the multi-arch gencodes (this module is sm_100a-only) and
        # the -Werror reorder/all-warnings promotions: CUTLASS's sm100
        # pipeline/collective headers emit member-init-order warnings in
        # their own code, which we cannot fix. cross-execution-space-call
        # stays. Host-side MSVC /WX still applies to our code.
        nvcc_flags_sm100 = []
        skip_next = False
        for i, f in enumerate(nvcc_flags):
            if skip_next:
                skip_next = False
                continue
            if f.startswith("-gencode"):
                continue
            if f == "-Werror" and i + 1 < len(nvcc_flags) and                     nvcc_flags[i + 1] in ("reorder", "all-warnings"):
                skip_next = True
                continue
            nvcc_flags_sm100.append(f)
        nvcc_flags_sm100 += ["-gencode=arch=compute_100a,code=sm_100a"]
        modules.append(
            CUDAExtension(
                name="flash_nystrom._C_sm100",
                sources=[
                    "csrc/sm100/sm100_smoke.cu",
                ],
                include_dirs=[
                    os.path.join(this_dir, "csrc"),
                    cutlass_include,
                ],
                extra_compile_args={"cxx": cxx_flags,
                                    "nvcc": nvcc_flags_sm100},
            )
        )

    return modules


def _cmdclass():
    """Return cmdclass dict. Imports BuildExtension lazily."""
    from torch.utils.cpp_extension import BuildExtension
    return {"build_ext": BuildExtension}


# ---------------------------------------------------------------------------
# Skip CUDA-touching work for commands that don't need it.
#
# Commands like `sdist`, `egg_info`, `--version` should not require torch
# to be importable or CUDA to be available. The PyPI sdist build runs setup.py
# inside an isolated env that may not have torch, and CI sdist jobs may not
# have CUDA on the runner. By gating the extension construction we keep
# those commands working in CUDA-less environments.
# ---------------------------------------------------------------------------

_CUDA_LESS_COMMANDS = {
    "sdist", "egg_info", "dist_info", "check", "clean",
    "--help", "--help-commands", "--version",
}

_needs_extension = not any(
    arg in _CUDA_LESS_COMMANDS for arg in sys.argv[1:]
)

if _needs_extension:
    ext_modules = _build_ext_modules()
    cmdclass = _cmdclass()
else:
    ext_modules = []
    cmdclass = {}

setup(
    name="flash-nystrom",
    # Version is the single source of truth in pyproject.toml; setup() reads
    # it from there via the PEP 621 mechanism. We omit version= here to
    # avoid drift.
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
)
