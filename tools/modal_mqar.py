# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Build and benchmark FlashNystrom on a real A100-80GB via Modal.

This removes the "I can't test on A100" gap: the extension is compiled on a
CUDA 12.8 devel image (sm_80) at image-build time, then the test suite and a
FN-vs-cuBLAS benchmark run on an actual A100-80GB.

Per the project owner's instruction the benchmark compares ONLY against the
cuBLAS path (the pure-PyTorch Nystrom reference, whose every matmul dispatches
to cuBLAS). No SDPA column.

USAGE
-----
One-time auth (opens a browser; you must do this, I can't):
    pip install modal
    modal setup

Run the tests:
    modal run tools/modal_a100.py::test
Run the benchmark:
    modal run tools/modal_a100.py::bench
Both (default entrypoint):
    modal run tools/modal_a100.py

Editing a kernel and re-running `modal run` rebuilds the extension layer
automatically (the local source is part of the image), so the loop is:
    edit -> modal run tools/modal_a100.py::bench -> read A100 numbers.
"""
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
REMOTE = "/root/FlashNystrom"

# CUDA 12.8 devel to match the strict-build nvcc flags (e.g.
# -static-global-template-stub, introduced in 12.8) and the cu128 torch wheels.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    # build-essential = gcc/g++ for nvcc's host compiler; clang because the
    # Modal Python's sysconfig records clang++ as the extension linker (the
    # nvcc compiles use gcc, but the final link shells out to clang++).
    .apt_install("git", "build-essential", "clang")
    # Pin the +cu128 local tag so pip is FORCED to take torch from the cu128
    # index (a bare "torch" with extra_index_url falls back to PyPI's default
    # build, which now bundles CUDA-13 wheels and mismatches the 12.8 nvcc in
    # this image). Pure-python deps (sympy, etc.) still resolve from PyPI.
    .pip_install(
        "torch==2.7.1+cu128",
        "pytest",
        "ninja",
        "numpy",
        # setuptools + wheel are required at build time because we use
        # --no-build-isolation (pip won't provision its own build env, so
        # bdist_wheel must already be importable).
        "setuptools",
        "wheel",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    # Build for A100 (sm_80), H100/H200 (sm_90), and B200 (sm_100): no nvidia-smi
    # at image-build time, so pin all three arches. The SM80 MMA / cp.async atoms
    # compile and run on Hopper and Blackwell in compatibility mode, so one image
    # serves every datacenter GPU (H200 is sm_90 like H100; B200 is sm_100).
    # 90a (not 90): the architecture-specific Hopper target that enables WGMMA +
    # TMA, needed by the native Hopper kernel family. SM80-idiom code still
    # compiles and runs on Hopper under sm_90a. 100 = B200; local builds are sm_120.
    .env({"FLASH_NYSTROM_CUDA_ARCH_LIST": "80 90a 100"})
    .add_local_dir(
        str(REPO),
        remote_path=REMOTE,
        copy=True,  # build-time layer so the run_commands below can compile it
        ignore=[
            "**/.git", "**/__pycache__", "**/*.pyc", "**/*.pyd", "**/*.so",
            "**/*.o", "**/*.a", "build/", "dist/", "**/*.egg-info",
            ".venv*/", "**/.pytest_cache",
            # Trim the CUTLASS tree to the header-only include/ we actually need.
            "third_party/cutlass/test", "third_party/cutlass/examples",
            "third_party/cutlass/tools", "third_party/cutlass/docs",
            "third_party/cutlass/media", "third_party/cutlass/python",
            "third_party/cutlass/.git",
        ],
    )
    .run_commands(
        f"cd {REMOTE} && pip install -e . --no-build-isolation -v"
    )
)

app = modal.App("flash-nystrom-mqar", image=image)


@app.function(gpu="A100-80GB", timeout=14400)
def mqar_diagnostic():
    """Reduced MQAR diagnostic: exact attention + the three Nystrom variants.

    Lives in its own Modal file because `modal run` builds every image defined
    in the file it is given, and the FlashAttention-3 image in modal_a100.py
    costs ~40 min of nvcc that this job never uses.
    """
    import subprocess
    r = subprocess.run(
        ["python", "-u", "-m", "paper.mqar.paper_sweep",
         "--methods", "sdpa", "nystrom_reference", "flash_nystrom",
         "flash_nystrom_tc", "--max_parallel", "4", "--out", "runs/mqar_paper"],
        cwd="/root/FlashNystrom", capture_output=True, text=True)
    print(r.stdout[-14000:])
    if r.returncode != 0:
        print("=== STDERR ==="); print(r.stderr[-4000:])
