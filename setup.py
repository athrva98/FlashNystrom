# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

import os
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
cutlass_include = os.path.join(this_dir, "third_party", "cutlass", "include")

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-lineinfo",
    "-std=c++17",
    # Resource visibility. Print registers, stack, spill, and shared-memory
    # usage per kernel at compile time. Required for any informed occupancy
    # tuning. Can be disabled by setting FLASH_NYSTROM_QUIET=1.
    *([] if os.environ.get("FLASH_NYSTROM_QUIET") else ["-Xptxas=-v", "--resource-usage"]),
]

# Detect GPU architecture from nvidia-smi
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        caps = set()
        for line in result.stdout.strip().split("\n"):
            cap = line.strip().replace(".", "")
            if cap:
                caps.add(cap)
        for cap in sorted(caps):
            nvcc_flags.append(f"-gencode=arch=compute_{cap},code=sm_{cap}")
    else:
        # Fallback: common architectures
        nvcc_flags.append("-gencode=arch=compute_80,code=sm_80")
except Exception:
    nvcc_flags.append("-gencode=arch=compute_80,code=sm_80")

ext_modules = [
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
        # cuBLAS is used for the m-bounded dense GEMMs (Newton-Schulz step,
        # Newton-Schulz final, landmark projection). Pure dense matmuls with
        # no softmax fusion, so we let NVIDIA's tuned kernels do the work.
        # Tensor-core flash kernels (kernel1_fused_tc, kernel3_fused_tc and
        # their backwards) keep streaming custom kernels.
        libraries=["cublas"],
        extra_compile_args={
            "cxx": ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags,
        },
    ),
]

setup(
    name="flash-nystrom",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
