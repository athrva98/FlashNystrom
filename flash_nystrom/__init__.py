# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

from flash_nystrom.flash_nystrom import FlashNystromAttention, flash_nystrom_attention
from flash_nystrom.nystrom_config import NystromConfig

__all__ = [
    "FlashNystromAttention",
    "flash_nystrom_attention",
    "NystromConfig",
    "__version__",
]


# Single source of truth: read the version from the installed package
# metadata (PEP 566), which is sourced from pyproject.toml at install time.
# Falls back to a sentinel for development checkouts where the package has
# not been pip-installed (e.g. someone running tests from a fresh clone
# without `pip install -e .`).
def _read_version() -> str:
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:  # pragma: no cover - Python < 3.8
        return "0.0.0+unknown"
    try:
        return version("flash-nystrom")
    except PackageNotFoundError:
        return "0.0.0+source"


__version__ = _read_version()


# Check for CUDA extension availability and warn if not found.
import warnings as _warnings
try:
    import flash_nystrom._C  # noqa: F401
except ImportError:
    _warnings.warn(
        "flash_nystrom._C CUDA extension not found. "
        "Falling back to pure-PyTorch reference implementation (much slower). "
        "Build with: pip install -e . --no-build-isolation "
        "(requires CUDA toolkit and the CUTLASS submodule).",
        UserWarning,
        stacklevel=1,
    )
else:
    # Blackwell-native kernels live in a separate sm_100a-only extension
    # (the main module is multi-arch and cannot carry tcgen05 code). On a
    # Blackwell datacenter GPU, hand its raw launcher to the main module;
    # launch_kernel3_bwd tries it before the sm80 path and falls back for
    # unsupported shapes. FLASH_NYSTROM_SM100=0 disables at kernel level.
    try:
        import flash_nystrom._C_sm100 as _C_sm100
        if _C_sm100.available():
            flash_nystrom._C._register_sm100_kernel3_bwd(
                _C_sm100._kernel3_bwd_hook_ptr())
    except ImportError:
        pass
