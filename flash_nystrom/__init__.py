# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

from flash_nystrom.flash_nystrom import FlashNystromAttention, flash_nystrom_attention
from flash_nystrom.nystrom_config import NystromConfig

__all__ = [
    "FlashNystromAttention",
    "flash_nystrom_attention",
    "NystromConfig",
]

__version__ = "0.1.0"

# Check for CUDA extension availability and warn if not found
import warnings as _warnings
try:
    import flash_nystrom._C  # noqa: F401
except ImportError:
    _warnings.warn(
        "flash_nystrom._C CUDA extension not found. "
        "Falling back to pure-PyTorch reference implementation (much slower). "
        "Build with: pip install -e . (requires CUDA toolkit and CUTLASS)",
        UserWarning,
        stacklevel=1,
    )
