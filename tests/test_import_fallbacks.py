# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Coverage for import-time fallback branches that only execute in a degraded
environment (package not pip-installed, or the _C CUDA extension not built).

In a normal built/installed env those branches never run, so we simulate the
failure by intercepting __import__ / patching importlib.metadata and reloading
the module, then always restore the real state in a finally block.
"""
import importlib
import sys

import pytest


# --------------------------------------------------------------------------- #
# __init__._read_version
# --------------------------------------------------------------------------- #

def test_read_version_source_fallback(monkeypatch):
    # version("flash-nystrom") raises PackageNotFoundError for a source checkout
    # that was never installed -> the "0.0.0+source" sentinel.
    import flash_nystrom
    import importlib.metadata as meta

    def _raise(_name):
        raise meta.PackageNotFoundError("flash-nystrom")

    monkeypatch.setattr(meta, "version", _raise)
    assert flash_nystrom._read_version() == "0.0.0+source"


def test_read_version_installed_path():
    # normal path: returns a non-empty string (dist metadata present)
    import flash_nystrom
    v = flash_nystrom._read_version()
    assert isinstance(v, str) and v


# --------------------------------------------------------------------------- #
# _C optional-extension import fallback
# --------------------------------------------------------------------------- #
#
# To make `import flash_nystrom._C` raise, two things must be undone: the
# sys.modules cache entry (set to None -> CPython raises ImportError) AND the
# `_C` attribute the parent package holds (else the import short-circuits to the
# cached submodule attribute and succeeds). We restore both in finally.

def _block_C():
    import flash_nystrom as pkg
    saved_mod = sys.modules.get("flash_nystrom._C", None)
    had_attr = hasattr(pkg, "_C")
    saved_attr = getattr(pkg, "_C", None)
    if had_attr:
        del pkg._C
    sys.modules["flash_nystrom._C"] = None
    return saved_mod, had_attr, saved_attr


def _restore_C(state):
    import flash_nystrom as pkg
    saved_mod, had_attr, saved_attr = state
    if saved_mod is not None:
        sys.modules["flash_nystrom._C"] = saved_mod
    else:
        sys.modules.pop("flash_nystrom._C", None)
    if had_attr:
        pkg._C = saved_attr


def test_flash_nystrom_module_C_import_fallback():
    # Covers `except ImportError: _C = None; HAS_CUDA = False` in flash_nystrom.py.
    # Grab the submodule via sys.modules: `import a.a as x` mis-binds when the
    # package and submodule share a leaf name.
    import flash_nystrom.flash_nystrom  # noqa: F401  (ensure imported)
    fn = sys.modules["flash_nystrom.flash_nystrom"]
    state = _block_C()
    try:
        importlib.reload(fn)
        assert fn._C is None
        assert fn.HAS_CUDA is False
    finally:
        _restore_C(state)
        importlib.reload(fn)  # restore the real _C / HAS_CUDA for later tests
    # reload mutates the module dict in place, so previously-captured references
    # (in other test modules) also see the restored state.
    assert fn.HAS_CUDA is True
    assert fn._C is not None


def test_package_init_warns_when_C_missing():
    # Covers the `except ImportError: warnings.warn(...)` in __init__.py.
    import flash_nystrom
    state = _block_C()
    try:
        with pytest.warns(UserWarning, match="_C CUDA extension not found"):
            importlib.reload(flash_nystrom)
    finally:
        _restore_C(state)
        importlib.reload(flash_nystrom)  # restore
