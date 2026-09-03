# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Coverage of benchmarks.train_three_way train_one() and main().

main() is orchestration -> smoke-tested on CPU with train_one/_max_batch_for
stubbed. train_one() is a real GPU training step -> smoke-tested with a TINY
model on fake in-memory data (no CIFAR/STL download), so it runs on the GPU
without risking the 8GB card (dim 32, batch 8, ~10 steps -> a few MB).
"""
import json
import sys

import pytest
import torch
import torchvision

import benchmarks.train_three_way as ttw
from benchmarks.train_three_way import train_one, main, _max_batch_for, SDPAAttention

HAS_GPU = torch.cuda.is_available()
gpu = pytest.mark.skipif(not HAS_GPU, reason="needs CUDA")


def _install_fake_vision(monkeypatch, H, n_train=40, n_test=16):
    """Replace CIFAR10/STL10 with an in-memory dataset of random HxH images so
    train_one never downloads and each epoch is a handful of tiny steps."""
    class _Fake(torch.utils.data.Dataset):
        def __init__(self, root=None, train=True, download=False, transform=None, split=None):
            is_train = train if split is None else (split == "train")
            self._n = n_train if is_train else n_test

        def __len__(self):
            return self._n

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return torch.randn(3, H, H, generator=g), i % 10

    monkeypatch.setattr(torchvision.datasets, "CIFAR10", _Fake)
    monkeypatch.setattr(torchvision.datasets, "STL10", _Fake)


def _sdpa(d, h):
    return SDPAAttention(d, h)


# =========================================================================== #
# main() -- orchestration, CPU, train_one stubbed
# =========================================================================== #

def _fake_train_one(label, fac, amp=True, **kw):
    return {"label": label, "test_acc": 50.0, "N_tokens": 65,
            "batch": kw.get("batch_size"), "samples_per_s": 100.0,
            "peak_gib": 0.3, "avg_epoch": 1.0, "params_M": 0.1}


def test_main_no_autobatch_writes_json(monkeypatch, tmp_path):
    monkeypatch.setattr(ttw, "train_one", _fake_train_one)
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--backends", "sdpa", "flash_nystrom",
        "--epochs", "1", "--out_json", str(out),
    ])
    main()
    rec = json.loads(out.read_text())
    assert len(rec["results"]) == 2
    assert {r["label"] for r in rec["results"]} == {"SDPA", "FlashNystrom"}


def test_main_autobatch_coordinated(monkeypatch, tmp_path):
    monkeypatch.setattr(ttw, "train_one", _fake_train_one)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    # different max per arm -> the min is used for all
    sizes = iter([512, 256])
    monkeypatch.setattr(ttw, "_max_batch_for", lambda *a, **k: next(sizes))
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--autobatch", "--backends", "sdpa", "nystrom_reference_fp32",
        "--epochs", "1", "--autobatch_margin", "0.85", "--out_json", str(out),
    ])
    main()
    rec = json.loads(out.read_text())
    assert len(rec["results"]) == 2


def test_main_autobatch_no_fit_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(ttw, "train_one", _fake_train_one)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(ttw, "_max_batch_for", lambda *a, **k: None)  # nothing fits
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--autobatch", "--backends", "sdpa",
        "--batch_size", "77", "--epochs", "1", "--out_json", str(out),
    ])
    main()
    rec = json.loads(out.read_text())
    assert len(rec["results"]) == 1


@pytest.mark.parametrize("backend", [
    "sdpa", "nystrom_reference", "nystrom_reference_fp32", "nystrom_vanilla",
    "flash_nystrom", "flash_nystrom_tc", "flash_nystrom_vanilla",
    "flash_nystrom_tc_vanilla",
])
def test_main_all_backend_factories(monkeypatch, tmp_path, backend):
    # exercises every entry of the factories dict + the per-arm loop
    monkeypatch.setattr(ttw, "train_one", _fake_train_one)
    out = tmp_path / f"{backend}.json"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--backends", backend, "--epochs", "1", "--out_json", str(out),
    ])
    main()
    assert out.exists()


# =========================================================================== #
# _max_batch_for -- run the real _trial closure once at a tiny batch (GPU)
# =========================================================================== #

@gpu
def test_max_batch_for_trial_closure_runs(monkeypatch):
    import autobatch

    def _fake_search(trial, lo, cap, warmup, iters):
        run = trial(lo)   # builds a tiny cuda model + inputs at batch=lo
        run()             # one real fwd+bwd+opt step on the GPU
        return {"batch": lo}

    monkeypatch.setattr(autobatch, "search_and_profile", _fake_search)
    got = _max_batch_for(_sdpa, dim=64, heads=2, patch_size=4, img_size=32,
                         lr=1e-3, autobatch_cap=64)
    assert got == 64


# =========================================================================== #
# train_one -- real GPU training step, tiny model + fake data
# =========================================================================== #

@gpu
def test_train_one_basic(monkeypatch):
    _install_fake_vision(monkeypatch, H=32)
    r = train_one("SDPA", _sdpa, epochs=2, batch_size=8, dim=32, heads=2,
                  dataset="cifar10", img_size=32, instrument=False, amp=True)
    assert 0.0 <= r["test_acc"] <= 100.0 and r["batch"] == 8
    assert r["N_tokens"] == (32 // 4) ** 2 + 1


@gpu
@pytest.mark.parametrize("amp", [True, False])
def test_train_one_amp_toggle(monkeypatch, amp):
    _install_fake_vision(monkeypatch, H=32)
    r = train_one("SDPA", _sdpa, epochs=2, batch_size=8, dim=32, heads=2,
                  instrument=False, amp=amp)
    assert "test_acc" in r


@gpu
@pytest.mark.parametrize("warmup_frac", [0.0, 0.1, 0.5])
def test_train_one_warmup_variants(monkeypatch, warmup_frac):
    _install_fake_vision(monkeypatch, H=32)
    r = train_one("SDPA", _sdpa, epochs=2, batch_size=8, dim=32, heads=2,
                  instrument=False, amp=True, warmup_frac=warmup_frac)
    assert "test_acc" in r


@gpu
@pytest.mark.parametrize("grad_clip", [0.0, 1.0])
def test_train_one_grad_clip(monkeypatch, grad_clip):
    _install_fake_vision(monkeypatch, H=32)
    r = train_one("SDPA", _sdpa, epochs=1, batch_size=8, dim=32, heads=2,
                  instrument=False, amp=True, grad_clip=grad_clip)
    assert "test_acc" in r


@gpu
def test_train_one_instrument(monkeypatch):
    _install_fake_vision(monkeypatch, H=32)
    # single-threaded autograd so the registered backward hooks run on THIS
    # thread and are visible to coverage (they fire on a native engine thread
    # otherwise). This exercises the per-layer dO instrumentation hook body.
    with torch.autograd.set_multithreading_enabled(False):
        r = train_one("SDPA", _sdpa, epochs=2, batch_size=8, dim=32, heads=2,
                      instrument=True, amp=True)
    assert "test_acc" in r


@gpu
def test_train_one_train_frac_subset(monkeypatch):
    _install_fake_vision(monkeypatch, H=32, n_train=40)
    r = train_one("SDPA", _sdpa, epochs=1, batch_size=8, dim=32, heads=2,
                  instrument=False, amp=True, train_frac=0.5)
    assert "test_acc" in r


@gpu
def test_train_one_stl10_and_resize(monkeypatch):
    # stl10 branch (native 96) + a non-native img_size to hit the Resize path
    _install_fake_vision(monkeypatch, H=48)
    r = train_one("SDPA", _sdpa, epochs=1, batch_size=8, dim=32, heads=2,
                  dataset="stl10", img_size=48, instrument=False, amp=True)
    assert "test_acc" in r


@gpu
def test_train_one_autobatch_path(monkeypatch):
    _install_fake_vision(monkeypatch, H=32)
    monkeypatch.setattr(ttw, "_max_batch_for", lambda *a, **k: 16)
    r = train_one("SDPA", _sdpa, epochs=1, dim=32, heads=2,
                  instrument=False, amp=True, autobatch=True, autobatch_margin=0.85)
    # 16 * 0.85 = 13.6 -> int 13 -> //8*8 = 8
    assert r["batch"] == 8


@gpu
def test_train_one_flash_nystrom_on_gpu(monkeypatch):
    # a real FlashNystrom attention training step (head_dim 64: dim 64, heads 1)
    from flash_nystrom import FlashNystromAttention, NystromConfig
    _install_fake_vision(monkeypatch, H=32)

    def _fn(d, h):
        return FlashNystromAttention(d, h, NystromConfig(num_landmarks=32, newton_iter=6))

    r = train_one("FlashNystrom", _fn, epochs=1, batch_size=8, dim=64, heads=1,
                  instrument=False, amp=True)
    assert "test_acc" in r
