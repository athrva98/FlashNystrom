# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Coverage of benchmarks.train_three_way helpers and modules.

subset_indices / _margin_batch are pure. SDPAAttention / NystromRefAttention /
TinyViT run on CPU in fp32. _max_batch_for's doubling search is mocked (its real
_trial builds a model on cuda and OOM-probes -- never run that on the 8GB card).

Importing the module runs _pin_cifar_mirror() (one HEAD request, 5s timeout with
graceful fallback) and imports torchvision; that is a one-time collection cost.
"""
import pytest
import torch
import torch.nn as nn

from benchmarks.train_three_way import (
    subset_indices,
    _margin_batch,
    SDPAAttention,
    NystromRefAttention,
    TinyViT,
    _max_batch_for,
)


# --------------------------------------------------------------------------- #
# subset_indices
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("frac", [1.0, 1.5, 2.0, 100.0])
def test_subset_full_returns_none(frac):
    assert subset_indices(1000, frac, 64, 0) is None


@pytest.mark.parametrize("n,frac,batch,exp", [
    (1000, 0.5, 64, 500),
    (1000, 0.1, 64, 100),
    (1000, 0.001, 64, 64),   # int(1) -> floored up to batch_size
    (50, 0.9, 64, 50),       # max(64,45)=64 -> clamped to n_full=50
    (100, 0.25, 8, 25),
])
def test_subset_length(n, frac, batch, exp):
    idx = subset_indices(n, frac, batch, 0)
    assert len(idx) == exp


@pytest.mark.parametrize("seed", [0, 1, 7, 123])
def test_subset_deterministic_same_seed(seed):
    a = subset_indices(1000, 0.3, 64, seed)
    b = subset_indices(1000, 0.3, 64, seed)
    assert a == b


def test_subset_different_seeds_differ():
    a = subset_indices(1000, 0.3, 64, 0)
    b = subset_indices(1000, 0.3, 64, 1)
    assert a != b


@pytest.mark.parametrize("n,frac,batch", [(1000, 0.3, 64), (500, 0.5, 32), (200, 0.1, 16)])
def test_subset_indices_valid_and_distinct(n, frac, batch):
    idx = subset_indices(n, frac, batch, 0)
    assert all(0 <= i < n for i in idx)
    assert len(set(idx)) == len(idx)  # randperm -> distinct


def test_subset_never_below_one_batch():
    # tiny frac still yields at least batch_size samples (one full batch)
    idx = subset_indices(10_000, 1e-9, 128, 0)
    assert len(idx) == 128


# --------------------------------------------------------------------------- #
# _margin_batch
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("b", [0, None])
def test_margin_falsy_passthrough(b):
    assert _margin_batch(b, 0.85) == b


@pytest.mark.parametrize("b,margin,exp", [
    (100, 0.85, 80),    # int(85)=85 -> 85//8*8=80
    (1024, 0.85, 864),  # int(870.4)=870 -> 870//8*8=864
    (10, 0.85, 8),      # int(8.5)=8 -> 8
    (5, 0.85, 8),       # int(4.25)=4 -> 0 -> floored to 8
    (1000, 1.0, 1000),  # margin 1.0, 1000//8*8=1000
    (256, 0.5, 128),    # int(128)=128
])
def test_margin_values(b, margin, exp):
    assert _margin_batch(b, margin) == exp


@pytest.mark.parametrize("b", [16, 64, 128, 333, 999, 4096])
@pytest.mark.parametrize("margin", [0.7, 0.85, 0.9, 0.95])
def test_margin_multiple_of_8_and_at_least_8(b, margin):
    out = _margin_batch(b, margin)
    assert out % 8 == 0 and out >= 8


@pytest.mark.parametrize("b", [64, 128, 512])
def test_margin_never_exceeds_input(b):
    assert _margin_batch(b, 0.85) <= b


# --------------------------------------------------------------------------- #
# SDPAAttention (CPU fp32)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dim,heads", [(64, 2), (128, 4), (256, 8)])
@pytest.mark.parametrize("B,N", [(1, 32), (2, 65)])
def test_sdpa_forward_shape(dim, heads, B, N):
    torch.manual_seed(0)
    out = SDPAAttention(dim, heads)(torch.randn(B, N, dim))
    assert out.shape == (B, N, dim) and torch.isfinite(out).all()


def test_sdpa_projections_bias_free():
    a = SDPAAttention(64, 2)
    assert a.head_dim == 32
    for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
        assert getattr(a, p).bias is None


# --------------------------------------------------------------------------- #
# NystromRefAttention (CPU fp32)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dim,heads", [(64, 2), (128, 4)])
def test_nysref_forward_shape(dim, heads):
    torch.manual_seed(0)
    a = NystromRefAttention(dim, heads, num_landmarks=16, newton_iter=6)
    out = a(torch.randn(2, 64, dim))
    assert out.shape == (2, 64, dim) and torch.isfinite(out).all()


@pytest.mark.parametrize("ks", [1, 3, 5])
def test_nysref_conv_weight_created(ks):
    a = NystromRefAttention(64, 2, num_landmarks=16, conv_kernel_size=ks)
    assert isinstance(a.conv_weight, nn.Parameter) and a.conv_weight.shape == (2, ks)
    out = a(torch.randn(2, 64, 64))
    assert torch.isfinite(out).all()


def test_nysref_no_conv_when_ks_zero():
    a = NystromRefAttention(64, 2, num_landmarks=16, conv_kernel_size=0)
    assert a.conv_weight is None


@pytest.mark.parametrize("m,j,kappa", [(16, 6, 0.0), (32, 20, 1e3), (8, 10, 1.0)])
def test_nysref_stores_hparams(m, j, kappa):
    a = NystromRefAttention(64, 2, num_landmarks=m, newton_iter=j, kappa_star=kappa)
    assert (a.m, a.newton_iter, a.kappa_star) == (m, j, kappa)


# --------------------------------------------------------------------------- #
# TinyViT (CPU fp32)
# --------------------------------------------------------------------------- #

def _sdpa_factory(dim, heads):
    return SDPAAttention(dim, heads)


def _nysref_factory(dim, heads):
    return NystromRefAttention(dim, heads, num_landmarks=16, newton_iter=6)


@pytest.mark.parametrize("factory", [_sdpa_factory, _nysref_factory])
@pytest.mark.parametrize("dim,heads,depth", [(64, 2, 1), (64, 4, 2), (128, 4, 2)])
def test_tinyvit_forward_shape(factory, dim, heads, depth):
    torch.manual_seed(0)
    m = TinyViT(factory, dim=dim, depth=depth, heads=heads,
                patch_size=4, num_classes=10, img_size=32)
    out = m(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10) and torch.isfinite(out).all()


@pytest.mark.parametrize("patch,img,exp_patches", [(4, 32, 64), (8, 32, 16), (4, 16, 16)])
def test_tinyvit_token_shapes(patch, img, exp_patches):
    m = TinyViT(_sdpa_factory, dim=64, depth=1, heads=2,
                patch_size=patch, img_size=img)
    assert m.cls_token.shape == (1, 1, 64)
    assert m.pos_embed.shape == (1, exp_patches + 1, 64)


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_tinyvit_depth(depth):
    m = TinyViT(_sdpa_factory, dim=64, depth=depth, heads=2)
    assert len(m.blocks) == depth


def test_tinyvit_init_biases_zero_and_layernorm_unit():
    m = TinyViT(_sdpa_factory, dim=64, depth=2, heads=2)
    for mod in m.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)) and mod.bias is not None:
            assert torch.count_nonzero(mod.bias) == 0
        if isinstance(mod, nn.LayerNorm):
            assert torch.allclose(mod.weight, torch.ones_like(mod.weight))
            assert torch.count_nonzero(mod.bias) == 0


def test_tinyvit_init_weights_trunc_normal_scale():
    # trunc_normal_ std=0.02 -> weights bounded well within a few * 0.02
    m = TinyViT(_sdpa_factory, dim=128, depth=2, heads=4)
    assert m.patch_embed.weight.abs().max() < 0.2  # 10*std, generous


def test_tinyvit_backward_flows():
    m = TinyViT(_sdpa_factory, dim=64, depth=2, heads=2)
    m(torch.randn(2, 3, 32, 32)).sum().backward()
    assert m.cls_token.grad is not None and torch.isfinite(m.cls_token.grad).all()


def test_tinyvit_init_weights_staticmethod_direct():
    # exercise _init_weights on each branch directly
    lin = nn.Linear(8, 8)
    TinyViT._init_weights(lin)
    assert torch.count_nonzero(lin.bias) == 0
    ln = nn.LayerNorm(8)
    ln.weight.data.fill_(5.0); ln.bias.data.fill_(3.0)
    TinyViT._init_weights(ln)
    assert torch.allclose(ln.weight, torch.ones(8)) and torch.count_nonzero(ln.bias) == 0
    # a module type that matches neither branch is a no-op (no error)
    TinyViT._init_weights(nn.ReLU())


# --------------------------------------------------------------------------- #
# _max_batch_for (search mocked -- no GPU)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("batch", [64, 128, 512, 2048])
def test_max_batch_for_returns_probed_batch(monkeypatch, batch):
    import autobatch
    monkeypatch.setattr(autobatch, "search_and_profile", lambda *a, **k: {"batch": batch})
    got = _max_batch_for(_sdpa_factory, dim=64, heads=2, patch_size=4,
                         img_size=32, lr=1e-3, autobatch_cap=4096)
    assert got == batch


def test_max_batch_for_none_when_all_oom(monkeypatch):
    import autobatch
    monkeypatch.setattr(autobatch, "search_and_profile", lambda *a, **k: None)
    got = _max_batch_for(_sdpa_factory, dim=64, heads=2, patch_size=4,
                         img_size=32, lr=1e-3, autobatch_cap=4096)
    assert got is None


@pytest.mark.parametrize("amp", [True, False])
def test_max_batch_for_amp_flag(monkeypatch, amp):
    import autobatch
    seen = {}

    def _fake(trial, lo, cap, warmup, iters):
        seen["lo"], seen["cap"] = lo, cap
        return {"batch": 256}

    monkeypatch.setattr(autobatch, "search_and_profile", _fake)
    got = _max_batch_for(_sdpa_factory, dim=64, heads=2, patch_size=4,
                         img_size=32, lr=1e-3, autobatch_cap=1024, amp=amp)
    assert got == 256
    assert seen["lo"] == 64 and seen["cap"] == 1024


# --------------------------------------------------------------------------- #
# _pin_cifar_mirror  (network branches, mocked)
# --------------------------------------------------------------------------- #

import benchmarks.train_three_way as ttw
import torchvision
import urllib.request


class _Resp:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_pin_cifar_mirror_success(monkeypatch):
    saved = torchvision.datasets.CIFAR10.url
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(200))
    try:
        ttw._pin_cifar_mirror()
        assert "brainchip" in torchvision.datasets.CIFAR10.url  # mirror pinned
    finally:
        torchvision.datasets.CIFAR10.url = saved


def test_pin_cifar_mirror_non_200_skips(monkeypatch):
    saved = torchvision.datasets.CIFAR10.url
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(404))
    try:
        ttw._pin_cifar_mirror()  # 404 -> not pinned; url unchanged
        assert torchvision.datasets.CIFAR10.url == saved
    finally:
        torchvision.datasets.CIFAR10.url = saved


def test_pin_cifar_mirror_exception_continues(monkeypatch):
    saved = torchvision.datasets.CIFAR10.url

    def _raise(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", _raise)
    try:
        ttw._pin_cifar_mirror()  # exception caught -> falls through cleanly
        assert torchvision.datasets.CIFAR10.url == saved
    finally:
        torchvision.datasets.CIFAR10.url = saved


# --------------------------------------------------------------------------- #
# SDPAAttention sdpa_kernel context fallback (covers the except -> nullcontext)
# --------------------------------------------------------------------------- #

def test_sdpa_attention_ctx_fallback(monkeypatch):
    import torch.nn.attention as tna

    def _raise(*a, **k):
        raise RuntimeError("sdpa_kernel unavailable")

    monkeypatch.setattr(tna, "sdpa_kernel", _raise)
    out = SDPAAttention(64, 2)(torch.randn(2, 32, 64))
    assert out.shape == (2, 32, 64) and torch.isfinite(out).all()
