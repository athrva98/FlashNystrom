# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Coverage of flash_nystrom.flash_nystrom (the package entry point).

Splits into (a) pure-Python / CPU paths that run without the CUDA extension
(budget helpers, depthwise conv, CPU fallback, module construction) and (b) the
CUDA forward/backward paths (skipped without a GPU + built _C).
"""
import os
import math
import pytest
import torch

import flash_nystrom.flash_nystrom as fnmod
from flash_nystrom.flash_nystrom import (
    flash_nystrom_attention, FlashNystromAttention,
    _depthwise_conv_residual, _reference_softmax_bytes, _check_reference_budget,
    _M_CUSTOM_KERNEL_MAX, _DEFAULT_REFERENCE_BYTE_BUDGET,
)
from flash_nystrom.nystrom_config import NystromConfig
from flash_nystrom.reference import nystrom_attention_reference

HAS_GPU = torch.cuda.is_available()
gpu = pytest.mark.skipif(not HAS_GPU, reason="needs CUDA + built _C")


def _fp32_d128_smem_ok():
    """fp32 D=128 uses the scalar kernel3, which needs ~150KB opt-in dynamic SMEM
    (fits A100/H100/B200; NOT consumer cards ~100KB). Skip that one case where the
    device can't provide the shared memory -- it's a hardware limit, not a bug."""
    if not HAS_GPU:
        return False
    p = torch.cuda.get_device_properties(0)
    smem = getattr(p, "shared_memory_per_block_optin", None) or \
        getattr(p, "shared_memory_per_block", 0)
    return smem >= 150 * 1024


FP32_D128_OK = _fp32_d128_smem_ok()


# =========================================================================== #
# _reference_softmax_bytes  (pure)
# =========================================================================== #

@pytest.mark.parametrize("B,H,N,D", [(1, 1, 256, 64), (2, 4, 1024, 64), (1, 8, 9216, 64), (3, 2, 512, 128)])
@pytest.mark.parametrize("m", [1, 16, 64, 128, 256])
def test_softmax_bytes_formula(B, H, N, D, m):
    assert _reference_softmax_bytes((B, H, N, D), m) == 2 * 2 * B * H * N * m


def test_softmax_bytes_scales_linearly_in_m():
    base = _reference_softmax_bytes((1, 1, 100, 64), 1)
    assert _reference_softmax_bytes((1, 1, 100, 64), 10) == 10 * base


# =========================================================================== #
# _check_reference_budget  (pure)
# =========================================================================== #

def test_budget_default_constant():
    assert _DEFAULT_REFERENCE_BYTE_BUDGET == 8 * (1024 ** 3)


# NOTE: the budget check only reads q.shape/q.device -- it never touches the data.
# So we use `device='meta'` tensors (shape metadata, ZERO storage). This makes it
# impossible for these tests to allocate real memory (never allocate a big tensor
# on the 8GB local GPU -- it hard-crashes the host).

def test_budget_small_ok():
    q = torch.empty(1, 1, 64, 64, device="meta")
    _check_reference_budget(q, 64)  # tiny, must not raise


def test_budget_exceeded_raises():
    q = torch.empty(4, 8, 2_000_000, 64, device="meta")  # 4*B*H*N*m = 16 GiB IF materialized, >8 GiB budget
    with pytest.raises(RuntimeError, match="materializes"):
        _check_reference_budget(q, 64)


@pytest.mark.parametrize("budget", [1, 1024, 10**6])
def test_budget_env_override_low(monkeypatch, budget):
    monkeypatch.setenv("FLASH_NYSTROM_REFERENCE_MAX_BYTES", str(budget))
    q = torch.empty(1, 1, 4096, 64, device="meta")
    with pytest.raises(RuntimeError):
        _check_reference_budget(q, 64)


def test_budget_env_override_high(monkeypatch):
    monkeypatch.setenv("FLASH_NYSTROM_REFERENCE_MAX_BYTES", str(10**15))
    q = torch.empty(2, 4, 9216, 64, device="meta")
    _check_reference_budget(q, 128)  # huge budget -> ok


# =========================================================================== #
# _depthwise_conv_residual  (CPU ok)
# =========================================================================== #

@pytest.mark.parametrize("B,H,N,D", [(1, 1, 32, 8), (2, 3, 64, 16), (1, 4, 128, 32)])
@pytest.mark.parametrize("ks", [1, 3, 5, 7])
def test_depthwise_conv_shape(B, H, N, D, ks):
    v = torch.randn(B, H, N, D)
    w = torch.randn(H, ks) * 0.02
    out = _depthwise_conv_residual(v, w)
    assert out.shape == (B, H, N, D) and torch.isfinite(out).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_depthwise_conv_dtype_matches_v(dtype):
    if dtype == torch.float16 and not HAS_GPU:
        pytest.skip("fp16 conv1d needs GPU")
    dev = "cuda" if HAS_GPU else "cpu"
    v = torch.randn(1, 2, 32, 8, device=dev, dtype=dtype)
    w = torch.randn(2, 3, device=dev, dtype=torch.float32)  # weight fp32 -> cast internally
    out = _depthwise_conv_residual(v, w)
    assert out.dtype == v.dtype


def test_depthwise_conv_ident_kernel():
    # a kernel [0,1,0] is identity (per-head, all channels share the weight)
    v = torch.randn(1, 2, 16, 4)
    w = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    out = _depthwise_conv_residual(v, w)
    torch.testing.assert_close(out, v, rtol=1e-5, atol=1e-5)


def test_depthwise_conv_per_head_weights_independent():
    v = torch.zeros(1, 2, 8, 3)
    v[0, 0, 4, :] = 1.0  # impulse in head 0 only
    w = torch.tensor([[0.0, 2.0, 0.0], [0.0, 5.0, 0.0]])  # head0 x2, head1 x5
    out = _depthwise_conv_residual(v, w)
    assert torch.allclose(out[0, 0, 4], torch.full((3,), 2.0))
    assert torch.allclose(out[0, 1], torch.zeros(8, 3))


# =========================================================================== #
# module construction (no forward)
# =========================================================================== #

@pytest.mark.parametrize("dim,heads", [(64, 1), (128, 2), (256, 4), (512, 8)])
def test_module_construct(dim, heads):
    mod = FlashNystromAttention(dim, heads=heads)
    assert mod.head_dim == dim // heads
    assert mod.q_proj.weight.shape == (dim, dim)


def test_module_default_config():
    mod = FlashNystromAttention(64, heads=1)
    assert isinstance(mod.config, NystromConfig)


def test_module_conv_weight_present_when_enabled():
    cfg = NystromConfig(use_conv_residual=True, conv_kernel_size=3)
    mod = FlashNystromAttention(64, heads=2, config=cfg)
    assert mod.conv_weight is not None and mod.conv_weight.shape == (2, 3)


def test_module_conv_weight_absent_when_disabled():
    cfg = NystromConfig(use_conv_residual=False, conv_kernel_size=0)
    mod = FlashNystromAttention(64, heads=2, config=cfg)
    assert mod.conv_weight is None


def test_module_dim_not_divisible_by_heads_rejected():
    with pytest.raises(AssertionError):
        FlashNystromAttention(65, heads=2)


# =========================================================================== #
# CPU fallback path (no GPU needed -- routes to the reference)
# =========================================================================== #

@pytest.mark.parametrize("m", [8, 16, 32, 64])
def test_cpu_path_runs_reference(m):
    q, k, v = (torch.randn(1, 2, 96, 16) for _ in range(3))  # cpu tensors
    out = flash_nystrom_attention(q, k, v, num_landmarks=m, newton_iter=6)
    assert out.shape == (1, 2, 96, 16) and torch.isfinite(out).all()


def test_cpu_path_equals_reference():
    q, k, v = (torch.randn(1, 1, 128, 16, generator=torch.Generator().manual_seed(s)) for s in (1, 2, 3))
    a = flash_nystrom_attention(q, k, v, 32, 6)
    b = nystrom_attention_reference(q, k, v, 32, 6)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("m", [100, 128])
def test_cpu_path_large_m(m):
    q, k, v = (torch.randn(1, 1, 200, 16) for _ in range(3))
    out = flash_nystrom_attention(q, k, v, num_landmarks=m, newton_iter=6)
    assert out.shape == (1, 1, 200, 16)


# =========================================================================== #
# GPU: m>64 routes to the reference + budget guard
# =========================================================================== #

@gpu
@pytest.mark.parametrize("m", [65, 96, 128])
def test_gpu_large_m_routes_reference(m):
    q, k, v = (torch.randn(1, 2, 256, 64, device="cuda") for _ in range(3))
    out = flash_nystrom_attention(q, k, v, num_landmarks=m, newton_iter=6)
    assert out.shape == (1, 2, 256, 64) and torch.isfinite(out).all()


@gpu
def test_gpu_large_m_budget_exceeded_raises(monkeypatch):
    # Force the guard with a 1-byte budget on a TINY tensor -- never allocate a big
    # GPU tensor here (a large device='cuda' empty() hard-crashes an 8GB host).
    monkeypatch.setenv("FLASH_NYSTROM_REFERENCE_MAX_BYTES", "1")
    q, k, v = (torch.randn(1, 1, 128, 64, device="cuda") for _ in range(3))
    with pytest.raises(RuntimeError, match="reference"):
        flash_nystrom_attention(q, k, v, num_landmarks=128, newton_iter=6)


# =========================================================================== #
# GPU: custom kernel forward -- shape, finiteness, match reference
# =========================================================================== #

@gpu
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("m", [1, 8, 16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gpu_forward_shape_finite(D, m, dtype):
    q, k, v = (torch.randn(1, 2, 256, D, device="cuda", dtype=dtype) for _ in range(3))
    out = flash_nystrom_attention(q, k, v, num_landmarks=m, newton_iter=6)
    assert out.shape == (1, 2, 256, D) and torch.isfinite(out).all()


@gpu
@pytest.mark.parametrize("m", [16, 32, 64])
@pytest.mark.parametrize("kappa", [0.0, 1e3])
def test_gpu_forward_matches_reference(m, kappa):
    torch.manual_seed(m)
    q = torch.randn(1, 2, 256, 64, device="cuda")
    k = torch.randn(1, 2, 256, 64, device="cuda")
    v = torch.randn(1, 2, 256, 64, device="cuda")
    ref = nystrom_attention_reference(q.float(), k.float(), v.float(), m, 6, kappa_star=kappa)
    got = flash_nystrom_attention(q.half(), k.half(), v.half(), m, 6, kappa_star=kappa)
    rel = (got.float() - ref).norm() / ref.norm()
    assert rel < 5e-2, rel


@gpu
@pytest.mark.parametrize("D", [64, 128])
def test_gpu_fp32_path(D):
    if D == 128 and not FP32_D128_OK:
        pytest.skip("fp32 D=128 scalar kernel needs ~150KB opt-in SMEM (datacenter GPU only)")
    q, k, v = (torch.randn(1, 1, 128, D, device="cuda", dtype=torch.float32) for _ in range(3))
    out = flash_nystrom_attention(q, k, v, 32, 6)
    assert out.shape == (1, 1, 128, D) and torch.isfinite(out).all()


@gpu
@pytest.mark.parametrize("j", [1, 3, 6, 10, 16])
def test_gpu_newton_iters(j):
    q, k, v = (torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    out = flash_nystrom_attention(q, k, v, 64, j)
    assert torch.isfinite(out).all()


@gpu
def test_gpu_use_tc_pinv_at_m64():
    q, k, v = (torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    out = flash_nystrom_attention(q, k, v, 64, 6, use_tc_pinv=True)
    assert torch.isfinite(out).all()


# =========================================================================== #
# GPU: backward
# =========================================================================== #

@gpu
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("m", [16, 32, 64])
def test_gpu_backward_finite(D, m):
    q = torch.randn(1, 2, 256, D, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 2, 256, D, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 2, 256, D, device="cuda", dtype=torch.float16, requires_grad=True)
    flash_nystrom_attention(q, k, v, m, 6).sum().backward()
    for t in (q, k, v):
        assert t.grad is not None and torch.isfinite(t.grad).all()


@gpu
@pytest.mark.parametrize("fast", [True, False])
def test_gpu_backward_fast_dk2inv_flag(fast):
    q = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    flash_nystrom_attention(q, k, v, 64, 6, fast_dk2inv=fast).sum().backward()
    assert torch.isfinite(q.grad).all()


@gpu
def test_gpu_backward_single_threaded_traced():
    # Autograd normally runs Function.backward on a native device thread that
    # coverage.py cannot trace. Forcing the single-threaded engine runs the
    # backward on THIS thread, so it is both exercised and visible to coverage.
    with torch.autograd.set_multithreading_enabled(False):
        q = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        flash_nystrom_attention(q, k, v, 64, 6).sum().backward()
    for t in (q, k, v):
        assert t.grad is not None and torch.isfinite(t.grad).all()


# =========================================================================== #
# GPU: module forward/backward
# =========================================================================== #

@gpu
@pytest.mark.parametrize("dim,heads", [(64, 1), (128, 2), (256, 4)])
@pytest.mark.parametrize("conv", [True, False])
def test_gpu_module_forward(dim, heads, conv):
    cfg = NystromConfig(num_landmarks=64, use_conv_residual=conv,
                        conv_kernel_size=3 if conv else 0)
    mod = FlashNystromAttention(dim, heads=heads, config=cfg).cuda().half()
    x = torch.randn(2, 256, dim, device="cuda", dtype=torch.float16)
    out = mod(x)
    assert out.shape == (2, 256, dim) and torch.isfinite(out).all()


@gpu
def test_gpu_module_backward():
    mod = FlashNystromAttention(128, heads=2, config=NystromConfig(num_landmarks=64)).cuda().half()
    x = torch.randn(2, 256, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    mod(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
