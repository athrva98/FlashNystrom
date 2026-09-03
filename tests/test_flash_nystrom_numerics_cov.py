# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""GPU numerical-correctness tests: the CUDA kernels vs the pure-PyTorch
reference (the same Nystrom math via cuBLAS/autograd).

The fp32 scalar Newton-Schulz path is faithful to ~1e-6 on the forward and
~1e-5 on the gradients (measured), so those get tight tolerances. fp16/bf16
tensor-core paths carry real precision noise and get looser bounds. All sizes
are tiny (N<=256, D=64) -- safe on the 8GB card.
"""
import pytest
import torch

from flash_nystrom.flash_nystrom import flash_nystrom_attention
from flash_nystrom.reference import nystrom_attention_reference

HAS_GPU = torch.cuda.is_available()
gpu = pytest.mark.skipif(not HAS_GPU, reason="needs CUDA + built _C")


def _triple(N, D, seed, dtype=torch.float32):
    torch.manual_seed(seed)
    return [torch.randn(1, 2, N, D, device="cuda", dtype=dtype) for _ in range(3)]


# =========================================================================== #
# fp32 forward: tight match to the reference (scalar Newton-Schulz path)
# =========================================================================== #

@gpu
@pytest.mark.parametrize("m", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("N", [128, 256])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fp32_forward_matches_reference_tight(m, N, seed):
    q, k, v = _triple(N, 64, seed)
    ref = nystrom_attention_reference(q.clone(), k.clone(), v.clone(), m, 6)
    got = flash_nystrom_attention(q.clone(), k.clone(), v.clone(),
                                  num_landmarks=m, newton_iter=6)
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 1e-4, rel  # measured ~7e-7


@gpu
@pytest.mark.parametrize("m", [16, 32, 64])
@pytest.mark.parametrize("kappa", [0.0, 1.0, 1e3])
def test_fp32_forward_matches_reference_with_ridge(m, kappa):
    q, k, v = _triple(256, 64, 0)
    ref = nystrom_attention_reference(q.clone(), k.clone(), v.clone(), m, 6, kappa_star=kappa)
    got = flash_nystrom_attention(q.clone(), k.clone(), v.clone(),
                                  num_landmarks=m, newton_iter=6, kappa_star=kappa)
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 1e-3, rel


@gpu
@pytest.mark.parametrize("j", [3, 6, 10, 16])
def test_fp32_forward_matches_reference_newton_iters(j):
    q, k, v = _triple(256, 64, 1)
    ref = nystrom_attention_reference(q.clone(), k.clone(), v.clone(), 64, j)
    got = flash_nystrom_attention(q.clone(), k.clone(), v.clone(), num_landmarks=64, newton_iter=j)
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 1e-4, rel


# =========================================================================== #
# fp32 backward: gradients match the reference autograd (dQ, dK, dV)
# =========================================================================== #

@gpu
@pytest.mark.parametrize("m", [16, 32, 64])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fp32_backward_matches_reference(m, seed):
    base = _triple(128, 64, seed)
    qs = [b.clone().requires_grad_(True) for b in base]
    qr = [b.clone().requires_grad_(True) for b in base]
    flash_nystrom_attention(*qs, num_landmarks=m, newton_iter=6).pow(2).sum().backward()
    nystrom_attention_reference(*qr, m, 6).pow(2).sum().backward()
    for name, a, b in zip("QKV", qs, qr):
        rel = ((a.grad - b.grad).norm() / (b.grad.norm() + 1e-9)).item()
        assert rel < 2e-3, f"d{name} rel={rel}"


@gpu
@pytest.mark.parametrize("kappa", [0.0, 1e3])
def test_fp32_backward_matches_reference_ridge(kappa):
    base = _triple(128, 64, 0)
    qs = [b.clone().requires_grad_(True) for b in base]
    qr = [b.clone().requires_grad_(True) for b in base]
    flash_nystrom_attention(*qs, num_landmarks=64, newton_iter=6, kappa_star=kappa).pow(2).sum().backward()
    nystrom_attention_reference(*qr, 64, 6, kappa_star=kappa).pow(2).sum().backward()
    for name, a, b in zip("QKV", qs, qr):
        rel = ((a.grad - b.grad).norm() / (b.grad.norm() + 1e-9)).item()
        assert rel < 1e-2, f"d{name} rel={rel}"


# =========================================================================== #
# fp16 / bf16 forward: looser match (tensor-core precision noise)
# =========================================================================== #

@gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [8, 16, 32, 64])
def test_half_forward_matches_reference(dtype, m):
    q, k, v = _triple(256, 64, 0)
    ref = nystrom_attention_reference(q.clone(), k.clone(), v.clone(), m, 6)
    got = flash_nystrom_attention(q.to(dtype), k.to(dtype), v.to(dtype),
                                  num_landmarks=m, newton_iter=6)
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    tol = 6e-2 if dtype == torch.float16 else 8e-2
    assert rel < tol, rel


@gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tc_pinv_forward_matches_reference_loose(dtype):
    # tf32 tensor-core pinv path carries an N-independent ~1-3% pinv error
    q, k, v = _triple(256, 64, 0)
    ref = nystrom_attention_reference(q.clone(), k.clone(), v.clone(), 64, 6)
    got = flash_nystrom_attention(q.to(dtype), k.to(dtype), v.to(dtype),
                                  num_landmarks=64, newton_iter=6, use_tc_pinv=True)
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    assert rel < 1e-1, rel


# =========================================================================== #
# determinism (same input -> identical output, both fp32 and half)
# =========================================================================== #

@gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [16, 32, 64])
def test_forward_deterministic(dtype, m):
    q, k, v = _triple(256, 64, 3, dtype=dtype)
    a = flash_nystrom_attention(q, k, v, num_landmarks=m, newton_iter=6)
    b = flash_nystrom_attention(q, k, v, num_landmarks=m, newton_iter=6)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


# =========================================================================== #
# conv residual: kernel + cuDNN conv matches reference + conv
# =========================================================================== #

@gpu
@pytest.mark.parametrize("ks", [3, 5])
@pytest.mark.parametrize("m", [32, 64])
def test_fp32_conv_residual_matches_reference(ks, m):
    q, k, v = _triple(256, 64, 0)
    cw = torch.randn(2, ks, device="cuda", dtype=torch.float32) * 0.02
    ref = nystrom_attention_reference(q.clone(), k.clone(), v.clone(), m, 6,
                                      cw.clone(), ks)
    got = flash_nystrom_attention(q.clone(), k.clone(), v.clone(),
                                  num_landmarks=m, newton_iter=6,
                                  conv_weight=cw.clone(), conv_kernel_size=ks)
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 1e-3, rel


# =========================================================================== #
# batch / head independence on GPU (a slice computes the same alone or stacked)
# =========================================================================== #

@gpu
@pytest.mark.parametrize("m", [16, 64])
def test_gpu_batch_head_independence(m):
    torch.manual_seed(0)
    q = torch.randn(3, 4, 256, 64, device="cuda", dtype=torch.float32)
    k = torch.randn(3, 4, 256, 64, device="cuda", dtype=torch.float32)
    v = torch.randn(3, 4, 256, 64, device="cuda", dtype=torch.float32)
    full = flash_nystrom_attention(q, k, v, num_landmarks=m, newton_iter=6)
    one = flash_nystrom_attention(q[1:2, 2:3].contiguous(), k[1:2, 2:3].contiguous(),
                                  v[1:2, 2:3].contiguous(), num_landmarks=m, newton_iter=6)
    rel = ((full[1:2, 2:3] - one).norm() / one.norm()).item()
    assert rel < 1e-4, rel
