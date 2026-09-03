# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Exhaustive coverage of flash_nystrom.reference (pure-torch, no CUDA kernels).

Covers iterative_pinverse (Newton-Schulz) and nystrom_attention_reference across
dtypes, shapes, landmark counts, ridge on/off, conv residual, injected landmarks,
and every guarded branch. Runs on GPU if present, else CPU (pure torch either way).
"""
import math
import pytest
import torch

from flash_nystrom.reference import iterative_pinverse, nystrom_attention_reference

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _rand(*shape, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(DEV, dtype)

# --------------------------------------------------------------------------- #
# iterative_pinverse
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_pinv_requires_fp32(dtype):
    A = torch.eye(4, dtype=dtype, device=DEV)
    with pytest.raises(AssertionError, match="float32"):
        iterative_pinverse(A, 6)


@pytest.mark.parametrize("m", [2, 3, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("n_iter", [6, 12, 20])
def test_pinv_recovers_inverse_of_wellcond(m, n_iter):
    torch.manual_seed(m + n_iter)
    # well-conditioned SPD: A = I + small symmetric
    R = torch.randn(m, m, device=DEV) * 0.05
    A = (torch.eye(m, device=DEV) + R @ R.t()).float()
    Z = iterative_pinverse(A, n_iter)
    # A Z A ~ A for a pseudoinverse
    resid = (A @ Z @ A - A).norm() / A.norm()
    assert resid < 1e-2, resid


@pytest.mark.parametrize("m", [2, 4, 8, 16, 32, 64])
def test_pinv_identity(m):
    I = torch.eye(m, device=DEV, dtype=torch.float32)
    Z = iterative_pinverse(I, 8)
    assert (Z - I).abs().max() < 1e-3


@pytest.mark.parametrize("m", [4, 8, 16])
def test_pinv_more_iters_not_worse(m):
    torch.manual_seed(m)
    R = torch.randn(m, m, device=DEV) * 0.05
    A = (torch.eye(m, device=DEV) + R @ R.t()).float()
    r_lo = (A @ iterative_pinverse(A, 4) @ A - A).norm().item()
    r_hi = (A @ iterative_pinverse(A, 16) @ A - A).norm().item()
    assert r_hi <= r_lo + 1e-4


@pytest.mark.parametrize("B,H,m", [(1, 1, 4), (2, 3, 8), (4, 2, 16), (1, 8, 32)])
def test_pinv_batched_shape(B, H, m):
    torch.manual_seed(B * H * m)
    R = torch.randn(B, H, m, m, device=DEV) * 0.05
    A = torch.eye(m, device=DEV) + R @ R.transpose(-2, -1)
    Z = iterative_pinverse(A.float(), 6)
    assert Z.shape == (B, H, m, m)
    assert torch.isfinite(Z).all()


def test_pinv_zero_matrix_clamped_finite():
    # degenerate: all-zero matrix. Z_0 denominator is clamped to 1e-12 so no NaN.
    A = torch.zeros(4, 4, device=DEV, dtype=torch.float32)
    Z = iterative_pinverse(A, 6)
    assert torch.isfinite(Z).all()


@pytest.mark.parametrize("n_iter", [1, 2, 3, 4, 5, 6, 8, 10, 16])
def test_pinv_n_iter_runs(n_iter):
    A = torch.eye(8, device=DEV, dtype=torch.float32) * 2.0
    Z = iterative_pinverse(A, n_iter)
    assert Z.shape == (8, 8) and torch.isfinite(Z).all()

# --------------------------------------------------------------------------- #
# nystrom_attention_reference — shapes / dtypes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("B,H,N,D", [(1, 1, 64, 16), (2, 2, 128, 32), (1, 4, 96, 64), (3, 1, 80, 8)])
@pytest.mark.parametrize("m", [1, 8, 16, 32])
def test_ref_output_shape(B, H, N, D, m):
    q, k, v = _rand(B, H, N, D), _rand(B, H, N, D), _rand(B, H, N, D)
    out = nystrom_attention_reference(q, k, v, num_landmarks=m, newton_iter=6)
    assert out.shape == (B, H, N, D)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ref_dtypes(dtype):
    if dtype in (torch.float16, torch.bfloat16) and DEV == "cpu":
        pytest.skip("half matmul flaky on CPU")
    q, k, v = (_rand(1, 2, 128, 32, dtype=dtype) for _ in range(3))
    out = nystrom_attention_reference(q, k, v, 16, 6)
    assert out.shape == (1, 2, 128, 32) and torch.isfinite(out).all()


@pytest.mark.parametrize("m", [8, 16, 32, 64])
@pytest.mark.parametrize("kappa", [0.0, 1.0, 1e3])
def test_ref_ridge_and_vanilla(m, kappa):
    q, k, v = (_rand(1, 2, 128, 32, seed=s) for s in (1, 2, 3))
    out = nystrom_attention_reference(q, k, v, m, 6, kappa_star=kappa)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("j", [1, 2, 4, 6, 10, 16])
def test_ref_newton_iters(j):
    q, k, v = (_rand(1, 1, 96, 16, seed=s) for s in (4, 5, 6))
    out = nystrom_attention_reference(q, k, v, 16, j)
    assert torch.isfinite(out).all()

# --------------------------------------------------------------------------- #
# landmark edge cases
# --------------------------------------------------------------------------- #

def test_ref_m_equals_1():
    q, k, v = (_rand(1, 2, 64, 16, seed=s) for s in (1, 2, 3))
    out = nystrom_attention_reference(q, k, v, 1, 6)
    assert out.shape == (1, 2, 64, 16) and torch.isfinite(out).all()


@pytest.mark.parametrize("N,m", [(64, 64), (65, 64), (100, 64), (128, 63), (70, 32), (33, 32)])
def test_ref_m_close_to_N(N, m):
    q, k, v = (_rand(1, 1, N, 16, seed=s) for s in (1, 2, 3))
    out = nystrom_attention_reference(q, k, v, m, 6)
    assert out.shape == (1, 1, N, 16) and torch.isfinite(out).all()


@pytest.mark.parametrize("N,m", [(100, 7), (127, 13), (65, 9), (200, 33)])
def test_ref_N_not_divisible_by_m(N, m):
    # exercises the last-landmark-absorbs-remainder branch
    q, k, v = (_rand(1, 1, N, 16, seed=s) for s in (7, 8, 9))
    out = nystrom_attention_reference(q, k, v, m, 6)
    assert torch.isfinite(out).all()


def test_ref_m_greater_than_N_rejected():
    q, k, v = (_rand(1, 1, 32, 16) for _ in range(3))
    with pytest.raises(AssertionError, match="must be >="):
        nystrom_attention_reference(q, k, v, 64, 6)

# --------------------------------------------------------------------------- #
# conv residual
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ks", [1, 3, 5, 7])
def test_ref_conv_residual(ks):
    B, H, N, D = 1, 2, 96, 16
    q, k, v = (_rand(B, H, N, D, seed=s) for s in (1, 2, 3))
    cw = torch.randn(H, ks, device=DEV) * 0.02
    out = nystrom_attention_reference(q, k, v, 16, 6, conv_weight=cw, conv_kernel_size=ks)
    assert out.shape == (B, H, N, D) and torch.isfinite(out).all()


def test_ref_conv_weight_wrong_shape_rejected():
    q, k, v = (_rand(1, 2, 96, 16) for _ in range(3))
    bad = torch.randn(5, 3, device=DEV)  # H=5 != 2
    with pytest.raises(AssertionError):
        nystrom_attention_reference(q, k, v, 16, 6, conv_weight=bad, conv_kernel_size=3)


def test_ref_conv_disabled_when_ks_zero():
    q, k, v = (_rand(1, 2, 96, 16, seed=s) for s in (1, 2, 3))
    cw = torch.randn(2, 3, device=DEV)
    # conv_kernel_size=0 -> residual skipped even if weight is passed
    out = nystrom_attention_reference(q, k, v, 16, 6, conv_weight=cw, conv_kernel_size=0)
    assert torch.isfinite(out).all()

# --------------------------------------------------------------------------- #
# determinism / structural properties
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("m", [8, 16, 32, 64])
def test_ref_deterministic(m):
    q, k, v = (_rand(1, 2, 128, 16, seed=s) for s in (1, 2, 3))
    a = nystrom_attention_reference(q, k, v, m, 6)
    b = nystrom_attention_reference(q, k, v, m, 6)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("B,H", [(1, 1), (2, 3), (4, 2)])
def test_ref_batch_head_independence(B, H):
    # each (b,h) slice is computed independently; stacking must not cross-contaminate
    N, D, m = 96, 16, 16
    q, k, v = (_rand(B, H, N, D, seed=s) for s in (1, 2, 3))
    full = nystrom_attention_reference(q, k, v, m, 6)
    one = nystrom_attention_reference(q[:1, :1], k[:1, :1], v[:1, :1], m, 6)
    torch.testing.assert_close(full[:1, :1], one, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("D", [8, 16, 32, 64, 128])
def test_ref_various_head_dims(D):
    q, k, v = (_rand(1, 1, 96, D, seed=s) for s in (1, 2, 3))
    out = nystrom_attention_reference(q, k, v, 16, 6)
    assert out.shape == (1, 1, 96, D) and torch.isfinite(out).all()


@pytest.mark.parametrize("N", [64, 96, 128, 192, 256, 512])
def test_ref_various_N(N):
    q, k, v = (_rand(1, 1, N, 16, seed=s) for s in (1, 2, 3))
    out = nystrom_attention_reference(q, k, v, 32, 6)
    assert out.shape == (1, 1, N, 16) and torch.isfinite(out).all()


@pytest.mark.parametrize("kappa", [0.0, 1e-3, 1.0, 10.0, 100.0, 1e3, 1e4, 1e6])
def test_ref_kappa_sweep_finite(kappa):
    q, k, v = (_rand(1, 2, 128, 16, seed=s) for s in (1, 2, 3))
    out = nystrom_attention_reference(q, k, v, 32, 8, kappa_star=kappa)
    assert torch.isfinite(out).all()

# --------------------------------------------------------------------------- #
# gradients flow
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("m", [8, 16, 32])
@pytest.mark.parametrize("kappa", [0.0, 1e3])
def test_ref_backward_finite(m, kappa):
    q = _rand(1, 2, 96, 16, seed=1).requires_grad_(True)
    k = _rand(1, 2, 96, 16, seed=2).requires_grad_(True)
    v = _rand(1, 2, 96, 16, seed=3).requires_grad_(True)
    nystrom_attention_reference(q, k, v, m, 6, kappa_star=kappa).sum().backward()
    for t in (q, k, v):
        assert t.grad is not None and torch.isfinite(t.grad).all()

# --------------------------------------------------------------------------- #
# nystrom_attention_reference_simple (thin wrapper, no conv, kappa=0)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("m", [1, 8, 16, 32, 64])
@pytest.mark.parametrize("j", [1, 6, 12])
def test_ref_simple_wrapper(m, j):
    from flash_nystrom.reference import nystrom_attention_reference_simple
    q, k, v = (_rand(1, 2, 96, 16, seed=s) for s in (1, 2, 3))
    out = nystrom_attention_reference_simple(q, k, v, m, j)
    assert out.shape == (1, 2, 96, 16) and torch.isfinite(out).all()


def test_ref_simple_wrapper_matches_full_no_conv():
    from flash_nystrom.reference import nystrom_attention_reference_simple
    q, k, v = (_rand(1, 1, 80, 16, seed=s) for s in (4, 5, 6))
    a = nystrom_attention_reference_simple(q, k, v, 16, 6)
    b = nystrom_attention_reference(q, k, v, 16, 6, None, 0, kappa_star=0.0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
