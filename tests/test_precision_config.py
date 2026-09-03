# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
# Regression tests for the config refactor and the precision findings:
#  - kappa_star is threaded identically to the kernel and the reference, is
#    validated, and actually changes the output (guards review #10/#20).
#  - fast_dk2inv does not bias the gradient vs the exact-fp32 dk2inv path
#    (guards review #3 — the precision tradeoff is unbiased, not a regression).
import pytest
import torch
import torch.nn.functional as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30)).item()


@cuda
def test_kappa_star_validated():
    """Invalid kappa_star is rejected at the C entry (no silent inf/neg ridge)."""
    from flash_nystrom.flash_nystrom import _C
    q = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16)
    for bad in (-1.0, float("inf"), float("nan")):
        with pytest.raises(RuntimeError):
            _C.forward(q, q, q, 64, 6, bad, True)
    # config dataclass rejects the same
    from flash_nystrom.nystrom_config import NystromConfig
    for bad in (-1.0, float("inf"), float("nan")):
        with pytest.raises(AssertionError):
            NystromConfig(kappa_star=bad)


@cuda
def test_kappa_star_changes_output():
    """The ridge must actually do something: kappa=5 differs from kappa=0 (off)."""
    from flash_nystrom.flash_nystrom import _C
    torch.manual_seed(0)
    q = torch.randn(2, 4, 1024, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 4, 1024, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 4, 1024, 64, device="cuda", dtype=torch.float16)
    o_off = _C.forward(q, k, v, 64, 6, 0.0, True)[0]
    o_on = _C.forward(q, k, v, 64, 6, 5.0, True)[0]
    assert _rel(o_on, o_off) > 1e-2, "ridge had no effect — kappa_star not wired through"


@cuda
def test_kappa_star_kernel_matches_reference():
    """#20: the kernel and the reference must use the SAME kappa_star. With both
    at 5.0 the m=64 kernel tracks the fp32 reference; a kappa mismatch (kernel 5
    vs reference 0) is far apart. Guards against the old env-var divergence."""
    from flash_nystrom.flash_nystrom import flash_nystrom_attention
    from flash_nystrom.reference import nystrom_attention_reference
    torch.manual_seed(0)
    q = torch.randn(2, 4, 512, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 4, 512, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 4, 512, 64, device="cuda", dtype=torch.float16)
    o_kernel = flash_nystrom_attention(q, k, v, num_landmarks=64, newton_iter=6,
                                       kappa_star=5.0)
    o_ref5 = nystrom_attention_reference(q.float(), k.float(), v.float(), 64, 6,
                                         kappa_star=5.0)
    o_ref0 = nystrom_attention_reference(q.float(), k.float(), v.float(), 64, 6,
                                         kappa_star=0.0)
    # Same kappa: kernel (fp16/tf32) tracks the fp32 reference within the
    # low-precision floor.
    assert _rel(o_kernel, o_ref5) < 2e-2, f"kernel vs same-kappa reference too far: {_rel(o_kernel, o_ref5)}"
    # Mismatched kappa: meaningfully different, proving kappa is not silently 0.
    assert _rel(o_ref5, o_ref0) > 1e-2


@cuda
def test_fast_dk2inv_unbiased():
    """#3: fast_dk2inv (fp16 P before GEMM2) must not BIAS the gradient vs the
    exact fp32 dk2inv path — zero-mean difference, ~unit magnitude ratio."""
    from flash_nystrom.flash_nystrom import FlashNystromFunction
    torch.manual_seed(0)
    B, H, N, D, m = 2, 4, 1024, 64, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    dO = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 1024.0  # scaled (avoid subnormals)

    def grads(fast):
        qg = q.clone().requires_grad_(); kg = k.clone().requires_grad_(); vg = v.clone().requires_grad_()
        FlashNystromFunction.apply(qg, kg, vg, m, 6, fast, 5.0, True).backward(dO)
        return qg.grad, kg.grad, vg.grad

    fast = grads(True)
    slow = grads(False)
    for name, gf, gs in zip("qkv", fast, slow):
        # magnitude ratio near 1 (no systematic shrink/grow)
        nrat = (gf.norm() / gs.norm().clamp(min=1e-30)).item()
        assert 0.9 < nrat < 1.1, f"d{name} norm ratio {nrat} (fast vs exact) — magnitude bias"
        # signed mean of the difference ~0 (unbiased, not a one-directional shift)
        smean = ((gf - gs).float().mean() / gs.float().abs().mean().clamp(min=1e-30)).abs().item()
        assert smean < 5e-2, f"d{name} signed-mean bias {smean} (fast vs exact)"
