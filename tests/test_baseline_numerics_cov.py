# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Numerical regressions for the bidirectional baselines at long context.

Linear attention's two accumulators are sums over N terms, so both grow like N
and leave fp16's range (max 65504) inside the lengths this paper trains at. The
normalizer goes first, around N=2304, and the failure is SILENT: den overflows
to inf, num/inf is 0, and the arm emits all-zero attention with a loss of
exactly 0 rather than crashing. The numerator follows near N=65536, which is
the NaN. These tests pin both, plus the single-definition rule that let the two
copies of the operator drift apart.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from benchmarks.baseline_attn import LinearAttention, _fp16_safe_scale
from benchmarks.baseline_ops import linear_attention_op

LONG = [2304, 9216, 32401]          # the vision context tiers


def _ref(m, x):
    """Same math in fp64 with autocast off: what the fp16 path must match."""
    with torch.amp.autocast("cpu", enabled=False):
        q, k, v = m._qkv(x.double())
        qf, kf = F.elu(q) + 1.0, F.elu(k) + 1.0
        num = qf @ (kf.transpose(-2, -1) @ v)
        den = (qf @ kf.sum(dim=-2).unsqueeze(-1)) + 1e-6
        return m._merge(num / den)


# --------------------------------------------------------------------------- #
# the scale helper
# --------------------------------------------------------------------------- #

def test_scale_is_identity_for_wide_dtypes():
    # bf16 and fp32 carry ~3.4e38; only fp16 needs protecting
    for dt in (torch.bfloat16, torch.float32, torch.float64):
        assert _fp16_safe_scale(10 ** 6, dt) == 1.0


def test_scale_is_identity_for_short_fp16():
    """No scaling below the bound: short contexts never leave fp16 range."""
    assert _fp16_safe_scale(512, torch.float16) == 1.0
    assert _fp16_safe_scale(256, torch.float16) == 1.0
    assert _fp16_safe_scale(1024, torch.float16) == 0.5     # past the bound


@pytest.mark.parametrize("n", [8192, 32401, 65536, 10 ** 6])
def test_scale_keeps_fp16_in_range_without_underflow(n):
    s = _fp16_safe_scale(n, torch.float16)
    assert 0.0 < s <= 1.0
    assert math.log2(s).is_integer()          # exact in binary floating point
    assert n * s <= 512                       # numerator bounded (n*s*D << 65504)
    assert s > 6.1e-5 * 5                     # clear of fp16's smallest normal


# --------------------------------------------------------------------------- #
# the operator itself
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [512, 2304, 9216])
def test_normalizer_does_not_overflow_fp16(n):
    """The regression is in den = phi(q).z, which accumulates over N AND over D:
    z alone still fits fp16 at these lengths, den does not."""
    qf = kf = torch.full((1, 2, n, 64), 1.5, dtype=torch.float16)
    z16 = kf.sum(dim=-2)
    den16 = qf @ z16.unsqueeze(-1)                  # the old path
    assert not torch.isfinite(den16).all(), "expected the fp16 denominator to blow up"
    z32 = kf.sum(dim=-2, dtype=torch.float32)       # the fix
    den32 = qf.float() @ z32.unsqueeze(-1)
    assert torch.isfinite(den32).all()
    assert float(den32.max()) == pytest.approx(1.5 * 1.5 * n * 64, rel=1e-3)


@pytest.mark.parametrize("n", LONG)
def test_linear_attention_output_is_not_silently_zero(n):
    """num/inf = 0 gave an all-zero output and a loss of exactly 0: an arm that
    trains nothing while reporting no error."""
    torch.manual_seed(0)
    m = LinearAttention(128, 2)
    x = torch.randn(1, n, 128)
    with torch.amp.autocast("cpu", dtype=torch.float16):
        out = m(x)
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) > 0.0


@pytest.mark.parametrize("n", LONG)
def test_linear_attention_matches_fp64_reference(n):
    torch.manual_seed(0)
    m = LinearAttention(128, 2)
    x = torch.randn(1, n, 128)
    with torch.amp.autocast("cpu", dtype=torch.float16):
        out = m(x)
    ref = _ref(m.double(), x)
    assert float((out.double() - ref).norm() / ref.norm()) < 5e-2


def test_linear_attention_gradients_are_finite_at_long_context():
    torch.manual_seed(0)
    m = LinearAttention(128, 2)
    x = torch.randn(1, 9216, 128, requires_grad=True)
    with torch.amp.autocast("cpu", dtype=torch.float16):
        out = m(x)
    (out.float().pow(2).mean() * 65536.0).backward()
    assert torch.isfinite(x.grad).all() and float(x.grad.abs().max()) > 0.0


def test_bare_op_normalizer_is_fp32():
    """baseline_ops feeds the latency table; it must not return NaN at the
    lengths the paper reports."""
    q = k = v = torch.full((1, 2, 8192, 64), 0.5, dtype=torch.float16)
    out = linear_attention_op(q, k, v)
    assert torch.isfinite(out).all()
    assert out.dtype == torch.float16


# --------------------------------------------------------------------------- #
# one definition, not two
# --------------------------------------------------------------------------- #

def test_every_harness_gets_the_same_linear_attention():
    """paper/mqar/model.py once carried a SECOND copy that never called the
    fused kernel and kept the overflowing fp16 accumulation, so MQAR and
    genomics silently ran a different, broken operator from vision."""
    from paper.mqar.model import build_attention
    m = build_attention("linear_attention", 128, 2, seq_len=512)
    assert type(m) is LinearAttention


def test_model_module_defines_no_rival_linear_attention():
    import paper.mqar.model as mm
    assert not any(n == "LinearAttention" and getattr(mm, n) is not LinearAttention
                   for n in dir(mm))
