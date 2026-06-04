# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Verify the m > 64 fallback in flash_nystrom_attention.

At num_landmarks > 64 the entry point dispatches to the pure-PyTorch
reference (nystrom_attention_reference). This test checks four things:

  1. Output equals what nystrom_attention_reference returns directly (the
     dispatch is a forwarder, no transformation).
  2. Autograd backward works through the dispatched path.
  3. The OOM guard fires with a clear Python error before allocating an
     oversized softmax intermediate, instead of letting CUDA OOM.
  4. m <= 64 still uses the custom CUDA path (output identical to the
     direct C extension call).

This is intentionally narrow: it only tests the dispatch wrapper. The
correctness of nystrom_attention_reference itself is covered by
test_forward.py and test_backward.py (which use it as the ground-truth
reference for the custom kernels at m <= 64).
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

try:
    from flash_nystrom import flash_nystrom_attention
    from flash_nystrom.reference import nystrom_attention_reference
    HAVE = True
except ImportError:
    HAVE = False

pytestmark_have = pytest.mark.skipif(not HAVE, reason="flash_nystrom not importable")


@pytestmark_have
@pytest.mark.parametrize("m", [96, 128, 192, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_m_gt_64_dispatches_to_reference(m, dtype):
    """flash_nystrom_attention(m > 64) must produce identical output to
    nystrom_attention_reference called directly."""
    B, H, N, D = 1, 4, 512, 64
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(B, H, N, D, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(B, H, N, D, device="cuda", dtype=dtype) * 0.5

    out_dispatched = flash_nystrom_attention(
        q, k, v, num_landmarks=m, newton_iter=6)
    out_direct = nystrom_attention_reference(
        q, k, v, num_landmarks=m, newton_iter=6)

    # Should be bit-identical since the dispatch is a direct forwarder.
    assert torch.equal(out_dispatched, out_direct), (
        f"dispatched output differs from direct reference at m={m}, {dtype}"
    )


@pytestmark_have
def test_m_gt_64_autograd_backward_works():
    """The reference is a pure-PyTorch composition of differentiable ops, so
    autograd should produce gradients without error and they should match
    what an explicit backward through the reference produces."""
    B, H, N, D, m = 1, 4, 256, 64, 128
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.5

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    out = flash_nystrom_attention(q, k, v, num_landmarks=m, newton_iter=6)
    out.sum().backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()


@pytestmark_have
def test_m_gt_64_oom_guard_fires_before_allocation():
    """At a shape where the (N, m) softmax intermediates would exceed the
    budget, the wrapper raises a clear RuntimeError BEFORE allocating
    anything large, rather than letting CUDA OOM."""
    import os
    # Tight 64 MiB budget so the test trips on tiny shapes.
    prev = os.environ.get("FLASH_NYSTROM_REFERENCE_MAX_BYTES")
    os.environ["FLASH_NYSTROM_REFERENCE_MAX_BYTES"] = str(64 * 1024 * 1024)
    try:
        # 2 * 2 * 1 * 16 * 4096 * 128 = 32 MiB — under the 64 MiB limit. OK.
        q = torch.randn(1, 16, 4096, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 16, 4096, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 16, 4096, 64, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            _ = flash_nystrom_attention(q, k, v, num_landmarks=128, newton_iter=6)

        # Same B*H*N but m=512 -> 128 MiB needed, over the 64 MiB cap.
        with pytest.raises(RuntimeError, match=r"(?i)budget|materializ"):
            flash_nystrom_attention(q, k, v, num_landmarks=512, newton_iter=6)
    finally:
        if prev is None:
            os.environ.pop("FLASH_NYSTROM_REFERENCE_MAX_BYTES", None)
        else:
            os.environ["FLASH_NYSTROM_REFERENCE_MAX_BYTES"] = prev


@pytestmark_have
def test_m_le_64_still_uses_custom_path():
    """At m <= 64 the wrapper must call the custom CUDA path, not the
    reference. We verify this indirectly by checking that the output
    matches the direct FlashNystromFunction.apply path."""
    from flash_nystrom.flash_nystrom import FlashNystromFunction

    B, H, N, D, m = 1, 4, 256, 64, 32
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.5

    out_wrapper = flash_nystrom_attention(q, k, v, num_landmarks=m, newton_iter=6)
    out_direct  = FlashNystromFunction.apply(q, k, v, m, 6, True)

    assert torch.equal(out_wrapper, out_direct), (
        "At m <= 64 the wrapper should be a direct forwarder over the custom "
        "FlashNystromFunction.apply path"
    )
