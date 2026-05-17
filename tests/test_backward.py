# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

"""Tests for FlashNystrom backward pass correctness."""

import pytest
import torch
from torch.autograd import gradcheck

from flash_nystrom.reference import nystrom_attention_reference_simple


class TestReferenceBackward:

    def test_gradient_finite(self):
        B, H, N, D, m = 1, 1, 64, 32, 16
        q = torch.randn(B, H, N, D, requires_grad=True)
        k = torch.randn(B, H, N, D, requires_grad=True)
        v = torch.randn(B, H, N, D, requires_grad=True)
        out = nystrom_attention_reference_simple(q, k, v, m)
        out.sum().backward()
        for name, p in [("q", q), ("k", k), ("v", v)]:
            assert p.grad is not None and not torch.isnan(p.grad).any()

    def test_gradcheck_v_only(self):
        B, H, N, D, m = 1, 1, 32, 16, 8
        torch.manual_seed(0)
        q = torch.randn(B, H, N, D, dtype=torch.float64)
        k = torch.randn(B, H, N, D, dtype=torch.float64)
        v = torch.randn(B, H, N, D, dtype=torch.float64, requires_grad=True)
        assert gradcheck(lambda v_: nystrom_attention_reference_simple(q, k, v_, m),
                         (v,), eps=1e-6, atol=1e-3, rtol=1e-2)

    def test_reference_qk_gradient_direction(self):
        """Verify Q/K gradients from the reference agree with finite
        differences via cosine similarity over a sample.

        The reference forward downcasts the Newton-Schulz pseudoinverse
        to FP32 internally (matching the CUDA kernel; see reference.py
        line ~103). That sets the effective precision of the function
        at FP32, not the dtype of the input tensors. A small FD
        perturbation in FP64 (eps=1e-5) sits below the FP32 NS noise
        floor and FD picks up noise rather than the gradient, which is
        what made earlier versions of this test fail on A100 but not
        on consumer Blackwell.

        Fix: run the check in FP32 with eps=1e-3, large enough that the
        perturbation propagates through the NS chain with signal-to-
        noise ratio above 1. Sample 32 elements and check cosine
        similarity rather than per-point sign so the result averages
        out per-point flakiness.
        """
        B, H, N, D, m = 1, 1, 32, 16, 8
        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float32, requires_grad=True)

        out = nystrom_attention_reference_simple(q, k, v, m)
        out.sum().backward()

        # Gradients themselves should be finite. Cheap sanity check before
        # the more expensive FD comparison below.
        for name, p in [("q", q), ("k", k), ("v", v)]:
            assert p.grad is not None, f"{name}.grad is None"
            assert not torch.isnan(p.grad).any(), f"{name}.grad has NaN"
            assert p.grad.abs().max() > 0, f"{name}.grad is all zeros"

        eps = 1e-3  # large enough to clear the FP32 NS noise floor
        n_samples = 32
        rng = torch.Generator().manual_seed(0)
        for name, param in [("q", q), ("k", k)]:
            flat_idx = torch.randperm(param.numel(), generator=rng)[:n_samples]
            analytical = []
            numerical = []
            for fi in flat_idx.tolist():
                # Unravel flat index into the 4D (B, H, N, D) tuple.
                i0 = fi // (H * N * D); rem = fi % (H * N * D)
                i1 = rem // (N * D);    rem = rem % (N * D)
                i2 = rem // D;          i3 = rem % D
                idx = (i0, i1, i2, i3)

                analytical.append(param.grad[idx].item())
                saved = param.data.clone()
                param.data[idx] += eps
                out_plus = nystrom_attention_reference_simple(q, k, v, m).sum().item()
                param.data[idx] -= 2 * eps
                out_minus = nystrom_attention_reference_simple(q, k, v, m).sum().item()
                param.data.copy_(saved)
                numerical.append((out_plus - out_minus) / (2 * eps))

            a = torch.tensor(analytical, dtype=torch.float64)
            n = torch.tensor(numerical, dtype=torch.float64)
            cos = torch.nn.functional.cosine_similarity(a, n, dim=0).item()
            # 0.5 is a loose floor; a correct gradient through the FP32 NS
            # chain typically lands in [0.7, 0.95]. A real backward bug
            # gives cosine near zero or negative.
            assert cos > 0.5, (
                f"{name}: grad-vs-FD cosine similarity {cos:.3f} below 0.5; "
                f"sample size {n_samples}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDABackward:

    def test_backward_produces_gradients(self):
        """Verify backward produces non-zero, finite gradients."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        B, H, N, D, m = 1, 2, 128, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)

        out = FlashNystromFunction.apply(q, k, v, m, 6, False)
        out.sum().backward()

        for name, p in [("q", q), ("k", k), ("v", v)]:
            assert p.grad is not None, f"{name}.grad is None"
            assert not torch.isnan(p.grad).any(), f"{name}.grad has NaN"
            assert not torch.isinf(p.grad).any(), f"{name}.grad has Inf"
            assert p.grad.abs().max() > 0, f"{name}.grad is all zeros"

    def test_dv_matches_reference(self):
        """dV should match reference closely (no N-S path)."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 1, 2, 128, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        v_cuda = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)

        out = FlashNystromFunction.apply(q, k, v_cuda, m, 6, False)
        out.sum().backward()

        v_ref = v_cuda.detach().float().cpu().requires_grad_(True)
        out_ref = nystrom_attention_reference_simple(
            q.float().cpu(), k.float().cpu(), v_ref, m)
        out_ref.sum().backward()

        cos_sim = torch.nn.functional.cosine_similarity(
            v_cuda.grad.float().cpu().flatten().unsqueeze(0),
            v_ref.grad.flatten().unsqueeze(0)
        ).item()
        assert cos_sim > 0.99, f"dV cosine sim {cos_sim:.4f} too low"

    def test_backward_d128_cosine(self):
        """D=128 gradients should match reference (cosine > 0.9)."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(0)
        B, H, N, D, m = 1, 2, 256, 128, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()

        q2 = q.detach().float().cpu().requires_grad_(True)
        k2 = k.detach().float().cpu().requires_grad_(True)
        v2 = v.detach().float().cpu().requires_grad_(True)
        nystrom_attention_reference_simple(q2, k2, v2, m).sum().backward()

        def cos(a, b):
            return torch.nn.functional.cosine_similarity(
                a.float().flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

        dq_cos = cos(q.grad.cpu(), q2.grad)
        dk_cos = cos(k.grad.cpu(), k2.grad)
        dv_cos = cos(v.grad.cpu(), v2.grad)
        assert dq_cos > 0.90, f"D=128 dQ cosine {dq_cos:.4f} too low"
        assert dk_cos > 0.90, f"D=128 dK cosine {dk_cos:.4f} too low"
        assert dv_cos > 0.99, f"D=128 dV cosine {dv_cos:.4f} too low"

    def test_backward_bf16(self):
        """BF16 backward should produce finite gradients."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 1, 2, 128, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()
        for name, p in [("q", q), ("k", k), ("v", v)]:
            assert not torch.isnan(p.grad).any(), f"{name}.grad has NaN"
            assert p.grad.abs().max() > 0, f"{name}.grad is all zeros"

    def test_backward_fp32(self):
        """FP32 backward (scalar path) should match reference closely."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 1, 2, 128, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()

        q2 = q.detach().cpu().requires_grad_(True)
        k2 = k.detach().cpu().requires_grad_(True)
        v2 = v.detach().cpu().requires_grad_(True)
        nystrom_attention_reference_simple(q2, k2, v2, m).sum().backward()

        def cos(a, b):
            return torch.nn.functional.cosine_similarity(
                a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

        assert cos(q.grad.cpu(), q2.grad) > 0.90, "FP32 dQ too far from reference"
        assert cos(k.grad.cpu(), k2.grad) > 0.90, "FP32 dK too far from reference"
        assert cos(v.grad.cpu(), v2.grad) > 0.99, "FP32 dV too far from reference"

    def test_backward_non_divisible_n(self):
        """N not divisible by m or tile size."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 1, 2, 300, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()
        for name, p in [("q", q), ("k", k), ("v", v)]:
            assert not torch.isnan(p.grad).any(), f"{name}.grad has NaN at N=300"

    def test_backward_batch(self):
        """Multi-batch backward."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 4, 4, 128, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()
        for name, p in [("q", q), ("k", k), ("v", v)]:
            assert not torch.isnan(p.grad).any(), f"{name}.grad has NaN for batch=4"
            assert p.grad.abs().max() > 0, f"{name}.grad is all zeros for batch=4"

    def test_all_gradients_match_reference(self):
        """All three gradients (dQ, dK, dV) should have >0.9 cosine with reference."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(0)
        B, H, N, D, m = 1, 2, 256, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()

        q2 = q.detach().float().cpu().requires_grad_(True)
        k2 = k.detach().float().cpu().requires_grad_(True)
        v2 = v.detach().float().cpu().requires_grad_(True)
        nystrom_attention_reference_simple(q2, k2, v2, m).sum().backward()

        def cos(a, b):
            return torch.nn.functional.cosine_similarity(
                a.float().flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

        dq_cos = cos(q.grad.cpu(), q2.grad)
        dk_cos = cos(k.grad.cpu(), k2.grad)
        dv_cos = cos(v.grad.cpu(), v2.grad)
        assert dq_cos > 0.90, f"dQ cosine {dq_cos:.4f} too low"
        assert dk_cos > 0.90, f"dK cosine {dk_cos:.4f} too low"
        assert dv_cos > 0.99, f"dV cosine {dv_cos:.4f} too low"

    # -- BF16 cosine similarity --

    def test_backward_bf16_cosine_d64(self):
        """BF16 gradients at D=64 should match reference with cosine > 0.9."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(0)
        B, H, N, D, m = 1, 2, 256, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()

        q2 = q.detach().float().cpu().requires_grad_(True)
        k2 = k.detach().float().cpu().requires_grad_(True)
        v2 = v.detach().float().cpu().requires_grad_(True)
        nystrom_attention_reference_simple(q2, k2, v2, m).sum().backward()

        def cos(a, b):
            return torch.nn.functional.cosine_similarity(
                a.float().flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

        assert cos(q.grad.cpu(), q2.grad) > 0.85, "BF16 dQ cosine too low"
        assert cos(k.grad.cpu(), k2.grad) > 0.85, "BF16 dK cosine too low"
        assert cos(v.grad.cpu(), v2.grad) > 0.99, "BF16 dV cosine too low"

    def test_backward_bf16_d128(self):
        """BF16 at D=128 should produce correct gradients."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(0)
        B, H, N, D, m = 1, 2, 256, 128, 64
        q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()

        q2 = q.detach().float().cpu().requires_grad_(True)
        k2 = k.detach().float().cpu().requires_grad_(True)
        v2 = v.detach().float().cpu().requires_grad_(True)
        nystrom_attention_reference_simple(q2, k2, v2, m).sum().backward()

        def cos(a, b):
            return torch.nn.functional.cosine_similarity(
                a.float().flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

        assert cos(q.grad.cpu(), q2.grad) > 0.85, "BF16 D=128 dQ cosine too low"
        assert cos(k.grad.cpu(), k2.grad) > 0.85, "BF16 D=128 dK cosine too low"
        assert cos(v.grad.cpu(), v2.grad) > 0.99, "BF16 D=128 dV cosine too low"

    # -- FP32 scalar path at D=128 --

    def test_backward_fp32_d128_raises(self):
        """FP32+D=128 backward should raise a clear error, not crash."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 1, 2, 128, 128, 64
        q = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float32, device="cuda", requires_grad=True)

        with pytest.raises(RuntimeError, match="FP32.*D=128 is not supported"):
            FlashNystromFunction.apply(q, k, v, m, 6, False)

    # -- partial tile edge cases --

    def test_backward_n65_partial_tile(self):
        """N=65: 1 full tile + 1-row partial. Tests partial tile in backward."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 1, 2, 65, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()

        q2 = q.detach().float().cpu().requires_grad_(True)
        k2 = k.detach().float().cpu().requires_grad_(True)
        v2 = v.detach().float().cpu().requires_grad_(True)
        nystrom_attention_reference_simple(q2, k2, v2, m).sum().backward()

        def cos(a, b):
            return torch.nn.functional.cosine_similarity(
                a.float().flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

        assert cos(v.grad.cpu(), v2.grad) > 0.99, f"N=65 dV cosine too low"

    def test_backward_n63_under_tile(self):
        """N=63: single partial tile, one row short."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 1, 2, 63, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()
        for name, p in [("q", q), ("k", k), ("v", v)]:
            assert not torch.isnan(p.grad).any(), f"{name}.grad has NaN at N=63"
            assert p.grad.abs().max() > 0, f"{name}.grad is all zeros at N=63"

    def test_backward_d128_partial_tile(self):
        """D=128, N=100: partial tile with D=128 SMEM path."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 1, 2, 100, 128, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()

        q2 = q.detach().float().cpu().requires_grad_(True)
        k2 = k.detach().float().cpu().requires_grad_(True)
        v2 = v.detach().float().cpu().requires_grad_(True)
        nystrom_attention_reference_simple(q2, k2, v2, m).sum().backward()

        def cos(a, b):
            return torch.nn.functional.cosine_similarity(
                a.float().flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

        assert cos(v.grad.cpu(), v2.grad) > 0.99, "D=128 N=100 dV cosine too low"

    # -- large N stress (atomicAdd accumulation across many tiles) --

    def test_backward_large_n_cosine(self):
        """N=4096: many tiles accumulate dKt/dstep2 via atomicAdd."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(0)
        B, H, N, D, m = 1, 2, 4096, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()

        q2 = q.detach().float().cpu().requires_grad_(True)
        k2 = k.detach().float().cpu().requires_grad_(True)
        v2 = v.detach().float().cpu().requires_grad_(True)
        nystrom_attention_reference_simple(q2, k2, v2, m).sum().backward()

        def cos(a, b):
            return torch.nn.functional.cosine_similarity(
                a.float().flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

        dq_cos = cos(q.grad.cpu(), q2.grad)
        dk_cos = cos(k.grad.cpu(), k2.grad)
        dv_cos = cos(v.grad.cpu(), v2.grad)
        assert dq_cos > 0.85, f"N=4096 dQ cosine {dq_cos:.4f} too low"
        assert dk_cos > 0.85, f"N=4096 dK cosine {dk_cos:.4f} too low"
        assert dv_cos > 0.99, f"N=4096 dV cosine {dv_cos:.4f} too low"

    # -- N%m != 0 with cosine check --

    def test_backward_non_divisible_n_cosine(self):
        """N=300, m=32: not divisible, check cosine not just NaN."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 1, 2, 300, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()

        q2 = q.detach().float().cpu().requires_grad_(True)
        k2 = k.detach().float().cpu().requires_grad_(True)
        v2 = v.detach().float().cpu().requires_grad_(True)
        nystrom_attention_reference_simple(q2, k2, v2, m).sum().backward()

        def cos(a, b):
            return torch.nn.functional.cosine_similarity(
                a.float().flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

        assert cos(v.grad.cpu(), v2.grad) > 0.99, "N=300 dV cosine too low"

    # -- m < kBlockN --

    def test_backward_m16_d128(self):
        """m=16, D=128: small landmark count with zero-padded SMEM."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 1, 2, 256, 128, 16
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()
        for name, p in [("q", q), ("k", k), ("v", v)]:
            assert not torch.isnan(p.grad).any(), f"{name}.grad NaN at m=16 D=128"
            assert p.grad.abs().max() > 0, f"{name}.grad zeros at m=16 D=128"

    # -- determinism --

    def test_backward_determinism(self):
        """Two identical backward passes should produce identical gradients."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 1, 2, 256, 64, 32

        def run():
            q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
            k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
            v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
            # use same data both runs
            return q, k, v

        q1, k1, v1 = run()
        q2 = q1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        v2 = v1.detach().clone().requires_grad_(True)

        FlashNystromFunction.apply(q1, k1, v1, m, 6, False).sum().backward()
        FlashNystromFunction.apply(q2, k2, v2, m, 6, False).sum().backward()

        # dV should be bit-exact (no atomicAdd in V path)
        assert torch.equal(v1.grad, v2.grad), \
            f"dV not deterministic, max diff: {(v1.grad-v2.grad).abs().max().item()}"

    # -- multi-batch with cosine --

    def test_backward_batch_cosine(self):
        """Multi-batch backward with cosine similarity check."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 4, 4, 128, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)

        FlashNystromFunction.apply(q, k, v, m, 6, False).sum().backward()

        q2 = q.detach().float().cpu().requires_grad_(True)
        k2 = k.detach().float().cpu().requires_grad_(True)
        v2 = v.detach().float().cpu().requires_grad_(True)
        nystrom_attention_reference_simple(q2, k2, v2, m).sum().backward()

        def cos(a, b):
            return torch.nn.functional.cosine_similarity(
                a.float().flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

        assert cos(q.grad.cpu(), q2.grad) > 0.90, "Batch dQ cosine too low"
        assert cos(k.grad.cpu(), k2.grad) > 0.90, "Batch dK cosine too low"
        assert cos(v.grad.cpu(), v2.grad) > 0.99, "Batch dV cosine too low"

    # -- end-to-end --

    def test_training_converges(self):
        """End-to-end: model can overfit to a fixed target."""
        try:
            from flash_nystrom import FlashNystromAttention, NystromConfig
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        config = NystromConfig(num_landmarks=32, conv_kernel_size=0, use_conv_residual=False)
        model = FlashNystromAttention(128, heads=2, config=config).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(1, 32, 128, device="cuda")
        target = torch.randn(1, 32, 128, device="cuda") * 0.01

        initial_loss = None
        for step in range(30):
            out = model(x)
            loss = (out - target).pow(2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step == 0:
                initial_loss = loss.item()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.1, \
            f"Loss did not decrease enough: {initial_loss:.6f} -> {final_loss:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
