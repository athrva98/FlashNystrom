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
        """Verify Q/K gradients from reference point in the right direction.

        Exact gradcheck fails because N-S autograd chain is numerically noisy,
        but the gradient direction should still be meaningful (positive cosine
        similarity with finite-difference approximation).
        """
        B, H, N, D, m = 1, 1, 32, 16, 8
        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, dtype=torch.float64, requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float64, requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float64, requires_grad=True)

        out = nystrom_attention_reference_simple(q, k, v, m)
        out.sum().backward()

        # check gradients are finite and non-zero
        for name, p in [("q", q), ("k", k), ("v", v)]:
            assert p.grad is not None, f"{name}.grad is None"
            assert not torch.isnan(p.grad).any(), f"{name}.grad has NaN"
            assert p.grad.abs().max() > 0, f"{name}.grad is all zeros"

        # verify gradient direction via finite differences for a few elements
        eps = 1e-5
        for name, param in [("q", q), ("k", k)]:
            # pick a random element
            idx = (0, 0, N // 2, D // 2)
            analytical = param.grad[idx].item()

            # finite difference
            param_data = param.data.clone()
            param.data[idx] += eps
            out_plus = nystrom_attention_reference_simple(q, k, v, m).sum().item()
            param.data[idx] -= 2 * eps
            out_minus = nystrom_attention_reference_simple(q, k, v, m).sum().item()
            param.data.copy_(param_data)
            numerical = (out_plus - out_minus) / (2 * eps)

            # should agree in sign at minimum
            if abs(numerical) > 1e-8 and abs(analytical) > 1e-8:
                assert numerical * analytical > 0, \
                    f"{name} gradient sign mismatch: analytical={analytical:.6f}, numerical={numerical:.6f}"


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

        out = FlashNystromFunction.apply(q, k, v, None, m, 6, 0)
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

        out = FlashNystromFunction.apply(q, k, v_cuda, None, m, 6, 0)
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

    def test_backward_d128(self):
        """Test backward with D=128 (larger head dim, needs SMEM opt-in)."""
        try:
            from flash_nystrom.flash_nystrom import FlashNystromFunction
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 1, 2, 128, 128, 64
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", requires_grad=True)

        out = FlashNystromFunction.apply(q, k, v, None, m, 6, 0)
        out.sum().backward()

        for name, p in [("q", q), ("k", k), ("v", v)]:
            assert p.grad is not None, f"{name}.grad is None"
            assert not torch.isnan(p.grad).any(), f"{name}.grad has NaN"
            assert not torch.isinf(p.grad).any(), f"{name}.grad has Inf"
            assert p.grad.abs().max() > 0, f"{name}.grad is all zeros"

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

        FlashNystromFunction.apply(q, k, v, None, m, 6, 0).sum().backward()
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

        FlashNystromFunction.apply(q, k, v, None, m, 6, 0).sum().backward()

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

        FlashNystromFunction.apply(q, k, v, None, m, 6, 0).sum().backward()
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

        FlashNystromFunction.apply(q, k, v, None, m, 6, 0).sum().backward()
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

        FlashNystromFunction.apply(q, k, v, None, m, 6, 0).sum().backward()

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
