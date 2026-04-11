# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

"""Tests for FlashNystrom forward pass correctness."""

import pytest
import torch
import math

from flash_nystrom.reference import (
    nystrom_attention_reference_simple,
    nystrom_attention_reference,
    iterative_pinverse,
)


class TestReferenceImplementation:
    """Validate the pure-PyTorch reference before trusting it as ground truth."""

    def test_output_shape(self):
        B, H, N, D = 2, 4, 512, 64
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        out = nystrom_attention_reference_simple(q, k, v, num_landmarks=32)
        assert out.shape == (B, H, N, D)

    def test_deterministic(self):
        B, H, N, D = 1, 2, 256, 64
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        out1 = nystrom_attention_reference_simple(q, k, v, num_landmarks=32)
        out2 = nystrom_attention_reference_simple(q, k, v, num_landmarks=32)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_gradient_flows(self):
        B, H, N, D = 1, 2, 128, 64
        q = torch.randn(B, H, N, D, requires_grad=True)
        k = torch.randn(B, H, N, D, requires_grad=True)
        v = torch.randn(B, H, N, D, requires_grad=True)
        out = nystrom_attention_reference_simple(q, k, v, num_landmarks=32)
        loss = out.sum()
        loss.backward()
        assert q.grad is not None and not torch.isnan(q.grad).any()
        assert k.grad is not None and not torch.isnan(k.grad).any()
        assert v.grad is not None and not torch.isnan(v.grad).any()

    def test_output_finite_and_reasonable(self):
        torch.manual_seed(0)
        B, H, N, D = 1, 1, 256, 64
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        out = nystrom_attention_reference_simple(q, k, v, num_landmarks=64)
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"
        # Output is a weighted average of V, should not exceed V's range wildly
        assert out.norm() < v.norm() * 5.0

    def test_conv_residual_changes_output(self):
        B, H, N, D = 1, 2, 256, 64
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        conv_w = torch.randn(H, 3) * 0.1

        out_no_conv = nystrom_attention_reference(q, k, v, 32, 6, None, 0)
        out_with_conv = nystrom_attention_reference(q, k, v, 32, 6, conv_w, 3)
        assert not torch.allclose(out_no_conv, out_with_conv, atol=1e-6)

    def test_newton_schulz_convergence(self):
        torch.manual_seed(42)
        m = 32
        A = torch.randn(m, m)
        A = A @ A.T / m
        A = torch.softmax(A, dim=-1)

        pinv = iterative_pinverse(A.unsqueeze(0).unsqueeze(0).float(), n_iter=6).squeeze()
        reconstructed = A @ pinv @ A
        max_err = (reconstructed - A).abs().max().item()
        assert max_err < 0.05, f"Newton-Schulz error: {max_err:.6f}"

    def test_n_not_divisible_by_m(self):
        """N=300, m=64: 300 is not divisible by 64."""
        B, H, N, D = 1, 2, 300, 64
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        out = nystrom_attention_reference_simple(q, k, v, num_landmarks=64)
        assert out.shape == (B, H, N, D)
        assert not torch.isnan(out).any()

    def test_m_equals_n(self):
        """Edge case: m == N."""
        B, H, N, D = 1, 1, 64, 32
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        out = nystrom_attention_reference_simple(q, k, v, num_landmarks=64)
        assert out.shape == (B, H, N, D)
        assert not torch.isnan(out).any()

    def test_batch_1_head_1(self):
        B, H, N, D = 1, 1, 128, 64
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        out = nystrom_attention_reference_simple(q, k, v, num_landmarks=16)
        assert out.shape == (B, H, N, D)
        assert not torch.isnan(out).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAForward:
    """Test CUDA kernels against reference. Requires compiled extension."""

    def _compare(self, B, H, N, D, m, dtype, atol):
        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
        k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
        v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")

        ref = nystrom_attention_reference_simple(
            q.float().cpu(), k.float().cpu(), v.float().cpu(), m
        ).to(dtype).cuda()

        try:
            from flash_nystrom._C import forward as cuda_forward
            results = cuda_forward(q, k, v, m, 6, 0, None)
            cuda_out = results[0]
            max_err = (cuda_out - ref).abs().max().item()
            assert not torch.isnan(cuda_out).any(), "CUDA output contains NaN"
            assert max_err < atol, f"Max error {max_err:.6f} > tolerance {atol}"
        except ImportError:
            pytest.skip("CUDA extension not compiled")

    def test_fp32_small(self):
        self._compare(1, 2, 256, 64, 32, torch.float32, 1e-3)

    def test_fp16_small(self):
        self._compare(1, 2, 256, 64, 32, torch.float16, 1e-3)

    def test_bf16_small(self):
        self._compare(1, 2, 256, 64, 32, torch.bfloat16, 5e-3)

    def test_fp16_large(self):
        self._compare(2, 4, 1024, 128, 64, torch.float16, 1e-3)

    def test_fp16_landmarks_32(self):
        self._compare(1, 2, 512, 128, 32, torch.float16, 1e-3)

    def test_fp16_non_divisible_n(self):
        self._compare(1, 2, 300, 64, 32, torch.float16, 1e-3)

    def test_fp16_stress_4k(self):
        self._compare(1, 8, 4096, 128, 64, torch.float16, 1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
