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
            results = cuda_forward(q, k, v, m, 6)
            cuda_out = results[0]
            assert not torch.isnan(cuda_out).any(), "CUDA output contains NaN"
            # K2_inv (exact LU) can have elements up to ~1000, so element-level
            # FP16 noise scales accordingly. Use cosine + relative norm error
            # for a tolerance that's meaningful at all output magnitudes.
            cos = torch.nn.functional.cosine_similarity(
                cuda_out.float().flatten().unsqueeze(0),
                ref.float().flatten().unsqueeze(0)).item()
            rel_err = ((cuda_out.float() - ref.float()).norm() /
                        (ref.float().norm() + 1e-12)).item()
            max_err = (cuda_out - ref).abs().max().item()
            assert cos > 0.999, f"cosine {cos:.6f} < 0.999 (max_err={max_err:.4f})"
            assert rel_err < atol, f"relative error {rel_err:.6f} > tolerance {atol}"
        except ImportError:
            pytest.skip("CUDA extension not compiled")

    def test_fp32_small(self):
        self._compare(1, 2, 256, 64, 32, torch.float32, 1e-3)

    def test_fp16_small(self):
        self._compare(1, 2, 256, 64, 32, torch.float16, 5e-3)

    def test_bf16_small(self):
        self._compare(1, 2, 256, 64, 32, torch.bfloat16, 5e-3)

    def test_fp16_large(self):
        self._compare(2, 4, 1024, 128, 64, torch.float16, 5e-3)

    def test_fp16_landmarks_32(self):
        self._compare(1, 2, 512, 128, 32, torch.float16, 5e-3)

    def test_fp16_non_divisible_n(self):
        self._compare(1, 2, 300, 64, 32, torch.float16, 5e-3)

    def test_fp16_stress_4k(self):
        self._compare(1, 8, 4096, 128, 64, torch.float16, 5e-3)

    # -- D=128 dtype coverage --

    def test_bf16_d128(self):
        self._compare(1, 2, 512, 128, 64, torch.bfloat16, 5e-3)

    def test_fp32_d128_runs_or_smem_gated(self):
        """FP32+D=128 is no longer hard-rejected (it's a gradient-checking path).
        On a GPU with enough opt-in SMEM (datacenter parts, ~150KB) it runs; on
        an undersized GPU it raises a clear 'insufficient smem' capability error
        — never the old blanket 'not supported', and never a silent crash."""
        try:
            from flash_nystrom._C import forward as cuda_forward
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        q = torch.randn(1, 2, 256, 128, dtype=torch.float32, device="cuda")
        k = torch.randn(1, 2, 256, 128, dtype=torch.float32, device="cuda")
        v = torch.randn(1, 2, 256, 128, dtype=torch.float32, device="cuda")

        try:
            out = cuda_forward(q, k, v, 64, 6)[0]
            assert torch.isfinite(out).all()  # capable GPU: it actually ran
        except RuntimeError as e:
            assert "insufficient smem" in str(e), \
                f"expected a SMEM capability error, got: {e}"

    # -- partial tile edge cases --

    def test_fp16_n_equals_tile_size(self):
        """N=64 = kBlockM: exactly one full tile, no partial."""
        self._compare(1, 2, 64, 64, 32, torch.float16, 5e-3)

    def test_fp16_n_one_over_tile(self):
        """N=65: one full tile + one partial tile with 1 row."""
        self._compare(1, 2, 65, 64, 32, torch.float16, 5e-3)

    def test_fp16_n_one_under_tile(self):
        """N=63: single partial tile, last row is missing."""
        self._compare(1, 2, 63, 64, 32, torch.float16, 5e-3)

    def test_fp16_n_one(self):
        """N=1: degenerate single-token sequence."""
        self._compare(1, 1, 1, 64, 1, torch.float16, 1e-2)

    def test_fp16_n_two_tiles_partial(self):
        """N=100: tile 0 full (64 rows), tile 1 partial (36 rows)."""
        self._compare(1, 2, 100, 64, 32, torch.float16, 5e-3)

    def test_fp16_d128_partial_tile(self):
        """D=128, N=100: partial tile at D=128 exercises SMEM opt-in path."""
        self._compare(1, 2, 100, 128, 64, torch.float16, 5e-3)

    # -- m < kBlockN --

    def test_fp16_m16_d128(self):
        """m=16 with D=128: landmark padding in SMEM (16 used, 48 zero-padded to 64)."""
        self._compare(1, 2, 256, 128, 16, torch.float16, 5e-3)

    # -- BF16 cosine similarity --

    def test_bf16_cosine_d64(self):
        """BF16 output should match FP32 reference with cosine > 0.99."""
        try:
            from flash_nystrom._C import forward as cuda_forward
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(0)
        B, H, N, D, m = 1, 2, 256, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")

        cuda_out = cuda_forward(q, k, v, m, 6)[0]
        ref = nystrom_attention_reference_simple(
            q.float().cpu(), k.float().cpu(), v.float().cpu(), m)

        cos = torch.nn.functional.cosine_similarity(
            cuda_out.float().cpu().flatten().unsqueeze(0),
            ref.flatten().unsqueeze(0)).item()
        assert cos > 0.99, f"BF16 forward cosine {cos:.4f} too low"

    # -- determinism --

    def test_determinism_cuda(self):
        """Two identical runs should produce bit-exact results."""
        try:
            from flash_nystrom._C import forward as cuda_forward
        except ImportError:
            pytest.skip("CUDA extension not compiled")

        torch.manual_seed(42)
        B, H, N, D, m = 1, 4, 512, 64, 32
        q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")

        out1 = cuda_forward(q, k, v, m, 6)[0]
        out2 = cuda_forward(q, k, v, m, 6)[0]
        assert torch.equal(out1, out2), f"Forward not deterministic, max diff: {(out1-out2).abs().max().item()}"

    # -- multi-batch at D=128 --

    def test_fp16_batch_d128(self):
        """Multi-batch, multi-head at D=128."""
        self._compare(4, 4, 256, 128, 64, torch.float16, 5e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
