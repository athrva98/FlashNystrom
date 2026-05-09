# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Isolation tests for the unrolled Newton-Schulz backward kernels.

These tests bypass the autograd Function and call the CUDA NS backward kernels
directly (via the `_C.debug_ns_bwd_step` / `_C.debug_ns_bwd_final` hooks),
comparing element-wise to a PyTorch reference implementation of the SAME math.

If these tests pass to FP32 noise, the NS backward kernels are correct.
If they fail, the test pinpoints which intermediate diverges.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

try:
    import flash_nystrom._C as _C
    HAS_DEBUG = hasattr(_C, "debug_ns_bwd_step") and hasattr(_C, "debug_ns_bwd_final")
    HAS_DK2INV_DEBUG = hasattr(_C, "debug_compute_dk2inv")
except ImportError:
    HAS_DEBUG = False
    HAS_DK2INV_DEBUG = False

pytestmark_debug = pytest.mark.skipif(not HAS_DEBUG, reason="_C debug hooks not built")


# ------- PyTorch reference for one NS backward iteration ----------------
# Forward iteration j (per-bh):
#   M = K2 @ Z_j
#   U = 7 I - M
#   V = 15 I - M @ U
#   T = 13 I - M @ V
#   Z_{j+1} = (1/4) Z_j @ T
#
# Backward (given dZ_{j+1} -> dZ_j and dK2_contrib_j):
#   dT       = (1/4) Z_j^T @ dZ_in
#   dZ_outer = (1/4) dZ_in @ T^T
#   dM_T     = -dT @ V^T
#   dV       = -M^T @ dT
#   dM_V     = -dV @ U^T
#   dU       = -M^T @ dV
#   dM_U     = -dU
#   dM       = dM_T + dM_V + dM_U
#   dK2_contrib = dM @ Z_j^T
#   dZ_inner    = K2^T @ dM
#   dZ_j       = dZ_outer + dZ_inner

def ref_ns_bwd_step(K2, Z_j, dZ_in):
    """Per-batch-head Python reference for one NS backward iteration.

    All inputs are (BH, m, m) FP32 tensors. Returns (dZ_j, dK2_contrib),
    each (BH, m, m) FP32.
    """
    BH, m, _ = K2.shape
    I = torch.eye(m, device=K2.device, dtype=K2.dtype).expand(BH, m, m)

    M = K2 @ Z_j
    U = 7.0 * I - M
    V = 15.0 * I - M @ U
    T = 13.0 * I - M @ V

    dT = 0.25 * Z_j.transpose(-2, -1) @ dZ_in
    dZ_outer = 0.25 * dZ_in @ T.transpose(-2, -1)

    dM_T = -dT @ V.transpose(-2, -1)
    dV = -M.transpose(-2, -1) @ dT
    dM_V = -dV @ U.transpose(-2, -1)
    dU = -M.transpose(-2, -1) @ dV
    dM_U = -dU
    dM = dM_T + dM_V + dM_U

    dK2_contrib = dM @ Z_j.transpose(-2, -1)
    dZ_inner = K2.transpose(-2, -1) @ dM
    dZ_j = dZ_outer + dZ_inner
    return dZ_j, dK2_contrib


def ref_ns_bwd_final(q_tilde, k_tilde, K2, dZ0, dK2_in):
    """Python reference for the NS backward final step.

    The forward Z_0 init is:  Z_0 = K2^T / c,  c = ||K2||_1 * ||K2||_inf,
    with both max() ops differentiable. Backward dK2 contributions:
        (a) Direct:        dK2[r, c]   += dZ_0[c, r] / c
        (b) via dnorm_1:   dK2[r, jc1] += -S * norm_inf / c^2  (all r)
        (c) via dnorm_inf: dK2[ir_inf, c] += -S * norm_1 / c^2 (all c)
    where S = trace(dZ_0 @ K2) and (jc1, ir_inf) are argmax positions for
    norm_1 and norm_inf respectively (first-occurrence on ties).
    """
    BH, m, _ = K2.shape
    abs_K2 = K2.abs()
    col_sums = abs_K2.sum(dim=-2)              # (BH, m)
    row_sums = abs_K2.sum(dim=-1)              # (BH, m)
    norm_1, jc1     = col_sums.max(dim=-1)     # (BH,), (BH,)
    norm_inf, ir_inf = row_sums.max(dim=-1)
    c_val  = torch.clamp(norm_1 * norm_inf, min=1e-12)
    inv_c  = 1.0 / c_val
    inv_c2 = inv_c * inv_c

    dK2_after_init = dK2_in + dZ0.transpose(-2, -1) * inv_c.view(BH, 1, 1)

    # S = trace(dZ_0 @ K2) per batch
    S = (dZ0 * K2.transpose(-2, -1)).sum(dim=(-2, -1))   # (BH,)

    # Column term:  dK2[*, jc1] += -S * norm_inf / c^2
    col_term = -S * norm_inf * inv_c2                  # (BH,)
    bh_range = torch.arange(BH, device=K2.device)
    dK2_after_init[bh_range, :, jc1] += col_term.view(BH, 1)

    # Row term:  dK2[ir_inf, *] += -S * norm_1 / c^2
    row_term = -S * norm_1 * inv_c2
    dK2_after_init[bh_range, ir_inf, :] += row_term.view(BH, 1)

    D_i = (dK2_after_init * K2).sum(dim=-1, keepdim=True)
    dS2 = K2 * (dK2_after_init - D_i)

    dQ_tilde = dS2 @ k_tilde
    dK_tilde = dS2.transpose(-2, -1) @ q_tilde
    return dQ_tilde, dK_tilde, dK2_after_init


def relerr(a, b):
    n = (a - b).norm()
    d = b.norm()
    return (n / torch.clamp(d, min=1e-30)).item()


def maxerr(a, b):
    return (a - b).abs().max().item()


# ------- Tests ----------------------------------------------------------

@pytest.mark.skipif(not HAS_DEBUG, reason="_C debug hooks not built")
class TestNSBwdStep:
    """Per-iteration NS backward kernel."""

    def _run(self, BH, m, seed=0, atol=1e-5):
        torch.manual_seed(seed)
        # Build a realistic K2 (softmax of random matrix, row-stochastic).
        A = torch.randn(BH, m, m, dtype=torch.float32, device="cuda") * 0.5
        K2 = torch.softmax(A, dim=-1).contiguous()
        Z_j = torch.randn(BH, m, m, dtype=torch.float32, device="cuda")
        dZ_in = torch.randn(BH, m, m, dtype=torch.float32, device="cuda")

        # PyTorch reference
        dZ_ref, dK2_ref = ref_ns_bwd_step(K2, Z_j, dZ_in)

        # CUDA kernel
        dZ_cuda, dK2_cuda = _C.debug_ns_bwd_step(K2, Z_j, dZ_in)

        # Element-wise relative error vs FP32 reference
        rdz = relerr(dZ_cuda, dZ_ref)
        rdk = relerr(dK2_cuda, dK2_ref)
        mdz = maxerr(dZ_cuda, dZ_ref)
        mdk = maxerr(dK2_cuda, dK2_ref)

        assert rdz < atol, (f"dZ_j relerr={rdz:.2e}, max abs err={mdz:.2e}, "
                            f"BH={BH}, m={m}")
        assert rdk < atol, (f"dK2_contrib relerr={rdk:.2e}, max abs err={mdk:.2e}, "
                            f"BH={BH}, m={m}")

    def test_BH1_m4(self):
        self._run(BH=1, m=4)

    def test_BH1_m8(self):
        self._run(BH=1, m=8)

    def test_BH1_m32(self):
        self._run(BH=1, m=32)

    def test_BH1_m64(self):
        self._run(BH=1, m=64)

    def test_BH4_m32(self):
        self._run(BH=4, m=32)

    def test_BH8_m64(self):
        self._run(BH=8, m=64)


@pytest.mark.skipif(not HAS_DEBUG, reason="_C debug hooks not built")
class TestNSBwdFinal:
    """NS backward final step (Z_0 init gradient + softmax backward)."""

    def _run(self, BH, m, D, seed=0, atol=1e-5):
        torch.manual_seed(seed)
        q_tilde = torch.randn(BH, m, D, dtype=torch.float32, device="cuda")
        k_tilde = torch.randn(BH, m, D, dtype=torch.float32, device="cuda")
        # Use an asymmetric positive K2 (not row-stochastic) so the argmax
        # row/column for the L1/Linf max() are unambiguous. A row-stochastic
        # K2 has all row sums tied at 1.0, which makes argmax_row sensitive
        # to FP summation order — kernel and torch.max can disagree on the
        # tie even though the resulting dQ_tilde/dK_tilde are unaffected
        # (the row term zero-cancels through softmax bwd downstream).
        K2 = torch.rand(BH, m, m, dtype=torch.float32, device="cuda") + 0.1
        K2 = K2.contiguous()
        dZ0 = torch.randn(BH, m, m, dtype=torch.float32, device="cuda")
        dK2_in = torch.randn(BH, m, m, dtype=torch.float32, device="cuda")

        dQ_ref, dK_ref, dK2_ref = ref_ns_bwd_final(q_tilde, k_tilde, K2, dZ0, dK2_in)
        dQ_cuda, dK_cuda, dK2_cuda = _C.debug_ns_bwd_final(
            q_tilde, k_tilde, K2, dZ0, dK2_in)

        rq = relerr(dQ_cuda, dQ_ref)
        rk = relerr(dK_cuda, dK_ref)
        rdk2 = relerr(dK2_cuda, dK2_ref)

        assert rq < atol, f"dQ_tilde relerr={rq:.2e}, BH={BH}, m={m}, D={D}"
        assert rk < atol, f"dK_tilde relerr={rk:.2e}, BH={BH}, m={m}, D={D}"
        assert rdk2 < atol, f"dK2 (after init) relerr={rdk2:.2e}"

    def test_BH1_m4_D8(self):
        self._run(BH=1, m=4, D=8)

    def test_BH1_m32_D64(self):
        self._run(BH=1, m=32, D=64)

    def test_BH1_m64_D128(self):
        self._run(BH=1, m=64, D=128)

    def test_BH4_m32_D64(self):
        self._run(BH=4, m=32, D=64)


@pytest.mark.skipif(not HAS_DEBUG, reason="_C debug hooks not built")
class TestNSBwdEndToEnd:
    """End-to-end: run several backward steps and compare to PyTorch unrolled chain."""

    def test_iter1(self):
        # Drive one step kernel; result must match the Python ref.
        torch.manual_seed(0)
        BH, m = 4, 32
        A = torch.randn(BH, m, m, dtype=torch.float32, device="cuda") * 0.5
        K2 = torch.softmax(A, dim=-1).contiguous()
        Z = torch.randn(BH, m, m, dtype=torch.float32, device="cuda")
        dZ = torch.randn(BH, m, m, dtype=torch.float32, device="cuda")

        dZ_out_ref, dK2_ref = ref_ns_bwd_step(K2, Z, dZ)
        dZ_out_cuda, dK2_cuda = _C.debug_ns_bwd_step(K2, Z, dZ)

        assert relerr(dZ_out_cuda, dZ_out_ref) < 1e-5
        assert relerr(dK2_cuda, dK2_ref) < 1e-5

    def test_iter3_chain(self):
        # Chain three backward steps as in the actual kernel orchestration.
        torch.manual_seed(0)
        BH, m = 2, 16
        A = torch.randn(BH, m, m, dtype=torch.float32, device="cuda") * 0.5
        K2 = torch.softmax(A, dim=-1).contiguous()

        # Three different "Z_j" (could be from a real forward; here random for the test)
        Zs = [torch.randn(BH, m, m, dtype=torch.float32, device="cuda") for _ in range(3)]
        dZ_seed = torch.randn(BH, m, m, dtype=torch.float32, device="cuda")

        # PyTorch chain: walk backward through iterations 2, 1, 0.
        dZ_pt = dZ_seed
        dK2_pt = torch.zeros_like(K2)
        for j in (2, 1, 0):
            dZ_new, dK2_inc = ref_ns_bwd_step(K2, Zs[j], dZ_pt)
            dZ_pt = dZ_new
            dK2_pt = dK2_pt + dK2_inc

        # CUDA chain
        dZ_cuda = dZ_seed.clone()
        dK2_cuda = torch.zeros_like(K2)
        for j in (2, 1, 0):
            dZ_new, dK2_inc = _C.debug_ns_bwd_step(K2, Zs[j], dZ_cuda)
            dZ_cuda = dZ_new
            dK2_cuda = dK2_cuda + dK2_inc

        assert relerr(dZ_cuda, dZ_pt) < 1e-5
        assert relerr(dK2_cuda, dK2_pt) < 1e-5


@pytest.mark.skipif(not HAS_DK2INV_DEBUG, reason="_C.debug_compute_dk2inv not built")
class TestComputeDK2Inv:
    """Regression: compute_dk2inv must compute the EXACT dL/dZ_N = dstep2 @ B^T,
    where B = softmax(Q_tilde @ K_s^T) @ V, with NO assumption on NS convergence.

    The previous implementation used dstep2 @ (K2 @ step2)^T which is only correct
    when K2 @ Z_N = I (full convergence). At newton_iter=6 that approximation
    introduces ~38% relerr in dK2_inv, which propagates through the entire NS
    backward chain and breaks training accuracy. This test locks in the correct
    formula.
    """

    def _ref(self, q_tilde, k_s, v, lse3, dstep2):
        # B = softmax(Q_tilde @ K_s^T) @ V, computed exactly using lse3.
        scores = q_tilde @ k_s.transpose(-2, -1)        # (BH, m, N)
        A = torch.exp(scores - lse3.unsqueeze(-1))      # (BH, m, N)
        B = A @ v                                        # (BH, m, D)
        return dstep2 @ B.transpose(-2, -1)             # (BH, m, m)

    def _run(self, BH, m, N, D, seed=0, atol=1e-5):
        torch.manual_seed(seed)
        q_tilde = torch.randn(BH, m, D, device="cuda", dtype=torch.float32)
        k_s     = torch.randn(BH, N, D, device="cuda", dtype=torch.float32)
        v       = torch.randn(BH, N, D, device="cuda", dtype=torch.float32)
        # lse3 must match what kernel3 forward produces: row-LSE of Q_tilde @ K_s^T.
        scores = q_tilde @ k_s.transpose(-2, -1)
        lse3 = torch.logsumexp(scores, dim=-1)           # (BH, m)
        dstep2 = torch.randn(BH, m, D, device="cuda", dtype=torch.float32)

        dK2_inv_ref = self._ref(q_tilde, k_s, v, lse3, dstep2)
        dK2_inv_cuda = _C.debug_compute_dk2inv(q_tilde, k_s, v, lse3, dstep2)

        rerr = relerr(dK2_inv_cuda, dK2_inv_ref)
        merr = maxerr(dK2_inv_cuda, dK2_inv_ref)
        assert rerr < atol, (f"dK2_inv relerr={rerr:.2e}, max abs err={merr:.2e}, "
                             f"BH={BH}, m={m}, N={N}, D={D}")

    def test_BH1_m32_N64_D64(self):
        self._run(BH=1, m=32, N=64, D=64)

    def test_BH1_m32_N128_D64(self):
        self._run(BH=1, m=32, N=128, D=64)

    def test_BH4_m32_N256_D64(self):
        self._run(BH=4, m=32, N=256, D=64)

    def test_BH1_m64_N1024_D64(self):
        self._run(BH=1, m=64, N=1024, D=64)

    def test_BH1_m32_N64_D128(self):
        self._run(BH=1, m=32, N=64, D=128)

    def test_BH4_m32_N256_D128(self):
        self._run(BH=4, m=32, N=256, D=128)

    def test_BH8_m64_N4096_D128(self):
        self._run(BH=8, m=64, N=4096, D=128)

    def test_partial_tile_N(self):
        # N not divisible by TILE_N=32 — verify masking is correct.
        self._run(BH=2, m=16, N=65, D=64)
        self._run(BH=2, m=16, N=33, D=64)
        self._run(BH=2, m=16, N=31, D=64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
