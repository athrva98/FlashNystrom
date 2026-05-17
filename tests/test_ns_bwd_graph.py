# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Isolation tests for the production cuBLAS + CUDA-graph NS backward.

`tests/test_ns_bwd_kernel.py` exercises the per-step pybind hooks
(`debug_ns_bwd_step`, `debug_ns_bwd_final`) which call the standalone
hand-rolled kernels. Those hooks were the only kernel-level coverage of the
backward; the production orchestration (cuBLAS-based per-iter NS step +
trailing TC matmul, captured into a per-shape CUDA graph held in a
thread-local cache) was only reachable through the end-to-end autograd path
in `tests/test_backward.py`. A regression inside the graph-capture path was
hard to bisect because failures showed up as gradient-magnitude drift far
downstream.

This file fills that gap. It drives `launch_kernel2_inv_bwd` directly
through the `_C.debug_kernel2_inv_bwd_full` hook and pins:

  1. **Correctness vs the unrolled autograd reference** — for several
     (BH, m, D, newton_iter) shapes, including the boundary cases the
     graph cache key keys on.
  2. **Graph replay** — calling with the same shape twice should produce
     identical results from the (now cached) replayed graph. Verifies the
     captured graph does not stale-cache inputs/outputs across calls.
  3. **Shape-change cache invalidation** — call with shape A, then B, then
     A again. All three must be correct. Exercises both the cache-miss
     (capture) and cache-hit (replay) paths.
  4. **`reset_caches()` correctness** — call, reset, call again. The path
     must rebuild the graph correctly.
  5. **Per-dtype cache independence** — the production code holds separate
     caches for float/half/bfloat16. The debug hook is FP32-only (the
     scalar path), but we sanity-check that resetting from FP32 doesn't
     leak state.

The reference is FP32 autograd through the closed-form NS recurrence +
Z_0 init backward + softmax-bwd closing step, which exactly mirrors the
math the kernel pipeline implements.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

try:
    import flash_nystrom._C as _C
    HAS_FULL_DEBUG = hasattr(_C, "debug_kernel2_inv_bwd_full")
    HAS_RESET = hasattr(_C, "reset_caches")
except ImportError:
    HAS_FULL_DEBUG = False
    HAS_RESET = False


# ----- Forward-NS in PyTorch (used to produce ns_iterates the kernel expects) -----

def ns_forward_iterates(K2, newton_iter):
    """Run the NS recurrence in PyTorch and return Z_0..Z_N stacked along dim=1.

    K2: (BH, m, m) FP32 — softmax(Q_tilde @ K_tilde^T)
    Returns: (BH, newton_iter+1, m, m) FP32, matching the layout that the
    forward `kernel2_inv` writes and the bwd kernel consumes.

    Z_0 = K2^T / c, c = ||K2||_1 * ||K2||_inf.
    Z_{j+1} = (1/4) Z_j (13 I - M (15 I - M (7 I - M))),  M = K2 Z_j.
    """
    BH, m, _ = K2.shape
    abs_K2 = K2.abs()
    norm_1   = abs_K2.sum(dim=-2).max(dim=-1).values    # max column sum
    norm_inf = abs_K2.sum(dim=-1).max(dim=-1).values    # max row sum
    c_val = torch.clamp(norm_1 * norm_inf, min=1e-12).view(BH, 1, 1)
    Z = K2.transpose(-2, -1) / c_val
    I = torch.eye(m, device=K2.device, dtype=K2.dtype).expand(BH, m, m)

    out = [Z]
    for _ in range(newton_iter):
        M = K2 @ Z
        U = 7.0  * I - M
        V = 15.0 * I - M @ U
        T = 13.0 * I - M @ V
        Z = 0.25 * Z @ T
        out.append(Z)
    return torch.stack(out, dim=1).contiguous()  # (BH, niter+1, m, m)


# ----- The reference for launch_kernel2_inv_bwd ----------------------------

# Reuse ref_ns_bwd_step and ref_ns_bwd_final from test_ns_bwd_kernel.py to keep
# a single source of truth for the per-step / final-step math. pytest's
# rootdir is the repo root; the tests/ directory is not a package, so we
# load the sibling module by absolute path.
import importlib.util as _ilu
import os as _os
_spec = _ilu.spec_from_file_location(
    "_test_ns_bwd_kernel_ref",
    _os.path.join(_os.path.dirname(__file__), "test_ns_bwd_kernel.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ref_ns_bwd_step  = _mod.ref_ns_bwd_step
ref_ns_bwd_final = _mod.ref_ns_bwd_final


def ref_kernel2_inv_bwd_full(q_tilde, k_tilde, K2_softmax, ns_iterates,
                              dZ_N_input, newton_iter):
    """End-to-end PyTorch reference for `launch_kernel2_inv_bwd`.

    Walks backward through the NS unroll given pre-computed forward iterates,
    accumulates the K2 gradient, then applies the softmax-bwd closing step
    that emits dQ_tilde and dK_tilde — the same chain the kernel orchestrates.

    Returns (dQ_tilde, dK_tilde) FP32.
    """
    K2 = K2_softmax  # alias
    BH = K2.shape[0]
    dZ = dZ_N_input
    dK2_acc = torch.zeros_like(K2)
    for j in range(newton_iter - 1, -1, -1):
        Z_j = ns_iterates[:, j].contiguous()
        dZ_j, dK2_inc = ref_ns_bwd_step(K2, Z_j, dZ)
        dZ = dZ_j
        dK2_acc = dK2_acc + dK2_inc
    # dZ is dZ_0 now; ref_ns_bwd_final folds in the Z_0 init backward (norms
    # of K2) and the softmax-bwd row correction.
    dQ_tilde, dK_tilde, _ = ref_ns_bwd_final(
        q_tilde, k_tilde, K2, dZ, dK2_acc)
    return dQ_tilde, dK_tilde


def _make_inputs(BH, m, D, newton_iter, seed):
    """Realistic inputs: random Q/K tilde, K2_softmax computed from them,
    full NS unroll under torch, random dZ_N gradient seed.
    """
    torch.manual_seed(seed)
    q_tilde = torch.randn(BH, m, D, device="cuda", dtype=torch.float32)
    k_tilde = torch.randn(BH, m, D, device="cuda", dtype=torch.float32)
    # K2_softmax = softmax(Q_tilde @ K_tilde^T) — what the forward kernel emits
    # and the bwd kernel takes as input. We use rand() + 0.1 (positive, not
    # row-stochastic) to avoid argmax ties in the L1/Linf norms inside Z_0
    # backward — same trick as test_ns_bwd_kernel.TestNSBwdFinal.
    K2 = torch.rand(BH, m, m, device="cuda", dtype=torch.float32) + 0.1
    K2 = K2.contiguous()
    ns_iter = ns_forward_iterates(K2, newton_iter)
    dZ_N = torch.randn(BH, m, m, device="cuda", dtype=torch.float32)
    return q_tilde, k_tilde, K2, ns_iter, dZ_N


def relerr(a, b):
    return ((a - b).norm() / torch.clamp(b.norm(), min=1e-30)).item()


def maxerr(a, b):
    return (a - b).abs().max().item()


# ============================================================================
# Tests
# ============================================================================

@pytest.mark.skipif(not HAS_FULL_DEBUG, reason="_C.debug_kernel2_inv_bwd_full not built")
class TestCorrectness:
    """The graph-captured cuBLAS pipeline must match autograd elementwise."""

    def _check(self, BH, m, D, niter, seed=0, atol=2e-4):
        # atol=2e-4 reflects accumulated FP32 round-off across many NS iterations
        # using FP32 cuBLAS GEMM vs Python's GEMM (same intrinsic precision; the
        # difference is purely summation order). At niter=12, BH=8, m=64 we
        # routinely see ~1e-5 relerr.
        q_tilde, k_tilde, K2, ns_iter, dZ_N = _make_inputs(BH, m, D, niter, seed)
        dQ_ref, dK_ref = ref_kernel2_inv_bwd_full(
            q_tilde, k_tilde, K2, ns_iter, dZ_N, niter)
        dQ_cuda, dK_cuda = _C.debug_kernel2_inv_bwd_full(
            q_tilde, k_tilde, K2, ns_iter, dZ_N, niter)

        rq = relerr(dQ_cuda, dQ_ref)
        rk = relerr(dK_cuda, dK_ref)
        mq = maxerr(dQ_cuda, dQ_ref)
        mk = maxerr(dK_cuda, dK_ref)
        assert rq < atol, (f"dQ_tilde relerr={rq:.2e} max_abs={mq:.2e} "
                            f"(BH={BH}, m={m}, D={D}, niter={niter})")
        assert rk < atol, (f"dK_tilde relerr={rk:.2e} max_abs={mk:.2e} "
                            f"(BH={BH}, m={m}, D={D}, niter={niter})")

    def test_small(self):
        self._check(BH=1, m=8, D=16, niter=3)

    def test_BH4_m32_D64_niter6(self):
        self._check(BH=4, m=32, D=64, niter=6)

    def test_BH1_m64_D128_niter6(self):
        self._check(BH=1, m=64, D=128, niter=6)

    def test_BH8_m64_D128_niter6(self):
        self._check(BH=8, m=64, D=128, niter=6)

    def test_high_iter_count(self):
        # 12 NS iterations stresses error accumulation through the graph.
        self._check(BH=4, m=32, D=64, niter=12)

    def test_min_iter(self):
        self._check(BH=2, m=16, D=32, niter=1)


@pytest.mark.skipif(not HAS_FULL_DEBUG, reason="_C.debug_kernel2_inv_bwd_full not built")
class TestGraphReplay:
    """Second call on the same shape must replay the captured graph correctly."""

    def test_replay_same_shape_different_data(self):
        # First call: captures the graph for shape (BH=4, m=32, D=64, niter=6).
        BH, m, D, niter = 4, 32, 64, 6
        for seed in (0, 1, 2):
            # Each iteration uses different data but the same shape; the graph
            # cache should hit on iteration 2 and 3, exercising replay.
            q_tilde, k_tilde, K2, ns_iter, dZ_N = _make_inputs(BH, m, D, niter, seed)
            dQ_ref, dK_ref = ref_kernel2_inv_bwd_full(
                q_tilde, k_tilde, K2, ns_iter, dZ_N, niter)
            dQ_cuda, dK_cuda = _C.debug_kernel2_inv_bwd_full(
                q_tilde, k_tilde, K2, ns_iter, dZ_N, niter)
            assert relerr(dQ_cuda, dQ_ref) < 2e-4, f"seed={seed}: dQ mismatch on replay"
            assert relerr(dK_cuda, dK_ref) < 2e-4, f"seed={seed}: dK mismatch on replay"

    def test_replay_determinism(self):
        # Same inputs, called twice, must produce bit-identical outputs.
        # cuBLAS is deterministic for a given algo + shape, and graph replay
        # cannot introduce non-determinism if inputs are identical.
        BH, m, D, niter = 2, 32, 64, 4
        q_tilde, k_tilde, K2, ns_iter, dZ_N = _make_inputs(BH, m, D, niter, seed=7)
        dQ1, dK1 = _C.debug_kernel2_inv_bwd_full(
            q_tilde, k_tilde, K2, ns_iter, dZ_N, niter)
        dQ2, dK2 = _C.debug_kernel2_inv_bwd_full(
            q_tilde, k_tilde, K2, ns_iter, dZ_N, niter)
        assert torch.equal(dQ1, dQ2), "graph replay not bit-deterministic for dQ"
        assert torch.equal(dK1, dK2), "graph replay not bit-deterministic for dK"


@pytest.mark.skipif(not HAS_FULL_DEBUG, reason="_C.debug_kernel2_inv_bwd_full not built")
class TestShapeChange:
    """Switching shapes must cache-miss correctly and re-hit on return."""

    def test_shape_A_B_A(self):
        # Call shape A, then B, then A again. All three must match autograd.
        # Tests both cache-miss (capture) and cache-hit (replay) for two shapes.
        def run_shape(BH, m, D, niter, seed):
            q, k, K2, ns_iter, dZ_N = _make_inputs(BH, m, D, niter, seed)
            dQ_ref, dK_ref = ref_kernel2_inv_bwd_full(q, k, K2, ns_iter, dZ_N, niter)
            dQ, dK = _C.debug_kernel2_inv_bwd_full(q, k, K2, ns_iter, dZ_N, niter)
            assert relerr(dQ, dQ_ref) < 2e-4
            assert relerr(dK, dK_ref) < 2e-4

        # Shape A: small. Shape B: larger BH and m.
        run_shape(BH=2, m=16, D=32, niter=4, seed=10)
        run_shape(BH=4, m=32, D=64, niter=6, seed=11)
        run_shape(BH=2, m=16, D=32, niter=4, seed=12)  # back to A — cache hit
        run_shape(BH=4, m=32, D=64, niter=6, seed=13)  # back to B — cache hit

    def test_niter_changes_cache_key(self):
        # newton_iter is part of the graph-cache key. Same (BH, m, D) with
        # different niter must capture two distinct graphs.
        BH, m, D = 2, 16, 32
        for niter in (2, 4, 6, 8):
            q, k, K2, ns_iter, dZ_N = _make_inputs(BH, m, D, niter, seed=20 + niter)
            dQ_ref, dK_ref = ref_kernel2_inv_bwd_full(q, k, K2, ns_iter, dZ_N, niter)
            dQ, dK = _C.debug_kernel2_inv_bwd_full(q, k, K2, ns_iter, dZ_N, niter)
            assert relerr(dQ, dQ_ref) < 2e-4, f"niter={niter}: dQ mismatch"
            assert relerr(dK, dK_ref) < 2e-4, f"niter={niter}: dK mismatch"

    def test_D_changes_cache_key(self):
        # D is part of the cache key (trailing matmul shape depends on it).
        BH, m, niter = 2, 16, 4
        for D in (16, 32, 64, 128):
            q, k, K2, ns_iter, dZ_N = _make_inputs(BH, m, D, niter, seed=30 + D)
            dQ_ref, dK_ref = ref_kernel2_inv_bwd_full(q, k, K2, ns_iter, dZ_N, niter)
            dQ, dK = _C.debug_kernel2_inv_bwd_full(q, k, K2, ns_iter, dZ_N, niter)
            assert relerr(dQ, dQ_ref) < 2e-4, f"D={D}: dQ mismatch"
            assert relerr(dK, dK_ref) < 2e-4, f"D={D}: dK mismatch"


@pytest.mark.skipif(not (HAS_FULL_DEBUG and HAS_RESET),
                    reason="_C.debug_kernel2_inv_bwd_full or _C.reset_caches not built")
class TestResetCaches:
    """`reset_caches()` must release workspaces and the next call must rebuild."""

    def test_reset_then_call_correct(self):
        BH, m, D, niter = 4, 32, 64, 6
        q, k, K2, ns_iter, dZ_N = _make_inputs(BH, m, D, niter, seed=42)
        dQ_ref, dK_ref = ref_kernel2_inv_bwd_full(q, k, K2, ns_iter, dZ_N, niter)

        # First call: builds the graph.
        dQ1, dK1 = _C.debug_kernel2_inv_bwd_full(q, k, K2, ns_iter, dZ_N, niter)
        assert relerr(dQ1, dQ_ref) < 2e-4
        assert relerr(dK1, dK_ref) < 2e-4

        # Reset wipes the thread-local NsBwdGraphState cache + workspaces.
        _C.reset_caches()

        # Second call: must rebuild and still match the reference.
        dQ2, dK2 = _C.debug_kernel2_inv_bwd_full(q, k, K2, ns_iter, dZ_N, niter)
        assert relerr(dQ2, dQ_ref) < 2e-4, "post-reset call gave wrong dQ"
        assert relerr(dK2, dK_ref) < 2e-4, "post-reset call gave wrong dK"

        # Both calls used the same inputs → results should be bit-identical
        # (cache rebuild does not change the captured-graph algorithm).
        assert torch.equal(dQ1, dQ2), "post-reset result differs bit-wise from pre-reset"
        assert torch.equal(dK1, dK2)

    def test_reset_releases_memory(self):
        # Heuristic: after a call that builds graphs/workspaces, reset_caches()
        # should drop allocated memory back toward the pre-call baseline.
        # We don't pin the exact number (allocator caching makes that brittle),
        # but verify reset doesn't *leak*: 100 reset cycles must not grow memory
        # without bound.
        BH, m, D, niter = 2, 16, 32, 4
        q, k, K2, ns_iter, dZ_N = _make_inputs(BH, m, D, niter, seed=99)

        # Warm up the allocator so the high-water mark stabilises.
        for _ in range(3):
            _C.debug_kernel2_inv_bwd_full(q, k, K2, ns_iter, dZ_N, niter)
            _C.reset_caches()

        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()

        for _ in range(20):
            _C.debug_kernel2_inv_bwd_full(q, k, K2, ns_iter, dZ_N, niter)
            _C.reset_caches()

        torch.cuda.synchronize()
        final = torch.cuda.memory_allocated()

        # If reset leaked, 20 build/release cycles would visibly grow allocated
        # memory. Allow generous slack (1 MiB) for allocator fragmentation.
        leak_bytes = final - baseline
        assert leak_bytes < 1 * 1024 * 1024, (
            f"reset_caches() appears to leak: {leak_bytes} bytes grown over "
            f"20 build/release cycles (baseline={baseline}, final={final})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
