# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""A hand-written Triton Nystrom attention forward, as a baseline for the paper.

The obvious objection to a hand-fused CUDA kernel is that Triton should get most
of the way there for a fraction of the effort. ``tab:compile`` answers the
automatic version of that (Inductor emits Triton), but not the deliberate one.
This is the deliberate one: what a competent engineer writing Triton directly
would produce.

The factorization is

    O = softmax(Q Kt') . pinv(softmax(Qt Kt')) . [softmax(Qt K') V]

with landmarks Qt, Kt the segment means of Q, K. Two stages are worth fusing and
both are written here:

  stage 1  ``_kernel_landmark_av``: softmax(Qt K') V with an online softmax over
           N. Structurally a FlashAttention forward with only m query rows, so
           the (m, N) probability matrix never reaches HBM.
  stage 2  ``_kernel_p1_z``: softmax(Q Kt') @ Z folded into one pass, so the
           (N, m) probability matrix never reaches HBM either. m <= 128 fits a
           single block, so this softmax needs no online pass.

The m x m pseudoinverse stays in torch. It is O(m^3) at m=64, far off the
critical path, and hand-writing Newton-Schulz in Triton would be a strawman
rather than a stronger baseline.

FORWARD ONLY, deliberately. A hand-written Triton backward for this chain is a
substantially larger undertaking than the forward, which is itself part of the
paper's argument: the analytic fused backward is the harder contribution. The
comparison in the paper is therefore forward-vs-forward and says so.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:                                          # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _kernel_landmark_av(Qt, K, V, Acc, Mx, Lx, N, stride_qb, stride_kb,
                            stride_vb, stride_ab, stride_sb, scale,
                            M: tl.constexpr, D: tl.constexpr,
                            BLOCK_N: tl.constexpr, SPLITS: tl.constexpr):
        """Partial softmax(Qt @ K') @ V over one slice of N.

        SPLIT-K over the sequence. One program per (batch, head) would launch
        only B*H programs -- 8 on a 108-SM A100 -- and serialize the whole N
        loop inside each, which measured SLOWER than unfused cuBLAS at large N.
        Each program now owns N/SPLITS of the sequence and emits its partial
        (acc, max, sumexp); the host combines them. This is the same
        decomposition FlashAttention uses for decoding.
        """
        split = tl.program_id(0)
        bh = tl.program_id(1)
        offs_m = tl.arange(0, M)
        offs_d = tl.arange(0, D)

        qt = tl.load(Qt + bh * stride_qb + offs_m[:, None] * D + offs_d[None, :])
        qt = qt.to(tl.float32)

        chunk = tl.cdiv(N, SPLITS)
        lo = split * chunk
        hi = tl.minimum(lo + chunk, N)

        m_i = tl.full((M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((M,), dtype=tl.float32)
        acc = tl.zeros((M, D), dtype=tl.float32)

        for start in range(lo, hi, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            valid = offs_n < hi
            k = tl.load(K + bh * stride_kb + offs_n[:, None] * D + offs_d[None, :],
                        mask=valid[:, None], other=0.0).to(tl.float32)
            v = tl.load(V + bh * stride_vb + offs_n[:, None] * D + offs_d[None, :],
                        mask=valid[:, None], other=0.0).to(tl.float32)

            s = tl.dot(qt, tl.trans(k)) * scale
            s = tl.where(valid[None, :], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])

            acc = acc * alpha[:, None] + tl.dot(p, v)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

        o = split * stride_ab + bh * (SPLITS * stride_ab)
        tl.store(Acc + o + offs_m[:, None] * D + offs_d[None, :], acc)
        so = split * stride_sb + bh * (SPLITS * stride_sb)
        tl.store(Mx + so + offs_m, m_i)
        tl.store(Lx + so + offs_m, l_i)

    @triton.jit
    def _kernel_p1_z(Q, Kt, Z, Out, N, stride_qb, stride_kb, stride_zb,
                     stride_ob, scale,
                     M: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
        """out[n, d] = softmax_over_M(Q @ Kt') @ Z.

        The softmax is over M <= 128, which fits one block, so no online pass is
        needed; the (BLOCK_N, M) probabilities stay in registers and the (N, M)
        matrix is never formed.
        """
        pid = tl.program_id(0)
        bh = tl.program_id(1)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, M)
        offs_d = tl.arange(0, D)
        valid = offs_n < N

        q = tl.load(Q + bh * stride_qb + offs_n[:, None] * D + offs_d[None, :],
                    mask=valid[:, None], other=0.0).to(tl.float32)
        kt = tl.load(Kt + bh * stride_kb + offs_m[:, None] * D + offs_d[None, :]
                     ).to(tl.float32)
        z = tl.load(Z + bh * stride_zb + offs_m[:, None] * D + offs_d[None, :]
                    ).to(tl.float32)

        s = tl.dot(q, tl.trans(kt)) * scale                      # (BLOCK_N, M)
        s = s - tl.max(s, 1)[:, None]
        p = tl.exp(s)
        p = p / tl.sum(p, 1)[:, None]

        o = tl.dot(p, z)                                         # (BLOCK_N, D)
        tl.store(Out + bh * stride_ob + offs_n[:, None] * D + offs_d[None, :],
                 o.to(Out.dtype.element_ty), mask=valid[:, None])


def _segment_means(x, m):
    """Landmarks as segment means, matching the reference implementation."""
    b, h, n, d = x.shape
    if n % m == 0:
        return x.view(b, h, m, n // m, d).mean(dim=3)
    # ragged tail: pad to a multiple of m and correct the divisor
    pad = (m - n % m) % m
    xp = torch.nn.functional.pad(x, (0, 0, 0, pad))
    seg = xp.shape[2] // m
    cnt = torch.full((m,), float(seg), device=x.device, dtype=torch.float32)
    cnt[-1] -= pad
    return (xp.view(b, h, m, seg, d).sum(dim=3)
            / cnt.view(1, 1, m, 1).to(xp.dtype))


def triton_nystrom_forward(q, k, v, num_landmarks=64, newton_iter=6,
                           block_n=64):
    """Nystrom attention forward, Triton for the two fusible stages.

    q, k, v: (B, H, N, D) contiguous. Returns (B, H, N, D).
    """
    if not HAS_TRITON:                                       # pragma: no cover
        raise ImportError("this baseline needs triton: pip install triton")
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    b, h, n, d = q.shape
    m = num_landmarks
    assert m <= 128, "stage 2 assumes the softmax over m fits one block"
    scale = d ** -0.5

    qt = _segment_means(q, m).contiguous()
    kt = _segment_means(k, m).contiguous()

    q2, k2, v2 = (t.view(b * h, n, d) for t in (q, k, v))
    qt2, kt2 = (t.view(b * h, m, d) for t in (qt, kt))

    # stage 1: F3 @ V, split-K online softmax over N
    splits = max(1, min(64, triton.cdiv(n, 4096)))
    acc = torch.empty((b * h, splits, m, d), device=q.device, dtype=torch.float32)
    mx = torch.empty((b * h, splits, m), device=q.device, dtype=torch.float32)
    lx = torch.empty((b * h, splits, m), device=q.device, dtype=torch.float32)
    _kernel_landmark_av[(splits, b * h)](
        qt2, k2, v2, acc, mx, lx, n,
        qt2.stride(0), k2.stride(0), v2.stride(0), m * d, m, scale,
        M=m, D=d, BLOCK_N=block_n, SPLITS=splits, num_warps=4,
    )
    # combine the partials: rescale each split to the global max, then sum
    gmax = mx.max(dim=1, keepdim=True).values                     # (BH,1,m)
    w = torch.exp(mx - gmax)                                      # (BH,splits,m)
    f3v = ((acc * w.unsqueeze(-1)).sum(1)
           / (lx * w).sum(1).unsqueeze(-1)).to(q.dtype).contiguous()

    # m x m pseudoinverse: torch, off the critical path at m=64.
    #
    # Newton-Schulz, NOT torch.linalg.pinv. K2 is ill-conditioned enough at
    # these lengths that six NS iterations have not converged to the true
    # pseudoinverse: substituting the exact one changes the OUTPUT by a
    # relative 1.2-2.2, an O(1) difference. Nystromformer's result is specific
    # to the NS iteration, so a baseline using exact pinv would be computing a
    # different function and its latency would not be comparable.
    from flash_nystrom.reference import iterative_pinverse
    f2 = torch.softmax(
        (qt2.float() @ kt2.float().transpose(-2, -1)) * scale, dim=-1)
    z = (iterative_pinverse(f2, n_iter=newton_iter)
         @ f3v.float()).to(q.dtype).contiguous()

    # stage 2: softmax(Q Kt') @ Z in one pass
    out = torch.empty_like(q2)
    grid = (triton.cdiv(n, block_n), b * h)
    _kernel_p1_z[grid](
        q2, kt2, z, out, n,
        q2.stride(0), kt2.stride(0), z.stride(0), out.stride(0), scale,
        M=m, D=d, BLOCK_N=block_n, num_warps=4,
    )
    return out.view(b, h, n, d)
