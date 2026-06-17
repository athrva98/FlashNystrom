# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Neutral investigation of the flash_nystrom MQAR collapses (the 0.0x% runs).

No prior that the kernels are at fault. We feed identical, increasingly
ill-conditioned (q, k, v) to every path and separately measure FORWARD and
BACKWARD finiteness, plus the numerical gap vs the pure-PyTorch reference:

  * reference            : nystrom_attention_reference (fp32 NS pinv + autograd)
  * fn fast_dk2inv=False  : custom fwd + fp32 scalar dk2inv bwd (claims ref-equiv)
  * fn fast_dk2inv=True   : custom fwd + FP16/BF16 tensor-core dk2inv bwd (default)

cond(K2) is the condition number of the landmark Gram matrix, computed in fp64.
We scale q,k to drive K2 toward singular (what aggressive-LR MQAR training does).

Whatever goes non-finite FIRST, and in which path, locates the fault empirically.
"""
from __future__ import annotations

import torch

from flash_nystrom import flash_nystrom_attention
from flash_nystrom.reference import nystrom_attention_reference


def _segment_landmarks(x, m):
    """Replicate the reference's segment-mean landmarks (fp64) for conditioning."""
    B, H, N, D = x.shape
    seg = N // m
    trunc = seg * (m - 1)
    first = x[:, :, :trunc, :].reshape(B, H, m - 1, seg, D).mean(dim=3)
    last = x[:, :, trunc:N, :].mean(dim=2, keepdim=True)
    return torch.cat([first, last], dim=2)


def cond_k2(q, k, m):
    """Condition number of K2 = softmax(q~ @ k~^T), in fp64."""
    D = q.shape[-1]
    s = D ** (-0.25)
    qt = _segment_landmarks((q * s).double(), m)
    kt = _segment_landmarks((k * s).double(), m)
    logits = qt @ kt.transpose(-2, -1)
    k2 = torch.softmax(logits, dim=-1)
    return torch.linalg.cond(k2).max().item()


def finite(t):
    return bool(torch.isfinite(t).all())


def run_path(fn, q, k, v):
    """Run forward + backward for one path; return (out, [dq,dk,dv])."""
    q = q.clone().detach().requires_grad_(True)
    k = k.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)
    out = fn(q, k, v)
    # Deterministic non-trivial scalar to backprop (fixed pseudo-random weights).
    g = torch.sin(torch.arange(out.numel(), device=out.device, dtype=torch.float32)
                  ).reshape(out.shape).to(out.dtype)
    (out * g).sum().backward()
    return out.detach(), [q.grad.detach(), k.grad.detach(), v.grad.detach()]


def grad_agreement(g_ref, g_test):
    """Cosine similarity and max relative error between two grad sets."""
    a = torch.cat([x.float().flatten() for x in g_ref])
    b = torch.cat([x.float().flatten() for x in g_test])
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((b - a).norm() / (a.norm() + 1e-12)).item()
    return cos, rel


def main():
    torch.manual_seed(0)
    dev = "cuda"
    B, H, N, D, m, ni = 4, 2, 256, 64, 64, 6
    dtype = torch.bfloat16  # the MQAR training dtype

    q0 = torch.randn(B, H, N, D, device=dev, dtype=dtype)
    k0 = torch.randn(B, H, N, D, device=dev, dtype=dtype)
    v0 = torch.randn(B, H, N, D, device=dev, dtype=dtype)

    paths = [
        ("reference",
         lambda q, k, v: nystrom_attention_reference(q, k, v, m, ni)),
        ("fn fast_dk2inv=False",
         lambda q, k, v: flash_nystrom_attention(q, k, v, m, ni, fast_dk2inv=False)),
        ("fn fast_dk2inv=True ",
         lambda q, k, v: flash_nystrom_attention(q, k, v, m, ni, fast_dk2inv=True)),
    ]

    print(f"shape B={B} H={H} N={N} D={D} m={m} newton_iter={ni} dtype={dtype}")
    print("Gradient agreement vs the exact-autograd reference (cosine sim, rel L2 err).")
    print("scale multiplies q,k -> sharper softmax -> worse cond(K2). scale 1-4 is the")
    print("realistic trained regime; higher scales probe the ill-conditioned tail.\n")
    print(f"{'scale':>5} {'cond(K2)':>11} | "
          f"{'fast_dk2inv=False':>28} | {'fast_dk2inv=True':>28}")
    print(f"{'':5} {'':11} | {'cos':>13} {'relerr':>13} | {'cos':>13} {'relerr':>13}")
    print("-" * 92)

    for scale in [1.0, 2.0, 4.0, 8.0, 16.0]:
        q, k, v = q0 * scale, k0 * scale, v0
        c = cond_k2(q, k, m)
        _, g_ref = run_path(paths[0][1], q, k, v)
        _, g_false = run_path(paths[1][1], q, k, v)
        _, g_true = run_path(paths[2][1], q, k, v)
        cos_f, rel_f = grad_agreement(g_ref, g_false)
        cos_t, rel_t = grad_agreement(g_ref, g_true)
        print(f"{scale:>5.0f} {c:>11.2e} | {cos_f:>13.5f} {rel_f:>13.2e} | "
              f"{cos_t:>13.5f} {rel_t:>13.2e}")


if __name__ == "__main__":
    main()
