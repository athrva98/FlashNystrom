# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Root-cause instrumentation for the flash_nystrom MQAR collapse.

Trains flash_nystrom and nystrom_reference from the IDENTICAL init and IDENTICAL
data (same seed), at the collapsing lr, and logs per epoch:

  loss, recall, cond(K2) (max/mean over the probe), |q~| landmark magnitude,
  and for flash_nystrom the forward divergence ||fn_attn - ref_attn|| measured
  on the actual training-time activations.

Because both runs share init+data, any divergence in their trajectories is
attributable to the attention operator alone. The question this answers:
does the kernel drive K2 into the ill-conditioned tail (cond >= 1e14) where the
controlled test showed kernel != reference, and does the collapse coincide with
it? cond(K2) is computed externally from the layer input with the same formula
for both backends, so it is directly comparable.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .data import generate_mqar
from .model import MQARModel
from .train import _autocast_ctx, evaluate


def _segment_landmarks(x, m):
    B, H, N, D = x.shape
    seg = N // m
    trunc = seg * (m - 1)
    first = x[:, :, :trunc, :].reshape(B, H, m - 1, seg, D).mean(dim=3)
    last = x[:, :, trunc:N, :].mean(dim=2, keepdim=True)
    return torch.cat([first, last], dim=2)


@torch.no_grad()
def probe_stats(attn, x_normed, m, want_fwd_div):
    """cond(K2), landmark magnitude, and (optional) fn-vs-ref forward divergence,
    all from the real normed activations entering the attention layer."""
    H, D = attn.heads, attn.head_dim
    B, N, _ = x_normed.shape
    q = attn.q_proj(x_normed).view(B, N, H, D).transpose(1, 2).contiguous()
    k = attn.k_proj(x_normed).view(B, N, H, D).transpose(1, 2).contiguous()
    v = attn.v_proj(x_normed).view(B, N, H, D).transpose(1, 2).contiguous()
    s = D ** (-0.25)
    qt = _segment_landmarks((q * s).double(), m)
    kt = _segment_landmarks((k * s).double(), m)
    k2 = torch.softmax(qt @ kt.transpose(-2, -1), dim=-1)
    cond = torch.linalg.cond(k2)  # (B, H)
    qt_norm = qt.norm(dim=-1).max().item()
    fwd_div = float("nan")
    if want_fwd_div:
        from flash_nystrom import flash_nystrom_attention
        from flash_nystrom.reference import nystrom_attention_reference
        o_fn = flash_nystrom_attention(q, k, v, m, 6).float()
        o_ref = nystrom_attention_reference(q, k, v, m, 6).float()
        fwd_div = (o_fn - o_ref).abs().max().item()
    return cond.max().item(), cond.mean().item(), qt_norm, fwd_div


def train_one(backend, train_x, train_y, test_x, test_y, probe_x, args):
    torch.manual_seed(args.seed)  # identical init across backends
    dev = args.device
    dtype = torch.bfloat16
    model = MQARModel(
        vocab_size=args.vocab_size, max_seq_len=args.seq_len, dim=args.dim,
        depth=2, heads=args.heads, backend=backend, init="normal",
        num_landmarks=args.num_landmarks, newton_iter=6,
    ).to(dev)

    attn = model.layers[1].mixer  # odd layer = attention backend
    holder = {}
    attn.register_forward_pre_hook(lambda mod, inp: holder.__setitem__("x", inp[0].detach()))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    probe = probe_x.to(dev)

    print(f"\n=== {backend} (lr={args.lr}, seed={args.seed}) ===")
    print(f"{'ep':>3} {'loss':>8} {'recall%':>8} {'condK2_max':>12} "
          f"{'condK2_mean':>12} {'|q~|max':>9} {'fwd_div':>9}")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(train_x.size(0))
        running = 0.0
        steps = 0
        for i in range(0, train_x.size(0), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = train_x[idx].to(dev), train_y[idx].to(dev)
            opt.zero_grad(set_to_none=True)
            mask = yb != -100
            with _autocast_ctx(dev, dtype):
                h = model.encode(xb)
                loss = F.cross_entropy(model.head(h[mask]).float(), yb[mask])
            loss.backward()
            opt.step()
            running += loss.item()
            steps += 1
        sched.step()
        recall = evaluate(model, test_x, test_y, args.batch_size, dev, dtype)
        model.eval()
        with _autocast_ctx(dev, dtype):
            model(probe)  # fire the hook
        cmax, cmean, qtn, fdiv = probe_stats(
            attn, holder["x"], args.num_landmarks, want_fwd_div=(backend == "flash_nystrom"))
        fdiv_s = f"{fdiv:9.3f}" if fdiv == fdiv else "      n/a"
        print(f"{epoch+1:>3} {running/steps:>8.4f} {recall*100:>8.2f} "
              f"{cmax:>12.2e} {cmean:>12.2e} {qtn:>9.2f} {fdiv_s}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--num_kv_pairs", type=int, default=16)
    p.add_argument("--num_train", type=int, default=20000)
    p.add_argument("--num_test", type=int, default=2000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--num_landmarks", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--backends", nargs="+", default=["flash_nystrom", "nystrom_reference"])
    args = p.parse_args()

    common = dict(vocab_size=args.vocab_size, seq_len=args.seq_len,
                  num_kv_pairs=args.num_kv_pairs, power_a=0.01)
    train_x, train_y = generate_mqar(num_examples=args.num_train, seed=args.seed, **common)
    test_x, test_y = generate_mqar(num_examples=args.num_test, seed=args.seed + 500_000, **common)
    probe_x = test_x[:128]

    for backend in args.backends:
        train_one(backend, train_x, train_y, test_x, test_y, probe_x, args)


if __name__ == "__main__":
    main()
