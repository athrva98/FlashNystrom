# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Train and evaluate a small transformer on MQAR.

Recall accuracy is the fraction of query positions where the argmax prediction
equals the bound value. Example:

    python -m paper.mqar.train --backend flash_nystrom --num_landmarks 64
    python -m paper.mqar.train --backend sdpa

To run the m-sweep that the paper uses (fidelity vs capacity), vary
``--num_landmarks`` and hold everything else fixed.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from .data import generate_mqar
from .model import MQARModel


@torch.no_grad()
def _diag_probe(model, probe_x, kappa_star, device, dtype):
    """Backend-agnostic conditioning probe of the LEARNED landmark Gram K2.

    Tests the hypothesis that MQAR (seq_len 256, seg_len 4) is well-conditioned,
    so the ridge is unnecessary and a too-strong ridge only adds bias. Captures
    the Nystrom attention layer's input on a fixed probe batch via a pre-hook,
    then recomputes K2 (segment-mean landmarks + softmax) and the ridge
    M = K2^T K2 + lambda*I in fp32 from the layer's OWN q/k projections (so it is
    independent of which forward kernel runs). Returns dict with cond(K2),
    cond(M), and the ridged-pinv residual ||I - K2 P|| / ||I||. NaN for sdpa."""
    from flash_nystrom.reference import iterative_pinverse

    mix = None
    for layer in model.layers:
        mm = getattr(layer, "mixer", None)
        if mm is not None and hasattr(mm, "q_proj") and (
            hasattr(mm, "num_landmarks") or hasattr(mm, "config")):
            mix = mm
            break
    if mix is None:  # sdpa has no landmarks
        return {"cond_K2": float("nan"), "cond_M": float("nan"), "pinv_resid": float("nan")}

    m_land = getattr(mix, "num_landmarks", None) or mix.config.num_landmarks
    H, D = mix.heads, mix.head_dim
    cap = {}
    handle = mix.register_forward_pre_hook(lambda mod, inp: cap.__setitem__("x", inp[0].detach()))
    with _autocast_ctx(device, dtype):
        model.encode(probe_x.to(device))
    handle.remove()

    x = cap["x"].float()
    B, N, _ = x.shape
    q = (x @ mix.q_proj.weight.float().t()).view(B, N, H, D).transpose(1, 2)
    k = (x @ mix.k_proj.weight.float().t()).view(B, N, H, D).transpose(1, 2)
    s = D ** -0.25
    qs, ks = q * s, k * s
    seg = N // m_land
    tn = seg * (m_land - 1)
    qf = qs[:, :, :tn].reshape(B, H, m_land - 1, seg, D).mean(3)
    kf = ks[:, :, :tn].reshape(B, H, m_land - 1, seg, D).mean(3)
    ql = qs[:, :, tn:].mean(2, keepdim=True)
    kl = ks[:, :, tn:].mean(2, keepdim=True)
    qt = torch.cat([qf, ql], 2)
    kt = torch.cat([kf, kl], 2)
    K2 = torch.softmax(qt @ kt.transpose(-2, -1), -1)  # (B,H,m,m)

    def condmax(A):
        try:
            return torch.linalg.cond(A).flatten().max().item()
        except Exception:
            return float("inf")

    Kt2 = K2.transpose(-2, -1)
    M0 = Kt2 @ K2
    eye = torch.eye(m_land, device=K2.device, dtype=K2.dtype)
    if kappa_star > 0:
        n1 = K2.abs().sum(-2).amax(-1)
        ninf = K2.abs().sum(-1).amax(-1)
        lam = (n1 * ninf / kappa_star)[..., None, None]
        M = M0 + lam * eye
        P = iterative_pinverse(M, n_iter=16) @ Kt2
    else:
        M = M0
        P = iterative_pinverse(K2, n_iter=16)
    resid = (eye - K2 @ P).norm(dim=(-2, -1)).max().item() / eye.norm().item()
    return {"cond_K2": condmax(K2), "cond_M": condmax(M), "pinv_resid": resid}


def _autocast_ctx(device: str, dtype: torch.dtype):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    # CPU: run in fp32 (FlashNystrom falls back to its pure-PyTorch reference).
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)


@torch.no_grad()
def evaluate(model, inputs, labels, batch_size, device, dtype):
    model.eval()
    correct = total = 0
    for i in range(0, inputs.size(0), batch_size):
        xb = inputs[i : i + batch_size].to(device)
        yb = labels[i : i + batch_size].to(device)
        mask = yb != -100
        if not bool(mask.any()):
            continue
        with _autocast_ctx(device, dtype):
            h = model.encode(xb)
            logits_q = model.head(h[mask])  # head only at query positions
        pred = logits_q.float().argmax(dim=-1)
        correct += (pred == yb[mask]).sum().item()
        total += int(mask.sum().item())
    return correct / max(total, 1)


def _autobatch_config(args, train_x, train_y, device, dtype):
    """Find the largest batch one real MQAR train step fits (gathered head, so
    the memory matches training), then scale epochs to preserve the total
    optimizer-step budget of the batch-256 recipe. Returns (batch, epochs,
    eval_every) where eval_every keeps the number of evaluations ~constant."""
    from benchmarks.autobatch import search_and_profile

    num_train = train_x.size(0)

    def make_trial(bs):
        m = MQARModel(
            vocab_size=args.vocab_size, max_seq_len=args.seq_len, dim=args.dim,
            depth=args.depth, heads=args.heads, backend=args.backend, init=args.init,
            num_landmarks=args.num_landmarks, newton_iter=args.newton_iter,
            use_conv_residual=args.conv, kappa_star=args.kappa_star,
        ).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        xb, yb = train_x[:bs].to(device), train_y[:bs].to(device)
        mask = yb != -100

        def run():
            opt.zero_grad(set_to_none=True)
            with _autocast_ctx(device, dtype):
                loss = F.cross_entropy(m.head(m.encode(xb)[mask]).float(), yb[mask])
            loss.backward()
            opt.step()

        return run

    cap = min(args.autobatch_cap, num_train)
    res = search_and_profile(make_trial, lo=8, cap=cap, warmup=2, iters=3)
    bs = res["batch"] if res else args.batch_size
    # Back off the probed max: the bare-model probe misses the DataLoader/eval
    # buffers and fragmentation a full run holds, so training at the exact max
    # can OOM partway. Reserve headroom (round down to a multiple of 8).
    if res:
        bs = max(8, int(bs * args.autobatch_margin) // 8 * 8)
    # Preserve total optimizer steps vs the validated batch-256 recipe; the
    # per-config LR sweep absorbs the LR<->batch coupling.
    target_steps = math.ceil(num_train / 256) * args.epochs
    epochs = max(1, round(target_steps / math.ceil(num_train / bs)))
    eval_every = max(1, epochs // 64)
    return bs, epochs, eval_every


def build_optimizer(model, lr, weight_decay):
    """AdamW that honors the per-parameter optimizer conventions the vendored
    baselines carry, so each method trains as its authors intend:

      * Hyena tags its implicit-filter and positional-embedding parameters with
        `_optim` (the filter is designed to train at lr=1e-3 and the positional
        embedding at lr=1e-5, far below the base lr).
      * Mamba tags A_log and D with `_no_weight_decay`.

    The other backends (sdpa, linear_attention, nystrom, flash_nystrom) set
    neither, so all of their parameters land in the default group at the base
    lr/wd. A single flat AdamW would train Hyena's filter 10--1000x too hot and
    weight-decay Mamba's A_log/D, i.e. run those two methods incorrectly."""
    default, no_decay, custom = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if hasattr(p, "_optim"):
            custom.append(p)
        elif getattr(p, "_no_weight_decay", False):
            no_decay.append(p)
        else:
            default.append(p)
    groups = [
        {"params": default, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # One group per unique `_optim` config (Hyena: filter lr=1e-3, pos-emb lr=1e-5).
    seen = set()
    for p in custom:
        key = frozenset(p._optim.items())
        if key in seen:
            continue
        seen.add(key)
        grp = {"params": [q for q in custom if frozenset(q._optim.items()) == key]}
        grp.update(dict(p._optim))
        groups.append(grp)
    groups = [g for g in groups if len(g["params"]) > 0]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def train(args):
    torch.manual_seed(args.seed)
    device = args.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    common = dict(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_kv_pairs=args.num_kv_pairs,
        power_a=args.power_a,
        random_non_queries=args.random_non_queries,
    )
    # Fresh synthetic data each epoch (standard MQAR practice): the model can
    # never overfit a fixed train set, which makes the phase transition far more
    # reliable. With --no-fresh_data a single fixed train set is reused.
    def make_train(epoch):
        s = args.seed * 1000 + epoch if args.fresh_data else args.seed
        return generate_mqar(num_examples=args.num_train, seed=s, **common)

    train_x, train_y = make_train(0)
    # Disjoint seed for the test set so we measure generalization, not memorization.
    test_x, test_y = generate_mqar(num_examples=args.num_test, seed=args.seed + 500_000, **common)

    eval_every = 1
    if args.autobatch and device == "cuda":
        args.batch_size, args.epochs, eval_every = _autobatch_config(
            args, train_x, train_y, device, dtype)
        print(f"autobatch: batch_size={args.batch_size} epochs={args.epochs} "
              f"eval_every={eval_every}")

    model = MQARModel(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        backend=args.backend,
        init=args.init,
        num_landmarks=args.num_landmarks,
        newton_iter=args.newton_iter,
        use_conv_residual=args.conv,
        kappa_star=args.kappa_star,
        layer_layout=args.layer_layout,
        causal=args.causal,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    arch = (
        f"backend={args.backend} dim={args.dim} depth={args.depth} "
        f"heads={args.heads} head_dim={args.dim // args.heads} init={args.init}"
    )
    if args.backend in ("flash_nystrom", "nystrom_reference"):
        arch += f" m={args.num_landmarks} newton_iter={args.newton_iter}"
        if args.conv:
            arch += " conv=on"
    arch += (f" layout={args.layer_layout} "
             f"causal={'on' if args.causal else 'off'} "
             f"pos_emb={'on' if model.use_pos_emb else 'off'} "
             f"params={n_params / 1e6:.2f}M")
    print(arch)
    print(
        f"data: vocab={args.vocab_size} seq_len={args.seq_len} kv_pairs={args.num_kv_pairs} "
        f"train={args.num_train} test={args.num_test}"
    )

    opt = build_optimizer(model, args.lr, args.weight_decay)
    steps_per_epoch = math.ceil(args.num_train / args.batch_size)
    # Zoology's MQAR recipe: cosine-anneal from the full LR over all epochs,
    # stepped once PER EPOCH with NO warmup. MQAR's phase transition fires late
    # and needs the LR held high early; OneCycle's warmup + per-step anneal drives
    # the LR toward zero before the transition can complete, which strands the
    # model on the ~22% pre-transition plateau.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.0)
    # GradScaler is only needed for fp16; bf16 has fp32 dynamic range.
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and dtype == torch.float16))

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Fixed probe batch for the per-eval conditioning diagnostic (--diag).
    diag_probe_x = test_x[: min(64, test_x.size(0))]
    epoch_times = []
    best_acc = 0.0
    for epoch in range(args.epochs):
        if args.fresh_data and epoch > 0:
            train_x, train_y = make_train(epoch)
        model.train()
        perm = torch.randperm(train_x.size(0))
        running = 0.0
        if device == "cuda":
            torch.cuda.synchronize()
        ep_start = time.perf_counter()
        last_gn = float("nan")
        for i in range(0, train_x.size(0), args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb = train_x[idx].to(device)
            yb = train_y[idx].to(device)
            opt.zero_grad(set_to_none=True)
            mask = yb != -100  # only query positions carry a label
            with _autocast_ctx(device, dtype):
                h = model.encode(xb)
                logits_q = model.head(h[mask])  # (num_queries, vocab)
                # Mean-reduced CE over query positions only: identical to
                # cross_entropy(full_logits, yb, ignore_index=-100) but it skips
                # the (B, N, vocab) head matmul over the ~94% ignored positions.
                loss = F.cross_entropy(logits_q.float(), yb[mask])
            scaler.scale(loss).backward()
            # No gradient clipping. No paper in this lineage specifies it for the
            # recall synthetics: Zoology (App E.2), Based and JRT never mention
            # clipping at all, Hyena never mentions it, and Mamba specifies it only
            # for LM scaling laws (App E.2.1, 1.0) and speech (E.4.2, 0.1), NOT for
            # its synthetic tasks (E.1). Zoology's trainer does not clip either.
            # An infinite max_norm only MEASURES the norm, it does not clip.
            if args.diag:
                last_gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        if device == "cuda":
            torch.cuda.synchronize()
        if epoch > 0:  # skip first epoch (warmup / lazy init)
            epoch_times.append(time.perf_counter() - ep_start)
        sched.step()  # per-epoch cosine step (Zoology recipe)
        if (epoch + 1) % eval_every == 0 or epoch == args.epochs - 1:
            acc = evaluate(model, test_x, test_y, args.batch_size, device, dtype)
            best_acc = max(best_acc, acc)
            line = (f"epoch {epoch+1:4d}/{args.epochs}  loss {running/steps_per_epoch:.4f}  "
                    f"test recall {acc*100:.2f}%")
            if args.diag:
                d = _diag_probe(model, diag_probe_x, args.kappa_star, device, dtype)
                line += (f"  grad_norm {last_gn:.3f}  cond_K2 {d['cond_K2']:.2e}  "
                         f"cond_M {d['cond_M']:.2e}  pinv_resid {d['pinv_resid']:.2e}")
            print(line)
            # Zoology stops a run once validation accuracy clears the threshold
            # (config.py:133-134, early_stopping_threshold=0.99). Only cuts epochs
            # off already-solved runs, so it cannot lower the reported best recall.
            if args.early_stop_acc > 0 and acc >= args.early_stop_acc:
                print(f"early stop at epoch {epoch+1}: test recall {acc*100:.2f}% "
                      f">= {args.early_stop_acc*100:.2f}%")
                break

    print(f"best test recall: {best_acc*100:.2f}%")
    # Training throughput + peak memory (median epoch after warmup).
    step_ms = samp_s = peak = None
    if epoch_times:
        med = sorted(epoch_times)[len(epoch_times) // 2]
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3) if device == "cuda" else 0.0
        step_ms = med / steps_per_epoch * 1000
        samp_s = args.num_train / med
        print(f"train profile: batch={args.batch_size} "
              f"step_ms={step_ms:.2f} "
              f"samples_per_s={samp_s:.1f} peak_GiB={peak:.2f}")
    # Structured result for notebook/aggregate collection.
    if getattr(args, "out_json", None):
        import json
        rec = {
            "backend": args.backend, "seed": args.seed,
            "best_recall": best_acc * 100.0,
            "seq_len": args.seq_len, "num_kv_pairs": args.num_kv_pairs,
            "num_landmarks": args.num_landmarks, "newton_iter": args.newton_iter,
            "kappa_star": args.kappa_star, "dim": args.dim, "depth": args.depth,
            "heads": args.heads, "epochs": args.epochs, "batch_size": args.batch_size,
            "num_train": args.num_train, "lr": args.lr,
            "step_ms": step_ms, "samples_per_s": samp_s, "peak_GiB": peak,
            # protocol provenance (which recipe this number came from)
            "num_test": args.num_test, "layer_layout": args.layer_layout,
            "random_non_queries": args.random_non_queries,
            "early_stop_acc": args.early_stop_acc,
            "use_pos_emb": model.use_pos_emb,
            "causal": args.causal,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"saved -> {args.out_json}")
    return best_acc


def build_parser():
    p = argparse.ArgumentParser(description="MQAR training for FlashNystrom vs full attention")
    # data
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--num_kv_pairs", type=int, default=16)
    p.add_argument("--power_a", type=float, default=0.01)
    p.add_argument("--num_train", type=int, default=20000)
    p.add_argument("--num_test", type=int, default=2000)
    p.add_argument("--random_non_queries", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="fill non-query slots with random tokens (generate_mqar "
                        "default). Zoology's iclr24_zoology_figure2 config sets this "
                        "False (blank slots); pass --no-random_non_queries to match it.")
    # model
    p.add_argument(
        "--backend",
        choices=["sdpa", "flash_nystrom", "flash_nystrom_tc", "nystrom_reference",
                 "linear_attention", "hyena", "mamba"],
        default="sdpa",
    )
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=2, help="dim/heads must be 64 or 128 for flash_nystrom")
    p.add_argument("--causal", action="store_true",
                   help="DIAGNOSTIC ONLY, not part of the paper benchmark: causal "
                        "attention mask for backends that support it (sdpa), for "
                        "reproducing Zoology's causal-attention setting when checking "
                        "the harness against published numbers. The benchmark runs "
                        "every maskable method bidirectionally (the Nystrom family has "
                        "no causal form; masking only the baselines would confound the "
                        "operator comparison). Hyena/Mamba are causal by construction "
                        "regardless; the Nystrom family rejects this flag.")
    p.add_argument("--layer_layout", choices=["hybrid", "uniform"], default="hybrid",
                   help="hybrid (default): even BaseConv / odd mixer, Zoology's "
                        "original_mqar_configs + models_repo recipe. uniform: every "
                        "layer is the mixer, no BaseConv -- the iclr24_zoology_figure2 "
                        "recipe the DeltaNet paper's MQAR figure uses.")
    p.add_argument("--num_landmarks", type=int, default=64)
    p.add_argument("--newton_iter", type=int, default=6)
    p.add_argument("--kappa_star", type=float, default=1.0e3,
                   help="Tikhonov ridge target cond(M), threaded identically to "
                        "flash_nystrom and the reference. 0 = no ridge (vanilla).")
    p.add_argument("--diag", action="store_true",
                   help="log per-eval conditioning diagnostics (grad_norm, cond(K2), "
                        "cond(M), ridged-pinv residual) on a fixed probe batch")
    p.add_argument("--init", choices=["normal", "orthogonal"], default="normal",
                   help="weight init for Linear layers (orthogonal = head-independence ablation)")
    p.add_argument("--conv", action="store_true", help="enable the Nystromformer depthwise-conv residual")
    # optim
    p.add_argument("--epochs", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--autobatch", action="store_true",
                   help="auto-select the largest batch that fits and scale epochs "
                        "to preserve the batch-256 optimizer-step budget")
    p.add_argument("--autobatch_cap", type=int, default=8192,
                   help="upper bound on the auto-selected batch size")
    p.add_argument("--autobatch_margin", type=float, default=0.85,
                   help="fraction of the probed max batch to train at (default "
                        "0.85); reserves headroom so a full run does not OOM after "
                        "the bare-model probe")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--early_stop_acc", type=float, default=0.0,
                   help="stop once test recall reaches this fraction (Zoology's "
                        "early_stopping_threshold=0.99, config.py:134). 0 disables. "
                        "Only trims already-solved runs, never lowers reported recall.")
    p.add_argument("--fresh_data", action=argparse.BooleanOptionalAction, default=False,
                   help="regenerate train data each epoch (standard MQAR; --no-fresh_data reuses a fixed set)")
    # run
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--out_json", type=str, default=None,
                   help="write a structured result record (backend, seed, recall, "
                        "config, timing) to this path for notebook/aggregate collection")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
