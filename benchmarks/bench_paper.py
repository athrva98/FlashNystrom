# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0

"""
Paper-quality benchmarks for FlashNystrom.

Produces tables and numbers suitable for direct inclusion in a SysML/NeurIPS
workshop paper. Covers:

  1. Latency: forward, backward, fwd+bwd across N=512..32768
  2. Memory: peak GPU memory at each N
  3. Throughput: tokens/sec for training iterations
  4. Correctness: gradient cosine similarity vs FP32 reference
  5. Training: end-to-end ViT on CIFAR-10 comparing FlashNystrom vs SDPA

Methodology:
  - CUDA events for timing (not wall-clock)
  - 10 warmup + 50 timed runs, report median (robust to outliers)
  - torch.cuda.reset_peak_memory_stats() for accurate peak memory
  - All runs on the same GPU, same dtype (FP16), same random seed
  - SDPA uses PyTorch's scaled_dot_product_attention (FlashAttention backend)
"""

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../benchmarks
_REPO = os.path.dirname(_HERE)                       # repo root
_DATA = os.environ.get("FN_DATA_DIR", os.path.join(_REPO, "data"))

import torch
import torch.nn as nn


# ---- timing utility ----

def benchmark_cuda(fn, warmup=10, repeat=50):
    """Returns dict with median_ms, min_ms, mean_ms, std_ms from CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    events = [(torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in events:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in events)
    n = len(times)
    median = times[n // 2]
    mean = sum(times) / n
    std = (sum((t - mean) ** 2 for t in times) / n) ** 0.5
    return {"median_ms": median, "min_ms": times[0], "mean_ms": mean, "std_ms": std}


def peak_memory_mb(fn, warmup=3):
    """Run fn and return peak GPU memory in MB."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


# ---- benchmark sections ----

def section_latency(B, H, D, m, dtype, seq_lengths):
    """Table 1: Forward, backward, and fwd+bwd latency comparison."""
    from flash_nystrom.flash_nystrom import FlashNystromFunction

    print("\n" + "=" * 110)
    print("TABLE 1: Latency (ms) — FlashNystrom vs SDPA")
    print(f"Config: B={B}, H={H}, D={D}, m={m}, dtype={'fp16' if dtype == torch.float16 else 'bf16'}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Timing: CUDA events, 10 warmup + 50 timed, median reported")
    print("=" * 110)
    hdr = f"{'N':>7} | {'FN fwd':>8} {'FN bwd':>8} {'FN total':>9} | {'SDPA fwd':>9} {'SDPA bwd':>9} {'SDPA total':>10} | {'Speedup':>8}"
    print(hdr)
    print("-" * 110)

    results = []
    for N in seq_lengths:
        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)

        # FlashNystrom forward
        fn_fwd = benchmark_cuda(
            lambda: FlashNystromFunction.apply(q, k, v, m, 6, True, 5.0, True))

        # FlashNystrom fwd+bwd
        def fn_fwd_bwd():
            qq = q.detach().requires_grad_(True)
            kk = k.detach().requires_grad_(True)
            vv = v.detach().requires_grad_(True)
            FlashNystromFunction.apply(qq, kk, vv, m, 6, True, 5.0, True).sum().backward()

        fn_fb = benchmark_cuda(fn_fwd_bwd, warmup=5, repeat=30)
        fn_bwd = fn_fb["median_ms"] - fn_fwd["median_ms"]

        # SDPA
        sdpa_fwd_t, sdpa_bwd_t, sdpa_total_t = None, None, None
        speedup = ""
        try:
            sdpa_fwd = benchmark_cuda(
                lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

            def sdpa_fwd_bwd():
                qq = q.detach().requires_grad_(True)
                kk = k.detach().requires_grad_(True)
                vv = v.detach().requires_grad_(True)
                torch.nn.functional.scaled_dot_product_attention(qq, kk, vv).sum().backward()

            sdpa_fb = benchmark_cuda(sdpa_fwd_bwd, warmup=5, repeat=30)
            sdpa_fwd_t = sdpa_fwd["median_ms"]
            sdpa_bwd_t = sdpa_fb["median_ms"] - sdpa_fwd["median_ms"]
            sdpa_total_t = sdpa_fb["median_ms"]
            speedup = f"{sdpa_total_t / fn_fb['median_ms']:.1f}x"
        except RuntimeError:
            sdpa_fwd_t = sdpa_bwd_t = sdpa_total_t = float("nan")
            speedup = "OOM"

        row = {
            "N": N,
            "fn_fwd": fn_fwd["median_ms"],
            "fn_bwd": fn_bwd,
            "fn_total": fn_fb["median_ms"],
            "sdpa_fwd": sdpa_fwd_t,
            "sdpa_bwd": sdpa_bwd_t,
            "sdpa_total": sdpa_total_t,
            "speedup": speedup,
        }
        results.append(row)

        sf = lambda v: f"{v:.2f}" if v and not (isinstance(v, float) and v != v) else "OOM"
        print(f"{N:>7} | {fn_fwd['median_ms']:>8.2f} {fn_bwd:>8.2f} {fn_fb['median_ms']:>9.2f} | "
              f"{sf(sdpa_fwd_t):>9} {sf(sdpa_bwd_t):>9} {sf(sdpa_total_t):>10} | {speedup:>8}")

        # free memory for next iteration
        del q, k, v
        torch.cuda.empty_cache()

    return results


def section_memory(B, H, D, m, dtype, seq_lengths):
    """Table 2: Peak GPU memory comparison."""
    from flash_nystrom.flash_nystrom import FlashNystromFunction

    print("\n" + "=" * 80)
    print("TABLE 2: Peak GPU Memory (MB) — FlashNystrom vs SDPA")
    print(f"Config: B={B}, H={H}, D={D}, m={m}")
    print("=" * 80)
    print(f"{'N':>7} | {'FN fwd+bwd':>12} | {'SDPA fwd+bwd':>13} | {'Savings':>8}")
    print("-" * 80)

    results = []
    for N in seq_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # FlashNystrom
        torch.manual_seed(42)
        def fn_run():
            q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
            k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
            v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
            FlashNystromFunction.apply(q, k, v, m, 6, True, 5.0, True).sum().backward()
        fn_mem = peak_memory_mb(fn_run, warmup=2)

        # SDPA
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        sdpa_mem = None
        try:
            def sdpa_run():
                q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
                k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
                v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
                torch.nn.functional.scaled_dot_product_attention(q, k, v).sum().backward()
            sdpa_mem = peak_memory_mb(sdpa_run, warmup=2)
        except RuntimeError:
            sdpa_mem = float("inf")

        savings = ""
        if sdpa_mem != float("inf"):
            savings = f"{sdpa_mem / fn_mem:.1f}x" if fn_mem > 0 else "N/A"
        else:
            savings = "OOM"

        row = {"N": N, "fn_mem_mb": fn_mem,
               "sdpa_mem_mb": sdpa_mem if sdpa_mem != float("inf") else None,
               "savings": savings}
        results.append(row)

        sm = f"{sdpa_mem:.0f}" if sdpa_mem != float("inf") else "OOM"
        print(f"{N:>7} | {fn_mem:>12.0f} | {sm:>13} | {savings:>8}")

        torch.cuda.empty_cache()

    return results


def section_correctness(D_vals, dtype):
    """Table 3: Gradient cosine similarity vs FP32 reference."""
    from flash_nystrom.flash_nystrom import FlashNystromFunction
    from flash_nystrom.reference import nystrom_attention_reference_simple

    print("\n" + "=" * 90)
    print("TABLE 3: Gradient Correctness — Cosine Similarity vs FP32 Reference")
    print("=" * 90)
    print(f"{'Config':>25} | {'dQ cos':>8} | {'dK cos':>8} | {'dV cos':>8}")
    print("-" * 90)

    results = []
    for D in D_vals:
        m = 64 if D == 128 else 32
        for N in [256, 1024, 4096]:
            torch.manual_seed(0)
            B, H = 1, 2
            q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
            k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
            v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)

            FlashNystromFunction.apply(q, k, v, m, 6, True, 5.0, True).sum().backward()

            q2 = q.detach().float().cpu().requires_grad_(True)
            k2 = k.detach().float().cpu().requires_grad_(True)
            v2 = v.detach().float().cpu().requires_grad_(True)
            nystrom_attention_reference_simple(q2, k2, v2, m).sum().backward()

            def cos(a, b):
                return torch.nn.functional.cosine_similarity(
                    a.float().flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

            dq_c = cos(q.grad.cpu(), q2.grad)
            dk_c = cos(k.grad.cpu(), k2.grad)
            dv_c = cos(v.grad.cpu(), v2.grad)

            label = f"D={D}, m={m}, N={N}"
            row = {"config": label, "dQ_cos": dq_c, "dK_cos": dk_c, "dV_cos": dv_c}
            results.append(row)
            print(f"{label:>25} | {dq_c:>8.4f} | {dk_c:>8.4f} | {dv_c:>8.4f}")

            del q, k, v, q2, k2, v2
            torch.cuda.empty_cache()

    return results


def section_cuda_graphs(B, H, D, m, dtype, seq_lengths):
    """Table 5: Latency with CUDA graphs (replay) vs eager (per-call kernel launches).

    Captures fwd+bwd into a CUDA graph and replays it. This is what production code
    would use for steady-state inference/training to amortize kernel launch overhead
    across the captured graph. Especially relevant at small N where FlashNystrom's
    ~18 kernel launches per fwd+bwd dominate.
    """
    from flash_nystrom.flash_nystrom import FlashNystromFunction

    print("\n" + "=" * 100)
    print("TABLE 5: CUDA Graph Replay — FlashNystrom eager vs graph-captured fwd+bwd")
    print("=" * 100)
    print(f"{'N':>7} | {'FN eager':>10} | {'FN graph':>10} | {'Speedup':>8} | {'SDPA eager':>11} | {'SDPA graph':>11}")
    print("-" * 100)

    results = []
    for N in seq_lengths:
        torch.manual_seed(42)
        # Static input buffers (CUDA graphs require fixed addresses)
        q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
        grad_out = torch.randn(B, H, N, D, dtype=dtype, device="cuda")

        def fn_step():
            # Re-attach graph (clone produces a fresh leaf with grad)
            for p in (q, k, v):
                if p.grad is not None:
                    p.grad = None
            out = FlashNystromFunction.apply(q, k, v, m, 6, True, 5.0, True)
            out.backward(grad_out)

        # Warmup on the side stream as required by torch.cuda.graph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(11):
                fn_step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Eager baseline (no graph)
        fn_eager = benchmark_cuda(fn_step, warmup=3, repeat=30)

        # Capture
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn_step()
            fn_graph_t = benchmark_cuda(lambda: graph.replay(), warmup=3, repeat=30)
            fn_graph_ms = fn_graph_t["median_ms"]
            speedup = f"{fn_eager['median_ms'] / fn_graph_ms:.2f}x"
        except Exception as e:
            fn_graph_ms = float("nan")
            speedup = f"failed: {type(e).__name__}"

        # SDPA eager + graph
        def sdpa_step():
            for p in (q, k, v):
                if p.grad is not None:
                    p.grad = None
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            out.backward(grad_out)

        sdpa_eager_t = sdpa_graph_ms = float("nan")
        try:
            s2 = torch.cuda.Stream()
            s2.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s2):
                for _ in range(11):
                    sdpa_step()
            torch.cuda.current_stream().wait_stream(s2)
            torch.cuda.synchronize()
            sdpa_eager = benchmark_cuda(sdpa_step, warmup=3, repeat=30)
            sdpa_eager_t = sdpa_eager["median_ms"]
            sdpa_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(sdpa_graph):
                sdpa_step()
            sdpa_graph_t = benchmark_cuda(lambda: sdpa_graph.replay(), warmup=3, repeat=30)
            sdpa_graph_ms = sdpa_graph_t["median_ms"]
        except Exception:
            pass

        row = {
            "N": N,
            "fn_eager_ms": fn_eager["median_ms"],
            "fn_graph_ms": fn_graph_ms,
            "fn_graph_speedup": speedup,
            "sdpa_eager_ms": sdpa_eager_t,
            "sdpa_graph_ms": sdpa_graph_ms,
        }
        results.append(row)

        sf = lambda x: f"{x:.2f}" if x and not (isinstance(x, float) and x != x) else "N/A"
        print(f"{N:>7} | {fn_eager['median_ms']:>10.2f} | {sf(fn_graph_ms):>10} | {speedup:>8} | "
              f"{sf(sdpa_eager_t):>11} | {sf(sdpa_graph_ms):>11}")

        del q, k, v, grad_out
        try: del graph
        except: pass
        try: del sdpa_graph
        except: pass
        torch.cuda.empty_cache()

    return results


def section_training():
    """Table 4: CIFAR-10 training comparison — FlashNystrom vs SDPA attention."""
    from flash_nystrom import FlashNystromAttention, NystromConfig

    print("\n" + "=" * 90)
    print("TABLE 4: CIFAR-10 Training — FlashNystrom vs SDPA")
    print("=" * 90)

    # Tiny ViT model
    # dim=256, heads=4 -> head_dim=64 (FlashNystrom requires head_dim in {64, 128})
    class TinyViT(nn.Module):
        def __init__(self, attn_module, dim=256, depth=4, heads=4, patch_size=4, num_classes=10):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.pos_embed = nn.Parameter(torch.randn(1, (32 // patch_size) ** 2 + 1, dim) * 0.02)
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    "attn": attn_module(dim, heads),
                    "norm1": nn.LayerNorm(dim),
                    "ff": nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                    "norm2": nn.LayerNorm(dim),
                }) for _ in range(depth)
            ])
            self.head = nn.Linear(dim, num_classes)
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1) + self.pos_embed
            for blk in self.blocks:
                x = x + blk["attn"](blk["norm1"](x))
                x = x + blk["ff"](blk["norm2"](x))
            return self.head(self.norm(x[:, 0]))

    # SDPA attention module (same interface)
    class SDPAAttention(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.heads = heads
            self.head_dim = dim // heads
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.out_proj = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            B, N, _ = x.shape
            H, D = self.heads, self.head_dim
            q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)
            k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
            v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return self.out_proj(out.transpose(1, 2).contiguous().view(B, N, -1))

    def make_fn_attn(dim, heads):
        # Conv residual via cuDNN (F.conv1d). Custom CUDA conv kernels were
        # replaced after the previous run produced NaN under FP16 autocast.
        config = NystromConfig(num_landmarks=32, conv_kernel_size=3, use_conv_residual=True)
        return FlashNystromAttention(dim, heads, config)

    # training loop
    def train_model(model_fn_name, attn_factory, epochs=20, batch_size=128, lr=1e-3):
        import torchvision
        import torchvision.transforms as T

        transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
                               T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_test = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root=_DATA, train=True, download=False, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=_DATA, train=False, download=False, transform=transform_test)
        # drop_last=True keeps batch shape constant -> torch.compile (CUDA graphs) avoids recapture
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                   num_workers=0, pin_memory=True, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False,
                                                  num_workers=0, pin_memory=True, drop_last=True)

        torch.manual_seed(42)
        model = TinyViT(attn_factory, dim=256, depth=4, heads=4).cuda()
        # Note: torch.compile would require Triton (not installed on this Windows env)
        # and would fail to trace through our pybind11 C++ extension anyway.
        # Direct CUDA graph capture is benchmarked separately — see section_cuda_graphs().
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        nparams = sum(p.numel() for p in model.parameters())
        print(f"\n  {model_fn_name}: {nparams/1e6:.1f}M params")

        train_times = []
        for epoch in range(epochs):
            model.train()
            epoch_start = time.time()
            total_loss, correct, total = 0.0, 0, 0
            for imgs, labels in trainloader:
                imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * imgs.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += imgs.size(0)
            scheduler.step()
            epoch_time = time.time() - epoch_start
            train_times.append(epoch_time)
            train_acc = 100.0 * correct / total
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:>2}/{epochs}: loss={total_loss/total:.4f} "
                      f"train_acc={train_acc:.1f}% time={epoch_time:.1f}s")

        # eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(imgs)
                correct += (logits.argmax(1) == labels).sum().item()
                total += imgs.size(0)
        test_acc = 100.0 * correct / total
        avg_time = sum(train_times) / len(train_times)
        print(f"    Final test accuracy: {test_acc:.1f}%  avg epoch time: {avg_time:.1f}s")
        return {"model": model_fn_name, "test_acc": test_acc, "avg_epoch_s": avg_time,
                "params_M": nparams / 1e6, "epochs": epochs}

    results = []
    results.append(train_model("FlashNystrom", make_fn_attn))
    torch.cuda.empty_cache()
    results.append(train_model("SDPA", SDPAAttention))

    print("\n  Summary:")
    for r in results:
        print(f"    {r['model']:>14}: test_acc={r['test_acc']:.1f}%  "
              f"avg_epoch={r['avg_epoch_s']:.1f}s  params={r['params_M']:.1f}M")

    return results


def main():
    print("=" * 110)
    print("FlashNystrom Paper Benchmarks")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print("=" * 110)

    B, H, D, m = 1, 8, 128, 64
    dtype = torch.float16
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    all_results = {}

    # Table 1: latency
    all_results["latency"] = section_latency(B, H, D, m, dtype, seq_lengths)

    # Table 2: memory
    all_results["memory"] = section_memory(B, H, D, m, dtype, seq_lengths)

    # Table 3: correctness
    all_results["correctness"] = section_correctness([64, 128], dtype)

    # Table 4: training
    all_results["training"] = section_training()

    # save raw results
    out_path = "bench_paper_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
