# FlashNystrom

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/athrva98/FlashNystrom/blob/master/notebooks/quickstart.ipynb)

CUDA kernels for Nystromformer approximate attention. Forward and backward run in linear time and memory with respect to sequence length. The matmul-heavy stages use tensor cores. Backward gradients are exact against PyTorch autograd at FP32 numerical noise.

Open the Colab notebook above for a one-click install + smoke test + short latency demo. Switch the Colab runtime to L4 or A100 first; free-tier T4 (sm_75) is not supported.

The Nystromformer factorization is

```
attention(Q, K, V) = softmax(Q @ Kt^T) @ softmax(Qt @ Kt^T)^+ @ softmax(Qt @ K^T) @ V
```

where Qt and Kt are landmarks formed by segmented mean pooling of Q and K. The pseudoinverse is computed by unrolled Newton-Schulz iteration in FP32. The backward pass differentiates through every NS iterate via the chain rule. There is no Implicit Function Theorem dependence and no requirement that NS has converged.

## Scope

FlashNystrom is not a FlashAttention competitor. FlashAttention (v1/v2/v3/v4) implements *exact* O(N²) attention with IO-aware tiling. Its version bumps are hardware-targeted rewrites of the same algorithm: FA2 for Ampere and Ada, FA3 for Hopper WGMMA and TMA, FA4 for Blackwell TMEM. FlashNystrom implements a *different* attention math: the Nyström low-rank factorization, which is O(m·N·D + m³) with m landmarks. The relevant comparison is FlashNystrom against SDPA (using any FA generation under the hood) at long sequence length, where O(N²) starts to dominate and the approximation becomes worthwhile. At short N (under ~1–2K), exact attention is faster and you should use it.

The kernels borrow the FA2-era CUTLASS SM80 mma atom and the tiled-softmax with running-LSE pattern, but apply them to the three Nyström softmaxes rather than to one big QK^T. They use the SM80 idioms deliberately: no WGMMA, no TMA, no warp specialization, no TMEM. That choice keeps a **single binary that runs on every Ampere through Blackwell card** (the build covers `sm_80;86;89;90;100;120` — verified running on A100, H100, H200, B200, and the RTX 5060) — Ampere consumer and datacenter, Ada, Hopper, and Blackwell consumer and datacenter. WGMMA and TMA are Hopper-only, and TMEM is Blackwell-only, so adopting them would fragment the codebase into per-arch builds; the FA3/FA4 codebases pay that complexity to extract Hopper- and Blackwell-native peak throughput. FlashNystrom keeps the one-binary contract and benefits from the larger SMEM and register files on Hopper and Blackwell via occupancy. (On H200 and B200 these SM80 kernels run in *compatibility mode* against native-kernel cuBLAS; the planned per-generation atom port to WGMMA/TMEM — a port of the *same* recipe, exactly as FA2→FA3→FA4 ported exact attention — closes the resulting constant-factor gap at long context. See the datacenter benchmarks below.) See [the SMEM sizing discussion](#smem-sizing-and-occupancy) below.

## Status

20-epoch CIFAR-10 ViT (default settings, FP16 autocast, num_landmarks=32, newton_iter=6) reaches the same test accuracy as the SDPA and pure-PyTorch Nystromformer baselines:

| Config                          | test acc |
|---------------------------------|---------:|
| `F.scaled_dot_product_attention`|    66.7% |
| Pure-PyTorch Nystromformer      |    66.3% |
| FlashNystrom (this repo)        |    66.7% |

99 tests cover forward, backward, kernel-level isolation, the production cuBLAS + CUDA-graph NS backward path, per-kernel regression against autograd-derived references, the `m > 64` reference-dispatch path, and the `kappa_star` / `fast_dk2inv` precision contracts (kernel-vs-reference consistency and gradient unbiasedness).

## Install

```
git clone --recursive https://github.com/athrva98/FlashNystrom.git
cd FlashNystrom
pip install -e . --no-build-isolation
```

If you cloned without `--recursive`, pull the CUTLASS submodule first:

```
git submodule update --init
```

Requirements:

* PyTorch 2.0+ with CUDA support
* CUDA toolkit 12.2+
* Compute capability 8.0+ (Ampere, Ada, Hopper, Blackwell). The kernels deliberately use SM80 idioms (16x8x16 mma atom, `cp.async`, up to ~96 KB of dynamic shared memory per CTA) so a single binary covers every arch from Ampere through Blackwell. WGMMA and TMA are Hopper-only and TMEM is Blackwell-only; those would require per-arch kernel families. SM75 and earlier are not supported.

## Quickstart

Module form:

```python
import torch
from flash_nystrom import FlashNystromAttention, NystromConfig

# kappa_star is the Tikhonov ridge target condition number (default 5.0);
# it keeps the Newton-Schulz pseudoinverse well-conditioned as N grows.
# Set kappa_star=0.0 to disable the ridge (original Nystromformer formulation).
cfg = NystromConfig(num_landmarks=64, newton_iter=6, conv_kernel_size=3,
                    kappa_star=5.0)
attn = FlashNystromAttention(dim=512, heads=8, config=cfg).cuda()

x = torch.randn(4, 4096, 512, device="cuda", dtype=torch.float16)
y = attn(x)
y.sum().backward()
```

Functional form (raw Q, K, V):

```python
from flash_nystrom import flash_nystrom_attention

q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

out = flash_nystrom_attention(q, k, v, num_landmarks=64, newton_iter=6,
                              kappa_star=5.0)
```

## Latency

Forward and backward latency in milliseconds on an RTX 5060 Laptop (Blackwell consumer, 8 GB VRAM, sm_120), FP16, B=1, H=4, head_dim=64, num_landmarks=32, newton_iter=6, measured with the current default config (Tikhonov ridge `kappa_star=5`, tf32 tensor-core pseudoinverse, `fast_dk2inv`). CUDA-event timed, median of 30 fwd+bwd runs after 5 warmups; reduced rep counts at N ≥ 16384 to keep wall-clock manageable. Three implementations:

- **FN**: this repo (custom CUDA forward + cuBLAS-graphs backward).
- **Ref**: the same Nyström algorithm written in plain PyTorch. Each matmul dispatches to cuBLAS via the `@` operator, each softmax to `torch.softmax`, each elementwise op to a torch CUDA kernel. No fusion across stages: every op is a separate launch with HBM round-trips between them, and the three softmaxes are not folded into a single pass. See `flash_nystrom/reference.py`.
- **SDPA**: `F.scaled_dot_product_attention`, which on PyTorch 2.x dispatches to the memory-efficient attention backend (a FlashAttention-class kernel). Exact O(N²) attention.

| N      | FN fwd | FN bwd | FN tot | Ref tot | SDPA fwd | SDPA bwd | SDPA tot | FN/Ref | FN/SDPA | SDPA − FN (ms) |
|-------:|-------:|-------:|-------:|--------:|---------:|---------:|---------:|-------:|--------:|---------------:|
|    128 |   0.18 |   1.09 |   1.27 |    5.54 |     0.05 |     0.46 |     0.52 |  4.4x  |   0.41x |          −0.75 |
|    256 |   0.18 |   1.00 |   1.18 |    5.59 |     0.07 |     0.40 |     0.47 |  4.7x  |   0.40x |          −0.70 |
|    512 |   0.19 |   0.52 |   0.71 |    5.29 |     0.04 |     0.21 |     0.25 |  7.5x  |   0.36x |          −0.46 |
|   1024 |   0.20 |   0.51 |   0.70 |    5.01 |     0.10 |     0.31 |     0.41 |  7.1x  |   0.58x |          −0.29 |
|   2048 |   0.20 |   0.54 |   0.74 |    5.18 |     0.29 |     0.95 |     1.24 |  7.0x  |   1.7x  |          +0.49 |
|   4096 |   0.22 |   0.59 |   0.81 |    5.25 |     1.06 |     3.51 |     4.56 |  6.5x  |   5.6x  |          +3.75 |
|   8192 |   0.25 |   0.75 |   0.99 |    6.03 |     4.11 |    13.48 |    17.59 |  6.1x  |  17.7x  |         +16.60 |
|  16384 |   0.40 |   1.43 |   1.83 |    5.63 |    16.19 |    57.44 |    73.62 |  3.1x  |  40.3x  |         +71.80 |
|  32768 |   0.68 |   2.18 |   2.86 |    7.09 |    67.82 |   219.70 |   287.52 |  2.5x  |   100x  |           +285 |
|  65536 |   1.48 |   4.68 |   6.16 |   11.12 |   278.70 |   916.43 |  1195.12 |  1.8x  |   194x  |         +1,189 |
| 131072 |   2.79 |   8.97 |  11.76 |   21.46 |  1117.17 |  3768.95 |  4886.12 |  1.8x  |   415x  |         +4,874 |
| 262144 |   5.40 |  15.19 |  20.59 |   56.81 |  4591.06 | 15291.11 | 19882.17 |  2.8x  |   966x  |        +19,862 |

![FlashNystrom vs cuBLAS-Nystrom vs SDPA fwd+bwd latency on an RTX 5060, log-log](assets/latency_5060.png)

The speedup columns are *base time / FN time*. Values > 1 mean FN is faster; values < 1 mean FN is slower than the base. The last column is the absolute time difference per fwd+bwd call (positive means FN is faster).

Reading the table:

- **The ratio compresses both ends. The absolute difference does not.** At N ≤ 1024 where SDPA wins, the loss is between 0.29 ms and 0.75 ms per call. That is below the noise floor of a typical training loop and well below any optimizer step. At N = 262144 where FN wins, the save is 19.9 seconds per fwd+bwd call. The ratio and the absolute column tell the same story but the absolute column is the one that matters for "does this make my training run actually finish."
- **At short N (≤ 1024), SDPA is faster than FN.** FN carries fixed overhead from its three softmaxes and the Newton-Schulz pseudoinverse. That overhead dominates while N² is still cheap. If your N stays under ~1 K, use SDPA.
- **The fwd+bwd crossover is between N = 1024 and N = 2048.** At N = 2048 FN is 1.7x faster than SDPA total. Above that point the gap widens monotonically.
- **Above N ≈ 8 K the speedup over SDPA grows roughly linearly with N**, as expected from FN's O(N) compute versus SDPA's O(N²). Doubling N from 16 K to 32 K roughly doubles the speedup (40x to 100x). Same at 32 K to 64 K (100x to 194x), 64 K to 128 K (194x to 415x), and 128 K to 256 K (415x to 966x).
- **FN beats Ref at every N tested.** Same algorithm; the gap is kernel fusion and GPU utilization. The FN/Ref ratio is largest at short N, where the reference pays fixed per-op launch overhead that FN folds into single kernels. It narrows to about 1.8x–2.5x in the mid-range (32 K–64 K) and holds at 1.8x to 2.8x out to N = 256 K, where the saving is HBM traffic and the multi-CTA split that keeps the GPU busy at this batch×head.
- **Neither method OOMs at N = 262144 on 8 GB.** SDPA's wall is wall-clock (~20 s per fwd+bwd at N = 256 K), not memory. PyTorch's SDPA uses memory-efficient attention internally, so it scales linearly in memory; the O(N²) compute is what makes it unusable past 32 K or so in practice.

Reproduce with `python benchmarks/bench_5060_refresh.py`.

## Datacenter GPUs: same algorithm, FlashNystrom vs cuBLAS

The 5060 table is FlashNystrom against *exact* attention. This one isolates kernel quality: FlashNystrom against the **same Nyström algorithm** in plain PyTorch (the `Ref` above, where every matmul is a cuBLAS call and every softmax a torch kernel, with no fusion across stages). Same math, same FLOPs; the only difference is the kernels. FP16, newton_iter=6. `f x` and `tot x` are cuBLAS_time / FN_time; values > 1 mean FN is faster.

**A100-80GB.** High batch×head (B=4, H=16, head_dim=128, m=64):

| N      | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|-------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   4096 |   1.96 |       1.87 | 0.95x |   6.97 |       7.36 | 1.06x |
|  16384 |   3.09 |       3.12 | 1.01x |  16.63 |      22.01 | 1.32x |
|  65536 |   9.00 |      10.62 | 1.18x |  57.58 |      82.42 | 1.43x |
| 131072 |  16.96 |      20.59 | 1.21x | 108.96 |     193.02 | 1.77x |

A100, long context, few heads (B=1, H=4, head_dim=64, m=32):

| N       | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|--------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   65536 |   0.81 |       1.86 | 2.29x |   4.71 |       6.60 | 1.40x |
|  131072 |   1.13 |       1.82 | 1.60x |   8.06 |       8.22 | 1.02x |
|  262144 |   1.80 |       2.77 | 1.53x |  14.52 |      17.90 | 1.23x |
|  524288 |   3.14 |       4.85 | 1.55x |  27.47 |      40.86 | 1.49x |
| 1048576 |   5.82 |       9.39 | 1.61x |  52.95 |      81.48 | 1.54x |
| 2097152 |  11.29 |      18.23 | 1.62x | 105.34 |     162.05 | 1.54x |

**H100-80GB.** High batch×head (B=4, H=16, head_dim=128, m=64):

| N      | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|-------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   4096 |   1.12 |       1.47 | 1.31x |   3.59 |       5.60 | 1.56x |
|  16384 |   1.96 |       1.77 | 0.90x |   8.56 |      13.04 | 1.52x |
|  65536 |   5.19 |       5.98 | 1.15x |  27.82 |      49.61 | 1.78x |
| 131072 |   9.53 |      11.64 | 1.22x |  53.56 |     101.88 | 1.90x |

H100, long context, few heads (B=1, H=4, head_dim=64, m=32):

| N       | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|--------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   65536 |   0.59 |       1.59 | 2.70x |   3.34 |       5.55 | 1.66x |
|  131072 |   0.77 |       1.53 | 1.98x |   5.62 |       5.65 | 1.00x |
|  262144 |   1.17 |       1.60 | 1.37x |  10.38 |       8.77 | 0.84x |
|  524288 |   1.99 |       2.40 | 1.20x |  19.91 |      21.40 | 1.08x |
| 1048576 |   3.61 |       4.45 | 1.23x |  38.92 |      43.80 | 1.13x |
| 2097152 |   6.86 |       8.59 | 1.25x |  77.01 |      87.02 | 1.13x |

**On the newest cards this is not a like-for-like comparison.** FlashNystrom runs the *same SM80-atom kernels in compatibility mode* on H200 and B200, while the cuBLAS reference dispatches to **native Hopper/Blackwell GEMM kernels**. Where cuBLAS is faster at long context below, that is a not-yet-written native atom port (future work) measured against a native vendor kernel, not the Nyström method being slower. FlashNystrom still wins at high batch×head, where its whole-pipeline fusion outweighs the per-GEMM gap, and the asymptotic win over *exact* attention is unaffected (it is set by O(mN) vs O(N²), not by the atom generation).

**H200-141GB.** High batch×head (B=4, H=16, head_dim=128, m=64):

| N      | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|-------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   4096 |   1.09 |       1.93 | 1.78x |   3.43 |       7.06 | 2.06x |
|  16384 |   1.83 |       1.94 | 1.06x |   8.42 |      10.88 | 1.29x |
|  65536 |   4.66 |       4.51 | 0.97x |  27.67 |      40.75 | 1.47x |
| 131072 |   8.51 |       8.73 | 1.03x |  53.62 |      82.03 | 1.53x |

H200, long context (B=1, H=4, head_dim=64, m=32):

| N       | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|--------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   65536 |   0.57 |       1.98 | 3.47x |   3.19 |       7.08 | 2.22x |
|  131072 |   0.75 |       1.93 | 2.58x |   5.77 |       7.43 | 1.29x |
|  262144 |   1.13 |       1.98 | 1.76x |  10.77 |       7.58 | 0.70x |
|  524288 |   1.88 |       2.10 | 1.12x |  20.75 |      17.21 | 0.83x |
| 1048576 |   3.38 |       3.86 | 1.14x |  40.62 |      35.20 | 0.87x |
| 2097152 |   6.39 |       7.40 | 1.16x |  80.53 |      69.96 | 0.87x |

**B200 (Blackwell, sm_100).** High batch×head (B=4, H=16, head_dim=128, m=64):

| N      | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|-------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   4096 |   1.14 |       1.00 | 0.88x |   3.17 |       4.25 | 1.34x |
|  16384 |   1.69 |       1.24 | 0.73x |   6.71 |       8.35 | 1.24x |
|  65536 |   3.84 |       3.29 | 0.86x |  20.97 |      28.94 | 1.38x |
| 131072 |   6.67 |       6.09 | 0.91x |  39.98 |      56.62 | 1.42x |

B200, long context (B=1, H=4, head_dim=64, m=32):

| N       | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|--------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   65536 |   0.57 |       1.11 | 1.96x |   3.31 |       4.39 | 1.32x |
|  131072 |   0.71 |       1.09 | 1.53x |   5.84 |       3.95 | 0.68x |
|  262144 |   0.96 |       1.17 | 1.21x |  10.75 |       5.92 | 0.55x |
|  524288 |   1.45 |       1.92 | 1.32x |  20.40 |      10.78 | 0.53x |
| 1048576 |   2.44 |       3.38 | 1.38x |  39.92 |      28.84 | 0.72x |
| 2097152 |   4.41 |       6.30 | 1.43x |  79.83 |      58.72 | 0.74x |

![FlashNystrom vs cuBLAS-Nystrom fwd+bwd latency on A100 and H100, log-log](assets/latency_datacenter_cublas.png)

Reading the tables:

- **The forward wins at low batch×head on every card** (1.1x–3.5x on A100/H100/H200; 1.2x–2.0x on B200). This is the regime the parallelized landmark kernel fixed: a single landmark's segment of N/m rows used to be summed by one thread serially (latency-bound at large N); splitting that reduction across threads made it bandwidth-bound, and the fused GEMMs already saved HBM traffic vs cuBLAS.
- **End-to-end, FlashNystrom wins across the whole range on A100 and H100** (A100 total 1.00x–1.77x, H100 1.00x–1.90x), where its SM80 atoms are the native path (Ampere) or run well in Hopper compatibility. The one dip is H100 long-context N=262144 at 0.84x (a fast cuBLAS backward; the forward there is still a win).
- **On H200 and B200, FN wins at high batch×head** (H200 up to 2.06x, B200 1.24x–1.42x total) and on the low-batch forward (1.2x–2.0x). At long context the native-kernel cuBLAS is faster by a constant factor (H200 0.70x–0.87x for N ≥ 262K, B200 0.53x–0.74x for N ≥ 131K). Per the note above, that is FN's SM80-compatibility build measured against native Hopper/Blackwell GEMMs — the missing per-generation atom port, not the method being slower.
- **This is an atom update, not an algorithm change — and it is on the roadmap.** FlashNystrom is a *recipe* for Nyströmformer kernels: the kernel structure and the math are generation-invariant; only the mma/copy atom changes per generation (SM80 mma + `cp.async` → Hopper WGMMA/TMA → Blackwell TMEM), exactly as FlashAttention-2 → 3 → 4 are atom/arch ports of the *same* attention math. The repo ships the SM80-atom recipe, which by design runs on every Ampere-through-Blackwell card from one binary; the Blackwell-native atom port (future work) closes the constant-factor gap to native cuBLAS at long context, the same kind of port FA4 was for exact attention. The O(mN)-vs-O(N²) advantage over *exact* attention (the FlashAttention tables below) is asymptotic and independent of the atom generation.

Reproduce with `modal run tools/modal_a100.py::bench_gaps` (A100), `::bench_gaps_h100`, `::bench_gaps_h200`, or `::bench_gaps_b200`. Requires a Modal account and a one-time `modal setup`.

## FlashAttention-2 / FlashAttention-3 (exact attention), H100

The tables above compare FlashNystrom to the *same* Nyström algorithm in cuBLAS. This one compares it to the alternative people actually reach for: **exact** attention via FlashAttention. FA2 and FA3 compute exact O(N²) attention (FA3 is the Hopper-native current SOTA); FlashNystrom computes approximate O(m·N) Nyström. They are not the same computation, so this is a speed comparison that only matters where the Nyström approximation is acceptable (it is for the CIFAR-10 ViT, which matches the exact-attention baseline accuracy). H100-80GB, FP16, fwd+bwd, newton_iter=6. `FA2/FN` and `FA3/FN` are FA_total / FN_total; > 1 means FlashNystrom is faster. FA3 was built with its cluster and hdim-64/128 kernels intact (only genuinely unused variants trimmed), so these are its best kernels for these shapes.

High batch×head (B=4, H=16, head_dim=128, m=64):

| N      | FN tot | FA2 tot | FA3 tot | FA2/FN | FA3/FN |
|-------:|-------:|--------:|--------:|-------:|-------:|
|   4096 |   3.61 |    5.90 |    3.47 |  1.6x  |  1.0x  |
|  16384 |   8.62 |   91.5  |   50.4  | 10.6x  |  5.8x  |
|  65536 |  28.0  | 1469    |  835    | 52.4x  | 29.8x  |
| 131072 |  53.9  | 5865    | 3395    |  109x  | 63.0x  |

Long context, few heads (B=1, H=4, head_dim=64, m=32):

| N       | FN tot | FA2 tot | FA3 tot | FA2/FN | FA3/FN |
|--------:|-------:|--------:|--------:|-------:|-------:|
|   16384 |   1.46 |    3.08 |    1.74 |  2.1x  |  1.2x  |
|   65536 |   3.22 |   48.9  |   32.2  | 15.2x  | 10.0x  |
|  131072 |   5.89 |  202    |  123    | 34.3x  | 20.8x  |
|  262144 |  10.7  |  806    |  478    | 75.2x  | 44.7x  |
|  524288 |  20.6  | 3295    | 1958    |  160x  | 95.2x  |
| 1048576 |  39.9  | 13338   | 7865    |  334x  |  197x  |
| 2097152 |  77.2  | n/r     | n/r     |   -    |   -    |

![FlashNystrom (approx O(mN)) vs FlashAttention-2/3 (exact O(N^2)) fwd+bwd latency on H100, log-log](assets/latency_flashattention_h100.png)

Reading the tables:

- **At short N, use exact attention.** At N=4096 (high batch×head) FA3 is roughly tied with FN (1.0x), and the two are close in long context at N=16384 (1.2x). Exact attention is cheap when N² is small and carries no approximation error. The crossover is roughly N=4K to 16K.
- **Past the crossover the O(N²) wall takes over.** FlashNystrom's O(m·N) cost grows linearly while exact attention grows quadratically, so the gap widens fast: 5.8x at 16K, 30x at 65K, 63x at 131K (high batch×head); and in long context from 21x at 131K up to **~197x at 1M tokens** versus FA3.
- **Exact attention eventually stops being practical.** At N=1M, FA2 is already 13 s per fwd+bwd call (FA3 ~8 s) and climbing quadratically; at 2M tokens (`n/r`) we no longer run it, while FlashNystrom finishes the full fwd+bwd in 77 ms.
- **FA3 is ~1.7x faster than FA2** here (Hopper-native kernels), so it is the right exact-attention baseline. FlashNystrom still pulls away from FA3 at long N.

Built and measured with `modal run tools/modal_a100.py::bench_fa_h100` (installs FA2 plus a trimmed FA3 Hopper build, then benchmarks).

### Exact attention on Blackwell (B200): FA2 measured, FA4 estimated

We can now run on a B200 (added to the Modal harness). FlashAttention-2 (exact O(N²)) builds and runs there, so the asymptotic comparison is **measured directly on Blackwell** — FN (the SM80-atom recipe, in compatibility mode) vs FA2, long context (B=1, H=4, head_dim=64, m=32):

| N       | FN tot | FA2 tot  | FA2/FN |
|--------:|-------:|---------:|-------:|
|   16384 |   1.34 |     3.10 |  2.3x  |
|   65536 |   3.38 |    42.81 | 12.7x  |
|  131072 |   5.86 |   171.12 | 29.2x  |
|  262144 |  10.78 |   685.24 | 63.6x  |
|  524288 |  20.30 |  2717.13 |  134x  |
| 1048576 |  39.76 | 10884.08 |  274x  |
| 2097152 |  78.94 |    n/r   |   -    |

Even on Blackwell, against exact attention, the O(N²) wall is the same shape: FN is **274x faster at 1M tokens** (FA2 OOMs at 2M; FN finishes in 79 ms).

**FA4 specifically is not yet measured.** flash-attn-4 is the Blackwell/Hopper-native exact kernel and the right constant-factor baseline there, but it ships only beta wheels (`4.0.0bN`) whose `nvidia-cutlass-dsl` dependency is currently unsatisfiable on the package index: the newer cutlass-dsl removed `cute.core.ThrMma` (import error), the older one removed `cutlass.utils.ampere_helpers` (different import error), and b19 needs an intermediate snapshot that is not published (cf. flash-attention issues [#2310](https://github.com/Dao-AILab/flash-attention/issues/2310), [#2334](https://github.com/Dao-AILab/flash-attention/issues/2334)). The harness is in place (`modal run tools/modal_a100.py::bench_fa4_b200`) and will produce numbers once FA4's packaging stabilizes.

Until then the estimate below projects FA4 from published throughput. FA4 is faster than FA2, so the true FN-vs-FA4 ratios sit *below* the measured FN-vs-FA2-on-B200 numbers above and *above* the derived floor here. We bridge through published peak attention throughput. FA4 reports **~1605 TFLOP/s** (BF16, 71% utilization) on B200; FA3 reports **~740 TFLOP/s** (FP16, 75% utilization) on H100. BF16 and FP16 run at the same tensor-core rate, so for compute-bound attention FA4-on-B200 is about **2.2x faster than FA3-on-H100** (1605 / 740). Dividing our *measured* FN-vs-FA3 ratios by that factor:

`FN/FA4 (derived)  ≈  (FN/FA3 measured on H100)  /  2.2`

Long context (B=1, H=4, head_dim=64, m=32):

| N       | FA3/FN (measured, H100) | FA4/FN (derived) |
|--------:|------------------------:|-----------------:|
|   16384 |                    1.2x |            ~0.5x |
|   65536 |                   10.0x |            ~4.5x |
|  131072 |                   20.8x |            ~9.5x |
|  262144 |                   44.7x |             ~20x |
|  524288 |                   95.2x |             ~43x |
| 1048576 |                    197x |             ~90x |

(At high batch×head the same division applies: the measured 63x vs FA3 at N=131072 becomes ~29x vs FA4.)

These numbers are *derived from published throughput, not measured.* They also handicap FlashNystrom on purpose: FN runs on H100, FA4 on its native B200, and the 2.2x bridge hands FA4 the entire B200-plus-next-gen-kernel improvement, so these ratios are a **floor** on FN's advantage. On equal hardware FN would look better, not worse. The throughput proxy is fair in the long-N compute-bound regime where this comparison matters (at short N exact attention wins anyway and is the right choice), and it uses forward throughput while the table is fwd+bwd.

The point holds: at long context FN's O(m·N) is far enough ahead that a ~2.2x faster exact kernel on a newer GPU is still tens of times slower at N ≥ 128K. FA4 moves the crossover out (roughly to N = 16K to 32K); it does not remove it.

Sources: FlashAttention-4 (Colfax Research / Together AI, arXiv:2603.05451); FlashAttention-3 (Shah et al., 2024).

## SMEM sizing and occupancy

The kernels are sized for the consumer SMEM envelope (~100 KB/SM on Ampere
consumer, Ada, and Blackwell consumer). The build does not auto-tune
tile sizes to the runtime device; the choice is fixed at compile time.

Per-kernel SMEM usage (probe output on an RTX 5060 Laptop, 100 KB/SM,
m=64, D=128, FP16, niter=6):

| Kernel                        | Dyn SMEM (KB) | Regs/thr | Blocks/SM (consumer) | Binding constraint |
|-------------------------------|--------------:|---------:|---------------------:|--------------------|
| `landmark_kernel` (fwd)       |           8   |      40  |                  1   | threads (1024/blk) |
| `kernel1_fused_tc` (fwd)      |          32   |      71  |                  3   | SMEM               |
| `kernel3_fused_tc` (fwd)      |          32   |     165  |                  3   | registers (= SMEM) |
| `kernel1_bwd_tc`              |          48   |     159  |                  2   | SMEM               |
| `kernel3_bwd_tc`              |          40   |     171  |                  2   | registers (= SMEM) |
| `compute_dk2inv_tc`           |          64   |     206  |                  1   | SMEM               |
| `kernel2_inv` (NS forward)    |          96   |      42  |                  1   | SMEM               |
| `ns_bwd_step`                 |          96   |      40  |                  1   | SMEM               |

Reproduce with `python tools/kernel_report.py`. (`landmark_kernel` is
threads-bound, not occupancy-starved: one 1024-thread block is 32 warps, and
it is bandwidth-bound after the segment-reduction parallelization.)

![Per-kernel dynamic SMEM and blocks/SM on an RTX 5060, colored by binding constraint](assets/occupancy_smem.png)

**Are we leaving performance on the table on bigger-SMEM GPUs?**

Yes and no, and not in the way most people assume.

What we get for free on bigger SMEM (H100 has 228 KB/SM, ~2.3× consumer):
- Occupancy scales automatically, because **most kernels are SMEM-bound**
  on the consumer card (see the Binding column). `kernel2_inv` and
  `ns_bwd_step` (96 KB, 1 block/SM at 100 KB) go to 2 blocks/SM. The 40 to
  64 KB kernels (`kernel3_bwd_tc`, `kernel1_bwd_tc`, `compute_dk2inv_tc`)
  each gain blocks/SM until their register count becomes the binder, e.g.
  `compute_dk2inv_tc` (64 KB) goes from 1 block/SM to its ~2-block register
  ceiling. So bigger SMEM does help these.
- The one kernel bigger SMEM does **not** help is the forward
  `kernel3_fused_tc`: registers and SMEM both allow only 3 blocks/SM at
  128 threads/block (165 regs/thr), so it is already at its register
  ceiling and extra SMEM changes nothing. A win there needs fewer
  registers (smaller accumulator fragments, recomputation), not more SMEM.

What we miss by not sizing for big SMEM:
- We do not multi-stage. Each kernel uses one SMEM buffer per role
  (sQ, sK, sV); the next tile cannot be prefetched while the current
  tile computes. FA2 uses a 2-stage `cp.async` pipeline on Ampere; FA3
  uses TMA-driven asynchronous loads with producer/consumer warp
  specialization on Hopper. Both trade SMEM for memory-latency hiding.
  Adding a second stage to our K/V buffer would roughly double its
  SMEM cost and is only a clear win where memory latency dominates
  compute, which is exactly the regime that benefits from bigger SMEM.
- We do not opt into the Hopper 228 KB envelope. The
  `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, ...)` calls request
  the kernel's compile-time SMEM size, not the device max. On Hopper a
  multi-stage rewrite could push tiles to 128 KB+ and use TMA bulk
  copies. That is an FA3-class engineering effort.

The TL;DR: for the kernels that *are* SMEM-bound, bigger SMEM helps via
occupancy automatically. For the kernels that are register- or
compute-bound, more SMEM does nothing. The structural win we leave on
the table is async multi-stage pipelining, which is a non-trivial
rewrite and is also the rewrite that would unlock FA3/FA4-style
hardware-native idioms. They are the same project.

## PyTorch compatibility

`FlashNystromAttention` is a regular `nn.Module` and `flash_nystrom_attention` is a regular function. Standard PyTorch idioms work without changes.

| Workflow                                  | Status |
|-------------------------------------------|--------|
| Eager forward + backward                  | works |
| FP16 / BF16 / FP32 input dtypes           | works |
| `torch.amp.autocast("cuda", dtype=...)`   | works |
| `nn.Module` composition, `state_dict`     | works |
| DDP / FSDP gradient sync                  | works (gradients flow through standard autograd; no custom collective is needed) |
| `torch.compile`                           | runs, with a graph break at the FlashNystrom forward call. The kernel itself executes normally, but Dynamo cannot fuse across the boundary. A `torch.library.custom_op` registration would eliminate the graph break and is the natural follow-up if `torch.compile` integration matters to you. |
| `torch.jit.script`                        | not supported. Custom autograd Functions are not scriptable. |
| `torch.export`                            | not currently supported. Depends on the `custom_op` registration above. |

Typical training loop with autocast (matches the CIFAR-10 example):

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for x, y in loader:
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(x.cuda())
        loss = F.cross_entropy(logits, y.cuda())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

## Configuration

`NystromConfig` fields:

| Field              | Default | Notes |
|--------------------|--------:|-------|
| `num_landmarks`    |      64 | Custom kernels handle `m <= 64`; `m > 64` falls back to a pure-PyTorch reference (see Limitations). |
| `newton_iter`      |       6 | NS iterations for the pseudoinverse. Backward correctness is independent of convergence. |
| `conv_kernel_size` |       3 | Depthwise conv1d residual on V. Set to 0 to disable. |
| `use_conv_residual`|    True | Master switch for the conv residual. |
| `kappa_star`       |     5.0 | Tikhonov ridge target condition number: the pinv inverts `M = K2ᵀK2 + λI` with `λ = (‖K2‖₁‖K2‖∞)/kappa_star`, so `cond(M) ≤ kappa_star`. Keeps Newton-Schulz well-conditioned as `cond(K2)` grows with N. `0.0` disables the ridge (raw-K2 pinv). |
| `use_tc_pinv`      |    True | Route the pseudoinverse through the tf32 tensor-core NS chain (faster; floor ~6e-4 vs the fp16 reference's ~1.2e-3). `m == 64` only; the fp32 scalar kernel is used otherwise. |
| `fast_dk2inv`      |    True | Tensor-core `compute_dk2inv` in the backward (fp16/bf16 only). Casts the softmax output P to 16-bit before GEMM2 — verified zero-mean unbiased vs the exact fp32 path. Set `False` for the fp32 scalar fallback. |

## Limitations

* `head_dim` is restricted to 64 or 128.
* `num_landmarks` (m):
  * `m <= 64` runs on the custom CUDA kernels (forward + backward). This is the regime the latency tables above were measured in.
  * `m > 64` is supported via dispatch to the pure-PyTorch reference (`flash_nystrom.reference.nystrom_attention_reference`) — mathematically the same algorithm, each matmul lowering to cuBLAS via `@`, with autograd handling the backward. The reference materializes the two `(B, H, N, m)` softmax matrices and runs slower than the custom path; the Python wrapper raises a clear `RuntimeError` before allocation when those matrices would exceed the memory budget (8 GiB default, configurable via `FLASH_NYSTROM_REFERENCE_MAX_BYTES`). Custom `m > 64` kernels are being added one at a time; this dispatch shrinks as each lands.
* FP32 at `head_dim=128` (forward and backward) needs ~150 KB of opt-in shared memory, so it runs only on datacenter GPUs (A100 164 KB, H100/B200 228 KB) and raises a clear `insufficient smem` error on consumer cards (~100 KB). It is a gradient-checking / verification path, not a performance path — use FP16 or BF16 for D=128 in production.
* Sequence length must be at least `num_landmarks`.
* Compute capability 8.0 or newer.

## Repository layout

```
csrc/                          CUDA source
  flash_nystrom.cu             pybind entry points
  flash_nystrom_kernels.cu     kernel orchestration
  kernels/                     forward kernels
  kernels/backward/            backward kernels and isolation hooks
flash_nystrom/                 Python package (autograd Function, config, reference)
tests/                         95 pytest tests
benchmarks/                    latency and CIFAR-10 training scripts
examples/                      end-to-end usage examples
notebooks/                     Colab quickstart
third_party/cutlass/           CUTLASS submodule
```

## Tests

```
pytest tests/
```

`tests/test_ns_bwd_kernel.py` contains element-wise isolation tests for every backward kernel, with the FP32 reference computed in PyTorch from the same algebra the CUDA kernel implements. The kernels are pinned to FP32 noise across `newton_iter` in {1, 2, 3, 6, 10, 15, 20} and across sequence lengths that exercise both tile-aligned and partial-tile code paths.

## References

* Xiong, Zeng, Chakraborty, Tan, Fung, Li, Singh. *Nystromformer: A Nystrom-based Algorithm for Approximating Self-Attention*. AAAI 2021.
* Dao, Fu, Ermon, Rudra, Re. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.
* Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. ICLR 2024.
* Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*. NeurIPS 2024.
* Colfax Research / Together AI. *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*. arXiv:2603.05451, 2026. (Used only for the indirect FA4 throughput estimate; see the latency section.)

The kernel layouts, the tiled-softmax running-LSE state machine, and the
CUTE SmemLayoutAtomQ/KV patterns are adapted from FlashAttention-2. We
intentionally stay on the FA2-era SM80 instruction set rather than
adopting FA3-style asynchrony (WGMMA + TMA + warp specialization): those
primitives are Hopper-only and would force a per-arch kernel split, and
FlashNystrom's sm_80 through sm_90 single-binary contract is worth more
to its users than the Hopper-only peak-throughput uplift would be.
FlashAttention solves exact O(N²) attention; FlashNystrom uses these
techniques to implement the Nyström low-rank factorization instead.

## License

Apache License 2.0. See `LICENSE`.

## Author

Athrva Pandhare. athrva98@gmail.com.

## AI assistance

Claude (Anthropic) was used as a coding aid, mostly for CUTLASS / CuTe
device-API syntax. The kernel designs and the algorithm are my own.
