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

The kernels borrow the FA2-era CUTLASS SM80 mma atom and the tiled-softmax with running-LSE pattern, but apply them to the three Nyström softmaxes rather than to one big QK^T. They use the SM80 idioms deliberately: no WGMMA, no TMA, no warp specialization, no TMEM. That choice keeps a **single binary that runs on every Ampere through Blackwell card** (the build covers `sm_80;86;89;90;100;120` — verified running on A100, H100, H200, B200, and the RTX 5060) — Ampere consumer and datacenter, Ada, Hopper, and Blackwell consumer and datacenter. WGMMA and TMA are Hopper-only, and TMEM is Blackwell-only, so adopting them would fragment the codebase into per-arch builds; the FA3/FA4 codebases pay that complexity to extract Hopper- and Blackwell-native peak throughput. FlashNystrom keeps the one-binary contract and benefits from the larger SMEM and register files on Hopper and Blackwell via occupancy. (On H200 and B200 these SM80 kernels run in *compatibility mode* against native-kernel cuBLAS and are still faster at every measured size; the planned per-generation atom port to WGMMA/TMEM — a port of the *same* recipe, exactly as FA2→FA3→FA4 ported exact attention — converts the remaining headroom into further wins. See the datacenter benchmarks below.) See [the SMEM sizing discussion](#smem-sizing-and-occupancy) below.

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
|    128 |   0.18 |   0.94 |   1.12 |    5.04 |     0.10 |     0.29 |     0.39 |  4.5x  |   0.35x |          −0.73 |
|    256 |   0.18 |   0.56 |   0.74 |    5.01 |     0.02 |     0.24 |     0.27 |  6.8x  |   0.36x |          −0.47 |
|    512 |   0.19 |   0.56 |   0.75 |    5.23 |     0.04 |     0.19 |     0.23 |  7.0x  |   0.30x |          −0.52 |
|   1024 |   0.20 |   0.54 |   0.74 |    5.21 |     0.10 |     0.31 |     0.41 |  7.1x  |   0.56x |          −0.33 |
|   2048 |   0.20 |   0.55 |   0.75 |    5.41 |     0.29 |     0.95 |     1.24 |  7.2x  |   1.7x  |          +0.49 |
|   4096 |   0.22 |   0.57 |   0.79 |    4.49 |     1.06 |     3.53 |     4.59 |  5.7x  |   5.8x  |          +3.80 |
|   8192 |   0.25 |   0.69 |   0.94 |    4.98 |     4.15 |    13.55 |    17.70 |  5.3x  |  18.8x  |         +16.76 |
|  16384 |   0.38 |   0.98 |   1.36 |    4.78 |    16.35 |    57.50 |    73.86 |  3.5x  |  54.4x  |         +72.50 |
|  32768 |   0.67 |   1.70 |   2.37 |    6.79 |    68.03 |   223.56 |   291.59 |  2.9x  |   123x  |           +289 |
|  65536 |   1.42 |   3.61 |   5.04 |   11.31 |   277.01 |   924.55 |  1201.56 |  2.2x  |   239x  |         +1,197 |
| 131072 |   2.72 |   6.88 |   9.59 |   21.69 |  1126.56 |  3978.38 |  5104.95 |  2.3x  |   532x  |         +5,095 |
| 262144 |   5.33 |  12.56 |  17.89 |   57.11 |  4635.00 | 15059.08 | 19694.07 |  3.2x  |  1101x  |        +19,676 |

![FlashNystrom vs cuBLAS-Nystrom vs SDPA fwd+bwd latency on an RTX 5060, log-log](assets/latency_5060.png)

The speedup columns are *base time / FN time*. Values > 1 mean FN is faster; values < 1 mean FN is slower than the base. The last column is the absolute time difference per fwd+bwd call (positive means FN is faster).

Reading the table:

- **The ratio compresses both ends. The absolute difference does not.** At N ≤ 1024 where SDPA wins, the loss is between 0.29 ms and 0.75 ms per call. That is below the noise floor of a typical training loop and well below any optimizer step. At N = 262144 where FN wins, the save is 19.7 seconds per fwd+bwd call. The ratio and the absolute column tell the same story but the absolute column is the one that matters for "does this make my training run actually finish."
- **At short N (≤ 1024), SDPA is faster than FN.** FN carries fixed overhead from its three softmaxes and the Newton-Schulz pseudoinverse. That overhead dominates while N² is still cheap. If your N stays under ~1 K, use SDPA.
- **The fwd+bwd crossover is between N = 1024 and N = 2048.** At N = 2048 FN is 1.7x faster than SDPA total. Above that point the gap widens monotonically.
- **Above N ≈ 8 K the speedup over SDPA grows roughly linearly with N**, as expected from FN's O(N) compute versus SDPA's O(N²). Doubling N from 16 K to 32 K roughly doubles the speedup (54x to 123x). Same at 32 K to 64 K (123x to 239x), 64 K to 128 K (239x to 532x), and 128 K to 256 K (532x to 1101x).
- **FN beats Ref at every N tested.** Same algorithm; the gap is kernel fusion and GPU utilization. The FN/Ref ratio is largest at short N, where the reference pays fixed per-op launch overhead that FN folds into single kernels. It narrows to about 2.2x–2.9x in the mid-range (32 K–64 K) and holds at 2.2x to 3.2x out to N = 256 K, where the saving is HBM traffic and the multi-CTA split that keeps the GPU busy at this batch×head.
- **Neither method OOMs at N = 262144 on 8 GB.** SDPA's wall is wall-clock (~20 s per fwd+bwd at N = 256 K), not memory. PyTorch's SDPA uses memory-efficient attention internally, so it scales linearly in memory; the O(N²) compute is what makes it unusable past 32 K or so in practice.

Reproduce with `python benchmarks/bench_5060_refresh.py`.

## Datacenter GPUs: same algorithm, FlashNystrom vs cuBLAS

The 5060 table is FlashNystrom against *exact* attention. This one isolates kernel quality: FlashNystrom against the **same Nyström algorithm** in plain PyTorch (the `Ref` above, where every matmul is a cuBLAS call and every softmax a torch kernel, with no fusion across stages). Same math, same FLOPs; the only difference is the kernels. FP16, newton_iter=6. `f x` and `tot x` are cuBLAS_time / FN_time; values > 1 mean FN is faster.

**A100-80GB.** High batch×head (B=4, H=16, head_dim=128, m=64):

| N      | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|-------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   4096 |   1.72 |       1.47 | 0.86x |   6.08 |       7.44 | 1.23x |
|  16384 |   3.11 |       3.19 | 1.03x |  13.64 |      22.25 | 1.63x |
|  65536 |   9.01 |      10.96 | 1.22x |  44.25 |      83.85 | 1.89x |
| 131072 |  17.01 |      21.54 | 1.27x |  85.96 |     193.96 | 2.26x |

A100, long context, few heads (B=1, H=4, head_dim=64, m=32):

| N       | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|--------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   65536 |   0.80 |       1.40 | 1.76x |   2.55 |       5.26 | 2.06x |
|  131072 |   1.12 |       1.68 | 1.50x |   3.75 |       8.29 | 2.21x |
|  262144 |   1.77 |       2.78 | 1.57x |   6.15 |      17.87 | 2.91x |
|  524288 |   3.06 |       4.96 | 1.62x |  10.88 |      40.74 | 3.75x |
| 1048576 |   5.67 |       9.34 | 1.65x |  20.21 |      81.27 | 4.02x |
| 2097152 |  11.01 |      18.30 | 1.66x |  39.83 |     161.07 | 4.04x |

**H100-80GB.** High batch×head (B=4, H=16, head_dim=128, m=64):

| N      | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|-------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   4096 |   1.10 |       1.15 | 1.04x |   3.47 |       4.15 | 1.19x |
|  16384 |   1.88 |       1.75 | 0.93x |   7.93 |      12.97 | 1.63x |
|  65536 |   4.89 |       5.89 | 1.20x |  25.19 |      49.32 | 1.96x |
| 131072 |   8.92 |      11.43 | 1.28x |  48.03 |     101.23 | 2.11x |

H100, long context, few heads (B=1, H=4, head_dim=64, m=32):

| N       | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|--------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   65536 |   0.57 |       1.02 | 1.78x |   1.56 |       3.75 | 2.41x |
|  131072 |   0.74 |       1.03 | 1.38x |   2.28 |       4.52 | 1.99x |
|  262144 |   1.11 |       1.38 | 1.24x |   3.70 |       8.77 | 2.37x |
|  524288 |   1.88 |       2.39 | 1.27x |   6.58 |      21.23 | 3.23x |
| 1048576 |   3.38 |       4.41 | 1.31x |  12.25 |      43.46 | 3.55x |
| 2097152 |   6.41 |       8.50 | 1.33x |  23.55 |      86.43 | 3.67x |

**On the newest cards this is not a like-for-like comparison — and FlashNystrom wins anyway.** FlashNystrom runs the *same SM80-atom kernels in compatibility mode* on H200 and B200, while the cuBLAS reference dispatches to **native Hopper/Blackwell GEMM kernels**. FlashNystrom is still faster at every measured size on both cards: whole-pipeline fusion, on-chip intermediates, and roofline-parallel gradient scatters outweigh the per-GEMM atom-generation gap. The planned native atom port (future work) is additional headroom, not a prerequisite.

**H200-141GB.** High batch×head (B=4, H=16, head_dim=128, m=64):

| N      | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|-------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   4096 |   1.06 |       1.18 | 1.11x |   3.27 |       4.58 | 1.40x |
|  16384 |   1.74 |       1.45 | 0.83x |   7.34 |      10.84 | 1.48x |
|  65536 |   4.29 |       4.52 | 1.05x |  23.36 |      40.67 | 1.74x |
| 131072 |   7.80 |       8.75 | 1.12x |  44.55 |      81.88 | 1.84x |

H200, long context (B=1, H=4, head_dim=64, m=32):

| N       | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|--------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   65536 |   0.56 |       1.21 | 2.18x |   1.50 |       4.61 | 3.07x |
|  131072 |   0.71 |       1.19 | 1.67x |   2.21 |       4.56 | 2.06x |
|  262144 |   1.05 |       1.24 | 1.19x |   3.59 |       7.52 | 2.09x |
|  524288 |   1.72 |       2.11 | 1.22x |   6.32 |      17.53 | 2.77x |
| 1048576 |   3.07 |       3.85 | 1.25x |  11.81 |      35.83 | 3.03x |
| 2097152 |   5.80 |       7.40 | 1.28x |  22.83 |      71.25 | 3.12x |

**B200 (Blackwell, sm_100).** High batch×head (B=4, H=16, head_dim=128, m=64):

| N      | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|-------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   4096 |   1.12 |       0.71 | 0.64x |   3.05 |       3.14 | 1.03x |
|  16384 |   1.62 |       1.24 | 0.76x |   6.15 |       8.30 | 1.35x |
|  65536 |   3.57 |       3.28 | 0.92x |  18.32 |      28.81 | 1.57x |
| 131072 |   6.13 |       6.08 | 0.99x |  34.51 |      56.33 | 1.63x |

B200, long context (B=1, H=4, head_dim=64, m=32):

| N       | FN fwd | cuBLAS fwd | f x   | FN tot | cuBLAS tot | tot x |
|--------:|-------:|-----------:|------:|-------:|-----------:|------:|
|   65536 |   0.56 |       0.64 | 1.14x |   1.46 |       2.29 | 1.58x |
|  131072 |   0.69 |       0.80 | 1.16x |   2.00 |       3.48 | 1.74x |
|  262144 |   0.90 |       1.16 | 1.29x |   3.04 |       5.89 | 1.94x |
|  524288 |   1.32 |       1.91 | 1.44x |   5.11 |      10.71 | 2.10x |
| 1048576 |   2.15 |       3.37 | 1.57x |   9.18 |      28.65 | 3.12x |
| 2097152 |   3.83 |       6.28 | 1.64x |  17.36 |      58.48 | 3.37x |

![FlashNystrom vs cuBLAS-Nystrom fwd+bwd latency on A100 and H100, log-log](assets/latency_datacenter_cublas.png)

Reading the tables:

- **The forward wins at low batch×head on every card** (1.2x–2.2x across A100/H100/H200/B200). This is the regime the parallelized landmark kernel fixed: a single landmark's segment of N/m rows used to be summed by one thread serially (latency-bound at large N); splitting that reduction across threads made it bandwidth-bound, and the fused GEMMs already saved HBM traffic vs cuBLAS.
- **End-to-end, FlashNystrom wins at every size on every card** (A100 total 1.19x–4.04x, H100 1.19x–3.67x, H200 1.40x–3.12x, B200 1.03x–3.37x), and the margin grows with N. The single largest contributor at long context is the output-parallel, 128-bit-vectorized landmark gradient scatter: the previous (BH, m)-grid scatter serialized at low batch×head and was 81–83% of the long-context backward (58.8 ms at N=2M on B200 against a ~0.5 ms memory roofline).
- **On H200 and B200 the win survives compatibility mode.** FN's SM80-atom build beats natively-dispatched cuBLAS at every size (H200 1.40x–3.12x, B200 1.03x–3.37x total). The one sub-parity metric left is the high batch×head *forward* on B200 (0.64x–0.99x), which is exactly what the Blackwell-native atom port targets.
- **The remaining headroom is an atom update, not an algorithm change — and it is on the roadmap.** FlashNystrom is a *recipe* for Nyströmformer kernels: the kernel structure and the math are generation-invariant; only the mma/copy atom changes per generation (SM80 mma + `cp.async` → Hopper WGMMA/TMA → Blackwell TMEM), exactly as FlashAttention-2 → 3 → 4 are atom/arch ports of the *same* attention math. The repo ships the SM80-atom recipe, which by design runs on every Ampere-through-Blackwell card from one binary; the Blackwell-native atom port (in progress on the `blackwell-native` branch) targets the remaining B200 high-batch forward gap. The O(mN)-vs-O(N²) advantage over *exact* attention (the FlashAttention tables below) is asymptotic and independent of the atom generation.

Reproduce with `modal run tools/modal_a100.py::bench_gaps` (A100), `::bench_gaps_h100`, `::bench_gaps_h200`, or `::bench_gaps_b200`. Requires a Modal account and a one-time `modal setup`.

## FlashAttention-2 / FlashAttention-3 (exact attention), H100

The tables above compare FlashNystrom to the *same* Nyström algorithm in cuBLAS. This one compares it to the alternative people actually reach for: **exact** attention via FlashAttention. FA2 and FA3 compute exact O(N²) attention (FA3 is the Hopper-native current SOTA); FlashNystrom computes approximate O(m·N) Nyström. They are not the same computation, so this is a speed comparison that only matters where the Nyström approximation is acceptable (it is for the CIFAR-10 ViT, which matches the exact-attention baseline accuracy). H100-80GB, FP16, fwd+bwd, newton_iter=6. `FA2/FN` and `FA3/FN` are FA_total / FN_total; > 1 means FlashNystrom is faster. FA3 was built with its cluster and hdim-64/128 kernels intact (only genuinely unused variants trimmed), so these are its best kernels for these shapes.

High batch×head (B=4, H=16, head_dim=128, m=64):

| N      | FN tot | FA2 tot | FA3 tot | FA2/FN | FA3/FN |
|-------:|-------:|--------:|--------:|-------:|-------:|
|   4096 |   3.74 |    5.89 |    3.41 |  1.6x  |  0.9x  |
|  16384 |   8.25 |   90.9  |   50.2  | 11.0x  |  6.1x  |
|  65536 |  25.5  | 1464    |  829    | 57.5x  | 32.6x  |
| 131072 |  48.7  | 5833    | 3378    |  120x  | 69.3x  |

Long context, few heads (B=1, H=4, head_dim=64, m=32):

| N       | FN tot | FA2 tot | FA3 tot | FA2/FN | FA3/FN |
|--------:|-------:|--------:|--------:|-------:|-------:|
|   16384 |   1.08 |    3.09 |    1.75 |  2.9x  |  1.6x  |
|   65536 |   1.59 |   48.6  |   34.2  | 30.6x  | 21.6x  |
|  131072 |   2.51 |  198    |  122    | 79.1x  | 48.8x  |
|  262144 |   3.92 |  799    |  479    |  204x  |  122x  |
|  524288 |   7.84 | 3288    | 1942    |  420x  |  248x  |
| 1048576 |  12.8  | 13285   | 7806    | 1041x  |  612x  |
| 2097152 |  23.6  | n/r     | n/r     |   -    |   -    |

![FlashNystrom (approx O(mN)) vs FlashAttention-2/3 (exact O(N^2)) fwd+bwd latency on H100, log-log](assets/latency_flashattention_h100.png)

Reading the tables:

- **At short N, use exact attention.** At N=4096 (high batch×head) FA3 slightly beats FN (0.9x), and the two are close in long context at N=16384 (1.6x). Exact attention is cheap when N² is small and carries no approximation error. The crossover is roughly N=4K to 16K.
- **Past the crossover the O(N²) wall takes over.** FlashNystrom's O(m·N) cost grows linearly while exact attention grows quadratically, so the gap widens fast: 6.1x at 16K, 33x at 65K, 69x at 131K (high batch×head); and in long context from 49x at 131K up to **~612x at 1M tokens** versus FA3.
- **Exact attention eventually stops being practical.** At N=1M, FA2 is already 13 s per fwd+bwd call (FA3 ~8 s) and climbing quadratically; at 2M tokens (`n/r`) we no longer run it, while FlashNystrom finishes the full fwd+bwd in 24 ms.
- **FA3 is ~1.7x faster than FA2** here (Hopper-native kernels), so it is the right exact-attention baseline. FlashNystrom still pulls away from FA3 at long N.

Built and measured with `modal run tools/modal_a100.py::bench_fa_h100` (installs FA2 plus a trimmed FA3 Hopper build, then benchmarks).

### Exact attention on Blackwell (B200): FA2 measured, FA4 estimated

We can now run on a B200 (added to the Modal harness). FlashAttention-2 (exact O(N²)) builds and runs there, so the asymptotic comparison is **measured directly on Blackwell** — FN (the SM80-atom recipe, in compatibility mode) vs FA2, long context (B=1, H=4, head_dim=64, m=32):

| N       | FN tot | FA2 tot  | FA2/FN |
|--------:|-------:|---------:|-------:|
|   16384 |   1.03 |     3.10 |  3.0x  |
|   65536 |   1.46 |    42.81 | 29.3x  |
|  131072 |   2.00 |   171.12 | 85.6x  |
|  262144 |   3.04 |   685.24 |  225x  |
|  524288 |   5.11 |  2717.13 |  532x  |
| 1048576 |   9.18 | 10884.08 | 1186x  |
| 2097152 |  17.36 |    n/r   |   -    |

Even on Blackwell, against exact attention, the O(N²) wall is the same shape: FN is **1186x faster at 1M tokens** (FA2 OOMs at 2M; FN finishes in 17 ms).

**FA4 specifically is not yet measured.** flash-attn-4 is the Blackwell/Hopper-native exact kernel and the right constant-factor baseline there, but it ships only beta wheels (`4.0.0bN`) whose `nvidia-cutlass-dsl` dependency is currently unsatisfiable on the package index: the newer cutlass-dsl removed `cute.core.ThrMma` (import error), the older one removed `cutlass.utils.ampere_helpers` (different import error), and b19 needs an intermediate snapshot that is not published (cf. flash-attention issues [#2310](https://github.com/Dao-AILab/flash-attention/issues/2310), [#2334](https://github.com/Dao-AILab/flash-attention/issues/2334)). The harness is in place (`modal run tools/modal_a100.py::bench_fa4_b200`) and will produce numbers once FA4's packaging stabilizes.

Until then the estimate below projects FA4 from published throughput. FA4 is faster than FA2, so the true FN-vs-FA4 ratios sit *below* the measured FN-vs-FA2-on-B200 numbers above and *above* the derived floor here. We bridge through published peak attention throughput. FA4 reports **~1605 TFLOP/s** (BF16, 71% utilization) on B200; FA3 reports **~740 TFLOP/s** (FP16, 75% utilization) on H100. BF16 and FP16 run at the same tensor-core rate, so for compute-bound attention FA4-on-B200 is about **2.2x faster than FA3-on-H100** (1605 / 740). Dividing our *measured* FN-vs-FA3 ratios by that factor:

`FN/FA4 (derived)  ≈  (FN/FA3 measured on H100)  /  2.2`

Long context (B=1, H=4, head_dim=64, m=32):

| N       | FA3/FN (measured, H100) | FA4/FN (derived) |
|--------:|------------------------:|-----------------:|
|   16384 |                    1.6x |            ~0.7x |
|   65536 |                   21.6x |             ~10x |
|  131072 |                   48.8x |             ~22x |
|  262144 |                    122x |             ~55x |
|  524288 |                    248x |            ~113x |
| 1048576 |                    612x |            ~278x |

(At high batch×head the same division applies: the measured 69x vs FA3 at N=131072 becomes ~32x vs FA4.)

These numbers are *derived from published throughput, not measured.* They also handicap FlashNystrom on purpose: FN runs on H100, FA4 on its native B200, and the 2.2x bridge hands FA4 the entire B200-plus-next-gen-kernel improvement, so these ratios are a **floor** on FN's advantage. On equal hardware FN would look better, not worse. The throughput proxy is fair in the long-N compute-bound regime where this comparison matters (at short N exact attention wins anyway and is the right choice), and it uses forward throughput while the table is fwd+bwd.

The point holds: at long context FN's O(m·N) is far enough ahead that a ~2.2x faster exact kernel on a newer GPU is still tens of times slower at N ≥ 128K. FA4 moves the crossover out (roughly to N = 16K to 32K); it does not remove it.

Sources: FlashAttention-4 (Colfax Research / Together AI, arXiv:2603.05451); FlashAttention-3 (Shah et al., 2024).

## SMEM sizing and occupancy

The kernels are sized for the consumer SMEM envelope (~100 KB/SM on Ampere
consumer, Ada, and Blackwell consumer). The build does not auto-tune
tile sizes to the runtime device; the choice is fixed at compile time.

Per-kernel SMEM usage (probe output on an RTX 5060 Laptop, 100 KB/SM,
m=64, D=128, FP16, niter=6):

| Kernel                          | Dyn SMEM (KB) | Regs/thr | Blocks/SM (consumer) | Binding constraint |
|---------------------------------|--------------:|---------:|---------------------:|--------------------|
| `landmark_kernel` (fwd)         |           8   |      40  |                  1   | threads (1024/blk) |
| `kernel1_fused_tc` (fwd)        |          32   |      71  |                  3   | SMEM               |
| `kernel3_fused_tc` (fwd, sync)  |          32   |     168  |                  3   | registers (= SMEM) |
| `kernel3_fused_tc` (fwd, pipe)  |          48   |     168  |                  2   | SMEM               |
| `kernel1_bwd_tc` (narrow)       |          48   |     156  |                  2   | SMEM               |
| `kernel1_bwd_tc` (wide)         |          64   |     158  |                  1   | SMEM               |
| `kernel3_bwd_tc` (narrow)       |          40   |     171  |                  2   | registers (= SMEM) |
| `kernel3_bwd_tc` (wide)         |          72   |     194  |                  1   | SMEM               |
| `compute_dk2inv_tc`             |          64   |     206  |                  1   | SMEM               |
| `kernel2_inv` (NS forward)      |          96   |      42  |                  1   | SMEM               |
| `ns_bwd_step`                   |          96   |      40  |                  1   | SMEM               |

The pipelined forward and the wide backward variants cost extra SMEM, so the
launchers gate them on a runtime occupancy check: on consumer parts
(~100 KB/SM) the sync/narrow variants keep 2–3 blocks/SM and win; on
datacenter parts (164–228 KB/SM) the pipelined/wide variants are free and are
selected automatically.

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
