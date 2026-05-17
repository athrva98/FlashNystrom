# FlashNystrom

CUDA kernels for Nystromformer approximate attention. Forward and backward run in linear time and memory with respect to sequence length. The matmul-heavy stages use tensor cores. Backward gradients are exact against PyTorch autograd at FP32 numerical noise.

The Nystromformer factorization is

```
attention(Q, K, V) = softmax(Q @ Kt^T) @ softmax(Qt @ Kt^T)^+ @ softmax(Qt @ K^T) @ V
```

where Qt and Kt are landmarks formed by segmented mean pooling of Q and K. The pseudoinverse is computed by unrolled Newton-Schulz iteration in FP32. The backward pass differentiates through every NS iterate via the chain rule. There is no Implicit Function Theorem dependence and no requirement that NS has converged.

## Scope

FlashNystrom is not a FlashAttention competitor. FlashAttention (v1/v2/v3/v4) implements *exact* O(N²) attention with IO-aware tiling. Its version bumps are hardware-targeted rewrites of the same algorithm: FA2 for Ampere and Ada, FA3 for Hopper WGMMA and TMA, FA4 for Blackwell TMEM. FlashNystrom implements a *different* attention math: the Nyström low-rank factorization, which is O(m·N·D + m³) with m landmarks. The relevant comparison is FlashNystrom against SDPA (using any FA generation under the hood) at long sequence length, where O(N²) starts to dominate and the approximation becomes worthwhile. At short N (under ~1–2K), exact attention is faster and you should use it.

The kernels borrow the FA2-era CUTLASS SM80 mma atom and the tiled-softmax with running-LSE pattern, but apply them to the three Nyström softmaxes rather than to one big QK^T. They are written in pre-Hopper idioms: no WGMMA, no TMA, no warp specialization, no TMEM. They run on Hopper and Blackwell (the build covers `sm_80;86;89;90`) and benefit from the higher SMEM and register counts on those parts via occupancy, but a Hopper-native rewrite would extract more peak throughput. See [the SMEM sizing discussion](#smem-sizing-and-occupancy) below.

## Status

20-epoch CIFAR-10 ViT (default settings, FP16 autocast, num_landmarks=32, newton_iter=6) reaches the same test accuracy as the SDPA and pure-PyTorch Nystromformer baselines:

| Config                          | test acc |
|---------------------------------|---------:|
| `F.scaled_dot_product_attention`|    66.7% |
| Pure-PyTorch Nystromformer      |    66.3% |
| FlashNystrom (this repo)        |    66.7% |

84 tests cover forward, backward, kernel-level isolation, the production cuBLAS + CUDA-graph NS backward path, and per-kernel regression against autograd-derived references.

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
* CUDA toolkit 11.8+
* Compute capability 8.0+ (Ampere, Ada, Hopper, Blackwell). The kernels use the SM80 16x8x16 mma atom and opt into up to ~96 KB of dynamic shared memory per CTA. They run on Hopper and Blackwell but are not Hopper-native (no WGMMA/TMA). SM75 and earlier are not supported.

## Quickstart

Module form:

```python
import torch
from flash_nystrom import FlashNystromAttention, NystromConfig

cfg = NystromConfig(num_landmarks=64, newton_iter=6, conv_kernel_size=3)
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

out = flash_nystrom_attention(q, k, v, num_landmarks=64, newton_iter=6)
```

## Latency

Forward and backward latency in milliseconds on an RTX 5060 Laptop (Blackwell consumer, 8 GB VRAM, sm_120), FP16, B=1, H=4, head_dim=64, num_landmarks=32, newton_iter=6. Three implementations: FN (this repo), Ref (pure-PyTorch Nyström, same algorithm), SDPA (`F.scaled_dot_product_attention`, which dispatches to PyTorch's memory-efficient attention backend). CUDA-event timed, median of 30 fwd+bwd runs after 5 warmups; reduced rep counts at N ≥ 16384 to keep wall-clock manageable.

| N      | FN fwd | FN bwd | FN tot | Ref tot | SDPA fwd | SDPA bwd | SDPA tot | FN/Ref | FN/SDPA |
|-------:|-------:|-------:|-------:|--------:|---------:|---------:|---------:|-------:|--------:|
|    128 |   0.16 |   0.71 |   0.87 |    4.68 |     0.03 |     0.23 |     0.26 |  5.4x  |   0.30x |
|    256 |   0.15 |   0.50 |   0.65 |    4.64 |     0.03 |     0.23 |     0.26 |  7.1x  |   0.40x |
|    512 |   0.16 |   0.49 |   0.65 |    5.30 |     0.04 |     0.19 |     0.23 |  8.2x  |   0.35x |
|   1024 |   0.18 |   0.48 |   0.66 |    4.78 |     0.10 |     0.31 |     0.41 |  7.2x  |   0.62x |
|   2048 |   0.21 |   0.50 |   0.72 |    5.68 |     0.29 |     0.96 |     1.24 |  7.9x  |   1.7x  |
|   4096 |   0.29 |   0.57 |   0.86 |    4.72 |     1.07 |     3.51 |     4.58 |  5.5x  |   5.3x  |
|   8192 |   0.43 |   0.78 |   1.21 |    4.79 |     4.15 |    13.71 |    17.86 |  4.0x  |  14.8x  |
|  16384 |   0.82 |   1.36 |   2.18 |    5.08 |    16.95 |    56.97 |    73.92 |  2.3x  |  33.9x  |
|  32768 |   1.65 |   2.58 |   4.23 |    8.06 |    69.22 |   222.01 |   291.24 |  1.9x  |  68.9x  |
|  65536 |   4.01 |   4.86 |   8.87 |   10.96 |   278.64 |   948.59 |  1227.23 |  1.2x  |   138x  |
| 131072 |   7.91 |   9.55 |  17.46 |   21.16 |  1125.10 |  3761.69 |  4886.79 |  1.2x  |   280x  |
| 262144 |  15.72 |  18.52 |  34.24 |   48.58 |  4599.10 | 15279.00 | 19878.10 |  1.4x  |   581x  |

The speedup columns are *base time / FN time*. Values > 1 mean FN is faster; values < 1 mean FN is slower than the base.

Reading the table:

- **At short N (≤ 1024), SDPA is faster than FN.** FN carries fixed overhead from its three softmaxes and the Newton-Schulz pseudoinverse; that overhead dominates while N² is still cheap. If your N stays under ~1 K, use SDPA.
- **The fwd+bwd crossover is between N = 1024 and N = 2048.** At N = 2048 FN is 1.7x faster than SDPA total. Above that point the gap widens monotonically.
- **Above N ≈ 8 K the speedup grows roughly linearly with N**, as expected from FN's O(N) compute versus SDPA's O(N²). Doubling N from 16 K to 32 K doubles the speedup (34x → 69x). Same at 32 K → 64 K (69x → 138x), 64 K → 128 K (138x → 280x), and 128 K → 256 K (280x → 581x).
- **FN beats the pure-PyTorch Nyström reference at every N tested.** The Ref column is the same algorithm in plain PyTorch (cuBLAS matmuls + torch softmax) and shows what's bought by the custom kernels. The FN/Ref ratio shrinks at large N because both methods are O(N) and the gap is the per-call kernel-launch + memory-traffic overhead, not asymptotic complexity.
- **Neither method OOMs at N = 262144 on 8 GB.** SDPA's wall is wall-clock (~20 s per fwd+bwd at N = 256 K), not memory. PyTorch's SDPA uses memory-efficient attention internally, so it scales linearly in memory; the O(N²) compute is what makes it unusable past 32 K or so in practice.

Reproduce with `python benchmarks/bench_fwd_bwd.py`.

## SMEM sizing and occupancy

The kernels are sized for the consumer SMEM envelope (~100 KB/SM on Ampere
consumer, Ada, and Blackwell consumer). The build does not auto-tune
tile sizes to the runtime device; the choice is fixed at compile time.

Per-kernel SMEM usage (probe output on an RTX 5060 Laptop, 100 KB/SM,
m=64, D=128, FP16, niter=6):

| Kernel                        | Dyn SMEM (KB) | Regs/thr | Blocks/SM (consumer) | Binding constraint |
|-------------------------------|--------------:|---------:|---------------------:|--------------------|
| `kernel1_fused_tc` (fwd)      |          32   |      71  |                  3   | registers          |
| `kernel3_fused_tc` (fwd)      |          32   |     165  |                  3   | registers          |
| `kernel1_bwd_tc`              |          48   |     163  |                  2   | SMEM + registers   |
| `kernel3_bwd_tc`              |          40   |     170  |                  2   | registers          |
| `compute_dk2inv_tc`           |          64   |     206  |                  1   | registers          |
| `kernel2_inv` (NS forward)    |          96   |      42  |                  1   | SMEM               |
| `ns_bwd_step`                 |          96   |      40  |                  1   | SMEM               |

Reproduce with `python tools/kernel_report.py`.

**Are we leaving performance on the table on bigger-SMEM GPUs?**

Yes and no, and not in the way most people assume.

What we get for free on bigger SMEM (H100 has 228 KB/SM, ~2.3× consumer):
- Occupancy scales automatically. The SMEM-bound kernels (`kernel2_inv`,
  `ns_bwd_step`) double their blocks/SM. The 40 to 48 KB bwd kernels
  gain roughly one extra block/SM until registers become the binder.
- The three matmul-heavy hot kernels (`kernel3_fused_tc`,
  `kernel3_bwd_tc`, `compute_dk2inv_tc`) are register-bound, not
  SMEM-bound. See the regs/thr column above (165 to 206 regs/thr at
  128 threads/block). Larger SMEM does nothing for those: register
  count caps occupancy first. A real win there requires fewer
  registers (smaller accumulator fragments, recomputation), not
  more SMEM.

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
| `num_landmarks`    |      64 | Capped at 64 by kernel tile size. |
| `newton_iter`      |       6 | NS iterations for the pseudoinverse. Backward correctness is independent of convergence. |
| `conv_kernel_size` |       3 | Depthwise conv1d residual on V. Set to 0 to disable. |
| `use_conv_residual`|    True | Master switch for the conv residual. |
| `fast_dk2inv`      |    True | Internal flag for a debug-only fallback path. Leave at the default. |

## Limitations

* `head_dim` is restricted to 64 or 128.
* `num_landmarks` is capped at 64.
* FP32 backward at `head_dim=128` is not supported (SMEM overflow). Use FP16 or BF16.
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
tests/                         84 pytest tests
benchmarks/                    latency and CIFAR-10 training scripts
examples/                      end-to-end usage examples
docs/                          longer technical writeup
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

The kernel layouts, the tiled-softmax running-LSE state machine, and the
CUTE SmemLayoutAtomQ/KV patterns are adapted from FlashAttention-2. We do
not implement FA3-style asynchrony (WGMMA + TMA + warp specialization);
those are Hopper-specific and would be a separate kernel family.
FlashAttention solves exact O(N²) attention; FlashNystrom uses these
techniques to implement the Nyström low-rank factorization instead.

## License

Apache License 2.0. See `LICENSE`.

## Author

Athrva Pandhare. athrva98@gmail.com.

Developed with Claude's help.
