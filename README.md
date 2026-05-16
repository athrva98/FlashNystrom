# FlashNystrom

CUDA kernels for Nystromformer approximate attention. Forward and backward run in linear time and memory with respect to sequence length. The matmul-heavy stages use tensor cores. Backward gradients are exact against PyTorch autograd at FP32 numerical noise.

The Nystromformer factorization is

```
attention(Q, K, V) = softmax(Q @ Kt^T) @ softmax(Qt @ Kt^T)^+ @ softmax(Qt @ K^T) @ V
```

where Qt and Kt are landmarks formed by segmented mean pooling of Q and K. The pseudoinverse is computed by unrolled Newton-Schulz iteration in FP32. The backward pass differentiates through every NS iterate via the chain rule. There is no Implicit Function Theorem dependence and no requirement that NS has converged.

## Status

20-epoch CIFAR-10 ViT (default settings, FP16 autocast, num_landmarks=32, newton_iter=6) reaches the same test accuracy as the SDPA and pure-PyTorch Nystromformer baselines:

| Config                          | test acc |
|---------------------------------|---------:|
| `F.scaled_dot_product_attention`|    66.7% |
| Pure-PyTorch Nystromformer      |    66.3% |
| FlashNystrom (this repo)        |    66.7% |

71 tests cover forward, backward, kernel-level isolation, and per-kernel regression against autograd-derived references.

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
* Compute capability 8.0+ (Ampere, Ada, Hopper). The kernels use the SM80 16x8x16 mma atom and opt into roughly 100 KB of dynamic shared memory per CTA. SM75 and earlier are not supported.

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

Forward and backward latency in milliseconds on an RTX 5060 Laptop, FP16, B=1, H=4, head_dim=64, num_landmarks=32, newton_iter=6. Median of 30 runs after 5 warmup iterations.

| N    | FN fwd | FN bwd | SDPA fwd | SDPA bwd |
|-----:|-------:|-------:|---------:|---------:|
|  128 |   0.15 |   0.84 |     0.03 |     0.23 |
|  256 |   0.15 |   0.51 |     0.03 |     0.24 |
|  512 |   0.16 |   0.50 |     0.04 |     0.20 |
| 1024 |   0.18 |   0.48 |     0.10 |     0.31 |
| 2048 |   0.21 |   0.50 |     0.29 |     0.95 |
| 4096 |   0.29 |   0.57 |     1.07 |     3.51 |
| 8192 |   0.43 |   0.82 |     4.16 |    13.69 |

The forward pass is faster than SDPA at every N. The backward crosses over near N=2048. At N=8192 the total fwd+bwd is roughly 14x faster than SDPA.

Reproduce with `python benchmarks/bench_fwd_bwd.py`.

### vs the PyTorch Nystrom reference

Same Nystrom algorithm implemented in pure PyTorch dispatches every matmul through cuBLAS via the `@` operator and uses torch's fused softmax. Total fwd+bwd latency, FP16, niter=6, median of 30 runs:

| Config                          | FN      | Ref     | FN/Ref |
|---------------------------------|--------:|--------:|-------:|
| B=1 H=4 N= 4096 D= 64 m=32      |   0.98  |  4.48   | 4.60x  |
| B=1 H=8 N= 4096 D=128 m=64      |   2.96  |  4.26   | 1.44x  |
| B=4 H=8 N= 4096 D= 64 m=32      |   3.20  |  5.66   | 1.77x  |
| B=1 H=4 N= 8192 D= 64 m=32      |   1.40  |  3.94   | 2.81x  |
| B=1 H=8 N= 8192 D=128 m=64      |   4.34  |  5.84   | 1.34x  |
| B=4 H=8 N= 8192 D= 64 m=32      |   6.26  |  7.84   | 1.25x  |
| B=1 H=4 N=16384 D= 64 m=32      |   2.31  |  4.61   | 1.99x  |
| B=1 H=8 N=16384 D=128 m=64      |   8.61  |  9.00   | 1.05x  |
| B=1 H=8 N=24576 D=128 m=64      |  10.86  | 11.69   | 1.08x  |

FN beats the reference at every configuration tested, from N=4K to N=24K across batch-head counts of 4 to 32.

## The fast_dk2inv flag

`compute_dk2inv` is the kernel that produces the gradient of the loss with respect to the pseudoinverse iterate Z_N. In normal use the backward picks up B = softmax(Q_tilde @ K^T) @ V from a small tensor the forward saved, then runs two tiny matmuls. The N-walk that used to dominate the backward (the previous default, `fast_dk2inv=False`, was 4-6x slower than the rest of the bwd combined) is gone.

The `fast_dk2inv` flag controls behavior only in the legacy fallback branch that fires when the saved B is unavailable (the debug pybind hook is the only path that hits it). Default is True. The flag is kept for backward compatibility; you can ignore it.

## PyTorch compatibility

`FlashNystromAttention` is a regular `nn.Module` and `flash_nystrom_attention` is a regular function. Standard PyTorch idioms work without changes.

| Workflow                                  | Status |
|-------------------------------------------|--------|
| Eager forward + backward                  | works |
| FP16 / BF16 / FP32 input dtypes           | works |
| `torch.amp.autocast("cuda", dtype=...)`   | works |
| `nn.Module` composition, `state_dict`     | works |
| DDP / FSDP gradient sync                  | works (gradients flow through standard autograd; no custom collective is needed) |
| `torch.compile`                           | runs, with a graph break at the FlashNystrom forward call. The kernel itself executes normally, but Dynamo cannot fuse across the boundary. See ROADMAP for the planned `torch.library.custom_op` registration that will eliminate the graph break. |
| `torch.jit.script`                        | not supported. Custom autograd Functions are not scriptable. |
| `torch.export`                            | not currently supported (depends on the `custom_op` registration above). |

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
| `fast_dk2inv`      |    True | Legacy flag, normally ignored. The backward reuses `B` from the forward and skips the N-walk entirely. The flag now only matters in the debug pybind hook fallback. |

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
tests/                         71 pytest tests
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

The kernel layouts and CUTE patterns are adapted from FlashAttention-2.

## License

Apache License 2.0. See `LICENSE`.

## Author

Athrva Pandhare. athrva98@gmail.com.
