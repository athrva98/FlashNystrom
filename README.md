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
git clone --recursive https://github.com/<your-fork>/FlashNystrom.git
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

Forward and backward latency in milliseconds on an RTX 5060 Laptop, FP16, B=1, H=4, head_dim=64, num_landmarks=32, newton_iter=6. Median of 50 runs after 10 warmup iterations.

| N    | FN fwd | FN bwd (default) | FN bwd (`fast_dk2inv=True`) | SDPA fwd | SDPA bwd |
|-----:|-------:|-----------------:|-----------------------------:|---------:|---------:|
|  128 |   0.15 |             0.74 |                         0.84 |     0.04 |     0.49 |
|  256 |   0.15 |             0.81 |                         0.58 |     0.03 |     0.25 |
|  512 |   0.16 |             1.05 |                         0.60 |     0.04 |     0.19 |
| 1024 |   0.18 |             1.57 |                         0.64 |     0.10 |     0.31 |
| 2048 |   0.21 |             2.52 |                         0.66 |     0.29 |     0.95 |
| 4096 |   0.29 |             4.49 |                         0.77 |     1.06 |     3.50 |
| 8192 |   0.43 |             8.80 |                         1.08 |     4.12 |    13.89 |

The forward pass is faster than SDPA at every N. The backward crosses over near N=4096 in the default configuration and near N=2048 in the `fast_dk2inv=True` configuration. At N=8192 the total fwd+bwd is roughly 12x faster than SDPA in the fast mode.

Reproduce with `python benchmarks/bench_fwd_bwd.py`.

## The fast_dk2inv flag

`compute_dk2inv` is the kernel that produces the gradient of the loss with respect to the pseudoinverse iterate Z_N. By default it runs in FP32 scalar so that its output is bit-for-bit consistent with PyTorch autograd modulo FP32 reduction order.

Setting `fast_dk2inv=True` switches it to a tensor-core path that converts the softmax output P from FP32 to FP16/BF16 before the second GEMM. The 10-bit mantissa truncation costs a small amount of precision in P. On 20-epoch CIFAR-10 ViT this falls within FP16 single-seed variance and accuracy is preserved within the noise floor. Set this flag when latency matters more than the last fraction of a percentage point of training accuracy.

The flag does nothing on FP32 input dtype. The tensor-core kernels require 16-bit operands.

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
| `fast_dk2inv`      |   False | Opt-in tensor-core path for `compute_dk2inv`. FP16/BF16 only. |

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
