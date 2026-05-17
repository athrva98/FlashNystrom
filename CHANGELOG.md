# Changelog

All notable changes to FlashNystrom are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version
numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `tests/test_ns_bwd_graph.py` — isolation tests for the production
  cuBLAS + CUDA-graph NS backward (`launch_kernel2_inv_bwd`). Pins
  graph-replay correctness, shape-change cache invalidation,
  `reset_caches()` behaviour, and a memory-leak smoke test. Previously
  this code path was only reachable through the end-to-end autograd
  pipeline, which made bisecting a graph-capture regression painful.
- `flash_nystrom._C.reset_caches()` — frees the thread-local NS-backward
  graph caches and workspaces across all dtypes. Useful between training
  runs of different shapes, or before measuring residual GPU memory.
- `FLASH_NYSTROM_CUDA_ARCH_LIST` env var to override the compute-capability
  detection in `setup.py`. Also accepts `TORCH_CUDA_ARCH_LIST` for
  compatibility with the PyTorch convention.
- Per-tensor input validation in the backward pybind binding: device,
  contiguity, dtype, and exact-shape checks for every saved tensor.
- `CHANGELOG.md` (this file).
- GitHub Actions sdist workflow (`sdist.yml`, CPU-only, always-on) and
  GPU test workflow scaffold (`tests-gpu.yml`, self-hosted runner, gated
  by `if: false` until a runner is attached).

### Removed
- The custom CUDA depthwise-conv kernels (`csrc/kernels/dconv_residual.cuh`
  and `csrc/kernels/backward/dconv_residual_bwd.cuh`) and all conv plumbing
  through the C extension boundary. The conv residual is now exclusively
  computed at the Python level via `F.conv1d` (cuDNN) inside
  `flash_nystrom_attention`; the bypassed C++ path was bit-rotting unused
  code reachable only by passing `conv_weight=` directly to `_C.forward`.
- Removed conv arguments from `_C.forward`, `_C.backward`,
  `NystromParams`, `NystromBwdParams`, and `FlashNystromFunction.{forward,
  backward}`. The public Python API (`flash_nystrom_attention` keyword
  args `conv_weight=` and `conv_kernel_size=`) is unchanged — those are
  applied via cuDNN.

### Changed
- CUDA errors and cuBLAS errors now throw `std::runtime_error` instead of
  calling `abort()`. A failure inside the kernel pipeline propagates as a
  normal Python `RuntimeError` that autograd can unwind, instead of
  killing the user's training process. Five call sites updated:
  `csrc/utils.h` (FN_CUDA_CHECK, FN_CUDA_KERNEL_CHECK, FN_CHECK),
  `csrc/cublas_helpers.cuh`, `csrc/kernels/backward/kernel2_inv_bwd.cu`.
- `setup.py` no longer requires `CUDA_HOME` at sdist-build time. The
  torch.utils.cpp_extension import and the `CUDAExtension(...)` call
  are deferred behind a command-line check; `sdist`, `egg_info`,
  `dist_info`, `check`, `clean`, and the help commands work in any
  CUDA-less environment (PyPI sdist builds, CI runners without GPUs).
- Compute-capability detection in `setup.py`: when `nvidia-smi` is
  unavailable, fall back to a multi-arch wheel covering
  `sm_80, sm_86, sm_89, sm_90` instead of a single-arch `sm_80` wheel
  that would not run on Hopper or Blackwell consumer GPUs.
- `flash_nystrom.__version__` is now read from the installed package
  metadata via `importlib.metadata.version`. Bumping the version in
  `pyproject.toml` is now the only required edit; `__init__.py` picks
  it up automatically.
- `launch_kernel2_inv_bwd` API: dropped five unused parameters
  (`lse2`, `k2_inv`, `dZ_workspace`, `dK2_workspace`, `ns_step_scratch`).
  The cuBLAS+graph path owns its own persistent workspaces (per-thread
  `NsBwdGraphState` cache), so caller-side allocation is no longer
  necessary and was being silently ignored.
- `NystromBwdParams` shrank correspondingly. The orchestrator no longer
  allocates `ns_dZ_workspace`, `ns_dK2_workspace`, or `ns_step_scratch`
  (saved ~352 KB GPU memory per backward at typical configs).
- `FLASH_NYSTROM_VERBOSE=1` replaces `FLASH_NYSTROM_QUIET=1` as the
  control for `nvcc -Xptxas=-v --resource-usage`. Default is now quiet
  (the noise was only useful for the kernel-tuning iterations).
- Built artifacts (`flash_nystrom/_C.*.pyd`, `_C.*.so`) are excluded from
  the sdist via `global-exclude` in `MANIFEST.in`.

### Fixed
- README install instructions used a `<your-fork>` placeholder. Replaced
  with the actual upstream URL.
- `pyproject.toml` had a stale `flash-attn>=2.0` dev dependency that
  nothing referenced. Removed.

### Packaging
- CUTLASS is now a proper git submodule pinned at NVIDIA/cutlass@b78588d
  (CUTLASS 3.7). Previously it was a local clone hidden by `.gitignore`
  with a README that lied about it being a submodule.
- `MANIFEST.in` added. The sdist now ships `csrc/`, the CUTLASS include
  tree, `LICENSE`, `README.md`, and excludes `*.pyd`, `*.so`, and the
  rest of the CUTLASS repository (tools, tests, examples, docs).
- `pyproject.toml` filled in: authors, URLs, classifiers, keywords,
  optional-deps split into `dev` and `bench`.

## [0.1.0] — initial release

Forward and backward CUDA kernels for Nyströmformer approximate attention.

### Kernels
- Multi-CTA flash-attention-style tensor-core kernels for the three
  softmaxes (`kernel1_output_fused`, `kernel2_inv`, `kernel3_output_fused`)
  and their backwards (`kernel1_bwd_tc`, `kernel2_inv_bwd`,
  `kernel3_bwd_tc`).
- FP32 scalar fallbacks for every kernel (the TC atom requires 16-bit
  operands; FP32 inputs go through the scalar path).
- Newton-Schulz pseudoinverse forward (`kernel2_inv`) with all iterates
  saved for the unrolled backward (no IFT shortcut; gradient is exact
  irrespective of convergence).

### Performance optimizations
- B-reuse: forward `kernel3_output_fused` saves `B = softmax(Qt @ K^T) @ V`
  to GMEM so the backward's `compute_dk2inv` can skip the N-walk.
- Split-K reduction for the `dQ_tilde` accumulator in `kernel3_bwd_tc`
  (replaces atomicAdd contention at long N).
- cuBLAS-based Newton-Schulz backward step, captured into a per-shape
  CUDA graph in a thread-local cache.
- cuBLAS GemmEx for the trailing matmuls in `ns_bwd_final`.

### Python / autograd
- `FlashNystromAttention` `nn.Module` and `flash_nystrom_attention`
  functional form.
- `torch.autograd.Function` wrapper with full saved-tensor protocol.
- Pure-PyTorch reference implementation (`flash_nystrom.reference`)
  used by the test suite as ground truth.

### Tests
- 71 pytest tests: 28 forward, 23 backward, 20 kernel-isolation tests
  for the debug pybind hooks.
- 5-way diagnostic CIFAR-10 harness (`benchmarks/train_five_way.py`)
  that compares SDPA, pure-PyTorch reference, full FlashNystrom, and
  the two mixed configurations (FN-fwd + torch-bwd, torch-fwd + FN-bwd).
