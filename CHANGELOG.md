# Changelog

All notable changes to FlashNystrom are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version
numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Tikhonov ridge** for the pseudoinverse, exposed as the `kappa_star`
  parameter (`NystromConfig.kappa_star`, default `5.0`). The pinv inverts
  `M = K2^T K2 + lambda*I` with `lambda = (||K2||_1 ||K2||_inf)/kappa_star`,
  guaranteeing `cond(M) <= kappa_star` and keeping the Newton-Schulz iteration
  well-conditioned as `cond(K2)` grows with N. `0.0` disables it (raw-K2 pinv,
  the original formulation). Forward and backward both use it (the backward
  inverts the same ridged `M`), and it is threaded identically to the kernel
  and the `m > 64` reference dispatch.
- **tf32 tensor-core Newton-Schulz pseudoinverse** forward, exposed as
  `use_tc_pinv` (default `True`, `m == 64` only), graph-captured per shape. Its
  precision floor (~6e-4) is tighter than the fp16-reference's (~1.2e-3); set
  `False` for the fp32 scalar kernel.
- **H200 + B200 (sm_100) support**: added `sm_100` to the build (one binary now
  spans A100/H100/H200/B200, verified `test_b200` 99/99) and Modal targets
  `bench_gaps_h200`, `bench_gaps_b200`, `test_h200`, `test_b200`, plus an
  `fa4_image` and `bench_fa4_b200`/`bench_fa4_h200` for the FlashAttention-4
  comparison. Note: on B200/H200 FN runs its SM80 atoms in compatibility mode
  while cuBLAS dispatches to native Blackwell/Hopper kernels (not like-for-like);
  at long context the native-kernel cuBLAS is faster by a constant factor that
  reflects the not-yet-written per-generation WGMMA/TMEM atom port, not the
  method. FA4 direct measurement is currently blocked by flash-attn-4's beta
  packaging (nvidia-cutlass-dsl version churn); FN-vs-FA2 is measured directly
  on B200.
- **FP32 at `head_dim=128`** is now allowed (previously hard-rejected). Mainly a
  gradient-checking / numerical-verification path; the scalar kernels opt into
  the ~150 KB SMEM they need, so it runs on datacenter GPUs (A100/H100/B200) and
  raises a clear `insufficient smem` error on smaller-SMEM consumer cards.
- `tests/test_precision_config.py` — regression tests pinning `kappa_star`
  validation, kernel-vs-reference consistency at matched `kappa_star`, and that
  `fast_dk2inv` does not bias the gradient.
- `benchmarks/README.md` indexing the diagnostic / performance / training
  scripts and marking the historical (archaeology) ones.
- `benchmarks/bench_5060_refresh.py` (local FN/SDPA/cuBLAS latency sweep at the
  current default config). The figure data in `benchmarks/plot_benchmarks.py`
  was refreshed from this plus the Modal A100/H100 runs.
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
  GPU test workflow (`tests-gpu.yml`): a `cpu-checks` job always runs the
  reference + config validation on GitHub-hosted runners, and the GPU job is
  enabled by setting the `GPU_RUNNER_AVAILABLE` repository variable once a
  self-hosted runner is attached.

### Removed
- **`FN_KAPPA_STAR` and `FN_K2INV_TC` environment variables.** These gated
  production numerics (the ridge and the pinv path) from the shell, with no
  validation and no visibility, and `FN_KAPPA_STAR` could silently desync the
  kernel from the reference. Replaced by the `kappa_star` / `use_tc_pinv`
  parameters (see Added). The remaining env vars (`FLASH_NYSTROM_PROFILE`,
  `FN_FP32_BWD`, `FLASH_NYSTROM_KERNEL3_SPLITS`) are diagnostic / perf-tuning
  only and never affect production correctness.
- Dead Bazel config (`MODULE.bazel`, `.bazelrc`, `.bazelversion`) — there were
  no `BUILD` files and CI never used it; setuptools is the only build system.
- Dead `NystromParams::seg_len` field (computed with ceil division but never
  read; the landmark kernel computes its own floor-division segments).
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
- **`kappa_star` and `use_tc_pinv` are now explicit, validated parameters**
  threaded through `NystromConfig` → `flash_nystrom_attention` →
  `FlashNystromFunction` → `_C.forward`/`_C.backward`/`NystromParams`, replacing
  the `FN_KAPPA_STAR` / `FN_K2INV_TC` env vars. Invalid `kappa_star`
  (negative / inf / NaN) is rejected at both the config and the C entry.
  `FlashNystromFunction.apply` arity is now 8 (`..., fast_dk2inv, kappa_star,
  use_tc_pinv`).
- `reset_caches()` now frees **all three** thread-local GPU caches on the
  calling thread (NS-backward graph, kernel3 split-N scratch, and the TC-pinv
  forward graph — the last was previously never freed). Docstring documents the
  per-thread lifetime and serving guidance.
- `kMaxLandmarks` constant corrected from `128` to `64` to match the enforced
  limit (the public entry hard-rejects `m > 64`).
- GPU CI is gated on a `GPU_RUNNER_AVAILABLE` repository variable instead of a
  hard `if: false`; an always-on `cpu-checks` job now runs the reference +
  config validation on GitHub-hosted runners every push.
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
- `kappa_star` could silently desync between the kernel and the reference: the
  kernel read `FN_KAPPA_STAR` from the environment while the Python reference
  used its own default, so a test or training run could compare a ridged kernel
  against an un-ridged reference. Both now take the same explicit parameter.
- `kernel1_bwd_scalar_kernel` rewritten from a single-line golfed form to
  readable, commented code (identical semantics; verified by the gradient tests
  plus clean compute-sanitizer memcheck/racecheck).
- Corrected a stale comment claiming the bf16 backward casts through fp16 — it
  does not (the backward runs in the native input dtype; the precision-sensitive
  softmax Jacobians are computed in fp32).
- `test_ns_bwd_kernel` tolerance loosened `1e-5` → `2e-4` (matching its sibling
  `test_ns_bwd_graph`): the fp32 cuBLAS Sgemm round-off in the trailing
  `dS2 @ k_tilde` matmul differs by GPU and reaches ~1.6e-5 at D=128 on A100
  (vs <1e-5 on consumer Blackwell) — fp32 round-off, not algorithmic error.
- Re-measured all latency tables (RTX 5060, A100, H100 vs cuBLAS, H100 vs
  FA2/FA3) at the current default config and regenerated the figures; corrected
  the `.tex` mislabel of the RTX 5060 as "SM89, Ada" (it is Blackwell sm_120).
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
