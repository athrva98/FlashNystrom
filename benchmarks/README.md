# benchmarks/

Diagnostic, performance, and correctness scripts. These are **not** part of the
shipped package (the sdist excludes this directory); they are developer tools.
Most write to `C:/tmp/` or stdout and call `os._exit(0)` at the end to avoid the
Windows CUDA-teardown hang.

Run from the repo root in the build env, e.g. `python benchmarks/bench_paper.py`.

## Performance
| script | what it measures |
|---|---|
| `bench_paper.py` | the headline FN vs FA2/FA3 latency table for the paper/README |
| `bench_forward.py` / `bench_backward.py` / `bench_fwd_bwd.py` | per-pass latency sweeps |
| `profile_scaling.py` | per-kernel breakdown vs N/BH (uses `FLASH_NYSTROM_PROFILE`) |
| `autobatch.py` | autobatch cap behavior |
| `plot_benchmarks.py` / `make_figures.py` | turn the JSON/CSV outputs into plots |

## Correctness / numerics
| script | what it checks |
|---|---|
| `adversarial_sweep.py` | per-stage FN-vs-reference error across N/B/H/m/d/dtype/kappa/regime |
| `pinv_bwd_sweep.py` | the pinv backward is fp32-exact vs an autograd unroll across cond(K2) |
| `grad_bias.py` | FN gradients are zero-mean unbiased vs the fp16 reference (no systematic bias) |
| `measure_bwd_ranges.py` | magnitude ranges of every backward intermediate |
| `train_mnist_seeds.py` | paired multi-seed MNIST: FN vs reference (the real comparison instrument) |

## Training harnesses
`train_three_way.py` (SDPA / reference / FN on CIFAR/STL), `train_five_way.py`,
`train_long_context.py`. Use multi-seed runs for any FN-vs-reference comparison;
single seeds are inside the run-to-run noise band.

## Fixtures
`capture_k2inv_fixture.py` writes a real trained-activation fixture to
`C:/tmp/fn_real_fixture.pt` used by the sanitizer/diagnostic drivers.

## Historical (kept for reproducibility, not routine use)
These reproduce bugs that were diagnosed and fixed, or validate TC-pinv
development checkpoints. They are archaeology — not maintained as live tests.
- `repro_stl10_collapse.py`, `mnist_diverge.py`, `mnist_internal.py` — training
  divergence investigations (resolved: FN is unbiased vs the reference).
- `k2inv_test.py`, `k2inv_cp2/cp5/cp6_test.py`, `k2inv_gemm_test.py` — the
  tf32 tensor-core pinv development checkpoints.
- `_san_*.py` — compute-sanitizer (racecheck/memcheck) driver scripts.

The maintained correctness gate is `tests/` (pytest), not this directory.
