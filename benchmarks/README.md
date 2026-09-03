# benchmarks/

Diagnostic, performance, and correctness scripts. These are **not** part of the
shipped package (the sdist excludes this directory); they are developer tools.
Run from the repo root in the build env, e.g. `python benchmarks/bench_paper.py`.

For the paper's experiments do **not** run these directly. There is one driver:

```
python run_all_paper_experiments.py --smoke      # first: minutes, finds bugs
python run_all_paper_experiments.py --preset paper12
```

## Paper experiments (driven by `run_all_paper_experiments.py`)
| script | role |
|---|---|
| `train_three_way.py` | the vision harness: every arm on CIFAR-10 / STL-10 at four context tiers |
| `run_genomics.py` | the genomics driver: species classification, Genomic Benchmarks, and the synthetic control |
| `genomics.py` / `genomics_data.py` | genomics models and datasets, with their provenance |
| `download_genomes.py` | fetches the Ensembl chromosomes the species task needs |
| `baseline_attn.py` | the bidirectional baselines as `nn.Module`s, shared by every harness |
| `baseline_ops.py` | the same operators bare, for latency measurement |

## Performance
| script | what it measures |
|---|---|
| `bench_bidir_latency.py` | the paper's bidirectional-operator table, every arm timed end to end |
| `bench_paper.py` | the headline FN vs FA2/FA3 latency table |
| `bench_fwd_bwd.py` | per-pass latency sweeps |
| `profile_scaling.py` | per-kernel breakdown vs N/BH (uses `FLASH_NYSTROM_PROFILE`) |
| `autobatch.py` | autobatch cap behavior |
| `bench_memory.py` | peak memory: FN vs the same algorithm unfused, vs exact attention |
| `bench_5060_refresh.py` | local FN / SDPA / cuBLAS latency sweep at the current defaults |
| `triton_nystrom.py` | hand-written Triton Nystrom forward, the compiler-baseline arm |
| `plot_benchmarks.py` / `make_figures.py` | turn the JSON/CSV outputs into plots |

## Correctness / numerics
These back specific claims in the paper; keep them runnable.

| script | what it checks |
|---|---|
| `adversarial_sweep.py` | per-stage FN-vs-reference error across N/B/H/m/d/dtype/kappa/regime |
| `pinv_bwd_sweep.py` | the pinv backward is fp32-exact vs an autograd unroll across cond(K2) |
| `grad_bias.py` | FN gradients are zero-mean unbiased vs the fp16 reference |
| `measure_bwd_ranges.py` | magnitude ranges of every backward intermediate |
| `train_mnist_seeds.py` | paired multi-seed MNIST: FN vs reference, the comparison instrument |
| `train_five_way.py` | splits FN forward from FN backward against the reference, to localize a discrepancy to one side |

Use multi-seed runs for any FN-vs-reference comparison; single seeds sit inside
the run-to-run noise band.

## Removed
One-off scripts for bugs that are now fixed and documented (the MNIST
divergence, the STL-10 collapse, the tensor-core pinv development checkpoints
and their fixture, the MQAR diagnose drivers, the Modal harnesses superseded by
`run_all_paper_experiments.py` and `bench_bidir_latency.py`) were deleted rather
than kept as archaeology. They are in git history if a result ever needs
re-deriving.
