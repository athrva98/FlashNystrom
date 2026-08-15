#!/usr/bin/env bash
# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
#
# One-command setup for a fresh GPU node (g7e.4xlarge / L40S, A100, or similar).
# Everything the paper's experiments need, in the order that fails fastest.
#
#   bash setup_g7e.sh          # deps, build, genomes, smoke, memory check
#   bash setup_g7e.sh --no-genomes
#
# Then, under tmux:
#   python run_all_paper_experiments.py --preset paper12 2>&1 | tee logs/sweep.log
#
# NOTE the absence of `| tail`. tail buffers the whole stream until the process
# exits, so a long run prints nothing and looks hung. Use tee alone.
set -euo pipefail
GENOMES=${GENOMES:-data/genomes}
DO_GENOMES=1
[[ "${1:-}" == "--no-genomes" ]] && DO_GENOMES=0

say() { printf '\n=== %s ===\n' "$1"; }

say "GPU"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
if [[ "$CC" -lt 80 ]]; then
  echo "!! compute capability $CC < 8.0; the fused kernels need Ampere or newer"; exit 1
fi

say "python deps"
pip install -q --upgrade pip
pip install -q einops ninja packaging pyfaidx genomic-benchmarks

say "flash_nystrom (--no-build-isolation: setup.py imports torch)"
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
export FLASH_NYSTROM_LAX_BUILD=1
pip install -e . --no-build-isolation 2>&1 | tail -2

say "baseline kernels (both are required for a FAIR comparison)"
# flash-attn compiles from source and is the slow step; MAX_JOBS bounds RAM so
# the build is not OOM-killed, which is the usual failure on a 16-vCPU node.
export MAX_JOBS=${MAX_JOBS:-4}
pip install flash-attn --no-build-isolation 2>&1 | tail -2 || \
  echo "!! flash-attn FAILED: the sliding_window arm cannot run"
pip install -q -e "git+https://github.com/fla-org/flash-bidirectional-linear-attention.git#egg=flash_bla" 2>&1 | tail -1 || \
  echo "!! flash_bla FAILED: linear_attention would run UNFUSED and is not a fair baseline"

say "dependency check"
python - <<'PY'
import importlib, sys
need = [("torch", "core"), ("flash_nystrom", "the kernels"),
        ("flash_attn", "sliding_window arm"),
        ("flash_bla", "linear_attention arm (unfused fallback is NOT fair)"),
        ("pyfaidx", "genomics species task"),
        ("genomic_benchmarks", "genomics regulatory task")]
bad = 0
for m, why in need:
    try:
        importlib.import_module(m); print(f"  {m:20s} OK       {why}")
    except Exception as e:
        print(f"  {m:20s} MISSING  {why}  ({type(e).__name__})"); bad += 1
sys.exit(0)
PY

say "kernel runs on this GPU"
python - <<'PY'
import torch
from flash_nystrom import flash_nystrom_attention
q = torch.randn(1, 2, 512, 64, device="cuda", dtype=torch.float16, requires_grad=True)
o = flash_nystrom_attention(q, q.clone(), q.clone(), num_landmarks=64)
o.sum().backward()
print(f"  fwd+bwd OK, finite={torch.isfinite(o).all().item()}")
PY

if [[ "$DO_GENOMES" == "1" ]]; then
  say "reference genomes (~1.5GB, skipped if already complete)"
  python benchmarks/download_genomes.py --out "$GENOMES" --chroms_per_split 4
fi

say "memory check at the sweep's real batch sizes"
python tools/memcheck.py || echo "!! reduce the flagged batch sizes before the sweep"

say "smoke: every stage, every arm, tiny budgets"
python run_all_paper_experiments.py --smoke --species_dir "$GENOMES"

cat <<'EOF'

=== setup done ===
If the smoke reported failures other than a missing optional kernel, fix them
before starting the sweep. Then, under tmux:

    python run_all_paper_experiments.py --preset paper12 --budget_hours 40 \
        2>&1 | tee logs/sweep.log

Progress: one line per job. Per-job detail is in logs/. Results are JSON under
runs/ and the sweep is resumable, so a disconnect costs only the run in flight.
Check what finished at any time, from another shell, with:

    python tools/inventory_runs.py --out runs

Budget. The paper12 grid was costed at ~13.5h on an A100-80GB. An L40S has
0.42x its memory bandwidth and 0.58x its fp16 tensor throughput, and this
workload is mostly memory-bound, so expect roughly 23-32h. That fits a two-day
window, but not with much to spare: --budget_hours 40 leaves margin and makes
the driver stop cleanly rather than be killed mid-run. Jobs run
highest-value-first, so anything dropped at the end is the least load-bearing.

If wall-clock gets tight, --preset minimal is the same grid without the 32401
vision tier (~2.8h on A100, so ~5-7h here). That tier runs one seed and carries
no error bar, and section 5.1 already scopes its claim accordingly, so dropping
it costs a column rather than a claim.
EOF
