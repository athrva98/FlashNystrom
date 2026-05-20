#!/usr/bin/env bash
# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
#
# Profile FlashNystrom kernel3 vs the equivalent cuBLAS GEMMs with Nsight
# Compute on Linux (e.g. a Colab A100). Linux counterpart of profile_ncu.ps1.
#
# Usage:
#   tools/profile_ncu.sh [B H N D m]
#   PY=python NCU=/usr/local/cuda/bin/ncu tools/profile_ncu.sh 1 8 4096 128 64
#
# On Colab/Linux ncu usually has counter access already. If you hit
# ERR_NVGPUCTRPERM, load the nvidia module with NVreg_RestrictProfilingToAdminUsers=0
# or run ncu as root.
set -euo pipefail

B=${1:-1}; H=${2:-8}; N=${3:-4096}; D=${4:-128}; M=${5:-64}
PY=${PY:-python}
NCU=${NCU:-$(command -v ncu || echo /usr/local/cuda/bin/ncu)}
OUTDIR=${OUTDIR:-/tmp/fn_ncu}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOAD="$SCRIPT_DIR/ncu_workload.py"

[ -x "$NCU" ] || { echo "ncu not found ($NCU). Set NCU=/path/to/ncu"; exit 1; }
[ -f "$WORKLOAD" ] || { echo "workload not found: $WORKLOAD"; exit 1; }
mkdir -p "$OUTDIR"
REP="$OUTDIR/fn_vs_cublas"

echo "ncu:      $NCU"
echo "python:   $PY"
echo "workload: $WORKLOAD  (B=$B H=$H N=$N D=$D m=$M)"
echo "out:      $REP.ncu-rep"
echo

KERNEL_REGEX='regex:(kernel3_(partial|combine|fused)|gemm|cutlass|ampere|xmma|16816|elementwise)'

echo "Profiling... ncu replays each kernel to read counters; takes a few minutes."
"$NCU" --target-processes all \
       --nvtx --nvtx-include "prof_fn/" --nvtx-include "prof_cublas/" \
       --kernel-name "$KERNEL_REGEX" \
       --section SpeedOfLight --section Occupancy \
       --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
       --section WarpStateStats \
       --force-overwrite --export "$REP" \
       "$PY" "$WORKLOAD" "$B" "$H" "$N" "$D" "$M"

echo
echo "==================== SUMMARY (paste this back) ===================="
"$NCU" --import "$REP.ncu-rep" --page details | \
    grep -E "void flash_nystrom|gemm|cutlass|ampere|Duration|Compute \(SM\)|Memory Throughput|Achieved Occupancy|Tensor|DRAM Throughput|Stall" || true
echo
echo "Full report: $REP.ncu-rep"
