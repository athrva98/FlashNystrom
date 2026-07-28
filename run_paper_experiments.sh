#!/usr/bin/env bash
# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
#
# Every training experiment behind the paper, in the order they should run.
# Designed for one long-lived GPU node (g7e / A100-class), NOT for ephemeral
# workers: each stage writes to disk under runs/ and is resumable, so a
# disconnect costs only the run in flight.
#
#   bash run_paper_experiments.sh            # everything
#   bash run_paper_experiments.sh vision     # one stage
#   bash run_paper_experiments.sh mqar genomics
#
# Run under tmux/nohup; the full set is many GPU-hours.
set -u
STAGES="${*:-vision mqar genomics}"
mkdir -p runs logs

# All bidirectional-native arms. Only the attention operator differs between
# them; the backbone, optimizer, schedule, seeds and data are identical, so a
# difference in accuracy is attributable to the operator.
ARMS="sdpa linear_attention linformer sliding_window nystrom_reference flash_nystrom flash_nystrom_tc"
SEEDS="0 1 2"

have() { python -c "import $1" >/dev/null 2>&1; }
echo "=== dependency check (fused baselines) ==="
have flash_attn && echo "  flash-attn        OK (sliding_window fused)" \
                || echo "  flash-attn        MISSING -> sliding_window WILL FAIL:
                     pip install flash-attn --no-build-isolation"
have flash_bla  && echo "  flash_bla         OK (linear_attention fused)" \
                || echo "  flash_bla         MISSING -> linear_attention falls back to
                     the UNFUSED torch path, which is ~2x slower and NOT a fair
                     baseline. Install:
                     pip install -e git+https://github.com/fla-org/flash-bidirectional-linear-attention.git#egg=flash_bla"
have pyfaidx    && echo "  pyfaidx           OK (species task)" \
                || echo "  pyfaidx           MISSING -> species task WILL FAIL:
                     pip install pyfaidx"
have genomic_benchmarks && echo "  genomic-benchmarks OK" \
                || echo "  genomic-benchmarks MISSING -> that stage WILL FAIL:
                     pip install genomic-benchmarks"
echo

# --------------------------------------------------------------------------
# 1. VISION -- the primary accuracy evidence, all arms at every context tier.
#    CIFAR-10 N=65, then STL-10 at N=2304 / 9216 / 32401. Batch is pinned per
#    tier (NOT autobatched) so every arm sees the same batch and the comparison
#    is not confounded; the 32401 tier uses a 50% subset to bound wall-clock.
# --------------------------------------------------------------------------
if [[ " $STAGES " == *" vision "* ]]; then
  echo "########## VISION ##########"
  for seed in $SEEDS; do
    # dataset patch_size img_size epochs batch train_frac
    for cfg in \
      "cifar10 4 32   20 128 1.0" \
      "stl10   2 96   50 128 1.0" \
      "stl10   1 96   50 48  1.0" \
      "stl10   1 180  50 16  0.5"
    do
      set -- $cfg; ds=$1; ps=$2; img=$3; ep=$4; bs=$5; frac=$6
      tag="${ds}_p${ps}_i${img}_seed${seed}"
      out="runs/vision_${tag}.json"
      if [[ -f "$out" ]]; then echo "skip $out"; continue; fi
      echo "--- vision $tag ---"
      python -u benchmarks/train_three_way.py \
        --dataset "$ds" --patch_size "$ps" --img_size "$img" \
        --epochs "$ep" --batch_size "$bs" --train_frac "$frac" \
        --backends $ARMS --seed "$seed" --kappa_star 0 \
        --no-instrument --out_json "$out" 2>&1 | tee "logs/vision_${tag}.log"
    done
  done
fi

# --------------------------------------------------------------------------
# 2. MQAR -- the adversarial recall probe. paper_sweep is the single certified
#    driver; it is resumable (finished cells skip via their result JSON) and
#    writes summary.json for the paper table.
# --------------------------------------------------------------------------
if [[ " $STAGES " == *" mqar "* ]]; then
  echo "########## MQAR ##########"
  python -u -m paper.mqar.paper_sweep \
    --methods $ARMS --max_parallel 4 --out runs/mqar_paper \
    2>&1 | tee logs/mqar.log
  python -m paper.mqar.paper_sweep --collect_only --out runs/mqar_paper \
    2>&1 | tee logs/mqar_summary.log
fi

# --------------------------------------------------------------------------
# 3. GENOMICS -- second bidirectional domain, on established benchmarks.
#
#    a) species classification (HyenaDNA, NeurIPS 2023): real genomes,
#       chromosome-disjoint splits, at 1024 and 32768 bp. This is the
#       long-range evidence and the one that matters most.
#    b) Genomic Benchmarks (Grešová et al., BMC Genomic Data 2023): the
#       standard regulatory-element check that HyenaDNA and Caduceus both
#       report, on the two longest datasets in the suite.
#    c) the synthetic needle-retrieval diagnostic, kept only to separate
#       "operator cannot do this" from "model did not train".
#
#    Each carries its own validity gate on the sdpa arm (see run_genomics.py);
#    a stage that prints "!! INVALID" must not go in the paper.
#
#    (a) needs reference genomes first, roughly 1.5 GB at the default
#    --chroms_per_split 4:
#        python benchmarks/download_genomes.py --out data/genomes
# --------------------------------------------------------------------------
if [[ " $STAGES " == *" genomics "* ]]; then
  echo "########## GENOMICS ##########"

  if [[ ! -d data/genomes ]]; then
    echo "--- fetching reference genomes ---"
    python -u benchmarks/download_genomes.py --out data/genomes \
      2>&1 | tee logs/genomes_download.log
  fi

  for N in 1024 32768; do
    bs=32; [[ $N -ge 32768 ]] && bs=4
    echo "--- species N=$N (batch $bs) ---"
    python -u benchmarks/run_genomics.py --task species \
      --arms $ARMS --seeds $SEEDS --lrs 1e-4 3e-4 1e-3 3e-3 \
      --seq_len "$N" --batch_size "$bs" --epochs 20 \
      --n_train 32768 --n_test 4096 --species_dir data/genomes \
      --out runs/genomics 2>&1 | tee "logs/genomics_species_${N}.log"
  done

  echo "--- genomic benchmarks ---"
  python -u benchmarks/run_genomics.py --task genomic_benchmarks \
    --arms $ARMS --seeds $SEEDS --lrs 1e-4 3e-4 1e-3 3e-3 \
    --epochs 40 --batch_size 32 --out runs/genomics \
    2>&1 | tee logs/genomics_gb.log

  echo "--- synthetic diagnostic ---"
  python -u benchmarks/run_genomics.py --task repeat --variant pointer \
    --arms $ARMS --seeds $SEEDS --lrs 1e-4 3e-4 1e-3 3e-3 \
    --seq_len 2048 --epochs 40 --n_train 32768 --out runs/genomics \
    2>&1 | tee logs/genomics_repeat.log
fi

echo "=== done: results under runs/, logs under logs/ ==="
