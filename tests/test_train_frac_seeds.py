# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Pin down --train_frac seed resolution for the multi-seed 32K sweep.

The 32K experiment runs 3 seeds x 6 backends on a 50% STL-10 subset. Two
properties MUST hold or the whole run is worthless:
  * within one seed, every backend arm trains on the *same* subset (fair);
  * across seeds, the subsets genuinely differ (the 3-seed spread is real).
Both reduce to: subset_indices is a pure function of the seed. These tests
run on CPU, no CUDA, no dataset download.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))

from train_three_way import subset_indices

# The exact 32K config: STL-10 train = 5000, --train_frac 0.5, --batch_size 32.
N_FULL, FRAC, BATCH = 5000, 0.5, 32


def test_frac_ge_one_uses_everything():
    # frac >= 1.0 -> None sentinel = "use the whole set", never a subset.
    assert subset_indices(N_FULL, 1.0, BATCH, 0) is None
    assert subset_indices(N_FULL, 2.0, BATCH, 0) is None


def test_deterministic_same_seed():
    # Same (config, seed) -> byte-identical indices. This is what makes every
    # backend arm in a run see the same images.
    a = subset_indices(N_FULL, FRAC, BATCH, 7)
    b = subset_indices(N_FULL, FRAC, BATCH, 7)
    assert a == b


def test_different_seeds_differ():
    # Distinct seeds -> distinct subsets (not a reshuffle of the same images).
    s0 = set(subset_indices(N_FULL, FRAC, BATCH, 0))
    s1 = set(subset_indices(N_FULL, FRAC, BATCH, 1))
    s2 = set(subset_indices(N_FULL, FRAC, BATCH, 2))
    assert s0 != s1 and s1 != s2 and s0 != s2
    # Two independent 50% draws of 5000 overlap ~50% in expectation; assert the
    # overlap is nowhere near identical (guards against a seed that silently
    # does nothing, e.g. randperm ignoring the generator).
    overlap = len(s0 & s1) / len(s0)
    assert 0.35 < overlap < 0.65, f"overlap {overlap:.2f} not consistent with independent draws"


def test_size_is_frac_of_full():
    idx = subset_indices(N_FULL, FRAC, BATCH, 0)
    assert len(idx) == int(N_FULL * FRAC) == 2500


def test_indices_valid_and_unique():
    idx = subset_indices(N_FULL, FRAC, BATCH, 0)
    assert len(set(idx)) == len(idx), "indices must be unique (no repeated samples)"
    assert all(0 <= i < N_FULL for i in idx), "indices in range"


def test_batch_floor():
    # A frac so small it would select < one batch is clamped up to one batch,
    # so the DataLoader (drop_last=True) is never empty.
    idx = subset_indices(N_FULL, 0.0001, BATCH, 0)
    assert len(idx) == BATCH


def test_clamped_to_full():
    # frac just under 1 must never ask for more than n_full samples.
    idx = subset_indices(100, 0.999, BATCH, 0)
    assert len(idx) <= 100


def test_three_seed_sweep_all_distinct_correct_size():
    # End-to-end shape of the actual sweep: seeds 0,1,2 each give a distinct,
    # correctly-sized, valid subset.
    subs = [subset_indices(N_FULL, FRAC, BATCH, s) for s in (0, 1, 2)]
    for idx in subs:
        assert len(idx) == 2500
        assert len(set(idx)) == 2500
    assert len({tuple(s) for s in subs}) == 3, "all three seed subsets must be distinct"
