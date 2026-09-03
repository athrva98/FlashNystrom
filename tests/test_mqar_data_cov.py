# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Exhaustive coverage of paper.mqar.data (MQAR dataset generation).

Pure numpy/torch, CPU, fully deterministic. Covers _topk_without_replacement,
generate_mqar (shapes, dtypes, every guard, token ranges, label semantics,
recall correctness, determinism, random_non_queries), and MQARDataset.
"""
import numpy as np
import pytest
import torch

from paper.mqar.data import (
    _topk_without_replacement,
    generate_mqar,
    MQARDataset,
)

# --------------------------------------------------------------------------- #
# _topk_without_replacement
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n,k,ne", [(10, 3, 5), (50, 10, 8), (4, 4, 2), (100, 1, 20)])
def test_topk_shape(n, k, ne):
    rng = np.random.default_rng(0)
    out = _topk_without_replacement(rng, np.zeros(n), k, ne)
    assert out.shape == (ne, k)


@pytest.mark.parametrize("n,k,ne", [(10, 3, 5), (50, 10, 8), (20, 20, 3)])
def test_topk_distinct_per_row(n, k, ne):
    rng = np.random.default_rng(1)
    out = _topk_without_replacement(rng, np.zeros(n), k, ne)
    for row in out:
        assert len(set(row.tolist())) == k  # no repeats within an example


@pytest.mark.parametrize("n,k", [(10, 3), (32, 8), (7, 7)])
def test_topk_indices_in_range(n, k):
    rng = np.random.default_rng(2)
    out = _topk_without_replacement(rng, np.zeros(n), k, 6)
    assert out.min() >= 0 and out.max() < n


def test_topk_weights_bias_toward_high_logweight():
    # with a strong log-weight on index 0, it should be selected almost always
    rng = np.random.default_rng(3)
    w = np.full(20, -10.0)
    w[0] = 100.0
    out = _topk_without_replacement(rng, w, 1, 200)
    frac0 = (out[:, 0] == 0).mean()
    assert frac0 > 0.99


def test_topk_deterministic_same_rng_seed():
    a = _topk_without_replacement(np.random.default_rng(7), np.zeros(15), 4, 10)
    b = _topk_without_replacement(np.random.default_rng(7), np.zeros(15), 4, 10)
    assert np.array_equal(a, b)

# --------------------------------------------------------------------------- #
# generate_mqar: shapes / dtypes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ne", [1, 4, 16, 64])
@pytest.mark.parametrize("seq_len,kv", [(64, 4), (128, 8), (256, 16), (32, 4)])
def test_gen_shapes(ne, seq_len, kv):
    x, y = generate_mqar(ne, vocab_size=128, seq_len=seq_len, num_kv_pairs=kv, seed=0)
    assert x.shape == (ne, seq_len)
    assert y.shape == (ne, seq_len)


@pytest.mark.parametrize("seq_len,kv", [(64, 4), (128, 8)])
def test_gen_dtypes_long(seq_len, kv):
    x, y = generate_mqar(4, 128, seq_len, kv, seed=0)
    assert x.dtype == torch.int64 and y.dtype == torch.int64

# --------------------------------------------------------------------------- #
# generate_mqar: input guards
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seq_len", [63, 65, 1, 127])
def test_gen_odd_seq_len_rejected(seq_len):
    with pytest.raises(ValueError, match="seq_len must be even"):
        generate_mqar(2, 128, seq_len, 4, seed=0)


@pytest.mark.parametrize("seq_len,kv", [(16, 8), (32, 16), (64, 20)])
def test_gen_too_many_kv_rejected(seq_len, kv):
    with pytest.raises(ValueError, match="num_kv_pairs"):
        generate_mqar(2, 512, seq_len, kv, seed=0)


@pytest.mark.parametrize("vocab,kv", [(8, 8), (10, 32), (6, 4)])
def test_gen_vocab_too_small_rejected(vocab, kv):
    # seq_len large enough that the kv*4 guard passes, so the vocab guard fires
    with pytest.raises(ValueError, match="vocab too small"):
        generate_mqar(2, vocab, seq_len=kv * 8, num_kv_pairs=kv, seed=0)

# --------------------------------------------------------------------------- #
# generate_mqar: token ranges
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("vocab", [64, 128, 256])
def test_gen_context_key_value_ranges(vocab):
    # random_non_queries=False keeps the structural layout visible (blanks = 0)
    x, y = generate_mqar(8, vocab, seq_len=128, num_kv_pairs=8,
                         random_non_queries=False, seed=0)
    ctx = x[:, :16]  # context_size = 2*kv
    keys = ctx[:, 0::2]
    vals = ctx[:, 1::2]
    half = vocab // 2
    assert (keys >= 1).all() and (keys < half).all()
    assert (vals >= half).all() and (vals < vocab).all()


@pytest.mark.parametrize("kv", [1, 2, 4, 8, 16])
def test_gen_num_labels_equals_kv(kv):
    x, y = generate_mqar(10, 256, seq_len=kv * 8, num_kv_pairs=kv, seed=1)
    per_example = (y != -100).sum(dim=1)
    assert (per_example == kv).all()  # exactly one query per kv pair


def test_gen_context_region_all_ignore():
    kv = 8
    x, y = generate_mqar(6, 128, seq_len=128, num_kv_pairs=kv, seed=2)
    assert (y[:, : 2 * kv] == -100).all()  # no labels in the kv-context region

# --------------------------------------------------------------------------- #
# generate_mqar: recall semantics (the actual task must be well-formed)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("kv", [2, 4, 8])
def test_gen_recall_is_consistent(seed, kv):
    # At each labelled (query) position, the input token is a key from the
    # context and the label is exactly the value that key was bound to.
    ne, seq_len, vocab = 12, kv * 8, 256
    x, y = generate_mqar(ne, vocab, seq_len, kv, random_non_queries=False, seed=seed)
    ctx = 2 * kv
    for e in range(ne):
        bind = {int(x[e, 2 * i]): int(x[e, 2 * i + 1]) for i in range(kv)}
        qpos = (y[e] != -100).nonzero(as_tuple=True)[0]
        assert len(qpos) == kv
        for p in qpos.tolist():
            key_at_p = int(x[e, p])
            assert key_at_p in bind, "query token must be a context key"
            assert int(y[e, p]) == bind[key_at_p]


@pytest.mark.parametrize("kv", [2, 4, 8])
def test_gen_queries_live_in_query_region(kv):
    x, y = generate_mqar(8, 256, kv * 8, kv, seed=5)
    ctx = 2 * kv
    qpos = (y != -100)
    assert not qpos[:, :ctx].any()  # every labelled position is past the context

# --------------------------------------------------------------------------- #
# generate_mqar: determinism
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", [0, 1, 42, 123])
def test_gen_deterministic_same_seed(seed):
    a = generate_mqar(8, 128, 128, 8, seed=seed)
    b = generate_mqar(8, 128, 128, 8, seed=seed)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_gen_different_seeds_differ():
    a = generate_mqar(8, 128, 128, 8, seed=0)
    b = generate_mqar(8, 128, 128, 8, seed=1)
    assert not torch.equal(a[0], b[0])

# --------------------------------------------------------------------------- #
# generate_mqar: random_non_queries
# --------------------------------------------------------------------------- #

def test_gen_random_non_queries_removes_blanks():
    x, y = generate_mqar(16, 128, 128, 8, random_non_queries=True, seed=0)
    assert (x != 0).all()  # no reserved-blank tokens survive


def test_gen_no_random_leaves_blanks_in_region():
    kv = 8
    x, y = generate_mqar(16, 128, 128, kv, random_non_queries=False, seed=0)
    # the query region has blanks (0) at the non-query slots
    region = x[:, 2 * kv :]
    assert (region == 0).any()


def test_gen_random_non_queries_keeps_labels_identical():
    # distractor fill must not change WHERE/what the labels are
    a = generate_mqar(8, 128, 128, 8, random_non_queries=True, seed=9)
    b = generate_mqar(8, 128, 128, 8, random_non_queries=False, seed=9)
    assert torch.equal(a[1], b[1])  # labels identical regardless of fill


@pytest.mark.parametrize("power_a", [0.001, 0.01, 0.1, 0.5, 1.0])
def test_gen_power_a_variants_valid(power_a):
    x, y = generate_mqar(8, 128, 128, 8, power_a=power_a, seed=0)
    assert x.shape == (8, 128) and (y != -100).sum() == 8 * 8

# --------------------------------------------------------------------------- #
# MQARDataset
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ne", [1, 4, 32])
def test_dataset_len(ne):
    ds = MQARDataset(num_examples=ne, vocab_size=128, seq_len=128, num_kv_pairs=8, seed=0)
    assert len(ds) == ne


def test_dataset_getitem_returns_input_label_pair():
    ds = MQARDataset(num_examples=8, vocab_size=128, seq_len=128, num_kv_pairs=8, seed=0)
    x, y = ds[0]
    assert x.shape == (128,) and y.shape == (128,)
    assert x.dtype == torch.int64 and y.dtype == torch.int64


def test_dataset_exposes_inputs_labels_attrs():
    ds = MQARDataset(num_examples=8, vocab_size=128, seq_len=128, num_kv_pairs=8, seed=0)
    assert ds.inputs.shape == (8, 128) and ds.labels.shape == (8, 128)


def test_dataset_matches_generate_mqar():
    kw = dict(num_examples=8, vocab_size=128, seq_len=128, num_kv_pairs=8, seed=3)
    ds = MQARDataset(**kw)
    x, y = generate_mqar(**kw)
    assert torch.equal(ds.inputs, x) and torch.equal(ds.labels, y)
