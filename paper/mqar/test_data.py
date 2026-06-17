# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Correctness tests for the MQAR generator.

These verify the data is a valid associative-recall problem: every query is a
key that was bound in the context, every label is the value that key was bound
to, and loss is masked everywhere a recall is not required. Run with:

    pytest paper/mqar/test_data.py
"""
import numpy as np
import pytest
import torch

from paper.mqar.data import generate_mqar

CFG = dict(vocab_size=512, seq_len=128, num_kv_pairs=8)


def _context_binding(inputs_row, num_kv_pairs):
    """Return the {key: value} dict encoded in the interleaved context prefix."""
    ctx = inputs_row[: num_kv_pairs * 2].tolist()
    keys, vals = ctx[0::2], ctx[1::2]
    return dict(zip(keys, vals))


def test_shapes_and_dtype():
    x, y = generate_mqar(num_examples=64, seed=0, **CFG)
    assert x.shape == (64, CFG["seq_len"])
    assert y.shape == (64, CFG["seq_len"])
    assert x.dtype == torch.int64 and y.dtype == torch.int64


def test_disjoint_key_value_vocab():
    x, _ = generate_mqar(num_examples=64, seed=1, **CFG)
    half = CFG["vocab_size"] // 2
    m = CFG["num_kv_pairs"]
    ctx = x[:, : 2 * m]
    keys, vals = ctx[:, 0::2], ctx[:, 1::2]
    assert (keys >= 1).all() and (keys < half).all()
    assert (vals >= half).all() and (vals < CFG["vocab_size"]).all()


def test_keys_and_values_distinct_per_example():
    x, _ = generate_mqar(num_examples=128, seed=2, **CFG)
    m = CFG["num_kv_pairs"]
    for row in x:
        keys = row[: 2 * m][0::2].tolist()
        vals = row[: 2 * m][1::2].tolist()
        assert len(set(keys)) == m, "context keys must be distinct"
        assert len(set(vals)) == m, "context values must be distinct"


def test_num_queries_equals_num_kv_pairs():
    m = CFG["num_kv_pairs"]
    x, y = generate_mqar(num_examples=128, seed=3, **CFG)
    n_queries = (y != -100).sum(dim=1)
    assert (n_queries == m).all(), "each key must be queried exactly once"


def test_context_labels_are_ignored():
    m = CFG["num_kv_pairs"]
    _, y = generate_mqar(num_examples=64, seed=4, **CFG)
    assert (y[:, : 2 * m] == -100).all(), "no labels inside the context prefix"


def test_every_query_recalls_the_correct_value():
    """The core property: at each query position, the input token is a context
    key and the label equals the value that key was bound to."""
    m = CFG["num_kv_pairs"]
    x, y = generate_mqar(num_examples=256, seed=5, **CFG)
    for row_x, row_y in zip(x, y):
        binding = _context_binding(row_x, m)
        q_positions = (row_y != -100).nonzero(as_tuple=True)[0]
        seen_keys = set()
        for p in q_positions.tolist():
            key = int(row_x[p])
            val = int(row_y[p])
            assert key in binding, "query token must be a bound key"
            assert binding[key] == val, "label must be the bound value"
            seen_keys.add(key)
        assert seen_keys == set(binding.keys()), "every bound key is queried once"


def test_no_blank_tokens_when_random_non_queries():
    x, _ = generate_mqar(num_examples=64, seed=6, random_non_queries=True, **CFG)
    assert (x != 0).all(), "blanks must be replaced by random distractors"


def test_blank_tokens_present_when_disabled():
    x, _ = generate_mqar(num_examples=64, seed=6, random_non_queries=False, **CFG)
    assert (x == 0).any(), "blank slots should remain when random fill is off"


def test_query_positions_on_even_offsets():
    """Queries are placed at even offsets within the query region."""
    m = CFG["num_kv_pairs"]
    ctx = 2 * m
    _, y = generate_mqar(num_examples=64, seed=7, **CFG)
    for row_y in y:
        offsets = (row_y[ctx:] != -100).nonzero(as_tuple=True)[0]
        assert (offsets % 2 == 0).all(), "query slots are at even offsets"


def test_determinism():
    a = generate_mqar(num_examples=32, seed=11, **CFG)
    b = generate_mqar(num_examples=32, seed=11, **CFG)
    c = generate_mqar(num_examples=32, seed=12, **CFG)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0]), "different seeds must differ"


@pytest.mark.parametrize(
    "bad",
    [
        dict(vocab_size=512, seq_len=127, num_kv_pairs=8),   # odd seq_len
        dict(vocab_size=512, seq_len=128, num_kv_pairs=33),  # num_kv_pairs*4 > seq_len
        dict(vocab_size=16, seq_len=128, num_kv_pairs=8),    # vocab too small for distinct keys
    ],
)
def test_invalid_args_raise(bad):
    with pytest.raises(ValueError):
        generate_mqar(num_examples=4, seed=0, **bad)


def test_harder_config_still_valid():
    """A larger, longer-recall config still satisfies every invariant."""
    cfg = dict(vocab_size=8192, seq_len=512, num_kv_pairs=64, power_a=0.01)
    x, y = generate_mqar(num_examples=64, seed=9, **cfg)
    for row_x, row_y in zip(x, y):
        binding = _context_binding(row_x, cfg["num_kv_pairs"])
        for p in (row_y != -100).nonzero(as_tuple=True)[0].tolist():
            assert binding[int(row_x[p])] == int(row_y[p])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
