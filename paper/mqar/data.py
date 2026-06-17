# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
r"""Multi-Query Associative Recall (MQAR) data generation.

MQAR is the recall probe from Zoology (Arora et al., ICLR 2024,
arXiv:2312.04927). A sequence interleaves key-value pairs and then poses
queries: each query is a key seen earlier, and the model must produce the
value that key was bound to. "Multi-query" because several independent
recalls are required in a single sequence, at input-dependent positions.

This is a self-contained reimplementation of the Zoology construction (no
zoology dependency). It differs from Zoology's in exactly one respect: the
label sits at the query-key position itself (predict-in-place) rather than
shifted by one (next-token). FlashNystrom is a bidirectional attention with
no causal form, so the model reads the whole sequence and emits the value at
each query position; the full-attention baseline runs in the same
bidirectional mode for an apples-to-apples comparison. The recall task is
identical; only the autoregressive shift is dropped.

Layout of one length-`seq_len` example (context_size = 2 * num_kv_pairs):

    [ k0 v0 k1 v1 ... k_{m-1} v_{m-1} | <query region> ]
      \------------ context ---------/  \-- queries + distractors --/

Keys are drawn from [1, vocab//2), values from [vocab//2, vocab); token 0 is
a reserved blank that never survives in the inputs (blanks are overwritten by
random distractor tokens). Each of the `num_kv_pairs` keys is queried exactly
once, at a slot chosen by a (truncated) power law over the query region.
Labels are -100 everywhere except the query-key positions, so the loss and
the recall accuracy are computed only where a recall is actually required.
"""
from __future__ import annotations

import numpy as np
import torch


def _topk_without_replacement(rng, weights_log, k, num_examples):
    """Sample `k` distinct indices per example, proportional to exp(weights_log),
    via the Gumbel-top-k trick (Plackett-Luce). `weights_log` is the log-weight
    vector of length n (shared across examples). Returns int array (num_examples, k)."""
    n = weights_log.shape[0]
    gumbel = -np.log(-np.log(rng.uniform(size=(num_examples, n))))
    scores = weights_log[None, :] + gumbel
    # top-k indices (unordered within the top-k, which is fine here)
    return np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]


def generate_mqar(
    num_examples: int,
    vocab_size: int,
    seq_len: int,
    num_kv_pairs: int,
    power_a: float = 0.01,
    random_non_queries: bool = True,
    seed: int = 0,
):
    """Generate an MQAR dataset.

    Args:
        num_examples: number of sequences.
        vocab_size: token vocabulary size. Keys use the lower half, values the
            upper half; must be large enough to draw `num_kv_pairs` distinct
            keys and values.
        seq_len: total sequence length (even). Must satisfy
            ``num_kv_pairs * 4 <= seq_len``.
        num_kv_pairs: number of key-value bindings (and number of queries).
        power_a: shape of the truncated power law over query-slot positions.
            Smaller -> more mass on early slots (shorter recall distance);
            Zoology's default is 0.01.
        random_non_queries: fill non-query slots with random tokens instead of
            the blank 0, so the model cannot identify query positions by
            token presence alone.
        seed: RNG seed (deterministic).

    Returns:
        inputs:  LongTensor (num_examples, seq_len) of token ids.
        labels:  LongTensor (num_examples, seq_len); the bound value at each
                 query-key position, and -100 (ignore) elsewhere.
    """
    if seq_len % 2 != 0:
        raise ValueError(f"seq_len must be even, got {seq_len}")
    if num_kv_pairs * 4 > seq_len:
        raise ValueError(
            f"need num_kv_pairs*4 <= seq_len, got {num_kv_pairs}*4 > {seq_len}"
        )
    key_vocab_size = vocab_size // 2
    n_key_choices = key_vocab_size - 1          # tokens [1, key_vocab_size)
    n_val_choices = vocab_size - key_vocab_size  # tokens [key_vocab_size, vocab)
    if num_kv_pairs > min(n_key_choices, n_val_choices):
        raise ValueError(
            f"vocab too small: need >= {num_kv_pairs} distinct keys and values, "
            f"have {n_key_choices} key tokens and {n_val_choices} value tokens"
        )

    rng = np.random.default_rng(seed)
    context_size = num_kv_pairs * 2

    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    # Distinct keys and values per example (uniform without replacement via
    # argpartition of uniform noise == Gumbel-top-k with zero log-weights).
    zero_log_keys = np.zeros(n_key_choices)
    zero_log_vals = np.zeros(n_val_choices)
    key_idx = _topk_without_replacement(rng, zero_log_keys, num_kv_pairs, num_examples)
    val_idx = _topk_without_replacement(rng, zero_log_vals, num_kv_pairs, num_examples)
    keys = key_choices[key_idx]      # (num_examples, num_kv_pairs)
    values = value_choices[val_idx]  # (num_examples, num_kv_pairs)

    # Interleaved key-value context.
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # Query slots: choose `num_kv_pairs` distinct slots in the query region
    # under a truncated power law. Slot s places a query token at offset 2*s.
    region_len = seq_len - context_size
    n_slots = region_len // 2
    slot_rank = np.arange(1, n_slots + 1, dtype=np.float64)
    weights_log = (power_a - 1.0) * np.log(slot_rank) + np.log(power_a)
    slots = _topk_without_replacement(rng, weights_log, num_kv_pairs, num_examples)

    queries = np.zeros((num_examples, region_len), dtype=np.int64)
    region_labels = np.full((num_examples, region_len), -100, dtype=np.int64)
    np.put_along_axis(queries, slots * 2, keys, axis=1)
    np.put_along_axis(region_labels, slots * 2, values, axis=1)

    inputs = np.concatenate([kvs, queries], axis=1)
    context_labels = np.full((num_examples, context_size), -100, dtype=np.int64)
    labels = np.concatenate([context_labels, region_labels], axis=1)

    inputs_t = torch.from_numpy(inputs)
    labels_t = torch.from_numpy(labels)

    if random_non_queries:
        blank = inputs_t == 0
        # Random distractor tokens in [1, vocab); never 0 so the blank sentinel
        # is fully removed from the inputs.
        rand = torch.from_numpy(
            rng.integers(1, vocab_size, size=inputs_t.shape, dtype=np.int64)
        )
        inputs_t = torch.where(blank, rand, inputs_t)

    return inputs_t, labels_t


class MQARDataset(torch.utils.data.TensorDataset):
    """Thin TensorDataset wrapper that yields (input_ids, labels)."""

    def __init__(self, **kwargs):
        inputs, labels = generate_mqar(**kwargs)
        super().__init__(inputs, labels)
        self.inputs = inputs
        self.labels = labels
