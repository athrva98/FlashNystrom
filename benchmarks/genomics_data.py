# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Genomic datasets for the bidirectional operator comparison.

DNA is bidirectional in the strict sense this paper cares about: a regulatory
element's effect does not depend on a scan order, and the sequence is read in
both directions, so a causal mask discards half the context for no modelling
reason. Caduceus (Schiff et al., ICML 2024) makes exactly this argument and
builds a bi-directional DNA model on it; that is the regime our operator
targets, so genomics is the natural second domain after vision.

Three tasks, in descending order of how much they should be trusted:

1. ``species``      -- HyenaDNA's species classification (Nguyen et al.,
                       NeurIPS 2023). Real genomes, chromosome-disjoint splits,
                       evaluated at 1024 and 32768 bp. This is the long-range
                       task and the primary evidence.
2. ``genomic_benchmarks`` -- the Grešová et al. (BMC Genomic Data 2023) suite,
                       the standard short-range regulatory check that both
                       HyenaDNA (Table 4.1) and Caduceus report. Real data,
                       cheap, certified splits shipped with the package.
3. ``repeat``       -- a synthetic needle-retrieval diagnostic. NOT evidence
                       about genomics; a controlled probe with a known 100%
                       ceiling, used to separate "the operator cannot do this"
                       from "the model did not train".

Protocol sources are quoted at each definition. Where we deviate we say so.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

# --------------------------------------------------------------------------- #
# tokenizer: single-nucleotide, the convention for DNA LMs (HyenaDNA, Caduceus,
# Nucleotide Transformer all tokenize at or near base resolution). Vocabulary
# is the four bases plus N for unknown/masked positions.
# --------------------------------------------------------------------------- #

DNA_VOCAB = "NACGT"
DNA_VOCAB_SIZE = len(DNA_VOCAB)
_BASE_TO_IDX = {b: i for i, b in enumerate(DNA_VOCAB)}

# Lookup table over the full byte range so tokenization is a single index_select
# instead of a Python loop. Anything not ACGT (including Ensembl's soft-masked
# lowercase and IUPAC ambiguity codes) maps to N.
_BYTE_LUT = torch.zeros(256, dtype=torch.long)
for _b, _i in _BASE_TO_IDX.items():
    _BYTE_LUT[ord(_b)] = _i
    _BYTE_LUT[ord(_b.lower())] = _i


def tokenize_dna(seq: str) -> torch.Tensor:
    """ACGT string -> LongTensor of token ids. Non-ACGT collapses to N."""
    # copy(): frombuffer aliases the immutable bytes object, which torch warns
    # about and which would let a downstream in-place op corrupt it.
    raw = torch.frombuffer(bytearray(seq, "ascii"), dtype=torch.uint8).long()
    return _BYTE_LUT[raw]


# --------------------------------------------------------------------------- #
# 1. species classification -- HyenaDNA, NeurIPS 2023
# --------------------------------------------------------------------------- #

# Verbatim from HyenaDNA's src/dataloaders/datasets/species_dataset.py. The
# split is by CHROMOSOME, not by window, so a test window cannot overlap any
# training window: the model must generalize across chromosomes rather than
# memorize loci. Only the species we can actually obtain are kept here; the
# lists themselves are unmodified.
SPECIES_CHROMOSOME_SPLITS = {
    "human": {
        "train": ["2", "4", "6", "8", "14", "15", "16", "17", "18", "19",
                  "20", "21", "22", "X", "Y"],
        "valid": ["1", "3", "12", "13"],
        "test": ["5", "7", "9", "10", "11"],
    },
    "lemur": {
        "train": ["2", "4", "6", "8", "14", "15", "16", "17", "18", "19",
                  "20", "21", "22", "23", "24", "25", "26", "27", "X", "Y"],
        "valid": ["1", "3", "12", "13"],
        "test": ["5", "7", "9", "10", "11"],
    },
    "goat": {
        "train": ["2", "4", "6", "8", "14", "15", "16", "17", "18", "19",
                  "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
                  "X", "Y"],
        "valid": ["1", "3", "12", "13"],
        "test": ["5", "7", "9", "10", "11"],
    },
    "pig": {
        "train": ["2", "4", "6", "8", "14", "15", "16", "17", "18", "X", "Y"],
        "valid": ["1", "3", "12", "13"],
        "test": ["5", "7", "9", "10", "11"],
    },
    "mouse": {
        "train": ["2", "4", "6", "8", "14", "15", "16", "17", "18", "19", "X"],
        "valid": ["1", "3", "12", "13"],
        "test": ["5", "7", "9", "10", "11"],
    },
}

# HyenaDNA's headline 5-way task is human/lemur/mouse/pig/hippo. Hippo has no
# chromosome-level Ensembl assembly, so goat stands in for it: goat is in
# HyenaDNA's own split table above, which keeps the substitution inside their
# design rather than inventing one. The other four are exactly theirs.
DEFAULT_SPECIES = ["human", "lemur", "mouse", "pig", "goat"]

# Verified against https://ftp.ensembl.org/pub/current_fasta/ . Pig and sheep
# publish "primary_assembly" rather than "chromosome"; the others use
# "chromosome". Downloading is handled by download_genomes.py.
ENSEMBL_SOURCES = {
    "human": ("homo_sapiens", "Homo_sapiens.GRCh38.dna.chromosome.{c}.fa.gz"),
    "lemur": ("microcebus_murinus",
              "Microcebus_murinus.Mmur_3.0.dna.chromosome.{c}.fa.gz"),
    "mouse": ("mus_musculus", "Mus_musculus.GRCm39.dna.chromosome.{c}.fa.gz"),
    "pig": ("sus_scrofa",
            "Sus_scrofa.Sscrofa11.1.dna.primary_assembly.{c}.fa.gz"),
    "goat": ("capra_hircus", "Capra_hircus.ARS1.dna.chromosome.{c}.fa.gz"),
}


class SpeciesDataset(torch.utils.data.Dataset):
    """Classify which species a random genomic window came from.

    Reimplements HyenaDNA's ``SpeciesDataset.__getitem__`` sampling exactly:
    pick a species uniformly, pick one of that species' chromosomes for this
    split, pick a uniform random start, take a fixed-length window, and label
    it with the species index. Windows are drawn on the fly rather than fixed
    in advance, which is sound here (unlike a synthetic task) because the
    underlying genome is a fixed finite object and train/test are disjoint
    chromosomes: fresh sampling cannot leak test data.

    Two documented deviations from the reference implementation:

    * HyenaDNA returns ``seq[:-1]`` because the same class also serves
      next-token prediction, where the last token is the final target. For
      classification that truncation is an artifact, so we sample
      ``seq_len + 1`` bases and drop the last, giving the model exactly
      ``seq_len`` tokens while keeping their window arithmetic.
    * ``chroms_per_split`` subsamples how many chromosomes per split are held
      open, to bound disk (a full 5-species set is ~12 GB uncompressed). The
      train/test chromosome BOUNDARY is untouched, which is the part that
      makes the split a generalization test.
    """

    def __init__(self, species_dir: str, split: str, seq_len: int,
                 total_size: int, species: Optional[list] = None,
                 chroms_per_split: int = 4, seed: int = 0,
                 rc_aug: bool = False):
        try:
            from pyfaidx import Fasta
        except ImportError as e:                              # pragma: no cover
            raise ImportError(
                "the species task needs pyfaidx: pip install pyfaidx") from e

        self.species = list(species or DEFAULT_SPECIES)
        self.seq_len = seq_len
        self.total_size = total_size
        self.split = split
        self.rc_aug = rc_aug
        self.seed = seed

        self.fastas: dict[str, list] = {}
        for spec in self.species:
            if spec not in SPECIES_CHROMOSOME_SPLITS:
                raise ValueError(f"unknown species {spec!r}; known: "
                                 f"{sorted(SPECIES_CHROMOSOME_SPLITS)}")
            chroms = SPECIES_CHROMOSOME_SPLITS[spec][split][:chroms_per_split]
            handles = []
            for c in chroms:
                path = os.path.join(species_dir, spec, f"{c}.fa")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"missing {path}. Fetch it with:\n"
                        f"  python benchmarks/download_genomes.py "
                        f"--species {spec} --out {species_dir}")
                fa = Fasta(path)
                handles.append(fa[list(fa.keys())[0]])
            if not handles:
                raise ValueError(f"no chromosomes for {spec}/{split}")
            self.fastas[spec] = handles

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Deterministic per index so an epoch is reproducible across arms:
        # every arm sees the identical stream of windows, which is what makes
        # an accuracy difference attributable to the operator.
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)

        si = int(torch.randint(len(self.species), (1,), generator=g))
        spec = self.species[si]
        handles = self.fastas[spec]
        ci = int(torch.randint(len(handles), (1,), generator=g))
        fasta = handles[ci]

        n = len(fasta)
        span = self.seq_len + 1          # sample one extra, drop the last
        right = max(n - span, 0)
        start = int(torch.randint(right + 1, (1,), generator=g))
        seq = str(fasta[start:start + span])
        seq = seq.rjust(span, "N")       # HyenaDNA's short-window padding

        tok = tokenize_dna(seq)[:self.seq_len]
        if self.rc_aug and bool(torch.randint(2, (1,), generator=g)):
            tok = reverse_complement(tok)
        return tok, si


def reverse_complement(tok: torch.Tensor) -> torch.Tensor:
    """Reverse-complement in token space. DNA_VOCAB is 'NACGT', so A<->T and
    C<->G is the permutation [0, 4, 3, 2, 1] applied to a reversed sequence.
    Caduceus builds RC equivariance into the architecture; we only expose it as
    optional augmentation, identically available to every arm."""
    perm = torch.tensor([0, 4, 3, 2, 1], dtype=torch.long, device=tok.device)
    return perm[tok.flip(-1)]


# --------------------------------------------------------------------------- #
# 2. Genomic Benchmarks -- Grešová et al., BMC Genomic Data 2023
# --------------------------------------------------------------------------- #

# Median lengths from Table 1 of the paper. The two long ones are the only
# members where a sub-quadratic operator has anything to prove; the 200-500 bp
# datasets are included because they are the standard check that a DNA model
# learns real regulatory signal at all.
GB_DATASETS = {
    "dummy_mouse_enhancers_ensembl": dict(median_len=2381, classes=2, n=1210),
    "drosophila_enhancers_stark": dict(median_len=2142, classes=2, n=6914),
    "human_enhancers_cohn": dict(median_len=500, classes=2, n=27791),
    "human_ensembl_regulatory": dict(median_len=401, classes=3, n=289061),
    "human_nontata_promoters": dict(median_len=251, classes=2, n=36131),
    "human_ocr_ensembl": dict(median_len=315, classes=2, n=174756),
    "human_enhancers_ensembl": dict(median_len=269, classes=2, n=154842),
    "demo_coding_vs_intergenomic_seqs": dict(median_len=200, classes=2, n=100000),
    "demo_human_or_worm": dict(median_len=200, classes=2, n=100000),
}

# Default to the two longest. They are the ones whose lengths make the operator
# choice matter, and both are real regulatory-element tasks.
DEFAULT_GB = ["dummy_mouse_enhancers_ensembl", "drosophila_enhancers_stark"]


def load_genomic_benchmark(name: str, split: str, max_len: Optional[int] = None):
    """Load one Genomic Benchmarks dataset through its official package.

    Uses the shipped train/test split verbatim (``get_dataset`` from
    ``genomic_benchmarks.dataset_getters.pytorch_datasets``), which is the
    split HyenaDNA and Caduceus report against. Sequences are variable length
    in several datasets, so they are right-padded with N to a common length and
    a padding mask is returned; padding to the dataset max rather than
    truncating avoids silently discarding the evidence the label depends on.
    """
    # Name first: a typo should say so whether or not the package is installed.
    if name not in GB_DATASETS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(GB_DATASETS)}")

    try:
        from genomic_benchmarks.dataset_getters.pytorch_datasets import get_dataset
    except ImportError as e:                                  # pragma: no cover
        raise ImportError(
            "pip install genomic-benchmarks  (official package for the "
            "Grešová et al. suite)") from e

    dset = get_dataset(name, split=split, version=0)
    seqs, labels = [], []
    for seq, y in dset:
        seqs.append(tokenize_dna(seq))
        labels.append(int(y))

    n = max_len or max(int(s.numel()) for s in seqs)
    x = torch.zeros(len(seqs), n, dtype=torch.long)     # 0 == N == pad
    mask = torch.zeros(len(seqs), n, dtype=torch.bool)
    for i, s in enumerate(seqs):
        k = min(int(s.numel()), n)
        x[i, :k] = s[:k]
        mask[i, :k] = True
    return x, torch.tensor(labels, dtype=torch.long), mask


# --------------------------------------------------------------------------- #
# 3. synthetic needle retrieval -- a diagnostic, not evidence
# --------------------------------------------------------------------------- #

KMER = 6                      # DNABERT-style k-mer tokens
KMER_VOCAB = 4 ** KMER        # 4096


def synth_repeat_dataset(num_examples: int, seq_len: int, seed: int = 0,
                         variant: str = "pointer"):
    """Long-range needle retrieval over k-mer tokens, with a 100% ceiling.

    An earlier version of this task was ill-posed and could not be trained to
    its own validity gate. Both defects are fixed here.

    FIX 1 -- the ceiling. The background was drawn uniformly over all 4096
    k-mers, so the query token recurred BY CHANCE in a negative with
    probability 1-(1-1/V)^(L-1): 12.7% at L=512 but 37.9% at L=2048 and 63.8%
    at L=4096. The label ("was a match planted") and the only observable ("does
    the query recur") therefore disagreed on that fraction, capping a perfect
    matcher at 81.1% at L=2048 while the driver gated validity at 85%. The
    experiment could not pass. The background now excludes the query token:
    drawing uniformly from [0, V-2) and adding 1 to any draw at or above the
    query gives an exactly uniform distribution over the other V-1 tokens with
    no rejection loop, so the planted match is the ONLY match and the ceiling
    is 100% at every length.

    FIX 2 -- the supervision. One binary label per sequence is ~1 bit, against
    the 64 x log2(8192) = 832 bits per sequence that MQAR supplies for the same
    induction-style matching circuit. That is 832x less signal per sequence and
    ~3300x less per token, which is why no arm left chance. The default
    ``pointer`` variant instead asks WHERE the match is: a softmax over
    positions, log2(L) = 11 bits at L=2048, with chance at 1/L rather than 1/2
    so any real signal is unmistakable.

    variant="pointer": returns (x, pos). Every sequence contains exactly one
        recurrence of x[:, 0] and the target is its index. Chance = 1/(L-1).
    variant="detect":  returns (x, y) binary, the original framing but now with
        a 100% ceiling. Chance = 1/2.
    """
    if variant not in ("pointer", "detect"):
        raise ValueError(f"variant must be 'pointer' or 'detect', got {variant!r}")
    g = torch.Generator().manual_seed(seed)

    query = torch.randint(0, KMER_VOCAB, (num_examples,), generator=g)
    # Background over the OTHER V-1 tokens, exactly uniform (fix 1).
    x = torch.randint(0, KMER_VOCAB - 1, (num_examples, seq_len), generator=g)
    x += (x >= query[:, None]).long()
    x[:, 0] = query

    pos = torch.randint(1, seq_len, (num_examples,), generator=g)
    rows = torch.arange(num_examples)

    if variant == "pointer":
        x[rows, pos] = query
        return x, pos

    y = torch.zeros(num_examples, dtype=torch.long)
    y[: num_examples // 2] = 1
    planted = torch.where(y == 1, query, x[rows, pos])
    x[rows, pos] = planted
    perm = torch.randperm(num_examples, generator=g)
    return x[perm], y[perm]
