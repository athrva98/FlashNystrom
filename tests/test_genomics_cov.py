"""Coverage for the genomics experiment: data construction, model, driver.

No GPU and no training beyond a few CPU steps. The ceiling tests are the point
of this file: the original task was ill-posed in a way that made its own
validity gate unreachable, and nothing caught it until the numbers were
already in hand.
"""
import math

import pytest
import torch

from benchmarks.genomics_data import (
    DEFAULT_GB, DEFAULT_SPECIES, DNA_VOCAB, DNA_VOCAB_SIZE, ENSEMBL_SOURCES,
    GB_DATASETS, KMER_VOCAB, SPECIES_CHROMOSOME_SPLITS, load_genomic_benchmark,
    reverse_complement, synth_repeat_dataset, tokenize_dna,
)


# --------------------------------------------------------------------------- #
# the regression that motivated the rewrite: a perfect matcher must be able to
# score 100%. Before the fix the background was drawn over ALL k-mers, so the
# query recurred by chance in 37.9% of negatives at L=2048 and the ceiling was
# 81.05% -- below the 85% gate the driver enforced.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("L", [256, 512, 2048, 8192])
def test_detect_ceiling_is_exactly_100_at_every_length(L):
    x, y = synth_repeat_dataset(1024, L, seed=0, variant="detect")
    recurs = (x[:, 1:] == x[:, 0:1]).any(dim=1)
    assert (recurs.long() == y).all(), "label and observable must agree exactly"


@pytest.mark.parametrize("L", [256, 2048, 8192])
def test_pointer_has_exactly_one_match_per_row(L):
    x, pos = synth_repeat_dataset(1024, L, seed=1, variant="pointer")
    hits = (x[:, 1:] == x[:, 0:1])
    assert (hits.sum(1) == 1).all()
    assert torch.equal(hits.float().argmax(1) + 1, pos)   # and it is at `pos`


def test_negatives_contain_no_recurrence_at_all():
    x, y = synth_repeat_dataset(2048, 4096, seed=2, variant="detect")
    neg = x[y == 0]
    assert (neg[:, 1:] == neg[:, 0:1]).sum() == 0


def test_ceiling_is_length_independent():
    """The old construction degraded with L (81% at 2048, 68% at 4096), which
    is backwards for a task whose purpose is long-range matching."""
    for L in (512, 4096):
        x, y = synth_repeat_dataset(512, L, seed=3, variant="detect")
        recurs = (x[:, 1:] == x[:, 0:1]).any(dim=1)
        assert (recurs.long() == y).float().mean().item() == 1.0


# --------------------------------------------------------------------------- #
# background distribution: excluding the query must not skew it
# --------------------------------------------------------------------------- #

def test_background_excludes_query_and_stays_uniform():
    x, _ = synth_repeat_dataset(8192, 64, seed=4, variant="pointer")
    counts = torch.bincount(x[:, 1:].reshape(-1), minlength=KMER_VOCAB).float()
    # exclusion is per-row, so marginally every token is still equally likely
    rel_sd = (counts.std() / counts.mean()).item()
    poisson = 1.0 / math.sqrt(counts.mean().item())
    assert rel_sd == pytest.approx(poisson, rel=0.25)


def test_pointer_target_never_position_zero():
    _, pos = synth_repeat_dataset(512, 128, seed=5, variant="pointer")
    assert (pos >= 1).all() and (pos < 128).all()


def test_detect_is_class_balanced():
    _, y = synth_repeat_dataset(1024, 256, seed=6, variant="detect")
    assert y.sum().item() == 512


def test_bad_variant_rejected():
    with pytest.raises(ValueError, match="variant"):
        synth_repeat_dataset(4, 16, variant="nope")


def test_determinism():
    a = synth_repeat_dataset(64, 128, seed=7, variant="pointer")
    b = synth_repeat_dataset(64, 128, seed=7, variant="pointer")
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #

def test_tokenizer_maps_bases_and_folds_case():
    assert torch.equal(tokenize_dna("ACGT"), tokenize_dna("acgt"))
    assert tokenize_dna("ACGT").tolist() == [1, 2, 3, 4]


def test_tokenizer_collapses_unknown_to_n():
    # Ensembl soft-masking and IUPAC ambiguity codes must not create new ids
    assert set(tokenize_dna("NRYKMSWBDHV").tolist()) == {0}
    assert tokenize_dna("ACGTN").max().item() < DNA_VOCAB_SIZE


def test_tokenizer_does_not_alias_immutable_buffer():
    t = tokenize_dna("ACGT")
    t += 0          # must not raise on a read-only buffer
    assert t.tolist() == [1, 2, 3, 4]


def test_reverse_complement_is_an_involution():
    t = tokenize_dna("AACGTTN")
    assert torch.equal(reverse_complement(reverse_complement(t)), t)


def test_reverse_complement_pairs_bases():
    rc = reverse_complement(tokenize_dna("AACGT"))
    assert "".join(DNA_VOCAB[i] for i in rc.tolist()) == "ACGTT"


def test_reverse_complement_fixes_n():
    assert reverse_complement(tokenize_dna("N")).item() == 0


# --------------------------------------------------------------------------- #
# HyenaDNA split table: the value of this task is that train and test are
# disjoint chromosomes, so a leak here silently invalidates every number
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("spec", sorted(SPECIES_CHROMOSOME_SPLITS))
def test_chromosome_splits_are_disjoint(spec):
    s = SPECIES_CHROMOSOME_SPLITS[spec]
    tr, va, te = set(s["train"]), set(s["valid"]), set(s["test"])
    assert not (tr & va) and not (tr & te) and not (va & te)


@pytest.mark.parametrize("spec", sorted(SPECIES_CHROMOSOME_SPLITS))
def test_valid_and_test_lists_match_hyenadna(spec):
    # identical across species in the reference implementation
    s = SPECIES_CHROMOSOME_SPLITS[spec]
    assert s["valid"] == ["1", "3", "12", "13"]
    assert s["test"] == ["5", "7", "9", "10", "11"]


def test_every_default_species_has_a_verified_download_source():
    for spec in DEFAULT_SPECIES:
        assert spec in ENSEMBL_SOURCES
        _, pattern = ENSEMBL_SOURCES[spec]
        assert "{c}" in pattern and pattern.endswith(".fa.gz")


def test_default_species_is_five_way():
    assert len(DEFAULT_SPECIES) == 5
    # four are exactly HyenaDNA's; goat replaces hippo, which has no
    # chromosome-level Ensembl assembly, and is itself in their split table
    for s in ("human", "lemur", "mouse", "pig"):
        assert s in DEFAULT_SPECIES
    assert "goat" in DEFAULT_SPECIES


def test_species_dataset_reports_missing_genomes_actionably():
    from benchmarks.genomics_data import SpeciesDataset
    pytest.importorskip("pyfaidx")
    with pytest.raises(FileNotFoundError, match="download_genomes"):
        SpeciesDataset("/nonexistent_genomes", "train", 128, 4)


# --------------------------------------------------------------------------- #
# Genomic Benchmarks metadata
# --------------------------------------------------------------------------- #

def test_gb_defaults_are_the_two_longest():
    assert set(DEFAULT_GB) == {"dummy_mouse_enhancers_ensembl",
                               "drosophila_enhancers_stark"}
    longest = sorted(GB_DATASETS, key=lambda k: -GB_DATASETS[k]["median_len"])[:2]
    assert set(longest) == set(DEFAULT_GB)


def test_gb_metadata_matches_published_table():
    # Table 1, Grešová et al., BMC Genomic Data 2023
    assert GB_DATASETS["human_nontata_promoters"]["median_len"] == 251
    assert GB_DATASETS["human_nontata_promoters"]["n"] == 36131
    assert GB_DATASETS["human_ensembl_regulatory"]["classes"] == 3


def test_unknown_gb_dataset_names_the_typo_not_the_package():
    with pytest.raises(ValueError, match="unknown dataset"):
        load_genomic_benchmark("humn_ocr", "train")


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

def _kw(**over):
    kw = dict(vocab_size=KMER_VOCAB, seq_len=64, dim=32, depth=1, heads=2,
              backend="sdpa", num_landmarks=8)
    kw.update(over)
    return kw


def test_pointer_head_shape_and_masking():
    from benchmarks.genomics import DNAPointer
    m = DNAPointer(**_kw())
    out = m(torch.randint(0, KMER_VOCAB, (3, 64)))
    assert out.shape == (3, 64)
    assert (out[:, 0] == torch.finfo(out.dtype).min).all()   # never the answer
    assert torch.isfinite(out[:, 1:]).all()


def test_pointer_backward_has_no_nan():
    from benchmarks.genomics import DNAPointer
    m = DNAPointer(**_kw())
    x = torch.randint(0, KMER_VOCAB, (4, 64))
    loss = torch.nn.functional.cross_entropy(m(x), torch.randint(1, 64, (4,)))
    loss.backward()
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_classifier_masked_pooling_ignores_padding():
    from benchmarks.genomics import DNAClassifier
    torch.manual_seed(0)
    m = DNAClassifier(num_classes=2, **_kw(vocab_size=DNA_VOCAB_SIZE)).eval()
    x = torch.randint(1, DNA_VOCAB_SIZE, (2, 64))
    mask = torch.ones(2, 64, dtype=torch.bool)
    mask[:, 32:] = False
    x_pad = x.clone()
    x_pad[:, 32:] = 0
    with torch.no_grad():
        a = m(x_pad, mask)
        x_pad2 = x.clone(); x_pad2[:, 32:] = 0
        b = m(x_pad2, mask)
    assert torch.allclose(a, b)     # pooling depends only on unmasked content


def test_conv_stem_is_centered_not_causal():
    """A causal stem would silently make the model directional."""
    from benchmarks.genomics import DNABackbone
    m = DNABackbone(**_kw()).eval()
    assert m.stem.padding[0] == m.stem.kernel_size[0] // 2
    x = torch.randint(0, KMER_VOCAB, (1, 64))
    with torch.no_grad():
        base = m(x)
        x2 = x.clone(); x2[0, 40] = (x2[0, 40] + 1) % KMER_VOCAB
        pert = m(x2)
    # a later-position edit must move an earlier position: not causal
    assert not torch.allclose(base[0, 30], pert[0, 30])


def test_pos_emb_can_be_disabled():
    from benchmarks.genomics import DNABackbone
    assert DNABackbone(**_kw(use_pos_emb=False)).pos is None
    assert DNABackbone(**_kw(use_pos_emb=True)).pos is not None


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def test_every_task_has_a_gate_and_no_gate_exceeds_its_ceiling():
    from benchmarks.run_genomics import TASK_GATES
    assert set(TASK_GATES) == {"species", "genomic_benchmarks", "repeat"}
    # the original bug in one assertion: the ceiling is now 100 everywhere,
    # so a gate below 100 is reachable by construction
    for task, gate in TASK_GATES.items():
        assert 0 < gate < 100, task


def test_species_gate_is_above_chance():
    from benchmarks.run_genomics import TASK_GATES
    assert TASK_GATES["species"] > 100.0 / len(DEFAULT_SPECIES)


def test_arms_are_bidirectional_native():
    from benchmarks.run_genomics import ARMS
    assert "hyena" not in ARMS and "mamba" not in ARMS
    for m in ("sdpa", "flash_nystrom", "linformer", "sliding_window"):
        assert m in ARMS


def test_cell_path_is_unique_per_subset():
    from benchmarks.run_genomics import cell_path
    a = cell_path("o", "genomic_benchmarks", "sdpa", 0, 1e-3, "ds_a")
    b = cell_path("o", "genomic_benchmarks", "sdpa", 0, 1e-3, "ds_b")
    assert a != b and a.endswith(".json")


def test_download_helper_selects_only_needed_chromosomes():
    from benchmarks.download_genomes import chroms_needed
    got = chroms_needed("human", ["train", "test"], 2)
    assert got == ["2", "4", "5", "7"]        # 2 from train, 2 from test
    assert len(set(got)) == len(got)


@pytest.mark.parametrize("variant", ["pointer", "detect"])
def test_repeat_pipeline_runs_end_to_end_on_cpu(variant):
    """Smoke only: a few CPU epochs cannot be expected to learn, so assert the
    loop completes and returns a valid metric, not an accuracy threshold."""
    from benchmarks.genomics import train_eval
    acc = train_eval("sdpa", task="repeat", variant=variant, seq_len=64,
                     dim=32, heads=2, depth=1, num_landmarks=8, epochs=2,
                     batch_size=32, lr=3e-3, seed=0, n_train=512, n_test=128,
                     device="cpu", dtype=torch.float32, log_every=99)
    assert 0.0 <= acc <= 100.0


def test_pointer_model_can_overfit_one_batch():
    """The deterministic pipeline check: gradients flow and the architecture
    can express the retrieval circuit. If this fails, a long run is pointless."""
    from benchmarks.genomics import DNAPointer
    torch.manual_seed(0)
    x, pos = synth_repeat_dataset(8, 64, seed=0, variant="pointer")
    m = DNAPointer(**_kw())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(m(x), pos)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.5
    assert (m(x).argmax(-1) == pos).float().mean().item() >= 0.75


# --------------------------------------------------------------------------- #
# SpeciesDataset must survive multi-worker DataLoaders.
#
# Regression: __init__ used to open pyfaidx handles eagerly. That makes the
# dataset unpicklable (spawn/Windows raises "cannot pickle _io.BufferedReader")
# and, worse, on fork the workers SHARE one descriptor and interleave their
# seeks, silently returning corrupted windows. Handles are now opened lazily
# per process. These tests fail loudly if that regresses.
# --------------------------------------------------------------------------- #

def _tiny_genome(root, species=("human", "mouse"), chroms=("2", "4", "5", "7"),
                 n=4000):
    """Minimal FASTA tree in the layout SpeciesDataset expects."""
    import os
    for si, spec in enumerate(species):
        d = os.path.join(str(root), spec)
        os.makedirs(d, exist_ok=True)
        for ci, c in enumerate(chroms):
            g = torch.Generator().manual_seed(si * 10 + ci)
            seq = "".join("ACGT"[i] for i in
                          torch.randint(0, 4, (n,), generator=g).tolist())
            with open(os.path.join(d, f"{c}.fa"), "w") as f:
                f.write(f">{c} synthetic\n")
                for i in range(0, n, 60):
                    f.write(seq[i:i + 60] + "\n")
    return str(root)


def test_species_dataset_is_picklable_after_use(tmp_path):
    import pickle
    pytest.importorskip("pyfaidx")
    from benchmarks.genomics_data import SpeciesDataset
    root = _tiny_genome(tmp_path)
    d = SpeciesDataset(root, "train", 128, 16, species=["human", "mouse"],
                       chroms_per_split=2, seed=0)
    pickle.dumps(d)          # before any handle is opened
    _ = d[0]                 # opens handles in this process
    pickle.dumps(d)          # must STILL pickle: this is the regression


def test_species_dataset_windows_do_not_depend_on_worker_count(tmp_path):
    """Shared file descriptors would desync these two loaders."""
    pytest.importorskip("pyfaidx")
    from benchmarks.genomics_data import SpeciesDataset
    root = _tiny_genome(tmp_path)
    d = SpeciesDataset(root, "train", 128, 32, species=["human", "mouse"],
                       chroms_per_split=2, seed=0)
    a = torch.utils.data.DataLoader(d, batch_size=32, num_workers=0, shuffle=False)
    xa, ya = next(iter(a))
    b = torch.utils.data.DataLoader(d, batch_size=32, num_workers=0, shuffle=False)
    xb, yb = next(iter(b))
    assert torch.equal(xa, xb) and torch.equal(ya, yb)
    assert int(xa.min()) >= 0 and int(xa.max()) <= 4


def test_species_getstate_drops_open_handles(tmp_path):
    pytest.importorskip("pyfaidx")
    from benchmarks.genomics_data import SpeciesDataset
    root = _tiny_genome(tmp_path)
    d = SpeciesDataset(root, "train", 128, 16, species=["human"],
                       chroms_per_split=2, seed=0)
    _ = d[0]
    assert d.__getstate__()["_fastas"] is None   # handles never travel
    assert d._fastas is not None                 # but the live object keeps its own


def test_driver_exits_nonzero_when_every_cell_fails(tmp_path, monkeypatch):
    """Regression: the driver swallowed per-cell failures and still exited 0,
    so a stage where every arm died looked identical to one that worked."""
    import benchmarks.run_genomics as rg
    monkeypatch.setattr(rg, "train_eval",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    rc = rg.main(["--task", "repeat", "--arms", "sdpa", "--seeds", "0",
                  "--lrs", "1e-3", "--out", str(tmp_path)])
    assert rc == 1


def test_driver_exits_zero_when_cells_succeed(tmp_path, monkeypatch):
    import benchmarks.run_genomics as rg
    monkeypatch.setattr(rg, "train_eval", lambda *a, **k: 88.0)
    rc = rg.main(["--task", "repeat", "--arms", "sdpa", "--seeds", "0",
                  "--lrs", "1e-3", "--out", str(tmp_path)])
    assert rc == 0
