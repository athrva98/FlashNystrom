"""Coverage for paper.mqar.paper_sweep, the certified paper driver.

No GPU and no training: job construction, protocol pinning, aggregation and
the CLI paths are exercised with stubbed/preseeded result files.
"""
import json
import os

import pytest

from paper.mqar import paper_sweep
from paper.mqar.paper_sweep import (
    METHODS, DIMS, LRS, PROTOCOL, heads_for, build_jobs, aggregate, main,
)
from paper.mqar.runner import build_cmd
from paper.mqar.train import build_parser


# --------------------------------------------------------------------------- #
# protocol certification: these values ARE the paper's protocol; a failure here
# means the certified experiment silently drifted
# --------------------------------------------------------------------------- #

def test_protocol_is_the_validated_figure2_recipe():
    assert PROTOCOL["seq_len"] == 512 and PROTOCOL["num_kv_pairs"] == 64
    assert PROTOCOL["layer_layout"] == "uniform"
    assert PROTOCOL["random_non_queries"] is False
    assert PROTOCOL["fresh_data"] is False
    assert PROTOCOL["num_train"] == 100_000 and PROTOCOL["num_test"] == 3_000
    assert PROTOCOL["batch_size"] == 128 and PROTOCOL["epochs"] == 64
    assert PROTOCOL["early_stop_acc"] == 0.99
    assert PROTOCOL["kappa_star"] == 0
    assert PROTOCOL["depth"] == 2 and PROTOCOL["vocab_size"] == 8192


def test_lr_grid_is_figure2_logspace():
    assert LRS[0] == pytest.approx(1e-4) and LRS[-1] == pytest.approx(1e-2)
    assert len(LRS) == 4
    # geometric spacing (logspace)
    assert LRS[1] / LRS[0] == pytest.approx(LRS[2] / LRS[1], rel=1e-3)


def test_method_set_is_bidirectional_native():
    # Hyena and Mamba are deliberately absent: causal by construction, so
    # comparing them would measure the masking regime, not the operator.
    assert len(METHODS) == 7
    for m in ("sdpa", "linear_attention", "linformer", "sliding_window",
              "nystrom_reference", "flash_nystrom", "flash_nystrom_tc"):
        assert m in METHODS, m
    assert "hyena" not in METHODS and "mamba" not in METHODS
    assert DIMS == [64, 128, 256, 512]


# --------------------------------------------------------------------------- #
# heads_for / build_jobs
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("d,h", [(64, 1), (128, 2), (256, 4), (512, 8)])
def test_heads_for_holds_head_dim_64(d, h):
    assert heads_for(d) == h and d // heads_for(d) == 64


def test_build_jobs_count():
    jobs = build_jobs(METHODS, DIMS, LRS, [0], "o")
    assert len(jobs) == 7 * 4 * 4


def test_build_jobs_seeds_multiply():
    assert len(build_jobs(["sdpa"], [64], LRS, [0, 1, 2], "o")) == 4 * 3


def test_build_jobs_out_json_is_resume_key():
    jobs = build_jobs(["sdpa"], [64], [1e-4], [0], "o")
    assert jobs[0]["out_json"] == os.path.join("o", "sdpa_d64_lr1.00e-04_seed0.json")
    assert jobs[0]["log_path"].endswith(".log")


def test_every_job_parses_under_train_argparse():
    """The certified commands must be accepted by train.py itself."""
    parser = build_parser()
    for j in build_jobs(METHODS, DIMS, LRS, [0], "o"):
        opts = {k: v for k, v in j.items() if k != "log_path"}
        parser.parse_args(build_cmd(**opts)[4:])


def test_jobs_pin_uniform_and_blanks():
    j = build_jobs(["hyena"], [512], [1e-2], [0], "o")[0]
    argv = build_cmd(**{k: v for k, v in j.items() if k != "log_path"})[4:]
    ns = build_parser().parse_args(argv)
    assert ns.layer_layout == "uniform"
    assert ns.random_non_queries is False and ns.fresh_data is False
    assert ns.early_stop_acc == 0.99 and ns.seq_len == 512


def test_benchmark_is_all_bidirectional():
    # ONE direction convention: no job may pass the causal flag. The Nystrom
    # family has no causal form, and masking only the baselines would confound
    # the operator comparison. Hyena/Mamba are causal by construction, which is
    # an operator property, not a protocol knob set here.
    parser = build_parser()
    for j in build_jobs(METHODS, [64], [1e-2], [0], "o"):
        assert "causal" not in j
        ns = parser.parse_args(build_cmd(**{k: v for k, v in j.items()
                                            if k != "log_path"})[4:])
        assert ns.causal is False   # train.py's diagnostic flag stays off


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #

def _fake(out, method, dim, lr, seed, recall):
    rec = {"backend": method, "dim": dim, "lr": lr, "seed": seed,
           "best_recall": recall}
    path = os.path.join(out, f"{method}_d{dim}_lr{lr:.2e}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(rec, f)


def test_aggregate_best_over_lr(tmp_path):
    out = str(tmp_path)
    for lr, r in zip(LRS, [10.0, 55.0, 92.0, 40.0]):
        _fake(out, "mamba", 128, lr, 0, r)
    table, edges = aggregate(out, ["mamba"], [128], LRS)
    assert table[("mamba", 128)]["mean"] == 92.0
    assert table[("mamba", 128)]["best_lrs"] == [LRS[2]]
    assert edges == []


def test_aggregate_mean_over_seeds(tmp_path):
    out = str(tmp_path)
    for s, r in [(0, 90.0), (1, 94.0)]:
        _fake(out, "sdpa", 64, LRS[2], s, r)
    table, _ = aggregate(out, ["sdpa"], [64], LRS)
    c = table[("sdpa", 64)]
    assert c["mean"] == 92.0 and c["n"] == 2 and c["sd"] > 0


def test_aggregate_flags_unsolved_boundary(tmp_path):
    out = str(tmp_path)
    # hyena-like: monotone toward the grid floor, never solved
    for lr, r in zip(LRS, [18.0, 12.0, 5.0, 0.1]):
        _fake(out, "hyena", 512, lr, 0, r)
    _, edges = aggregate(out, ["hyena"], [512], LRS)
    assert len(edges) == 1
    m, d, s, lr_b, r_b = edges[0]
    assert (m, d, lr_b) == ("hyena", 512, LRS[0])


def test_aggregate_no_flag_when_solved_at_boundary(tmp_path):
    out = str(tmp_path)
    for lr, r in zip(LRS, [99.5, 99.2, 99.7, 40.0]):
        _fake(out, "sdpa", 256, lr, 0, r)
    _, edges = aggregate(out, ["sdpa"], [256], LRS)
    assert edges == []   # solved cells are capped by the task, not the grid


def test_aggregate_missing_cells_render_as_dashes(tmp_path, capsys):
    out = str(tmp_path)
    _fake(out, "sdpa", 64, LRS[2], 0, 99.0)
    aggregate(out, ["sdpa", "mamba"], [64, 512], LRS)
    printed = capsys.readouterr().out
    assert "--" in printed and "99.00" in printed


def test_aggregate_writes_summary_json(tmp_path):
    out = str(tmp_path)
    _fake(out, "sdpa", 64, LRS[2], 0, 99.0)
    aggregate(out, ["sdpa"], [64], LRS)
    s = json.load(open(os.path.join(out, "summary.json")))
    assert s["protocol"]["seq_len"] == 512
    assert "sdpa|d64" in s["table"]
    # a later aggregate must not choke on its own summary file
    table, _ = aggregate(out, ["sdpa"], [64], LRS)
    assert table[("sdpa", 64)]["mean"] == 99.0


# --------------------------------------------------------------------------- #
# CLI paths (no GPU): --dry_run and --collect_only
# --------------------------------------------------------------------------- #

def test_main_dry_run_lists_everything(tmp_path, capsys):
    main(["--dry_run", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "112 runs" in out
    assert out.count(".json") == 112


def test_main_collect_only_aggregates(tmp_path, capsys, monkeypatch):
    _fake(str(tmp_path), "flash_nystrom", 256, LRS[2], 0, 99.08)
    monkeypatch.setattr(paper_sweep, "run_many",
                        lambda *a, **k: pytest.fail("collect_only must not run"))
    main(["--collect_only", "--out", str(tmp_path)])
    assert "99.08" in capsys.readouterr().out


def test_main_runs_then_aggregates(tmp_path, capsys, monkeypatch):
    calls = {}
    def fake_run_many(jobs, max_parallel):
        calls["n"] = len(jobs); calls["p"] = max_parallel
        for j in jobs:   # simulate train.py writing its record
            _fake(os.path.dirname(j["out_json"]), j["backend"], j["dim"],
                  float(j["lr"]), j["seed"], 50.0)
        return []
    monkeypatch.setattr(paper_sweep, "run_many", fake_run_many)
    main(["--methods", "sdpa", "--dims", "64", "--out", str(tmp_path),
          "--max_parallel", "2"])
    assert calls == {"n": 4, "p": 2}
    assert "50.00" in capsys.readouterr().out
