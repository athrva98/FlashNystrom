# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Orchestrator: resume, budget, and the paper12 grid.

The resume tests matter most. MQAR and genomics skip finished cells inside
their own drivers, but train_three_way.py has no such check, so without the
orchestrator's marker the vision stage retrains from scratch after a runtime
reset. On Colab that is the difference between losing one run and losing hours.
"""
import json
import os

import pytest

import run_all_paper_experiments as R
from run_all_paper_experiments import ARMS, build_jobs, completion_marker, run


def test_vision_jobs_declare_a_completion_marker():
    for stage, name, argv in build_jobs(["vision"], ARMS, [0], "runs", False):
        m = completion_marker(argv)
        assert m and m.endswith(".json"), name


def test_self_resuming_stages_declare_no_marker():
    # their drivers skip finished cells themselves; a marker here would skip the
    # whole sweep after its first cell completed
    for stage in ("mqar", "genomics"):
        for _, name, argv in build_jobs([stage], ARMS, [0], "runs", False):
            assert completion_marker(argv) is None, name


def test_run_skips_a_job_whose_marker_exists(tmp_path, capsys, monkeypatch):
    out = tmp_path / "vision_x.json"
    out.write_text("{}")
    argv = ["python", "x.py", "--out_json", str(out)]
    monkeypatch.setattr(R.subprocess, "run",
                        lambda *a, **k: pytest.fail("must not launch a finished job"))
    run([("vision", "x", argv)], str(tmp_path / "logs"))
    assert "done" in capsys.readouterr().out


def test_run_launches_a_job_whose_marker_is_absent(tmp_path, monkeypatch):
    calls = []

    class P:
        returncode = 0
    monkeypatch.setattr(R.subprocess, "run",
                        lambda *a, **k: (calls.append(a), P())[1])
    argv = ["python", "x.py", "--out_json", str(tmp_path / "missing.json")]
    run([("vision", "x", argv)], str(tmp_path / "logs"))
    assert len(calls) == 1


def test_paper12_runs_the_largest_vision_tier_at_one_seed_only():
    """The reduced grid the paper's 5.1 describes: 3 seeds everywhere except the
    most expensive tier."""
    jobs = build_jobs(["vision"], ARMS, [0, 1, 2], "runs", False, preset="paper12")
    names = [n for _, n, _ in jobs]
    assert sum("i180" in n for n in names) == 1
    for tier in ("cifar10_p4_i32", "stl10_p2_i96", "stl10_p1_i96"):
        assert sum(n.startswith(tier) for n in names) == 3


def test_paper12_carries_the_learning_rate_up_to_the_long_genomics_tier():
    jobs = build_jobs(["genomics"], ARMS, [0], "runs", False, preset="paper12")
    long_job = [a for _, n, a in jobs if n == "species_N32768"][0]
    assert "--lr_from" in long_job              # reuses the N=1024 winner
    assert long_job[long_job.index("--lrs") + 1:][:1] != []
    short = [a for _, n, a in jobs if n == "species_N1024"][0]
    assert short.count("--lrs") == 1 and len(short[short.index("--lrs") + 1:]) > 4
