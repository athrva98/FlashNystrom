"""Coverage for paper.mqar.runner, the shared MQAR subprocess driver.

Runs on CPU with no GPU and no real training: the subprocess layer is stubbed,
so these exercise command construction, output parsing and the sweep logic.
The parsed formats are pinned against real train.py output strings.
"""
import json
import subprocess

import pytest

from paper.mqar import runner
from paper.mqar.runner import (
    build_cmd,
    parse_output,
    run_train,
    sweep_lr,
    run_many,
    finite_diag,
    already_done,
)
from paper.mqar.train import build_parser

# Verbatim lines emitted by paper/mqar/train.py.
REAL_OUT = (
    "backend=sdpa dim=128 depth=2 heads=2 head_dim=64 init=normal params=1.21M\n"
    "epoch    1/2  loss 8.5132  test recall 0.02%  grad_norm 0.218  "
    "cond_K2 1.23e+03  cond_M 4.56e+05  pinv_resid 7.80e-07\n"
    "epoch    2/2  loss 8.3784  test recall 12.50%  grad_norm 0.250  "
    "cond_K2 nan  cond_M nan  pinv_resid nan\n"
    "best test recall: 12.50%\n"
    "train profile: batch=256 step_ms=25.83 samples_per_s=9680.0 peak_GiB=0.82\n"
)


# --------------------------------------------------------------------------- #
# build_cmd
# --------------------------------------------------------------------------- #

def test_build_cmd_invokes_train_module():
    cmd = build_cmd(backend="sdpa")
    assert cmd[1:4] == ["-u", "-m", "paper.mqar.train"]


def test_build_cmd_pairs_values():
    assert build_cmd(backend="sdpa", seed=3)[4:] == ["--backend", "sdpa", "--seed", "3"]


def test_build_cmd_drops_none():
    assert "--dim" not in build_cmd(backend="sdpa", dim=None)


def test_build_cmd_true_is_bare_flag():
    assert build_cmd(backend="sdpa", diag=True)[4:] == ["--backend", "sdpa", "--diag"]


def test_build_cmd_false_emits_nothing():
    # store_true flags have no --no- form; emitting one would be an argparse error
    assert build_cmd(backend="sdpa", diag=False)[4:] == ["--backend", "sdpa"]


def test_build_cmd_optional_bool_false_emits_no_form():
    # BooleanOptionalAction flags default True, so False MUST emit --no-<name>;
    # omitting it would silently leave the default on.
    assert build_cmd(backend="sdpa", random_non_queries=False)[4:] == \
           ["--backend", "sdpa", "--no-random_non_queries"]


def test_build_cmd_optional_bool_true_emits_bare_flag():
    assert build_cmd(backend="sdpa", fresh_data=True)[4:] == \
           ["--backend", "sdpa", "--fresh_data"]


@pytest.mark.parametrize("flag,val,expected", [
    ("random_non_queries", False, "--no-random_non_queries"),
    ("random_non_queries", True, "--random_non_queries"),
    ("fresh_data", False, "--no-fresh_data"),
    ("layer_layout", "uniform", "--layer_layout"),
])
def test_build_cmd_new_flags_parse_under_train(flag, val, expected):
    # the generated token(s) must be accepted by train.py's own parser
    argv = build_cmd(backend="hyena", **{flag: val})[4:]
    assert expected in argv
    build_parser().parse_args(argv)


def test_build_cmd_appends_extra_verbatim():
    assert build_cmd(backend="sdpa", extra=["--epochs", "1"])[-2:] == ["--epochs", "1"]


def test_build_cmd_honors_python_override():
    assert build_cmd(python="/usr/bin/python3", backend="sdpa")[0] == "/usr/bin/python3"


@pytest.mark.parametrize("kw", [
    dict(backend="hyena", seed=0, lr=5e-4, kappa_star=0, seq_len=256, num_kv_pairs=16),
    dict(backend="sdpa", heads=2, dim=128, init="normal", seed=0, lr=1e-2),
    dict(backend="flash_nystrom", num_landmarks=64, newton_iter=6, autobatch=True, diag=True),
    dict(backend="sdpa", autobatch=False, diag=False, conv=False),
])
def test_build_cmd_output_is_accepted_by_train_parser(kw):
    """Every command this module generates must parse under train.py's argparse."""
    build_parser().parse_args(build_cmd(**kw)[4:])


# --------------------------------------------------------------------------- #
# parse_output
# --------------------------------------------------------------------------- #

def test_parse_recall():
    assert parse_output(REAL_OUT)["recall"] == 12.50


def test_parse_profile_fields():
    r = parse_output(REAL_OUT)
    assert (r["batch"], r["step_ms"], r["samples_per_s"], r["peak_GiB"]) == \
           (256, 25.83, 9680.0, 0.82)


def test_parse_takes_last_diag_line():
    r = parse_output(REAL_OUT)
    assert r["cond_K2"] != r["cond_K2"]      # last line is nan
    assert r["grad_norm"] == 0.250           # last line's value


def test_parse_handles_nan_tokens():
    # sdpa has no landmarks -> train.py prints nan; the regex must not choke
    r = parse_output("cond_K2 nan  cond_M nan  pinv_resid nan\n")
    assert all(r[k] != r[k] for k in ("cond_K2", "cond_M", "pinv_resid"))


def test_parse_missing_recall_is_none():
    assert parse_output("crashed\n")["recall"] is None


def test_parse_absent_profile_omits_keys():
    assert "step_ms" not in parse_output("best test recall: 5.00%\n")


# --------------------------------------------------------------------------- #
# finite_diag
# --------------------------------------------------------------------------- #

def test_finite_diag_drops_nan():
    assert finite_diag(parse_output(REAL_OUT)) == {"grad_norm": 0.250}


def test_finite_diag_keeps_finite():
    d = finite_diag({"cond_K2": 1.5, "cond_M": 2.5, "pinv_resid": 1e-7, "grad_norm": 0.2})
    assert d == {"cond_K2": 1.5, "cond_M": 2.5, "pinv_resid": 1e-7, "grad_norm": 0.2}


def test_finite_diag_drops_inf_and_missing():
    assert finite_diag({"cond_K2": float("inf")}) == {}


def test_finite_diag_output_is_json_serializable():
    json.dumps(finite_diag(parse_output(REAL_OUT)))  # NaN would still dump; absence is the point


# --------------------------------------------------------------------------- #
# already_done
# --------------------------------------------------------------------------- #

def test_already_done_false_for_none():
    assert already_done(None) is False


def test_already_done_true_for_existing(tmp_path):
    f = tmp_path / "r.json"
    f.write_text("{}")
    assert already_done(str(f)) is True


def test_already_done_false_for_missing(tmp_path):
    assert already_done(str(tmp_path / "nope.json")) is False


# --------------------------------------------------------------------------- #
# run_train / sweep_lr (subprocess stubbed)
# --------------------------------------------------------------------------- #

class _Proc:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.returncode = stdout, returncode


@pytest.fixture
def fake_run(monkeypatch):
    calls = []

    def _run(cmd, **kw):
        calls.append((cmd, kw))
        return _Proc(REAL_OUT, 0)

    monkeypatch.setattr(subprocess, "run", _run)
    return calls


def test_run_train_parses_and_reports_rc(fake_run):
    r = run_train(backend="sdpa")
    assert r["recall"] == 12.50 and r["returncode"] == 0


def test_run_train_keeps_raw_output(fake_run):
    assert "best test recall" in run_train(backend="sdpa")["output"]


def test_run_train_forwards_cwd(fake_run):
    run_train(backend="sdpa", cwd="/repo")
    assert fake_run[0][1]["cwd"] == "/repo"


def test_run_train_writes_log(tmp_path, fake_run):
    log = tmp_path / "sub" / "run.log"
    run_train(backend="sdpa", log_path=str(log))
    assert "best test recall" in log.read_text(encoding="utf-8")


def test_run_train_merges_out_json(tmp_path, fake_run):
    j = tmp_path / "r.json"
    j.write_text(json.dumps({"best_recall": 99.9, "backend": "sdpa", "step_ms": None}))
    r = run_train(backend="sdpa", out_json=str(j))
    assert r["recall"] == 99.9                 # structured record wins
    assert r["step_ms"] == 25.83               # its None must NOT clobber the scraped value


def test_run_train_stream_echoes(monkeypatch, capsys):
    class _Popen:
        def __init__(self, *a, **kw):
            self.stdout = iter(REAL_OUT.splitlines(keepends=True))
            self.returncode = 0
        def wait(self): return 0
    monkeypatch.setattr(subprocess, "Popen", _Popen)
    r = run_train(backend="sdpa", stream=True)
    assert r["recall"] == 12.50
    assert "best test recall" in capsys.readouterr().out


def test_sweep_lr_picks_best(monkeypatch):
    scores = {0.001: 10.0, 0.01: 42.0, 0.1: 5.0}
    monkeypatch.setattr(runner, "run_train",
                        lambda **kw: {"recall": scores[kw["lr"]]})
    best = sweep_lr([0.001, 0.01, 0.1], backend="sdpa")
    assert best["lr"] == 0.01 and best["recall"] == 42.0


def test_sweep_lr_skips_failed_runs(monkeypatch):
    monkeypatch.setattr(runner, "run_train",
                        lambda **kw: {"recall": None if kw["lr"] == 0.001 else 7.0})
    assert sweep_lr([0.001, 0.01], backend="sdpa")["lr"] == 0.01


def test_sweep_lr_all_failed_returns_empty(monkeypatch):
    monkeypatch.setattr(runner, "run_train", lambda **kw: {"recall": None})
    assert sweep_lr([0.001, 0.01], backend="sdpa") == {}


def test_sweep_lr_invokes_callback(monkeypatch):
    monkeypatch.setattr(runner, "run_train", lambda **kw: {"recall": 1.0})
    seen = []
    sweep_lr([0.001, 0.01], backend="sdpa", on_result=lambda lr, r: seen.append(lr))
    assert seen == [0.001, 0.01]


# --------------------------------------------------------------------------- #
# run_many (concurrent driver; run_train stubbed)
# --------------------------------------------------------------------------- #

def test_run_many_runs_all_jobs(monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "run_train",
                        lambda **kw: (seen.append(kw["dim"]), {"recall": 50.0})[1])
    jobs = [{"backend": "hyena", "dim": d, "out_json": None} for d in (64, 128, 256)]
    res = run_many(jobs, max_parallel=3)
    assert len(res) == 3 and sorted(seen) == [64, 128, 256]


def test_run_many_skips_already_done(monkeypatch, tmp_path):
    done = tmp_path / "done.json"; done.write_text("{}")
    ran = []
    monkeypatch.setattr(runner, "run_train",
                        lambda **kw: (ran.append(kw["out_json"]), {"recall": 1.0})[1])
    jobs = [{"backend": "sdpa", "out_json": str(done)},
            {"backend": "sdpa", "out_json": str(tmp_path / "todo.json")}]
    res = run_many(jobs, max_parallel=2)
    assert len(res) == 1 and ran == [str(tmp_path / "todo.json")]


def test_run_many_all_done_returns_empty(monkeypatch, tmp_path):
    f = tmp_path / "a.json"; f.write_text("{}")
    monkeypatch.setattr(runner, "run_train", lambda **kw: pytest.fail("should not run"))
    assert run_many([{"out_json": str(f)}], max_parallel=2) == []


def test_run_many_isolates_crashed_job(monkeypatch):
    def flaky(**kw):
        if kw["dim"] == 128:
            raise RuntimeError("boom")
        return {"recall": 42.0}
    monkeypatch.setattr(runner, "run_train", flaky)
    jobs = [{"backend": "mamba", "dim": d, "out_json": None} for d in (64, 128, 256)]
    res = run_many(jobs, max_parallel=3)
    assert len(res) == 3                                   # pool survives the crash
    crashed = [r for r in res if r.get("error")]
    assert len(crashed) == 1 and crashed[0]["dim"] == 128
    assert sum(r.get("recall") == 42.0 for r in res) == 2


def test_run_many_progress_callback(monkeypatch):
    monkeypatch.setattr(runner, "run_train", lambda **kw: {"recall": 1.0})
    calls = []
    run_many([{"backend": "sdpa", "out_json": None}], max_parallel=1,
             on_done=lambda job, res, n, total: calls.append((n, total)))
    assert calls == [(1, 1)]


def test_run_many_default_progress_prints(monkeypatch, capsys):
    monkeypatch.setattr(runner, "run_train",
                        lambda **kw: {"recall": 88.5, "elapsed_s": 90.0})
    run_many([{"backend": "mamba", "dim": 128, "lr": 3e-3, "out_json": None}],
             max_parallel=1)
    out = capsys.readouterr().out
    assert "mamba d=128" in out and "88.5" in out and "1.5 min" in out
