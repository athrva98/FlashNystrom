# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Exhaustive coverage of benchmarks.autobatch.

The batch-doubling search must NEVER run a real OOM here (an actual OOM on the
local 8GB card crashes the host). Instead we mock torch.cuda (empty_cache,
Event, timers, memory stats) and pass a fake make_trial that raises a simulated
OOM at a chosen batch. This exercises the search/break/return logic on CPU with
zero GPU allocation.
"""
import pytest
import torch

from benchmarks.autobatch import _is_oom, _time_trial, search_and_profile


# --------------------------------------------------------------------------- #
# mocks
# --------------------------------------------------------------------------- #

class _FakeEvent:
    def __init__(self, enable_timing=False):
        pass

    def record(self):
        pass

    def elapsed_time(self, other):
        return 5.0  # ms, constant so median is deterministic


@pytest.fixture
def mock_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    return monkeypatch


def _factory(oom_at=None, raise_exc=None):
    """make_trial(bs) -> run(); run() raises OOM when bs >= oom_at, or raises
    raise_exc (a non-OOM exception) when bs >= its threshold."""
    seen = []

    def make_trial(bs):
        def run():
            seen.append(bs)
            if raise_exc is not None and bs >= raise_exc[0]:
                raise raise_exc[1]
            if oom_at is not None and bs >= oom_at:
                raise RuntimeError("CUDA out of memory. Tried to allocate 999 GiB")
        return run

    make_trial.seen = seen
    return make_trial


# --------------------------------------------------------------------------- #
# _is_oom
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("msg", [
    "CUDA out of memory. Tried to allocate...",
    "out of memory",
    "OUT OF MEMORY",
    "RuntimeError: CUDA out of memory",
])
def test_is_oom_runtimeerror_true(msg):
    assert _is_oom(RuntimeError(msg)) is True


@pytest.mark.parametrize("msg", [
    "an illegal memory access was encountered",
    "device-side assert triggered",
    "shape mismatch",
    "",
])
def test_is_oom_runtimeerror_false(msg):
    assert _is_oom(RuntimeError(msg)) is False


@pytest.mark.parametrize("exc", [ValueError("out of memory"), KeyError("x"),
                                 TypeError("out of memory"), RuntimeError.__new__(RuntimeError)])
def test_is_oom_non_runtime_or_empty(exc):
    # ValueError/KeyError/TypeError are not RuntimeError -> not OOM even if the
    # text says so; a bare RuntimeError with no message -> not OOM.
    assert _is_oom(exc) is False


def test_is_oom_true_for_cuda_oom_error():
    # torch.cuda.OutOfMemoryError is matched by the isinstance branch. Construct
    # one if the runtime allows it (it subclasses RuntimeError in recent torch).
    try:
        e = torch.cuda.OutOfMemoryError("simulated")
    except Exception:
        pytest.skip("OutOfMemoryError not directly constructible in this torch")
    assert _is_oom(e) is True


# --------------------------------------------------------------------------- #
# _time_trial (mocked cuda)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bs", [2, 8, 64, 256])
def test_time_trial_metrics(mock_cuda, bs):
    m = _time_trial(_factory(), bs, warmup=2, iters=5)
    assert m["batch"] == bs
    assert m["step_ms"] == pytest.approx(5.0)
    assert m["samples_per_s"] == pytest.approx(bs * 1000.0 / 5.0)
    assert m["peak_gib"] == pytest.approx(1.0)


def test_time_trial_runs_warmup_and_iters(mock_cuda):
    mk = _factory()
    _time_trial(mk, 4, warmup=3, iters=7)
    assert len(mk.seen) == 3 + 7  # warmup + timed iters, all at bs=4
    assert set(mk.seen) == {4}


def test_time_trial_propagates_oom(mock_cuda):
    with pytest.raises(RuntimeError, match="out of memory"):
        _time_trial(_factory(oom_at=1), 8, warmup=1, iters=1)


# --------------------------------------------------------------------------- #
# search_and_profile (mocked cuda)
# --------------------------------------------------------------------------- #

def test_search_stops_at_cap_when_all_fit(mock_cuda):
    best = search_and_profile(_factory(), lo=2, cap=8, warmup=1, iters=1)
    assert best["batch"] == 8  # doubled 2 -> 4 -> 8, then 16 > cap


@pytest.mark.parametrize("oom_at,expected", [(8, 4), (16, 8), (4, 2), (64, 32)])
def test_search_returns_last_fitting_batch(mock_cuda, oom_at, expected):
    best = search_and_profile(_factory(oom_at=oom_at), lo=2, cap=4096, warmup=1, iters=1)
    assert best["batch"] == expected


def test_search_returns_none_if_lo_ooms(mock_cuda):
    best = search_and_profile(_factory(oom_at=2), lo=2, cap=1024, warmup=1, iters=1)
    assert best is None


def test_search_reraises_non_oom(mock_cuda):
    with pytest.raises(ValueError, match="boom"):
        search_and_profile(_factory(raise_exc=(2, ValueError("boom"))),
                           lo=2, cap=64, warmup=1, iters=1)


def test_search_verbose_fit_and_oom(mock_cuda, capsys):
    best = search_and_profile(_factory(oom_at=8), lo=2, cap=64,
                              warmup=1, iters=1, verbose=True)
    out = capsys.readouterr().out
    assert best["batch"] == 4
    assert "batch=2" in out and "OOM (stop)" in out


def test_search_verbose_all_fit_prints_each(mock_cuda, capsys):
    search_and_profile(_factory(), lo=2, cap=8, warmup=1, iters=1, verbose=True)
    out = capsys.readouterr().out
    for b in (2, 4, 8):
        assert f"batch={b}" in out


@pytest.mark.parametrize("lo", [1, 2, 4, 16])
def test_search_respects_lo(mock_cuda, lo):
    mk = _factory(oom_at=10 ** 9)  # never OOM
    best = search_and_profile(mk, lo=lo, cap=lo, warmup=1, iters=1)
    assert best["batch"] == lo
    assert min(mk.seen) == lo  # never probed below lo
