# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Single source of truth for driving ``paper.mqar.train`` as a subprocess.

Every MQAR call site (``sweep.py``, ``run_scaling_sweep.py``, and the notebooks)
used to build its own command string, re-declare its own copy of the
``best test recall:`` regex, and reimplement skip-if-exists and best-over-LR.
Five copies of the same parsing meant a change to train.py's output format
silently broke some drivers and not others. They all call in here now.

The output formats parsed below are produced by ``paper/mqar/train.py``:

    best test recall: 99.71%
    train profile: batch=256 step_ms=25.83 samples_per_s=9680.0 peak_GiB=0.82
    epoch 1/64  loss 8.70  test recall 0.02%  grad_norm 0.218  cond_K2 1.2e+03 ...

``--out_json`` is preferred when the caller asks for it: train.py writes a
structured record there, so no regex is involved at all. The regexes remain the
fallback for callers that only have stdout (the tee-to-log notebooks).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys

# A float token, tolerating the nan/inf that the diagnostics emit for backends
# without landmarks (sdpa reports nan for cond_K2 / cond_M / pinv_resid).
_NUM = r"([-+]?(?:nan|inf|\d+\.?\d*(?:[eE][-+]?\d+)?))"

BEST_RE = re.compile(r"best test recall:\s*([\d.]+)%")
PROF_RE = re.compile(
    r"train profile: batch=(\d+) step_ms=([\d.]+) "
    r"samples_per_s=([\d.]+) peak_GiB=([\d.]+)"
)
DIAG_RE = re.compile(
    rf"cond_K2 {_NUM}\s+cond_M {_NUM}\s+pinv_resid {_NUM}"
)
GRAD_RE = re.compile(rf"grad_norm {_NUM}")

TRAIN_MODULE = "paper.mqar.train"


# train.py's BooleanOptionalAction flags: for these, False must emit --no-<name>
# (their default may be True, so omitting the flag does NOT turn them off). Every
# other boolean flag is plain store_true, for which argparse rejects a --no- form
# and False must emit nothing. Keep this in sync with build_parser() in train.py.
_OPTIONAL_BOOL_FLAGS = frozenset({"fresh_data", "random_non_queries"})


def _flag(name: str, value) -> list[str]:
    """One CLI token pair, or nothing.

    Booleans: a BooleanOptionalAction flag (see ``_OPTIONAL_BOOL_FLAGS``) emits
    ``--name`` for True and ``--no-name`` for False, so the caller's intent
    survives regardless of the flag's default. A plain store_true flag emits
    ``--name`` for True and NOTHING for False (argparse rejects ``--no-`` there).
    """
    if isinstance(value, bool):
        if name in _OPTIONAL_BOOL_FLAGS:
            return [f"--{name}"] if value else [f"--no-{name}"]
        return [f"--{name}"] if value else []
    return [f"--{name}", str(value)]


def build_cmd(python: str | None = None, extra: list[str] | None = None,
              **opts) -> list[str]:
    """Build the ``python -m paper.mqar.train`` argv.

    Keyword names are the train.py flag names with underscores, e.g.
    ``build_cmd(backend="sdpa", seq_len=256, num_kv_pairs=16)``. ``None`` values
    are dropped so callers can pass through optional settings uniformly.
    ``extra`` is appended verbatim, for drivers that forward argparse's
    ``parse_known_args`` leftovers straight to train.py.
    """
    cmd = [python or sys.executable, "-u", "-m", TRAIN_MODULE]
    for name, value in opts.items():
        if value is None:
            continue
        cmd += _flag(name, value)
    return cmd + list(extra or [])


def parse_output(text: str) -> dict:
    """Extract whatever train.py reported from its stdout.

    Returns a dict with ``recall`` (percent, or None if the run failed) and,
    when present, the training profile and the last diagnostic line's values.
    """
    out: dict = {"recall": None}
    m = BEST_RE.search(text)
    if m:
        out["recall"] = float(m.group(1))

    p = PROF_RE.search(text)
    if p:
        out.update(
            batch=int(p.group(1)), step_ms=float(p.group(2)),
            samples_per_s=float(p.group(3)), peak_GiB=float(p.group(4)),
        )

    diag = DIAG_RE.findall(text)   # last eval line wins
    if diag:
        c_k2, c_m, resid = diag[-1]
        out.update(cond_K2=float(c_k2), cond_M=float(c_m), pinv_resid=float(resid))
    gn = GRAD_RE.findall(text)
    if gn:
        out["grad_norm"] = float(gn[-1])
    return out


def run_train(python: str | None = None, stream: bool = False,
              log_path: str | None = None, extra: list[str] | None = None,
              cwd: str | None = None, **opts) -> dict:
    """Run one training job and return its parsed result.

    stream=True echoes the child's output live (Colab: a captured subprocess
    writes to the OS stdout fd, which the notebook front-end never shows).
    log_path tees the full output to a file. If ``out_json`` is passed through
    to train.py and the file lands, that structured record is merged in and
    wins over the regex-scraped values.
    """
    cmd = build_cmd(python=python, extra=extra, **opts)
    if stream:
        chunks = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                cwd=cwd)
        for line in proc.stdout:
            print(line, end="")
            sys.stdout.flush()
            chunks.append(line)
        proc.wait()
        text = "".join(chunks)
        rc = proc.returncode
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        text = proc.stdout
        rc = proc.returncode

    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(text)

    result = parse_output(text)
    result["returncode"] = rc
    result["output"] = text   # kept so callers can print diagnostics on failure

    # train.py's own structured record is authoritative when it exists. Merge
    # only the fields it actually populated: it writes None for the timing keys
    # on runs too short to profile, which must not clobber a scraped value.
    out_json = opts.get("out_json")
    if out_json and os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as f:
            rec = json.load(f)
        if rec.get("best_recall") is not None:
            rec["recall"] = rec["best_recall"]
        result.update({k: v for k, v in rec.items() if v is not None})
    return result


def sweep_lr(lrs, python: str | None = None, stream: bool = False,
             on_result=None, **opts) -> dict:
    """Run one job per learning rate and return the best by recall.

    This is the Zoology convention every MQAR driver here follows: sweep the LR
    per configuration and keep the best, because the optimum shifts with model
    width and a single fixed LR gives width-confounded numbers. The returned
    dict carries the winning run's fields plus ``lr``; empty if every run failed.
    ``on_result(lr, result)`` is called after each run for progress reporting.
    """
    best: dict = {}
    for lr in lrs:
        res = run_train(python=python, stream=stream, lr=lr, **opts)
        if on_result is not None:
            on_result(lr, res)
        if res.get("recall") is None:
            continue
        if not best or res["recall"] > best["recall"]:
            best = {**res, "lr": lr}
    return best


def finite_diag(result: dict) -> dict:
    """The diagnostic fields of a result, dropping non-finite entries.

    sdpa has no landmarks, so train.py prints nan for cond_K2 / cond_M /
    pinv_resid. Those must not reach a JSON record: json.dump writes a bare
    ``NaN`` token, which strict JSON parsers reject.
    """
    keys = ("cond_K2", "cond_M", "pinv_resid", "grad_norm")
    return {k: result[k] for k in keys
            if isinstance(result.get(k), float) and math.isfinite(result[k])}


def already_done(path: str | None) -> bool:
    """True when a result file already exists, so a resumed sweep can skip it.
    Every driver here is restartable; Colab sessions get reclaimed mid-sweep."""
    return bool(path) and os.path.exists(path)
