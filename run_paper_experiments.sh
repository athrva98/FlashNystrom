#!/usr/bin/env bash
# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
#
# Thin wrapper kept for muscle memory. The experiment plan now lives in ONE
# place, run_all_paper_experiments.py, so the two cannot drift apart:
#
#   python run_all_paper_experiments.py --smoke     # FIRST: minutes, finds bugs
#   python run_all_paper_experiments.py             # the real thing
#   python run_all_paper_experiments.py --stages genomics
#   python run_all_paper_experiments.py --list      # print the plan, run nothing
#
# Run the full sweep under tmux or nohup; it is many GPU-hours.
set -eu
exec python run_all_paper_experiments.py ${@:+--stages "$@"}
