#!/usr/bin/env bash
# Reproducible public -> nested-HPO -> verification completion sequence.
# SelGenPalm and private oil-palm data are intentionally out of scope.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/delete/miniconda3/bin/python3}"

bash experiments/run_campaign_guarded.sh
bash experiments/run_loss_specific_hpo_guarded.sh
"$PYTHON_BIN" -m pytest -q tests
"$PYTHON_BIN" -m compileall -q ccgp experiments analysis figures
git diff --check
