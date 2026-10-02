#!/usr/bin/env bash
# Resume the completion workflow after the strictly assembled public campaign.
# SelGenPalm and private oil-palm data are intentionally outside this workflow.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/delete/miniconda3/bin/python3}"
LOG_PREFIX="[post-public]"

echo "$LOG_PREFIX $(date -Is) starting nested guarded campaign"
bash experiments/run_loss_specific_hpo_guarded.sh
echo "$LOG_PREFIX $(date -Is) nested campaign complete; running verification"
"$PYTHON_BIN" -m pytest -q tests
"$PYTHON_BIN" -m compileall -q ccgp experiments analysis figures
git diff --check
echo "$LOG_PREFIX $(date -Is) verification complete"
