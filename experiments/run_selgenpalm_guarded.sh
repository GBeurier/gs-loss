#!/usr/bin/env bash
# Guarded, resumable SelGenPalm campaign on the private data drive.
set -euo pipefail

cd "$(dirname "$0")/.."

# SelGenPalm is intentionally outside the active completion scope.  Keep this
# check before any Python invocation so a queued/in-memory supervisor cannot
# inspect or load private inputs after the scope was narrowed.
SELGENPALM_SKIP_SENTINEL="${SELGENPALM_SKIP_SENTINEL:-results/SKIP_SELGENPALM}"
if [[ -f "$SELGENPALM_SKIP_SENTINEL" ]]; then
  echo "[selgenpalm] skipped: sentinel $SELGENPALM_SKIP_SENTINEL is present"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-/home/delete/miniconda3/bin/python3}"
PHYSICAL_GPU=1
MAX_GPU_UTIL="${MAX_GPU_UTIL:-79}"
MAX_LOAD="${MAX_LOAD:-20}"
MIN_AVAIL_GB="${MIN_AVAIL_GB:-8}"
MAX_SHM_PCT="${MAX_SHM_PCT:-75}"
MIN_SHM_AVAIL_GB="${MIN_SHM_AVAIL_GB:-8}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MODELS="${SELGENPALM_MODELS:-gblup,ridge,mlp,cnn,transformer}"

resources_available() {
  local gpu_util load_one available_kb minimum_kb
  local shm_total shm_used shm_avail shm_pct minimum_shm_bytes
  gpu_util=$(nvidia-smi --id="$PHYSICAL_GPU" --query-gpu=utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' ')
  load_one=$(awk '{print $1}' /proc/loadavg)
  available_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  minimum_kb=$((MIN_AVAIL_GB * 1024 * 1024))
  read -r shm_total shm_used shm_avail shm_pct <<EOF
$(df -B1 --output=size,used,avail,pcent /dev/shm | awk 'NR==2 {gsub(/%/, "", $4); print $1, $2, $3, $4}')
EOF
  minimum_shm_bytes=$((MIN_SHM_AVAIL_GB * 1024 * 1024 * 1024))
  awk -v gpu="$gpu_util" -v max_gpu="$MAX_GPU_UTIL" \
      -v load="$load_one" -v max_load="$MAX_LOAD" \
      -v mem="$available_kb" -v min_mem="$minimum_kb" \
      -v shm_pct="$shm_pct" -v max_shm_pct="$MAX_SHM_PCT" \
      -v shm_avail="$shm_avail" -v min_shm_avail="$minimum_shm_bytes" \
      'BEGIN {exit !(gpu <= max_gpu && load < max_load && mem >= min_mem &&
                     shm_pct <= max_shm_pct && shm_avail >= min_shm_avail)}'
}

wait_for_resources() {
  while ! resources_available; do
    echo "[guard] machine busy; waiting ${POLL_SECONDS}s" >&2
    sleep "$POLL_SECONDS"
  done
}

IFS=',' read -r -a model_array <<< "$MODELS"
for model in "${model_array[@]}"; do
  if "$PYTHON_BIN" - "$model" <<'PY'
import json
import sys
from pathlib import Path
path = Path("results/hpo_selgenpalm.json")
model = sys.argv[1]
done = path.exists() and model in json.loads(path.read_text()).get("params", {})
raise SystemExit(0 if done else 1)
PY
  then
    echo "[hpo] ${model} already cached"
  else
    wait_for_resources
    echo "[hpo] starting ${model} on physical GPU ${PHYSICAL_GPU}"
    nice -n 10 env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" "$PYTHON_BIN" \
      experiments/run_selgenpalm.py hpo --models "$model"
  fi
done

while true; do
  next_cell=$("$PYTHON_BIN" experiments/run_selgenpalm.py fit --models "$MODELS" \
    --list-cells | awk '$2 == "missing" {print $1; exit}')
  if [[ -z "$next_cell" ]]; then
    break
  fi
  wait_for_resources
  echo "[fit] starting cell ${next_cell} on physical GPU ${PHYSICAL_GPU}"
  nice -n 10 env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" "$PYTHON_BIN" \
    experiments/run_selgenpalm.py fit --models "$MODELS" --cell-index "$next_cell"
done

wait_for_resources
"$PYTHON_BIN" experiments/run_selgenpalm.py evaluate --models "$MODELS"
