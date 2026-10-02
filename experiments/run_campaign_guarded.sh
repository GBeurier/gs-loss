#!/usr/bin/env bash
# Resume S2 one cell at a time without competing with other work on the machine.
# A cell starts only when CPU/RAM are healthy and a candidate GPU is below the
# user-defined utilization ceiling. Between cells the script waits rather than
# reserving resources. Restrict GPU_CANDIDATES when another project owns a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-/home/delete/miniconda3/bin/python3}
GPU_CANDIDATES=${GPU_CANDIDATES:-1}
MAX_GPU_UTIL=${MAX_GPU_UTIL:-79}
MAX_LOAD=${MAX_LOAD:-20}
MIN_AVAIL_GB=${MIN_AVAIL_GB:-8}
MAX_SHM_PCT=${MAX_SHM_PCT:-75}
MIN_SHM_AVAIL_GB=${MIN_SHM_AVAIL_GB:-8}
POLL_SECONDS=${POLL_SECONDS:-60}

next_cell() {
    "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import sys
import pandas as pd
from ccgp.config import CAMPAIGN
from experiments.run import unique_keys, make_dataset, traits_for, _shard_path
key_columns = ["dataset", "trait", "model", "loss", "scheme", "repeat",
               "fold", "calibration", "seed"]
jobs = [(key, trait) for key in unique_keys(CAMPAIGN)
        for trait in traits_for(make_dataset(key, CAMPAIGN), CAMPAIGN)]
for i, (key, trait) in enumerate(jobs):
    path = _shard_path(CAMPAIGN, key, trait)
    if not path.exists():
        print(i)
        break
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise SystemExit(f"invalid existing shard {path}: unreadable: {type(exc).__name__}: {exc}")
    missing = sorted(set(key_columns) - set(frame.columns))
    if len(frame) != 900 or missing or frame.duplicated(key_columns).any():
        raise SystemExit(
            f"invalid existing shard {path}: rows={len(frame)} missing_columns={missing} "
            f"duplicate_keys={int(frame.duplicated(key_columns).sum()) if not missing else 'n/a'}")
else:
    print("DONE")
PY
}

choose_gpu() {
    local candidate util
    IFS=',' read -ra candidates <<< "$GPU_CANDIDATES"
    for candidate in "${candidates[@]}"; do
        util=$(nvidia-smi --id="$candidate" --query-gpu=utilization.gpu \
            --format=csv,noheader,nounits | tr -d ' ')
        if [ "$util" -le "$MAX_GPU_UTIL" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

# Read-only diagnostic mode used to verify the resume decision without starting a cell.
if [ "${1:-}" = "--next-cell" ]; then
    next_cell
    exit 0
fi

# This campaign is explicitly restricted to the physical GPU 1.  Do not permit
# an environment override to redirect a resumed cell to GPU 0.
if [ "$GPU_CANDIDATES" != "1" ]; then
    echo "[guard] refusing GPU_CANDIDATES=$GPU_CANDIDATES; this campaign is pinned to physical GPU 1" >&2
    exit 2
fi

mkdir -p results/campaign_guarded_logs
while true; do
    cell=$(next_cell)
    if [ "$cell" = "DONE" ]; then
        echo "[guard] $(date -Is) all campaign cells complete"
        break
    fi
    load=$(cut -d' ' -f1 /proc/loadavg)
    avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
    read -r shm_total shm_used shm_avail shm_pct <<EOF
$(df -B1 --output=size,used,avail,pcent /dev/shm | awk 'NR==2 {gsub(/%/, "", $4); print $1, $2, $3, $4}')
EOF
    shm_avail_gb=$((shm_avail / 1024 / 1024 / 1024))
    gpu=""
    gpu=$(choose_gpu || true)
    if awk -v x="$load" -v max="$MAX_LOAD" 'BEGIN {exit !(x >= max)}' || \
       [ "$avail_gb" -lt "$MIN_AVAIL_GB" ] || [ "$shm_pct" -gt "$MAX_SHM_PCT" ] || \
       [ "$shm_avail_gb" -lt "$MIN_SHM_AVAIL_GB" ] || [ -z "$gpu" ]; then
        echo "[guard] $(date -Is) paused: cell=$cell load=$load avail=${avail_gb}GiB " \
             "shm=${shm_pct}%/${shm_avail_gb}GiB gpu=${gpu:-none<80%}"
        sleep "$POLL_SECONDS"
        continue
    fi
    echo "[guard] $(date -Is) start cell=$cell physical_gpu=$gpu load=$load avail=${avail_gb}GiB " \
         "shm=${shm_pct}%/${shm_avail_gb}GiB"
    CUDA_VISIBLE_DEVICES="$gpu" nice -n 10 "$PYTHON_BIN" experiments/run.py main \
        --preset campaign --gpus 0 --cell-index "$cell" \
        > "results/campaign_guarded_logs/cell_${cell}.log" 2>&1
    echo "[guard] $(date -Is) finished cell=$cell"
done

"$PYTHON_BIN" experiments/run.py main --preset campaign --gpus 0 --assemble
