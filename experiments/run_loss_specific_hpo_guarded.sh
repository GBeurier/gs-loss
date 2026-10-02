#!/usr/bin/env bash
# Resume the loss-specific nested-HPO sensitivity grid one cell at a time.
# This intentionally serializes inner HPO and the outer-fold refit: the same
# physical GPU is shared with other work on the machine.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-/home/delete/miniconda3/bin/python3}
PHYSICAL_GPU=1
MAX_GPU_UTIL=${MAX_GPU_UTIL:-79}
MAX_LOAD=${MAX_LOAD:-20}
MIN_AVAIL_GB=${MIN_AVAIL_GB:-8}
MAX_SHM_PCT=${MAX_SHM_PCT:-75}
MIN_SHM_AVAIL_GB=${MIN_SHM_AVAIL_GB:-8}
POLL_SECONDS=${POLL_SECONDS:-60}

resources_available() {
    local gpu_util load_one avail_gb shm_total shm_used shm_avail shm_pct shm_avail_gb
    gpu_util=$(nvidia-smi --id="$PHYSICAL_GPU" --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' ')
    load_one=$(awk '{print $1}' /proc/loadavg)
    avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
    read -r shm_total shm_used shm_avail shm_pct <<EOF
$(df -B1 --output=size,used,avail,pcent /dev/shm | awk 'NR==2 {gsub(/%/, "", $4); print $1, $2, $3, $4}')
EOF
    shm_avail_gb=$((shm_avail / 1024 / 1024 / 1024))
    awk -v gpu="$gpu_util" -v max_gpu="$MAX_GPU_UTIL" \
        -v current_load="$load_one" -v max_load="$MAX_LOAD" \
        -v mem="$avail_gb" -v min_mem="$MIN_AVAIL_GB" \
        -v shm_pct="$shm_pct" -v max_shm_pct="$MAX_SHM_PCT" \
        -v shm_avail="$shm_avail_gb" -v min_shm_avail="$MIN_SHM_AVAIL_GB" \
        'BEGIN {exit !(gpu <= max_gpu && current_load < max_load && mem >= min_mem &&
                       shm_pct <= max_shm_pct && shm_avail >= min_shm_avail)}'
}

while true; do
    # List columns are: index, HPO status, outer-result status, model, loss, fold.
    next_cell=$("$PYTHON_BIN" experiments/run_loss_specific_hpo.py hpo --list-cells |
        awk '$3 != "main_done" {print $1; exit}')
    if [[ -z "$next_cell" ]]; then
        break
    fi
    while ! resources_available; do
        echo "[nested-guard] $(date -Is) paused: waiting ${POLL_SECONDS}s" >&2
        sleep "$POLL_SECONDS"
    done
    echo "[nested-guard] $(date -Is) start cell=$next_cell physical_gpu=$PHYSICAL_GPU"
    nice -n 10 env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" "$PYTHON_BIN" \
        experiments/run_loss_specific_hpo.py hpo --cell-index "$next_cell"
    nice -n 10 env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" "$PYTHON_BIN" \
        experiments/run_loss_specific_hpo.py main --cell-index "$next_cell"
    echo "[nested-guard] $(date -Is) finished cell=$next_cell"
done

"$PYTHON_BIN" experiments/run_loss_specific_hpo.py assemble
