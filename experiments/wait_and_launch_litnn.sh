#!/usr/bin/env bash
# Wait until the box's CPUs are free, then launch the full literature-NN S2 campaign.
# "Free" = 1-min load average below THRESHOLD for HOLD consecutive checks (so a brief
# dip does not trigger a multi-hour run). Polls every INTERVAL seconds.
#
# Usage: nohup bash experiments/wait_and_launch_litnn.sh > results/litnn_wait.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

THRESHOLD=${THRESHOLD:-10}      # 24-core box; <10 means >~14 cores idle
HOLD=${HOLD:-3}                 # consecutive good checks required
INTERVAL=${INTERVAL:-300}      # 5 min between checks

echo "[wait] $(date) waiting for load1<$THRESHOLD x$HOLD checks, every ${INTERVAL}s"
good=0
while true; do
    load=$(cut -d' ' -f1 /proc/loadavg)
    li=${load%.*}
    if [ "${li:-99}" -lt "$THRESHOLD" ]; then
        good=$((good+1))
        echo "[wait] $(date) load=$load OK ($good/$HOLD)"
        [ "$good" -ge "$HOLD" ] && break
    else
        [ "$good" -ne 0 ] && echo "[wait] $(date) load=$load reset"
        good=0
    fi
    sleep "$INTERVAL"
done

echo "[wait] $(date) CPUs free (load=$load) -> launching full litnn S2 campaign"
export PYTHONPATH=/home/delete/gs-loss
exec nice -n 10 python3 experiments/run_litnn.py --gpus 0,1 --streams 2
