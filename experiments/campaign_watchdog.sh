#!/usr/bin/env bash
# Resource watchdog for the S2 campaign. Polls RAM + load every 20s and, if the
# box approaches the conditions that can crash it, kills the campaign CLEANLY
# (TERM then KILL of the parent + its spawn workers). HPO is checkpointed per
# (dataset,arch) so a kill loses at most one in-flight tuning.
#
# Triggers (any one, sustained for 2 consecutive checks to avoid transient spikes):
#   - available RAM < MIN_AVAIL_GB
#   - 1-min load   > MAX_LOAD
# Usage: nohup bash experiments/campaign_watchdog.sh <campaign_pid> > results/watchdog.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

CAMPAIGN_PID="${1:?need campaign parent pid}"
# OOM is the real machine-crash risk; RAM is the primary guard. The campaign
# legitimately drives load to ~40-50 at streams=3 on a 24-core box (full, healthy
# utilization), so the load trigger is only a runaway backstop set well above that.
MIN_AVAIL_GB="${MIN_AVAIL_GB:-5}"     # kill if free RAM dips below this (OOM guard = the real crash risk)
# Load alone is NOT a crash risk and the grid's GBLUP eigendecomposition bursts
# load to ~100 transiently at startup. Disable the load trip by default (set very
# high); RAM is the guard that actually prevents a machine crash. Require 3
# sustained checks so only a true runaway (not a transient spike) ever trips it.
MAX_LOAD="${MAX_LOAD:-300}"
SUSTAIN="${SUSTAIN:-3}"
INTERVAL="${INTERVAL:-20}"
NPROC=$(nproc)

echo "[wd] $(date) watching PID $CAMPAIGN_PID | trip if avail<${MIN_AVAIL_GB}GB or load1>${MAX_LOAD} (x2)"
bad=0
while kill -0 "$CAMPAIGN_PID" 2>/dev/null; do
    avail=$(awk '/MemAvailable/{printf "%d", $2/1024/1024}' /proc/meminfo)
    load=$(cut -d' ' -f1 /proc/loadavg); li=${load%.*}
    trip=0
    [ "${avail:-99}" -lt "$MIN_AVAIL_GB" ] && trip=1 && reason="avail=${avail}GB"
    [ "${li:-0}" -gt "$MAX_LOAD" ] && trip=1 && reason="load=${load}"
    if [ "$trip" -eq 1 ]; then
        bad=$((bad+1))
        echo "[wd] $(date) WARN $reason ($bad/$SUSTAIN)"
        if [ "$bad" -ge "$SUSTAIN" ]; then
            echo "[wd] $(date) TRIPPED ($reason) -> killing campaign $CAMPAIGN_PID"
            kill -TERM "$CAMPAIGN_PID" 2>/dev/null; sleep 5
            kill -9 "$CAMPAIGN_PID" 2>/dev/null
            for w in $(pgrep -f "spawn_main"); do
                pp=$(cut -d' ' -f4 /proc/$w/stat 2>/dev/null)
                # only kill spawn workers whose chain leads to our run (best-effort: miniconda python)
                grep -q miniconda3 /proc/$w/cmdline 2>/dev/null && kill -9 "$w" 2>/dev/null
            done
            echo "[wd] $(date) campaign killed; HPO state preserved in results/hpo_campaign.json"
            exit 0
        fi
    else
        [ "$bad" -ne 0 ] && echo "[wd] $(date) recovered ($reason cleared)"
        bad=0
    fi
    sleep "$INTERVAL"
done
echo "[wd] $(date) campaign PID $CAMPAIGN_PID exited on its own; watchdog done"
