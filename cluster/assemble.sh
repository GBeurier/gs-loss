#!/usr/bin/env bash
# Run AFTER the array job finishes: merge all per-cell shards into the final
# results/results_main_campaign.parquet. Reports how many of 101 cells are present.
set -euo pipefail
REPO=${REPO:-$HOME/gs-loss}
PYTHON=${PYTHON:-python}
cd "$REPO"

n=$(ls results/main_campaign_shards/*.parquet 2>/dev/null | wc -l)
echo "shards present: $n / 101"
if [ "$n" -lt 101 ]; then
  echo "WARNING: not all cells done. Missing array indices:"
  "$PYTHON" - <<'PY'
from ccgp.config import CAMPAIGN as C
from experiments.run import unique_keys, traits_for, make_dataset, _shard_path
jobs=[(k,t) for k in unique_keys(C) for t in traits_for(make_dataset(k,C),C)]
missing=[i for i,(k,t) in enumerate(jobs) if not _shard_path(C,k,t).exists()]
print("  re-submit:  sbatch --array="+",".join(map(str,missing))+" cluster/submit_grid.slurm")
PY
fi
"$PYTHON" experiments/run.py main --preset campaign --assemble
echo "Done -> results/results_main_campaign.parquet"
