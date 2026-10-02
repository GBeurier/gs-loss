#!/usr/bin/env bash
#SBATCH --job-name=gsloss-grid
#SBATCH --array=0-100%32          # 101 cells (0..100); %32 = max 32 concurrent (tune to your quota)
#SBATCH --gres=gpu:1              # 1 GPU per cell (cells are independent)
#SBATCH --cpus-per-task=4         # OMP threads per cell; 4 is plenty (models are small)
#SBATCH --mem=12G                 # per cell; bump to 24G if a node OOMs on the big panels
#SBATCH --time=02:00:00           # per cell wall limit; heaviest cell (rice/soydngp) ~1-1.5h
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err
#
# One SLURM array task = one (dataset,trait) cell of the S2 campaign grid.
# Each task writes results/main_campaign_shards/<cell>.parquet independently, so:
#   - tasks are fully parallel (no shared state)
#   - already-done cells (the 14 from the local run) are skipped automatically
#   - a failed/killed task only loses its own cell; just re-submit that array index
# HPO is NOT run here: results/hpo_campaign.json (135/135) must be present (shipped).
#
# EDIT THESE for your cluster:
REPO=${REPO:-$HOME/gs-loss}                 # repo location on the cluster
PYTHON=${PYTHON:-python}                     # python with the env (see handoff doc)
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$REPO"
mkdir -p logs results/main_campaign_shards

echo "[task $SLURM_ARRAY_TASK_ID] $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(date)"
# --gpus 0 because SLURM already pins one GPU via CUDA_VISIBLE_DEVICES; index 0 = that GPU.
srun "$PYTHON" experiments/run.py main --preset campaign --gpus 0 \
     --cell-index "$SLURM_ARRAY_TASK_ID"
echo "[task $SLURM_ARRAY_TASK_ID] done $(date)"
