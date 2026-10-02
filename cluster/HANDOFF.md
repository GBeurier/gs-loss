# gs-loss S2 grid — cluster handoff

Goal: finish the **S2 campaign grid** (the test-set evaluation of 9 models × 4 losses ×
101 trait-datasets) on a SLURM cluster, because it is ~200h on one local GPU but
**embarrassingly parallel** — 101 independent cells → a SLURM array job finishes in a
few hours.

The expensive HPO phase (135/135 tunings) is **already done** and shipped; the cluster
only runs the grid. 14 of 101 cells are also already done (from the local run) and will
be auto-skipped.

---

## What to ship
Tarball: `/tmp/gsloss_cluster_handoff.tar.gz` (~3.6 GB, dominated by `data_cache/`).
Contents:
- `ccgp/`, `experiments/`, `analysis/`, `pyproject.toml` — the package + runner
- `cluster/` — the SLURM scripts (this dir)
- `results/hpo_campaign.json` — **the 135/135 HPO cache (required; grid won't tune)**
- `results/main_campaign_shards/` — the 14 already-computed cells (auto-skipped)
- `data_cache/` — the 15 preprocessed dataset pickles (so nodes don't rebuild/download)

Copy to the cluster, e.g.:
```bash
scp /tmp/gsloss_cluster_handoff.tar.gz user@cluster:~/
ssh user@cluster 'mkdir -p ~/gs-loss && tar xzf ~/gsloss_cluster_handoff.tar.gz -C ~/gs-loss'
```

## Environment on the cluster
Python 3.13, torch 2.10 (cu128), scikit-learn 1.8, pandas 2.3, plus pyarrow/scipy/xgboost/statsmodels.
Easiest: recreate with conda/uv from `pyproject.toml`, then `pip install -e .` inside the repo so
`import ccgp` works. Verify:
```bash
cd ~/gs-loss && python -c "import ccgp, torch; print(torch.__version__, torch.cuda.is_available())"
```
(If the cluster has its own torch module, `module load` it and `pip install -e . --no-deps`.)

## Run it
```bash
cd ~/gs-loss
export REPO=$PWD PYTHON=$(which python)
sbatch cluster/submit_grid.slurm          # array 0-100, 1 GPU/cell, max 32 concurrent
```
- Each array task runs ONE cell and writes `results/main_campaign_shards/<cell>.parquet`.
- The 14 done cells skip instantly. Failed/killed tasks lose only their own cell.
- Tune in `submit_grid.slurm`: `--array=...%N` (concurrency vs your quota), `--mem`, `--time`.

## When the array finishes
```bash
cd ~/gs-loss && bash cluster/assemble.sh   # merges shards -> results/results_main_campaign.parquet
```
If some cells failed, `assemble.sh` prints the exact `sbatch --array=<missing>` line to re-run them.

## Bring results back
Only `results/results_main_campaign.parquet` (and optionally the shards) need to come back:
```bash
scp user@cluster:~/gs-loss/results/results_main_campaign.parquet ./results/
```
Then the local analysis (`analysis/reanalyze_p1.py` style + the LMM/LOSO scripts) runs on it to
answer: **does the Pearson loss help the literature NNs (DeepGS/DNNGP/PNNGS/SoyDNGP), and does
PNNGS's "Pearson can't be a loss" claim fall?**

---

## Cell index reference
- 101 cells, index 0-100, deterministic order = `unique_keys(CAMPAIGN) × traits`.
- `python experiments/run.py main --preset campaign --gpus 0 --cell-index I` runs cell I.
- Already-done at handoff: indices 0-12 and 14 (barley/bean/lentil + 1 maize).
- Heaviest cells: rice (36 traits, low r → big Pearson-loss effect expected), pine (17), wheatG (16);
  soydngp is the slowest arch (~28s/fit). Give those `--time` headroom.

## Key facts / gotchas
- **Use the right python**: locally it's `/home/delete/miniconda3/bin/python3` (NOT /usr/bin/python3,
  which lacks torch). On the cluster, whatever python has the env.
- Grid does NOT run HPO — it reads `results/hpo_campaign.json`. If that file is missing the run aborts
  with a clear message.
- Partial results so far (14 easy-species cells) show ~0 Pearson-loss effect — EXPECTED, because those
  are high-r "easy" species; the headroom mechanism predicts the gain appears on the hard species
  (rice etc.) still to be computed. Don't conclude from the 14 alone.
