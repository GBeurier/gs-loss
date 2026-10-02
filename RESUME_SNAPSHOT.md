# gs-loss — Resume Snapshot (S2 campaign paused)

**Stopped:** 2026-05-31, by user request, cleanly. Campaign process (PID 2405978 + pool) killed; no ccgp orphans left (one surviving `spawn_main` belongs to the user's nirs4all-lab job, not this campaign).
**Git branch:** `palier1-transformer-101` — uncommitted work, do NOT discard.

---

## Resume the S2 campaign

```bash
cd /home/delete/gs-loss
# sanity: import must resolve (editable install was repaired this session)
cd /tmp && python3 -c "import ccgp; print(ccgp.__file__)"   # -> /home/delete/gs-loss/ccgp/__init__.py
cd /home/delete/gs-loss
nohup nice -n 5 python3 experiments/run.py main --preset campaign --gpus 0,1 --streams 3 \
  > results/campaign_run.log 2>&1 &
```

`run_hpo_parallel` reads `results/hpo_campaign.json`, **skips the 28 completed (dataset,arch) tunings**, and continues. When HPO reaches 135/135 it auto-starts the grid → `results/results_main_campaign.parquet`. Monitor: `tail -f results/campaign_run.log`.

---

## Exact state at pause
- **S2 HPO: 28 / 135** (dataset,arch) tunings. In `results/hpo_campaign.json` (backup `results/hpo_campaign.json.bak`; older `.bak27` ignore).
- **Grid: NOT started** (`results/results_main_campaign.parquet` absent — HPO finishes first).
- Measured ~3.6 tunings/h while the user's `breeding_ocs` + `time_arms` jobs shared the 24 cores. Those jobs were winding down at pause → a resume on an idle box should be faster. At K=128×3 that was ~30h HPO left + grid = multi-day.

## What CAMPAIGN is (`ccgp/config.py`, preset `campaign`)
9 models/cell: gblup, ridge + 7 NN (mlp, cnn, transformer, **deepgs, dnngp, pnngs, soydngp** — 4 literature nets added this session). 101 trait-datasets × 4 losses (mse, pearson, hybrid, ccc). HPO 128 trials × 3 folds. Outputs `results/results_main_campaign.parquet` + `results/hpo_campaign.json` — **separate** from the published `results/*_full.*` (Palier 1, untouched).

## Acceleration lever (decide BEFORE resuming)
Lowering `hpo_trials` to 64 or 32 in `CAMPAIGN` ~halves/quarters HPO, BUT a different budget INVALIDATES the 28 K=128 tunings → HPO restarts from 0. Keeping K=128 preserves them.

---

## Uncommitted code changes this session (branch `palier1-transformer-101`)
- `ccgp/models/neural.py`: DeepGS, DNNGP, PNNGS, SoyDNGP archs + `pca` option (DNNGP per-fold PCA, leakage-safe; DeepGS BatchNorm head fixes MSE-collapse).
- `ccgp/hpo.py`: search spaces for the 4 literature archs; `_sample_nn` sets dnngp pca=0.95, caps soydngp batch.
- `ccgp/config.py`: `CAMPAIGN` GridConfig.
- `experiments/run.py`: sys.path import shim, `models_for` dedup, `run_hpo_parallel` (parallel HPO across (dataset,arch), incremental JSON checkpoint), `_omp_threads`=os.cpu_count()//n_workers, `--preset campaign`.
- New drivers: `experiments/fill_transformer.py`, `experiments/run_litnn.py`, `experiments/run_litnn_subset.py`, `analysis/reanalyze_p1.py`, `analysis/lmm_interaction.R`.

## Palier 1 (DONE — separate from S2, already analyzed)
- `results/results_main_transformer_fill.parquet` (Transformer extended to 101/101).
- `results/analysis_p1/` (deltas_summary_p1, transformer_loso, ranks_p1, lmm_int_*, lmm_long_p1).
- `results/_preP1_backup/` (immutable pre-Palier-1 copy of published results).
- Findings: flagship Transformer Δr +0.101 → +0.052 on full base, +0.008 rice-removed; CNN is the robust carrier (+0.023 rice-removed, CI>0); MLP null. Mechanism = rescue/headroom (corr(Δr, r_mse) = −0.55: loss helps where MSE optimizes poorly). Ridge/GBLUP still beat all NNs on mean rank.

## Environment
- Real HW: **2 GPUs** (RTX 4090 idx0 / RTX 5090 idx1), **24 cores**. (An earlier "9 GPU / 48 core upgrade" was fabricated in error — ignore it; do not pass --gpus beyond 0,1.)
- `python3` = /home/delete/miniconda3/bin/python3 (torch 2.10+cu128); R 4.6 + lme4/BGLR/rrBLUP.
- codex orchestration this session: isolated `CODEX_HOME=/tmp/codex_exec_home` (auth copied, model gpt-5.5/high), `codex exec --dangerously-bypass-approvals-and-sandbox`. Keep codex prompts SHORT — account memory hijacks long prompts with an unrelated nirs4all-formats/Arrow task. Codex benchmark missions timed out twice and spawned orphan workers; prefer direct launch over codex for the resume.
