# ccgp — Correlation-Consistent Genomic Prediction

Reproducible code for the paper *"Metric-consistent neural networks for genomic
prediction and selection: standardized-MSE Pearson loss, affine calibration, and
selection-aware evaluation."*

Genomic prediction is trained almost universally with **MSE**, but evaluated and
used for selection with **Pearson correlation** and **ranking**. This package
makes the training objective consistent with that goal:

1. **Pearson loss = standardized MSE.** For population-standardized `y` and `ŷ`,
   `MSE(z_y, z_ŷ) = 2(1 − r)`, so minimizing a standardized MSE *is* maximizing
   Pearson correlation — a stable, differentiable Pearson loss (Proposition 1).
2. **Affine calibration.** A correlation-trained model has the right shape but the
   wrong scale; the optimal affine map `ŷ = a·p + b` restores RMSE/bias with
   `MSE_min = σ²_y(1 − r²)` while leaving Pearson and all rankings unchanged
   (Proposition 2).
3. **Selection-aware evaluation.** Top-k overlap, precision/recall@k, NDCG@k,
   selection differential and relative efficiency, alongside the usual metrics.

Repository: <https://github.com/GBeurier/selgen-loss>

## Install

```bash
git clone https://github.com/GBeurier/selgen-loss.git
cd selgen-loss
pip install -e .                 # core library (PyTorch, scikit-learn, xgboost, ...)
# R (with BGLR, SoyNAM, rrBLUP, lme4) is used for the wheat/SoyNAM datasets and the
# mixed-model analysis; EasyGeSe data download automatically from Zenodo.
```

## Quick checks

```bash
ccgp datasets        # list public datasets
ccgp verify          # Experiment A: numerical verification of the equivalences
pytest -q            # unit tests for the scientific core
```

## Reproduce the study

```bash
# Experiments B–F (HPO, headline grid, calibration, selection, batch size, splits)
python experiments/run.py all --preset full --gpus 0,1 --streams 2
# Statistical analysis (deltas, bootstrap CIs, Holm/FDR, lme4 mixed model)
python analysis/analyze.py
# Figures 1–6
python figures/make_figures.py
```

Results are written to `results/` as tidy Parquet tables
(`dataset, species, trait, model, loss, calibration, scheme, repeat, fold, <metrics>`).

## Data

| Source | Content | Access |
|---|---|---|
| EasyGeSe (Quesada-Traver et al. 2025) | 10 species, 93 traits, predefined CV folds | Zenodo 15348871 (auto-download) |
| CIMMYT wheat (Crossa et al. 2010) | 599 lines × 1279 markers, 4 environments | R package `BGLR` |
| SoyNAM (Xavier et al. 2016) | ~5500 RILs, 40 families | R package `SoyNAM` |

> The sorghum crop-model-parameter case study (manuscript §4.8) uses **proprietary
> data not included in this repository**; `scripts/sorghum_trial.py` points to a
> local path and is provided for transparency only. All benchmark datasets above
> are public and download/build automatically.

## Layout

```
ccgp/        losses, metrics, calibration, data, splits, models/, experiment, hpo, stats, cli
experiments/ run.py (Exp B–F driver), exp_a_numerical.py
analysis/    analyze.py, lmm.R (lme4 mixed model)
figures/     make_figures.py
paper/       main.tex, refs.bib
tests/       unit tests
```

## License

MIT. See `LICENSE`.
