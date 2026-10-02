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
3. **Upper-tail evaluation.** Top-k overlap and NDCG complement the usual
   metrics. They recover the largest observed phenotypes and require
   trait-specific desirability directions before interpretation as breeding gain.

Repository: <https://github.com/GBeurier/gs-loss>

## Install

```bash
git clone https://github.com/GBeurier/gs-loss.git
cd gs-loss
pip install -e .                 # core library (PyTorch, scikit-learn, xgboost, ...)
# R (with BGLR, SoyNAM and rrBLUP) supplies two datasets and a numerical baseline
# check; EasyGeSe downloads automatically from Zenodo.
```

## Quick checks

```bash
ccgp datasets        # list public datasets
ccgp verify          # Experiment A: numerical verification of the equivalences
pytest -q            # unit tests for the scientific core
```

## Reproduce the study

```bash
# Balanced seven-network, 101-task shared-configuration campaign
python experiments/run.py all --preset campaign --gpus 0,1 --streams 2
# Three-network loss-specific nested-HPO sensitivity on CIMMYT wheat
python experiments/run_loss_specific_hpo.py all
# Panel-balanced analysis and interaction sensitivity
python analysis/final_analysis.py
python analysis/mixed_model_final.py
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

All datasets used in the reported benchmark are public. No private-data result is
included in the manuscript or Supporting Information.

## Layout

```
ccgp/        losses, metrics, calibration, data, splits, models/, experiment, hpo, stats, cli
experiments/ run.py (Exp B–F driver), exp_a_numerical.py
analysis/    final_analysis.py, mixed_model_final.py, supplement generator
figures/     make_figures.py, source_data/, text alternatives
paper/       main.tex, refs.bib, cover_letter.md; g3/ (GSA G3 two-column build)
tests/       unit tests
```

## Manuscript builds

| Build | File | How |
|---|---|---|
| Submission (single-column, figures inline) | `paper/main.pdf` | `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` |
| bioRxiv preprint (line-numbered) | `paper/main_preprint.pdf` | `pdflatex --jobname=main_preprint "\def\PREPRINTMODE{1}\input{main}"` (+ bibtex, ×2) |
| Official GSA **G3** template (two-column) | `paper/g3/main_g3.pdf` | `cd paper/g3 && pdflatex main_g3 && bibtex main_g3 && pdflatex main_g3 && pdflatex main_g3` |
| Supporting Information (File S1) | `paper/supplement.pdf` | `python analysis/make_final_supplement.py && cd paper && pdflatex supplement && bibtex supplement && pdflatex supplement && pdflatex supplement` |

`paper/supplement.pdf` is generated from the frozen campaign and nested-HPO
outputs by `analysis/make_final_supplement.py`. The cover letter is
`paper/cover_letter.md`. G3 accepts any format for initial
submission; the `paper/g3/` build follows the official `gsag3jnl` template
(`articletype{gs}`, Genomic Selection).

For manuscript edits, share the entire `paper/` directory and preserve its folder
structure. The entry points are `main.tex` (submission), `g3/main_g3.tex` (G3),
and `supplement.tex` (Supporting Information). Both manuscript versions share
`abstract_body.tex`, `sec_results.tex`, and `discussion_body.tex`; the G3 version
imports them from its parent directory. Update both entry points when editing
their separately maintained sections, and keep both `refs.bib` files in sync.
The directory also includes the figure PDFs, supplementary table fragments,
G3 class, bibliography style, styles, and logos needed to compile. The checked-in
table fragments allow the supplement to compile directly. Figure descriptions
and underlying data are available in `figures/text_alternatives.md` and
`figures/source_data/`.

## License

MIT. See `LICENSE`.
