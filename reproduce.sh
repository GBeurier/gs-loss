#!/usr/bin/env bash
# End-to-end reproduction of the ccgp study.
# Requires: Python 3.11+, the dependencies in pyproject.toml, and R with
# BGLR, SoyNAM, rrBLUP and lme4. Public data download automatically.
set -euo pipefail
cd "$(dirname "$0")"

pip install -e . --no-deps

# Experiment A: numerical verification of the equivalences (Propositions 1-2).
ccgp verify

# Experiments B-F: HPO, headline grid, calibration, selection, batch size, splits.
python experiments/run.py all --preset full --gpus 0,1 --streams 3

# Statistical analysis (deltas, bootstrap CIs, Holm/FDR, lme4 mixed model).
python analysis/analyze.py full

# Figures 1-6.
python figures/make_figures.py
mkdir -p paper/figures && cp figures/fig*.pdf paper/figures/

# Manuscript.
( cd paper && pdflatex -interaction=nonstopmode main.tex \
    && bibtex main && pdflatex -interaction=nonstopmode main.tex \
    && pdflatex -interaction=nonstopmode main.tex )

echo "Done. Manuscript at paper/main.pdf"
