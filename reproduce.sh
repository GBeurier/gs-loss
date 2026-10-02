#!/usr/bin/env bash
# End-to-end reproduction of the ccgp study.
# Requires Python 3.11+, R with BGLR/SoyNAM/rrBLUP, and a CUDA GPU for the full
# neural campaign. Public data download automatically.
set -euo pipefail
cd "$(dirname "$0")"

python -m pip install -e '.[dev]'

# Experiment A: numerical verification of the equivalences (Propositions 1-2).
ccgp verify

# Balanced seven-network, 101-task shared-configuration ablation.
python experiments/run.py all --preset campaign --gpus 0,1 --streams 2

# Loss-specific nested-HPO sensitivity.
python experiments/run_loss_specific_hpo.py all

# Panel-balanced inference and corroborative interaction models.
python analysis/final_analysis.py
python analysis/mixed_model_final.py
python analysis/make_final_supplement.py

# Figures 1-6.
python figures/make_figures.py
mkdir -p paper/figures && cp figures/fig*.pdf paper/figures/
mkdir -p paper/g3/figures && cp figures/fig*.pdf paper/g3/figures/

# Manuscript.
( cd paper && pdflatex -interaction=nonstopmode main.tex \
    && bibtex main && pdflatex -interaction=nonstopmode main.tex \
    && pdflatex -interaction=nonstopmode main.tex \
    && pdflatex -interaction=nonstopmode -jobname=main_preprint '\def\PREPRINTMODE{1}\input{main}' \
    && bibtex main_preprint \
    && pdflatex -interaction=nonstopmode -jobname=main_preprint '\def\PREPRINTMODE{1}\input{main}' \
    && pdflatex -interaction=nonstopmode -jobname=main_preprint '\def\PREPRINTMODE{1}\input{main}' \
    && pdflatex -interaction=nonstopmode supplement.tex \
    && bibtex supplement && pdflatex -interaction=nonstopmode supplement.tex \
    && pdflatex -interaction=nonstopmode supplement.tex )

( cd paper/g3 && pdflatex -interaction=nonstopmode main_g3.tex \
    && bibtex main_g3 && pdflatex -interaction=nonstopmode main_g3.tex \
    && pdflatex -interaction=nonstopmode main_g3.tex )

python analysis/make_repro_manifest.py

echo "Done: paper/main.pdf, paper/main_preprint.pdf, paper/supplement.pdf, paper/g3/main_g3.pdf"
