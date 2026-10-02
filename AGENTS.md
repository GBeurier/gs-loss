# Repository Guidelines

## Project Structure & Module Organization

`ccgp/` is the installable Python package. Core losses, metrics, calibration, data loading, and split logic live directly in that package; model implementations are under `ccgp/models/`. Experiment drivers are in `experiments/`, statistical workflows in `analysis/`, and reusable utilities in `scripts/`. Tests belong in `tests/`. Figure-generation code and outputs are in `figures/`, while manuscript sources and publication builds are under `paper/`. Treat `results/` tables and generated PDFs/PNGs as reproducibility artifacts: update them only when the corresponding experiment or build is intentionally rerun.

## Build, Test, and Development Commands

- `python -m pip install -e '.[dev]'` installs the package, CLI, and Pytest in editable mode.
- `pytest -q` runs the fast scientific-core test suite.
- `ccgp verify` checks the numerical equivalences described by the project.
- `python experiments/run.py all --preset full --gpus 0,1 --streams 2` reproduces experiments B–F; adjust GPU and stream values for local hardware.
- `python analysis/analyze.py` regenerates statistical summaries, and `python figures/make_figures.py` rebuilds Figures 1–6.
- `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` builds the primary manuscript.

Full experiments are expensive and may download public datasets. Use focused tests or a smaller experiment preset while developing.

## Coding Style & Naming Conventions

Use Python 3.11+, four-space indentation, and a maximum line length of 110 characters (configured in `pyproject.toml`). Follow existing conventions: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and descriptive CLI option names. Keep scientific routines deterministic by passing explicit seeds. Ruff is the configured linter; run `ruff check .` when available. Add docstrings where assumptions, formulas, or array shapes are not obvious.

## Testing Guidelines

Pytest discovers `test_*.py` files and `test_*` functions. Add focused regression tests to `tests/`, using fixed NumPy or Torch seeds and tolerances appropriate to floating-point calculations. There is no stated coverage threshold; prioritize losses, calibration invariants, ranking metrics, splits, and model interfaces. Run `pytest -q` before submitting changes.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative summaries such as `Add Supporting Information` or `Strengthen Exp F`; optional component prefixes such as `ccgp:` are acceptable. Keep each commit scoped to one logical change. Pull requests should explain the scientific or implementation motivation, list validation commands, link relevant issues, and identify regenerated results, figures, or PDFs. Include before/after figures when visual output changes, and note any proprietary-data dependency that reviewers cannot reproduce.
