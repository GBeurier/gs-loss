"""Experiment configuration (datasets, models, losses, budgets).

Two presets: ``SMOKE`` (tiny, for CI/debugging) and ``FULL`` (the paper run).
Everything here is deterministic given the seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .data import EASYGESE_SPECIES, SOYNAM_TRAITS

# Core losses for the headline comparison. ``pearson`` == ``std_mse`` (Prop. 1);
# ``neg_pearson`` shares gradients with ``pearson`` -- both are checked in Exp A
# rather than re-run across the whole grid.
CORE_LOSSES = ["mse", "pearson", "hybrid", "ccc"]
NN_ARCHS = ["mlp", "cnn"]
# Canonical genomic-prediction baselines: GBLUP (kernel/REML) + ridge (RR-BLUP).
# Tree ensembles (RF/XGBoost) and coordinate-descent ElasticNet are supported by
# make_classical but excluded from the default grid -- they are uncompetitive on
# and/or prohibitively slow over 20k-marker panels.
CLASSICAL = ["gblup", "ridge"]


@dataclass
class GridConfig:
    name: str = "full"
    easygese_species: list = field(default_factory=lambda: list(EASYGESE_SPECIES))
    max_traits_per_species: int | None = None          # None = all traits
    classical_models: list = field(default_factory=lambda: list(CLASSICAL))
    nn_archs: list = field(default_factory=lambda: list(NN_ARCHS))
    transformer_species: list = field(default_factory=lambda: ["lentil", "soybean", "rice", "maize"])
    losses: list = field(default_factory=lambda: list(CORE_LOSSES))
    hybrid_lam: float = 0.1
    fracs: tuple = (0.05, 0.10, 0.15, 0.20)
    calibrators: tuple = ("raw", "affine", "isotonic")
    n_repeats: int = 5                                  # predefined CV repeats to use (<=5)
    maf: float = 0.01
    max_features: int = 20000
    hpo_trials: int = 12
    hpo_folds: int = 3
    soynam_traits: list = field(default_factory=lambda: list(SOYNAM_TRAITS))
    soynam_max_families: int = 20                        # leave-family-out folds
    seed: int = 0


SMOKE = GridConfig(
    name="smoke",
    easygese_species=["lentil", "bean"],
    max_traits_per_species=2,
    classical_models=["gblup", "ridge"],
    nn_archs=["mlp"],
    transformer_species=[],
    losses=["mse", "pearson", "hybrid"],
    n_repeats=1,
    hpo_trials=4,
    hpo_folds=2,
    soynam_traits=["yield"],
    soynam_max_families=4,
)

# Use all 93 EasyGeSe traits (+ 4 wheat envs + 4 SoyNAM traits = 101
# trait-datasets across 12 species). 2 CV repeats x 5 folds = 10 evaluations per
# trait x 101 traits is ample statistical power; HPO is light (8 trials x 2
# folds) since the architecture only needs to be reasonable and is shared across
# losses.
FULL = GridConfig(name="full", max_traits_per_species=None, n_repeats=2,
                  hpo_trials=8, hpo_folds=2)
