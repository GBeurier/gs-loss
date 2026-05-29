"""ccgp -- Correlation-Consistent Genomic Prediction.

A lean, reproducible framework for the experiments in
"Metric-consistent neural networks for genomic prediction and selection":
standardized-MSE Pearson loss, affine calibration, and selection-aware
evaluation.

Submodules
----------
losses       Differentiable training objectives (MSE, Pearson/std-MSE, CCC, hybrid).
calibration  Post-hoc calibrators (raw, affine, isotonic).
metrics      Predictive and selection-oriented evaluation metrics.
data         Public dataset loaders (EasyGeSe, CIMMYT wheat, SoyNAM).
splits       Cross-validation schemes (predefined, random, leave-family-out, env-holdout).
models       Genomic predictors (GBLUP/RR-BLUP, Ridge, ElasticNet, RF, XGBoost, MLP, CNN, Transformer).
experiment   Config-driven experiment runner producing tidy results.
stats        Statistical analysis of method deltas.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
