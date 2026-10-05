Neural genomic predictors are usually trained with mean squared error but
evaluated by Pearson correlation and used for selection ranking. This study
shows that Pearson correlation can be optimized directly through an equivalent
standardized-MSE loss, then evaluates what that alignment changes across 101
public trait-dataset tasks. Pearson training modestly improves predictive
correlation, affine calibration restores phenotypic scale, and upper-tail
selection changes little. The results provide a reproducible framework for
separating correlation, calibration, and selection performance when assessing
loss functions for neural genomic prediction.
It also clarifies why linear baselines remain essential comparators in practical
breeding evaluations.
