# Figure text alternatives

## Figure 1

Five hundred simulations show no fixed relation between raw MSE and Pearson
correlation, whereas standardized MSE lies exactly on the line `2(1-r)`.

## Figure 2

A simulated prediction is strongly correlated but incorrectly scaled. A fitted
affine map reduces RMSE and aligns predictions with observations while leaving
the top-20% overlap unchanged.

## Figure 3

The 101 tasks span 12 panels with substantial variation in training size,
marker count and GBLUP predictive ability. Rice, pine and wheatG contribute most
traits, motivating panel-balanced inference.

## Figure 4

The comparison follows three strategies: raw predictions from MSE training,
raw predictions from Pearson training, and affine-calibrated predictions from
Pearson training. Relative to raw MSE, raw Pearson predictions improve
correlation by 0.0048 but worsen normalized RMSE by 0.346 because their scale is
unidentified. Validation-fitted affine calibration retains a 0.0047 correlation
gain and changes normalized RMSE by -0.0033, effectively restoring the MSE
baseline. Small points show the 12 individual panel contrasts; uncertainty bars
are 95% panel-cluster bootstrap intervals. The architecture panel includes all
seven neural networks. Pearson-minus-MSE correlation contrasts range from
-0.0067 for DeepGS to +0.0176 for Transformer; PNNGS is +0.0096, SoyDNGP
+0.0056, MLP +0.0044, DNNGP +0.0022 and CNN +0.0008. The global interaction is
inconclusive (`p=0.414`).

## Figure 5

Across Pearson-trained task-model cells, affine calibration lowers normalized
error and moves calibration slopes toward one. Most correlations are unchanged,
but 12 of 7,014 comparable fold cells reverse sign because the validation-fitted
affine slope is negative.

## Figure 6

The main shared-HPO panel includes all seven neural networks. Loss-specific
nested tuning was run only for MLP, CNN and Transformer on CIMMYT wheat; there
are no nested-HPO estimates for DeepGS, DNNGP, PNNGS or SoyDNGP. Within the
three-network sensitivity, Transformer is positive, MLP near zero and CNN
negative on average. CNN changes direction across three seeds while Transformer
stays positive; the experiment represents one panel and is descriptive.
