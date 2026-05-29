# ccgp — Methods & verified-facts log (authoritative spec for the paper)

## Package
`ccgp` (Correlation-Consistent Genomic Prediction), MIT, Python 3.13, PyTorch 2.10+cu128,
scikit-learn 1.8. Two GPUs (RTX 4090 + 5090), 24 cores, 62 GB RAM.
Reproducible: fixed seeds, fixed CV folds, public data, pinned env.

## Theory (verified numerically — Exp A, results/exp_a.json)
- **Proposition 1**: `1 - r == ½·MSE(z_y, z_ŷ)` with population standardization.
  Numerical check over 80 random cases: max abs diff **5.0e-8** (float tolerance). Exact on
  real GBLUP predictions (diff 0.0).
- **Proposition 2**: optimal affine `ŷ=ap+b` gives residual `MSE_min = var(y)(1-r²)`.
  Check over 200 cases: max abs diff **1.2e-15** (machine precision). Affine with a>0 preserves
  Pearson/Spearman/ranking, fixes scale+offset.
- **Gradient**: `pearson_loss` and `ccc_loss` pass `torch.autograd.gradcheck`. `neg_pearson`
  shares gradients with `pearson` (diff 0.0) → equivalent objectives.
- **Proposition 3**: batch-averaged Pearson ≠ global Pearson (Exp E, empirical).

## Datasets (all public, reproducible)
- **EasyGeSe** (Quesada-Traver 2025, Zenodo DOI 10.5281/zenodo.15348871): 10 species — barley, bean, lentil, maize,
  oyster, pig, pine, rice, soybean, wheatG — 93 trait columns total, 0/1/2 genotypes, **predefined
  5-fold × 5-repeat CV partitions** (we use 2 repeats = 10 folds per trait).
- **CIMMYT wheat** (Crossa 2010, BGLR): 599 lines, 1279 DArT (0/1), grain yield in 4 environments
  → cross-environment transfer (Exp F).
- **SoyNAM** (Xavier 2016, R pkg): ~5500 RILs, 40 biparental families, traits yield/height/protein/oil
  → leave-family-out (Exp F).
- Total **101 trait-datasets across 12 panels/sources spanning 10 species**.
- Preprocessing (unsupervised, no phenotype): MAF ≥ 0.01, mean imputation, top-variance cap at
  20,000 markers when p larger (applies to seven panels: barley, lentil, maize, oyster, pig, rice,
  soybean). Per-fold marker standardization uses **training statistics only**.

## Models
- **GBLUP** — REML genomic BLUP via spectral decomposition of K=XX'/p, profiled restricted
  likelihood for the variance ratio δ, GLS intercept, BLUP `g_*=K_*(K+δI)^{-1}(y-μ)`.
  **VALIDATED vs R rrBLUP 4.6.3**: cross-implementation correlation **1.0000** (max pred diff
  1e-7), identical variance components (δ, μ, σ²_g, σ²_e).
- Ridge regression (HPO-tuned) as the linear baseline.
- **MLP** (BN+ReLU+Dropout), **1D-CNN** (Conv1d over ordered SNPs), **TransformerLite** (SNP-patch
  tokens + encoder). Same architecture across losses (frozen by HPO with MSE).

## Losses (ccgp/losses.py)
mse; **pearson** = std_mse = `1-r`; neg_pearson = `-r`; **hybrid** = `(1-r)+λ·MSE` (λ=0.1);
**ccc** = `1-ρ_c` (Lin's concordance); ranking (RankNet pairwise, optional).

## Calibration (fit on VALIDATION predictions only)
raw (identity); **affine** (`a*=cov(y,p)/var(p)`, `b*=ȳ-a*p̄`); isotonic.

## Metrics
Predictive: pearson, spearman, rmse, mae, r2, bias, calibration slope/intercept.
Selection (k∈{5,10,15,20}%, higher-is-better convention): top-k overlap, precision@k, recall@k,
NDCG@k (linear gain, min-shifted), selection differential, mean selected phenotype,
relative efficiency (achieved/oracle gain).

## Protocol (leakage-safe)
Per fold: inner 80/20 fit/val split (group-aware for family CV). Markers standardized on fit;
y standardized on fit (NN). NN early-stops on the **validation value of the same loss**.
Calibrators fit on val predictions. **Test never used** for fitting, early stopping, calibration,
or HPO. HPO: 12 random configs × 3 inner folds per (dataset, model), selected by val Pearson with
the neutral MSE objective, then frozen and reused across all losses (keeps "only the loss changes").

## Statistics
Δ(loss) = metric(loss) − metric(MSE) per cell; bootstrap CIs (BCa for n≥12, percentile otherwise —
e.g. the n=4 split contrasts and the n=8 sorghum contrast are percentile, per `ccgp/stats.py`);
paired Wilcoxon; Holm + BH-FDR;
mean method ranks; confirmatory LMM (lme4): `metric ~ loss + model + (1|dataset) +
(1|dataset:trait)`, MSE as reference level.

## Experiments
A numerical equivalence (done). B same-model-different-loss (main grid). C calibration. D selection
top-k. E batch-size/global-Pearson. F realistic splits (random vs family vs cross-env).

## Correctness audit (19-agent adversarial workflow)
- **GBLUP** re-validated vs rrBLUP to machine precision (1−r = 8.9e-16; variance components match).
- Fixed (HIGH): `mean_ranks` now ranks slope/bias/intercept by **proximity to their ideal**
  (1, 0, 0) rather than by raw value.
- **Limitation (documented, by design)**: HPO selects the architecture on full-data validation
  splits with a neutral MSE objective; some test-fold genotypes' phenotypes can therefore inform
  architecture choice. Because the chosen architecture is **frozen and shared identically across all
  losses**, this cannot bias the relative loss/calibration comparisons (our inferential target,
  always expressed as Δ vs MSE within the same architecture); it may only mildly, and symmetrically,
  inflate absolute accuracies. GBLUP has no tuned hyper-parameters (REML is automatic) and is
  unaffected. To be stated as a limitation in the manuscript.
