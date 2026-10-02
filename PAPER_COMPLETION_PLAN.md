# Paper Completion Plan

## Scientific questions and estimands

1. **Controlled loss ablation (primary):** within each dataset, trait and neural
   architecture, compare MSE with Pearson, hybrid and CCC while keeping splits,
   preprocessing and MSE-tuned hyperparameters identical. The primary endpoint
   is raw test Pearson correlation; calibrated error and selection metrics are
   secondary.
2. **Literature-network breadth:** run MLP, CNN, Transformer and protocol-matched
   adaptations of DeepGS, DNNGP, PNNGS and SoyDNGP on all 101 public tasks, each
   with and without Pearson. The four named networks must be called
   *literature-inspired adaptations*, with every structural/input deviation
   disclosed; no reproduction claim is allowed.
3. **Loss-specific tuning sensitivity:** on CIMMYT wheat, retune MSE and Pearson
   separately inside each of five outer folds using identical candidate banks
   and budgets. This answers the practical tuning question without replacing the
   controlled primary ablation.
SelGenPalm and all private oil-palm data are explicitly outside the completion
scope. No SelGenPalm HPO, prediction, evaluation, or private-outcome access is
permitted in this workflow.

## Resource policy and compute order

- Use physical GPU 1 only. Before every new HPO/model cell require GPU utilization
  `<=79%`, one-minute load `<20`, and available RAM `>=8 GiB`; otherwise wait 60 s.
- Run one cell at a time with `nice -n 10` and `CUDA_VISIBLE_DEVICES=1`. Never
  compete with the CIGE process on GPU 0. A running cell may finish; no next cell
  starts while the guard fails.
- Resume, never overwrite, readable completed shards. Order:

  1. finish current public cell 13;
  2. complete 101 public campaign shards, then assemble;
  3. run 30 nested loss-specific wheat cells;
  4. run targeted tests, bytecode compilation and `git diff --check`;
  5. freeze results, regenerate analyses/figures, and complete manuscript audits.

## Compute acceptance checks

- Public: exactly 101 readable shards, 900 rows each and 90,900 assembled rows;
  all seven neural models have 4 losses, 10 test folds and 3 calibrations per task;
  GBLUP/ridge have the expected neutral-loss rows. No duplicate experimental key.
- Nested HPO: 30 selected configurations and 30 readable result shards; MSE and
  Pearson use the same candidate bank and only outer-training observations.
- Run targeted tests, bytecode compilation and `git diff --check` after campaigns.
  Any failure pauses downstream work and is logged; scientific protocol changes
  require explicit review rather than an automatic workaround.

## Analysis after results freeze

- Build one immutable analysis table with provenance, normalized RMSE (phenotype
  SD denominator), raw Pearson as the scale-free primary outcome, and calibrated
  RMSE/NRMSE as scale outcomes.
- Form paired loss-minus-MSE contrasts at dataset × trait × architecture level.
  Report architecture-specific effects, balanced pooled effects, dataset-clustered
  bootstrap intervals, paired tests with Holm correction, and a mixed model with
  loss-by-architecture interaction and dataset/trait random effects. Never treat
  folds as independent replicates.
- Analyze the nested-HPO experiment separately as sensitivity, not pooled with
  shared-HPO results.
- Audit negative affine slopes. Use raw predictions for correlation/ranking claims
  and validation-fitted positive affine calibration for held-out scale metrics.

## Figures, manuscript and review

- Regenerate every result figure from frozen tables. Each figure gets an exported
  source table and a text alternative/caption stating conclusion, uncertainty,
  sample unit and limitations; color is never the only grouping cue.
- Rewrite abstract, Methods, Results, Discussion, supplement, data availability
  and cover letter. Include the balanced seven-network benchmark, shared versus
  loss-specific HPO distinction, without adding SelGenPalm results or private-data
  claims.
- Remove all stale numerical claims and all “exact clone” language. Mirror the
  canonical text into the G3 version only after numbers freeze.
- Finish with four audits: claims-to-table/statistics, architecture/citation
  fidelity, figure legibility/data alternatives, and clean-environment
  reproducibility. Build main paper and supplement without unresolved references.

Completion means that all acceptance checks pass, every reported number is
generated from the frozen outputs, private data remain private, and the final
claims match both confirmatory and sensitivity analyses.
