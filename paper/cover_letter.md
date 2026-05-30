# Cover letter — submission to *G3: Genes|Genomes|Genetics*

Dear Editors,

We submit our manuscript, **"Metric-consistent neural networks for genomic
prediction and selection: a standardized-MSE Pearson loss, affine calibration,
and selection-aware evaluation,"** for consideration as an Investigation /
Genomic Prediction article in *G3*.

Genomic prediction models are trained almost universally by minimizing the mean
squared error, yet they are evaluated — and used to make selection decisions —
with the Pearson correlation between predicted and observed phenotypes and with
the ranking of candidate genotypes. Our manuscript makes this training/evaluation
mismatch explicit and resolvable, and contributes:

1. **A standardized-MSE Pearson loss.** We prove and numerically verify (to
   machine precision) that maximizing the Pearson correlation is exactly
   minimizing a standardized MSE, $\mathrm{MSE}(z_y,z_{\hat y})=2(1-r)$. This
   directly refutes the common assertion that correlation "cannot be used as a
   loss function," and yields a stable, differentiable training objective.
2. **Affine calibration with a closed form.** Because correlation is
   scale-invariant, we prove that a single post-hoc affine map restores
   calibrated predictions with residual error $\sigma_y^2(1-r^2)$ while leaving
   the correlation and all rankings unchanged.
3. **Selection-aware evaluation** (top-$k$ overlap, NDCG@$k$, selection
   differential, relative efficiency) alongside the usual metrics, across
   101 trait–dataset combinations from 12 panels/sources spanning 10 species, with architectures, splits and
   tuning budgets held fixed so that only the loss changes.

Our central finding is deliberately nuanced. Holding architecture, splits and
tuning fixed so that only the loss changes, correlation-consistent and CCC losses
improve predictive ability (Δr ≈ +0.04, Holm-corrected paired Wilcoxon p<10⁻⁷) and
selection-aware metrics (NDCG@10, relative efficiency) over MSE; but the benefit is
strongly architecture-dependent (large for the Transformer and CNN, negligible for
the MLP), only the concordance loss reliably improves calibrated RMSE, and no neural
loss unseats GBLUP or ridge on mean rank. We therefore provide a principled,
reproducible framework — and an honest map of when it helps — rather than a claim
that one loss universally wins.

This work fits *G3*'s scope for computational tools and statistical methodology
for genomic prediction. All benchmark datasets are public (EasyGeSe; the CIMMYT
wheat panel; the SoyNAM population), and we release the complete software (`ccgp`),
the raw and aggregated results, the exact cross-validation partitions, tuned
configurations and a pinned environment at <https://github.com/GBeurier/gs-loss>,
so that every reported number is reproducible; a versioned archive will be deposited
at Zenodo (reserved DOI to be inserted at submission). Per *G3*'s initial-submission
policy any format is accepted; a manuscript prepared in the official GSA G3 template
is also provided (`paper/g3/`).

The manuscript is original, not under consideration elsewhere, and all authors
approve submission. We declare no competing interests.

Thank you for considering our work.

Sincerely,

Grégory Beurier, on behalf of all authors (G. Beurier, D. Cornet, L. Rouan, D. Cros)
CIRAD, UMR AGAP Institut, Montpellier, France · gregory.beurier@cirad.fr
