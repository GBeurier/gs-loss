# Cover letter — submission to *G3: Genes|Genomes|Genetics*

Dear Editors,

We submit our manuscript, **"Metric-consistent neural networks for genomic
prediction and selection: a standardized-MSE Pearson loss, affine calibration,
and selection-aware evaluation,"** for consideration as an Investigation /
Genomic Prediction article in *G3*.

Genomic prediction models are trained almost universally by minimizing the mean
squared error, yet they are evaluated with the Pearson correlation between
predicted and observed phenotypes and used to rank candidate genotypes. Our
manuscript separates three objectives that are often conflated in this workflow:
predictive correlation, calibration to the phenotypic scale and recovery of
selection candidates. It contributes:

1. **A standardized-MSE Pearson loss.** We prove and numerically verify (to
   machine precision) that maximizing the Pearson correlation is exactly
   minimizing a standardized MSE, $\mathrm{MSE}(z_y,z_{\hat y})=2(1-r)$. This
   directly refutes the common assertion that correlation "cannot be used as a
   loss function," and yields a stable, differentiable training objective.
2. **Affine calibration with a closed form.** Because correlation is
   scale-invariant, we prove the optimal affine residual identity and audit the
   practical case where a validation-fitted negative slope can reverse ranking.
3. **A balanced public benchmark** across seven networks, 101 trait–dataset
   tasks and 12 panels, with panel-clustered inference, normalized error,
   upper-tail ranking metrics and a separate loss-specific nested-HPO
   sensitivity analysis.

Our central finding is deliberately nuanced. In the panel-balanced
shared-configuration ablation, Pearson training improves raw test correlation by
only +0.0048 (95% panel-cluster bootstrap interval 0.0008–0.0091), while NDCG@10
and normalized-error changes are smaller. The Transformer has the clearest
positive estimate, DeepGS is a negative counterexample, and the omnibus
loss-by-architecture interaction is inconclusive. Nested tuning on CIMMYT wheat
supports a Transformer signal but also exposes initialization variability. No
neural loss unseats GBLUP or ridge on mean rank. We therefore provide a
reproducible framework for objective alignment, calibration and decision-level
evaluation rather than a universal-win claim for any one loss.

This work fits *G3*'s scope for computational tools and statistical methodology
for genomic prediction. All benchmark datasets are public (EasyGeSe; the CIMMYT
wheat panel; the SoyNAM population), and we release the complete software (`ccgp`),
the raw and aggregated results, cross-validation definitions, tuned
configurations and figure source data at <https://github.com/GBeurier/gs-loss>,
so that every reported number is reproducible; a versioned archive will be deposited
at Zenodo (reserved DOI to be inserted at submission). Per *G3*'s initial-submission
policy any format is accepted; a manuscript prepared in the official GSA G3 template
is also provided (`paper/g3/`).

The manuscript is original, not under consideration elsewhere, and all authors
approve submission. We declare no competing interests.

Thank you for considering our work.

Sincerely,

Grégory Beurier, on behalf of all authors (G. Beurier, D. Cornet, L. Rouan, C. Noûs, D. Cros)
CIRAD, UMR AGAP Institut, Montpellier, France · gregory.beurier@cirad.fr
