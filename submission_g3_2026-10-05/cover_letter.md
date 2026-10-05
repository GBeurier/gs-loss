# Cover letter — G3: Genes|Genomes|Genetics

5 October 2026

Dear Editors,

We submit our manuscript, “Metric-consistent neural networks for genomic prediction and selection: a standardized-MSE Pearson loss, affine calibration, and selection-aware evaluation,” for consideration as an Investigation in G3: Genes|Genomes|Genetics, in the area of genomic prediction and selection.

Genomic predictors are commonly trained with mean squared error, assessed by Pearson correlation, and used to rank candidates. Our study separates three targets that are often conflated: predictive correlation, calibration to the phenotypic scale, and recovery of selection candidates. We establish the standardized-MSE identity MSE(z_y, z_prediction) = 2(1 − r), implement a differentiable Pearson objective, and evaluate objective alignment alongside validation-fitted calibration and upper-tail ranking. Numerical checks agree to single-precision tolerance for the implemented loss and to float64 precision for the affine residual identity.

The public benchmark compares four losses across seven neural architectures on 101 trait–dataset tasks from 12 panels spanning ten species. Its panel-balanced Pearson-minus-MSE correlation estimate is small: +0.0048 (95% panel-cluster bootstrap interval 0.0008–0.0091; exact 12-panel sign-flip p = 0.052). Raw Pearson predictions are poorly scaled; validation-fitted affine calibration removes the scale penalty while retaining nearly all of the correlation gain. Raw NDCG at the top 10% changes by only +0.0018. Architecture-specific results are heterogeneous, the global interaction is inconclusive, and ridge and GBLUP retain the best mean ranks. A separate nested-tuning sensitivity on CIMMYT wheat supports a Transformer signal while exposing initialization variability. We present a practical evaluation framework and its limitations rather than claiming universal superiority of one loss.

The work addresses G3’s interest in computational tools and statistical methodology for genomic prediction. All benchmark datasets are public. Code, frozen results, cross-validation definitions, selected configurations, analysis scripts, and figure source data are available at https://github.com/GBeurier/gs-loss. Supporting Information, analysis and simulation code, and figure source tables accompany this submission as Supplementary Files 1–3.

The study was supported by the Bana+ project, funded by the French Ministry of Agriculture under the PARSADA programme. During preparation of this manuscript, the authors used Anthropic Claude and OpenAI Codex (GPT-6 for the final submission preparation) to assist with code review, implementation checks, repository organization, benchmark aggregation scripts, LaTeX editing and language revision. All numerical results, references, code and manuscript content were reviewed and verified by the authors, who take full responsibility for the final content of the article.

The manuscript is original, is not under consideration elsewhere, and all authors approve its submission. The authors declare no competing interests.

Thank you for considering our work.

Sincerely,

Grégory Beurier, on behalf of Denis Cornet, Lauriane Rouan, and David Cros
CIRAD, UMR AGAP Institut
Avenue Agropolis, 34398 Montpellier Cedex 5, France
gregory.beurier@cirad.fr
