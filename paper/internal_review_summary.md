# Internal Review Summary

Date: 3 September 2026

These reports are internal simulated peer reviews used to improve the manuscript. They do not constitute editorial review or acceptance by G3 or another journal.

## Scientific and Statistical Review — Gauss

**Final recommendation:** scientifically defensible after revision.

The reviewer requested correction of the calibration sign-reversal count, a conditional statement about preservation of Pearson correlation under affine calibration, an exact description of the SoyNAM hyperparameter registry, and explicit reporting of the borderline panel sign-flip result. These points were resolved: the manuscript reports 12 reversals among 7,014 comparable cells; distinguishes positive- from negative-slope calibration; documents the four phenotype-specific SoyNAM keys; and reports the exact 12-panel test (`p = 0.052`) in the abstract. Architecture contrasts are presented as descriptive because the global loss-by-architecture interaction is inconclusive (`p = 0.414`).

## Manuscript and Argumentation Review — Locke

**Final recommendation:** acceptable after editorial revision.

The reviewer asked for a tighter focus on public-data comparisons of neural networks trained with and without the Pearson loss. The revision distinguishes literature-inspired implementations from exact reproductions, removes validation designs that were not run, replaces claims of a pinned environment with a recorded runtime manifest, and tempers the conclusion to a small, heterogeneous, non-automatic benefit. SelGenPalm and private sorghum analyses are excluded from the active manuscript.

## Reproducibility and Figure Review — James

**Final recommendation:** reproducibility package acceptable after regeneration and audit.

The reviewer required all figures and manuscript variants to be regenerated from the frozen results, with editable exports, source data, text alternatives, checksums, and an executable environment record. The final package includes PDF/SVG figures, CSV source tables, text alternatives, four compiled manuscript variants, and a SHA-256 reproducibility manifest. The documented `pytest -q` and `ccgp verify` commands were repaired and pass.

## Final Resolution

All blocking comments from the three internal reviewers were addressed. The final scientific claim is deliberately limited: Pearson-loss training has a small positive panel-balanced estimate, but evidence for a universal improvement or a general architecture interaction is insufficient.
