"""Corroborative loss-by-architecture mixed models for final paired deltas.

Panel is fitted as a fixed blocking factor because only 12 panels are observed;
a random task intercept accounts for the 21 repeated loss-by-model contrasts
within each trait-dataset.  The primary uncertainty remains the panel-balanced
cluster bootstrap in ``final_analysis.py``.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


INFILE = Path("results/analysis_final/paired_deltas.csv")
OUT = Path("results/analysis_final")


def fit(metric: str):
    d = pd.read_csv(INFILE)
    d = d[(d.metric == metric) & np.isfinite(d.delta)].copy()
    d["task"] = d.dataset + "::" + d.trait
    common = dict(data=d, groups=d.task, re_formula="1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        full = smf.mixedlm(
            "delta ~ C(loss) * C(model) + C(panel)", **common
        ).fit(reml=False, method="powell", maxiter=3000, disp=False)
        reduced = smf.mixedlm(
            "delta ~ C(loss) + C(model) + C(panel)", **common
        ).fit(reml=False, method="powell", maxiter=3000, disp=False)
    lr = max(0.0, 2.0 * (full.llf - reduced.llf))
    df = int(full.df_modelwc - reduced.df_modelwc)
    test = pd.DataFrame([{
        "metric": metric, "n_rows": len(d), "n_tasks": d.task.nunique(),
        "n_panels": d.panel.nunique(), "full_converged": full.converged,
        "reduced_converged": reduced.converged, "lr": lr, "df": df,
        "p_chi2": float(chi2.sf(lr, df)),
        "task_random_intercept_variance": float(full.cov_re.iloc[0, 0]),
    }])
    ci = full.conf_int().loc[full.params.index]
    coef = pd.DataFrame({
        "term": full.params.index, "estimate": full.params.values,
        "ci_lo": ci.iloc[:, 0].values, "ci_hi": ci.iloc[:, 1].values,
        "metric": metric,
    })
    return test, coef


def main() -> None:
    tests, coefficients = [], []
    for metric in ["pearson", "ndcg@10", "nrmse"]:
        test, coef = fit(metric)
        tests.append(test)
        coefficients.append(coef)
    tests = pd.concat(tests, ignore_index=True)
    coefficients = pd.concat(coefficients, ignore_index=True)
    tests.to_csv(OUT / "mixed_model_interaction_tests.csv", index=False)
    coefficients.to_csv(OUT / "mixed_model_coefficients.csv", index=False)
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
