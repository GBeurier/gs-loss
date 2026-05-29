"""Statistical analysis of the tidy results table.

Per (dataset, trait, model, calibration, scheme) the metric is averaged over CV
folds; deltas of each neural loss versus the MSE baseline are then summarized
with bootstrap confidence intervals and paired Wilcoxon tests (Holm- and
FDR-corrected). A linear mixed model is fit in R (lme4) for the confirmatory
analysis.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, wilcoxon
from statsmodels.stats.multitest import multipletests

PRED_METRICS = ["pearson", "spearman", "rmse", "mae", "r2", "bias", "slope", "intercept"]
SEL_METRICS = [f"{m}@{k}" for k in (5, 10, 15, 20)
               for m in ("overlap", "ndcg", "seldiff", "releff", "meansel")]
CELL = ["dataset", "species", "trait", "model", "calibration", "scheme"]
HIGHER_BETTER = {"pearson", "spearman", "r2"} | {m for m in SEL_METRICS
                                                 if m.split("@")[0] in
                                                 ("overlap", "ndcg", "releff", "seldiff", "meansel")}
# Metrics best when *closest to an ideal value* (not when largest or smallest).
IDEAL_VALUE = {"slope": 1.0, "bias": 0.0, "intercept": 0.0}


def load_results(*paths) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def fold_means(df: pd.DataFrame, metrics) -> pd.DataFrame:
    return df.groupby(CELL + ["loss"], as_index=False)[metrics].mean()


def compute_deltas(df: pd.DataFrame, metrics, baseline="mse",
                   nn_models=("mlp", "cnn", "transformer")) -> pd.DataFrame:
    """Delta(loss) = metric(loss) - metric(MSE), per cell, averaged over folds."""
    d = df[df.model.isin(nn_models)]
    fm = d.groupby(CELL + ["loss"], as_index=False)[metrics].mean()
    base = fm[fm.loss == baseline].set_index(CELL)[metrics]
    out = []
    for loss in fm.loss.unique():
        if loss in (baseline, "na"):
            continue
        sub = fm[fm.loss == loss].set_index(CELL)
        idx = sub.index.intersection(base.index)
        delta = (sub.loc[idx, metrics] - base.loc[idx, metrics]).reset_index()
        delta["loss"] = loss
        out.append(delta)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def bootstrap_ci(x, n_boot=5000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float(np.nanmean(x)) if len(x) else np.nan, np.nan, np.nan
    method = "BCa" if len(x) >= 12 and np.ptp(x) > 0 else "percentile"
    try:
        res = bootstrap((x,), np.mean, n_resamples=n_boot, random_state=seed, method=method)
        return float(x.mean()), float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        return float(x.mean()), np.nan, np.nan


def summarize_deltas(deltas: pd.DataFrame, metric: str,
                     group=("loss", "model", "calibration")) -> pd.DataFrame:
    rows = []
    for keys, sub in deltas.groupby(list(group)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        vals = sub[metric].dropna().values
        m, lo, hi = bootstrap_ci(vals)
        p = np.nan
        if len(vals) >= 6 and np.ptp(vals) > 0:
            try:
                p = float(wilcoxon(vals).pvalue)
            except Exception:
                p = np.nan
        rows.append({**dict(zip(group, keys)), "metric": metric, "n": len(vals),
                     "mean": m, "ci_lo": lo, "ci_hi": hi, "p": p})
    res = pd.DataFrame(rows)
    mask = res.p.notna()
    res["p_holm"], res["p_fdr"] = np.nan, np.nan
    if mask.sum() > 0:
        res.loc[mask, "p_holm"] = multipletests(res.loc[mask, "p"], method="holm")[1]
        res.loc[mask, "p_fdr"] = multipletests(res.loc[mask, "p"], method="fdr_bh")[1]
    return res.sort_values(list(group)).reset_index(drop=True)


def mean_ranks(df: pd.DataFrame, metric: str,
               within=("dataset", "trait", "scheme", "calibration")) -> pd.DataFrame:
    d = df.copy()
    d["method"] = d.model + "/" + d.loss
    fm = d.groupby(list(within) + ["method"], as_index=False)[metric].mean()
    if metric in IDEAL_VALUE:                       # rank by proximity to the ideal value
        fm["_score"] = (fm[metric] - IDEAL_VALUE[metric]).abs()
        fm["rank"] = fm.groupby(list(within))["_score"].rank(ascending=True)
    else:
        fm["rank"] = fm.groupby(list(within))[metric].rank(ascending=metric not in HIGHER_BETTER)
    return (fm.groupby("method", as_index=False)["rank"].mean()
            .sort_values("rank").reset_index(drop=True))


def write_lmm_long(df: pd.DataFrame, metrics, path) -> Path:
    # Aggregate to one row per cell (mean over CV folds): the LMM then has one
    # observation per dataset x trait x model x loss x calibration, avoiding
    # fold-level pseudoreplication that would inflate precision. Scheme is
    # dropped (it is confounded with dataset source in the main grid).
    keys = ["dataset", "species", "trait", "model", "loss", "calibration"]
    keys = [c for c in keys if c in df.columns]
    agg = df.groupby(keys, as_index=False)[list(metrics)].mean()
    path = Path(path)
    agg.to_csv(path, index=False)
    return path


def fit_lmm(csv, metric, calibration="affine", out_csv=None,
            r_script="analysis/lmm.R") -> pd.DataFrame:
    out_csv = out_csv or str(Path(csv).with_name(f"lmm_{metric}_{calibration}.csv"))
    subprocess.run(["Rscript", r_script, str(csv), metric, calibration, out_csv], check=True)
    return pd.read_csv(out_csv, index_col=0)
