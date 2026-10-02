"""Final analysis of the frozen 101-task campaign and nested-HPO sensitivity.

The inferential unit is a trait-dataset, never an individual CV fold.  The
primary pooled contrast first averages the seven architecture-specific paired
contrasts within each task.  Confidence intervals resample the 12 public panels
as clusters, preserving dependence among traits from the same panel.  An exact
panel-level sign-flip test is reported as a small-cluster robustness check.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ccgp.config import CAMPAIGN
from experiments.run import build_splits, make_dataset, traits_for, unique_keys


RESULTS = Path("results")
OUT = RESULTS / "analysis_final"
MAIN = RESULTS / "results_main_campaign.parquet"
NESTED = RESULTS / "results_loss_specific_wheat.parquet"
MODELS = ["mlp", "cnn", "transformer", "deepgs", "dnngp", "pnngs", "soydngp"]
LOSSES = ["mse", "pearson", "hybrid", "ccc"]
ENDPOINTS = {
    "pearson": "raw",
    "spearman": "raw",
    "ndcg@10": "raw",
    "overlap@10": "raw",
    "releff@10": "raw",
    "rmse": "affine",
    "nrmse": "affine",
}


def panel_id(dataset: pd.Series) -> pd.Series:
    """Collapse the four SoyNAM trait files to their common biological panel."""
    return dataset.mask(dataset.str.startswith("soynam:"), "soynam")


def outcome_sd_table() -> pd.DataFrame:
    rows: list[dict] = []
    for key in unique_keys(CAMPAIGN):
        ds = make_dataset(key, CAMPAIGN)
        for trait in traits_for(ds, CAMPAIGN):
            _, y, _, _ = ds.get_xy(trait)
            for split in build_splits(ds, trait, CAMPAIGN):
                rows.append({
                    "dataset": ds.name,
                    "trait": trait,
                    "repeat": split.repeat,
                    "fold": split.fold,
                    "y_sd": float(np.std(y[split.test], ddof=0)),
                })
    out = pd.DataFrame(rows)
    if len(out) != 1010 or out.duplicated(["dataset", "trait", "repeat", "fold"]).any():
        raise RuntimeError("unexpected fold-outcome metadata")
    return out


def validate_main(d: pd.DataFrame) -> None:
    key = ["dataset", "trait", "model", "loss", "scheme", "repeat", "fold",
           "calibration", "seed"]
    expected = {m: set(LOSSES) for m in MODELS}
    if len(d) != 90_900 or d.duplicated(key).any():
        raise RuntimeError("campaign must contain 90,900 unique experimental rows")
    if d[["dataset", "trait"]].drop_duplicates().shape[0] != 101:
        raise RuntimeError("campaign must contain 101 trait-datasets")
    for model, losses in expected.items():
        got = set(d.loc[d.model == model, "loss"])
        if got != losses:
            raise RuntimeError(f"{model}: losses {got}, expected {losses}")
    counts = d[d.model.isin(MODELS)].groupby(["dataset", "trait", "model", "loss"]).size()
    if not (counts == 30).all():
        raise RuntimeError("each neural task/model/loss must have 10 folds x 3 calibrations")


def clustered_ci(values: pd.DataFrame, weighting: str,
                 n_boot: int = 20_000, seed: int = 20260902):
    """Cluster bootstrap, either task-weighted or equally weighted by panel."""
    x = values[["panel", "delta"]].dropna()
    panels = np.asarray(sorted(x.panel.unique()))
    rng = np.random.default_rng(seed)
    groups = {p: x.loc[x.panel == p, "delta"].to_numpy() for p in panels}
    indices = rng.integers(0, len(panels), size=(n_boot, len(panels)))
    sums = np.asarray([groups[p].sum() for p in panels])
    counts = np.asarray([len(groups[p]) for p in panels])
    means = sums / counts
    if weighting == "task_weighted":
        draws = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    elif weighting == "panel_balanced":
        draws = means[indices].mean(axis=1)
    else:
        raise ValueError(weighting)
    observed = (float(x.delta.mean()) if weighting == "task_weighted" else
                float(x.groupby("panel").delta.mean().mean()))
    return observed, *np.quantile(draws, [0.025, 0.975]).tolist()


def panel_signflip_p(values: pd.DataFrame, weighting: str) -> float:
    """Exact two-sided sign-flip test, flipping all tasks within a panel together."""
    x = values[["panel", "delta"]].dropna()
    stats = x.groupby("panel").delta.agg(["sum", "count"])
    observed = (abs(stats["sum"].sum() / stats["count"].sum())
                if weighting == "task_weighted" else
                abs((stats["sum"] / stats["count"]).mean()))
    permuted = []
    sums = stats["sum"].to_numpy()
    n = int(stats["count"].sum())
    for signs in itertools.product((-1.0, 1.0), repeat=len(stats)):
        if weighting == "task_weighted":
            permuted.append(abs(np.dot(signs, sums) / n))
        else:
            permuted.append(abs(np.mean(np.asarray(signs) * sums / stats["count"].to_numpy())))
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def paired_tables(d: pd.DataFrame):
    keys = ["panel", "dataset", "trait", "model", "loss"]
    chunks = []
    for metric, calibration in ENDPOINTS.items():
        cell = (d[(d.model.isin(MODELS)) & (d.calibration == calibration)]
                .groupby(keys, as_index=False)[metric].mean())
        base = (cell[cell.loss == "mse"].drop(columns="loss")
                .rename(columns={metric: "baseline"}))
        cmp = cell[cell.loss != "mse"].rename(columns={metric: "value"})
        delta = cmp.merge(base, on=["panel", "dataset", "trait", "model"],
                          validate="many_to_one")
        delta["metric"] = metric
        delta["calibration"] = calibration
        delta["delta"] = delta.value - delta.baseline
        chunks.append(delta)
    paired = pd.concat(chunks, ignore_index=True)

    rows = []
    for metric in ENDPOINTS:
        for loss in LOSSES[1:]:
            selected = paired[(paired.metric == metric) & (paired.loss == loss)]
            for model in ["pooled"] + MODELS:
                if model == "pooled":
                    # Equal architecture weight inside each task; n=101 tasks.
                    x = (selected.groupby(["panel", "dataset", "trait"], as_index=False)
                         .delta.mean())
                else:
                    x = selected[selected.model == model][["panel", "dataset", "trait", "delta"]]
                for weighting in ["panel_balanced", "task_weighted"]:
                    mean, lo, hi = clustered_ci(x, weighting)
                    panel_means = x.groupby("panel").delta.mean()
                    loo = [panel_means.drop(p).mean() for p in panel_means.index]
                    rows.append({
                        "metric": metric,
                        "calibration": ENDPOINTS[metric],
                        "loss": loss,
                        "model": model,
                        "weighting": weighting,
                        "n_tasks": int(x.delta.notna().sum()),
                        "n_panels": int(x.loc[x.delta.notna(), "panel"].nunique()),
                        "mean_delta": mean,
                        "median_delta": float(x.delta.median()),
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "panel_signflip_p": panel_signflip_p(x, weighting),
                        "win_fraction": float((x.delta > 0).mean()),
                        "loo_panel_min": float(min(loo)),
                        "loo_panel_max": float(max(loo)),
                    })
    return paired, pd.DataFrame(rows)


def method_ranks(d: pd.DataFrame) -> pd.DataFrame:
    raw = d[d.calibration == "raw"].copy()
    raw["method"] = np.where(raw.loss == "na", raw.model, raw.model + "/" + raw.loss)
    rows = []
    for metric in ["pearson", "ndcg@10"]:
        task = (raw.groupby(["dataset", "trait", "method"], as_index=False)[metric].mean())
        task["rank"] = task.groupby(["dataset", "trait"])[metric].rank(ascending=False)
        rank = task.groupby("method", as_index=False).agg(
            mean_rank=("rank", "mean"), tasks=("rank", "count"))
        rank["metric"] = metric
        rows.append(rank)
    return pd.concat(rows, ignore_index=True).sort_values(["metric", "mean_rank"])


def calibration_audit(d: pd.DataFrame) -> pd.DataFrame:
    nn = d[(d.model.isin(MODELS)) & (d.loss == "pearson")]
    keys = ["dataset", "trait", "model", "repeat", "fold", "seed"]
    p = nn.pivot_table(index=keys, columns="calibration",
                       values=["pearson", "rmse", "bias", "slope", "intercept"])
    raw_r, affine_r = p["pearson"]["raw"], p["pearson"]["affine"]
    comparable = raw_r.notna() & affine_r.notna()
    flips = comparable & (np.sign(raw_r) != np.sign(affine_r))
    rows = []
    for cal in ["raw", "affine", "isotonic"]:
        row = {"calibration": cal, "n_fold_cells": len(p)}
        for metric in ["pearson", "rmse", "bias", "slope", "intercept"]:
            row[f"mean_{metric}"] = float(p[metric][cal].mean())
            row[f"median_{metric}"] = float(p[metric][cal].median())
        row["raw_affine_sign_flips"] = int(flips.sum())
        row["raw_affine_sign_flip_fraction"] = float(flips.sum() / comparable.sum())
        row["raw_affine_comparable_cells"] = int(comparable.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def nested_tables(sd: pd.DataFrame):
    d = pd.read_parquet(NESTED).merge(
        sd[sd.dataset == "wheat"], on=["dataset", "trait", "repeat", "fold"],
        validate="many_to_one")
    d["nrmse"] = d.rmse / d.y_sd
    rows, seed_rows = [], []
    for metric, calibration in {"pearson": "raw", "ndcg@10": "raw", "nrmse": "affine"}.items():
        x = d[d.calibration == calibration]
        p = x.pivot_table(index=["trait", "model", "fold", "seed"],
                          columns="loss", values=metric).dropna()
        p["delta"] = p.pearson - p.mse
        p = p.reset_index()
        for model in ["pooled"] + ["mlp", "cnn", "transformer"]:
            q = p if model == "pooled" else p[p.model == model]
            task_model = q.groupby(["trait"] + ([] if model != "pooled" else ["model"])).delta.mean()
            rows.append({
                "metric": metric, "calibration": calibration, "model": model,
                "mean_delta": float(q.delta.mean()),
                "median_task_model_delta": float(task_model.median()),
                "min_task_model_delta": float(task_model.min()),
                "max_task_model_delta": float(task_model.max()),
                "n_fold_seed_pairs": len(q),
            })
        for (model, seed), q in p.groupby(["model", "seed"]):
            seed_rows.append({"metric": metric, "model": model, "seed": seed,
                              "mean_delta": float(q.delta.mean())})
    seed_variability = (d[d.calibration == "raw"]
        .groupby(["trait", "model", "loss", "fold"]).pearson.std()
        .rename("sd_across_seeds").reset_index())
    return pd.DataFrame(rows), pd.DataFrame(seed_rows), seed_variability


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(MAIN)
    validate_main(d)
    d["panel"] = panel_id(d.dataset)
    sd = outcome_sd_table()
    d = d.merge(sd, on=["dataset", "trait", "repeat", "fold"], validate="many_to_one")
    d["nrmse"] = d.rmse / d.y_sd.replace(0, np.nan)

    paired, summary = paired_tables(d)
    ranks = method_ranks(d)
    calibration = calibration_audit(d)
    nested, nested_seeds, seed_variability = nested_tables(sd)

    sd.to_csv(OUT / "fold_outcome_sd.csv", index=False)
    paired.to_csv(OUT / "paired_deltas.csv", index=False)
    summary.to_csv(OUT / "paired_summary.csv", index=False)
    ranks.to_csv(OUT / "method_ranks.csv", index=False)
    calibration.to_csv(OUT / "calibration_audit.csv", index=False)
    nested.to_csv(OUT / "nested_hpo_summary.csv", index=False)
    nested_seeds.to_csv(OUT / "nested_hpo_seed_deltas.csv", index=False)
    seed_variability.to_csv(OUT / "nested_hpo_seed_variability.csv", index=False)

    audit = {
        "main_file": str(MAIN), "main_rows": len(d), "tasks": 101,
        "panels": int(d.panel.nunique()), "neural_models": MODELS,
        "losses": LOSSES, "fold_rows_per_task_model_loss": 10,
        "primary_endpoint": "raw Pearson",
        "scale_endpoint": "affine RMSE / test-fold phenotype SD",
        "nested_file": str(NESTED), "nested_rows": 1080,
        "notes": [
            "Main campaign uses one fixed training seed; folds are averaged.",
            "Nested sensitivity uses seeds 0, 1, and 2.",
            "No SelGenPalm data or results are included.",
        ],
    }
    (OUT / "provenance.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(summary[(summary.model == "pooled") & (summary.weighting == "panel_balanced") &
                  (summary.metric.isin(["pearson", "ndcg@10", "nrmse"]))]
          .to_string(index=False))
    print(f"\nWrote final analysis to {OUT}/")


if __name__ == "__main__":
    main()
