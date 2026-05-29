"""Statistical analysis of the ccgp results -> tables for the paper.

Reads results/results_{main,batch,splits}_<preset>.parquet and writes summary
CSVs under results/analysis/ plus a console digest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ccgp.stats import (compute_deltas, fit_lmm, mean_ranks, summarize_deltas,
                        write_lmm_long)

RESULTS = Path("results")
OUT = RESULTS / "analysis"
KEY_METRICS = ["pearson", "spearman", "rmse", "r2", "ndcg@10", "overlap@10",
               "releff@10", "seldiff@10"]


def _load(preset):
    main = pd.read_parquet(RESULTS / f"results_main_{preset}.parquet")
    out = {"main": main}
    for k in ("batch", "splits"):
        p = RESULTS / f"results_{k}_{preset}.parquet"
        if p.exists():
            out[k] = pd.read_parquet(p)
    return out


def headline_deltas(main, out):
    """Delta(loss) vs MSE for affine-calibrated NN, pooled and per architecture."""
    d = compute_deltas(main[main.calibration == "affine"], KEY_METRICS)
    rows = []
    for met in KEY_METRICS:
        rows.append(summarize_deltas(d, met, group=("loss",)).assign(scope="pooled"))
        rows.append(summarize_deltas(d, met, group=("loss", "model")).assign(scope="per_model"))
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out / "deltas_summary.csv", index=False)
    return res


def calibration_summary(main, out):
    """Effect of calibration on Pearson-trained NN (Pearson unchanged; RMSE/slope improve)."""
    d = main[(main.model.isin(["mlp", "cnn", "transformer"])) & (main.loss == "pearson")]
    s = d.groupby("calibration")[["pearson", "rmse", "slope", "intercept", "bias"]].agg(["mean", "std"])
    s.to_csv(out / "calibration_summary.csv")
    return s


def ranks(main, out):
    r_pearson = mean_ranks(main[main.calibration == "affine"], "pearson")
    r_ndcg = mean_ranks(main[main.calibration == "affine"], "ndcg@10")
    r = r_pearson.merge(r_ndcg, on="method", suffixes=("_pearson", "_ndcg10"))
    r.to_csv(out / "ranks.csv", index=False)
    return r


def lmm(main, out):
    nn = main[main.model.isin(["mlp", "cnn", "transformer"])]
    csv = write_lmm_long(nn, KEY_METRICS, out / "lmm_long.csv")
    tables = {}
    for met in ["pearson", "rmse", "ndcg@10"]:
        try:
            tables[met] = fit_lmm(csv, met, calibration="affine",
                                  out_csv=str(out / f"lmm_{met.replace('@','')}_affine.csv"))
        except Exception as e:
            print(f"[lmm] {met} failed: {e}")
    return tables


def splits_summary(out):
    """Exp F: does the loss benefit persist under leave-family-out / cross-env splits?"""
    p = RESULTS / "results_splits_full.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    # unified split label: random / family (SoyNAM), within_env / cross_env (wheat)
    if "split_type" in d.columns:
        d["scheme"] = d["split_type"].fillna(d["scheme"])
    rows = []
    for met in ["pearson", "ndcg@10", "overlap@10", "releff@10"]:
        dl = compute_deltas(d[d.calibration == "affine"], [met])
        if dl.empty:
            continue
        rows.append(summarize_deltas(dl, met, group=("loss", "scheme")))
    res = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    res.to_csv(out / "splits_summary.csv", index=False)
    return res


def batch_summary(out):
    """Exp E: predictive ability vs batch size for the Pearson loss (Prop. 3)."""
    p = RESULTS / "results_batch_full.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d = d[d.calibration == "affine"]
    s = (d.groupby(["loss", "batch_size"])[["pearson", "rmse", "ndcg@10"]]
         .agg(["mean", "std"]))
    s.to_csv(out / "batch_summary.csv")
    return s


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else (
        "full" if (RESULTS / "results_main_full.parquet").exists() else "smoke")
    OUT.mkdir(parents=True, exist_ok=True)
    data = _load(preset)
    main_df = data["main"]
    print(f"=== analysis ({preset}): {len(main_df)} rows, "
          f"{main_df[['dataset','trait']].drop_duplicates().shape[0]} trait-datasets ===\n")

    dsum = headline_deltas(main_df, OUT)
    print("--- Delta vs MSE (affine-calibrated NN, pooled) ---")
    pooled = dsum[dsum.scope == "pooled"][["loss", "metric", "mean", "ci_lo", "ci_hi", "p_holm", "n"]]
    print(pooled.round(4).to_string(index=False))

    print("\n--- Calibration effect (Pearson-trained NN) ---")
    print(calibration_summary(main_df, OUT).round(3).to_string())

    print("\n--- Mean ranks (affine) ---")
    print(ranks(main_df, OUT).round(2).to_string(index=False))

    bs = batch_summary(OUT)
    if bs is not None:
        print("\n--- Batch-size effect (Exp E, Pearson loss, affine) ---")
        print(bs.round(3).to_string())
    ss = splits_summary(OUT)
    if ss is not None:
        print("\n--- Loss benefit by split type (Exp F) ---")
        print(ss[["loss", "scheme", "metric", "mean", "ci_lo", "ci_hi", "p_holm", "n"]]
              .round(4).to_string(index=False))

    print("\n--- Linear mixed model (affine) ---")
    for met, tab in lmm(main_df, OUT).items():
        print(f"[{met}]")
        print(tab.round(4).to_string())
    print(f"\nTables written to {OUT}/")


if __name__ == "__main__":
    main()
