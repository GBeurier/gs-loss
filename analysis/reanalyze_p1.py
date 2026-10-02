"""Palier-1 re-analysis: merge the Transformer fill into the main grid and
recompute the headline numbers on the now-balanced base (Transformer on all 101
trait-datasets), plus the two robustness checks codex flagged:

  (1) leave-one-species-out (LOSO) sensitivity of the per-architecture Transformer
      loss benefit -- in particular, recompute Transformer Delta r with rice
      removed, since 36/49 of the OLD subset were rice.
  (2) the pooled/per-model Delta-vs-MSE on the full base, to report the new
      (likely smaller, honest) Transformer effect.

Writes results/analysis_p1/*.csv. Does NOT touch the deposited files; reads the
backup main grid + the fill and concatenates in memory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ccgp.stats import compute_deltas, summarize_deltas, mean_ranks, bootstrap_ci

RESULTS = Path("results")
MAIN = RESULTS / "results_main_full.parquet"
FILL = RESULTS / "results_main_transformer_fill.parquet"
OUT = RESULTS / "analysis_p1"
KEY = ["pearson", "spearman", "rmse", "r2", "ndcg@10", "overlap@10", "releff@10"]


def merged() -> pd.DataFrame:
    m = pd.read_parquet(MAIN)
    f = pd.read_parquet(FILL)
    # guard: the fill must not duplicate any existing transformer cell
    mk = set(map(tuple, m[m.model == "transformer"][["dataset", "trait"]].values))
    fk = set(map(tuple, f[["dataset", "trait"]].values))
    assert not (mk & fk), f"fill overlaps existing transformer cells: {mk & fk}"
    df = pd.concat([m, f], ignore_index=True)
    return df


def headline(df, out):
    aff = df[df.calibration == "affine"]
    d = compute_deltas(aff, KEY)
    rows = []
    for met in KEY:
        rows.append(summarize_deltas(d, met, group=("loss",)).assign(scope="pooled"))
        rows.append(summarize_deltas(d, met, group=("loss", "model")).assign(scope="per_model"))
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out / "deltas_summary_p1.csv", index=False)
    return res


def transformer_loso(df, out):
    """Leave-one-species-out Transformer Delta r (pearson loss), to test whether the
    moderation claim survives removing any single species (esp. rice)."""
    aff = df[df.calibration == "affine"]
    d = compute_deltas(aff, ["pearson"])
    tf = d[(d.model == "transformer") & (d.loss == "pearson")]
    rows = []
    full_m, full_lo, full_hi = bootstrap_ci(tf["pearson"].values)
    rows.append({"dropped": "(none)", "n": len(tf), "mean": full_m, "ci_lo": full_lo, "ci_hi": full_hi})
    for sp in sorted(tf.species.unique()):
        sub = tf[tf.species != sp]["pearson"].values
        m, lo, hi = bootstrap_ci(sub)
        rows.append({"dropped": sp, "n": len(sub), "mean": m, "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(rows)
    res.to_csv(out / "transformer_loso_pearson.csv", index=False)
    return res


def ranks(df, out):
    aff = df[df.calibration == "affine"]
    rp = mean_ranks(aff, "pearson")
    rn = mean_ranks(aff, "ndcg@10")
    r = rp.merge(rn, on="method", suffixes=("_pearson", "_ndcg10"))
    r.to_csv(out / "ranks_p1.csv", index=False)
    return r


def coverage(df):
    print("=== coverage (trait-datasets per model) ===")
    for mdl in sorted(df.model.unique()):
        n = df[df.model == mdl][["dataset", "trait"]].drop_duplicates().shape[0]
        print(f"  {mdl:12s} {n}")
    aff = df[df.calibration == "affine"]
    d = compute_deltas(aff, ["pearson"])
    npool = d[d.loss == "pearson"].shape[0]
    print(f"  pooled pearson-loss cells (was 251): {npool}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = merged()
    coverage(df)

    print("\n=== Palier-1 pooled & per-model Delta r (affine) ===")
    h = headline(df, OUT)
    show = h[(h.metric == "pearson")][["scope", "loss", "model", "n", "mean", "ci_lo", "ci_hi", "p_holm"]]
    print(show.round(4).to_string(index=False))

    print("\n=== Transformer leave-one-species-out (pearson loss, Delta r) ===")
    print(transformer_loso(df, OUT).round(4).to_string(index=False))

    print("\n=== mean ranks (affine), full base ===")
    print(ranks(df, OUT).round(2).to_string(index=False))

    # cell-level long table for the interaction LMM (one row per cell)
    from ccgp.stats import write_lmm_long
    nn = df[df.model.isin(["mlp", "cnn", "transformer"])]
    write_lmm_long(nn, KEY, OUT / "lmm_long_p1.csv")
    print(f"\nWrote {OUT}/ (deltas_summary_p1, transformer_loso, ranks_p1, lmm_long_p1)")


if __name__ == "__main__":
    main()
