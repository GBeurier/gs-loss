"""Real use-case trial (Task 12): apply the ccgp framework to the sorghum
crop-growth-model-parameter genomic-prediction dataset at
/home/delete/modelxgenetic (predict 8 Ecomeristem-type model parameters from
~31k SNP markers). Non-public data -> reported as a 'real breeding case study'.

Runs the same loss comparison (MSE vs Pearson/std-MSE/hybrid/CCC), affine
calibration, and predictive + selection metrics used in the main study.
"""
import numpy as np
import pandas as pd

from ccgp.data import GenomicDataset, preprocess_markers
from ccgp.experiment import run_cell
from ccgp.splits import repeated_kfold
from ccgp.stats import compute_deltas, summarize_deltas

BASE = "/home/delete/modelxgenetic"


def load_sorghum():
    M = pd.read_csv(f"{BASE}/data/Markers.csv", index_col=0)
    M = M.drop(index=[i for i in M.index if str(i).lower() == "trait"], errors="ignore")
    ids = np.asarray(M.columns, dtype=object)
    X = M.T.to_numpy(np.float32) + 1.0          # -1/0/1 -> 0/1/2
    X, info = preprocess_markers(X, maf=0.01, max_features=20000)
    P = pd.read_csv(f"{BASE}/data/Parameters.csv", index_col=0).reindex(ids)
    traits = {c: P[c].to_numpy(float) for c in P.columns}
    meta = {"source": "private: sorghum crop-model parameters (GPCGM)", "n": X.shape[0], **info}
    return GenomicDataset(name="sorghum", species="sorghum", X=X, ids=ids, traits=traits, meta=meta)


def main():
    ds = load_sorghum()
    print(f"sorghum: X={ds.X.shape} traits={ds.trait_names}")
    nn = dict(max_epochs=150, patience=20, batch_size=64,
              arch_kwargs=dict(hidden_dims=(128, 32)))
    rows = []
    for trait in ds.trait_names:
        X, y, ids, _ = ds.get_xy(trait)
        splits = repeated_kfold(len(y), n_splits=5, n_repeats=3, seed=0)
        specs = [("gblup", "na", {}), ("ridge", "na", {"alpha": 100.0})]
        for loss in ["mse", "pearson", "hybrid", "ccc"]:
            p = dict(nn)
            if loss == "hybrid":
                p["loss_kwargs"] = {"lam": 0.1}
            specs.append(("mlp", loss, p))
        for m, loss, params in specs:
            rows += run_cell(ds, trait, m, loss, splits, params=params,
                             base_meta={"exp": "sorghum"}, seed=0)
        print(f"  {trait} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_parquet("results/results_sorghum.parquet")

    print("\n=== Sorghum case study: mean test metrics by model/loss (affine) ===")
    a = res[res.calibration == "affine"]
    print(a.groupby(["model", "loss"])[["pearson", "ndcg@10", "rmse"]].mean().round(4).to_string())
    print("\n=== Delta vs MSE (affine MLP) ===")
    dl = compute_deltas(a, ["pearson", "ndcg@10", "overlap@10", "rmse"])
    if not dl.empty:
        print(summarize_deltas(dl, "pearson", group=("loss",))[["loss", "mean", "ci_lo", "ci_hi", "p_holm", "n"]].round(4).to_string(index=False))
        print(summarize_deltas(dl, "ndcg@10", group=("loss",))[["loss", "mean", "ci_lo", "ci_hi", "p_holm", "n"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
