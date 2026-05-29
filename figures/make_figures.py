"""Publication figures for the paper.

Fig 1  Geometry: the standardized-MSE / Pearson identity, demonstrated numerically.
Fig 2  Controlled simulation: affine calibration restores scale, leaves ranking intact.
Fig 3  Dataset map: size, dimensionality, predictability across species.
Fig 4  Loss comparison: deltas vs MSE for Pearson / hybrid / CCC (affine-calibrated NN).
Fig 5  Calibration: slope, RMSE and Pearson before vs after affine calibration.
Fig 6  Meta-analysis: when does correlation-consistent training help?

Usage: python figures/make_figures.py [results_dir] [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "font.family": "sans-serif",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
# Okabe-Ito colorblind-safe palette
CB = {"mse": "#000000", "pearson": "#0072B2", "hybrid": "#009E73",
      "ccc": "#D55E00", "raw": "#999999", "affine": "#CC79A7", "isotonic": "#E69F00"}
SPECIES_C = plt.cm.tab20(np.linspace(0, 1, 20))


def _save(fig, out, name):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


# --- Fig 1 -------------------------------------------------------------------

def fig1_geometry(out):
    import torch

    from ccgp import losses
    rng = np.random.default_rng(0)
    rs, raw_mse, std_mse = [], [], []
    for _ in range(500):
        n = int(rng.integers(40, 400))
        y = rng.normal(size=n)
        p = (rng.uniform(-2, 2) * y + rng.normal(size=n) * rng.uniform(0.1, 3)
             + rng.normal() * rng.uniform(0, 3))
        yt, pt = torch.tensor(y), torch.tensor(p)
        rs.append(losses.pearson_corr(pt, yt).item())
        raw_mse.append(float(((pt - yt) ** 2).mean()))
        std_mse.append(losses.std_mse_loss(pt, yt).item() * 2)   # MSE(z_y,z_p) = 2(1-r)
    rs = np.array(rs)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax[0].scatter(rs, raw_mse, s=9, c=CB["raw"], alpha=0.5)
    ax[0].set_xlabel("correlation $r$"); ax[0].set_ylabel(r"MSE$(y,\hat y)$ (raw)")
    ax[0].set_title("raw MSE conflates $r$ with scale")
    ax[1].scatter(rs, std_mse, s=9, c=CB["pearson"], alpha=0.5, label="data")
    grid = np.linspace(-1, 1, 100)
    ax[1].plot(grid, 2 * (1 - grid), "k--", lw=1, label="$2(1-r)$")
    ax[1].set_xlabel("correlation $r$"); ax[1].set_ylabel(r"MSE$(z_y,z_{\hat y})$")
    ax[1].set_title("standardized MSE $=2(1-r)$"); ax[1].legend()
    _save(fig, out, "fig1_geometry")


# --- Fig 2 -------------------------------------------------------------------

def fig2_simulation(out):
    from ccgp.calibration import AffineCalibrator
    from ccgp.metrics import pearson, rmse, top_k_overlap
    rng = np.random.default_rng(3)
    n = 300
    y = rng.normal(10, 3, n)
    p = 0.4 * (y - 10) + 5.0 + rng.normal(0, 0.8, n)   # well-correlated, badly scaled & offset
    cal = AffineCalibrator().fit(p, y)
    pc = cal.transform(p)
    fig, ax = plt.subplots(1, 3, figsize=(8.2, 2.9))
    for a, (pp, ttl) in zip(ax[:2], [(p, "raw prediction"), (pc, "affine-calibrated")]):
        a.scatter(pp, y, s=9, c=CB["pearson"], alpha=0.6)
        lim = [min(pp.min(), y.min()), max(pp.max(), y.max())]
        a.plot(lim, lim, "k--", lw=1)
        a.set_xlabel("prediction"); a.set_ylabel("observed")
        a.set_title(f"{ttl}\n r={pearson(y, pp):.2f}, RMSE={rmse(y, pp):.2f}")
    ov = top_k_overlap(y, p, 0.2)
    ax[2].bar(["raw", "affine"], [top_k_overlap(y, p, 0.2), top_k_overlap(y, pc, 0.2)],
              color=[CB["raw"], CB["affine"]])
    ax[2].set_ylim(0, 1.05); ax[2].set_ylabel("top-20% overlap")
    ax[2].set_title(f"ranking unchanged\n(overlap={ov:.2f})")
    _save(fig, out, "fig2_simulation")


# --- helpers for data-driven figures ----------------------------------------

def _cell_covariates(main: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, trait): n, p, species, and GBLUP predictive ability."""
    g = main[main.model == "gblup"]
    pred = (g[g.calibration == "raw"].groupby(["dataset", "species", "trait"])
            .agg(gblup_r=("pearson", "mean"), n=("n_train", "mean"), p=("p", "mean"))
            .reset_index())
    return pred


# --- Fig 3 -------------------------------------------------------------------

def fig3_datasets(main, out):
    cov = _cell_covariates(main)
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.0))
    species = sorted(cov.species.unique())
    cmap = {s: SPECIES_C[i] for i, s in enumerate(species)}
    for s in species:
        d = cov[cov.species == s]
        ax[0].scatter(d.n, d.p, s=28, color=cmap[s], label=s, alpha=0.8, edgecolor="w", lw=0.4)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("n (training)"); ax[0].set_ylabel("p (markers used)")
    ax[0].set_title(f"{cov.shape[0]} trait-datasets")
    ax[0].legend(fontsize=6, ncol=2, loc="lower right")
    cnt = cov.groupby("species").size().sort_values()
    ax[1].barh(cnt.index, cnt.values, color=[cmap[s] for s in cnt.index])
    ax[1].set_xlabel("# trait-datasets"); ax[1].set_title("traits per species")
    ax[2].hist(cov.gblup_r.dropna(), bins=20, color=CB["pearson"], alpha=0.8)
    ax[2].set_xlabel("GBLUP predictive ability (r)")
    ax[2].set_title("difficulty spread")
    _save(fig, out, "fig3_datasets")


# --- Fig 4 -------------------------------------------------------------------

def fig4_loss_comparison(main, out):
    from ccgp.stats import compute_deltas
    metrics = ["pearson", "rmse", "ndcg@10", "overlap@10"]
    d = compute_deltas(main[main.calibration == "affine"], metrics)
    if d.empty:
        return
    losses_order = [l for l in ["pearson", "hybrid", "ccc"] if l in d.loss.unique()]
    fig, ax = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics), 3.2))
    for a, met in zip(ax, metrics):
        data = [d[d.loss == l][met].dropna().values for l in losses_order]
        parts = a.violinplot(data, showmeans=True, showextrema=False)
        for pc, l in zip(parts["bodies"], losses_order):
            pc.set_facecolor(CB[l]); pc.set_alpha(0.6)
        a.axhline(0, color="k", lw=0.8, ls="--")
        a.set_xticks(range(1, len(losses_order) + 1)); a.set_xticklabels(losses_order, rotation=30)
        a.set_title(f"$\\Delta$ {met}")
        a.set_ylabel("loss - MSE")
    fig.suptitle("Headline comparison vs MSE (affine-calibrated MLP/CNN)", y=1.02)
    _save(fig, out, "fig4_loss_comparison")


# --- Fig 5 -------------------------------------------------------------------

def fig5_calibration(main, out):
    d = main[(main.model.isin(["mlp", "cnn"])) & (main.loss == "pearson")]
    piv = d.pivot_table(index=["dataset", "trait", "model", "repeat", "fold"],
                        columns="calibration", values=["slope", "rmse", "pearson"])
    fig, ax = plt.subplots(1, 3, figsize=(9.2, 3.0))
    for a, met, ttl in zip(ax, ["slope", "rmse", "pearson"],
                           ["calibration slope", "RMSE", "Pearson r"]):
        if ("raw" not in piv[met]) or ("affine" not in piv[met]):
            continue
        x, y = piv[met]["raw"], piv[met]["affine"]
        a.scatter(x, y, s=6, alpha=0.3, c=CB["pearson"])
        lo, hi = np.nanpercentile(np.concatenate([x.values, y.values]), [1, 99])
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        if met == "slope":
            a.axhline(1, color=CB["affine"], lw=1)
        a.set_xlabel(f"{ttl} (raw)"); a.set_ylabel(f"{ttl} (affine)"); a.set_title(ttl)
    fig.suptitle("Affine calibration: scale restored, ranking (Pearson) unchanged", y=1.02)
    _save(fig, out, "fig5_calibration")


# --- Fig 6 -------------------------------------------------------------------

def fig6_meta(main, out, batch=None):
    from ccgp.stats import compute_deltas
    d = compute_deltas(main[main.calibration == "affine"], ["ndcg@10", "pearson"])
    cov = _cell_covariates(main)
    if d.empty:
        return
    dd = d[d.loss == "pearson"].merge(cov, on=["dataset", "species", "trait"], how="left")
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.0))
    ax[0].scatter(dd.n, dd["ndcg@10"], s=14, c=CB["pearson"], alpha=0.7)
    ax[0].axhline(0, color="k", ls="--", lw=0.8); ax[0].set_xscale("log")
    ax[0].set_xlabel("n (training)"); ax[0].set_ylabel(r"$\Delta$NDCG@10 (Pearson-MSE)")
    ax[1].scatter(dd.gblup_r, dd["ndcg@10"], s=14, c=CB["hybrid"], alpha=0.7)
    ax[1].axhline(0, color="k", ls="--", lw=0.8)
    ax[1].set_xlabel("GBLUP predictive ability"); ax[1].set_ylabel(r"$\Delta$NDCG@10")
    if batch is not None and len(batch):
        b = batch[(batch.loss == "pearson") & (batch.calibration == "affine")]
        bb = b.groupby("batch_size")["pearson"].mean()
        ax[2].plot(bb.index.astype(str), bb.values, "o-", c=CB["pearson"])
        ax[2].set_xlabel("batch size (-1 = full)"); ax[2].set_ylabel("test Pearson")
        ax[2].set_title("batch size (Exp E)")
    else:
        ax[2].axis("off")
    fig.suptitle("When does correlation-consistent training help?", y=1.02)
    _save(fig, out, "fig6_meta")


def main():
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures")
    suffix = "full" if (res / "results_main_full.parquet").exists() else "smoke"
    main_df = pd.read_parquet(res / f"results_main_{suffix}.parquet")
    batch_df = None
    bp = res / f"results_batch_{suffix}.parquet"
    if bp.exists():
        batch_df = pd.read_parquet(bp)
    fig1_geometry(out)
    fig2_simulation(out)
    fig3_datasets(main_df, out)
    fig4_loss_comparison(main_df, out)
    fig5_calibration(main_df, out)
    fig6_meta(main_df, out, batch_df)


if __name__ == "__main__":
    main()
