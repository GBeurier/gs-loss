"""Publication figures for the paper.

Fig 1  Geometry: the standardized-MSE / Pearson identity, demonstrated numerically.
Fig 2  Controlled simulation: affine calibration restores scale, leaves ranking intact.
Fig 3  Dataset map: size, dimensionality, predictability across species.
Fig 4  Headline comparison: MSE, raw Pearson, and affine-calibrated Pearson.
Fig 5  Calibration: slope, RMSE and Pearson before vs after affine calibration.
Fig 6  Meta-analysis: when does correlation-consistent training help?

Usage: python figures/make_figures.py [results_dir] [out_dir]
"""
from __future__ import annotations

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
MARKERS = {"pooled": "D", "mlp": "o", "cnn": "s", "transformer": "^",
           "deepgs": "v", "dnngp": "P", "pnngs": "X", "soydngp": "h"}


def _save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def _source(out, name, frame):
    path = Path(out) / "source_data"
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / f"{name}.csv", index=False)


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
    ax[0].set_xlabel("correlation $r$")
    ax[0].set_ylabel(r"MSE$(y,\hat y)$ (raw)")
    ax[0].set_title("raw MSE conflates $r$ with scale")
    ax[1].scatter(rs, std_mse, s=9, c=CB["pearson"], alpha=0.5, label="data")
    grid = np.linspace(-1, 1, 100)
    ax[1].plot(grid, 2 * (1 - grid), "k--", lw=1, label="$2(1-r)$")
    ax[1].set_xlabel("correlation $r$")
    ax[1].set_ylabel(r"MSE$(z_y,z_{\hat y})$")
    ax[1].set_title("standardized MSE $=2(1-r)$")
    ax[1].legend()
    _source(out, "fig1_geometry_points", pd.DataFrame({
        "pearson_r": rs, "raw_mse": raw_mse, "standardized_mse": std_mse,
        "identity_2_one_minus_r": 2 * (1 - rs)}))
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
        a.set_xlabel("prediction")
        a.set_ylabel("observed")
        a.set_title(f"{ttl}\n r={pearson(y, pp):.2f}, RMSE={rmse(y, pp):.2f}")
    ov = top_k_overlap(y, p, 0.2)
    ax[2].bar(["raw", "affine"], [top_k_overlap(y, p, 0.2), top_k_overlap(y, pc, 0.2)],
              color=[CB["raw"], CB["affine"]])
    ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("top-20% overlap")
    ax[2].set_title(f"ranking unchanged\n(overlap={ov:.2f})")
    _source(out, "fig2_simulation_points", pd.DataFrame({
        "observed": y, "raw_prediction": p, "affine_prediction": pc}))
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
    _source(out, "fig3_datasets", cov)
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.0))
    species = sorted(cov.species.unique())
    cmap = {s: SPECIES_C[i] for i, s in enumerate(species)}
    for s in species:
        d = cov[cov.species == s]
        ax[0].scatter(d.n, d.p, s=28, color=cmap[s], label=s, alpha=0.8, edgecolor="w", lw=0.4)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("n (training)")
    ax[0].set_ylabel("p (markers used)")
    ax[0].set_title(f"{cov.shape[0]} trait-datasets")
    ax[0].legend(fontsize=6, ncol=2, loc="lower right")
    cnt = cov.groupby("species").size().sort_values()
    ax[1].barh(cnt.index, cnt.values, color=[cmap[s] for s in cnt.index])
    ax[1].set_xlabel("# trait-datasets")
    ax[1].set_title("traits per species")
    ax[2].hist(cov.gblup_r.dropna(), bins=20, color=CB["pearson"], alpha=0.8)
    ax[2].set_xlabel("GBLUP predictive ability (r)")
    ax[2].set_title("difficulty spread")
    _save(fig, out, "fig3_datasets")


# --- Fig 4 -------------------------------------------------------------------

def fig4_loss_comparison(main, out):
    models = ["mlp", "cnn", "transformer", "deepgs", "dnngp", "pnngs", "soydngp"]
    d = main[main.model.isin(models)].copy()
    d["nrmse"] = d.rmse / d.y_sd.replace(0, np.nan)
    d["panel"] = d.dataset.mask(d.dataset.str.startswith("soynam:"), "soynam")
    conditions = [
        ("mse", "raw", "MSE training\nraw output"),
        ("pearson", "raw", "Pearson training\nraw output"),
        ("pearson", "affine", "Pearson training\n+ affine calibration"),
    ]
    selected = []
    for loss, calibration, strategy in conditions:
        q = d[(d.loss == loss) & (d.calibration == calibration)].copy()
        q["strategy"] = strategy
        selected.append(q)
    d = pd.concat(selected, ignore_index=True)

    # Preserve the inferential hierarchy: folds, then architectures, then traits within panels.
    task = (d.groupby(["panel", "dataset", "trait", "model", "strategy"], as_index=False)
            [["pearson", "nrmse", "ndcg@10"]].mean()
            .groupby(["panel", "dataset", "trait", "strategy"], as_index=False)
            [["pearson", "nrmse", "ndcg@10"]].mean())
    panel = (task.groupby(["panel", "strategy"], as_index=False)
             [["pearson", "nrmse", "ndcg@10"]].mean())
    strategy_order = [x[2] for x in conditions]
    rng = np.random.default_rng(20260903)
    metric_strategy_orders = {
        "pearson": strategy_order,
        "nrmse": strategy_order,
        "ndcg@10": strategy_order[:2],
    }
    rows = []
    for metric, strategies in metric_strategy_orders.items():
        pivot = panel.pivot(index="panel", columns="strategy", values=metric)[strategy_order]
        baseline = pivot[strategy_order[0]].to_numpy()
        for strategy in strategies:
            values = pivot[strategy].to_numpy()
            delta = values - baseline
            draws = rng.integers(0, len(delta), size=(20_000, len(delta)))
            lo, hi = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
            rows.append({
                "metric": metric,
                "strategy": strategy,
                "panel_balanced_mean": values.mean(),
                "delta_vs_mse_raw": delta.mean(),
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
                "n_panels": len(delta),
            })
    headline = pd.DataFrame(rows)
    _source(out, "fig4_headline_strategy_summary", headline)
    _source(out, "fig4_headline_panel_values", panel)

    fig = plt.figure(figsize=(12.8, 4.45))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.95, 1.02, 0.95, 1.35], wspace=0.55)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[0, 2])
    ax_arch = fig.add_subplot(gs[0, 3])

    labels = ["MSE\nraw", "Pearson\nraw", "Pearson\n+ affine"]
    colors = ["#555555", CB["pearson"], CB["affine"]]
    markers = ["s", "o", "D"]

    def effect_panel(axis, metric, title, xlabel, better, strategies=None):
        strategies = strategy_order if strategies is None else strategies
        pivot = panel.pivot(index="panel", columns="strategy", values=metric)[strategy_order]
        baseline = pivot[strategy_order[0]]
        q = headline[headline.metric == metric].set_index("strategy").loc[strategies]
        y = np.arange(len(strategies))
        jitter = np.linspace(-0.105, 0.105, len(pivot))
        for i, strategy in enumerate(strategies):
            delta = (pivot[strategy] - baseline).to_numpy()
            axis.scatter(delta, y[i] + jitter, s=13, color=colors[i], alpha=0.36,
                         edgecolor="none", zorder=1)
            row = q.loc[strategy]
            axis.errorbar(row.delta_vs_mse_raw, y[i],
                          xerr=[[row.delta_vs_mse_raw - row.delta_ci_lo],
                                [row.delta_ci_hi - row.delta_vs_mse_raw]],
                          fmt=markers[i], color=colors[i], mfc="white", mec=colors[i],
                          mew=1.3, ms=7, capsize=3, lw=1.5, zorder=3)
        axis.axvline(0, color="#555555", lw=0.9, ls=(0, (3, 2)), zorder=0)
        axis.set_yticks(y, labels[:len(strategies)])
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.text(0.98, 0.04, better, transform=axis.transAxes, ha="right", va="bottom",
                  fontsize=7.5, color="#555555")
        axis.grid(axis="x", color="#E7E7E7", lw=0.7, zorder=0)
        axis.tick_params(axis="y", length=0)

    effect_panel(ax_r, "pearson", "A   Predictive ability", r"change in Pearson $r$ vs MSE raw",
                 "higher is better ->")
    effect_panel(ax_e, "nrmse", "B   Phenotypic-scale error",
                 "change in normalized RMSE vs MSE raw", "<- lower is better")
    effect_panel(ax_s, "ndcg@10", "C   Upper-tail recovery",
                 r"change in NDCG@10 vs MSE raw", "higher is better ->",
                 strategies=strategy_order[:2])
    ax_r.text(0.98, 0.93, "+0.0048\n[0.0008, 0.0091]", transform=ax_r.transAxes,
              ha="right", va="top", color=CB["pearson"], fontsize=8, fontweight="bold")
    ax_e.text(0.98, 0.93, "raw: +0.346", transform=ax_e.transAxes,
              ha="right", va="top", color=CB["pearson"], fontsize=8, fontweight="bold")
    ax_e.text(0.98, 0.86, "affine: −0.0033", transform=ax_e.transAxes,
              ha="right", va="top", color=CB["affine"], fontsize=8, fontweight="bold")
    ax_s.text(0.98, 0.93, "+0.0018\n[-0.0001, 0.0035]", transform=ax_s.transAxes,
              ha="right", va="top", color=CB["pearson"], fontsize=8, fontweight="bold")

    summary = pd.read_csv("results/analysis_final/paired_summary.csv")
    architecture_order = ["pooled", "transformer", "pnngs", "soydngp", "mlp",
                          "dnngp", "cnn", "deepgs"]
    architecture = (summary[(summary.weighting == "panel_balanced") &
                            (summary.loss == "pearson") &
                            (summary.metric == "pearson") &
                            (summary.model.isin(architecture_order))]
                    .set_index("model").loc[architecture_order].reset_index())
    _source(out, "fig4_architecture_effects", architecture)
    y = np.arange(len(architecture))
    for i, row in architecture.iterrows():
        color = ("#222222" if row.model == "pooled" else
                 CB["ccc"] if row.model == "deepgs" else CB["pearson"])
        ax_arch.errorbar(row.mean_delta, y[i],
                         xerr=[[row.mean_delta - row.ci_lo], [row.ci_hi - row.mean_delta]],
                         fmt=MARKERS[row.model], color=color, mfc="white", mec=color,
                         mew=1.2, ms=6.5, capsize=2.5, lw=1.3, zorder=3)
        ax_arch.text(0.98, y[i], f"{row.mean_delta:+.4f}",
                     transform=ax_arch.get_yaxis_transform(), ha="right", va="center",
                     fontsize=7.5, color=color,
                     fontweight="bold" if row.model == "pooled" else None)
    names = ["Pooled", "Transformer", "PNNGS", "SoyDNGP", "MLP", "DNNGP", "CNN", "DeepGS"]
    ax_arch.set_yticks(y, names)
    ax_arch.invert_yaxis()
    ax_arch.axvline(0, color="#555555", lw=0.9, ls=(0, (3, 2)), zorder=0)
    ax_arch.grid(axis="x", color="#E7E7E7", lw=0.7, zorder=0)
    ax_arch.tick_params(axis="y", length=0)
    ax_arch.set_xlabel(r"Pearson loss $-$ MSE ($\Delta r$)")
    ax_arch.set_title("D   All seven architectures", loc="left", fontweight="bold")
    ax_arch.set_xlim(-0.017, 0.043)

    fig.suptitle("Pearson loss separates correlation, calibration, and selection effects",
                 x=0.5, y=1.01, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.008,
             "Small points (A-C): 12 panel contrasts. Open symbols: panel-balanced means. "
             "Bars: 95% panel-cluster bootstrap intervals. Global interaction: $p=0.414$.",
             fontsize=7.5, ha="center")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.19, top=0.84)
    _save(fig, out, "fig4_loss_comparison")


# --- Fig 5 -------------------------------------------------------------------

def fig5_calibration(main, out):
    models = ["mlp", "cnn", "transformer", "deepgs", "dnngp", "pnngs", "soydngp"]
    d = main[(main.model.isin(models)) & (main.loss == "pearson")].copy()
    d["nrmse"] = d.rmse / d.y_sd.replace(0, np.nan)
    cell = (d.groupby(["dataset", "trait", "model", "calibration"], as_index=False)
            .agg(nrmse=("nrmse", "mean"), slope=("slope", "mean"), pearson=("pearson", "mean")))
    piv = cell.pivot_table(index=["dataset", "trait", "model"], columns="calibration",
                           values=["nrmse", "slope", "pearson"])
    fig, ax = plt.subplots(1, 3, figsize=(9.7, 3.2))
    for i, cal in enumerate(["raw", "affine"]):
        vals = piv["nrmse"][cal].dropna()
        ax[0].boxplot(vals, positions=[i], widths=.55, showfliers=False,
                      patch_artist=True, boxprops={"facecolor": CB[cal], "alpha": .55})
    ax[0].set_xticks([0,1],["raw","affine"])
    ax[0].set_ylabel("RMSE / test-fold SD")
    ax[0].set_title("Normalized error")
    slope_dev = pd.DataFrame({c: (piv["slope"][c]-1).abs() for c in ["raw","affine"]})
    ax[1].boxplot([slope_dev.raw.dropna(), slope_dev.affine.dropna()], positions=[0,1], widths=.55,
                  showfliers=False, patch_artist=True,
                  boxprops={"facecolor": CB["affine"], "alpha": .55})
    ax[1].set_xticks([0,1],["raw","affine"])
    ax[1].set_ylabel("|calibration slope − 1|")
    ax[1].set_title("Scale calibration")
    fold = d.pivot_table(index=["dataset","trait","model","repeat","fold","seed"],
                         columns="calibration", values="pearson")
    comparable = fold.raw.notna() & fold.affine.notna()
    change = (fold.affine-fold.raw)[comparable]
    ax[2].hist(change, bins=np.linspace(-1.1,1.1,45), color=CB["pearson"], alpha=.75)
    ax[2].axvline(0,color="black",lw=.8,ls="--")
    ax[2].set_xlabel("affine r − raw r")
    ax[2].set_ylabel("fold-level cells")
    ax[2].set_title("Occasional negative-slope flips")
    flips = comparable & (np.sign(fold.raw) != np.sign(fold.affine))
    ax[2].text(.03,.94,f"{flips.sum()}/{comparable.sum()} sign reversals",
               transform=ax[2].transAxes,va="top",fontsize=8)
    source = cell.copy()
    source["figure_note"] = "task-model mean across ten CV folds"
    _source(out, "fig5_calibration_task_model", source)
    fold_out = fold.reset_index()
    fold_out["delta_r"] = fold_out.affine - fold_out.raw
    fold_out["comparable"] = fold_out.raw.notna() & fold_out.affine.notna()
    _source(out, "fig5_calibration_fold_changes", fold_out)
    fig.suptitle("Validation-fitted affine calibration reduces scale error but is not constrained positive", y=1.02)
    fig.tight_layout()
    _save(fig, out, "fig5_calibration")


# --- Fig 6 -------------------------------------------------------------------

def fig6_meta(main, out, batch=None):
    nested = pd.read_csv("results/analysis_final/nested_hpo_summary.csv")
    seeds = pd.read_csv("results/analysis_final/nested_hpo_seed_deltas.csv")
    shared = pd.read_csv("results/analysis_final/paired_summary.csv")
    fig, ax = plt.subplots(1, 3, figsize=(10.3, 3.35))
    common=["mlp","cnn","transformer"]
    all_models=["mlp","cnn","transformer","deepgs","dnngp","pnngs","soydngp"]
    q=shared[(shared.metric=="pearson")&(shared.loss=="pearson")&
             (shared.weighting=="panel_balanced")&shared.model.isin(all_models)].set_index("model").loc[all_models]
    y=np.arange(len(q))
    xerr=np.vstack([q.mean_delta-q.ci_lo,q.ci_hi-q.mean_delta])
    ax[0].barh(y,q.mean_delta,xerr=xerr,color=CB["pearson"],alpha=.72,
               error_kw={"ecolor":"#333333","lw":1,"capsize":2})
    ax[0].axvline(0,color="black",lw=.8,ls="--")
    ax[0].set_yticks(y,["MLP","CNN","Transformer","DeepGS","DNNGP","PNNGS","SoyDNGP"])
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Δr (Pearson − MSE)")
    ax[0].set_title("Main ablation: 7 networks")
    n=nested[(nested.metric=="pearson")&nested.model.isin(common)].set_index("model").loc[common]
    model_colors=[CB["pearson"],CB["isotonic"],CB["hybrid"]]
    ax[1].bar(common,n.mean_delta,color=model_colors,alpha=.8)
    ax[1].axhline(0,color="black",lw=.8,ls="--")
    ax[1].set_ylabel("Δr (Pearson − MSE)")
    ax[1].set_title("Nested HPO: 3-network subset")
    s=seeds[(seeds.metric=="pearson")&seeds.model.isin(common)]
    for model in common:
        z=s[s.model==model]
        ax[2].plot(z.seed,z.mean_delta,marker=MARKERS[model],label=model)
    ax[2].axhline(0,color="black",lw=.8,ls="--")
    ax[2].set_xticks([0,1,2])
    ax[2].set_xlabel("training seed")
    ax[2].set_ylabel("mean Δr")
    ax[2].set_title("Initialization sensitivity")
    ax[2].legend(fontsize=7)
    _source(out, "fig6_nested_hpo_summary", nested)
    _source(out, "fig6_nested_hpo_seed_deltas", seeds)
    _source(out, "fig6_shared_hpo_all_models", q.reset_index())
    fig.suptitle("Nested-HPO sensitivity: MLP, CNN and Transformer only", y=1.02)
    fig.tight_layout()
    _save(fig, out, "fig6_meta")


def main():
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures")
    main_df = pd.read_parquet(res / "results_main_campaign.parquet")
    sd = pd.read_csv(res / "analysis_final" / "fold_outcome_sd.csv")
    main_df = main_df.merge(sd, on=["dataset", "trait", "repeat", "fold"],
                            validate="many_to_one")
    batch_df = None
    bp = res / "results_batch_full.parquet"
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
