"""Generate the Supporting Information (File S1) for the ccgp manuscript.

Every number in the supplement is computed directly from the deposited result
files (no manual transcription). Run from the repo root:

    python analysis/make_supplement.py [preset]   # preset defaults to "full"

It writes LaTeX table fragments (tabS_*.tex) and figures (figS_*.pdf) into
paper/supp/. The standalone paper/supplement.tex inputs these fragments.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ccgp.stats import compute_deltas, mean_ranks, summarize_deltas
from ccgp import hpo

# --- paths / config ----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "analysis"
OUT = ROOT / "paper" / "supp"

NN_MODELS = ["mlp", "cnn", "transformer"]
KS = (5, 10, 15, 20)
# metrics for the full delta table: predictive accuracy + the four selection
# families at every k.
PRED_DELTA_METRICS = ["pearson", "spearman", "rmse", "mae", "r2"]
SEL_FAMILIES = ["overlap", "ndcg", "seldiff", "releff", "meansel"]
SEL_DELTA_METRICS = [f"{m}@{k}" for m in SEL_FAMILIES for k in KS]
ALL_DELTA_METRICS = PRED_DELTA_METRICS + SEL_DELTA_METRICS

LOSS_ORDER = ["pearson", "hybrid", "ccc"]

# --- figure style (mirrors figures/make_figures.py) --------------------------
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "font.family": "sans-serif",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
CB = {"mse": "#000000", "pearson": "#0072B2", "hybrid": "#009E73",
      "ccc": "#D55E00", "raw": "#999999", "affine": "#CC79A7", "isotonic": "#E69F00"}
SPECIES_C = plt.cm.tab20(np.linspace(0, 1, 20))


def _save_fig(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.pdf")


def _write_tex(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body)
    print(f"wrote {name}")


def _fmt_p(p):
    if pd.isna(p):
        return "--"
    if p < 1e-4:
        return "$<10^{-4}$"
    return f"{p:.4f}"


def _ci(lo, hi, nd=4):
    if pd.isna(lo) or pd.isna(hi):
        return "--"
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


# =============================================================================
# Tables
# =============================================================================

def table_deltas_full(main):
    """tabS_deltas_full: Delta-vs-MSE (affine NN), all metrics, pooled + per-arch."""
    aff = main[main.calibration == "affine"]
    d = compute_deltas(aff, ALL_DELTA_METRICS)
    rows = []
    for met in ALL_DELTA_METRICS:
        pooled = summarize_deltas(d, met, group=("loss",))
        for _, r in pooled.iterrows():
            rows.append({"Metric": met, "Loss": r["loss"], "Architecture": "pooled",
                         "n": int(r["n"]), "Mean": r["mean"],
                         "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p_holm"]})
        per = summarize_deltas(d, met, group=("loss", "model"))
        for _, r in per.iterrows():
            rows.append({"Metric": met, "Loss": r["loss"], "Architecture": r["model"],
                         "n": int(r["n"]), "Mean": r["mean"],
                         "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p_holm"]})
    df = pd.DataFrame(rows)
    # ordering
    met_rank = {m: i for i, m in enumerate(ALL_DELTA_METRICS)}
    loss_rank = {l: i for i, l in enumerate(LOSS_ORDER)}
    arch_rank = {"pooled": 0, "mlp": 1, "cnn": 2, "transformer": 3}
    df["_m"] = df.Metric.map(met_rank)
    df["_l"] = df.Loss.map(loss_rank)
    df["_a"] = df.Architecture.map(arch_rank)
    df = df.sort_values(["_m", "_l", "_a"]).reset_index(drop=True)

    out = pd.DataFrame({
        "Metric": df.Metric,
        "Loss": df.Loss,
        "Architecture": df.Architecture,
        "$n$": df.n,
        "Mean $\\Delta$": df.Mean.map(lambda v: f"{v:.4f}"),
        "95\\% CI": [_ci(lo, hi) for lo, hi in zip(df.ci_lo, df.ci_hi)],
        "$p_{\\mathrm{Holm}}$": df.p.map(_fmt_p),
    })
    body = out.to_latex(index=False, escape=False, longtable=True,
                        column_format="lllrrcc",
                        caption=("Delta-versus-MSE for the affine-calibrated neural network, "
                                 "for every predictive and selection metric, pooled across "
                                 "architectures and broken down per architecture (mlp/cnn/transformer). "
                                 "Mean change relative to the MSE baseline with BCa 95\\% bootstrap "
                                 "confidence interval and Holm-corrected paired-Wilcoxon $p$-value; "
                                 "$n$ is the number of (dataset, trait) cells. Positive values favour "
                                 "the correlation-consistent loss for pearson/spearman/$R^2$ and the "
                                 "selection metrics, negative values for rmse/mae."),
                        label="tab:S-deltas-full")
    _write_tex("tabS_deltas_full.tex", body)


def table_lmm():
    """tabS_lmm: three mixed-model coefficient tables (pearson, ndcg@10, rmse)."""
    files = [("pearson", "Pearson $r$", "lmm_pearson_affine.csv"),
             ("ndcg10", "NDCG@10", "lmm_ndcg10_affine.csv"),
             ("rmse", "RMSE", "lmm_rmse_affine.csv")]
    parts = []
    for _, pretty, fname in files:
        df = pd.read_csv(ANALYSIS / fname)
        out = pd.DataFrame({
            "Term": df["term"],
            "Estimate": df["Estimate"].map(lambda v: f"{v:.4f}"),
            "SE": df["Std. Error"].map(lambda v: f"{v:.4f}"),
            "$t$": df["t value"].map(lambda v: f"{v:.2f}"),
            "95\\% CI": [_ci(lo, hi) for lo, hi in zip(df["ci_lo"], df["ci_hi"])],
        })
        tex = out.to_latex(index=False, escape=False, column_format="lrrrc")
        # strip the tabular wrapper produced by to_latex so we can stack panels
        parts.append((pretty, tex))
    blocks = []
    for pretty, tex in parts:
        blocks.append("\\paragraph{Response: %s.}\\mbox{}\\\\[2pt]\n%s" % (pretty, tex))
    _write_tex("tabS_lmm.tex", "\n\n".join(blocks))


def table_ranks():
    """tabS_ranks: full mean method ranks (ranks.csv)."""
    df = pd.read_csv(ANALYSIS / "ranks.csv")
    out = pd.DataFrame({
        "Method (model/loss)": df["method"],
        "Mean rank (Pearson)": df["rank_pearson"].map(lambda v: f"{v:.2f}"),
        "Mean rank (NDCG@10)": df["rank_ndcg10"].map(lambda v: f"{v:.2f}"),
    })
    body = out.to_latex(index=False, escape=True, column_format="lrr",
                        caption=("Mean rank of every model/loss method across all trait-datasets "
                                 "(affine-calibrated), lower is better, for predictive ability "
                                 "(Pearson $r$) and selection quality (NDCG@10). Ranks are computed "
                                 "within each (dataset, trait, scheme, calibration) block and averaged. "
                                 "GBLUP and ridge use no neural loss (\\texttt{na})."),
                        label="tab:S-ranks", position="ht")
    _write_tex("tabS_ranks.tex", body)


def table_calibration():
    """tabS_calibration: calibration_summary.csv as mean +/- sd per metric."""
    df = pd.read_csv(ANALYSIS / "calibration_summary.csv", header=[0, 1], index_col=0)
    metrics = ["pearson", "rmse", "slope", "intercept", "bias"]
    rows = []
    for cal in ["raw", "affine", "isotonic"]:
        row = {"Calibration": cal}
        for met in metrics:
            m = df.loc[cal, (met, "mean")]
            s = df.loc[cal, (met, "std")]
            row[met] = f"{m:.3f} $\\pm$ {s:.3f}"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.columns = ["Calibration", "Pearson $r$", "RMSE", "Slope", "Intercept", "Bias"]
    body = out.to_latex(index=False, escape=False, column_format="lccccc",
                        caption=("Effect of post-hoc calibration on the Pearson-trained neural "
                                 "network (mlp/cnn/transformer): mean $\\pm$ s.d. of each diagnostic "
                                 "across all folds. Affine calibration restores the regression slope "
                                 "towards 1 and reduces RMSE while leaving the rank correlation "
                                 "unchanged; isotonic calibration is monotone and similar."),
                        label="tab:S-calibration", position="ht")
    _write_tex("tabS_calibration.tex", body)


def table_batch():
    """tabS_batch: batch_summary.csv (Exp E)."""
    df = pd.read_csv(ANALYSIS / "batch_summary.csv", header=[0, 1], index_col=[0, 1])
    rows = []
    for (loss, bs), r in df.iterrows():
        rows.append({
            "Loss": loss,
            "Batch size": "full" if int(bs) == -1 else str(int(bs)),
            "Pearson $r$": f"{r[('pearson', 'mean')]:.4f} $\\pm$ {r[('pearson', 'std')]:.4f}",
            "RMSE": f"{r[('rmse', 'mean')]:.3f} $\\pm$ {r[('rmse', 'std')]:.3f}",
            "NDCG@10": f"{r[('ndcg@10', 'mean')]:.4f} $\\pm$ {r[('ndcg@10', 'std')]:.4f}",
        })
    out = pd.DataFrame(rows)
    body = out.to_latex(index=False, escape=False, column_format="llccc",
                        caption=("Experiment E: minibatch size and the Pearson loss. "
                                 "Mean $\\pm$ s.d. of test Pearson $r$, RMSE and NDCG@10 "
                                 "(affine-calibrated MLP) over four EasyGeSe datasets as a "
                                 "function of minibatch size (full = full-batch gradient). The "
                                 "minibatch estimator of the standardized-MSE loss is "
                                 "consistent: accuracy is stable across batch sizes."),
                        label="tab:S-batch", position="ht")
    _write_tex("tabS_batch.tex", body)


def table_splits(splits):
    """tabS_splits: scheme x loss summary + per-architecture Delta breakdown."""
    # Panel A: loss x scheme x metric (from splits_summary.csv, all metrics).
    s = pd.read_csv(ANALYSIS / "splits_summary.csv")
    scheme_order = {"random": 0, "within_env": 1, "family": 2, "cross_env": 3}
    met_order = {"pearson": 0, "ndcg@10": 1, "overlap@10": 2, "releff@10": 3}
    s["_s"] = s.scheme.map(scheme_order)
    s["_m"] = s.metric.map(met_order)
    s["_l"] = s.loss.map({l: i for i, l in enumerate(LOSS_ORDER)})
    s = s.sort_values(["_m", "_l", "_s"]).reset_index(drop=True)
    panelA = pd.DataFrame({
        "Metric": s.metric,
        "Loss": s.loss,
        "Scheme": s.scheme.str.replace("_", "\\_", regex=False),
        "$n$": s.n.astype(int),
        "Mean $\\Delta$": s["mean"].map(lambda v: f"{v:.4f}"),
        "95\\% CI": [_ci(lo, hi) for lo, hi in zip(s.ci_lo, s.ci_hi)],
        "$p_{\\mathrm{Holm}}$": s.p_holm.map(_fmt_p),
    })
    texA = panelA.to_latex(index=False, escape=False, longtable=True,
                           column_format="lllrrcc",
                           caption=("Experiment F: does correlation-consistent training help under "
                                    "harder splits? Delta-versus-MSE (affine-calibrated NN) by "
                                    "cross-validation scheme (random, within-environment, "
                                    "leave-family-out, cross-environment), pooled across architectures. "
                                    "Mean with BCa 95\\% CI and Holm-corrected paired-Wilcoxon $p$. "
                                    "The benefit is reliable under random/within-environment splits "
                                    "and attenuates under family/environment extrapolation."),
                           label="tab:S-splits")

    # Panel B: per-architecture Delta from results_splits_full.parquet.
    d = splits.copy()
    if "split_type" in d.columns:
        d["scheme"] = d["split_type"].fillna(d["scheme"])
    aff = d[d.calibration == "affine"]
    dl = compute_deltas(aff, ["pearson", "ndcg@10"])
    rows = []
    for met in ["pearson", "ndcg@10"]:
        per = summarize_deltas(dl, met, group=("loss", "scheme", "model"))
        for _, r in per.iterrows():
            rows.append({"Metric": met, "Loss": r["loss"], "Scheme": r["scheme"],
                         "Architecture": r["model"], "n": int(r["n"]),
                         "Mean": r["mean"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    b = pd.DataFrame(rows)
    b["_s"] = b.Scheme.map(scheme_order)
    b["_m"] = b.Metric.map(met_order)
    b["_l"] = b.Loss.map({l: i for i, l in enumerate(LOSS_ORDER)})
    b["_a"] = b.Architecture.map({"mlp": 0, "cnn": 1, "transformer": 2})
    b = b.sort_values(["_m", "_l", "_s", "_a"]).reset_index(drop=True)
    panelB = pd.DataFrame({
        "Metric": b.Metric,
        "Loss": b.Loss,
        "Scheme": b.Scheme.str.replace("_", "\\_", regex=False),
        "Architecture": b.Architecture,
        "$n$": b.n,
        "Mean $\\Delta$": b.Mean.map(lambda v: f"{v:.4f}"),
        "95\\% CI": [_ci(lo, hi) for lo, hi in zip(b.ci_lo, b.ci_hi)],
    })
    texB = panelB.to_latex(index=False, escape=False, longtable=True,
                           column_format="llllrrc",
                           caption=("Experiment F, per-architecture breakdown. Delta-versus-MSE "
                                    "(affine-calibrated) of Pearson $r$ and NDCG@10 by split scheme "
                                    "and architecture, computed directly from the difficult-splits "
                                    "grid. Bootstrap mean and BCa 95\\% CI."),
                           label="tab:S-splits-arch")
    _write_tex("tabS_splits.tex", texA + "\n\n" + texB)


def table_perdataset(main):
    """tabS_perdataset: longtable, one row per (dataset, trait)."""
    def cell_mean(df, model, loss, cal, met="pearson"):
        sub = df[(df.model == model) & (df.loss == loss) & (df.calibration == cal)]
        return sub[met].mean() if len(sub) else np.nan

    fm = (main.groupby(["dataset", "species", "trait", "model", "loss", "calibration"], as_index=False)
          ["pearson"].mean())
    rows = []
    for (dataset, species, trait), g in main.groupby(["dataset", "species", "trait"]):
        n = int((g.n_train + g.n_test).max())
        p = int(g.p.max())
        gblup = cell_mean(g, "gblup", "na", "raw")
        ridge = cell_mean(g, "ridge", "na", "affine")
        mlp_mse = cell_mean(g, "mlp", "mse", "affine")
        # best correlation-loss NN (affine): over models x {pearson,hybrid,ccc}
        best, best_lab = np.nan, "--"
        for model in NN_MODELS:
            for loss in LOSS_ORDER:
                v = cell_mean(g, model, loss, "affine")
                if not np.isnan(v) and (np.isnan(best) or v > best):
                    best, best_lab = v, f"{model}/{loss}"
        rows.append({"Dataset": dataset, "Species": species, "Trait": trait,
                     "n": n, "p": p, "gblup": gblup, "ridge": ridge,
                     "mlp_mse": mlp_mse, "best": best, "best_lab": best_lab})
    df = pd.DataFrame(rows).sort_values(["Species", "Dataset", "Trait"]).reset_index(drop=True)

    def f(v):
        return "--" if pd.isna(v) else f"{v:.3f}"

    out = pd.DataFrame({
        "Dataset": df.Dataset.str.replace("_", "\\_\\allowbreak{}", regex=False),
        "Trait": df.Trait.str.replace("_", "\\_\\allowbreak{}", regex=False),
        "Species": df.Species,
        "$n$": df.n,
        "$p$": df.p,
        "GBLUP": df.gblup.map(f),
        "Ridge": df.ridge.map(f),
        "MLP/MSE": df.mlp_mse.map(f),
        "Best corr.\\ NN": [f"{f(v)} ({lab})" for v, lab in zip(df.best, df.best_lab)],
    })
    body = out.to_latex(index=False, escape=False, longtable=True,
                        column_format="@{}p{2.25cm}p{1.35cm}p{1.85cm}rrcccp{2.7cm}@{}",
                        caption=("Per (dataset, trait) summary. Test-set Pearson $r$ (mean over "
                                 "cross-validation folds) for GBLUP (raw), ridge (affine), the "
                                 "MSE-trained MLP (affine) and the best correlation-loss neural "
                                 "network (affine; winning model/loss in parentheses). $n$ is the "
                                 "maximum train+test sample count and $p$ the number of markers used. "
                                 "Sorghum is excluded (proprietary)."),
                        label="tab:S-perdataset")
    _write_tex("tabS_perdataset.tex", body)


def table_hpospaces():
    """tabS_hpospaces: NN + classical search spaces from ccgp/hpo.py."""
    def fmt(v):
        return str(v).replace("_", "\\_")

    rows = []
    for arch in hpo.NN_ARCHS:
        for k, vals in hpo.NN_SPACES[arch].items():
            rows.append({"Component": f"NN: {arch}", "Hyper-parameter": fmt(k),
                         "Search values": ", ".join(fmt(v) for v in vals)})
    for k, vals in hpo.COMMON.items():
        rows.append({"Component": "NN: common (optimizer)", "Hyper-parameter": fmt(k),
                     "Search values": ", ".join(fmt(v) for v in vals)})
    for name, space in hpo.CLS_SPACES.items():
        if not space:
            rows.append({"Component": f"Classical: {name}", "Hyper-parameter": "--",
                         "Search values": "no tuning"})
            continue
        for k, vals in space.items():
            rows.append({"Component": f"Classical: {name}", "Hyper-parameter": fmt(k),
                         "Search values": ", ".join(fmt(v) for v in vals)})
    df = pd.DataFrame(rows)
    body = df.to_latex(index=False, escape=False, longtable=True, column_format="llp{7cm}",
                       caption=("Hyper-parameter search spaces. Architecture and optimizer settings "
                                "are tuned once per (dataset, model) with a neutral MSE objective, "
                                "selected by validation Pearson, then frozen and reused across every "
                                "loss so that only the loss changes in the headline comparison. "
                                "Classical models tune their regularization analogously; GBLUP is "
                                "tuning-free. Reproduced from \\texttt{ccgp/hpo.py}."),
                       label="tab:S-hpospaces")
    _write_tex("tabS_hpospaces.tex", body)


def table_hpoconfig():
    """tabS_hpoconfig: selected hyper-parameters per dataset from hpo_full.json."""
    cfg = json.loads((RESULTS / "hpo_full.json").read_text())

    def arch_str(model, p):
        ak = p.get("arch_kwargs", {})
        if model == "mlp":
            return f"hidden={tuple(ak.get('hidden_dims', []))}, drop={ak.get('dropout')}"
        if model == "cnn":
            return (f"ch={tuple(ak.get('channels', []))}, k={ak.get('kernel_size')}, "
                    f"stride={ak.get('first_stride')}, fc={ak.get('fc_dim')}, drop={ak.get('dropout')}")
        if model == "transformer":
            return (f"tok={ak.get('n_tokens')}, d={ak.get('d_model')}, heads={ak.get('n_heads')}, "
                    f"layers={ak.get('n_layers')}, drop={ak.get('dropout')}")
        return ""

    def opt_str(p):
        return f"lr={p.get('lr')}, wd={p.get('weight_decay')}, bs={p.get('batch_size')}"

    rows = []
    for dataset, entry in cfg.items():
        params = entry["params"]
        ridge_alpha = params.get("ridge", {}).get("alpha", "--")
        for model in ["mlp", "cnn", "transformer"]:
            if model not in params or not params[model]:
                continue
            rows.append({
                "Dataset": dataset,
                "Model": model,
                "Architecture": arch_str(model, params[model]),
                "Optimizer": opt_str(params[model]),
                "Ridge $\\alpha$": ridge_alpha if model == "mlp" else "",
            })
    df = pd.DataFrame(rows)

    def esc(s):
        return str(s).replace("_", "\\_")

    out = pd.DataFrame({
        "Dataset": df.Dataset.map(esc),
        "Model": df.Model,
        "Architecture": df.Architecture.map(esc),
        "Optimizer": df.Optimizer.map(esc),
        "Ridge $\\alpha$": df["Ridge $\\alpha$"].map(lambda v: esc(v) if v != "" else ""),
    })
    body = out.to_latex(index=False, escape=False, longtable=True,
                        column_format="llp{4.6cm}p{3.4cm}r",
                        caption=("Selected (frozen) hyper-parameters per dataset, from "
                                 "\\texttt{results/hpo\\_full.json}. Architecture and optimizer "
                                 "(learning rate, weight decay, batch size) for each neural model, "
                                 "plus the tuned ridge penalty $\\alpha$ (shown once per dataset on "
                                 "the MLP row). These are reused unchanged across all losses."),
                        label="tab:S-hpoconfig")
    _write_tex("tabS_hpoconfig.tex", body)


def table_expA():
    """tabS_expA: numerical-verification summary from exp_a.json."""
    j = json.loads((RESULTS / "exp_a.json").read_text())
    p1, p2 = j["proposition1"], j["proposition2"]
    g, ng, rp = j["gradient"], j["neg_pearson_gradient"], j["real_predictions"]

    def sci(v):
        return f"{v:.2e}"

    rows = [
        ("Prop.\\ 1: standardized-MSE $=2(1-r)$", "max abs.\\ diff.", sci(p1["max_abs_diff"]), str(p1["n_cases"])),
        ("Prop.\\ 1: standardized-MSE $=2(1-r)$", "mean abs.\\ diff.", sci(p1["mean_abs_diff"]), str(p1["n_cases"])),
        ("Prop.\\ 2: affine-optimal MSE identity", "max abs.\\ diff.", sci(p2["max_abs_diff"]), str(p2["n_cases"])),
        ("Prop.\\ 2: affine-optimal MSE identity", "mean abs.\\ diff.", sci(p2["mean_abs_diff"]), str(p2["n_cases"])),
        ("Gradient check (Pearson loss)", "\\texttt{gradcheck} passes",
         "yes" if g["pearson_gradcheck"] else "no", "--"),
        ("Gradient check (CCC loss)", "\\texttt{gradcheck} passes",
         "yes" if g["ccc_gradcheck"] else "no", "--"),
        ("Neg-Pearson gradient vs analytic", "max grad.\\ diff.", sci(ng["max_grad_diff"]), "--"),
        (f"Real data identity ({rp['dataset']}/{rp['trait']}, $r={rp['r']:.3f}$)",
         "abs.\\ diff.\\ (identity)", sci(rp["abs_diff_identity"]), "--"),
    ]
    out = pd.DataFrame(rows, columns=["Check", "Quantity", "Value", "$n$"])
    body = out.to_latex(index=False, escape=False, column_format="llrr",
                        caption=("Experiment A: numerical verification of the equivalences. "
                                 "Proposition~1 (standardized MSE equals $2(1-r)$) and "
                                 "Proposition~2 (the affine-optimal MSE identity) are confirmed to "
                                 "machine precision across randomized cases; analytic gradients of "
                                 "the Pearson and CCC losses pass \\texttt{torch.autograd.gradcheck}; "
                                 "and the identity holds exactly on a real dataset. From "
                                 "\\texttt{results/exp\\_a.json}."),
                        label="tab:S-expA", position="ht")
    _write_tex("tabS_expA.tex", body)


# =============================================================================
# Figures
# =============================================================================

def fig_arch_deltas(main):
    """figS_arch_deltas: per-architecture Delta r and Delta NDCG@10 for each loss."""
    aff = main[main.calibration == "affine"]
    d = compute_deltas(aff, ["pearson", "ndcg@10"])
    metrics = [("pearson", r"$\Delta$ Pearson $r$"), ("ndcg@10", r"$\Delta$ NDCG@10")]
    fig, axes = plt.subplots(2, len(NN_MODELS), figsize=(2.7 * len(NN_MODELS), 5.4),
                             sharey="row")
    for i, (met, mlab) in enumerate(metrics):
        for j, model in enumerate(NN_MODELS):
            ax = axes[i, j]
            data = [d[(d.loss == l) & (d.model == model)][met].dropna().values
                    for l in LOSS_ORDER]
            parts = ax.violinplot(data, showmeans=True, showextrema=False)
            for pc, l in zip(parts["bodies"], LOSS_ORDER):
                pc.set_facecolor(CB[l]); pc.set_alpha(0.6)
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xticks(range(1, len(LOSS_ORDER) + 1))
            ax.set_xticklabels(LOSS_ORDER, rotation=30)
            if i == 0:
                ax.set_title(model)
            if j == 0:
                ax.set_ylabel(mlab)
    fig.suptitle("Per-architecture loss benefit vs MSE (affine-calibrated)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save_fig(fig, "figS_arch_deltas")


def fig_selection_k(main):
    """figS_selection_k: pooled Delta of overlap@k, ndcg@k, releff@k at each k."""
    aff = main[main.calibration == "affine"]
    fams = [("overlap", "$\\Delta$ overlap@k"), ("ndcg", "$\\Delta$ NDCG@k"),
            ("releff", "$\\Delta$ relative efficiency@k")]
    mets = [f"{fam}@{k}" for fam, _ in fams for k in KS]
    d = compute_deltas(aff, mets)
    fig, axes = plt.subplots(1, len(fams), figsize=(3.0 * len(fams), 3.4), sharex=True)
    width = 0.25
    x = np.arange(len(KS))
    for ax, (fam, lab) in zip(axes, fams):
        for li, loss in enumerate(LOSS_ORDER):
            means, los, his = [], [], []
            for k in KS:
                s = summarize_deltas(d[d.loss == loss], f"{fam}@{k}", group=("loss",))
                r = s.iloc[0]
                means.append(r["mean"]); los.append(r["mean"] - r["ci_lo"])
                his.append(r["ci_hi"] - r["mean"])
            ax.bar(x + (li - 1) * width, means, width, yerr=[los, his],
                   color=CB[loss], alpha=0.8, label=loss, capsize=2, ecolor="0.3")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in KS])
        ax.set_title(lab)
    axes[0].set_ylabel("loss $-$ MSE (pooled)")
    axes[-1].legend(title="loss", fontsize=8)
    fig.suptitle("Selection benefit vs selection intensity (pooled, affine-calibrated NN)", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "figS_selection_k")


def fig_calibration(main):
    """figS_calibration: raw-vs-affine slope, RMSE, intercept for Pearson-trained NN."""
    d = main[(main.model.isin(NN_MODELS)) & (main.loss == "pearson")]
    piv = d.pivot_table(index=["dataset", "trait", "model", "repeat", "fold"],
                        columns="calibration", values=["slope", "rmse", "intercept"])
    fig, ax = plt.subplots(1, 3, figsize=(9.2, 3.0))
    specs = [("slope", "calibration slope", 1.0), ("rmse", "RMSE", None),
             ("intercept", "intercept", 0.0)]
    for a, (met, ttl, ideal) in zip(ax, specs):
        if ("raw" not in piv[met]) or ("affine" not in piv[met]):
            continue
        x, y = piv[met]["raw"].values, piv[met]["affine"].values
        m = ~(np.isnan(x) | np.isnan(y))
        x, y = x[m], y[m]
        lo, hi = np.nanpercentile(np.concatenate([x, y]), [1, 99])
        a.scatter(x, y, s=6, alpha=0.3, c=CB["pearson"])
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        if ideal is not None:
            a.axhline(ideal, color=CB["affine"], lw=1)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_xlabel(f"{ttl} (raw)"); a.set_ylabel(f"{ttl} (affine)"); a.set_title(ttl)
    fig.suptitle("Affine calibration of Pearson-trained NN: slope$\\to$1, intercept$\\to$0, "
                 "RMSE reduced", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "figS_calibration")


def fig_meta(main):
    """figS_meta: Delta NDCG@10 (Pearson loss) vs training n and vs GBLUP ability, by species."""
    aff = main[main.calibration == "affine"]
    d = compute_deltas(aff, ["ndcg@10"])
    dd = d[d.loss == "pearson"]
    # covariates per (dataset, trait): training n and GBLUP raw ability
    g = main[(main.model == "gblup") & (main.calibration == "raw")]
    cov = (g.groupby(["dataset", "species", "trait"])
           .agg(gblup_r=("pearson", "mean"), n=("n_train", "mean")).reset_index())
    dd = dd.merge(cov, on=["dataset", "species", "trait"], how="left")
    species = sorted(dd.species.unique())
    cmap = {s: SPECIES_C[i] for i, s in enumerate(species)}
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for s in species:
        sub = dd[dd.species == s]
        ax[0].scatter(sub.n, sub["ndcg@10"], s=22, color=cmap[s], alpha=0.85,
                      edgecolor="w", lw=0.3, label=s)
        ax[1].scatter(sub.gblup_r, sub["ndcg@10"], s=22, color=cmap[s], alpha=0.85,
                      edgecolor="w", lw=0.3)
    for a in ax:
        a.axhline(0, color="k", ls="--", lw=0.8)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("n (training)"); ax[0].set_ylabel(r"$\Delta$NDCG@10 (Pearson $-$ MSE)")
    ax[1].set_xlabel("GBLUP predictive ability (r)"); ax[1].set_ylabel(r"$\Delta$NDCG@10")
    ax[0].legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle("When does the Pearson loss help selection?", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "figS_meta")


def fig_batch(batch):
    """figS_batch: test Pearson vs batch size per dataset for the Pearson loss (Exp E)."""
    b = batch[(batch.loss == "pearson") & (batch.calibration == "affine")]
    g = (b.groupby(["dataset", "batch_size"])["pearson"].mean().reset_index())
    datasets = sorted(g.dataset.unique())
    cmap = {d: SPECIES_C[i] for i, d in enumerate(datasets)}
    order = sorted(g.batch_size.unique())
    labels = ["full" if bs == -1 else str(bs) for bs in order]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for d in datasets:
        sub = g[g.dataset == d].set_index("batch_size").reindex(order)
        ax.plot(range(len(order)), sub["pearson"].values, "o-", color=cmap[d],
                label=d.replace("easygese:", ""))
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels)
    ax.set_xlabel("minibatch size"); ax.set_ylabel("test Pearson $r$ (affine)")
    ax.set_title("Pearson loss is batch-size robust (Exp E)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, "figS_batch")


# =============================================================================
# driver
# =============================================================================

def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "full"
    OUT.mkdir(parents=True, exist_ok=True)
    main_df = pd.read_parquet(RESULTS / f"results_main_{preset}.parquet")
    batch_df = pd.read_parquet(RESULTS / f"results_batch_{preset}.parquet")
    splits_df = pd.read_parquet(RESULTS / f"results_splits_{preset}.parquet")
    print(f"=== make_supplement ({preset}): main {main_df.shape}, "
          f"batch {batch_df.shape}, splits {splits_df.shape} ===")

    # tables
    table_expA()
    table_hpospaces()
    table_hpoconfig()
    table_deltas_full(main_df)
    table_lmm()
    table_ranks()
    table_calibration()
    table_batch()
    table_splits(splits_df)
    table_perdataset(main_df)

    # figures
    fig_arch_deltas(main_df)
    fig_selection_k(main_df)
    fig_calibration(main_df)
    fig_meta(main_df)
    fig_batch(batch_df)
    print(f"\nAll fragments and figures written to {OUT}/")


if __name__ == "__main__":
    main()
