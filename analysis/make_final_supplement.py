"""Generate compact LaTeX tables for the final Supporting Information."""
from pathlib import Path

import pandas as pd


SRC = Path("results/analysis_final")
OUT = Path("paper/supp")


def write(df, name, caption, label, float_format="%.4f"):
    OUT.mkdir(parents=True, exist_ok=True)
    text = df.to_latex(index=False, longtable=True, escape=True,
                       float_format=float_format, caption=caption, label=label)
    (OUT / name).write_text(text)


def main():
    s = pd.read_csv(SRC / "paired_summary.csv")
    pearson = s[(s.metric == "pearson") & (s.weighting == "panel_balanced")][
        ["loss", "model", "n_tasks", "n_panels", "mean_delta", "ci_lo", "ci_hi",
         "panel_signflip_p", "win_fraction"]]
    write(pearson, "tabS_final_pearson.tex",
          "Panel-balanced raw-Pearson contrasts relative to MSE. Intervals resample panels; the exact test flips signs at panel level.",
          "tab:s-final-pearson")

    pooled = s[(s.model == "pooled") & (s.weighting == "panel_balanced")][
        ["metric", "calibration", "loss", "n_tasks", "mean_delta", "ci_lo", "ci_hi",
         "panel_signflip_p"]]
    write(pooled, "tabS_final_pooled.tex",
          "Panel-balanced pooled contrasts for all prespecified endpoints.",
          "tab:s-final-pooled")

    ranks = pd.read_csv(SRC / "method_ranks.csv")
    wide = ranks.pivot(index="method", columns="metric", values="mean_rank").reset_index()
    wide = wide.sort_values("pearson")
    write(wide, "tabS_final_ranks.tex", "Mean ranks over 101 tasks using raw predictions.",
          "tab:s-final-ranks", "%.2f")

    cal = pd.read_csv(SRC / "calibration_audit.csv")[[
        "calibration", "n_fold_cells", "median_pearson", "median_rmse",
        "median_slope", "raw_affine_sign_flips", "raw_affine_sign_flip_fraction"]]
    cal.columns = ["map", "n", "median r", "median RMSE", "median slope",
                   "sign flips", "flip fraction"]
    write(cal, "tabS_final_calibration.tex",
          "Calibration audit for Pearson-trained networks. Raw-unit means are descriptive across heterogeneous traits.",
          "tab:s-final-calibration")

    nested = pd.read_csv(SRC / "nested_hpo_summary.csv")[[
        "metric", "model", "mean_delta", "min_task_model_delta",
        "max_task_model_delta", "n_fold_seed_pairs"]]
    nested.columns = ["metric", "model", "mean delta", "minimum", "maximum", "pairs"]
    write(nested, "tabS_final_nested.tex",
          "Loss-specific nested-HPO sensitivity on CIMMYT wheat; repeated outcomes are summarized descriptively.",
          "tab:s-final-nested")

    mixed = pd.read_csv(SRC / "mixed_model_interaction_tests.csv")[[
        "metric", "n_tasks", "n_panels", "lr", "df", "p_chi2",
        "task_random_intercept_variance"]]
    mixed.columns = ["metric", "tasks", "panels", "LR", "df", "p", "task variance"]
    write(mixed, "tabS_final_mixed.tex",
          "Likelihood-ratio tests for the loss-by-architecture interaction in corroborative mixed models.",
          "tab:s-final-mixed")


if __name__ == "__main__":
    main()
