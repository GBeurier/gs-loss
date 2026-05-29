"""Predictive and selection-oriented evaluation metrics (NumPy).

Two families:

* **Predictive** -- Pearson r, Spearman rho, RMSE, MAE, R^2, mean bias, and the
  calibration regression (slope/intercept of ``y_true ~ y_pred``). Ideal
  calibration has slope 1 and intercept 0.
* **Selection** -- for a selection fraction alpha, the candidates ranked highest
  by ``y_pred`` are "selected"; we score how good that selection is against the
  truth: top-k overlap, precision@k, recall@k, NDCG@k, selection differential,
  mean selected phenotype, and relative efficiency (achieved gain / oracle gain).

Selection metrics assume **larger phenotype = better** (a standard genetic-gain
convention). Because all methods are compared on the same trait with the same
convention, the resulting deltas between methods are well defined regardless of
the agronomic direction of the trait.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

DEFAULT_FRACS = (0.05, 0.10, 0.15, 0.20)


# --- predictive --------------------------------------------------------------

def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_pred) < 1e-12:
        return np.nan
    return float(spearmanr(y_true, y_pred).statistic)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    ss_res = np.sum((y_true - np.asarray(y_pred, float)) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error E[pred - true]."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def calibration_slope_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of the regression ``y_true ~ y_pred``.

    Ideal predictions give slope 1, intercept 0.
    """
    y_pred = np.asarray(y_pred, float)
    y_true = np.asarray(y_true, float)
    vp = np.var(y_pred)
    if vp < 1e-12:
        return np.nan, np.nan
    slope = np.cov(y_pred, y_true, bias=True)[0, 1] / vp
    intercept = y_true.mean() - slope * y_pred.mean()
    return float(slope), float(intercept)


def predictive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    slope, intercept = calibration_slope_intercept(y_true, y_pred)
    return {
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "slope": slope,
        "intercept": intercept,
    }


# --- selection ---------------------------------------------------------------

def _top_idx(values: np.ndarray, m: int) -> np.ndarray:
    """Indices of the ``m`` largest values (unordered)."""
    return np.argpartition(values, -m)[-m:]


def top_k_overlap(y_true: np.ndarray, y_pred: np.ndarray, frac: float) -> float:
    """Fraction of the true top-alpha set recovered by the predicted top-alpha set.

    With equal selected/relevant set sizes this equals precision@k and recall@k.
    """
    n = len(y_true)
    m = max(1, int(np.ceil(frac * n)))
    sel = set(_top_idx(np.asarray(y_pred, float), m).tolist())
    true = set(_top_idx(np.asarray(y_true, float), m).tolist())
    return len(sel & true) / m


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, frac: float) -> float:
    """Normalized Discounted Cumulative Gain at the top-alpha cut.

    Linear gains equal to the (min-shifted, non-negative) phenotype, items
    ordered by predicted score. NDCG = DCG / ideal-DCG in [0, 1].
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    m = max(1, int(np.ceil(frac * n)))
    rel = y_true - y_true.min()  # non-negative gains
    order_pred = np.argsort(-y_pred, kind="stable")[:m]
    order_ideal = np.argsort(-rel, kind="stable")[:m]
    discounts = 1.0 / np.log2(np.arange(2, m + 2))
    dcg = float(np.sum(rel[order_pred] * discounts))
    idcg = float(np.sum(rel[order_ideal] * discounts))
    if idcg < 1e-12:
        return np.nan
    return dcg / idcg


def selection_differential(y_true: np.ndarray, y_pred: np.ndarray, frac: float) -> float:
    """Mean phenotype of selected minus population mean (original units)."""
    y_true = np.asarray(y_true, float)
    n = len(y_true)
    m = max(1, int(np.ceil(frac * n)))
    sel = _top_idx(np.asarray(y_pred, float), m)
    return float(y_true[sel].mean() - y_true.mean())


def mean_selected(y_true: np.ndarray, y_pred: np.ndarray, frac: float) -> float:
    y_true = np.asarray(y_true, float)
    n = len(y_true)
    m = max(1, int(np.ceil(frac * n)))
    sel = _top_idx(np.asarray(y_pred, float), m)
    return float(y_true[sel].mean())


def relative_efficiency(y_true: np.ndarray, y_pred: np.ndarray, frac: float) -> float:
    """Achieved selection gain divided by oracle (truth-based) gain in [.,1]."""
    y_true = np.asarray(y_true, float)
    n = len(y_true)
    m = max(1, int(np.ceil(frac * n)))
    mu = y_true.mean()
    gain_model = y_true[_top_idx(np.asarray(y_pred, float), m)].mean() - mu
    gain_oracle = y_true[_top_idx(y_true, m)].mean() - mu
    if abs(gain_oracle) < 1e-12:
        return np.nan
    return float(gain_model / gain_oracle)


def selection_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      fracs: tuple[float, ...] = DEFAULT_FRACS) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in fracs:
        pct = int(round(f * 100))
        ov = top_k_overlap(y_true, y_pred, f)
        out[f"overlap@{pct}"] = ov
        out[f"precision@{pct}"] = ov       # equal-size sets => precision == recall == overlap
        out[f"recall@{pct}"] = ov
        out[f"ndcg@{pct}"] = ndcg_at_k(y_true, y_pred, f)
        out[f"seldiff@{pct}"] = selection_differential(y_true, y_pred, f)
        out[f"meansel@{pct}"] = mean_selected(y_true, y_pred, f)
        out[f"releff@{pct}"] = relative_efficiency(y_true, y_pred, f)
    return out


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                fracs: tuple[float, ...] = DEFAULT_FRACS) -> dict[str, float]:
    """Merged predictive + selection metrics."""
    m = predictive_metrics(y_true, y_pred)
    m.update(selection_metrics(y_true, y_pred, fracs))
    return m
