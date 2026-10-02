"""Differentiable training objectives for genomic prediction.

The central object is the **Pearson loss**, written as a standardized MSE
(Proposition 1 of the paper):

    For z_y = (y - mean y)/std y and z_p = (p - mean p)/std p (population std),
        MSE(z_y, z_p) = 2 (1 - r),     hence     1 - r = 1/2 MSE(z_y, z_p).

All losses take ``(pred, target)`` 1-D tensors and return a scalar tensor. They
also accept an optional non-negative ``weight`` vector. This supports breeding
programs such as SelGenPalm, where cross means have heterogeneous precision and
the operational endpoint is a precision-weighted Pearson correlation.
Population (1/N) moments are used throughout so the identity above holds
exactly; the small ``eps`` guards the variance denominators when a batch is
(near-)constant early in training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def _normalized_weight(x: torch.Tensor, weight: torch.Tensor | None,
                       eps: float = EPS) -> torch.Tensor | None:
    """Return non-negative weights summing to one, or ``None`` for uniform moments."""
    if weight is None:
        return None
    w = torch.as_tensor(weight, dtype=x.dtype, device=x.device).reshape(-1)
    if w.shape != x.reshape(-1).shape:
        raise ValueError("weight must have the same number of elements as the target")
    if not torch.isfinite(w).all() or torch.any(w < 0):
        raise ValueError("weight must be finite and non-negative")
    total = w.sum()
    if total <= eps:
        raise ValueError("weight must have a positive sum")
    return w / total


def _mean(x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    return x.mean() if weight is None else torch.sum(weight * x)


def _standardize(x: torch.Tensor, eps: float = EPS,
                 weight: torch.Tensor | None = None) -> torch.Tensor:
    """Center and scale to unit population variance."""
    w = _normalized_weight(x, weight, eps)
    x = x - _mean(x, w)
    var = torch.mean(x * x) if w is None else torch.sum(w * x * x)
    return x / torch.sqrt(var + eps)


def pearson_corr(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS,
                 weight: torch.Tensor | None = None) -> torch.Tensor:
    """Pearson correlation coefficient r in [-1, 1] (differentiable)."""
    w = _normalized_weight(target, weight, eps)
    p = pred - _mean(pred, w)
    t = target - _mean(target, w)
    if w is None:
        num, vp, vt = torch.mean(p * t), torch.mean(p * p), torch.mean(t * t)
    else:
        num, vp, vt = torch.sum(w * p * t), torch.sum(w * p * p), torch.sum(w * t * t)
    den = torch.sqrt(vp + eps) * torch.sqrt(vt + eps)
    return num / den


def concordance_corr(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS,
                     weight: torch.Tensor | None = None) -> torch.Tensor:
    """Lin's concordance correlation coefficient rho_c (differentiable).

    rho_c = 2 cov(p,t) / (var(p) + var(t) + (mean p - mean t)^2)
    Penalizes correlation, scale and location jointly.
    """
    w = _normalized_weight(target, weight, eps)
    mp, mt = _mean(pred, w), _mean(target, w)
    if w is None:
        vp = torch.mean((pred - mp) ** 2)
        vt = torch.mean((target - mt) ** 2)
        cov = torch.mean((pred - mp) * (target - mt))
    else:
        vp = torch.sum(w * (pred - mp) ** 2)
        vt = torch.sum(w * (target - mt) ** 2)
        cov = torch.sum(w * (pred - mp) * (target - mt))
    return 2 * cov / (vp + vt + (mp - mt) ** 2 + eps)


# --- losses -----------------------------------------------------------------

def mse_loss(pred: torch.Tensor, target: torch.Tensor,
             weight: torch.Tensor | None = None) -> torch.Tensor:
    w = _normalized_weight(target, weight)
    return F.mse_loss(pred, target) if w is None else torch.sum(w * (pred - target) ** 2)


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS,
                 weight: torch.Tensor | None = None) -> torch.Tensor:
    """1 - r. Maximizing correlation == minimizing this."""
    return 1.0 - pearson_corr(pred, target, eps, weight)


def std_mse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS,
                 weight: torch.Tensor | None = None) -> torch.Tensor:
    """1/2 * MSE(standardize(pred), standardize(target)).

    Numerically identical to :func:`pearson_loss` (Proposition 1); provided so
    the equivalence is explicit and directly usable as a training loss.
    """
    w = _normalized_weight(target, weight, eps)
    zp = _standardize(pred, eps, w)
    zt = _standardize(target, eps, w)
    return 0.5 * (F.mse_loss(zp, zt) if w is None else torch.sum(w * (zp - zt) ** 2))


def neg_pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS,
                     weight: torch.Tensor | None = None) -> torch.Tensor:
    """-r. Differs from :func:`pearson_loss` by a constant -> identical gradients."""
    return -pearson_corr(pred, target, eps, weight)


def hybrid_loss(pred: torch.Tensor, target: torch.Tensor, lam: float = 0.1,
                eps: float = EPS, weight: torch.Tensor | None = None) -> torch.Tensor:
    """(1 - r) + lam * MSE(pred, target): correlation shape plus a scale anchor."""
    return pearson_loss(pred, target, eps, weight) + lam * mse_loss(pred, target, weight)


def ccc_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS,
             weight: torch.Tensor | None = None) -> torch.Tensor:
    """1 - rho_c."""
    return 1.0 - concordance_corr(pred, target, eps, weight)


def ranking_loss(pred: torch.Tensor, target: torch.Tensor, n_pairs: int = 4096,
                 eps: float = EPS, weight: torch.Tensor | None = None) -> torch.Tensor:
    """RankNet-style pairwise logistic ranking loss (optional, for top-k).

    Samples ``n_pairs`` ordered pairs (i, j) with target_i > target_j and
    penalizes -log sigmoid(pred_i - pred_j). Pairs are weighted by the target
    gap so that getting the extremes right matters more.
    """
    n = pred.shape[0]
    if n < 2:
        return pred.sum() * 0.0
    g = torch.Generator(device=pred.device)
    i = torch.randint(0, n, (n_pairs,), generator=g, device=pred.device)
    j = torch.randint(0, n, (n_pairs,), generator=g, device=pred.device)
    gap = target[i] - target[j]
    sign = torch.sign(gap)
    mask = sign != 0
    if mask.sum() == 0:
        return pred.sum() * 0.0
    diff = (pred[i] - pred[j]) * sign
    pair_weight = gap.abs()
    if weight is not None:
        sample_weight = _normalized_weight(target, weight, eps)
        pair_weight = pair_weight * 0.5 * (sample_weight[i] + sample_weight[j])
    loss = F.softplus(-diff) * pair_weight
    return loss[mask].sum() / (pair_weight[mask].sum() + eps)


_REGISTRY = {
    "mse": mse_loss,
    "pearson": pearson_loss,        # 1 - r
    "std_mse": std_mse_loss,        # 1/2 MSE(z_y, z_p)  == pearson
    "neg_pearson": neg_pearson_loss,
    "hybrid": hybrid_loss,
    "ccc": ccc_loss,
    "ranking": ranking_loss,
}


def get_loss(name: str, **kwargs):
    """Return a loss callable ``f(pred, target, weight=None) -> scalar`` by name.

    Extra kwargs (e.g. ``lam`` for ``hybrid``) are bound into the callable.
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {sorted(_REGISTRY)}")
    base = _REGISTRY[name]
    if kwargs:
        return lambda pred, target, weight=None: base(pred, target, weight=weight, **kwargs)
    return base


def available_losses() -> list[str]:
    return sorted(_REGISTRY)
