"""Differentiable training objectives for genomic prediction.

The central object is the **Pearson loss**, written as a standardized MSE
(Proposition 1 of the paper):

    For z_y = (y - mean y)/std y and z_p = (p - mean p)/std p (population std),
        MSE(z_y, z_p) = 2 (1 - r),     hence     1 - r = 1/2 MSE(z_y, z_p).

All losses take ``(pred, target)`` 1-D tensors and return a scalar tensor.
Population (1/N) moments are used throughout so the identity above holds
exactly; the small ``eps`` guards the variance denominators when a batch is
(near-)constant early in training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def _standardize(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Center and scale to unit population variance."""
    x = x - x.mean()
    return x / torch.sqrt(torch.mean(x * x) + eps)


def pearson_corr(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Pearson correlation coefficient r in [-1, 1] (differentiable)."""
    p = pred - pred.mean()
    t = target - target.mean()
    num = torch.mean(p * t)
    den = torch.sqrt(torch.mean(p * p) + eps) * torch.sqrt(torch.mean(t * t) + eps)
    return num / den


def concordance_corr(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Lin's concordance correlation coefficient rho_c (differentiable).

    rho_c = 2 cov(p,t) / (var(p) + var(t) + (mean p - mean t)^2)
    Penalizes correlation, scale and location jointly.
    """
    mp, mt = pred.mean(), target.mean()
    vp = torch.mean((pred - mp) ** 2)
    vt = torch.mean((target - mt) ** 2)
    cov = torch.mean((pred - mp) * (target - mt))
    return 2 * cov / (vp + vt + (mp - mt) ** 2 + eps)


# --- losses -----------------------------------------------------------------

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """1 - r. Maximizing correlation == minimizing this."""
    return 1.0 - pearson_corr(pred, target, eps)


def std_mse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """1/2 * MSE(standardize(pred), standardize(target)).

    Numerically identical to :func:`pearson_loss` (Proposition 1); provided so
    the equivalence is explicit and directly usable as a training loss.
    """
    return 0.5 * F.mse_loss(_standardize(pred, eps), _standardize(target, eps))


def neg_pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """-r. Differs from :func:`pearson_loss` by a constant -> identical gradients."""
    return -pearson_corr(pred, target, eps)


def hybrid_loss(pred: torch.Tensor, target: torch.Tensor, lam: float = 0.1,
                eps: float = EPS) -> torch.Tensor:
    """(1 - r) + lam * MSE(pred, target): correlation shape plus a scale anchor."""
    return pearson_loss(pred, target, eps) + lam * F.mse_loss(pred, target)


def ccc_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """1 - rho_c."""
    return 1.0 - concordance_corr(pred, target, eps)


def ranking_loss(pred: torch.Tensor, target: torch.Tensor, n_pairs: int = 4096,
                 eps: float = EPS) -> torch.Tensor:
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
    w = gap.abs()
    loss = F.softplus(-diff) * w
    return loss[mask].sum() / (w[mask].sum() + eps)


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
    """Return a loss callable ``f(pred, target) -> scalar`` by name.

    Extra kwargs (e.g. ``lam`` for ``hybrid``) are bound into the callable.
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {sorted(_REGISTRY)}")
    base = _REGISTRY[name]
    if kwargs:
        return lambda pred, target: base(pred, target, **kwargs)
    return base


def available_losses() -> list[str]:
    return sorted(_REGISTRY)
