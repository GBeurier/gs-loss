"""Classical genomic predictors: GBLUP (REML) and ML baselines.

All estimators expose scikit-learn-style ``fit(X, y)`` / ``predict(X)`` on a
(standardized) marker matrix ``X`` (n x p).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge


def _kernel_eigh(X: np.ndarray, p: int):
    """Linear genomic kernel K = X X'/p and its symmetric eigendecomposition.

    Uses the GPU (torch, float64) when available -- an O(n^3) eigendecomposition
    is ~10x faster there for n in the thousands -- and falls back to NumPy.
    """
    try:
        import torch

        if torch.cuda.is_available():
            Xt = torch.as_tensor(X, device="cuda", dtype=torch.float64)
            K = (Xt @ Xt.T) / p
            lam, Q = torch.linalg.eigh(K)
            return lam.cpu().numpy(), Q.cpu().numpy()
    except Exception:
        pass
    K = (X @ X.T) / p
    return np.linalg.eigh(K)


class GBLUP:
    """Genomic BLUP / RR-BLUP with REML-estimated variance components.

    Model ``y = 1*mu + g + e`` with ``g ~ N(0, sigma_g^2 K)``,
    ``e ~ N(0, sigma_e^2 I)`` and linear genomic kernel ``K = X X' / p``.
    The variance ratio ``delta = sigma_e^2 / sigma_g^2`` is fit by maximizing the
    restricted likelihood (profiled, intercept as the only fixed effect), solved
    in the spectral basis of K. Predictions use the BLUP
    ``g_* = K_* (K + delta I)^{-1} (y - mu)``.
    """

    def __init__(self, n_grid: int = 0) -> None:
        self.n_grid = n_grid  # reserved; optimization is via bounded Brent

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBLUP":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape
        self.X_train_ = X
        self.p_ = p
        # symmetric eigendecomposition of K = X X'/p (ascending eigenvalues)
        lam, Q = _kernel_eigh(X, p)
        lam = np.clip(lam, 0.0, None)
        self.lam_, self.Q_ = lam, Q
        yt = Q.T @ y                      # transformed response
        xt = Q.T @ np.ones(n)             # transformed intercept design
        q = 1

        def neg_restricted_ll(t: float) -> float:
            delta = np.exp(t)
            w = 1.0 / (lam + delta)
            xax = float(np.sum(xt * xt * w))
            xay = float(np.sum(xt * yt * w))
            yay = float(np.sum(yt * yt * w))
            yp0y = yay - xay * xay / xax
            if yp0y <= 0:
                return 1e18
            return (n - q) * np.log(yp0y) + float(np.sum(np.log(lam + delta))) + np.log(xax)

        res = minimize_scalar(neg_restricted_ll, bounds=(-15.0, 15.0), method="bounded")
        delta = float(np.exp(res.x))
        w = 1.0 / (lam + delta)
        xax = float(np.sum(xt * xt * w))
        xay = float(np.sum(xt * yt * w))
        yay = float(np.sum(yt * yt * w))
        yp0y = yay - xay * xay / xax
        self.delta_ = delta
        self.mu_ = xay / xax                       # GLS intercept
        self.sigma_g2_ = yp0y / (n - q)
        self.sigma_e2_ = delta * self.sigma_g2_
        self.h2_ = 1.0 / (1.0 + delta)             # K with ~unit scale -> ratio proxy
        # c = (K + delta I)^{-1} (y - mu)
        r = y - self.mu_
        self.c_ = Q @ (w * (Q.T @ r))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        k_star = (X @ self.X_train_.T) / self.p_
        return self.mu_ + k_star @ self.c_


def _xgb(**params):
    from xgboost import XGBRegressor

    defaults = dict(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
        tree_method="hist", n_jobs=params.pop("n_jobs", 4), verbosity=0,
    )
    defaults.update(params)
    return XGBRegressor(**defaults)


def make_classical(name: str, random_state: int = 0, **params):
    """Factory for a classical estimator with fit/predict."""
    name = name.lower()
    if name == "gblup" or name == "rrblup":
        return GBLUP()
    if name == "ridge":
        # lsqr is an iterative solver that is fast when p >> n (genomic markers),
        # unlike the default cholesky which forms a p x p matrix.
        return Ridge(alpha=params.get("alpha", 1.0), solver="lsqr")
    if name == "elasticnet":
        return ElasticNet(
            alpha=params.get("alpha", 0.01),
            l1_ratio=params.get("l1_ratio", 0.5),
            max_iter=params.get("max_iter", 5000),
        )
    if name == "rf" or name == "randomforest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", None),
            max_features=params.get("max_features", "sqrt"),
            n_jobs=params.get("n_jobs", 4),
            random_state=random_state,
        )
    if name == "xgboost" or name == "xgb":
        return _xgb(random_state=random_state, **params)
    raise ValueError(f"Unknown classical model '{name}'")


CLASSICAL_MODELS = ["gblup", "ridge", "elasticnet", "rf", "xgboost"]
