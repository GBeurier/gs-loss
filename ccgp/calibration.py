"""Post-hoc calibration of raw predictions.

Calibrators are *fit on held-out (validation) predictions* and applied to test
predictions -- never fit on the test set. They map a raw prediction ``p`` to a
calibrated ``y_hat``:

* ``raw``      -- identity (control).
* ``affine``   -- ``y_hat = a p + b`` with the least-squares optimum
  ``a* = cov(y, p)/var(p)``, ``b* = mean(y) - a* mean(p)`` (Proposition 2).
  With ``a* > 0`` this is monotone increasing, so it leaves Pearson r, Spearman
  rho and every ranking/selection metric unchanged while restoring scale and
  offset (and hence RMSE, bias, calibration slope).
* ``isotonic`` -- monotone non-parametric fit (handles a monotone but non-linear
  raw-vs-true relationship); also rank-preserving.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class RawCalibrator:
    name = "raw"

    def fit(self, p: np.ndarray, y: np.ndarray) -> "RawCalibrator":
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p, float)


class AffineCalibrator:
    name = "affine"

    def __init__(self) -> None:
        self.a_ = 1.0
        self.b_ = 0.0

    def fit(self, p: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "AffineCalibrator":
        p = np.asarray(p, float)
        y = np.asarray(y, float)
        if sample_weight is None:
            mp, my = p.mean(), y.mean()
            vp = np.mean((p - mp) ** 2)
            cov = np.mean((p - mp) * (y - my))
        else:
            w = np.asarray(sample_weight, float).reshape(-1)
            if len(w) != len(y) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
                raise ValueError("sample_weight must be finite, non-negative, and aligned")
            w = w / w.sum()
            mp, my = np.sum(w * p), np.sum(w * y)
            vp = np.sum(w * (p - mp) ** 2)
            cov = np.sum(w * (p - mp) * (y - my))
        if vp < 1e-12:
            self.a_, self.b_ = 0.0, float(my)
        else:
            self.a_ = float(cov / vp)
            self.b_ = float(my - self.a_ * mp)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.a_ * np.asarray(p, float) + self.b_


class PositiveAffineCalibrator(AffineCalibrator):
    """Weighted-capable affine fit constrained to a non-decreasing map.

    The unconstrained validation slope occasionally becomes negative and flips
    every test-set rank. Clipping it at a tiny positive value implements the
    constrained least-squares boundary while preserving prediction order.
    """
    name = "affine_positive"

    def __init__(self, min_slope: float = 1e-8) -> None:
        super().__init__()
        self.min_slope = min_slope

    def fit(self, p: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "PositiveAffineCalibrator":
        super().fit(p, y, sample_weight=sample_weight)
        if self.a_ < self.min_slope:
            p = np.asarray(p, float)
            y = np.asarray(y, float)
            if sample_weight is None:
                mp, my = p.mean(), y.mean()
            else:
                w = np.asarray(sample_weight, float)
                w = w / w.sum()
                mp, my = np.sum(w * p), np.sum(w * y)
            self.a_ = float(self.min_slope)
            self.b_ = float(my - self.a_ * mp)
        return self


class IsotonicCalibrator:
    name = "isotonic"

    def __init__(self) -> None:
        self.iso_ = IsotonicRegression(out_of_bounds="clip", increasing=True)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        p = np.asarray(p, float)
        y = np.asarray(y, float)
        if np.var(p) < 1e-12:
            self._const = float(y.mean())
        else:
            self._const = None
            self.iso_.fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        if getattr(self, "_const", None) is not None:
            return np.full_like(p, self._const)
        return self.iso_.predict(p)


_CALIBRATORS = {
    "raw": RawCalibrator,
    "affine": AffineCalibrator,
    "affine_positive": PositiveAffineCalibrator,
    "isotonic": IsotonicCalibrator,
}


def get_calibrator(name: str):
    name = name.lower()
    if name not in _CALIBRATORS:
        raise ValueError(f"Unknown calibrator '{name}'. Available: {sorted(_CALIBRATORS)}")
    return _CALIBRATORS[name]()


def available_calibrators() -> list[str]:
    return list(_CALIBRATORS)
