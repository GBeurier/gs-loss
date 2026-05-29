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

    def fit(self, p: np.ndarray, y: np.ndarray) -> "AffineCalibrator":
        p = np.asarray(p, float)
        y = np.asarray(y, float)
        vp = np.var(p)
        if vp < 1e-12:
            self.a_, self.b_ = 0.0, float(y.mean())
        else:
            self.a_ = float(np.cov(p, y, bias=True)[0, 1] / vp)
            self.b_ = float(y.mean() - self.a_ * p.mean())
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.a_ * np.asarray(p, float) + self.b_


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
    "isotonic": IsotonicCalibrator,
}


def get_calibrator(name: str):
    name = name.lower()
    if name not in _CALIBRATORS:
        raise ValueError(f"Unknown calibrator '{name}'. Available: {sorted(_CALIBRATORS)}")
    return _CALIBRATORS[name]()


def available_calibrators() -> list[str]:
    return list(_CALIBRATORS)
