"""Fast unit tests for the scientific core (run with `pytest -q`)."""
import numpy as np
import torch

from ccgp import losses
from ccgp.calibration import AffineCalibrator, IsotonicCalibrator, RawCalibrator
from ccgp.metrics import (all_metrics, ndcg_at_k, pearson, relative_efficiency,
                          rmse, top_k_overlap)
from ccgp.models.classical import GBLUP


def test_proposition1_identity():
    """1 - r == 1/2 MSE(z_y, z_p) == pearson_loss."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = torch.tensor(rng.normal(size=rng.integers(30, 400)))
        p = rng.uniform(-2, 2) * y + torch.tensor(rng.normal(size=len(y))) * 1.3
        r = losses.pearson_corr(p, y).item()
        assert abs(losses.pearson_loss(p, y).item() - (1 - r)) < 1e-6
        assert abs(losses.std_mse_loss(p, y).item() - (1 - r)) < 1e-6


def test_proposition2_affine_mse_min():
    """min MSE(ap+b, y) == var(y)(1 - r^2)."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        y = rng.normal(size=500)
        p = rng.uniform(0.2, 3) * y + rng.normal(size=500) * 2 + 5
        cal = AffineCalibrator().fit(p, y)
        mse_min = np.mean((cal.transform(p) - y) ** 2)
        theory = np.var(y) * (1 - np.corrcoef(y, p)[0, 1] ** 2)
        assert abs(mse_min - theory) < 1e-8


def test_affine_preserves_ranking_metrics():
    """Affine (a>0) leaves Pearson and ranking unchanged but can change RMSE."""
    rng = np.random.default_rng(2)
    y = rng.normal(10, 3, 300)
    p = 0.5 * y + 4 + rng.normal(0, 1, 300)
    pc = AffineCalibrator().fit(p, y).transform(p)
    assert abs(pearson(y, p) - pearson(y, pc)) < 1e-9
    assert top_k_overlap(y, p, 0.1) == top_k_overlap(y, pc, 0.1)
    assert rmse(y, pc) <= rmse(y, p) + 1e-9


def test_metrics_perfect_prediction():
    y = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    m = all_metrics(y, y.copy())
    assert abs(m["pearson"] - 1) < 1e-9
    assert abs(m["overlap@20"] - 1) < 1e-9
    assert abs(m["ndcg@20"] - 1) < 1e-9
    assert abs(m["releff@20"] - 1) < 1e-9


def test_ndcg_bounds_and_random():
    rng = np.random.default_rng(3)
    y = rng.normal(size=200)
    assert abs(ndcg_at_k(y, y, 0.1) - 1) < 1e-9          # perfect ranking
    vals = [ndcg_at_k(y, rng.normal(size=200), 0.1) for _ in range(20)]
    assert all(0 <= v <= 1.0001 for v in vals)


def test_relative_efficiency_oracle_and_random():
    rng = np.random.default_rng(4)
    y = rng.normal(size=400)
    assert abs(relative_efficiency(y, y, 0.1) - 1) < 1e-9
    re = relative_efficiency(y, rng.normal(size=400), 0.1)
    assert re < 0.95                                      # random << oracle


def test_gblup_runs_and_predicts():
    rng = np.random.default_rng(5)
    n, p = 120, 300
    X = rng.integers(0, 3, size=(n, p)).astype(float)
    beta = rng.normal(size=p) * (rng.random(p) < 0.1)
    y = X @ beta + rng.normal(size=n) * 5
    tr, te = slice(0, 90), slice(90, n)
    g = GBLUP().fit(X[tr], y[tr])
    pred = g.predict(X[te])
    assert pred.shape == (30,)
    assert 0 <= g.h2_ <= 1


def test_loss_registry_and_ccc():
    for name in ("mse", "pearson", "hybrid", "ccc"):
        fn = losses.get_loss(name, **({"lam": 0.1} if name == "hybrid" else {}))
        v = fn(torch.randn(50), torch.randn(50))
        assert torch.isfinite(v)
