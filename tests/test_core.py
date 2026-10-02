"""Fast unit tests for the scientific core (run with `pytest -q`)."""
import numpy as np
import torch

from ccgp import losses
from ccgp.calibration import (AffineCalibrator, IsotonicCalibrator,
                              PositiveAffineCalibrator, RawCalibrator)
from ccgp.metrics import (all_metrics, ndcg_at_k, pearson, relative_efficiency,
                          rmse, top_k_overlap, weighted_pearson,
                          weighted_predictive_metrics)
from ccgp.models.classical import GBLUP
from ccgp.models.neural import NEURAL_ARCHS, NeuralRegressor
from ccgp.selgenpalm import aggregate_cross_phenotypes, load_daphne_folds


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


def test_weighted_pearson_identity_and_calibration():
    """Weighted standardized MSE and affine fit match weighted Pearson theory."""
    rng = np.random.default_rng(7)
    y = rng.normal(size=200)
    p = 1.7 * y + rng.normal(size=200)
    w = rng.uniform(0.1, 3.0, size=200)
    yt, pt, wt = map(torch.tensor, (y, p, w))
    r = weighted_pearson(y, p, w)
    assert abs(losses.pearson_loss(pt, yt, weight=wt).item() - (1 - r)) < 1e-8
    assert abs(losses.std_mse_loss(pt, yt, weight=wt).item() - (1 - r)) < 1e-8

    cal = AffineCalibrator().fit(p, y, sample_weight=w)
    pc = cal.transform(p)
    assert (weighted_predictive_metrics(y, pc, w)["weighted_rmse"] <=
            weighted_predictive_metrics(y, p, w)["weighted_rmse"])


def test_positive_affine_never_reverses_ranking():
    y = np.arange(20.0)
    p = -y
    cal = PositiveAffineCalibrator().fit(p, y)
    assert cal.a_ > 0
    assert np.array_equal(np.argsort(cal.transform(p)), np.argsort(p))


def test_selgenpalm_aggregation_is_cross_level_and_precision_weighted():
    import pandas as pd

    frame = pd.DataFrame({
        "Cross_Recod": ["a", "a", "a", "b", "b", "b"],
        "ffb_adj": [1.0, 2.0, 3.0, 10.0, 12.0, 14.0],
    })
    crosses, means, weights, counts = aggregate_cross_phenotypes(frame, "ffb_adj")
    assert crosses.tolist() == ["a", "b"]
    assert np.allclose(means, [2.0, 12.0])
    assert counts.tolist() == [3, 3]
    assert np.allclose(weights, [3.0, 0.75])  # 1 / (sample variance / n)


def test_daphne_folds_are_exhaustive_disjoint_and_ordered(tmp_path):
    import pandas as pd

    crosses = np.asarray(["c0", "c1", "c2", "c3", "c4"])
    for fold, cross in enumerate(crosses, start=1):
        pd.DataFrame({"cross": [cross]}).to_csv(
            tmp_path / f"kfold_{fold}_test.csv", index=False)
    folds = load_daphne_folds(crosses, tmp_path)
    assert [fold.fold for fold in folds] == [1, 2, 3, 4, 5]
    assert [fold.validation.tolist() for fold in folds] == [[0], [1], [2], [3], [4]]
    assert all(len(fold.train) == 4 for fold in folds)


def test_daphne_folds_reject_overlap(tmp_path):
    import pandas as pd
    import pytest

    crosses = np.asarray(["c0", "c1", "c2", "c3", "c4"])
    for fold, cross in enumerate(["c0", "c1", "c2", "c3", "c0"], start=1):
        pd.DataFrame({"cross": [cross]}).to_csv(
            tmp_path / f"kfold_{fold}_test.csv", index=False)
    with pytest.raises(ValueError, match="overlap"):
        load_daphne_folds(crosses, tmp_path)


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


def test_all_neural_architectures_fit_and_predict():
    """Every generic/literature architecture completes the shared training API."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(30, 64)).astype(np.float32)
    y = (X[:, :4].sum(axis=1) + rng.normal(size=len(X))).astype(np.float32)
    kwargs = {
        "mlp": {"hidden_dims": (8,), "dropout": 0.0},
        "cnn": {"channels": (2, 4), "kernel_size": 5, "first_stride": 2,
                "fc_dim": 4, "dropout": 0.0},
        "transformer": {"n_tokens": 8, "d_model": 8, "n_heads": 2,
                        "n_layers": 1, "dropout": 0.0},
        "deepgs": {"n_filters": 2, "kernel_size": 7, "fc_dim": 4, "dropout": 0.0},
        "dnngp": {"channels": (2, 2, 2), "kernel_size": 3, "fc_dim": 4,
                  "dropout": 0.0},
        "pnngs": {"n_paths": 2, "channels": 2, "stem_stride": 2, "dropout": 0.0},
        "soydngp": {"side": 16, "width": 2, "dropout": 0.0},
    }
    assert set(kwargs) == set(NEURAL_ARCHS)
    for arch in NEURAL_ARCHS:
        model = NeuralRegressor(
            arch=arch, loss="pearson", arch_kwargs=kwargs[arch], device="cpu",
            batch_size=8, max_epochs=2, patience=1, use_amp=False, seed=6,
        ).fit(X[:24], y[:24], X[24:], y[24:])
        pred = model.predict(X[24:])
        assert pred.shape == (6,)
        assert np.isfinite(pred).all()
