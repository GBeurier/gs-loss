"""Per-cell experiment logic: fit one (model, loss) under a CV scheme and score
raw / affine / isotonic calibrated predictions with every metric.

A "cell" = (dataset, trait, model, loss). For each fold the training data is
split into a fit set and a validation set; markers are standardized with fit
statistics only; the model is trained on the fit set (neural nets early-stop on
the validation set); calibrators are fit on validation predictions and applied
to the test set. The test set is never used for fitting, early stopping,
calibration or hyper-parameter choice.
"""
from __future__ import annotations

import time

import numpy as np

from .calibration import get_calibrator
from .metrics import DEFAULT_FRACS, all_metrics
from .models import make_classical
from .models.neural import NEURAL_ARCHS, NeuralRegressor
from .splits import Split, inner_split

DEFAULT_CALIBRATORS = ("raw", "affine", "isotonic")


def _make_model(model_name: str, loss: str, params: dict, seed: int):
    if model_name in NEURAL_ARCHS:
        return NeuralRegressor(arch=model_name, loss=loss, seed=seed, **params), True
    return make_classical(model_name, random_state=seed, **params), False


def _standardize(X_fit: np.ndarray, *others: np.ndarray):
    mu = X_fit.mean(axis=0)
    sd = X_fit.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return tuple((X - mu) / sd for X in (X_fit, *others))


def run_fold(X, y, split: Split, groups, model_name, loss, params,
             fracs=DEFAULT_FRACS, cal_names=DEFAULT_CALIBRATORS,
             seed=0, val_frac=0.2):
    fit_idx, val_idx = inner_split(split.train, groups, val_frac,
                                   seed=10_000 * seed + 100 * split.repeat + split.fold)
    Xfit, Xval, Xtest = _standardize(X[fit_idx], X[val_idx], X[split.test])
    yfit, yval, ytest = y[fit_idx], y[val_idx], y[split.test]

    model, is_nn = _make_model(model_name, loss, params, seed)
    t0 = time.perf_counter()
    if is_nn:
        model.fit(Xfit, yfit, Xval, yval)
    else:
        model.fit(Xfit, yfit)
    ttime = getattr(model, "train_time_", None) or (time.perf_counter() - t0)

    p_val = model.predict(Xval)
    p_test = model.predict(Xtest)
    rows = []
    for cal in cal_names:
        c = get_calibrator(cal).fit(p_val, yval)
        m = all_metrics(ytest, c.transform(p_test), fracs)
        rows.append({"calibration": cal, **m})
    return rows, float(ttime), getattr(model, "best_epoch_", None)


def run_cell(dataset, trait, model_name, loss, splits, params=None,
             fracs=DEFAULT_FRACS, cal_names=DEFAULT_CALIBRATORS,
             base_meta=None, seed=0, val_frac=0.2):
    params = params or {}
    base_meta = base_meta or {}
    X, y, ids, groups = dataset.get_xy(trait)
    rows = []
    for sp in splits:
        fold_rows, ttime, bep = run_fold(
            X, y, sp, groups, model_name, loss, params, fracs, cal_names, seed, val_frac)
        for r in fold_rows:
            r.update({
                "dataset": dataset.name, "species": dataset.species, "trait": trait,
                "model": model_name, "loss": loss, "scheme": sp.scheme,
                "repeat": sp.repeat, "fold": sp.fold, "label": sp.label,
                "n_train": int(len(sp.train)), "n_test": int(len(sp.test)),
                "p": int(X.shape[1]), "train_time": ttime, "best_epoch": bep,
                "seed": seed, **base_meta,
            })
            rows.append(r)
    return rows


def run_cross_env(dataset, env_train, env_test, model_name, loss, splits,
                  params=None, fracs=DEFAULT_FRACS, cal_names=DEFAULT_CALIBRATORS,
                  seed=0, val_frac=0.2):
    """Cross-environment transfer (wheat): train on ``env_train`` phenotype, score
    held-out lines against their ``env_test`` phenotype."""
    params = params or {}
    Xa, ya, _, _ = dataset.get_xy(env_train)
    yb = dataset.traits[env_test]                 # same line order as Xa (no NaN in wheat)
    rows = []
    for sp in splits:
        fit_idx, val_idx = inner_split(sp.train, None, val_frac,
                                       seed=10_000 * seed + 100 * sp.repeat + sp.fold)
        Xfit, Xval, Xtest = _standardize(Xa[fit_idx], Xa[val_idx], Xa[sp.test])
        yfit, yval = ya[fit_idx], ya[val_idx]
        ytest = yb[sp.test]                        # evaluate in the OTHER environment
        model, is_nn = _make_model(model_name, loss, params, seed)
        if is_nn:
            model.fit(Xfit, yfit, Xval, yval)
        else:
            model.fit(Xfit, yfit)
        p_val, p_test = model.predict(Xval), model.predict(Xtest)
        for cal in cal_names:
            c = get_calibrator(cal).fit(p_val, yval)
            m = all_metrics(ytest, c.transform(p_test), fracs)
            rows.append({
                "calibration": cal, "dataset": dataset.name, "species": dataset.species,
                "trait": f"{env_train}->{env_test}", "model": model_name, "loss": loss,
                "scheme": "cross_env", "repeat": sp.repeat, "fold": sp.fold,
                "n_train": int(len(sp.train)), "n_test": int(len(sp.test)),
                "p": int(Xa.shape[1]), "seed": seed, **m,
            })
    return rows
