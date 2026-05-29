"""Fixed-budget hyper-parameter search.

To keep the headline comparison fair ("only the loss changes"), the
architecture and optimizer settings are tuned **once per (dataset, model)** with
a neutral MSE objective, selected by validation Pearson, then frozen and reused
across every loss. Classical models tune their regularization the same way.
The test folds are never used for selection. Results are cached to JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .experiment import _make_model, _standardize
from .metrics import pearson
from .models.classical import GBLUP  # noqa: F401  (documents the no-tuning case)

NN_ARCHS = ["mlp", "cnn", "transformer"]

NN_SPACES = {
    "mlp": {"hidden_dims": [(64,), (128, 32), (256, 64), (256, 128, 32), (512, 128)],
            "dropout": [0.1, 0.2, 0.3, 0.5]},
    "cnn": {"channels": [(8, 16), (16, 32), (32, 64)], "kernel_size": [7, 11, 21],
            "first_stride": [4, 8, 16], "fc_dim": [32, 64, 128], "dropout": [0.1, 0.2, 0.3]},
    "transformer": {"n_tokens": [32, 64, 128], "d_model": [32, 64], "n_heads": [2, 4],
                    "n_layers": [1, 2], "dropout": [0.1, 0.2]},
}
COMMON = {"lr": [3e-3, 1e-3, 3e-4], "weight_decay": [1e-3, 1e-4, 1e-5], "batch_size": [64, 128, 256]}
CLS_SPACES = {
    "gblup": {},
    "ridge": {"alpha": [1.0, 10.0, 100.0, 1000.0, 1e4]},
    "elasticnet": {"alpha": [1e-3, 1e-2, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
    "rf": {"n_estimators": [300], "max_depth": [None, 8, 16], "max_features": ["sqrt", 0.1]},
    "xgboost": {"max_depth": [3, 4, 6], "learning_rate": [0.03, 0.05, 0.1],
                "n_estimators": [300, 500], "subsample": [0.7, 0.9], "colsample_bytree": [0.3, 0.5]},
}


def _choice(rng, values):
    v = values[rng.integers(len(values))]
    return v.item() if isinstance(v, np.generic) else v


def _sample_nn(arch, rng):
    ak = {k: _choice(rng, v) for k, v in NN_SPACES[arch].items()}
    p = {"arch_kwargs": ak, "max_epochs": 150, "patience": 15}
    for k, v in COMMON.items():
        p[k] = _choice(rng, v)
    return p


def _eval_config(X, y, name, params, n_folds, seed, is_nn):
    rng = np.random.default_rng(seed)
    scores = []
    for f in range(n_folds):
        idx = rng.permutation(len(X))
        nv = max(10, int(0.2 * len(X)))
        val, tr = idx[:nv], idx[nv:]
        Xtr, Xval = _standardize(X[tr], X[val])
        model, _ = _make_model(name, "mse", params, seed=1000 + f)
        if is_nn:
            model.fit(Xtr, y[tr], Xval, y[val])
        else:
            model.fit(Xtr, y[tr])
        scores.append(pearson(y[val], model.predict(Xval)))
    return float(np.nanmean(scores))


def tune_model(X, y, name, n_trials=12, n_folds=3, seed=0):
    is_nn = name in NN_ARCHS
    rng = np.random.default_rng(seed)
    if not is_nn:
        space = CLS_SPACES[name]
        if not space:                       # GBLUP: no tuning
            return {}
        keys = list(space)
        combos = []
        # bounded random sampling of the discrete grid
        for _ in range(n_trials):
            combos.append({k: _choice(rng, space[k]) for k in keys})
        candidates = combos
    else:
        candidates = [_sample_nn(name, rng) for _ in range(n_trials)]

    best, best_score = candidates[0], -np.inf
    for params in candidates:
        try:
            s = _eval_config(X, y, name, params, n_folds, seed, is_nn)
        except Exception:
            s = -np.inf
        if s > best_score:
            best, best_score = params, s
    return best


def _representative_trait(dataset):
    counts = {t: int(np.sum(~np.isnan(y))) for t, y in dataset.traits.items()}
    return max(counts, key=counts.get)


def tune_dataset(dataset, models, n_trials=12, n_folds=3, seed=0):
    trait = _representative_trait(dataset)
    X, y, _, _ = dataset.get_xy(trait)
    out = {}
    for m in models:
        out[m] = tune_model(X, y, m, n_trials=n_trials, n_folds=n_folds, seed=seed)
    return {"trait": trait, "params": out}


def load_or_tune(datasets, models, cfg, path: Path, force=False):
    """Tune (or load cached) hyper-parameters for each dataset. ``datasets`` is a
    dict name -> GenomicDataset."""
    path = Path(path)
    cache = json.loads(path.read_text()) if (path.exists() and not force) else {}
    changed = False
    for name, ds in datasets.items():
        if name in cache:
            continue
        print(f"[hpo] tuning {name} ...", flush=True)
        cache[name] = tune_dataset(ds, models, cfg.hpo_trials, cfg.hpo_folds, cfg.seed)
        changed = True
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1))
    if changed:
        path.write_text(json.dumps(cache, indent=1))
    return cache
