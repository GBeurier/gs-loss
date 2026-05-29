"""Experiment A -- numerical verification that the equivalences are exact, not
merely symbolic.

Checks:
  (1) Proposition 1:  1 - r  ==  1/2 * MSE(z_y, z_p)  ==  pearson_loss
  (2) Proposition 2:  min_{a,b} MSE(a p + b, y)  ==  var(y) (1 - r^2)
  (3) Gradient check of the differentiable Pearson loss (autograd vs finite diff)
  (4) neg_pearson and pearson share gradients (differ by a constant)
  (5) The identity holds on real model predictions (GBLUP on a real fold)

Writes results/exp_a.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ccgp import losses
from ccgp.calibration import AffineCalibrator
from ccgp.data import load_easygese
from ccgp.experiment import _standardize
from ccgp.models.classical import GBLUP
from ccgp.splits import predefined_folds

RESULTS = Path("results")


def check_prop1(n_list=(50, 200, 1000, 5000), seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for n in n_list:
        for _ in range(20):
            y = rng.normal(size=n)
            p = rng.uniform(0.1, 2.0) * y + rng.normal(size=n) * rng.uniform(0.1, 3.0) + rng.normal()
            yt, pt = torch.tensor(y), torch.tensor(p)
            r = losses.pearson_corr(pt, yt).item()
            pl = losses.pearson_loss(pt, yt).item()
            sm = losses.std_mse_loss(pt, yt).item()
            diffs.append(max(abs(pl - (1 - r)), abs(sm - (1 - r)), abs(pl - sm)))
    return {"max_abs_diff": float(max(diffs)), "mean_abs_diff": float(np.mean(diffs)), "n_cases": len(diffs)}


def check_prop2(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(200):
        y = rng.normal(size=n)
        p = rng.uniform(0.2, 3.0) * y + rng.normal(size=n) * rng.uniform(0.1, 4.0) + rng.normal() * 5
        cal = AffineCalibrator().fit(p, y)
        mse_min = np.mean((cal.transform(p) - y) ** 2)
        r = np.corrcoef(y, p)[0, 1]
        theory = np.var(y) * (1 - r ** 2)
        diffs.append(abs(mse_min - theory))
    return {"max_abs_diff": float(max(diffs)), "mean_abs_diff": float(np.mean(diffs)), "n_cases": len(diffs)}


def check_gradient(n=64, seed=2):
    torch.manual_seed(seed)
    y = torch.randn(n, dtype=torch.double)
    p = torch.randn(n, dtype=torch.double, requires_grad=True)
    ok_pearson = torch.autograd.gradcheck(lambda x: losses.pearson_loss(x, y), (p,), eps=1e-6, atol=1e-4)
    ok_ccc = torch.autograd.gradcheck(lambda x: losses.ccc_loss(x, y), (p,), eps=1e-6, atol=1e-4)
    return {"pearson_gradcheck": bool(ok_pearson), "ccc_gradcheck": bool(ok_ccc)}


def check_neg_pearson_gradient(n=128, seed=3):
    torch.manual_seed(seed)
    y = torch.randn(n)
    p1 = torch.randn(n, requires_grad=True)
    p2 = p1.detach().clone().requires_grad_(True)
    losses.pearson_loss(p1, y).backward()
    losses.neg_pearson_loss(p2, y).backward()
    return {"max_grad_diff": float((p1.grad - p2.grad).abs().max().item())}


def check_real(seed=4):
    ds = load_easygese("lentil")
    trait = ds.trait_names[0]
    X, y, ids, _ = ds.get_xy(trait)
    sp = predefined_folds(ds.folds, trait, ids)[0]
    Xfit, Xtest = _standardize(X[sp.train], X[sp.test])
    g = GBLUP().fit(Xfit, y[sp.train])
    pred = g.predict(Xtest)
    yt, pt = torch.tensor(y[sp.test]), torch.tensor(pred)
    r = losses.pearson_corr(pt, yt).item()
    return {"dataset": "lentil", "trait": trait, "r": r,
            "abs_diff_identity": float(abs(losses.pearson_loss(pt, yt).item() - (1 - r)))}


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {
        "proposition1": check_prop1(),
        "proposition2": check_prop2(),
        "gradient": check_gradient(),
        "neg_pearson_gradient": check_neg_pearson_gradient(),
        "real_predictions": check_real(),
    }
    (RESULTS / "exp_a.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
