"""End-to-end smoke test of the ccgp pipeline (fast)."""
import time

import numpy as np
import pandas as pd
import torch

from ccgp import losses
from ccgp.data import load_easygese
from ccgp.experiment import run_cell
from ccgp.splits import predefined_folds

# 1. Proposition 1: 1 - r == 1/2 MSE(z_y, z_p) == pearson_loss
torch.manual_seed(0)
y = torch.randn(300)
p = 0.7 * y + 0.5 * torch.randn(300)
r = losses.pearson_corr(p, y).item()
pl = losses.pearson_loss(p, y).item()
sm = losses.std_mse_loss(p, y).item()
print(f"[identity] r={r:.6f}  1-r={1-r:.6f}  pearson_loss={pl:.6f}  std_mse={sm:.6f}")
assert abs(pl - sm) < 1e-5 and abs(pl - (1 - r)) < 1e-5, "Proposition 1 violated"
print("[identity] OK\n")

# 2. End-to-end on lentil
ds = load_easygese("lentil")
print(f"[data] lentil X={ds.X.shape} traits={ds.trait_names}")
print(f"[data] meta={ds.meta}")
trait = ds.trait_names[0]
X, y, ids, _ = ds.get_xy(trait)
sp = predefined_folds(ds.folds, trait, ids)
print(f"[splits] {len(sp)} predefined folds; first: train={len(sp[0].train)} test={len(sp[0].test)}\n")

sub = sp[:2]
for model, loss in [("gblup", "na"), ("ridge", "na"), ("mlp", "mse"), ("mlp", "pearson"),
                    ("cnn", "pearson")]:
    if model in ("gblup", "ridge"):
        params = {} if model == "gblup" else {"alpha": 100.0}
    else:
        params = dict(max_epochs=80, patience=12, batch_size=128,
                      arch_kwargs=dict(hidden_dims=(128, 32)) if model == "mlp" else {})
    t0 = time.perf_counter()
    rows = run_cell(ds, trait, model, loss, sub, params=params)
    df = pd.DataFrame(rows)
    g = df.groupby("calibration")[["pearson", "rmse", "ndcg@10", "overlap@10", "slope"]].mean()
    print(f"=== {model}/{loss}  ({time.perf_counter()-t0:.1f}s, train_time~{df['train_time'].mean():.2f}s) ===")
    print(g.round(4).to_string())
    # calibration must preserve Pearson (rank) but change RMSE/slope
    assert abs(g.loc['raw', 'pearson'] - g.loc['affine', 'pearson']) < 1e-6, "affine changed Pearson!"
    print()
print("[smoke] all checks passed")
