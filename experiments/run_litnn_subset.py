"""Subset validation for the literature-inspired NN benchmark: four adapted
architectures with paper-anchored hyper-parameters (no HPO), on a subset chosen to
span the difficulty spectrum (rice/pig-like low-r ... pine/maize high-r), under MSE
and the Pearson loss. Purpose: (1) confirm the reimplementations reproduce sane
accuracy, (2) validate the headroom mechanism at scale, (3) measure real wall-time
under current CPU contention before committing to the full S2 campaign.

CPU-polite: few workers, low thread count, nice'd by the launcher.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ccgp.config import FULL, GridConfig
from experiments.run import RESULTS, build_splits, make_dataset, traits_for, _nn_params

# Paper-anchored defaults for protocol-matched adaptations (no tuning). Structural
# and input changes in ccgp.models.neural mean these are not canonical clones.
CLONE_DEFAULTS = {
    "deepgs":  {"arch_kwargs": {"n_filters": 8, "kernel_size": 18, "fc_dim": 32, "dropout": 0.2},
                "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 64},
    "dnngp":   {"arch_kwargs": {"channels": (16, 16, 16), "kernel_size": 4, "fc_dim": 64, "dropout": 0.2},
                "pca": 0.95, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 128},
    "pnngs":   {"arch_kwargs": {"n_paths": 4, "channels": 8, "stem_stride": 8, "dropout": 0.5},
                "lr": 1e-3, "weight_decay": 0.1, "batch_size": 128},
    "soydngp": {"arch_kwargs": {"side": 96, "width": 32, "dropout": 0.3},
                "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 128},
}
SUBSET_KEYS = ["easygese:rice", "easygese:pine", "easygese:maize", "wheat:_", "easygese:lentil"]
SUB_LOSSES = ["mse", "pearson"]
MAX_TRAITS = 3
OUT = RESULTS / "results_litnn_subset.parquet"


def run_one(payload):
    cfg = GridConfig(**payload["cfg"])
    key, trait = payload["key"], payload["trait"]
    ds = make_dataset(key, cfg)
    splits = build_splits(ds, trait, cfg)
    from ccgp.experiment import run_cell
    rows = []
    for arch, base in CLONE_DEFAULTS.items():
        for loss in SUB_LOSSES:
            p = _nn_params(dict(base), loss, cfg)
            p.setdefault("max_epochs", 150)
            p.setdefault("patience", 15)
            rows += run_cell(ds, trait, arch, loss, splits, params=p, fracs=cfg.fracs,
                             cal_names=("raw", "affine"), base_meta={"exp": "litnn_subset"},
                             seed=cfg.seed)
    return pd.DataFrame(rows)


def _gpu_init(q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(q.get())
    os.environ["OMP_NUM_THREADS"] = "2"          # CPU-polite: cores are saturated by other jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")             # single GPU; CPUs are saturated by other jobs
    args = ap.parse_args()
    cfg = FULL
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "2"

    jobs = []
    for key in SUBSET_KEYS:
        ds = make_dataset(key, cfg)
        for t in traits_for(ds, cfg)[:MAX_TRAITS]:
            jobs.append((key, t))
    print(f"[litnn-subset] {len(jobs)} trait-datasets x 4 archs x 2 losses (serial, GPU {args.gpu})",
          flush=True)
    cfgd = asdict(cfg)

    # Serial, in-process: robust under CPU contention and avoids the spawn-pool
    # re-import fork cascade seen with ProcessPoolExecutor on this box.
    t0 = time.perf_counter()
    shards = []
    for i, (k, t) in enumerate(jobs):
        shards.append(run_one({"key": k, "trait": t, "cfg": cfgd}))
        print(f"[litnn-subset] {i+1}/{len(jobs)} {k}:{t} done "
              f"({time.perf_counter()-t0:.0f}s elapsed)", flush=True)
    df = pd.concat(shards, ignore_index=True)
    df.to_parquet(OUT)
    wall = time.perf_counter() - t0
    print(f"[litnn-subset] wrote {OUT} ({len(df)} rows) in {wall:.0f}s wall; "
          f"per trait-dataset {wall/len(jobs):.1f}s", flush=True)


if __name__ == "__main__":
    main()
