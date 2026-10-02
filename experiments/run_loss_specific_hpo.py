#!/usr/bin/env python3
"""Nested loss-specific HPO sensitivity analysis on CIMMYT wheat.

The 101-task campaign freezes one MSE-tuned configuration across losses, which
is the controlled ablation needed to identify the effect of changing the loss.
This complementary experiment asks a different practical question: what happens
when MSE and Pearson each receive their own tuning? To avoid optimistic HPO bias,
every configuration is selected using only the outer-training portion of one of
five wheat folds, then evaluated on that fold. The same sampled configurations
and budget are used for both losses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ccgp.config import CAMPAIGN
from ccgp.experiment import _make_model, _standardize, run_fold
from ccgp.hpo import sample_nn_params
from ccgp.metrics import pearson
from experiments.run import build_splits, make_dataset

DATASET_KEY = "wheat:_"
DEFAULT_MODELS = ("mlp", "cnn", "transformer")
LOSSES = ("mse", "pearson")
RESULTS = Path("results")
HPO_PATH = RESULTS / "hpo_loss_specific_wheat.json"
SHARD_DIR = RESULTS / "loss_specific_wheat_shards"


def hpo_cells(models: list[str], n_folds: int):
    return [(model, loss, fold) for model in models for loss in LOSSES
            for fold in range(1, n_folds + 1)]


def _hpo_key(cell) -> str:
    model, loss, fold = cell
    return f"{model}__{loss}__fold{fold}"


def _result_shard(cell) -> Path:
    return SHARD_DIR / f"{_hpo_key(cell)}.parquet"


def _candidate_bank(model: str, n_trials: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed + sum(model.encode()))
    return [sample_nn_params(model, rng) for _ in range(n_trials)]


def _score_candidate(X, y, outer_train, model_name, loss, params,
                     n_inner_folds, seed, candidate_index):
    local = np.asarray(outer_train, int)
    splitter = KFold(n_splits=n_inner_folds, shuffle=True,
                     random_state=seed + 1000 * candidate_index)
    scores = []
    for inner_fold, (tr_local, val_local) in enumerate(splitter.split(local), start=1):
        train = local[tr_local]
        validation = local[val_local]
        Xtr, Xval = _standardize(X[train], X[validation])
        model, _ = _make_model(
            model_name, loss, params,
            seed=seed + 10_000 * candidate_index + 100 * inner_fold)
        model.fit(Xtr, y[train], Xval, y[validation])
        scores.append(pearson(y[validation], model.predict(Xval)))
    return float(np.nanmean(scores))


def tune_cell(cell, cache, X, y, outer_splits, n_trials, n_inner_folds, seed):
    key = _hpo_key(cell)
    if key in cache["selected"]:
        print(f"[nested-hpo] {key}: cached", flush=True)
        return
    model_name, loss, fold_number = cell
    outer = outer_splits[fold_number - 1]
    candidates = _candidate_bank(model_name, n_trials, seed)
    trials = []
    for index, params in enumerate(candidates):
        error = None
        try:
            score = _score_candidate(
                X, y, outer.train, model_name, loss, params,
                n_inner_folds, seed + 100_000 * fold_number, index)
        except Exception as exc:
            score = None
            error = f"{type(exc).__name__}: {exc}"
        trials.append({"trial": index, "params": params,
                       "mean_inner_pearson": score, "error": error})
        display = "FAILED" if score is None else f"{score:.4f}"
        print(f"[nested-hpo] {key} {index + 1}/{n_trials} r={display}", flush=True)
    valid = [trial for trial in trials if trial["mean_inner_pearson"] is not None and
             np.isfinite(trial["mean_inner_pearson"])]
    if not valid:
        raise RuntimeError(f"all nested HPO trials failed for {key}")
    best = max(valid, key=lambda trial: trial["mean_inner_pearson"])
    cache["selected"][key] = best["params"]
    cache["trials"][key] = trials
    HPO_PATH.parent.mkdir(parents=True, exist_ok=True)
    HPO_PATH.write_text(json.dumps(cache, indent=2))


def load_cache(n_trials, n_inner_folds, n_outer_folds, seed):
    if HPO_PATH.exists():
        return json.loads(HPO_PATH.read_text())
    return {
        "_meta": {
            "dataset": "CIMMYT wheat",
            "mode": "loss-specific nested HPO sensitivity",
            "training_losses": list(LOSSES),
            "selection_metric": "validation Pearson",
            "same_candidate_bank_across_losses": True,
            "n_trials_per_loss": n_trials,
            "n_inner_folds": n_inner_folds,
            "n_outer_folds": n_outer_folds,
            "seed": seed,
            "external_test_used_for_hpo": False,
        },
        "selected": {},
        "trials": {},
    }


def run_result_cell(cell, cache, dataset, outer_splits, seeds):
    shard = _result_shard(cell)
    if shard.exists():
        print(f"[nested-main] {shard.name}: cached", flush=True)
        return
    model_name, loss, fold_number = cell
    params = cache["selected"][_hpo_key(cell)]
    split = outer_splits[fold_number - 1]
    rows = []
    for trait in dataset.trait_names:
        X, y, _, groups = dataset.get_xy(trait)
        for seed in seeds:
            fold_rows, train_time, best_epoch = run_fold(
                X, y, split, groups, model_name, loss, params,
                cal_names=("raw", "affine", "isotonic"), seed=seed)
            for row in fold_rows:
                row.update({
                    "dataset": dataset.name,
                    "species": dataset.species,
                    "trait": trait,
                    "model": model_name,
                    "loss": loss,
                    "scheme": "nested_loss_specific_hpo",
                    "repeat": 1,
                    "fold": fold_number,
                    "n_train": len(split.train),
                    "n_test": len(split.test),
                    "p": X.shape[1],
                    "seed": seed,
                    "train_time": train_time,
                    "best_epoch": best_epoch,
                    "hpo_mode": "loss_specific_nested",
                })
                rows.append(row)
    shard.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(shard, index=False)
    print(f"[nested-main] {shard.name}: {len(rows)} rows", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["hpo", "main", "assemble", "all"])
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--cell-index", type=int, default=None)
    parser.add_argument("--list-cells", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=RESULTS / "results_loss_specific_wheat.parquet")
    args = parser.parse_args()

    models = [item for item in args.models.split(",") if item]
    seeds = [int(item) for item in args.seeds.split(",") if item]
    cells = hpo_cells(models, args.outer_folds)
    if args.list_cells:
        for index, cell in enumerate(cells):
            hpo_done = HPO_PATH.exists() and _hpo_key(cell) in json.loads(
                HPO_PATH.read_text()).get("selected", {})
            result_done = _result_shard(cell).exists()
            print(index, "hpo_done" if hpo_done else "hpo_missing",
                  "main_done" if result_done else "main_missing", *cell)
        return

    dataset = make_dataset(DATASET_KEY, CAMPAIGN)
    representative_trait = dataset.trait_names[0]
    outer_splits = build_splits(dataset, representative_trait, CAMPAIGN)[:args.outer_folds]
    X, y, _, _ = dataset.get_xy(representative_trait)
    cache = load_cache(args.n_trials, args.inner_folds, args.outer_folds, args.seed)
    selected_cells = cells if args.cell_index is None else [cells[args.cell_index]]

    if args.cmd in ("hpo", "all"):
        for cell in selected_cells:
            tune_cell(cell, cache, X, y, outer_splits, args.n_trials,
                      args.inner_folds, args.seed)
    if args.cmd in ("main", "all"):
        for cell in selected_cells:
            if _hpo_key(cell) not in cache["selected"]:
                raise SystemExit(f"missing nested HPO for {_hpo_key(cell)}")
            run_result_cell(cell, cache, dataset, outer_splits, seeds)
    if args.cmd in ("assemble", "all"):
        missing = [_result_shard(cell) for cell in cells if not _result_shard(cell).exists()]
        if missing:
            raise SystemExit(f"cannot assemble: {len(missing)} shards missing")
        frame = pd.concat([pd.read_parquet(_result_shard(cell)) for cell in cells],
                          ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.out, index=False)
        print(f"[assemble] {len(frame)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
