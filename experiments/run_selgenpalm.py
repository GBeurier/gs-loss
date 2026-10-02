#!/usr/bin/env python3
"""Leakage-safe SelGenPalm 2025 external-validation campaign.

The workflow is deliberately split into three commands:

``hpo``
    Tune once on the official DAPHNE development folds under neutral MSE.
``fit``
    Train every fold/seed/objective and write outcome-blind predictions to the
    private data drive. The external phenotype file is not loaded.
``evaluate``
    Refuse to run until every expected prediction exists, then reveal and score
    the locked external endpoint. Only aggregate metrics enter this repository.

Raw private data, cross identifiers, and individual predictions are never
written to the repository.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ccgp.calibration import PositiveAffineCalibrator
from ccgp.experiment import _make_model, _standardize
from ccgp.hpo import NN_ARCHS, sample_nn_params
from ccgp.metrics import all_metrics, nrmse, weighted_predictive_metrics
from ccgp.selgenpalm import (
    DEFAULT_2025_ROOT,
    DEFAULT_FOLDS_ROOT,
    load_daphne_folds,
    load_selgenpalm_development_2025,
    load_selgenpalm_external_features_2025,
    load_selgenpalm_external_outcomes_2025,
)

SUPPORTED_MODELS = ("gblup", "ridge", "mlp", "cnn", "transformer",
                    "deepgs", "dnngp", "pnngs", "soydngp")
# The four adapted literature architectures are already exhaustively benchmarked
# in the 101-task public campaign. Keep the private validation focused enough to
# finish while covering linear baselines and the three generic neural families.
DEFAULT_MODELS = ("gblup", "ridge", "mlp", "cnn", "transformer")
OBJECTIVES = {
    "mse": ("mse", False),
    "mse_weighted": ("mse", True),
    "pearson": ("pearson", False),
    "pearson_weighted": ("pearson", True),
}
CLASSICAL = {"gblup", "ridge"}


def _cross_hash(crosses: np.ndarray) -> str:
    payload = "\0".join(np.asarray(crosses, str).tolist()).encode()
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _candidate_params(model: str, n_trials: int, seed: int) -> list[dict]:
    if model == "gblup":
        return [{}]
    if model == "ridge":
        values = [1.0, 10.0, 100.0, 1000.0, 1e4]
        return [{"alpha": alpha} for alpha in values[:max(1, min(n_trials, len(values)))]]
    rng = np.random.default_rng(seed)
    return [sample_nn_params(model, rng) for _ in range(n_trials)]


def _fit_one(model_name: str, loss: str, params: dict, seed: int,
             X_train: np.ndarray, y_train: np.ndarray,
             X_validation: np.ndarray, y_validation: np.ndarray,
             train_weight: np.ndarray | None = None,
             validation_weight: np.ndarray | None = None):
    model, is_nn = _make_model(model_name, loss, params, seed)
    if is_nn:
        model.fit(X_train, y_train, X_validation, y_validation,
                  sample_weight=train_weight, val_weight=validation_weight)
    else:
        model.fit(X_train, y_train)
    return model


def run_hpo(models: list[str], hpo_path: Path, data_root: Path, folds_root: Path,
            n_trials: int, n_folds: int, seed: int) -> dict:
    """Shared-HPO ablation: tune each architecture under unweighted MSE on FFB."""
    data = load_selgenpalm_development_2025("ffb", data_root)
    folds = load_daphne_folds(data.crosses, folds_root)[:n_folds]
    cache = json.loads(hpo_path.read_text()) if hpo_path.exists() else {
        "_meta": {
            "dataset": "SelGenPalm 2025 private",
            "tuning_trait": "ffb",
            "objective": "unweighted MSE; selected by validation Pearson",
            "fold_source": str(folds_root),
            "n_hpo_folds": n_folds,
            "seed": seed,
            "external_outcomes_loaded": False,
            "reuse": "parameters frozen across traits and loss/weight objectives",
        },
        "params": {},
        "trials": {},
    }
    for model_name in models:
        if model_name in cache["params"]:
            print(f"[hpo] {model_name}: cached", flush=True)
            continue
        trials = []
        candidates = _candidate_params(model_name, n_trials, seed + 1009 * len(cache["params"]))
        for trial_index, params in enumerate(candidates):
            scores = []
            error = None
            try:
                for fold in folds:
                    Xtr, Xval = _standardize(data.X[fold.train], data.X[fold.validation])
                    fitted = _fit_one(
                        model_name, "mse", params, seed + trial_index + 100 * fold.fold,
                        Xtr, data.y[fold.train], Xval, data.y[fold.validation])
                    pred = fitted.predict(Xval)
                    if np.std(pred) < 1e-12:
                        scores.append(np.nan)
                    else:
                        scores.append(float(np.corrcoef(data.y[fold.validation], pred)[0, 1]))
                    del fitted
                    gc.collect()
            except Exception as exc:  # record failures; do not silently select them
                error = f"{type(exc).__name__}: {exc}"
            score = float(np.nanmean(scores)) if scores and np.isfinite(scores).any() else None
            trials.append({"trial": trial_index, "params": _jsonable(params),
                           "fold_pearson": scores, "mean_pearson": score, "error": error})
            display = f"{score:.4f}" if score is not None else "FAILED"
            print(f"[hpo] {model_name} {trial_index + 1}/{len(candidates)} r={display}", flush=True)
        valid = [trial for trial in trials if trial["mean_pearson"] is not None and
                 np.isfinite(trial["mean_pearson"])]
        if not valid:
            raise RuntimeError(f"all HPO trials failed for {model_name}")
        best = max(valid, key=lambda trial: trial["mean_pearson"])
        cache["params"][model_name] = best["params"]
        cache["trials"][model_name] = trials
        hpo_path.parent.mkdir(parents=True, exist_ok=True)
        hpo_path.write_text(json.dumps(cache, indent=2, allow_nan=True))
    return cache


def campaign_cells(traits: list[str], models: list[str], seeds: list[int], n_folds: int):
    cells = []
    for trait in traits:
        for model in models:
            objectives = ["na"] if model in CLASSICAL else list(OBJECTIVES)
            model_seeds = [0] if model in CLASSICAL else seeds
            for objective in objectives:
                for seed in model_seeds:
                    for fold in range(1, n_folds + 1):
                        cells.append((trait, model, objective, seed, fold))
    return cells


def _shard_path(private_root: Path, cell) -> Path:
    trait, model, objective, seed, fold = cell
    return private_root / "predictions" / (
        f"{trait}__{model}__{objective}__seed{seed}__fold{fold}.npz")


def fit_cell(cell, hpo: dict, data_root: Path, folds_root: Path,
             private_root: Path) -> Path:
    """Create one external prediction shard without loading external outcomes."""
    trait, model_name, objective, seed, fold_number = cell
    shard = _shard_path(private_root, cell)
    if shard.exists():
        print(f"[fit] cached {shard.name}", flush=True)
        return shard
    data = load_selgenpalm_development_2025(trait, data_root)
    external = load_selgenpalm_external_features_2025(data_root)
    folds = load_daphne_folds(data.crosses, folds_root)
    fold = folds[fold_number - 1]
    if data.X.shape[1] != external.X.shape[1]:
        raise ValueError("development and external marker counts differ")
    Xtr, Xval, Xext = _standardize(
        data.X[fold.train], data.X[fold.validation], external.X)
    params = hpo["params"].get(model_name)
    if params is None:
        raise KeyError(f"missing SelGenPalm HPO parameters for {model_name}")
    if objective == "na":
        loss, weighted = "na", False
    else:
        loss, weighted = OBJECTIVES[objective]
    train_weight = data.weight[fold.train] if weighted else None
    validation_weight = data.weight[fold.validation] if weighted else None
    started = time.perf_counter()
    fitted = _fit_one(
        model_name, loss, params, seed + 100 * fold_number,
        Xtr, data.y[fold.train], Xval, data.y[fold.validation],
        train_weight, validation_weight)
    validation_prediction = fitted.predict(Xval)
    external_prediction = fitted.predict(Xext)
    calibrator = PositiveAffineCalibrator().fit(
        validation_prediction, data.y[fold.validation],
        sample_weight=validation_weight)
    calibrated_prediction = calibrator.transform(external_prediction)
    metadata = {
        "trait": trait,
        "model": model_name,
        "objective": objective,
        "seed": seed,
        "fold": fold_number,
        "n_train": len(fold.train),
        "n_validation": len(fold.validation),
        "n_external": len(external.X),
        "p": data.X.shape[1],
        "train_seconds": time.perf_counter() - started,
        "best_epoch": getattr(fitted, "best_epoch_", None),
        "calibration_slope": calibrator.a_,
        "calibration_intercept": calibrator.b_,
        "external_cross_sha256": _cross_hash(external.crosses),
        "external_outcomes_loaded": False,
    }
    shard.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        raw=np.asarray(external_prediction, np.float32),
        affine_positive=np.asarray(calibrated_prediction, np.float32),
        metadata=json.dumps(_jsonable(metadata)),
    )
    os.replace(temporary, shard)
    print(f"[fit] {shard.name} ({metadata['train_seconds']:.1f}s)", flush=True)
    return shard


def run_fit(cells, hpo, data_root, folds_root, private_root, cell_index=None):
    selected = cells if cell_index is None else [cells[cell_index]]
    for index, cell in enumerate(selected, start=1):
        print(f"[fit] {index}/{len(selected)} {cell}", flush=True)
        fit_cell(cell, hpo, data_root, folds_root, private_root)
        gc.collect()


def _read_prediction(path: Path, calibration: str, expected_hash: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as shard:
        metadata = json.loads(str(shard["metadata"]))
        if metadata["external_cross_sha256"] != expected_hash:
            raise ValueError(f"external cross order differs in {path}")
        return np.asarray(shard[calibration], float)


def evaluate_locked_endpoint(cells, traits, models, seeds, n_folds,
                             data_root, private_root, output_path):
    """Score only after checking that the outcome-blind prediction grid is complete."""
    missing = [str(_shard_path(private_root, cell)) for cell in cells
               if not _shard_path(private_root, cell).exists()]
    if missing:
        raise RuntimeError(
            f"external outcomes remain locked: {len(missing)} prediction shards missing; "
            f"first is {missing[0]}")

    rows = []
    for trait in traits:
        outcomes = load_selgenpalm_external_outcomes_2025(trait, data_root)
        expected_hash = _cross_hash(outcomes.crosses)
        for model in models:
            objectives = ["na"] if model in CLASSICAL else list(OBJECTIVES)
            model_seeds = [0] if model in CLASSICAL else seeds
            for objective in objectives:
                for calibration in ("raw", "affine_positive"):
                    seed_predictions = []
                    for seed in model_seeds:
                        fold_predictions = [
                            _read_prediction(
                                _shard_path(private_root, (trait, model, objective, seed, fold)),
                                calibration, expected_hash)
                            for fold in range(1, n_folds + 1)
                        ]
                        prediction = np.mean(fold_predictions, axis=0)
                        seed_predictions.append(prediction)
                        metrics = all_metrics(outcomes.y, prediction)
                        metrics["nrmse"] = nrmse(outcomes.y, prediction)
                        metrics.update(weighted_predictive_metrics(
                            outcomes.y, prediction, outcomes.weight))
                        rows.append({
                            "dataset": "selgenpalm_2025_external",
                            "trait": trait,
                            "model": model,
                            "objective": objective,
                            "calibration": calibration,
                            "ensemble": "folds_within_seed",
                            "seed": seed,
                            "n_folds": n_folds,
                            "n_test": len(outcomes.y),
                            **metrics,
                        })
                    prediction = np.mean(seed_predictions, axis=0)
                    metrics = all_metrics(outcomes.y, prediction)
                    metrics["nrmse"] = nrmse(outcomes.y, prediction)
                    metrics.update(weighted_predictive_metrics(
                        outcomes.y, prediction, outcomes.weight))
                    rows.append({
                        "dataset": "selgenpalm_2025_external",
                        "trait": trait,
                        "model": model,
                        "objective": objective,
                        "calibration": calibration,
                        "ensemble": "all_folds_and_seeds",
                        "seed": -1,
                        "n_folds": n_folds,
                        "n_test": len(outcomes.y),
                        **metrics,
                    })
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    manifest = {
        "evaluated_at_unix": time.time(),
        "prediction_root": str(private_root),
        "aggregate_output": str(output_path),
        "n_prediction_shards": len(cells),
        "external_outcomes_loaded_after_completeness_check": True,
        "private_identifiers_written_to_repository": False,
    }
    (private_root / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[evaluate] {len(result)} aggregate rows -> {output_path}", flush=True)
    return result


def _csv_arg(value: str, cast=str):
    return [cast(item) for item in value.split(",") if item]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["hpo", "fit", "evaluate", "all"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_2025_ROOT)
    parser.add_argument("--folds-root", type=Path, default=DEFAULT_FOLDS_ROOT)
    parser.add_argument("--private-root", type=Path, default=Path(os.environ.get(
        "SELGENPALM_RESULTS_ROOT", "/mnt/d/oil_palm/gs_loss_results")))
    parser.add_argument("--hpo-path", type=Path, default=Path("results/hpo_selgenpalm.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/results_selgenpalm_external.parquet"))
    parser.add_argument("--traits", default="ffb,bn,bw")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--hpo-trials", type=int, default=16)
    parser.add_argument("--hpo-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cell-index", type=int, default=None,
                        help="Fit only one stable campaign cell (for guarded schedulers).")
    parser.add_argument("--list-cells", action="store_true")
    args = parser.parse_args()

    traits = _csv_arg(args.traits)
    models = _csv_arg(args.models)
    seeds = _csv_arg(args.seeds, int)
    unknown = sorted(set(models) - set(SUPPORTED_MODELS))
    if unknown:
        raise SystemExit(f"unsupported SelGenPalm models: {unknown}")
    if not 1 <= args.n_folds <= 5 or not 1 <= args.hpo_folds <= 5:
        raise SystemExit("n-folds and hpo-folds must be between 1 and 5")
    cells = campaign_cells(traits, models, seeds, args.n_folds)
    if args.list_cells:
        for index, cell in enumerate(cells):
            status = "done" if _shard_path(args.private_root, cell).exists() else "missing"
            print(index, status, *cell)
        return

    hpo = None
    if args.cmd in ("hpo", "all"):
        hpo = run_hpo(models, args.hpo_path, args.data_root, args.folds_root,
                      args.hpo_trials, args.hpo_folds, args.seed)
    if args.cmd in ("fit", "all"):
        if hpo is None:
            if not args.hpo_path.exists():
                raise SystemExit(f"missing HPO cache: {args.hpo_path}")
            hpo = json.loads(args.hpo_path.read_text())
        run_fit(cells, hpo, args.data_root, args.folds_root,
                args.private_root, args.cell_index)
    if args.cmd in ("evaluate", "all"):
        evaluate_locked_endpoint(cells, traits, models, seeds, args.n_folds,
                                 args.data_root, args.private_root, args.out)


if __name__ == "__main__":
    main()
