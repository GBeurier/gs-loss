"""Driver for the ccgp experimental suite (Experiments B-F).

Subcommands:
  hpo      tune & cache hyper-parameters
  main     headline grid: all (dataset, trait, model, loss, calibration) cells
  batch    batch-size / global-Pearson study (Exp E)
  splits   realistic-split study: random vs leave-family-out vs cross-environment (Exp F)
  all      hpo -> main -> batch -> splits

Parallelism: one worker per (GPU x stream); each worker is pinned to a GPU and
processes (dataset, trait) jobs. Datasets are built once and cached to disk so
worker reloads are cheap.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ccgp.config import CLASSICAL, FULL, SMOKE, GridConfig
from ccgp.data import CACHE, load_easygese, load_soynam, load_wheat
from ccgp.experiment import run_cell, run_cross_env
from ccgp.hpo import tune_dataset
from ccgp.splits import leave_family_out, predefined_folds, repeated_kfold

RESULTS = Path("results")
DS_CACHE = CACHE / "datasets"


# --- dataset registry --------------------------------------------------------

def unique_keys(cfg: GridConfig) -> list[str]:
    keys = [f"easygese:{s}" for s in cfg.easygese_species]
    keys.append("wheat:_")
    keys += [f"soynam:{t}" for t in cfg.soynam_traits]
    return keys


def _build_dataset(key: str, cfg: GridConfig):
    src, name = key.split(":", 1)
    if src == "easygese":
        return load_easygese(name, maf=cfg.maf, max_features=cfg.max_features)
    if src == "wheat":
        return load_wheat()
    if src == "soynam":
        return load_soynam(name)
    raise ValueError(key)


def make_dataset(key: str, cfg: GridConfig):
    DS_CACHE.mkdir(parents=True, exist_ok=True)
    p = DS_CACHE / f"{key.replace(':', '_')}_maf{cfg.maf}_mf{cfg.max_features}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    ds = _build_dataset(key, cfg)
    p.write_bytes(pickle.dumps(ds))
    return ds


def models_for(key: str, cfg: GridConfig) -> list[str]:
    src, name = key.split(":", 1)
    archs = list(cfg.nn_archs)
    species = name if src == "easygese" else src
    if species in cfg.transformer_species:
        archs = archs + ["transformer"]
    return list(cfg.classical_models) + archs


def traits_for(ds, cfg: GridConfig) -> list[str]:
    ts = ds.trait_names
    if cfg.max_traits_per_species is not None:
        ts = ts[: cfg.max_traits_per_species]
    return ts


def build_splits(ds, trait, cfg):
    if ds.folds is not None:
        _, _, ids, _ = ds.get_xy(trait)
        sp = predefined_folds(ds.folds, trait, ids)
        return [s for s in sp if s.repeat <= cfg.n_repeats]
    _, y, _, _ = ds.get_xy(trait)
    return repeated_kfold(len(y), 5, cfg.n_repeats, seed=cfg.seed)


def _nn_params(base, loss, cfg):
    p = dict(base)
    if loss == "hybrid":
        p["loss_kwargs"] = {"lam": cfg.hybrid_lam}
    return p


# --- worker job --------------------------------------------------------------

def run_job(args):
    key, trait, cfg_dict, hpo = args
    cfg = GridConfig(**cfg_dict)
    ds = make_dataset(key, cfg)
    params_map = hpo[key]["params"]
    splits = build_splits(ds, trait, cfg)
    rows = []
    for m in models_for(key, cfg):
        if m in CLASSICAL:
            rows += run_cell(ds, trait, m, "na", splits, params=params_map.get(m, {}),
                             fracs=cfg.fracs, cal_names=cfg.calibrators,
                             base_meta={"exp": "main"}, seed=cfg.seed)
        else:
            base = params_map.get(m, {})
            for loss in cfg.losses:
                rows += run_cell(ds, trait, m, loss, splits, params=_nn_params(base, loss, cfg),
                                 fracs=cfg.fracs, cal_names=cfg.calibrators,
                                 base_meta={"exp": "main"}, seed=cfg.seed)
    return pd.DataFrame(rows)


# --- parallel execution ------------------------------------------------------

def _gpu_init(q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(q.get())
    os.environ["OMP_NUM_THREADS"] = "4"


def execute(jobs, cfg, hpo, gpus, streams, serial=False):
    payloads = [(k, t, asdict(cfg), hpo) for (k, t) in jobs]
    shards = []
    if serial:
        for i, pl in enumerate(payloads):
            shards.append(run_job(pl))
            print(f"[run] {i+1}/{len(payloads)} {pl[0]}:{pl[1]} done", flush=True)
        return pd.concat(shards, ignore_index=True)

    n_workers = len(gpus) * streams
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for i in range(n_workers):
        q.put(gpus[i % len(gpus)])
    with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_gpu_init, initargs=(q,)) as ex:
        futs = {ex.submit(run_job, pl): (pl[0], pl[1]) for pl in payloads}
        for i, fut in enumerate(as_completed(futs)):
            k, t = futs[fut]
            shards.append(fut.result())
            print(f"[run] {i+1}/{len(futs)} {k}:{t} done", flush=True)
    return pd.concat(shards, ignore_index=True)


# --- HPO ---------------------------------------------------------------------

def run_hpo(cfg, path):
    path = Path(path)
    cache = json.loads(path.read_text()) if path.exists() else {}
    for key in unique_keys(cfg):
        if key in cache:
            continue
        ds = make_dataset(key, cfg)
        print(f"[hpo] {key}: {models_for(key, cfg)}", flush=True)
        cache[key] = tune_dataset(ds, models_for(key, cfg), cfg.hpo_trials, cfg.hpo_folds, cfg.seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1))
    return cache


# --- Exp E: batch size -------------------------------------------------------

def run_batch(cfg, hpo, gpus, streams, serial=False):
    species = [s for s in ["lentil", "soybean", "rice", "maize"] if s in cfg.easygese_species]
    batch_sizes = [32, 128, 512, None]            # None = full-batch (global Pearson)
    rows = []
    for sp in species:
        key = f"easygese:{sp}"
        ds = make_dataset(key, cfg)
        base = hpo[key]["params"].get("mlp", {})
        for trait in traits_for(ds, cfg)[:3]:
            splits = build_splits(ds, trait, cfg)
            for bs in batch_sizes:
                for loss in ["pearson", "mse"]:
                    p = dict(base); p["batch_size"] = bs
                    if loss == "hybrid":
                        p["loss_kwargs"] = {"lam": cfg.hybrid_lam}
                    r = run_cell(ds, trait, "mlp", loss, splits, params=p,
                                 fracs=cfg.fracs, cal_names=("raw", "affine"),
                                 base_meta={"exp": "batch", "batch_size": -1 if bs is None else bs},
                                 seed=cfg.seed)
                    rows += r
            print(f"[batch] {sp}:{trait} done", flush=True)
    return pd.DataFrame(rows)


# --- Exp F: realistic splits -------------------------------------------------

def _split_specs(cfg):
    """Full grid for the split study: classical baselines + every neural
    architecture under every core loss (so Delta-vs-MSE is defined for ALL
    architectures, not just the MLP)."""
    return ([("gblup", "na"), ("ridge", "na")]
            + [(a, l) for a in ("mlp", "cnn", "transformer") for l in cfg.losses])


def run_split_payload(payload):
    """Run one (dataset, trait/env-pair, scheme) split cell over the full grid."""
    cfg = GridConfig(**payload["cfg"])
    hpo = payload["hpo"]
    key = payload["key"]
    ds = make_dataset(key, cfg)
    pm = hpo[key]["params"]
    scheme = payload["scheme"]
    rows = []
    if payload["kind"] == "cell":
        trait = payload["trait"]
        X, y, ids, groups = ds.get_xy(trait)
        if scheme == "family":
            splits = leave_family_out(groups, min_test=5,
                                      max_families=cfg.soynam_max_families, seed=cfg.seed)
        else:                                       # random / within_env
            splits = repeated_kfold(len(y), 5, cfg.n_repeats, seed=cfg.seed)
        for m, loss in _split_specs(cfg):
            p = _nn_params(pm.get(m, {}), loss, cfg) if m not in CLASSICAL else pm.get(m, {})
            rows += run_cell(ds, trait, m, loss, splits, params=p, fracs=cfg.fracs,
                             cal_names=cfg.calibrators,
                             base_meta={"exp": "splits", "split_type": scheme}, seed=cfg.seed)
    else:                                           # cross-environment transfer
        ea, eb = payload["ea"], payload["eb"]
        splits = repeated_kfold(len(ds.traits[ea]), 5, cfg.n_repeats, seed=cfg.seed)
        for m, loss in _split_specs(cfg):
            p = _nn_params(pm.get(m, {}), loss, cfg) if m not in CLASSICAL else pm.get(m, {})
            rows += run_cross_env(ds, ea, eb, m, loss, splits, params=p,
                                  fracs=cfg.fracs, cal_names=cfg.calibrators, seed=cfg.seed)
    return pd.DataFrame(rows)


def run_splits(cfg, hpo, gpus, streams, serial=False, hpo_path=None):
    """Exp F across all architectures, parallelized over GPUs. SoyNAM random vs
    leave-family-out; wheat within- vs cross-environment."""
    from ccgp.hpo import _representative_trait, tune_model
    split_keys = [f"soynam:{t}" for t in cfg.soynam_traits] + ["wheat:_"]
    changed = False
    for key in split_keys:                          # ensure Transformer is tuned for these datasets
        if "transformer" not in hpo[key]["params"]:
            ds = make_dataset(key, cfg)
            trait = hpo[key].get("trait") or _representative_trait(ds)
            X, y, _, _ = ds.get_xy(trait)
            print(f"[splits-hpo] transformer {key}", flush=True)
            hpo[key]["params"]["transformer"] = tune_model(X, y, "transformer",
                                                           cfg.hpo_trials, cfg.hpo_folds, cfg.seed)
            changed = True
    if changed and hpo_path:
        Path(hpo_path).write_text(json.dumps(hpo, indent=1))

    cfgd = asdict(cfg)
    payloads = []
    for t in cfg.soynam_traits:
        for scheme in ("random", "family"):
            payloads.append({"kind": "cell", "key": f"soynam:{t}", "trait": t,
                             "scheme": scheme, "cfg": cfgd, "hpo": hpo})
    wk = "wheat:_"
    envs = list(make_dataset(wk, cfg).trait_names)
    for env in envs:
        payloads.append({"kind": "cell", "key": wk, "trait": env,
                         "scheme": "within_env", "cfg": cfgd, "hpo": hpo})
    for ea in envs:
        for eb in envs:
            if ea != eb:
                payloads.append({"kind": "crossenv", "key": wk, "ea": ea, "eb": eb,
                                 "scheme": "cross_env", "cfg": cfgd, "hpo": hpo})

    shards = []
    if serial:
        for i, pl in enumerate(payloads):
            shards.append(run_split_payload(pl))
            print(f"[splits] {i+1}/{len(payloads)} done", flush=True)
    else:
        n_workers = len(gpus) * streams
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        for i in range(n_workers):
            q.put(gpus[i % len(gpus)])
        with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_gpu_init, initargs=(q,)) as ex:
            futs = {ex.submit(run_split_payload, pl): pl for pl in payloads}
            for i, fut in enumerate(as_completed(futs)):
                shards.append(fut.result())
                print(f"[splits] {i+1}/{len(futs)} done", flush=True)
    return pd.concat(shards, ignore_index=True)


# --- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hpo", "main", "batch", "splits", "all"])
    ap.add_argument("--preset", default="full", choices=["full", "smoke"])
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = SMOKE if args.preset == "smoke" else FULL
    gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    RESULTS.mkdir(exist_ok=True)

    hpo = run_hpo(cfg, RESULTS / f"hpo_{cfg.name}.json")
    if args.cmd == "hpo":
        return

    if args.cmd in ("main", "all"):
        jobs = [(k, t) for k in unique_keys(cfg) for t in traits_for(make_dataset(k, cfg), cfg)]
        print(f"[main] {len(jobs)} (dataset,trait) jobs", flush=True)
        df = execute(jobs, cfg, hpo, gpus, args.streams, serial=args.serial)
        out = args.out or RESULTS / f"results_main_{cfg.name}.parquet"
        df.to_parquet(out)
        print(f"[main] wrote {out} ({len(df)} rows)")
    if args.cmd in ("batch", "all"):
        df = run_batch(cfg, hpo, gpus, args.streams, serial=args.serial)
        df.to_parquet(RESULTS / f"results_batch_{cfg.name}.parquet")
        print(f"[batch] wrote {len(df)} rows")
    if args.cmd in ("splits", "all"):
        df = run_splits(cfg, hpo, gpus, args.streams, serial=args.serial,
                        hpo_path=RESULTS / f"hpo_{cfg.name}.json")
        df.to_parquet(RESULTS / f"results_splits_{cfg.name}.parquet")
        print(f"[splits] wrote {len(df)} rows")


if __name__ == "__main__":
    main()
