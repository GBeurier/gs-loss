"""Palier-1 surgical fill: run the light Transformer on the trait-datasets it was
not run on in the main grid (52 of 101), so the loss comparison covers all 101
trait-datasets for every neural architecture.

It reuses the exact main-grid code paths (build_splits, run_cell, the FULL config,
cached/auto-tuned HPO) and appends rows schema-identical to
results_main_full.parquet. Existing rows are never recomputed -- the merge in
analysis stays deterministic.

Usage:
  python experiments/fill_transformer.py --gpus 0,1 --streams 2
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ccgp.config import FULL, GridConfig
from ccgp.hpo import tune_model
from experiments.run import RESULTS, build_splits, make_dataset, _nn_params

MAIN = RESULTS / "results_main_full.parquet"
HPO_PATH = RESULTS / "hpo_full.json"
OUT = RESULTS / "results_main_transformer_fill.parquet"


def missing_cells() -> list[tuple[str, str]]:
    """(dataset, trait) cells where CNN ran but Transformer did not."""
    m = pd.read_parquet(MAIN, columns=["dataset", "trait", "model"])
    cnn = set(map(tuple, m[m.model == "cnn"][["dataset", "trait"]].drop_duplicates().values))
    tf = set(map(tuple, m[m.model == "transformer"][["dataset", "trait"]].drop_duplicates().values))
    return sorted(cnn - tf)


def dataset_key(ds_name: str) -> str:
    """Map the stored dataset name back to the loader key used by make_dataset."""
    if ds_name.startswith("easygese:") or ds_name == "wheat":
        return "wheat:_" if ds_name == "wheat" else ds_name
    if ds_name.startswith("soynam:"):
        return ds_name
    raise ValueError(ds_name)


def ensure_transformer_hpo(cfg: GridConfig) -> dict:
    """Tune the Transformer for any dataset that lacks cached params, then persist."""
    hpo = json.loads(HPO_PATH.read_text())
    keys = sorted({dataset_key(d) for d, _ in missing_cells()})
    changed = False
    for key in keys:
        params = hpo.setdefault(key, {"params": {}})["params"]
        if "transformer" in params:
            continue
        ds = make_dataset(key, cfg)
        trait = hpo[key].get("trait")
        if trait is None or trait not in ds.traits:
            from ccgp.hpo import _representative_trait
            trait = _representative_trait(ds)
            hpo[key]["trait"] = trait
        X, y, _, _ = ds.get_xy(trait)
        print(f"[fill-hpo] tuning transformer for {key} (trait={trait})", flush=True)
        params["transformer"] = tune_model(X, y, "transformer", cfg.hpo_trials, cfg.hpo_folds, cfg.seed)
        changed = True
    if changed:
        HPO_PATH.write_text(json.dumps(hpo, indent=1))
    return hpo


def run_cell_job(payload):
    cfg = GridConfig(**payload["cfg"])
    hpo = payload["hpo"]
    key, trait = payload["key"], payload["trait"]
    ds = make_dataset(key, cfg)
    base = hpo[key]["params"]["transformer"]
    splits = build_splits(ds, trait, cfg)
    rows = []
    for loss in cfg.losses:
        rows += run_cell_import(ds, trait, "transformer", loss, splits,
                                params=_nn_params(base, loss, cfg),
                                fracs=cfg.fracs, cal_names=cfg.calibrators,
                                base_meta={"exp": "main"}, seed=cfg.seed)
    return pd.DataFrame(rows)


def run_cell_import(*a, **k):
    from ccgp.experiment import run_cell
    return run_cell(*a, **k)


def _gpu_init(q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(q.get())
    os.environ["OMP_NUM_THREADS"] = "4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--serial", action="store_true")
    args = ap.parse_args()

    cfg = FULL
    gpus = [int(g) for g in args.gpus.split(",") if g != ""]

    cells = missing_cells()
    print(f"[fill] {len(cells)} missing transformer trait-datasets", flush=True)
    hpo = ensure_transformer_hpo(cfg)

    cfgd = asdict(cfg)
    payloads = [{"key": dataset_key(d), "trait": t, "cfg": cfgd, "hpo": hpo} for d, t in cells]

    shards = []
    if args.serial:
        for i, pl in enumerate(payloads):
            shards.append(run_cell_job(pl))
            print(f"[fill] {i+1}/{len(payloads)} {pl['key']}:{pl['trait']} done", flush=True)
    else:
        n_workers = len(gpus) * args.streams
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        for i in range(n_workers):
            q.put(gpus[i % len(gpus)])
        with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_gpu_init, initargs=(q,)) as ex:
            futs = {ex.submit(run_cell_job, pl): pl for pl in payloads}
            for i, fut in enumerate(as_completed(futs)):
                pl = futs[fut]
                shards.append(fut.result())
                print(f"[fill] {i+1}/{len(futs)} {pl['key']}:{pl['trait']} done", flush=True)

    df = pd.concat(shards, ignore_index=True)
    df.to_parquet(OUT)
    print(f"[fill] wrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
