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

import sys

# Make the package importable no matter how this script is launched (python
# experiments/run.py, nohup, nice, mp-spawn workers) without relying on
# PYTHONPATH: put the repo root on sys.path before importing ccgp.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ccgp.config import CAMPAIGN, CLASSICAL, FULL, SMOKE, GridConfig
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
    out = list(cfg.classical_models) + archs
    seen, deduped = set(), []                      # CAMPAIGN puts transformer in nn_archs AND
    for m in out:                                  # in transformer_species -> drop the duplicate
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


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

def _omp_threads(n_workers):
    return max(1, (os.cpu_count() or 1) // max(1, n_workers))


def _gpu_init(q, omp_threads):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(q.get())
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)


def _shard_path(cfg, key, trait):
    safe = f"{key}__{trait}".replace(":", "_").replace("/", "_").replace(" ", "_").replace("*", "x")
    return RESULTS / f"main_{cfg.name}_shards" / f"{safe}.parquet"


def validate_main_shards(cfg: GridConfig, jobs):
    """Read and validate every expected main-campaign shard before assembly.

    A partially written Parquet file can otherwise look like a completed cell,
    and concatenating whatever happens to be present produces a deceptively
    plausible partial result table.  Assembly is therefore intentionally
    all-or-nothing: every expected (dataset, trait) shard must be readable,
    contain the full 900 rows of the campaign grid, and have unique result
    keys.  This is a reproducibility guard, not an analysis filter.
    """
    frames = []
    failures = []
    key_columns = ["dataset", "trait", "model", "loss", "scheme", "repeat",
                   "fold", "calibration", "seed"]
    for key, trait in jobs:
        path = _shard_path(cfg, key, trait)
        if not path.exists():
            failures.append(f"missing {path}")
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            failures.append(f"unreadable {path}: {type(exc).__name__}: {exc}")
            continue
        if len(frame) != 900:
            failures.append(f"wrong row count {path}: {len(frame)} (expected 900)")
            continue
        missing_columns = sorted(set(key_columns) - set(frame.columns))
        if missing_columns:
            failures.append(f"missing columns {path}: {missing_columns}")
            continue
        if frame.duplicated(key_columns).any():
            failures.append(f"duplicate experimental key in {path}")
            continue
        frames.append(frame)
    if failures:
        preview = "\n  ".join(failures[:10])
        raise SystemExit(
            f"cannot assemble campaign: {len(failures)} shard integrity failure(s)\n  {preview}")
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 900 * len(jobs):
        raise SystemExit(
            f"cannot assemble campaign: {len(result)} rows (expected {900 * len(jobs)})")
    if result.duplicated(key_columns).any():
        raise SystemExit("cannot assemble campaign: duplicate experimental key across shards")
    return result


def _run_job_sharded(args):
    """Run one (dataset,trait) cell and persist it as a shard immediately, so a
    kill/crash never loses completed cells. Skips cells whose shard already exists."""
    key, trait, cfg_dict, hpo = args
    cfg = GridConfig(**cfg_dict)
    sp = _shard_path(cfg, key, trait)
    if sp.exists():
        return key, trait, "skip"
    df = run_job(args)
    sp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(sp)
    return key, trait, "done"


def execute(jobs, cfg, hpo, gpus, streams, serial=False):
    # Resume: drop cells whose shard already exists.
    todo = [(k, t) for (k, t) in jobs if not _shard_path(cfg, k, t).exists()]
    skipped = len(jobs) - len(todo)
    if skipped:
        print(f"[run] resuming: {skipped} cells already done (shards), {len(todo)} to do", flush=True)
    payloads = [(k, t, asdict(cfg), hpo) for (k, t) in todo]

    if serial:
        for i, pl in enumerate(payloads):
            _, _, st = _run_job_sharded(pl)
            print(f"[run] {i+1}/{len(payloads)} {pl[0]}:{pl[1]} {st}", flush=True)
    else:
        n_workers = len(gpus) * streams
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        for i in range(n_workers):
            q.put(gpus[i % len(gpus)])
        with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_gpu_init,
                                 initargs=(q, _omp_threads(n_workers))) as ex:
            futs = {ex.submit(_run_job_sharded, pl): (pl[0], pl[1]) for pl in payloads}
            for i, fut in enumerate(as_completed(futs)):
                k, t, st = fut.result()
                print(f"[run] {i+1}/{len(futs)} {k}:{t} {st}", flush=True)

    # Assemble the full table from all shards (this run's + any prior).
    shard_files = sorted(_shard_path(cfg, "*", "*").parent.glob("*.parquet"))
    return pd.concat([pd.read_parquet(s) for s in shard_files], ignore_index=True)


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


def _hpo_one(payload):
    """Tune ONE (dataset, arch) on the representative trait. Returns (key, arch, params)."""
    from ccgp.hpo import tune_model, _representative_trait
    cfg = GridConfig(**payload["cfg"])
    key, arch = payload["key"], payload["arch"]
    ds = make_dataset(key, cfg)
    trait = payload["trait"] or _representative_trait(ds)
    X, y, _, _ = ds.get_xy(trait)
    params = tune_model(X, y, arch, cfg.hpo_trials, cfg.hpo_folds, cfg.seed)
    return key, trait, arch, params


def run_hpo_parallel(cfg, path, gpus, streams, serial=False):
    """Parallel HPO across (dataset, arch) jobs -- each is independent. Writes the
    JSON incrementally as jobs complete, so a crash resumes from the last arch
    (not the last dataset). Falls back to the serial ``run_hpo`` when serial=True."""
    if serial:
        return run_hpo(cfg, path)
    from ccgp.hpo import _representative_trait
    path = Path(path)
    cache = json.loads(path.read_text()) if path.exists() else {}
    jobs = []
    for key in unique_keys(cfg):
        entry = cache.setdefault(key, {"trait": None, "params": {}})
        ds = make_dataset(key, cfg)
        if entry.get("trait") is None:
            entry["trait"] = _representative_trait(ds)
        for arch in models_for(key, cfg):
            if arch in entry["params"]:
                continue                              # resume: skip already-tuned
            jobs.append({"key": key, "arch": arch, "trait": entry["trait"], "cfg": asdict(cfg)})
    if not jobs:
        return cache
    print(f"[hpo] {len(jobs)} (dataset,arch) tuning jobs across {len(gpus)*streams} workers", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_workers = len(gpus) * streams
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for i in range(n_workers):
        q.put(gpus[i % len(gpus)])
    with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_gpu_init,
                             initargs=(q, _omp_threads(n_workers))) as ex:
        futs = {ex.submit(_hpo_one, pl): pl for pl in jobs}
        for i, fut in enumerate(as_completed(futs)):
            key, trait, arch, params = fut.result()
            cache[key]["trait"] = trait
            cache[key]["params"][arch] = params
            path.write_text(json.dumps(cache, indent=1))   # incremental checkpoint
            print(f"[hpo] {i+1}/{len(jobs)} {key}:{arch} done", flush=True)
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
        with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_gpu_init,
                                 initargs=(q, _omp_threads(n_workers))) as ex:
            futs = {ex.submit(run_split_payload, pl): pl for pl in payloads}
            for i, fut in enumerate(as_completed(futs)):
                shards.append(fut.result())
                print(f"[splits] {i+1}/{len(futs)} done", flush=True)
    return pd.concat(shards, ignore_index=True)


# --- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hpo", "main", "batch", "splits", "all"])
    ap.add_argument("--preset", default="full", choices=["full", "smoke", "campaign"])
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cell-index", type=int, default=None,
                    help="Run ONLY the i-th (dataset,trait) cell and write its shard, then exit. "
                         "For SLURM array jobs (one cell per task). HPO must be pre-cached.")
    ap.add_argument("--assemble", action="store_true",
                    help="Merge all main shards into results_main_<preset>.parquet and exit.")
    args = ap.parse_args()

    cfg = {"smoke": SMOKE, "full": FULL, "campaign": CAMPAIGN}[args.preset]
    gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    RESULTS.mkdir(exist_ok=True)

    # --- SLURM array mode: one cell per task -------------------------------
    if args.cell_index is not None or args.assemble:
        import json as _json
        hpo_path = RESULTS / f"hpo_{cfg.name}.json"
        if not hpo_path.exists():
            raise SystemExit(f"HPO cache {hpo_path} missing; ship it to the cluster first.")
        hpo = _json.loads(hpo_path.read_text())
        jobs = [(k, t) for k in unique_keys(cfg) for t in traits_for(make_dataset(k, cfg), cfg)]
        if args.assemble:
            df = validate_main_shards(cfg, jobs)
            out = args.out or RESULTS / f"results_main_{cfg.name}.parquet"
            df.to_parquet(out)
            print(f"[assemble] {len(jobs)}/{len(jobs)} cells -> {out} ({len(df)} rows)")
            return
        k, t = jobs[args.cell_index]
        sp = _shard_path(cfg, k, t)
        if sp.exists():
            print(f"[cell {args.cell_index}] {k}:{t} already done (shard exists), skip", flush=True)
            return
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpus[0]) if gpus else "0")
        df = run_job((k, t, asdict(cfg), hpo))
        sp.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(sp)
        print(f"[cell {args.cell_index}] {k}:{t} done -> {sp} ({len(df)} rows)", flush=True)
        return

    hpo = run_hpo_parallel(cfg, RESULTS / f"hpo_{cfg.name}.json", gpus, args.streams,
                           serial=args.serial)
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
