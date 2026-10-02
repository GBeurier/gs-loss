"""Literature-inspired NN benchmark (legacy Palier-2 driver): run protocol-matched
adaptations of genomic-prediction networks reported competitive with GBLUP
-- DeepGS, DNNGP, PNNGS, SoyDNGP -- across all 101 trait-datasets under every
core loss, so each is compared MSE vs the differentiable Pearson loss (and the
hybrid/CCC) under the same frozen-architecture protocol as the main grid. GBLUP
and ridge baselines are reused from results_main_full.parquet (not re-run).

Resumable: writes one parquet shard per (dataset, trait) under
results/litnn_shards/; re-running skips completed shards. Merge with:
  python -c "import pandas as pd,glob; pd.concat([pd.read_parquet(f) for f in glob.glob('results/litnn_shards/*.parquet')]).to_parquet('results/results_litnn_full.parquet')"

These share the papers' defining architectural motifs but use common preprocessing
and tractability modifications; they are not claimed as exact reproductions.

Usage:
  python experiments/run_litnn.py --gpus 0,1 --streams 2
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
from ccgp.hpo import tune_model, _representative_trait
from experiments.run import RESULTS, build_splits, make_dataset, unique_keys, traits_for, _nn_params

LIT_ARCHS = ["deepgs", "dnngp", "pnngs", "soydngp"]
HPO_PATH = RESULTS / "hpo_full.json"
SHARDS = RESULTS / "litnn_shards"


def shard_path(key: str, trait: str) -> Path:
    safe = f"{key}__{trait}".replace(":", "_").replace("/", "_").replace(" ", "_").replace("*", "x")
    return SHARDS / f"{safe}.parquet"


def ensure_hpo(cfg: GridConfig) -> dict:
    """Tune each literature arch once per dataset (representative trait, MSE objective),
    cache to hpo_full.json. Incremental: persists after every tune so a crash is cheap."""
    hpo = json.loads(HPO_PATH.read_text())
    for key in unique_keys(cfg):
        entry = hpo.setdefault(key, {"params": {}})
        params = entry["params"]
        ds = None
        for arch in LIT_ARCHS:
            if arch in params:
                continue
            if ds is None:
                ds = make_dataset(key, cfg)
                trait = entry.get("trait") or _representative_trait(ds)
                entry["trait"] = trait
                X, y, _, _ = ds.get_xy(trait)
            print(f"[litnn-hpo] {arch} {key}", flush=True)
            params[arch] = tune_model(X, y, arch, cfg.hpo_trials, cfg.hpo_folds, cfg.seed)
            HPO_PATH.write_text(json.dumps(hpo, indent=1))
    return hpo


def run_one(payload):
    cfg = GridConfig(**payload["cfg"])
    hpo = payload["hpo"]
    key, trait = payload["key"], payload["trait"]
    out = shard_path(key, trait)
    if out.exists():
        return ("skip", key, trait, 0)
    ds = make_dataset(key, cfg)
    splits = build_splits(ds, trait, cfg)
    pm = hpo[key]["params"]
    rows = []
    from ccgp.experiment import run_cell
    for arch in LIT_ARCHS:
        base = pm[arch]
        for loss in cfg.losses:
            rows += run_cell(ds, trait, arch, loss, splits, params=_nn_params(base, loss, cfg),
                             fracs=cfg.fracs, cal_names=cfg.calibrators,
                             base_meta={"exp": "litnn"}, seed=cfg.seed)
    df = pd.DataFrame(rows)
    df.to_parquet(out)
    return ("done", key, trait, len(df))


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
    SHARDS.mkdir(parents=True, exist_ok=True)

    hpo = ensure_hpo(cfg)
    cfgd = asdict(cfg)
    jobs = [(k, t) for k in unique_keys(cfg) for t in traits_for(make_dataset(k, cfg), cfg)]
    todo = [(k, t) for k, t in jobs if not shard_path(k, t).exists()]
    print(f"[litnn] {len(jobs)} trait-datasets, {len(todo)} remaining", flush=True)
    payloads = [{"key": k, "trait": t, "cfg": cfgd, "hpo": hpo} for k, t in todo]

    if args.serial:
        for i, pl in enumerate(payloads):
            st, k, t, n = run_one(pl)
            print(f"[litnn] {i+1}/{len(payloads)} {k}:{t} {st} ({n})", flush=True)
    else:
        n_workers = len(gpus) * args.streams
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        for i in range(n_workers):
            q.put(gpus[i % len(gpus)])
        with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_gpu_init, initargs=(q,)) as ex:
            futs = {ex.submit(run_one, pl): pl for pl in payloads}
            for i, fut in enumerate(as_completed(futs)):
                st, k, t, n = fut.result()
                print(f"[litnn] {i+1}/{len(futs)} {k}:{t} {st} ({n})", flush=True)

    # merge shards
    shards = sorted(SHARDS.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    df.to_parquet(RESULTS / "results_litnn_full.parquet")
    print(f"[litnn] merged {len(shards)} shards -> results_litnn_full.parquet ({len(df)} rows)")


if __name__ == "__main__":
    main()
