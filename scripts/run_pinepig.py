"""Recover the pine + pig main-grid cells (missed due to the now-fixed
genotype-ID type bug) and write results/results_pinepig.parquet to merge."""
import json
import sys
from dataclasses import asdict

import pandas as pd

sys.path.insert(0, ".")
from ccgp.config import FULL
import experiments.run as R

cfg = FULL
hpo = json.load(open("results/hpo_full.json"))
rows = []
for key in ["easygese:pine", "easygese:pig"]:
    ds = R.make_dataset(key, cfg)
    for t in R.traits_for(ds, cfg):
        df = R.run_job((key, t, asdict(cfg), hpo))
        rows.append(df)
        print(f"{key}:{t} -> {len(df)} rows", flush=True)
out = pd.concat(rows, ignore_index=True)
out.to_parquet("results/results_pinepig.parquet")
print("WROTE", len(out), "rows,", out[["dataset", "trait"]].drop_duplicates().shape[0], "trait-datasets")
