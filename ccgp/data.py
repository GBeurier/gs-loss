"""Public dataset loaders and genotype preprocessing.

Sources (all public, reproducible):

* **EasyGeSe** (Quesada-Traver et al. 2025) -- 10 species, genotypes, phenotypes
  and *predefined* 5-fold x 5-repeat cross-validation partitions, downloaded once
  from Zenodo (record 15348871) and cached locally.
* **CIMMYT wheat** (Crossa et al. 2010) via the BGLR R package -- 599 lines,
  1279 DArT markers, grain yield in 4 environments (enables cross-environment
  evaluation).
* **SoyNAM** (Xavier et al. 2016) via the SoyNAM R package -- ~5500 RILs in 40
  biparental families (enables leave-family-out evaluation).

Marker preprocessing is unsupervised (no phenotype used): MAF filtering, mean
imputation, and an optional top-variance cap to keep very wide matrices
tractable. Per-fold *standardization* (which uses training statistics only) is
applied later by the experiment runner to avoid leakage.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data_cache"
# EasyGeSe dataset index (Zenodo URLs) vendored inside the package so the
# pipeline does not depend on a separate clone of the EasyGeSe repository.
INDEX_PATH = Path(__file__).resolve().parent / "easygese_index.json"
R_SCRIPT = ROOT / "scripts" / "export_r_data.R"

EASYGESE_SPECIES = ["barley", "bean", "lentil", "maize", "oyster",
                    "pig", "pine", "rice", "soybean", "wheatG"]


# --- dataset container -------------------------------------------------------

@dataclass
class GenomicDataset:
    name: str
    species: str
    X: np.ndarray                      # (n, p) float32 marker dosages (preprocessed)
    ids: np.ndarray                    # (n,) genotype identifiers
    traits: dict[str, np.ndarray]      # trait -> (n,) phenotype (may contain NaN)
    folds: dict | None = None          # EasyGeSe Z object (predefined CV), else None
    groups: np.ndarray | None = None   # (n,) family/group label, else None
    env: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def trait_names(self) -> list[str]:
        return list(self.traits)

    def get_xy(self, trait: str):
        """Return (X, y, ids, groups) with NaN-phenotype rows dropped."""
        y = self.traits[trait]
        m = ~np.isnan(y)
        g = self.groups[m] if self.groups is not None else None
        return self.X[m], y[m], self.ids[m], g


# --- preprocessing -----------------------------------------------------------

def _maf_filter(X: np.ndarray, thresh: float) -> np.ndarray:
    if thresh <= 0:
        return np.ones(X.shape[1], bool)
    dmax = 2.0 if np.nanmax(X) > 1.0 else 1.0
    freq = np.nanmean(X, axis=0) / dmax
    maf = np.minimum(freq, 1.0 - freq)
    return maf >= thresh


def _mean_impute(X: np.ndarray) -> np.ndarray:
    if not np.isnan(X).any():
        return X
    col_mean = np.nanmean(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(col_mean, idx[1])
    return X


def preprocess_markers(X: np.ndarray, maf: float = 0.01,
                       max_features: int | None = 20000) -> tuple[np.ndarray, dict]:
    """MAF filter -> mean impute -> optional top-variance cap. Returns (X, info)."""
    X = np.asarray(X, np.float32)
    p0 = X.shape[1]
    keep = _maf_filter(X, maf)
    X = X[:, keep]
    X = _mean_impute(X)
    capped = False
    if max_features is not None and X.shape[1] > max_features:
        var = X.var(axis=0)
        top = np.argpartition(var, -max_features)[-max_features:]
        top.sort()
        X = X[:, top]
        capped = True
    info = {"p_raw": p0, "p_after_maf": int(keep.sum()), "p_used": X.shape[1],
            "maf": maf, "variance_capped": capped}
    return np.ascontiguousarray(X), info


# --- EasyGeSe ----------------------------------------------------------------

def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    return dest


def _read_matrix_csv(path) -> tuple[np.ndarray, np.ndarray]:
    """Memory-efficient read of a wide genotype CSV (col 0 = id, rest numeric).

    pandas is prohibitively memory-hungry on matrices with 100k+ columns (maize,
    barley); pyarrow reads them columnar into a preallocated float32 array.
    Returns (ids: object array, X: float32 (n, p)).
    """
    import pyarrow.csv as pacsv

    # a 256 MB block holds the very wide header row (100k+ columns) that overflows
    # pyarrow's default 1 MB block.
    tbl = pacsv.read_csv(str(path),
                         read_options=pacsv.ReadOptions(use_threads=True, block_size=1 << 28))
    names = tbl.column_names
    # Coerce IDs to str: some panels (pine, pig) have purely numeric genotype IDs
    # that parse as integers and then fail to align with the string-keyed fold JSON.
    ids = np.asarray([str(v) for v in tbl.column(0).to_pylist()], dtype=object)
    n, p = tbl.num_rows, len(names) - 1
    X = np.empty((n, p), dtype=np.float32)
    for j in range(1, len(names)):
        X[:, j - 1] = tbl.column(j).to_numpy(zero_copy_only=False)
    return ids, X


def load_easygese(species: str, maf: float = 0.01, max_features: int | None = 20000,
                  cache: Path = CACHE) -> GenomicDataset:
    index = json.loads(INDEX_PATH.read_text())
    entry = index[species]
    base = cache / "easygese"
    xp = _download(entry["X"], base / f"{species}X.csv")
    yp = _download(entry["Y"], base / f"{species}Y.csv")
    zp = _download(entry["Z"], base / f"{species}Z.json")

    ids, Xraw = _read_matrix_csv(xp)
    Ydf = pd.read_csv(yp, index_col=0)
    Ydf.index = Ydf.index.map(str)
    Ydf = Ydf.reindex(ids)                             # align phenotypes to genotype order
    Z = json.loads(Path(zp).read_text())

    X, info = preprocess_markers(Xraw, maf=maf, max_features=max_features)
    traits = {c: Ydf[c].to_numpy(np.float64) for c in Ydf.columns}
    meta = {"source": "EasyGeSe", "citation": entry.get("citation", ""),
            "n": X.shape[0], **info}
    return GenomicDataset(name=f"easygese:{species}", species=species, X=X,
                          ids=np.asarray(ids), traits=traits, folds=Z, meta=meta)


# --- R-hosted datasets -------------------------------------------------------

def _ensure_r_export(what: str, cache: Path = CACHE) -> None:
    targets = {"wheat": cache / "wheat" / "X.csv",
               "soynam": cache / "soynam" / "yield_pheno.csv"}
    if targets[what].exists():
        return
    subprocess.run(["Rscript", str(R_SCRIPT), str(cache), what], check=True)


def load_wheat(maf: float = 0.0, max_features: int | None = None,
               cache: Path = CACHE) -> GenomicDataset:
    _ensure_r_export("wheat", cache)
    wd = cache / "wheat"
    X = pd.read_csv(wd / "X.csv", header=None).to_numpy(np.float32)
    Y = pd.read_csv(wd / "Y.csv")
    Xp, info = preprocess_markers(X, maf=maf, max_features=max_features)
    traits = {c: Y[c].to_numpy(np.float64) for c in Y.columns}
    meta = {"source": "BGLR::wheat (Crossa et al. 2010)", "n": Xp.shape[0],
            "marker_type": "DArT 0/1", "environments": list(Y.columns), **info}
    return GenomicDataset(name="wheat", species="wheat", X=Xp,
                          ids=np.arange(Xp.shape[0]).astype(str), traits=traits,
                          env=np.asarray(list(Y.columns)), meta=meta)


def load_soynam(trait: str = "yield", maf: float = 0.0, max_features: int | None = None,
                cache: Path = CACHE) -> GenomicDataset:
    _ensure_r_export("soynam", cache)
    sd = cache / "soynam"
    G = pd.read_csv(sd / f"{trait}_geno.csv.gz", header=None).to_numpy(np.float32)
    ph = pd.read_csv(sd / f"{trait}_pheno.csv")
    Xp, info = preprocess_markers(G, maf=maf, max_features=max_features)
    meta = {"source": "SoyNAM (Xavier et al. 2016)", "n": Xp.shape[0],
            "n_families": int(ph["family"].nunique()), "marker_type": "0/1/2", **info}
    return GenomicDataset(name=f"soynam:{trait}", species="soybeanNAM", X=Xp,
                          ids=np.arange(Xp.shape[0]).astype(str),
                          traits={trait: ph["y"].to_numpy(np.float64)},
                          groups=ph["family"].to_numpy(), meta=meta)


SOYNAM_TRAITS = ["yield", "height", "protein", "oil"]
