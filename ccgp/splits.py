"""Cross-validation schemes.

* ``predefined_folds`` -- the fixed EasyGeSe 5-fold x 5-repeat partitions
  (column ``Split{fold}CV{repeat}``; value 1 = test, 0 = train).
* ``repeated_kfold``   -- standard random repeated K-fold.
* ``leave_family_out`` -- one biparental family held out per fold (SoyNAM).
* ``inner_split``      -- training/validation split for early stopping and
  calibration; group-aware so a held-out family is not split across fit/val.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    scheme: str
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray
    label: str = ""


def predefined_folds(folds_obj: dict, trait: str, ids: np.ndarray) -> list[Split]:
    """Build splits from an EasyGeSe Z object, aligned to ``ids`` (post NaN-drop)."""
    df = pd.DataFrame.from_dict(folds_obj[trait], orient="index").reindex(ids)
    out: list[Split] = []
    for col in df.columns:
        m = re.match(r"Split(\d+)CV(\d+)", str(col))
        if not m:
            continue
        fold, rep = int(m.group(1)), int(m.group(2))
        test_mask = df[col].to_numpy() == 1
        test = np.where(test_mask)[0]
        train = np.where(~test_mask & ~np.isnan(df[col].to_numpy()))[0]
        if len(test) and len(train):
            out.append(Split("predefined", rep, fold, train, test, col))
    return out


def repeated_kfold(n: int, n_splits: int = 5, n_repeats: int = 10, seed: int = 0) -> list[Split]:
    from sklearn.model_selection import RepeatedKFold

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    out = []
    for i, (tr, te) in enumerate(rkf.split(np.arange(n))):
        out.append(Split("random", i // n_splits + 1, i % n_splits + 1, tr, te))
    return out


def leave_family_out(groups: np.ndarray, min_test: int = 5,
                     max_families: int | None = None, seed: int = 0) -> list[Split]:
    fams = pd.unique(groups)
    if max_families is not None and len(fams) > max_families:
        rng = np.random.default_rng(seed)
        fams = rng.choice(fams, size=max_families, replace=False)
    out = []
    for i, fam in enumerate(sorted(fams, key=str)):
        test = np.where(groups == fam)[0]
        if len(test) < min_test:
            continue
        train = np.where(groups != fam)[0]
        out.append(Split("family", 1, i + 1, train, test, f"fam={fam}"))
    return out


def inner_split(train: np.ndarray, groups: np.ndarray | None = None,
                val_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (fit_idx, val_idx) as positions; group-aware when groups given."""
    rng = np.random.default_rng(seed)
    if groups is None:
        perm = rng.permutation(train)
        nv = max(2, int(val_frac * len(train)))
        return perm[nv:], perm[:nv]
    g = groups[train]
    ufams = pd.unique(g)
    rng.shuffle(ufams)
    nval = max(1, int(val_frac * len(ufams)))
    val_fams = set(ufams[:nval].tolist())
    mask = np.array([x in val_fams for x in g])
    if mask.all() or (~mask).all():            # degenerate -> fall back to random
        perm = rng.permutation(train)
        nv = max(2, int(val_frac * len(train)))
        return perm[nv:], perm[:nv]
    return train[~mask], train[mask]
