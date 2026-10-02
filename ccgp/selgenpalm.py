"""Private SelGenPalm external-validation adapter.

Raw data are never copied into this repository. The current 2025 endpoint lives
under ``D:/oil_palm`` (``/mnt/d/oil_palm`` in WSL); the historical neural-network
experiments and original Pearson-loss implementations live under
``D:/dcros_oilpalm``. Paths can be overridden with ``SELGENPALM_2025_ROOT`` and
``SELGENPALM_LEGACY_ROOT``.

The 2025 training table has repeated phenotypes for only 382 crosses and repeats
the same expected parental genotype for every record. This loader therefore
aggregates phenotypes and genotypes at the *cross* level before modelling. It
must not be replaced by a row-wise random split, which would place identical
genotypes in train and validation and grossly overstate generalization.
"""
from __future__ import annotations

import os
import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_2025_ROOT = Path(os.environ.get(
    "SELGENPALM_2025_ROOT", "/mnt/d/oil_palm/Data_for_DeepLearning2025"))
DEFAULT_LEGACY_ROOT = Path(os.environ.get(
    "SELGENPALM_LEGACY_ROOT", "/mnt/d/dcros_oilpalm"))
DEFAULT_FOLDS_ROOT = Path(os.environ.get(
    "SELGENPALM_FOLDS_ROOT", "/mnt/d/oil_palm/OptValSets_Daphne/kfold5_2025"))

TRAITS = {
    "ffb": ("ffb_adj", "ffb_adj_iSSEM"),
    "bn": ("bn_adj", "bn_adj_iSSEM"),
    "bw": ("bw_adj", "bw_adj_iSSEM"),
}


@dataclass
class SelGenPalmExternalSplit:
    trait: str
    X_train: np.ndarray
    y_train: np.ndarray
    weight_train: np.ndarray
    train_crosses: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    weight_test: np.ndarray
    test_crosses: np.ndarray
    marker_names: np.ndarray
    meta: dict


@dataclass
class SelGenPalmDevelopmentSet:
    """Cross-level training data; contains no external-test outcomes."""

    trait: str
    X: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    crosses: np.ndarray
    meta: dict


@dataclass
class SelGenPalmExternalFeatures:
    """Locked external-test inputs; contains identifiers but no outcomes."""

    X: np.ndarray
    crosses: np.ndarray
    marker_names: np.ndarray


@dataclass
class SelGenPalmExternalOutcomes:
    """External-test outcomes, loaded only by the final evaluation step."""

    trait: str
    y: np.ndarray
    weight: np.ndarray
    crosses: np.ndarray


@dataclass(frozen=True)
class DaphneFold:
    """One pre-specified cross-level model-development split."""

    fold: int
    train: np.ndarray
    validation: np.ndarray
    validation_crosses: np.ndarray


def load_daphne_folds(train_crosses: np.ndarray,
                      fold_dir: Path = DEFAULT_FOLDS_ROOT,
                      n_folds: int = 5) -> list[DaphneFold]:
    """Map the official DAPHNE cross lists to exhaustive, disjoint indices.

    These folds are model-development folds, not the external test. Strict
    validation is intentional: silently dropping or duplicating a cross would
    change the historical SelGenPalm protocol.
    """
    train_crosses = np.asarray(train_crosses, str).reshape(-1)
    if len(np.unique(train_crosses)) != len(train_crosses):
        raise ValueError("training cross identifiers must be unique")
    index = {cross: i for i, cross in enumerate(train_crosses)}
    fold_dir = Path(fold_dir)
    assigned: list[str] = []
    validation_lists: list[list[str]] = []
    for fold in range(1, n_folds + 1):
        path = fold_dir / f"kfold_{fold}_test.csv"
        frame = pd.read_csv(path)
        if "cross" not in frame.columns:
            raise ValueError(f"{path} must contain a 'cross' column")
        crosses = frame["cross"].astype(str).tolist()
        if len(crosses) != len(set(crosses)):
            raise ValueError(f"{path} contains duplicate crosses")
        unknown = sorted(set(crosses) - set(index))
        if unknown:
            raise ValueError(f"{path} contains crosses absent from training data: {unknown[:5]}")
        assigned.extend(crosses)
        validation_lists.append(crosses)
    if len(assigned) != len(set(assigned)):
        raise ValueError("DAPHNE validation folds overlap")
    missing = sorted(set(index) - set(assigned))
    if missing or len(assigned) != len(train_crosses):
        raise ValueError(f"DAPHNE folds are not exhaustive; missing {missing[:5]}")

    all_idx = np.arange(len(train_crosses))
    out = []
    for fold, crosses in enumerate(validation_lists, start=1):
        validation = np.asarray([index[cross] for cross in crosses], dtype=int)
        train = np.setdiff1d(all_idx, validation, assume_unique=True)
        out.append(DaphneFold(fold, train, validation, np.asarray(crosses, dtype=str)))
    return out


def aggregate_cross_phenotypes(frame: pd.DataFrame, trait_column: str):
    """Return cross means and inverse-SEM-squared precision weights.

    Cross order follows first appearance in the source table so the returned
    arrays stay aligned with first-row genotype extraction from the HDF5 file.
    """
    required = {"Cross_Recod", trait_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"phenotype table is missing {sorted(required - set(frame.columns))}")
    grouped = frame.groupby("Cross_Recod", sort=False)[trait_column].agg(["mean", "var", "count"])
    sem2 = grouped["var"].to_numpy(float) / grouped["count"].to_numpy(float)
    if not np.isfinite(sem2).all() or np.any(sem2 <= 0):
        raise ValueError("every training cross must have a finite, positive phenotype SEM")
    return (
        grouped.index.to_numpy(str),
        grouped["mean"].to_numpy(np.float64),
        (1.0 / sem2).astype(np.float64),
        grouped["count"].to_numpy(int),
    )


def _load_train_cross_genotypes(h5_path: Path, phenotype: pd.DataFrame):
    import h5py

    first = phenotype.reset_index(drop=True).drop_duplicates("Cross_Recod", keep="first").index.to_numpy()
    if not np.all(first[:-1] < first[1:]):
        raise ValueError("first-occurrence genotype indices must be strictly increasing")
    with h5py.File(h5_path, "r") as h5:
        ds = h5["genotypes"]
        if ds.shape[0] != len(phenotype):
            raise ValueError("training genotype and phenotype row counts differ")
        X = np.asarray(ds[first, :], dtype=np.float32)
    return np.ascontiguousarray(X)


def _load_test_genotypes(path: Path):
    """Read the moderately sized 536 x 23,937 compressed external-test matrix."""
    import pyarrow.csv as pacsv

    table = pacsv.read_csv(
        str(path), read_options=pacsv.ReadOptions(use_threads=True, block_size=1 << 26))
    X = np.empty((table.num_rows, table.num_columns), dtype=np.float32)
    for j in range(table.num_columns):
        X[:, j] = table.column(j).to_numpy(zero_copy_only=False)
    return np.ascontiguousarray(X), np.asarray(table.column_names, dtype=object)


def _compressed_csv_header(path: Path) -> np.ndarray:
    """Read only a compressed matrix header, without scanning genotype rows."""
    with gzip.open(path, "rt", newline="") as stream:
        return np.asarray(next(csv.reader(stream)), dtype=object)


def load_selgenpalm_development_2025(
        trait: str = "ffb", root: Path = DEFAULT_2025_ROOT) -> SelGenPalmDevelopmentSet:
    """Load cross-level model-development data without touching test outcomes."""
    trait = trait.lower()
    if trait not in TRAITS:
        raise ValueError(f"unknown SelGenPalm trait {trait!r}; choose from {sorted(TRAITS)}")
    root = Path(root)
    y_col, _ = TRAITS[trait]
    train_pheno = pd.read_csv(root / "Ytrain2025_Prod_RECOD.csv",
                              usecols=["Cross_Recod", y_col])
    train_crosses, y_train, weight_train, record_counts = aggregate_cross_phenotypes(
        train_pheno, y_col)
    first_crosses = train_pheno.drop_duplicates("Cross_Recod", keep="first")["Cross_Recod"].astype(str)
    if not np.array_equal(train_crosses, first_crosses.to_numpy()):
        raise ValueError("training cross order is inconsistent")
    X_train = _load_train_cross_genotypes(
        root / "Xtrain2025_Categ_ExpectedGenotypes_Prod.h5", train_pheno)

    return SelGenPalmDevelopmentSet(
        trait=trait,
        X=X_train,
        y=y_train,
        weight=weight_train,
        crosses=train_crosses,
        meta={
            "source": "SelGenPalm 2025 private model development",
            "legacy_source": str(DEFAULT_LEGACY_ROOT),
            "n_train_crosses": len(train_crosses),
            "p": X_train.shape[1],
            "training_records_min": int(record_counts.min()),
            "training_records_max": int(record_counts.max()),
            "genotype_level": "expected parental genotype per cross",
            "data_redistribution": "prohibited unless separately authorized",
        },
    )


def load_selgenpalm_external_features_2025(
        root: Path = DEFAULT_2025_ROOT) -> SelGenPalmExternalFeatures:
    """Load external genotypes and IDs, deliberately excluding phenotypes."""
    root = Path(root)
    X_test, marker_names = _load_test_genotypes(
        root / "Xtest2025_Categ_ExpectedGenotypes_ProdMean3to10y.csv.gz")
    train_marker_names = _compressed_csv_header(
        root / "Xtrain2025_Categ_ExpectedGenotypes_Prod.csv.gz")
    if not np.array_equal(marker_names, train_marker_names):
        raise ValueError("SelGenPalm train/test marker names or order differ")
    test_ids = pd.read_csv(root / "Ytest2025_ProdMean3to10y_RECOD.csv",
                           usecols=["Cross_Recod"])["Cross_Recod"].astype(str).to_numpy()
    if len(test_ids) != len(X_test) or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError("external genotype rows and unique cross identifiers are not aligned")
    return SelGenPalmExternalFeatures(X_test, test_ids, marker_names)


def load_selgenpalm_external_outcomes_2025(
        trait: str = "ffb", root: Path = DEFAULT_2025_ROOT) -> SelGenPalmExternalOutcomes:
    """Reveal the locked endpoint; call only after predictions are complete."""
    trait = trait.lower()
    if trait not in TRAITS:
        raise ValueError(f"unknown SelGenPalm trait {trait!r}; choose from {sorted(TRAITS)}")
    root = Path(root)
    y_col, w_col = TRAITS[trait]
    test_pheno = pd.read_csv(root / "Ytest2025_ProdMean3to10y_RECOD.csv",
                             usecols=["Cross_Recod", y_col, w_col])
    weight = test_pheno[w_col].to_numpy(np.float64)
    y = test_pheno[y_col].to_numpy(np.float64)
    if not np.isfinite(y).all():
        raise ValueError("external-test phenotypes must be finite")
    if not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError("external-test precision weights must be finite and positive")
    return SelGenPalmExternalOutcomes(
        trait, y, weight, test_pheno["Cross_Recod"].astype(str).to_numpy())


def load_selgenpalm_2025(trait: str = "ffb",
                         root: Path = DEFAULT_2025_ROOT) -> SelGenPalmExternalSplit:
    """Load the complete endpoint for audit/analysis, not model development."""
    development = load_selgenpalm_development_2025(trait, root)
    features = load_selgenpalm_external_features_2025(root)
    outcomes = load_selgenpalm_external_outcomes_2025(trait, root)
    if not np.array_equal(features.crosses, outcomes.crosses):
        raise ValueError("SelGenPalm external features and outcomes are not aligned")
    if development.X.shape[1] != features.X.shape[1]:
        raise ValueError("SelGenPalm train/test marker counts differ")

    return SelGenPalmExternalSplit(
        trait=trait,
        X_train=development.X,
        y_train=development.y,
        weight_train=development.weight,
        train_crosses=development.crosses,
        X_test=features.X,
        y_test=outcomes.y,
        weight_test=outcomes.weight,
        test_crosses=features.crosses,
        marker_names=features.marker_names,
        meta={
            "source": "SelGenPalm 2025 private external validation",
            "legacy_source": str(DEFAULT_LEGACY_ROOT),
            "n_train_crosses": len(development.crosses),
            "n_test_crosses": len(features.crosses),
            "p": development.X.shape[1],
            "training_records_min": development.meta["training_records_min"],
            "training_records_max": development.meta["training_records_max"],
            "genotype_level": "expected parental genotype per cross",
            "data_redistribution": "prohibited unless separately authorized",
        },
    )


def legacy_single_record_paths(root: Path = DEFAULT_LEGACY_ROOT) -> dict[str, Path]:
    """Inventory the historical production files without loading or redistributing them."""
    base = Path(root) / "Data_for_DeepLearning" / "SingleRecords"
    names = {
        "X_train": "Xtr_PROD_SingleRecords.csv",
        "y_train": "Ytr_PROD_SingleRecords.csv",
        "train_ids": "XYtr_PROD_rowNames.txt",
        "X_validation": "Xval_PROD_SingleRecords.csv",
        "y_validation": "Yval_PROD_SingleRecords.csv",
        "validation_ids": "XYval_PROD_rowNames.txt",
    }
    paths = {key: base / value for key, value in names.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing historical SelGenPalm files: {missing}")
    return paths
