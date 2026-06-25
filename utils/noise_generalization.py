#!/usr/bin/env python3
"""Generalized synthetic noise-injection experiments.

This module intentionally does not modify the original Polaris-only noise
experiment. It consumes existing BenchAudit ``runs/*/*`` artifacts and runs
task-aware perturbations across single-task regression and binary
classification datasets.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import sys
import zlib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import DataStructs, rdBase
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    explained_variance_score,
    f1_score,
    log_loss,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    matthews_corrcoef,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from utils.analysis import morgan_fps, scaffold_fps

rdBase.DisableLog("rdApp.warning")

DEFAULT_SCENARIOS = ("conflicts", "cliffs", "contamination", "random_label_noise")
DEFAULT_FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
DEFAULT_MODELS = ("rf", "lgbm", "torch_mlp")
BENCHMARK_FAMILIES = ("moleculenet", "tdc", "polaris", "dti")

CLASSIFICATION_METRICS = (
    "roc_auc",
    "pr_auc",
    "average_precision",
    "balanced_accuracy",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "mcc",
    "log_loss",
    "brier_score",
    "n_eval_rows",
    "positive_rate",
)

REGRESSION_METRICS = (
    "mse",
    "rmse",
    "mae",
    "median_ae",
    "r2",
    "explained_variance",
    "pearson",
    "spearman",
    "kendall_tau",
    "max_error",
    "n_eval_rows",
)


@dataclass(frozen=True)
class DatasetBundle:
    """Loaded run artifacts and metadata for one dataset."""

    family: str
    dataset: str
    run_dir: Path
    task_type: str
    is_dti: bool
    summary: Dict[str, Any]
    records: pd.DataFrame
    sequence_id_map: Dict[str, str]


@dataclass(frozen=True)
class NoiseDataset:
    """Dataset bundle with its inferred label-task count."""

    bundle: DatasetBundle
    n_label_tasks: int


@dataclass(frozen=True)
class BackendAvailability:
    """Availability flags for optional model-training backends."""

    have_lgbm: bool
    have_torch: bool
    have_lightning: bool


def _config_name(summary: Dict[str, Any]) -> str:
    return str(summary.get("config", {}).get("name") or "")


def normalize_records(records: pd.DataFrame, dataset: str, is_dti: bool) -> pd.DataFrame:
    """Normalize a ``records.csv`` frame into the experiment schema."""
    df = records.copy()
    df["_record_id"] = np.arange(len(df), dtype=int)
    df["_dataset"] = dataset
    if "split" not in df.columns:
        raise ValueError(f"{dataset}: records.csv is missing split column")
    df["split"] = df["split"].astype(str).str.lower()
    df = df[df["split"].isin({"train", "valid", "test"})].copy()
    if "valid" in df.columns:
        df = df[df["valid"].fillna(False).astype(bool)].copy()
    if "smiles_clean" not in df.columns:
        raise ValueError(f"{dataset}: records.csv is missing smiles_clean")
    df["smiles_clean"] = df["smiles_clean"].astype("string").fillna("").astype(str).str.strip()
    df = df[df["smiles_clean"] != ""].copy()
    if "label_raw" not in df.columns:
        raise ValueError(f"{dataset}: records.csv is missing label_raw")
    if is_dti:
        if "sequence_aa" not in df.columns:
            raise ValueError(f"{dataset}: DTI records.csv is missing sequence_aa")
        df["sequence_aa"] = df["sequence_aa"].astype("string").fillna("").astype(str).str.strip().str.upper()
        df = df[df["sequence_aa"] != ""].copy()
    else:
        df["sequence_aa"] = ""
    return df.reset_index(drop=True)


def _parse_label(value: Any) -> List[float]:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        seq = list(value)
    elif value is None:
        seq = [np.nan]
    elif isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            seq = [np.nan]
        elif text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                seq = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
            except Exception:
                seq = [np.nan]
        else:
            seq = [text]
    else:
        seq = [value]

    out: List[float] = []
    for item in seq:
        if item is None:
            out.append(np.nan)
            continue
        if isinstance(item, str) and item.strip().lower() in {"", "nan", "none", "null"}:
            out.append(np.nan)
            continue
        try:
            out.append(float(item))
        except Exception:
            out.append(np.nan)
    return out or [np.nan]


def label_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return labels as a two-dimensional numeric array."""
    rows = [_parse_label(value) for value in df["label_raw"].tolist()]
    width = max((len(row) for row in rows), default=1)
    arr = np.full((len(rows), width), np.nan, dtype=float)
    for idx, row in enumerate(rows):
        arr[idx, : len(row)] = row
    return arr


def label_task_count(df: pd.DataFrame) -> int:
    """Return the maximum number of label tasks represented by a dataset."""
    return int(label_matrix(df).shape[1])


def _fingerprint_matrix(smiles: Sequence[str], radius: int, nbits: int) -> np.ndarray:
    fps = morgan_fps(list(smiles), radius=radius, n_bits=nbits)
    arr = np.zeros((len(fps), nbits), dtype=np.float32)
    for idx, fp in enumerate(fps):
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, arr[idx])
    return arr


def _target_hash_features(sequences: Sequence[str], nbits: int = 512, k: int = 3) -> np.ndarray:
    arr = np.zeros((len(sequences), nbits), dtype=np.float32)
    for row, seq in enumerate(sequences):
        seq = str(seq or "").upper()
        if len(seq) < k:
            if seq:
                arr[row, zlib.crc32(seq.encode("utf-8")) % nbits] = 1.0
            continue
        for i in range(len(seq) - k + 1):
            token = seq[i : i + k]
            arr[row, zlib.crc32(token.encode("utf-8")) % nbits] += 1.0
        norm = float(np.linalg.norm(arr[row]))
        if norm > 0:
            arr[row] /= norm
    return arr


def feature_matrix(df: pd.DataFrame, bundle: DatasetBundle, fp_radius: int, fp_nbits: int, target_bits: int) -> np.ndarray:
    """Build molecular and optional target-sequence features for model training."""
    mol = _fingerprint_matrix(df["smiles_clean"].astype(str).tolist(), fp_radius, fp_nbits)
    if not bundle.is_dti:
        return mol
    target = _target_hash_features(df["sequence_aa"].astype(str).tolist(), nbits=target_bits)
    return np.concatenate([mol, target], axis=1)


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_floats(raw: str) -> List[float]:
    values = [float(item) for item in _parse_csv(raw)]
    if not values:
        raise ValueError("At least one fraction is required.")
    if any((not np.isfinite(v)) or v < 0 for v in values):
        raise ValueError("Fractions must be finite values >= 0.")
    return sorted(set(float(v) for v in values))


def _parse_ints(raw: str) -> List[int]:
    values = [int(item) for item in _parse_csv(raw)]
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def _matches_filters(name: str, config_name: str, filters: Sequence[str]) -> bool:
    if not filters:
        return True
    candidates = {str(name), str(config_name)}
    return any(item in candidates for item in filters)


def _counts(records: pd.DataFrame) -> Dict[str, int]:
    return {
        "rows_total": int(len(records)),
        "rows_train": int(records["split"].eq("train").sum()) if "split" in records else 0,
        "rows_valid": int(records["split"].eq("valid").sum()) if "split" in records else 0,
        "rows_test": int(records["split"].eq("test").sum()) if "split" in records else 0,
    }


def _unique_finite_labels(values: np.ndarray) -> List[float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    return sorted(float(v) for v in np.unique(vals))


def _is_binary_label_set(labels: Sequence[float]) -> bool:
    return all(np.isclose(v, 0.0) or np.isclose(v, 1.0) for v in labels)


def discover_noise_datasets(
    runs_root: Path,
    dataset_filters: Sequence[str],
    skip_multitask: bool = True,
) -> Tuple[List[NoiseDataset], List[Dict[str, Any]]]:
    """Discover datasets and return both included bundles and manifest rows."""

    filters = [item for item in dataset_filters if item and item.lower() != "all"]
    include_all = not filters
    datasets: List[NoiseDataset] = []
    manifest: List[Dict[str, Any]] = []

    for family in BENCHMARK_FAMILIES:
        family_root = runs_root / family
        if not family_root.exists():
            continue
        for summary_path in sorted(family_root.glob("*/summary.json")):
            run_dir = summary_path.parent
            records_path = run_dir / "records.csv"
            if not records_path.exists():
                continue
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

            dataset = run_dir.name
            config_name = _config_name(summary)
            if not include_all and not _matches_filters(dataset, config_name, filters):
                continue

            task_type = str(summary.get("task", {}).get("type") or summary.get("config", {}).get("task") or "")
            if task_type not in {"classification", "regression"}:
                task_type = str(summary.get("config", {}).get("task") or "classification")
            is_dti = family == "dti" or str(summary.get("config", {}).get("modality", "")).lower() == "dti"

            try:
                records = normalize_records(pd.read_csv(records_path), dataset=dataset, is_dti=is_dti)
                n_tasks = label_task_count(records)
            except Exception as exc:
                manifest.append(
                    {
                        "family": family,
                        "dataset": dataset,
                        "status": "skipped",
                        "skip_reason": f"load_error: {exc}",
                        "task_type": task_type,
                        "n_label_tasks": np.nan,
                        **{"rows_total": 0, "rows_train": 0, "rows_valid": 0, "rows_test": 0},
                    }
                )
                continue

            row = {
                "family": family,
                "dataset": dataset,
                "task_type": task_type,
                "n_label_tasks": int(n_tasks),
                **_counts(records),
            }
            if skip_multitask and n_tasks > 1:
                manifest.append({**row, "status": "skipped", "skip_reason": f"multitask:{n_tasks}"})
                continue

            y = label_matrix(records)
            finite = y[np.isfinite(y)]
            if task_type == "classification" and finite.size:
                labels = _unique_finite_labels(finite)
                if labels and not _is_binary_label_set(labels):
                    manifest.append({**row, "status": "skipped", "skip_reason": f"non_binary_labels:{labels}"})
                    continue

            bundle = DatasetBundle(
                family=family,
                dataset=dataset,
                run_dir=run_dir,
                task_type=task_type,
                is_dti=is_dti,
                summary=summary,
                records=records,
                sequence_id_map={},
            )
            datasets.append(NoiseDataset(bundle=bundle, n_label_tasks=n_tasks))
            manifest.append({**row, "status": "included", "skip_reason": ""})

    return datasets, manifest


def _label_values(df: pd.DataFrame) -> np.ndarray:
    y = label_matrix(df)
    if y.shape[1] != 1:
        raise ValueError(f"Expected single-task labels, got {y.shape[1]} tasks")
    return y[:, 0].astype(float)


def _set_label_values(df: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    vals = np.asarray(values, dtype=float)
    out["label_raw"] = vals
    return out


def _finite_label_frame(df: pd.DataFrame) -> pd.DataFrame:
    y = _label_values(df)
    return df[np.isfinite(y)].copy().reset_index(drop=True)


def label_sigma3(values: np.ndarray) -> float:
    """Return a robust three-sigma scale estimate for finite label values."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return 1.0
    std = float(np.std(vals))
    sigma3 = 3.0 * std
    return sigma3 if np.isfinite(sigma3) and sigma3 > 0 else 1.0


def _n_from_fraction(n_rows: int, fraction: float) -> int:
    return int(math.ceil(float(fraction) * int(n_rows))) if fraction > 0 else 0


def _sample_indices(n_rows: int, n_select: int, rng: np.random.Generator) -> np.ndarray:
    if n_rows <= 0 or n_select <= 0:
        return np.empty((0,), dtype=int)
    return rng.choice(n_rows, size=n_select, replace=n_select > n_rows).astype(int)


def _perturb_regression(values: np.ndarray, sigma3: float, rng: np.random.Generator) -> np.ndarray:
    sign = rng.choice(np.asarray([-1.0, 1.0]), size=len(values))
    scale = rng.uniform(0.9, 1.1, size=len(values))
    return np.asarray(values, dtype=float) + sign * sigma3 * scale


def inject_conflicts(
    train: pd.DataFrame,
    task_type: str,
    fraction: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Inject duplicate-label conflicts into a training frame."""
    y = _label_values(train)
    idx = _sample_indices(len(train), _n_from_fraction(len(train), fraction), rng)
    if idx.size == 0:
        return train.copy(), {"n_conflicts_added": 0.0}

    dup = train.iloc[idx].copy()
    dup_y = y[idx].copy()
    if task_type == "classification":
        dup_y = 1.0 - dup_y.astype(int)
    else:
        dup_y = _perturb_regression(dup_y, label_sigma3(y), rng)
    dup = _set_label_values(dup, dup_y)
    out = pd.concat([train, dup], ignore_index=True)
    return out, {"n_conflicts_added": float(len(dup))}


def _valid_neighbor_indices(
    row_idx: int,
    train: pd.DataFrame,
    fps: Sequence[Any],
    scaffolds: Sequence[Any],
    sim_threshold: float,
    rng: np.random.Generator,
    candidate_pool: int,
) -> List[int]:
    if len(train) <= 1:
        return []
    if candidate_pool >= len(train) - 1:
        candidates = np.arange(len(train), dtype=int)
        candidates = candidates[candidates != row_idx]
    else:
        sampled = rng.choice(len(train) - 1, size=max(1, candidate_pool), replace=False)
        candidates = sampled + (sampled >= row_idx)

    if "sequence_aa" in train.columns and str(train.iloc[row_idx].get("sequence_aa", "")):
        row_seq = str(train.iloc[row_idx].get("sequence_aa", ""))
        candidates = np.asarray([j for j in candidates.tolist() if str(train.iloc[int(j)].get("sequence_aa", "")) == row_seq], dtype=int)
        if candidates.size == 0:
            return []

    fp_i = fps[row_idx]
    scaf_i = scaffolds[row_idx]
    valid: List[int] = []
    for j in candidates.tolist():
        fp_j = fps[int(j)]
        scaf_j = scaffolds[int(j)]
        similar = False
        if fp_i is not None and fp_j is not None:
            similar = float(DataStructs.TanimotoSimilarity(fp_i, fp_j)) >= sim_threshold
        if not similar and scaf_i is not None and scaf_j is not None:
            similar = float(DataStructs.TanimotoSimilarity(scaf_i, scaf_j)) >= sim_threshold
        if similar:
            valid.append(int(j))
    return valid


def inject_cliffs(
    train: pd.DataFrame,
    task_type: str,
    fraction: float,
    rng: np.random.Generator,
    sim_threshold: float,
    fp_radius: int,
    fp_nbits: int,
    candidate_pool: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Inject activity-cliff style perturbations into a training frame."""
    y = _label_values(train)
    n_select = _n_from_fraction(len(train), fraction)
    idx = _sample_indices(len(train), n_select, rng)
    if idx.size == 0:
        return train.copy(), {"n_cliffs_modified": 0.0, "n_cliff_candidates": 0.0}

    smiles = train["smiles_clean"].astype(str).tolist()
    fps = morgan_fps(smiles, radius=fp_radius, n_bits=fp_nbits)
    scaffolds = scaffold_fps(smiles, radius=fp_radius, n_bits=fp_nbits)
    out_y = y.copy()
    sigma3 = label_sigma3(y)
    modified = 0
    candidates_seen = 0

    for row_idx in idx.tolist():
        neighbors = _valid_neighbor_indices(
            int(row_idx),
            train,
            fps,
            scaffolds,
            sim_threshold=sim_threshold,
            rng=rng,
            candidate_pool=candidate_pool,
        )
        if not neighbors:
            continue
        candidates_seen += len(neighbors)
        neighbor_idx = int(rng.choice(np.asarray(neighbors, dtype=int)))
        if task_type == "classification":
            out_y[int(row_idx)] = 1.0 - int(y[neighbor_idx])
        else:
            out_y[int(row_idx)] = _perturb_regression(np.asarray([y[neighbor_idx]]), sigma3, rng)[0]
        modified += 1

    return _set_label_values(train, out_y), {
        "n_cliffs_modified": float(modified),
        "n_cliff_candidates": float(candidates_seen),
    }


def inject_contamination(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fraction: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Inject train/test overlap by replacing selected training rows."""
    idx = _sample_indices(len(test), _n_from_fraction(len(test), fraction), rng)
    if idx.size == 0:
        return train.copy(), {"n_contamination_added": 0.0}
    copied = test.iloc[idx].copy()
    copied["split"] = "train"
    out = pd.concat([train, copied], ignore_index=True)
    return out, {"n_contamination_added": float(len(copied))}


def inject_random_label_noise(
    train: pd.DataFrame,
    fraction: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Randomly perturb selected training labels."""
    y = _label_values(train)
    idx = _sample_indices(len(train), _n_from_fraction(len(train), fraction), rng)
    finite = y[np.isfinite(y)]
    if idx.size == 0 or finite.size == 0:
        return train.copy(), {"n_random_labels_modified": 0.0}
    out_y = y.copy()
    out_y[idx] = rng.choice(finite, size=len(idx), replace=True)
    return _set_label_values(train, out_y), {"n_random_labels_modified": float(len(idx))}


def apply_noise(
    train: pd.DataFrame,
    test: pd.DataFrame,
    task_type: str,
    scenario: str,
    fraction: float,
    rng: np.random.Generator,
    sim_threshold: float = 0.9,
    fp_radius: int = 2,
    fp_nbits: int = 2048,
    cliff_candidate_pool: int = 256,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Apply one configured noise scenario to the training frame."""
    if scenario == "conflicts":
        return inject_conflicts(train, task_type, fraction, rng)
    if scenario == "cliffs":
        return inject_cliffs(
            train,
            task_type,
            fraction,
            rng,
            sim_threshold=sim_threshold,
            fp_radius=fp_radius,
            fp_nbits=fp_nbits,
            candidate_pool=cliff_candidate_pool,
        )
    if scenario == "contamination":
        return inject_contamination(train, test, fraction, rng)
    if scenario == "random_label_noise":
        return inject_random_label_noise(train, fraction, rng)
    raise ValueError(f"Unknown noise scenario: {scenario}")


def evaluate_classification(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute the classification metrics used by the experiment."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_score, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask].astype(int)
    yp = np.clip(yp[mask], 1e-7, 1.0 - 1e-7)
    out = {metric: float("nan") for metric in CLASSIFICATION_METRICS}
    out["n_eval_rows"] = float(len(yt))
    out["positive_rate"] = float(np.mean(yt)) if len(yt) else float("nan")
    if len(yt) == 0:
        return out

    pred = (yp >= 0.5).astype(int)
    out["balanced_accuracy"] = float(balanced_accuracy_score(yt, pred))
    out["accuracy"] = float(accuracy_score(yt, pred))
    out["f1"] = float(f1_score(yt, pred, zero_division=0))
    out["precision"] = float(precision_score(yt, pred, zero_division=0))
    out["recall"] = float(recall_score(yt, pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(yt, pred)) if len(np.unique(yt)) > 1 or len(np.unique(pred)) > 1 else 0.0
    out["brier_score"] = float(brier_score_loss(yt, yp))
    try:
        out["log_loss"] = float(log_loss(yt, yp, labels=[0, 1]))
    except Exception:
        out["log_loss"] = float("nan")
    if len(np.unique(yt)) >= 2:
        out["roc_auc"] = float(roc_auc_score(yt, yp))
        ap = float(average_precision_score(yt, yp))
        out["pr_auc"] = ap
        out["average_precision"] = ap
    return out


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the regression metrics used by the experiment."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    out = {metric: float("nan") for metric in REGRESSION_METRICS}
    out["n_eval_rows"] = float(len(yt))
    if len(yt) == 0:
        return out
    mse = float(mean_squared_error(yt, yp))
    out["mse"] = mse
    out["rmse"] = float(math.sqrt(mse))
    out["mae"] = float(mean_absolute_error(yt, yp))
    out["median_ae"] = float(median_absolute_error(yt, yp))
    if len(yt) >= 2:
        out["r2"] = float(r2_score(yt, yp))
        out["explained_variance"] = float(explained_variance_score(yt, yp))
        if float(np.std(yt)) > 0 and float(np.std(yp)) > 0:
            out["pearson"] = float(getattr(pearsonr(yt, yp), "statistic", pearsonr(yt, yp)[0]))
            out["spearman"] = float(getattr(spearmanr(yt, yp), "statistic", spearmanr(yt, yp)[0]))
            out["kendall_tau"] = float(getattr(kendalltau(yt, yp), "statistic", kendalltau(yt, yp)[0]))
    try:
        out["max_error"] = float(max_error(yt, yp))
    except Exception:
        out["max_error"] = float("nan")
    return out


def evaluate_predictions(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Dispatch metric computation based on task type."""
    if task_type == "classification":
        return evaluate_classification(y_true, y_pred)
    return evaluate_regression(y_true, y_pred)


def backend_availability() -> BackendAvailability:
    """Detect optional training backend availability."""
    have_lgbm = importlib.util.find_spec("lightgbm") is not None
    have_torch = importlib.util.find_spec("torch") is not None
    have_lightning = importlib.util.find_spec("lightning") is not None
    return BackendAvailability(have_lgbm=have_lgbm, have_torch=have_torch, have_lightning=have_lightning)


def validate_model_backends(models: Sequence[str], availability: Optional[BackendAvailability] = None) -> None:
    """Raise if requested models require unavailable backends."""
    availability = availability or backend_availability()
    missing: List[str] = []
    if "lgbm" in models and not availability.have_lgbm:
        missing.append("lightgbm")
    if "torch_mlp" in models:
        if not availability.have_torch:
            missing.append("torch")
        if not availability.have_lightning:
            missing.append("lightning")
    if missing:
        raise RuntimeError(
            "Missing required model dependencies: "
            + ", ".join(sorted(set(missing)))
            + ". Install project dependencies before running the requested models."
        )


def _normalize_mlp_accelerator(accelerator: str) -> str:
    value = str(accelerator).strip().lower()
    if value == "cuda":
        return "gpu"
    if value not in {"auto", "cpu", "gpu", "mps"}:
        raise ValueError("--mlp-accelerator must be one of: auto, cpu, gpu, cuda, mps")
    return value


def _lightning_devices_arg(raw: Any) -> Any:
    text = str(raw).strip()
    if text.lower() == "auto":
        return "auto"
    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if parts and all(part.lstrip("-").isdigit() for part in parts):
            return [int(part) for part in parts]
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def validate_accelerator_request(models: Sequence[str], args: argparse.Namespace) -> None:
    """Validate MLP accelerator settings before starting an experiment run."""
    if "torch_mlp" not in models:
        return
    accelerator = _normalize_mlp_accelerator(args.mlp_accelerator)
    if accelerator not in {"gpu", "mps"}:
        return
    try:
        import torch
    except Exception as exc:  # pragma: no cover - validated separately
        raise RuntimeError("torch is required to validate torch_mlp accelerator settings") from exc
    if accelerator == "gpu" and not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "torch_mlp requested --mlp-accelerator gpu, but torch.cuda.is_available() is False. "
            "Use a CUDA-enabled torch install/container or pass --mlp-accelerator cpu."
        )
    if accelerator == "mps":
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is None or not bool(mps_backend.is_available()):
            raise RuntimeError(
                "torch_mlp requested --mlp-accelerator mps, but torch.backends.mps.is_available() is False."
            )


def _fit_predict_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task_type: str,
    seed: int,
    n_jobs: int,
    rf_estimators: int,
) -> np.ndarray:
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=rf_estimators, random_state=seed, n_jobs=n_jobs)
        model.fit(X_train, np.rint(y_train).astype(int))
        proba = model.predict_proba(X_test)
        if proba.shape[1] == 1:
            return np.full(len(X_test), float(model.classes_[0]), dtype=float)
        pos_idx = int(np.where(model.classes_ == 1)[0][0]) if 1 in model.classes_ else -1
        return np.asarray(proba[:, pos_idx], dtype=float)
    model = RandomForestRegressor(n_estimators=rf_estimators, random_state=seed, n_jobs=n_jobs)
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=float)


def _fit_predict_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task_type: str,
    seed: int,
    n_jobs: int,
    n_estimators: int,
    learning_rate: float,
    device_type: str,
) -> np.ndarray:
    import lightgbm as lgb

    if task_type == "classification":
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
            device_type=device_type,
        )
        model.fit(X_train, np.rint(y_train).astype(int))
        proba = model.predict_proba(X_test)
        if proba.shape[1] == 1:
            return np.full(len(X_test), float(model.classes_[0]), dtype=float)
        pos_idx = int(np.where(model.classes_ == 1)[0][0]) if 1 in model.classes_ else -1
        return np.asarray(proba[:, pos_idx], dtype=float)
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
        device_type=device_type,
    )
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=float)


def _lightning_imports():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    try:
        import lightning.pytorch as pl
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("lightning is required for torch_mlp") from exc
    return torch, TensorDataset, DataLoader, pl


def _fit_predict_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task_type: str,
    seed: int,
    hidden_size: int,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    accelerator: str,
    devices: Any,
) -> np.ndarray:
    torch, TensorDataset, DataLoader, pl = _lightning_imports()
    accelerator = _normalize_mlp_accelerator(accelerator)
    trainer_devices = _lightning_devices_arg(devices)

    pl.seed_everything(seed, workers=True)
    scaler = StandardScaler(with_mean=False)
    X_tr = scaler.fit_transform(X_train).astype(np.float32)
    X_te = scaler.transform(X_test).astype(np.float32)
    y_tr = y_train.astype(np.float32)

    class LitMLP(pl.LightningModule):
        def __init__(self, in_dim: int) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            if task_type == "classification":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y)
            else:
                loss = torch.nn.functional.mse_loss(pred, y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = DataLoader(ds, batch_size=min(batch_size, max(1, len(ds))), shuffle=True)
    model = LitMLP(X_tr.shape[1])
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=trainer_devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
    )
    trainer.fit(model, loader)
    model.eval()
    device = model.device
    pred_loader = DataLoader(TensorDataset(torch.from_numpy(X_te)), batch_size=min(batch_size, max(1, len(X_te))))
    chunks = []
    with torch.no_grad():
        for (x_batch,) in pred_loader:
            chunks.append(model(x_batch.to(device)).detach().cpu().numpy())
    raw = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
    if task_type == "classification":
        return 1.0 / (1.0 + np.exp(-raw))
    return raw.astype(float)


def fit_predict_model(
    model_name: str,
    bundle: DatasetBundle,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    """Fit one supported model and return predictions for the test features."""
    y_train = _label_values(train)
    finite = np.isfinite(y_train)
    train = train.loc[finite].reset_index(drop=True)
    y_train = y_train[finite]
    if len(train) == 0:
        raise RuntimeError("No finite training labels after noise injection.")

    if bundle.task_type == "classification":
        raw_labels = _unique_finite_labels(y_train)
        if not _is_binary_label_set(raw_labels):
            raise RuntimeError(f"Non-binary classification labels after noise injection: {raw_labels}")
        labels = sorted(int(round(v)) for v in raw_labels)
        if not set(labels).issubset({0, 1}):
            raise RuntimeError(f"Non-binary classification labels after noise injection: {labels}")
        if len(labels) < 2:
            return np.full(len(test), float(labels[0]) if labels else 0.0, dtype=float)

    X_train = feature_matrix(train, bundle, args.fp_radius, args.fp_nbits, args.target_feature_bits)
    X_test = feature_matrix(test, bundle, args.fp_radius, args.fp_nbits, args.target_feature_bits)

    if model_name == "rf":
        return _fit_predict_rf(X_train, y_train, X_test, bundle.task_type, seed, args.n_jobs, args.rf_estimators)
    if model_name == "lgbm":
        return _fit_predict_lgbm(
            X_train,
            y_train,
            X_test,
            bundle.task_type,
            seed,
            args.n_jobs,
            args.lgbm_estimators,
            args.lgbm_learning_rate,
            args.lgbm_device_type,
        )
    if model_name == "torch_mlp":
        return _fit_predict_torch_mlp(
            X_train,
            y_train,
            X_test,
            bundle.task_type,
            seed,
            hidden_size=args.mlp_hidden_size,
            max_epochs=args.mlp_max_epochs,
            lr=args.mlp_lr,
            weight_decay=args.mlp_weight_decay,
            batch_size=args.mlp_batch_size,
            accelerator=args.mlp_accelerator,
            devices=args.mlp_devices,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _version_or_none(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except Exception:
        return None


def _scenario_seed(seed: int, dataset: str, scenario: str, fraction: float) -> int:
    token = f"{dataset}\x1f{scenario}\x1f{fraction:.12g}".encode("utf-8")
    return int(seed) + int(zlib.crc32(token) % 1_000_000)


def _empty_metric_row() -> Dict[str, float]:
    return {metric: float("nan") for metric in (*CLASSIFICATION_METRICS, *REGRESSION_METRICS)}


def run_experiment(args: argparse.Namespace) -> None:
    """Run all configured noise-generalization experiments."""
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    models = _parse_csv(args.models)
    unknown_models = sorted(set(models) - set(DEFAULT_MODELS))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}. Valid: {DEFAULT_MODELS}")
    scenarios = _parse_csv(args.scenarios)
    unknown_scenarios = sorted(set(scenarios) - set(DEFAULT_SCENARIOS))
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios: {unknown_scenarios}. Valid: {DEFAULT_SCENARIOS}")
    fractions = _parse_floats(args.fractions)
    seeds = _parse_ints(args.seeds)
    train_splits = tuple(_parse_csv(args.train_splits))
    dataset_filters = [] if str(args.datasets).strip().lower() == "all" else _parse_csv(args.datasets)

    validate_model_backends(models)
    validate_accelerator_request(models, args)
    datasets, manifest_rows = discover_noise_datasets(args.runs_root.resolve(), dataset_filters, skip_multitask=args.skip_multitask)
    if not datasets:
        raise RuntimeError("No matching datasets to run.")

    result_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    for ds in datasets:
        bundle = ds.bundle
        records = bundle.records.copy()
        train_base = _finite_label_frame(records[records["split"].isin(train_splits)].copy())
        test = _finite_label_frame(records[records["split"].eq("test")].copy())
        if train_base.empty or test.empty:
            error_rows.append(
                {
                    "family": bundle.family,
                    "dataset": bundle.dataset,
                    "task_type": bundle.task_type,
                    "model": "",
                    "scenario": "",
                    "fraction": np.nan,
                    "seed": np.nan,
                    "error": "empty train/test after filtering finite labels",
                }
            )
            continue
        y_test = _label_values(test)

        for seed in seeds:
            for scenario in scenarios:
                for fraction in fractions:
                    rng = np.random.default_rng(_scenario_seed(seed, bundle.dataset, scenario, fraction))
                    try:
                        noisy_train, noise_stats = apply_noise(
                            train_base,
                            test,
                            bundle.task_type,
                            scenario,
                            fraction,
                            rng,
                            sim_threshold=args.sim_threshold,
                            fp_radius=args.fp_radius,
                            fp_nbits=args.fp_nbits,
                            cliff_candidate_pool=args.cliff_candidate_pool,
                        )
                    except Exception as exc:
                        error_rows.append(
                            {
                                "family": bundle.family,
                                "dataset": bundle.dataset,
                                "task_type": bundle.task_type,
                                "model": "",
                                "scenario": scenario,
                                "fraction": fraction,
                                "seed": seed,
                                "error": f"noise_error: {exc}",
                            }
                        )
                        continue

                    for model in models:
                        base_row: Dict[str, Any] = {
                            "family": bundle.family,
                            "dataset": bundle.dataset,
                            "task_type": bundle.task_type,
                            "model": model,
                            "scenario": scenario,
                            "fraction": float(fraction),
                            "seed": int(seed),
                            "n_train_original": int(len(train_base)),
                            "n_train_noisy": int(len(noisy_train)),
                            "n_test": int(len(test)),
                            **noise_stats,
                        }
                        try:
                            pred = fit_predict_model(model, bundle, noisy_train, test, seed, args)
                            metrics = evaluate_predictions(bundle.task_type, y_test, pred)
                            result_rows.append({**base_row, **_empty_metric_row(), **metrics, "error": ""})
                        except Exception as exc:
                            error_rows.append({**base_row, "error": str(exc)})
                            result_rows.append({**base_row, **_empty_metric_row(), "error": str(exc)})

    results = pd.DataFrame(result_rows)
    errors = pd.DataFrame(error_rows)
    if errors.empty:
        errors = pd.DataFrame(columns=["family", "dataset", "task_type", "model", "scenario", "fraction", "seed", "error"])
    manifest = pd.DataFrame(manifest_rows)
    results.to_csv(out_dir / "results.csv", index=False)
    manifest.to_csv(out_dir / "dataset_manifest.csv", index=False)
    errors.to_csv(out_dir / "errors.csv", index=False)

    if results.empty:
        summary = pd.DataFrame()
    else:
        metric_cols = [c for c in (*CLASSIFICATION_METRICS, *REGRESSION_METRICS) if c in results.columns]
        ok = results[results["error"].isna() | (results["error"] == "")]
        summary = (
            ok.groupby(["task_type", "model", "scenario", "fraction"])[metric_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.columns = [
            "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
            for col in summary.columns
        ]
    summary.to_csv(out_dir / "summary_by_fraction.csv", index=False)

    metadata_payload = {
        "runs_root": str(args.runs_root.resolve()),
        "out_dir": str(out_dir),
        "datasets_arg": args.datasets,
        "included_datasets": [ds.bundle.dataset for ds in datasets],
        "models": models,
        "scenarios": scenarios,
        "fractions": fractions,
        "seeds": seeds,
        "train_splits": train_splits,
        "skip_multitask": bool(args.skip_multitask),
        "fp_radius": args.fp_radius,
        "fp_nbits": args.fp_nbits,
        "target_feature_bits": args.target_feature_bits,
        "sim_threshold": args.sim_threshold,
        "cliff_candidate_pool": args.cliff_candidate_pool,
        "rf_estimators": args.rf_estimators,
        "lgbm_estimators": args.lgbm_estimators,
        "lgbm_learning_rate": args.lgbm_learning_rate,
        "lgbm_device_type": args.lgbm_device_type,
        "mlp_hidden_size": args.mlp_hidden_size,
        "mlp_max_epochs": args.mlp_max_epochs,
        "mlp_lr": args.mlp_lr,
        "mlp_weight_decay": args.mlp_weight_decay,
        "mlp_batch_size": args.mlp_batch_size,
        "mlp_accelerator": _normalize_mlp_accelerator(args.mlp_accelerator),
        "mlp_devices": args.mlp_devices,
        "versions": {
            "lightgbm": _version_or_none("lightgbm"),
            "torch": _version_or_none("torch"),
            "lightning": _version_or_none("lightning"),
            "scikit-learn": _version_or_none("scikit-learn"),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    print(f"Wrote generalized noise-injection outputs to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for noise-generalization experiments."""
    parser = argparse.ArgumentParser(description="Run generalized synthetic noise-injection experiments.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/degradation/noise_generalization"))
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--fractions", type=str, default=",".join(str(x) for x in DEFAULT_FRACTIONS))
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--train-splits", type=str, default="train,valid")
    parser.add_argument("--skip-multitask", action="store_true")
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--fp-nbits", type=int, default=2048)
    parser.add_argument("--target-feature-bits", type=int, default=512)
    parser.add_argument("--sim-threshold", type=float, default=0.9)
    parser.add_argument("--cliff-candidate-pool", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--lgbm-estimators", type=int, default=400)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.05)
    parser.add_argument("--lgbm-device-type", type=str, choices=("cpu", "gpu", "cuda"), default="cpu")
    parser.add_argument("--mlp-hidden-size", type=int, default=100)
    parser.add_argument("--mlp-max-epochs", type=int, default=200)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-batch-size", type=int, default=200)
    parser.add_argument("--mlp-accelerator", type=str, default="gpu", help="Lightning accelerator for torch_mlp: gpu, cpu, auto, cuda, or mps.")
    parser.add_argument("--mlp-devices", type=str, default="1", help="Lightning devices setting for torch_mlp, e.g. 1, 0,1, or auto.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Execute the noise-generalization command line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_experiment(args)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"{parser.prog}: error: {exc}\n")


if __name__ == "__main__":
    main()
