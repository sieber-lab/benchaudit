"""Opt-in cleaning utilities for curated benchmark splits."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd
from rdkit import Chem, rdBase

from .analysis import (
    _compute_sigma3,
    _delta_exceeds_threshold,
    _label_to_tuple,
    _regression_delta,
)


DEFAULT_REFERENCE_SPLITS: Tuple[str, ...] = ("train", "valid")
_REMOVAL_REASONS = ("invalid", "conflict", "contaminant")


def _normalize_reference_splits(reference_splits: Sequence[str] | str) -> Tuple[str, ...]:
    if isinstance(reference_splits, str):
        parts = [part.strip() for part in reference_splits.split(",")]
    else:
        parts = [str(part).strip() for part in reference_splits]
    normalized = tuple(part for part in parts if part)
    if not normalized:
        raise ValueError("reference_splits must contain at least one split name")
    return normalized


def _normalize_task_type(task_type: str) -> str:
    token = str(task_type).strip().lower()
    if token not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'")
    return token


def _smiles_column(df: pd.DataFrame) -> str:
    if "smiles_clean" in df.columns:
        return "smiles_clean"
    if "smiles" in df.columns:
        return "smiles"
    raise ValueError("Benchmark cleaning requires a 'smiles_clean' or 'smiles' column")


def _smiles_key(value: Any) -> Optional[str]:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _rdkit_valid_smiles(value: Any) -> bool:
    key = _smiles_key(value)
    if key is None:
        return False
    try:
        with rdBase.BlockLogs():
            return Chem.MolFromSmiles(key) is not None
    except Exception:
        return False


def _valid_flag(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y"}:
            return True
        if token in {"false", "0", "no", "n", "none", "null", "nan", ""}:
            return False
    return bool(value)


def _invalid_mask(df: pd.DataFrame) -> pd.Series:
    smiles_col = _smiles_column(df)
    invalid = ~df[smiles_col].map(_rdkit_valid_smiles)
    if "valid" in df.columns:
        invalid = invalid | ~df["valid"].map(_valid_flag)
    return invalid.fillna(True)


def _effective_reference_splits(
    splits: Mapping[str, pd.DataFrame],
    reference_splits: Tuple[str, ...],
) -> Tuple[str, ...]:
    return tuple(split for split in reference_splits if split in splits)


def _label_series_for_splits(
    splits: Mapping[str, pd.DataFrame],
    split_names: Iterable[str],
) -> pd.Series:
    series = [
        splits[split]["label_raw"]
        for split in split_names
        if split in splits and "label_raw" in splits[split]
    ]
    if not series:
        return pd.Series([], dtype=object)
    return pd.concat(series, ignore_index=True)


def _has_regression_conflict(labels: Sequence[Any], sigma3: Any) -> bool:
    if len(labels) < 2:
        return False
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if _delta_exceeds_threshold(_regression_delta(labels[i], labels[j]), sigma3):
                return True
    return False


def _conflicting_smiles(
    splits: Mapping[str, pd.DataFrame],
    task_type: str,
    reference_splits: Tuple[str, ...],
) -> set[str]:
    for split_name, df in splits.items():
        if "label_raw" not in df.columns:
            raise ValueError(f"Benchmark cleaning requires a 'label_raw' column in split '{split_name}'")

    by_smiles: Dict[str, list[Any]] = defaultdict(list)
    for df in splits.values():
        smiles_col = _smiles_column(df)
        for smi, label in zip(df[smiles_col].tolist(), df["label_raw"].tolist()):
            key = _smiles_key(smi)
            if key is not None:
                by_smiles[key].append(label)

    conflicts: set[str] = set()
    if task_type == "classification":
        for smi, labels in by_smiles.items():
            if len({_label_to_tuple(label) for label in labels}) > 1:
                conflicts.add(smi)
        return conflicts

    label_series = _label_series_for_splits(splits, reference_splits)
    sigma3, _ = _compute_sigma3(label_series)
    for smi, labels in by_smiles.items():
        if _has_regression_conflict(labels, sigma3):
            conflicts.add(smi)
    return conflicts


def _remove_smiles(splits: Dict[str, pd.DataFrame], smiles: set[str]) -> Dict[str, pd.Series]:
    masks: Dict[str, pd.Series] = {}
    for split_name, df in splits.items():
        smiles_col = _smiles_column(df)
        masks[split_name] = df[smiles_col].map(lambda value: _smiles_key(value) in smiles)
    return masks


def _apply_masks(
    splits: Dict[str, pd.DataFrame],
    masks: Mapping[str, pd.Series],
    removed_counts: Dict[str, Dict[str, int]],
    reason: str,
) -> Dict[str, pd.DataFrame]:
    cleaned: Dict[str, pd.DataFrame] = {}
    for split_name, df in splits.items():
        mask = masks.get(split_name)
        if mask is None:
            mask = pd.Series(False, index=df.index)
        mask = mask.reindex(df.index, fill_value=False).astype(bool)
        n_removed = int(mask.sum())
        removed_counts[split_name][reason] += n_removed
        removed_counts[split_name]["total"] += n_removed
        cleaned[split_name] = df.loc[~mask].reset_index(drop=True)
    return cleaned


def _contaminant_masks(
    splits: Mapping[str, pd.DataFrame],
    reference_splits: Tuple[str, ...],
) -> Tuple[Dict[str, pd.Series], set[str]]:
    reference_smiles: set[str] = set()
    for split_name in reference_splits:
        if split_name not in splits:
            continue
        df = splits[split_name]
        smiles_col = _smiles_column(df)
        reference_smiles.update(key for key in df[smiles_col].map(_smiles_key).tolist() if key is not None)

    masks: Dict[str, pd.Series] = {}
    contaminant_smiles: set[str] = set()
    reference_set = set(reference_splits)
    for split_name, df in splits.items():
        if split_name in reference_set:
            masks[split_name] = pd.Series(False, index=df.index)
            continue
        smiles_col = _smiles_column(df)
        mask = df[smiles_col].map(lambda value: _smiles_key(value) in reference_smiles)
        masks[split_name] = mask
        contaminant_smiles.update(
            key for key in df.loc[mask, smiles_col].map(_smiles_key).tolist() if key is not None
        )
    return masks, contaminant_smiles


def _empty_removed_counts(splits: Mapping[str, pd.DataFrame]) -> Dict[str, Dict[str, int]]:
    return {
        split_name: {**{reason: 0 for reason in _REMOVAL_REASONS}, "total": 0}
        for split_name in splits
    }


def clean_benchmark_splits(
    splits: Mapping[str, pd.DataFrame],
    task_type: str,
    *,
    reference_splits: Sequence[str] | str = DEFAULT_REFERENCE_SPLITS,
    remove_invalid: bool = True,
    remove_conflicts: bool = True,
    remove_contaminants: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Return cleaned benchmark splits and a JSON-serializable cleaning report.

    Cleaning is intentionally opt-in and operates only on in-memory split frames.
    Removal precedence is invalid rows, label-conflicting molecules, then exact
    contaminants in non-reference splits.
    """
    task = _normalize_task_type(task_type)
    reference_splits_norm = _normalize_reference_splits(reference_splits)
    working = {split_name: df.copy().reset_index(drop=True) for split_name, df in splits.items()}
    original_counts = {split_name: int(len(df)) for split_name, df in working.items()}
    removed_counts = _empty_removed_counts(working)
    conflict_smiles: set[str] = set()
    contaminant_smiles: set[str] = set()

    effective_reference_splits = _effective_reference_splits(working, reference_splits_norm)
    if not effective_reference_splits:
        raise ValueError("None of the requested reference_splits are present in the provided splits")

    if remove_invalid:
        invalid_masks = {split_name: _invalid_mask(df) for split_name, df in working.items()}
        working = _apply_masks(working, invalid_masks, removed_counts, "invalid")

    if remove_conflicts:
        conflict_smiles = _conflicting_smiles(working, task, effective_reference_splits)
        working = _apply_masks(working, _remove_smiles(working, conflict_smiles), removed_counts, "conflict")

    if remove_contaminants:
        contaminant_masks, contaminant_smiles = _contaminant_masks(working, effective_reference_splits)
        working = _apply_masks(working, contaminant_masks, removed_counts, "contaminant")

    cleaned_counts = {split_name: int(len(df)) for split_name, df in working.items()}
    totals = {
        reason: int(sum(split_counts[reason] for split_counts in removed_counts.values()))
        for reason in _REMOVAL_REASONS
    }
    totals["removed"] = int(sum(split_counts["total"] for split_counts in removed_counts.values()))

    report: Dict[str, Any] = {
        "options": {
            "reference_splits": list(reference_splits_norm),
            "effective_reference_splits": list(effective_reference_splits),
            "remove_invalid": bool(remove_invalid),
            "remove_conflicts": bool(remove_conflicts),
            "remove_contaminants": bool(remove_contaminants),
        },
        "original_counts": original_counts,
        "cleaned_counts": cleaned_counts,
        "removed_counts": removed_counts,
        "totals": totals,
        "n_conflict_smiles": int(len(conflict_smiles)),
        "n_contaminant_smiles": int(len(contaminant_smiles)),
    }
    return working, report
