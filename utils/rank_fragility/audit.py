"""Molecular audit annotations for rank-fragility analysis."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .chem import max_tanimoto_to_train, morgan_fingerprint, murcko_scaffold_smiles, standardize_smiles
from .config import AuditConfig

LOG = logging.getLogger(__name__)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")


def _label_key(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, str):
        text = value.strip()
        try:
            f = float(text)
            return int(f) if f.is_integer() else f
        except ValueError:
            return text
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _classification_conflict(values: pd.Series) -> tuple[bool, float]:
    labels = {_label_key(v) for v in values.tolist()}
    labels.discard(None)
    return len(labels) > 1, float(max(len(labels) - 1, 0))


def _regression_conflict(values: pd.Series, threshold: float) -> tuple[bool, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return False, 0.0
    severity = float(numeric.max() - numeric.min())
    return severity > float(threshold), severity


def _group_columns(df: pd.DataFrame) -> list[str]:
    cols = ["canonical_smiles"]
    if "dataset" in df.columns:
        cols.append("dataset")
    if "task" in df.columns:
        cols.append("task")
    return cols


def _assign_audit_group(row: pd.Series) -> str:
    if str(row.get("split", "")).lower() != "test":
        return "not_test"
    if bool(row.get("label_conflict", False)):
        return "label_conflict"
    if bool(row.get("exact_train_test_leak", False)):
        return "exact_leak"
    if bool(row.get("near_train_analogue", False)):
        return "near_train_analogue"
    if bool(row.get("same_scaffold_as_train", False)):
        return "same_scaffold"
    return "audit_clean"


def audit_dataset(df: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Annotate dataset rows with chemistry and train-test audit flags."""
    _require_columns(df, [config.id_col, config.smiles_col, config.label_col, config.split_col])
    out = df.copy()
    out["split"] = out[config.split_col].astype(str).str.lower()
    out["audit_label"] = out[config.label_col]

    out["canonical_smiles"] = [standardize_smiles(smi) for smi in out[config.smiles_col].tolist()]
    out["valid_mol"] = out["canonical_smiles"].notna()
    out["murcko_scaffold"] = [
        murcko_scaffold_smiles(smi) if smi is not None else None for smi in out["canonical_smiles"].tolist()
    ]

    canonical_counts = out["canonical_smiles"].value_counts(dropna=True).to_dict()
    out["duplicate_count_full_dataset"] = out["canonical_smiles"].map(canonical_counts).fillna(0).astype(int)

    out["label_conflict"] = False
    out["conflict_severity"] = 0.0
    valid = out[out["canonical_smiles"].notna()]
    if not valid.empty:
        for _, idx in valid.groupby(_group_columns(valid), dropna=False).groups.items():
            labels = out.loc[idx, config.label_col]
            if config.task == "classification":
                conflict, severity = _classification_conflict(labels)
            else:
                conflict, severity = _regression_conflict(labels, config.regression_conflict_threshold)
            out.loc[idx, "label_conflict"] = bool(conflict)
            out.loc[idx, "conflict_severity"] = float(severity)

    train = out[out["split"] == "train"]
    test_mask = out["split"] == "test"
    train_smiles = set(train["canonical_smiles"].dropna().tolist())
    train_scaffolds = {s for s in train["murcko_scaffold"].dropna().tolist() if s}

    out["exact_train_test_leak"] = False
    out.loc[test_mask, "exact_train_test_leak"] = out.loc[test_mask, "canonical_smiles"].isin(train_smiles)

    out["max_train_tanimoto"] = np.nan
    if test_mask.any():
        train_fps = [morgan_fingerprint(smi) for smi in train["canonical_smiles"].dropna().tolist()]
        test_indices = out.index[test_mask].tolist()
        test_fps = [morgan_fingerprint(smi) if smi is not None else None for smi in out.loc[test_indices, "canonical_smiles"]]
        out.loc[test_indices, "max_train_tanimoto"] = max_tanimoto_to_train(test_fps, train_fps)

    out["near_train_analogue"] = False
    out.loc[test_mask, "near_train_analogue"] = (
        out.loc[test_mask, "max_train_tanimoto"] >= float(config.primary_near_leak_threshold)
    ).fillna(False)

    out["same_scaffold_as_train"] = False
    out.loc[test_mask, "same_scaffold_as_train"] = out.loc[test_mask, "murcko_scaffold"].isin(train_scaffolds)

    out["audit_group"] = out.apply(_assign_audit_group, axis=1)
    return out


def _count_fraction_rows(metric: str, count: int, denominator: int, group: str | None = None) -> dict[str, Any]:
    fraction = float(count / denominator) if denominator else np.nan
    return {
        "metric": metric,
        "group": group,
        "label": None,
        "count": int(count),
        "fraction": fraction,
        "value": float(count),
    }


def summarize_audit(audited_df: pd.DataFrame) -> pd.DataFrame:
    """Return a long-form audit summary table."""
    rows: list[dict[str, Any]] = []
    n_rows = len(audited_df)
    rows.append(_count_fraction_rows("rows_total", n_rows, n_rows))

    split_counts = audited_df["split"].value_counts(dropna=False)
    for split in ["train", "valid", "test"]:
        rows.append(_count_fraction_rows(f"rows_{split}", int(split_counts.get(split, 0)), n_rows, split))

    rows.append(_count_fraction_rows("valid_molecules", int(audited_df["valid_mol"].sum()), n_rows))

    test = audited_df[audited_df["split"] == "test"].copy()
    n_test = len(test)
    rows.extend(
        [
            _count_fraction_rows("test_label_conflicts", int(test["label_conflict"].sum()), n_test),
            _count_fraction_rows("exact_test_leaks", int(test["exact_train_test_leak"].sum()), n_test),
            _count_fraction_rows("near_train_analogues", int(test["near_train_analogue"].sum()), n_test),
            _count_fraction_rows("same_scaffold_test_molecules", int(test["same_scaffold_as_train"].sum()), n_test),
            _count_fraction_rows("audit_clean_test_molecules", int((test["audit_group"] == "audit_clean").sum()), n_test),
        ]
    )

    if not test.empty:
        group_counts = test["audit_group"].value_counts(dropna=False)
        for group, count in group_counts.items():
            rows.append(_count_fraction_rows("audit_group_count", int(count), n_test, str(group)))

        label_col = "audit_label" if "audit_label" in test.columns else ("y" if "y" in test.columns else None)
        if label_col in test.columns:
            for (group, label), count in test.groupby(["audit_group", label_col], dropna=False).size().items():
                denom = int((test["audit_group"] == group).sum())
                rows.append(
                    {
                        "metric": "label_distribution_by_audit_group",
                        "group": str(group),
                        "label": label,
                        "count": int(count),
                        "fraction": float(count / denom) if denom else np.nan,
                        "value": float(count),
                    }
                )

    return pd.DataFrame(rows, columns=["metric", "group", "label", "count", "fraction", "value"])
