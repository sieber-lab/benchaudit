"""Prediction loading and audit-merge helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


PREDICTION_COLUMNS = {"molecule_id", "y_true", "y_pred"}


def load_prediction_directory(pred_dir: str) -> pd.DataFrame:
    """Load one prediction CSV per model into long format."""
    root = Path(pred_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"prediction directory not found: {pred_dir}")
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("*.csv")):
        frame = pd.read_csv(path)
        missing = sorted(PREDICTION_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required prediction column(s): {', '.join(missing)}")
        out = frame[["molecule_id", "y_true", "y_pred"]].copy()
        out["model"] = path.stem
        out["molecule_id"] = out["molecule_id"].astype(str)
        if out["molecule_id"].duplicated().any():
            dupes = out.loc[out["molecule_id"].duplicated(), "molecule_id"].head(5).tolist()
            raise ValueError(f"{path} contains duplicate molecule_id values, e.g. {dupes}")
        frames.append(out[["model", "molecule_id", "y_true", "y_pred"]])
    if not frames:
        raise ValueError(f"no prediction CSV files found in {pred_dir}")
    return pd.concat(frames, ignore_index=True)


def _labels_agree(pred: pd.Series, dataset: pd.Series, task: str) -> np.ndarray:
    if task == "regression":
        return np.isclose(
            pd.to_numeric(pred, errors="coerce"),
            pd.to_numeric(dataset, errors="coerce"),
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True,
        )
    return pred.map(_label_key).to_numpy() == dataset.map(_label_key).to_numpy()


def _label_key(value: Any) -> Any:
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
        return value.item()
    return value


def merge_predictions_with_audit(pred_df: pd.DataFrame, audited_df: pd.DataFrame, config) -> pd.DataFrame:
    """Validate prediction labels and attach audit annotations for test rows."""
    for column in PREDICTION_COLUMNS | {"model"}:
        if column not in pred_df.columns:
            raise ValueError(f"prediction dataframe is missing required column: {column}")

    test = audited_df[audited_df["split"].astype(str).str.lower() == "test"].copy()
    if test.empty:
        raise ValueError("audited dataset contains no test rows")
    if config.id_col not in test.columns or config.label_col not in test.columns:
        raise ValueError(f"audited dataframe must contain {config.id_col!r} and {config.label_col!r}")

    test["molecule_id"] = test[config.id_col].astype(str)
    test = test.drop_duplicates("molecule_id", keep="first")
    test_labels = test[["molecule_id", config.label_col]].rename(columns={config.label_col: "dataset_y_true"})

    merged_for_validation = pred_df.merge(test_labels, on="molecule_id", how="inner")
    if merged_for_validation.empty:
        raise ValueError("no prediction molecule_id values overlap test rows")
    agree = _labels_agree(merged_for_validation["y_true"], merged_for_validation["dataset_y_true"], config.task)
    if not bool(np.all(agree)):
        bad = merged_for_validation.loc[~agree, ["model", "molecule_id", "y_true", "dataset_y_true"]].head(10)
        raise ValueError(f"prediction y_true does not agree with dataset labels:\n{bad.to_string(index=False)}")

    all_test_ids = set(test["molecule_id"].tolist())
    model_to_ids = {model: set(group["molecule_id"].tolist()) for model, group in pred_df.groupby("model")}
    common_ids = set(all_test_ids)
    warnings_list: list[str] = []
    for model, ids in model_to_ids.items():
        missing = sorted(all_test_ids - ids)
        if missing:
            message = f"model {model} is missing {len(missing)} test prediction(s); dropping them for all models"
            LOG.warning(message)
            warnings_list.append(message)
        common_ids &= ids

    if not common_ids:
        raise ValueError("no common predicted test molecules across all models")

    dropped = len(all_test_ids - common_ids)
    if dropped:
        message = f"dropping {dropped} test molecule(s) absent from at least one model"
        LOG.warning(message)
        warnings_list.append(message)

    annotations = test.copy()
    annotations["molecule_id"] = annotations[config.id_col].astype(str)

    out = pred_df[pred_df["molecule_id"].isin(common_ids)].merge(annotations, on="molecule_id", how="left")
    out = out.sort_values(["model", "molecule_id"], kind="mergesort").reset_index(drop=True)
    out.attrs["warnings"] = warnings_list
    return out
