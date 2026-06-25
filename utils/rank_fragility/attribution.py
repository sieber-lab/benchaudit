"""Advantage decomposition helpers for audited prediction tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import per_sample_loss


def _similarity_bins(values: pd.Series) -> pd.Series:
    bins = [0.0, 0.4, 0.6, 0.8, 0.9, 1.0000001]
    labels = ["[0, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 0.9)", "[0.9, 1.0]"]
    out = pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, right=False, include_lowest=True)
    return out.astype(object).where(out.notna(), "unknown").astype(str)


def _aggregate(per_example: pd.DataFrame, group_type: str, group_values: pd.Series, total_positive: float) -> list[dict]:
    rows: list[dict] = []
    tmp = per_example.assign(_group=group_values.astype(str))
    for group, frame in tmp.groupby("_group", dropna=False):
        positive = frame["advantage"].clip(lower=0).sum()
        rows.append(
            {
                "group_type": group_type,
                "group": str(group),
                "n": int(len(frame)),
                "total_advantage": float(frame["advantage"].sum()),
                "mean_advantage": float(frame["advantage"].mean()) if len(frame) else np.nan,
                "fraction_of_total_positive_advantage": float(positive / total_positive) if total_positive > 0 else np.nan,
            }
        )
    return rows


def compute_advantage_decomposition(
    pred_audit_df: pd.DataFrame,
    sota_model: str,
    baseline_model: str,
    task: str,
    loss: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-example SOTA advantage and aggregate by chemistry audit strata."""
    required = {"model", "molecule_id", "y_true", "y_pred", "audit_group", "max_train_tanimoto"}
    missing = sorted(required - set(pred_audit_df.columns))
    if missing:
        raise ValueError(f"pred_audit_df is missing required column(s): {', '.join(missing)}")

    sota = pred_audit_df[pred_audit_df["model"] == sota_model].copy()
    baseline = pred_audit_df[pred_audit_df["model"] == baseline_model].copy()
    if sota.empty:
        raise ValueError(f"sota_model not found in predictions: {sota_model}")
    if baseline.empty:
        raise ValueError(f"baseline_model not found in predictions: {baseline_model}")

    merged = sota[
        ["molecule_id", "y_true", "y_pred", "audit_group", "max_train_tanimoto"]
    ].rename(columns={"y_pred": "sota_pred"}).merge(
        baseline[["molecule_id", "y_pred"]].rename(columns={"y_pred": "baseline_pred"}),
        on="molecule_id",
        how="inner",
    )
    merged["loss_sota"] = per_sample_loss(merged["y_true"], merged["sota_pred"], task=task, loss=loss)
    merged["loss_baseline"] = per_sample_loss(merged["y_true"], merged["baseline_pred"], task=task, loss=loss)
    merged["advantage"] = merged["loss_baseline"] - merged["loss_sota"]

    per_example = merged[
        [
            "molecule_id",
            "audit_group",
            "y_true",
            "sota_pred",
            "baseline_pred",
            "loss_sota",
            "loss_baseline",
            "advantage",
            "max_train_tanimoto",
        ]
    ].copy()

    total_positive = float(per_example["advantage"].clip(lower=0).sum())
    rows: list[dict] = []
    rows.extend(_aggregate(per_example, "audit_group", per_example["audit_group"], total_positive))
    rows.extend(
        _aggregate(
            per_example,
            "clean_vs_nonclean",
            np.where(per_example["audit_group"] == "audit_clean", "audit_clean", "nonclean"),
            total_positive,
        )
    )
    rows.extend(_aggregate(per_example, "max_train_tanimoto_bin", _similarity_bins(per_example["max_train_tanimoto"]), total_positive))
    decomposition = pd.DataFrame(
        rows,
        columns=[
            "group_type",
            "group",
            "n",
            "total_advantage",
            "mean_advantage",
            "fraction_of_total_positive_advantage",
        ],
    )
    return per_example, decomposition
