"""Leaderboard scoring and ranking helpers."""

from __future__ import annotations

import pandas as pd

from .metrics import compute_metric, higher_is_better


def evaluate_models(pred_audit_df: pd.DataFrame, subset_ids, task: str, metric: str) -> pd.DataFrame:
    """Evaluate every model on a molecule-id subset."""
    if subset_ids is None:
        subset = pred_audit_df.copy()
    else:
        ids = set(map(str, subset_ids))
        subset = pred_audit_df[pred_audit_df["molecule_id"].astype(str).isin(ids)].copy()

    rows = []
    for model, group in subset.groupby("model", sort=True):
        rows.append(
            {
                "model": model,
                "n": int(len(group)),
                "score": compute_metric(group["y_true"], group["y_pred"], task=task, metric=metric),
            }
        )
    return pd.DataFrame(rows, columns=["model", "n", "score"])


def rank_models(scores_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Assign average ranks with rank 1 as best."""
    if scores_df.empty:
        return pd.DataFrame(columns=["model", "score", "rank"])
    out = scores_df[["model", "score"]].copy()
    ascending = not higher_is_better(metric)
    out["rank"] = out["score"].rank(method="average", ascending=ascending, na_option="bottom")
    return out.sort_values(["rank", "model"], kind="mergesort").reset_index(drop=True)


def original_leaderboard(pred_audit_df: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    """Evaluate and rank models on the full original test set."""
    subset_ids = pred_audit_df["molecule_id"].drop_duplicates().tolist()
    scores = evaluate_models(pred_audit_df, subset_ids, task=task, metric=metric)
    ranked = rank_models(scores, metric=metric)
    return ranked.merge(scores[["model", "n"]], on="model", how="left")[["model", "n", "score", "rank"]]
