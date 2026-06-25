"""Counterfactual panel evaluation and aggregate stability summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .leaderboard import evaluate_models, original_leaderboard, rank_models
from .metrics import higher_is_better


def _delta(sota_score: float, baseline_score: float, metric: str) -> float:
    if pd.isna(sota_score) or pd.isna(baseline_score):
        return float("nan")
    return float(sota_score - baseline_score) if higher_is_better(metric) else float(baseline_score - sota_score)


def _ci(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan"), float("nan")
    return float(clean.quantile(0.025)), float(clean.quantile(0.975))


def run_counterfactual_evaluation(
    pred_audit_df: pd.DataFrame,
    panel_manifest: pd.DataFrame,
    task: str,
    metric: str,
    baseline_model: str,
    sota_model: str,
) -> dict[str, pd.DataFrame]:
    """Evaluate all models on each counterfactual panel and aggregate stability summaries."""
    if panel_manifest.empty:
        empty = pd.DataFrame()
        return {
            "counterfactual_scores": empty,
            "counterfactual_ranks": empty,
            "sota_margin_by_panel": empty,
            "kendall_tau_by_panel": empty,
            "rank_probabilities": empty,
            "sota_margin_by_composition": empty,
            "kendall_tau_by_composition": empty,
        }

    original = original_leaderboard(pred_audit_df, task=task, metric=metric)
    original_rank = original.set_index("model")["rank"]

    score_frames: list[pd.DataFrame] = []
    rank_frames: list[pd.DataFrame] = []
    margin_rows: list[dict] = []
    tau_rows: list[dict] = []

    panel_meta = panel_manifest[["panel_id", "mode", "target_rate"]].drop_duplicates("panel_id")
    for panel_id, panel in panel_manifest.groupby("panel_id", sort=False):
        ids = panel["molecule_id"].astype(str).tolist()
        mode = panel["mode"].iloc[0]
        target_rate = panel["target_rate"].iloc[0]

        scores = evaluate_models(pred_audit_df, ids, task=task, metric=metric)
        scores.insert(0, "target_rate", target_rate)
        scores.insert(0, "mode", mode)
        scores.insert(0, "panel_id", panel_id)
        score_frames.append(scores)

        ranks = rank_models(scores[["model", "score"]], metric=metric)
        ranks.insert(0, "target_rate", target_rate)
        ranks.insert(0, "mode", mode)
        ranks.insert(0, "panel_id", panel_id)
        rank_frames.append(ranks)

        score_map = scores.set_index("model")["score"]
        margin_rows.append(
            {
                "panel_id": panel_id,
                "mode": mode,
                "target_rate": target_rate,
                "sota_model": sota_model,
                "baseline_model": baseline_model,
                "delta": _delta(score_map.get(sota_model, np.nan), score_map.get(baseline_model, np.nan), metric),
                "positive_means_sota_better": True,
            }
        )

        panel_rank = ranks.set_index("model")["rank"]
        common = sorted(set(original_rank.index) & set(panel_rank.index))
        if len(common) < 2:
            tau = np.nan
        else:
            tau = kendalltau(original_rank.loc[common].to_numpy(), panel_rank.loc[common].to_numpy()).statistic
        tau_rows.append(
            {
                "panel_id": panel_id,
                "mode": mode,
                "target_rate": target_rate,
                "kendall_tau_to_original": float(tau) if tau is not None else np.nan,
            }
        )

    counterfactual_scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    counterfactual_ranks = pd.concat(rank_frames, ignore_index=True) if rank_frames else pd.DataFrame()
    sota_margin_by_panel = pd.DataFrame(margin_rows)
    kendall_tau_by_panel = pd.DataFrame(tau_rows)

    rank_rows = []
    for (mode, target_rate, model), group in counterfactual_ranks.groupby(["mode", "target_rate", "model"], dropna=False):
        ranks = pd.to_numeric(group["rank"], errors="coerce")
        rank_rows.append(
            {
                "mode": mode,
                "target_rate": target_rate,
                "model": model,
                "probability_rank_1": float((ranks == 1.0).mean()) if len(ranks) else np.nan,
                "median_rank": float(ranks.median()) if len(ranks) else np.nan,
                "rank_q025": float(ranks.quantile(0.025)) if len(ranks) else np.nan,
                "rank_q975": float(ranks.quantile(0.975)) if len(ranks) else np.nan,
            }
        )
    rank_probabilities = pd.DataFrame(rank_rows)

    margin_rows_agg = []
    for (mode, target_rate), group in sota_margin_by_panel.groupby(["mode", "target_rate"], dropna=False):
        lower, upper = _ci(group["delta"])
        delta = pd.to_numeric(group["delta"], errors="coerce")
        margin_rows_agg.append(
            {
                "mode": mode,
                "target_rate": target_rate,
                "mean_delta": float(delta.mean()) if len(delta) else np.nan,
                "median_delta": float(delta.median()) if len(delta) else np.nan,
                "ci_lower": lower,
                "ci_upper": upper,
                "fraction_delta_positive": float((delta > 0).mean()) if len(delta) else np.nan,
            }
        )
    sota_margin_by_composition = pd.DataFrame(margin_rows_agg)

    tau_rows_agg = []
    for (mode, target_rate), group in kendall_tau_by_panel.groupby(["mode", "target_rate"], dropna=False):
        lower, upper = _ci(group["kendall_tau_to_original"])
        values = pd.to_numeric(group["kendall_tau_to_original"], errors="coerce")
        tau_rows_agg.append(
            {
                "mode": mode,
                "target_rate": target_rate,
                "mean_kendall_tau": float(values.mean()) if len(values) else np.nan,
                "median_kendall_tau": float(values.median()) if len(values) else np.nan,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    kendall_tau_by_composition = pd.DataFrame(tau_rows_agg)

    return {
        "counterfactual_scores": counterfactual_scores,
        "counterfactual_ranks": counterfactual_ranks,
        "sota_margin_by_panel": sota_margin_by_panel,
        "kendall_tau_by_panel": kendall_tau_by_panel,
        "rank_probabilities": rank_probabilities,
        "sota_margin_by_composition": sota_margin_by_composition,
        "kendall_tau_by_composition": kendall_tau_by_composition,
    }
