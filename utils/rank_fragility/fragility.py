"""Summary helpers for composition-driven leaderboard fragility."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_target(value) -> float | None:
    try:
        if isinstance(value, str) and value.lower() == "observed":
            return None
        return float(value)
    except Exception:
        return None


def _smallest_numeric_rate(df: pd.DataFrame, predicate) -> float | None:
    candidates: list[float] = []
    for _, row in df.iterrows():
        rate = _numeric_target(row.get("target_rate"))
        if rate is None:
            continue
        try:
            if predicate(row):
                candidates.append(rate)
        except Exception:
            continue
    return min(candidates) if candidates else None


def compute_fragility_summary(
    rank_probabilities: pd.DataFrame,
    sota_margin_by_composition: pd.DataFrame,
    sota_model: str,
) -> pd.DataFrame:
    """Summarize composition rates where the SOTA conclusion becomes fragile."""
    modes = sorted(
        set(rank_probabilities.get("mode", pd.Series(dtype=object)).dropna().tolist())
        | set(sota_margin_by_composition.get("mode", pd.Series(dtype=object)).dropna().tolist())
    )
    rows: list[dict] = []
    for mode in modes:
        rp = rank_probabilities[(rank_probabilities["mode"] == mode) & (rank_probabilities["model"] == sota_model)]
        margins = sota_margin_by_composition[sota_margin_by_composition["mode"] == mode]
        observed_rate = "observed" if (
            (rp.get("target_rate", pd.Series(dtype=object)).astype(str) == "observed").any()
            or (margins.get("target_rate", pd.Series(dtype=object)).astype(str) == "observed").any()
        ) else np.nan

        rank_fragile_rate = _smallest_numeric_rate(
            rp, lambda row: pd.notna(row.get("probability_rank_1")) and row.get("probability_rank_1") < 0.5
        )
        ci_zero_rate = _smallest_numeric_rate(
            margins,
            lambda row: pd.notna(row.get("ci_lower"))
            and pd.notna(row.get("ci_upper"))
            and row.get("ci_lower") <= 0 <= row.get("ci_upper"),
        )
        fraction_rate = _smallest_numeric_rate(
            margins,
            lambda row: pd.notna(row.get("fraction_delta_positive")) and row.get("fraction_delta_positive") < 0.95,
        )

        if rp.empty and margins.empty:
            interpretation = "inconclusive due to infeasible or small panels"
        elif rank_fragile_rate is not None:
            interpretation = "rank-fragile"
        elif ci_zero_rate is not None or fraction_rate is not None:
            interpretation = "statistically indistinguishable"
        elif margins["mean_delta"].isna().all() if not margins.empty else False:
            interpretation = "inconclusive due to infeasible or small panels"
        else:
            interpretation = "stable"

        rows.append(
            {
                "mode": mode,
                "original_observed_rate": observed_rate,
                "smallest_target_rate_sota_rank_probability_below_0_5": rank_fragile_rate,
                "smallest_target_rate_sota_baseline_ci_includes_zero": ci_zero_rate,
                "smallest_target_rate_fraction_delta_positive_below_0_95": fraction_rate,
                "interpretation": interpretation,
                "interpretation_text": f"SOTA conclusion is {interpretation}.",
            }
        )
    return pd.DataFrame(rows)
