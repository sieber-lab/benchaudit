"""Counterfactual evaluation-panel sampling utilities."""

from __future__ import annotations

import logging
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PanelConfig

LOG = logging.getLogger(__name__)


def _warn(message: str, collected: list[str]) -> None:
    LOG.warning(message)
    collected.append(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _rate_label(rate: float | str) -> str:
    if isinstance(rate, str):
        return rate
    return f"{float(rate):.6g}"


def _parse_rate(rate: float | str, observed: float) -> float:
    if isinstance(rate, str):
        if rate.lower() == "observed":
            return float(observed)
        return float(rate)
    return float(rate)


def _numeric_rate_for_output(rate: float | str) -> float | str:
    if isinstance(rate, str) and rate.lower() == "observed":
        return "observed"
    return float(rate)


def _strata(df: pd.DataFrame, task: str, label_col: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    if label_col not in df.columns:
        return pd.Series(["all"] * len(df), index=df.index, dtype=object)
    if task == "classification":
        return df[label_col].astype(str).fillna("missing")
    y = pd.to_numeric(df[label_col], errors="coerce")
    try:
        return pd.qcut(y.rank(method="first"), q=min(5, max(1, y.notna().sum())), duplicates="drop").astype(str)
    except Exception:
        return pd.Series(["all"] * len(df), index=df.index, dtype=object)


def _allocate_counts(pool: pd.DataFrame, n: int, task: str, label_col: str) -> dict[object, int]:
    if n <= 0:
        return {}
    strata = _strata(pool, task, label_col)
    counts = strata.value_counts()
    raw = counts / counts.sum() * n
    alloc = np.floor(raw).astype(int)
    remainder = int(n - alloc.sum())
    if remainder > 0:
        order = (raw - alloc).sort_values(ascending=False).index.tolist()
        for key in order[:remainder]:
            alloc.loc[key] += 1
    alloc = pd.concat([alloc, counts], axis=1)
    alloc.columns = ["requested", "available"]
    alloc["requested"] = alloc[["requested", "available"]].min(axis=1)
    deficit = int(n - alloc["requested"].sum())
    if deficit > 0:
        capacity = (alloc["available"] - alloc["requested"]).sort_values(ascending=False)
        for key, room in capacity.items():
            if deficit <= 0:
                break
            add = min(int(room), deficit)
            alloc.loc[key, "requested"] += add
            deficit -= add
    return {key: int(value) for key, value in alloc["requested"].items() if int(value) > 0}


def _sample_pool(pool: pd.DataFrame, n: int, rng: np.random.Generator, task: str, label_col: str) -> pd.DataFrame:
    if n <= 0:
        return pool.iloc[0:0].copy()
    if len(pool) < n:
        raise ValueError(f"cannot sample {n} unique molecules from pool of size {len(pool)}")
    if n == len(pool):
        return pool.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).copy()

    strata = _strata(pool, task, label_col)
    work = pool.assign(_stratum=strata)
    counts = _allocate_counts(work, n, task, label_col)
    pieces = []
    used: set[int] = set()
    for stratum, count in counts.items():
        group = work[work["_stratum"] == stratum]
        take = min(count, len(group))
        if take:
            sample = group.sample(n=take, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))
            pieces.append(sample)
            used.update(sample.index.tolist())
    selected = pd.concat(pieces, axis=0) if pieces else work.iloc[0:0].copy()
    deficit = n - len(selected)
    if deficit > 0:
        remaining = work.loc[~work.index.isin(used)]
        fill = remaining.sample(n=deficit, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))
        selected = pd.concat([selected, fill], axis=0)
    return selected.drop(columns=["_stratum"], errors="ignore").sample(
        frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))
    )


def _max_feasible_panel_size(pos_n: int, neg_n: int, rates: Iterable[float], total_cap: int) -> int:
    best = 0
    for n in range(int(total_cap), 0, -1):
        feasible = True
        for rate in rates:
            k = int(round(n * rate))
            if k > pos_n or (n - k) > neg_n:
                feasible = False
                break
        if feasible:
            best = n
            break
    return best


def _fixed_or_auto_size(config: PanelConfig, default: int) -> int:
    if isinstance(config.panel_size, str) and config.panel_size.lower() == "auto":
        return int(default)
    return int(config.panel_size)


def _panel_rows(panel_id: str, mode: str, target_rate, observed_rate: float, sample: pd.DataFrame, matched_mode=None):
    rows = []
    for mol_id in sample["molecule_id"].astype(str).tolist():
        rows.append(
            {
                "panel_id": panel_id,
                "mode": mode,
                "target_rate": target_rate,
                "observed_rate": float(observed_rate),
                "molecule_id": mol_id,
                "matched_mode": matched_mode,
            }
        )
    return rows


def _prepare_test_df(audited_test_df: pd.DataFrame, config: PanelConfig) -> pd.DataFrame:
    if config.id_col not in audited_test_df.columns:
        raise ValueError(f"audited_test_df is missing id column {config.id_col!r}")
    required = {"audit_group", "exact_train_test_leak", "near_train_analogue", "label_conflict"}
    missing = sorted(required - set(audited_test_df.columns))
    if missing:
        raise ValueError(f"audited_test_df is missing audit column(s): {', '.join(missing)}")
    out = audited_test_df.copy()
    out["molecule_id"] = out[config.id_col].astype(str)
    return out.drop_duplicates("molecule_id", keep="first").reset_index(drop=True)


def _sample_with_fallback(
    primary: pd.DataFrame,
    fallback: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    task: str,
    label_col: str,
) -> pd.DataFrame:
    if n <= len(primary):
        return _sample_pool(primary, n, rng, task, label_col)
    if len(primary) + len(fallback) < n:
        raise ValueError("not enough molecules in primary + fallback pools")
    primary_sample = _sample_pool(primary, len(primary), rng, task, label_col) if len(primary) else primary
    remaining_pool = fallback.loc[~fallback["molecule_id"].isin(primary_sample["molecule_id"])]
    fill = _sample_pool(remaining_pool, n - len(primary_sample), rng, task, label_col)
    return pd.concat([primary_sample, fill], axis=0)


def generate_counterfactual_panels(audited_test_df: pd.DataFrame, config: PanelConfig) -> pd.DataFrame:
    """Generate a long-form counterfactual panel manifest."""
    test = _prepare_test_df(audited_test_df, config)
    warnings_list: list[str] = []
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict] = []

    clean = test[test["audit_group"] == "audit_clean"].copy()
    n_clean = _fixed_or_auto_size(config, len(clean))
    if n_clean > 0 and len(clean) >= n_clean:
        for i in range(config.n_panels):
            sample = _sample_pool(clean, n_clean, rng, config.task, config.label_col)
            rows.extend(_panel_rows(f"clean_reference_{i:04d}", "clean_reference", 0.0, 0.0, sample))
    else:
        _warn("clean_reference is infeasible because there are no audit_clean test molecules", warnings_list)

    observed_size = _fixed_or_auto_size(config, len(test))
    if observed_size <= len(test) and observed_size > 0:
        observed_nonclean = float((test["audit_group"] != "audit_clean").mean()) if len(test) else np.nan
        groups = [g for _, g in test.groupby("audit_group", sort=True)]
        group_probs = np.asarray([len(g) for g in groups], dtype=float)
        group_probs = group_probs / group_probs.sum()
        for i in range(config.n_panels):
            desired = np.floor(group_probs * observed_size).astype(int)
            while desired.sum() < observed_size:
                desired[int(rng.choice(np.arange(len(desired)), p=group_probs))] += 1
            desired = np.minimum(desired, np.asarray([len(g) for g in groups]))
            deficit = observed_size - int(desired.sum())
            if deficit > 0:
                all_sample = _sample_pool(test, observed_size, rng, config.task, config.label_col)
            else:
                parts = [
                    _sample_pool(group, int(k), rng, config.task, config.label_col)
                    for group, k in zip(groups, desired)
                    if int(k) > 0
                ]
                all_sample = pd.concat(parts, axis=0) if parts else test.iloc[0:0]
            rows.extend(
                _panel_rows(
                    f"observed_composition_{i:04d}",
                    "observed_composition",
                    "observed",
                    observed_nonclean,
                    all_sample,
                )
            )
    else:
        _warn("observed_composition is infeasible for the requested panel_size", warnings_list)

    leakage_base = test[~test["label_conflict"].astype(bool)].copy()
    shortcut = leakage_base[
        leakage_base["exact_train_test_leak"].astype(bool) | leakage_base["near_train_analogue"].astype(bool)
    ].copy()
    leakage_clean = leakage_base[leakage_base["audit_group"] == "audit_clean"].copy()
    leakage_fallback = leakage_base[
        (~leakage_base["molecule_id"].isin(shortcut["molecule_id"]))
        & (~leakage_base["molecule_id"].isin(leakage_clean["molecule_id"]))
    ].copy()
    observed_shortcut_rate = float(len(shortcut) / len(leakage_base)) if len(leakage_base) else 0.0
    requested_leakage_rates = [_parse_rate(r, observed_shortcut_rate) for r in config.target_rates]
    leakage_size_auto = _max_feasible_panel_size(
        len(shortcut), len(leakage_clean) + len(leakage_fallback), requested_leakage_rates, len(leakage_base)
    )
    leakage_size = _fixed_or_auto_size(config, leakage_size_auto)
    feasible_leakage_rates: list[tuple[float | str, float]] = []
    for raw_rate in config.target_rates:
        rate = _parse_rate(raw_rate, observed_shortcut_rate)
        if not 0 <= rate <= 1:
            _warn(f"skipping leakage_curve target_rate={raw_rate}: rate must be between 0 and 1", warnings_list)
            continue
        k = int(round(leakage_size * rate))
        if leakage_size <= 0 or k > len(shortcut) or leakage_size - k > len(leakage_clean) + len(leakage_fallback):
            _warn(f"skipping leakage_curve target_rate={raw_rate}: infeasible with available pools", warnings_list)
            continue
        feasible_leakage_rates.append((_numeric_rate_for_output(raw_rate), rate))
        for i in range(config.n_panels):
            pos = _sample_pool(shortcut, k, rng, config.task, config.label_col)
            remaining = _sample_with_fallback(leakage_clean, leakage_fallback, leakage_size - k, rng, config.task, config.label_col)
            sample = pd.concat([pos, remaining], axis=0)
            observed = float(
                (
                    sample["exact_train_test_leak"].astype(bool) | sample["near_train_analogue"].astype(bool)
                ).mean()
            )
            rows.extend(
                _panel_rows(
                    f"leakage_curve_{_rate_label(raw_rate)}_{i:04d}",
                    "leakage_curve",
                    _numeric_rate_for_output(raw_rate),
                    observed,
                    sample,
                )
            )

    conflict = test[test["label_conflict"].astype(bool)].copy()
    conflict_bg_primary = test[(test["audit_group"] == "audit_clean") & (~test["exact_train_test_leak"].astype(bool))].copy()
    conflict_bg_fallback = test[
        (~test["label_conflict"].astype(bool))
        & (~test["molecule_id"].isin(conflict_bg_primary["molecule_id"]))
        & (~test["exact_train_test_leak"].astype(bool))
    ].copy()
    if len(conflict_bg_primary) + len(conflict_bg_fallback) == 0:
        conflict_bg_fallback = test[~test["label_conflict"].astype(bool)].copy()
    conflict_universe = pd.concat([conflict, conflict_bg_primary, conflict_bg_fallback], axis=0).drop_duplicates("molecule_id")
    observed_conflict_rate = float(len(conflict) / len(test)) if len(test) else 0.0
    requested_conflict_rates = [_parse_rate(r, observed_conflict_rate) for r in config.target_rates]
    conflict_size_auto = _max_feasible_panel_size(
        len(conflict), len(conflict_bg_primary) + len(conflict_bg_fallback), requested_conflict_rates, len(conflict_universe)
    )
    conflict_size = _fixed_or_auto_size(config, conflict_size_auto)
    feasible_conflict_rates: list[tuple[float | str, float]] = []
    for raw_rate in config.target_rates:
        rate = _parse_rate(raw_rate, observed_conflict_rate)
        if not 0 <= rate <= 1:
            _warn(f"skipping conflict_curve target_rate={raw_rate}: rate must be between 0 and 1", warnings_list)
            continue
        k = int(round(conflict_size * rate))
        if conflict_size <= 0 or k > len(conflict) or conflict_size - k > len(conflict_bg_primary) + len(conflict_bg_fallback):
            _warn(f"skipping conflict_curve target_rate={raw_rate}: infeasible with available pools", warnings_list)
            continue
        feasible_conflict_rates.append((_numeric_rate_for_output(raw_rate), rate))
        for i in range(config.n_panels):
            pos = _sample_pool(conflict, k, rng, config.task, config.label_col)
            remaining = _sample_with_fallback(
                conflict_bg_primary, conflict_bg_fallback, conflict_size - k, rng, config.task, config.label_col
            )
            sample = pd.concat([pos, remaining], axis=0)
            observed = float(sample["label_conflict"].astype(bool).mean())
            rows.extend(
                _panel_rows(
                    f"conflict_curve_{_rate_label(raw_rate)}_{i:04d}",
                    "conflict_curve",
                    _numeric_rate_for_output(raw_rate),
                    observed,
                    sample,
                )
            )

    for matched_mode, feasible_rates, size in [
        ("leakage_curve", feasible_leakage_rates, leakage_size),
        ("conflict_curve", feasible_conflict_rates, conflict_size),
    ]:
        if size <= 0 or len(test) < size:
            continue
        for raw_rate, _ in feasible_rates:
            for i in range(config.n_panels):
                sample = _sample_pool(test, size, rng, config.task, config.label_col)
                observed = float((sample["audit_group"] != "audit_clean").mean())
                rows.extend(
                    _panel_rows(
                        f"random_matched_control_{matched_mode}_{_rate_label(raw_rate)}_{i:04d}",
                        "random_matched_control",
                        raw_rate,
                        observed,
                        sample,
                        matched_mode=matched_mode,
                    )
                )

    manifest = pd.DataFrame(
        rows,
        columns=["panel_id", "mode", "target_rate", "observed_rate", "molecule_id", "matched_mode"],
    )
    if manifest.empty:
        _warn("no counterfactual panels were generated", warnings_list)
    manifest.attrs["warnings"] = warnings_list
    return manifest
