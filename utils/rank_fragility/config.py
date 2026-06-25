"""Configuration dataclasses for rank-fragility analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Task = Literal["classification", "regression"]


@dataclass(frozen=True)
class AuditConfig:
    """Column names and thresholds used to annotate audited molecules."""

    id_col: str = "molecule_id"
    smiles_col: str = "smiles"
    label_col: str = "y"
    split_col: str = "split"
    task: Task = "classification"
    near_leak_thresholds: tuple[float, ...] = (0.85, 0.90)
    primary_near_leak_threshold: float = 0.85
    regression_conflict_threshold: float = 1.0
    regression_conflict_threshold_sensitivity: float | None = None
    random_seed: int = 13


@dataclass(frozen=True)
class PanelConfig:
    """Sampling controls for generated counterfactual evaluation panels."""

    id_col: str = "molecule_id"
    label_col: str = "y"
    task: Task = "classification"
    panel_size: int | str = "auto"
    n_panels: int = 1000
    target_rates: tuple[float | str, ...] = (0.0, 0.05, 0.10, 0.25, "observed", 0.50, 0.75)
    random_seed: int = 13
    output_dir: Path | str = Path("runs/rank_fragility")


@dataclass(frozen=True)
class MetricConfig:
    """Metric and model-selection settings for leaderboard comparisons."""

    task: Task = "classification"
    metric: str = "auroc"
    baseline_model: str = "ecfp_rf"
    sota_model: str = "auto"


@dataclass(frozen=True)
class RunConfig:
    """Complete input, audit, panel, and output settings for one analysis run."""

    data: Path
    pred_dir: Path
    id_col: str = "molecule_id"
    smiles_col: str = "smiles"
    label_col: str = "y"
    split_col: str = "split"
    task: Task = "classification"
    metric: str = "auroc"
    near_leak_thresholds: tuple[float, ...] = (0.85, 0.90)
    primary_near_leak_threshold: float = 0.85
    regression_conflict_threshold: float = 1.0
    regression_conflict_threshold_sensitivity: float | None = None
    random_seed: int = 13
    panel_size: int | str = "auto"
    n_panels: int = 1000
    target_rates: tuple[float | str, ...] = field(
        default_factory=lambda: (0.0, 0.05, 0.10, 0.25, "observed", 0.50, 0.75)
    )
    baseline_model: str = "ecfp_rf"
    sota_model: str = "auto"
    output_dir: Path = Path("runs/rank_fragility")

    def audit_config(self) -> AuditConfig:
        """Return the audit-specific subset of this run configuration."""
        return AuditConfig(
            id_col=self.id_col,
            smiles_col=self.smiles_col,
            label_col=self.label_col,
            split_col=self.split_col,
            task=self.task,
            near_leak_thresholds=self.near_leak_thresholds,
            primary_near_leak_threshold=self.primary_near_leak_threshold,
            regression_conflict_threshold=self.regression_conflict_threshold,
            regression_conflict_threshold_sensitivity=self.regression_conflict_threshold_sensitivity,
            random_seed=self.random_seed,
        )

    def panel_config(self) -> PanelConfig:
        """Return the panel-sampling subset of this run configuration."""
        return PanelConfig(
            id_col=self.id_col,
            label_col=self.label_col,
            task=self.task,
            panel_size=self.panel_size,
            n_panels=self.n_panels,
            target_rates=self.target_rates,
            random_seed=self.random_seed,
            output_dir=self.output_dir,
        )

    def metric_config(self) -> MetricConfig:
        """Return the metric/model subset of this run configuration."""
        return MetricConfig(
            task=self.task,
            metric=self.metric,
            baseline_model=self.baseline_model,
            sota_model=self.sota_model,
        )
