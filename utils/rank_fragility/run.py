"""Command-line driver for rank-fragility analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from .audit import audit_dataset, summarize_audit
from .attribution import compute_advantage_decomposition
from .config import RunConfig
from .counterfactual import run_counterfactual_evaluation
from .fragility import compute_fragility_summary
from .leaderboard import evaluate_models, original_leaderboard, rank_models
from .panels import generate_counterfactual_panels
from .predictions import load_prediction_directory, merge_predictions_with_audit


LOG = logging.getLogger("rank_fragility")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_panel_size(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--panel_size must be a positive integer or 'auto'")
    return parsed


def _parse_target_rates(values: Sequence[str]) -> tuple[float | str, ...]:
    rates: list[float | str] = []
    for value in values:
        if value.lower() == "observed":
            rates.append("observed")
        else:
            rates.append(float(value))
    return tuple(rates)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for single-run and batch analysis."""
    parser = argparse.ArgumentParser(description="Counterfactual benchmark-composition analysis")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--pred_dir", type=Path)
    parser.add_argument(
        "--from-runs-root",
        type=Path,
        help="Batch mode: discover existing BenchAudit records.csv artifacts under this runs root.",
    )
    parser.add_argument("--batch-out-dir", type=Path, default=Path("runs/rank_fragility"))
    parser.add_argument("--datasets", type=str, default="all", help="Batch dataset filter: all, name, or family/name CSV.")
    parser.add_argument("--skip-multitask", action="store_true", help="Batch mode: skip records with multitask labels.")
    parser.add_argument("--include-dti", action="store_true", help="Batch mode: include DTI pair datasets; default skips them.")
    parser.add_argument("--batch-models", type=str, default="ecfp_linear,ecfp_rf")
    parser.add_argument("--train-splits", type=str, default="train,valid")
    parser.add_argument("--classification-metric", default="auroc")
    parser.add_argument("--regression-metric", default="rmse")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--fp-nbits", type=int, default=2048)
    parser.add_argument("--lgbm-estimators", type=int, default=400)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.05)
    parser.add_argument("--lgbm-device-type", choices=["cpu", "gpu", "cuda"], default="cpu")
    parser.add_argument("--lgbm-basic-num-leaves", type=int, default=31)
    parser.add_argument("--lgbm-basic-min-child-samples", type=int, default=20)
    parser.add_argument("--lgbm-advanced-estimators", type=int, default=800)
    parser.add_argument("--lgbm-advanced-learning-rate", type=float, default=0.03)
    parser.add_argument("--lgbm-advanced-num-leaves", type=int, default=63)
    parser.add_argument("--lgbm-advanced-min-child-samples", type=int, default=10)
    parser.add_argument("--lgbm-advanced-subsample", type=float, default=0.8)
    parser.add_argument("--lgbm-advanced-colsample", type=float, default=0.8)
    parser.add_argument("--lgbm-advanced-reg-alpha", type=float, default=0.0)
    parser.add_argument("--lgbm-advanced-reg-lambda", type=float, default=1.0)
    parser.add_argument("--mlp-hidden-size", type=int, default=100)
    parser.add_argument("--mlp-max-epochs", type=int, default=200)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-batch-size", type=int, default=200)
    parser.add_argument("--mlp-accelerator", choices=["auto", "cpu", "gpu", "cuda", "mps"], default="auto")
    parser.add_argument("--mlp-devices", default="auto")
    parser.add_argument("--mlp-advanced-hidden-sizes", default="256,128")
    parser.add_argument("--mlp-advanced-max-epochs", type=int, default=300)
    parser.add_argument("--mlp-advanced-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-advanced-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-advanced-dropout", type=float, default=0.1)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--id_col", default="molecule_id")
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument("--label_col", default="y")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--task", choices=["classification", "regression"])
    parser.add_argument("--metric")
    parser.add_argument("--baseline_model", default="ecfp_linear")
    parser.add_argument("--sota_model", default="auto")
    parser.add_argument("--near_leak_thresholds", type=float, nargs="+", default=[0.85, 0.90])
    parser.add_argument("--primary_near_leak_threshold", type=float, default=0.85)
    parser.add_argument("--regression_conflict_threshold", type=float, default=1.0)
    parser.add_argument("--regression_conflict_threshold_sensitivity", type=float, default=None)
    parser.add_argument("--n_panels", type=int, default=1000)
    parser.add_argument("--panel_size", type=_parse_panel_size, default="auto")
    parser.add_argument(
        "--target_rates",
        nargs="+",
        default=["0", "0.05", "0.10", "0.25", "observed", "0.50", "0.75"],
    )
    parser.add_argument("--random_seed", type=int, default=13)
    parser.add_argument(
        "--out",
        "--out-dir",
        dest="output_dir",
        type=Path,
        default=Path("runs/rank_fragility"),
        help="Output directory (default: runs/rank_fragility).",
    )
    parser.add_argument("--log_level", default="INFO")
    return parser


def _resolve_sota(leaderboard: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in set(leaderboard["model"]):
            raise ValueError(f"sota_model not found in original leaderboard: {requested}")
        return requested
    valid = leaderboard.dropna(subset=["rank"]).sort_values(["rank", "model"])
    if valid.empty:
        raise ValueError("cannot resolve --sota_model auto because the original leaderboard has no valid ranks")
    return str(valid.iloc[0]["model"])


def _loss_for_advantage(task: str, metric: str) -> str:
    if task == "classification":
        return "brier" if metric == "brier" else "log_loss"
    return "absolute_error" if metric == "mae" else "squared_error"


def _write_csvs(output_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def run_analysis(config: RunConfig) -> dict[str, pd.DataFrame]:
    """Run one rank-fragility analysis and write its CSV outputs."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("loading dataset: %s", config.data)
    data = pd.read_csv(config.data)

    LOG.info("auditing dataset")
    audited = audit_dataset(data, config.audit_config())
    audit_summary = summarize_audit(audited)
    audit_summary.to_csv(out_dir / "audit_summary.csv", index=False)

    LOG.info("loading predictions: %s", config.pred_dir)
    predictions = load_prediction_directory(str(config.pred_dir))
    pred_audit = merge_predictions_with_audit(predictions, audited, config)

    LOG.info("computing original leaderboard")
    orig = original_leaderboard(pred_audit, task=config.task, metric=config.metric)
    sota_model = _resolve_sota(orig, config.sota_model)
    if config.baseline_model not in set(orig["model"]):
        raise ValueError(f"baseline_model not found in predictions: {config.baseline_model}")
    orig.to_csv(out_dir / "original_leaderboard.csv", index=False)

    clean_ids = audited[(audited["split"] == "test") & (audited["audit_group"] == "audit_clean")][config.id_col].astype(str)
    clean_scores = evaluate_models(pred_audit, clean_ids.tolist(), task=config.task, metric=config.metric)
    clean_leaderboard = rank_models(clean_scores, metric=config.metric).merge(clean_scores[["model", "n"]], on="model", how="left")
    clean_leaderboard = clean_leaderboard[["model", "n", "score", "rank"]] if not clean_leaderboard.empty else clean_leaderboard
    clean_leaderboard.to_csv(out_dir / "clean_reference_leaderboard.csv", index=False)

    LOG.info("generating counterfactual panels")
    test_audit = audited[audited["split"] == "test"].copy()
    panel_manifest = generate_counterfactual_panels(test_audit, config.panel_config())

    LOG.info("evaluating counterfactual panels")
    cf = run_counterfactual_evaluation(
        pred_audit,
        panel_manifest,
        task=config.task,
        metric=config.metric,
        baseline_model=config.baseline_model,
        sota_model=sota_model,
    )
    _write_csvs(
        out_dir,
        {
            "rank_probabilities": cf["rank_probabilities"],
            "sota_margin_by_composition": cf["sota_margin_by_composition"],
            "kendall_tau_by_composition": cf["kendall_tau_by_composition"],
        },
    )

    LOG.info("summarizing fragility")
    fragility_summary = compute_fragility_summary(
        cf["rank_probabilities"], cf["sota_margin_by_composition"], sota_model=sota_model
    )
    fragility_summary.to_csv(out_dir / "fragility_summary.csv", index=False)

    LOG.info("computing advantage decomposition")
    _per_example_advantage, advantage_decomposition = compute_advantage_decomposition(
        pred_audit,
        sota_model=sota_model,
        baseline_model=config.baseline_model,
        task=config.task,
        loss=_loss_for_advantage(config.task, config.metric),
    )
    advantage_decomposition.to_csv(out_dir / "advantage_decomposition.csv", index=False)

    outputs: dict[str, pd.DataFrame] = {
        "audit_summary": audit_summary,
        "original_leaderboard": orig,
        "clean_reference_leaderboard": clean_leaderboard,
        "rank_probabilities": cf["rank_probabilities"],
        "sota_margin_by_composition": cf["sota_margin_by_composition"],
        "kendall_tau_by_composition": cf["kendall_tau_by_composition"],
        "fragility_summary": fragility_summary,
        "advantage_decomposition": advantage_decomposition,
    }
    return outputs


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Convert parsed command-line arguments into a run configuration."""
    return RunConfig(
        data=args.data,
        pred_dir=args.pred_dir,
        id_col=args.id_col,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        split_col=args.split_col,
        task=args.task,
        metric=args.metric,
        near_leak_thresholds=tuple(args.near_leak_thresholds),
        primary_near_leak_threshold=args.primary_near_leak_threshold,
        regression_conflict_threshold=args.regression_conflict_threshold,
        regression_conflict_threshold_sensitivity=args.regression_conflict_threshold_sensitivity,
        random_seed=args.random_seed,
        panel_size=args.panel_size,
        n_panels=args.n_panels,
        target_rates=_parse_target_rates(args.target_rates),
        baseline_model=args.baseline_model,
        sota_model=args.sota_model,
        output_dir=args.output_dir,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Execute rank-fragility analysis from command-line arguments."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)
    if args.from_runs_root is not None:
        from .batch import run_batch_rank_fragility

        run_batch_rank_fragility(args)
        return
    if args.data is None or args.pred_dir is None or args.task is None or args.metric is None:
        parser.error("--data, --pred_dir, --task, and --metric are required unless --from-runs-root is used")
    run_analysis(config_from_args(args))
    LOG.info("completed analysis -> %s", args.output_dir)


if __name__ == "__main__":
    main()
