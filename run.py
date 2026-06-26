"""CLI runner for BenchAudit dataset audits."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from utils import (
    ResultWriter,
    build_analyzer,
    build_loader,
    clean_benchmark_splits,
    make_logger,
    resolve_output_dir,
)
from utils.config_models import normalize_echo_config, normalize_runtime_config, validate_yaml_mapping
from utils.baselines import run_baselines


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dict."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return validate_yaml_mapping(data, source=path)


def echo_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a lightweight echo of the config for inclusion in summary.json."""
    cfg = normalize_echo_config(cfg)
    keys = ["type", "name", "task", "modality", "info", "seed", "out"]
    return {k: cfg.get(k) for k in keys if k in cfg}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _elapsed_seconds(start: float) -> float:
    return round(time.perf_counter() - start, 6)


def _benchmark_cleaning_options(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    setting = (cfg.get("info", {}) or {}).get("clean_benchmark", False)
    if setting is None or setting is False:
        return None
    if setting is True:
        return {}
    if isinstance(setting, str):
        token = setting.strip().lower()
        if token in {"", "false", "0", "no", "none", "null"}:
            return None
        if token in {"true", "1", "yes"}:
            return {}
        raise ValueError("info.clean_benchmark string value must be true or false")
    if not isinstance(setting, Mapping):
        raise TypeError("info.clean_benchmark must be a boolean or mapping")

    options = dict(setting)
    allowed = {"reference_splits", "remove_invalid", "remove_conflicts", "remove_contaminants"}
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"Unsupported info.clean_benchmark option(s): {unknown}")
    return options


def discover_yaml_files(configs_dir: Optional[Path], single_config: Optional[Path]) -> List[Path]:
    """Collect unique YAML files from a folder or a single path."""
    files: List[Path] = []
    if single_config:
        if not single_config.exists():
            raise FileNotFoundError(f"--config not found: {single_config}")
        if single_config.suffix.lower() in {".yml", ".yaml"}:
            files.append(single_config)
        else:
            raise ValueError(f"--config must be a YAML file: {single_config}")
    if configs_dir:
        if not configs_dir.exists():
            raise FileNotFoundError(f"--configs folder not found: {configs_dir}")
        if not configs_dir.is_dir():
            raise ValueError(f"--configs must be a directory: {configs_dir}")
        files.extend(
            sorted(
                p
                for p in configs_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in {".yml", ".yaml"}
            )
        )

    seen = set()
    uniq: List[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            uniq.append(path)
            seen.add(resolved)
    return uniq


def run_one_config(
    cfg: Dict[str, Any],
    config_path: Path,
    out_root: Path,
    log: logging.Logger,
    do_benchmark: bool = False,
    configs_root: Optional[Path] = None,
    force: bool = False,
) -> None:
    """Run the loader, analyzer, and optional baselines for a single config."""
    cfg = normalize_runtime_config(cfg)
    typ = cfg.get("type", "unknown")
    name = cfg.get("name", "unnamed")
    out_dir = resolve_output_dir(cfg, out_root, config_path=config_path, configs_root=configs_root)
    writer = ResultWriter(out_dir, log)

    if not force:
        expected = [out_dir / "summary.json"]
        if do_benchmark:
            expected.append(out_dir / "performance.json")
        if all(path.exists() for path in expected):
            log.info("skipping existing results -> %s (use --force to rerun)", out_dir)
            return

    log.info("run: %s/%s -> %s", typ, name, out_dir)

    run_started_at = _utc_now_iso()
    run_timer = time.perf_counter()
    stage_timings: Dict[str, float] = {}

    stage_timer = time.perf_counter()
    loader = build_loader(cfg)
    splits = loader.get_splits()
    stage_timings["load_splits_seconds"] = _elapsed_seconds(stage_timer)
    if "train" not in splits or "test" not in splits:
        raise RuntimeError("loader must provide at least 'train' and 'test' splits")

    cleaning_report: Optional[Dict[str, Any]] = None
    cleaning_options = _benchmark_cleaning_options(cfg)
    if cleaning_options is not None:
        stage_timer = time.perf_counter()
        splits, cleaning_report = clean_benchmark_splits(
            splits,
            str(cfg.get("task") or ""),
            **cleaning_options,
        )
        stage_timings["clean_benchmark_seconds"] = _elapsed_seconds(stage_timer)
        removed_total = cleaning_report["totals"]["removed"]
        log.info(
            "benchmark cleaning removed %d rows (%s)",
            removed_total,
            ", ".join(
                f"{reason}={cleaning_report['totals'][reason]}"
                for reason in ("invalid", "conflict", "contaminant")
            ),
        )

    split_sizes = {split_name: len(df) for split_name, df in splits.items()}
    log.info("splits: %s", ", ".join(f"{k}={v}" for k, v in split_sizes.items()))

    stage_timer = time.perf_counter()
    analyzer = build_analyzer(cfg, logger=log)
    analysis_result = analyzer.run(splits)
    stage_timings["analysis_seconds"] = _elapsed_seconds(stage_timer)
    summary: Dict[str, Any] = dict(getattr(analysis_result, "summary", {}))
    if cleaning_report is not None:
        summary["benchmark_cleaning"] = cleaning_report
    summary["config"] = echo_config(cfg)
    analysis_result.summary = summary

    stage_timer = time.perf_counter()
    writer.write_analysis(analysis_result, write_summary=False)
    stage_timings["write_analysis_artifacts_seconds"] = _elapsed_seconds(stage_timer)

    if do_benchmark:
        stage_timer = time.perf_counter()
        try:
            perf = run_baselines(cfg, splits)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("benchmark failed: %s", exc)
            perf = {"error": str(exc)}
        stage_timings["benchmark_seconds"] = _elapsed_seconds(stage_timer)

        stage_timer = time.perf_counter()
        writer.write_performance(perf)
        stage_timings["write_performance_seconds"] = _elapsed_seconds(stage_timer)

    total_elapsed = _elapsed_seconds(run_timer)
    summary["runtime"] = {
        "started_at_utc": run_started_at,
        "completed_at_utc": _utc_now_iso(),
        "elapsed_seconds": total_elapsed,
        "elapsed_minutes": round(total_elapsed / 60.0, 6),
        "stages": stage_timings,
    }
    analysis_result.summary = summary
    writer.write_summary(summary)
    log.info("completed %s/%s in %.2f seconds", typ, name, total_elapsed)


def main() -> None:
    """Parse CLI args and run one or more benchmark analyses."""
    parser = argparse.ArgumentParser(description="BenchAudit: run dataset analyzer (+ optional baselines)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--configs", type=Path, help="Folder with YAML configs")
    group.add_argument("--config", type=Path, help="Single YAML config")
    parser.add_argument("--out-root", type=Path, default=Path("runs"), help="Output root folder")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Train baselines (train-only) and write performance.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun configs even if outputs already exist (default is to skip)",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    args = parser.parse_args()

    log = make_logger("runner", args.log_level)

    configs_root = args.configs.resolve() if args.configs else None
    single_config = args.config.resolve() if args.config else None

    files = discover_yaml_files(configs_root, single_config)
    if not files:
        log.info("no configs found")
        return

    for yml in files:
        try:
            cfg = load_yaml(yml)
            run_one_config(
                cfg,
                yml,
                args.out_root,
                log,
                do_benchmark=args.benchmark,
                configs_root=configs_root,
                force=args.force,
            )
        except Exception as exc:
            log.error("failed: %s: %s", yml.name, exc)


if __name__ == "__main__":
    main()
