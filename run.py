from __future__ import annotations

"""CLI runner for bench (future AuditBench) dataset audits."""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from utils import ResultWriter, build_analyzer, build_loader, make_logger, resolve_output_dir
from utils.baselines import run_baselines


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dict."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def echo_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a lightweight echo of the config for inclusion in summary.json."""
    keys = ["type", "name", "task", "modality", "info", "seed", "out"]
    return {k: cfg.get(k) for k in keys if k in cfg}


def discover_yaml_files(configs_dir: Optional[Path], single_config: Optional[Path]) -> List[Path]:
    """Collect unique YAML files from a folder or a single path."""
    files: List[Path] = []
    if single_config:
        if single_config.suffix.lower() in {".yml", ".yaml"}:
            files.append(single_config)
        else:
            raise ValueError(f"--config must be a YAML file: {single_config}")
    if configs_dir:
        files.extend(sorted(p for p in configs_dir.iterdir() if p.suffix.lower() in {".yml", ".yaml"}))

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
) -> None:
    """Run the loader, analyzer, and optional baselines for a single config."""
    typ = cfg.get("type", "unknown")
    name = cfg.get("name", "unnamed")
    out_dir = resolve_output_dir(cfg, out_root, config_path=config_path)
    writer = ResultWriter(out_dir, log)

    log.info("run: %s/%s -> %s", typ, name, out_dir)

    loader = build_loader(cfg)
    splits = loader.get_splits()
    if "train" not in splits or "test" not in splits:
        raise RuntimeError("loader must provide at least 'train' and 'test' splits")

    split_sizes = {split_name: len(df) for split_name, df in splits.items()}
    log.info("splits: %s", ", ".join(f"{k}={v}" for k, v in split_sizes.items()))

    analyzer = build_analyzer(cfg, logger=log)
    analysis_result = analyzer.run(splits)
    summary: Dict[str, Any] = dict(getattr(analysis_result, "summary", {}))
    summary["config"] = echo_config(cfg)
    analysis_result.summary = summary
    writer.write_analysis(analysis_result)

    if do_benchmark:
        try:
            perf = run_baselines(cfg, splits)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("benchmark failed: %s", exc)
            perf = {"error": str(exc)}
        writer.write_performance(perf)


def main() -> None:
    """Parse CLI args and run one or more benchmark analyses."""
    parser = argparse.ArgumentParser(description="Bench: run dataset analyzer (+ optional baselines)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--configs", type=Path, help="Folder with YAML configs")
    group.add_argument("--config", type=Path, help="Single YAML config")
    parser.add_argument("--out-root", type=Path, default=Path("runs"), help="Output root folder")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Train baselines (train-only) and write performance.json",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    args = parser.parse_args()

    log = make_logger("runner", args.log_level)

    files = discover_yaml_files(args.configs, args.config)
    if not files:
        log.info("no configs found")
        return

    for yml in files:
        try:
            cfg = load_yaml(yml)
            run_one_config(cfg, yml, args.out_root, log, do_benchmark=args.benchmark)
        except Exception as exc:
            log.error("failed: %s: %s", yml.name, exc)


if __name__ == "__main__":
    main()
