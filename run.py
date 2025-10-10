from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import numpy as np

from utils import build_loader, build_analyzer
from utils.baselines import run_baselines


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _json_default(o):
    """Safe JSON encoder for numpy/pandas types."""
    if o is None:
        return None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    for attr in ("model_dump", "dict", "to_dict", "as_dict"):
        if hasattr(o, attr) and callable(getattr(o, attr)):
            try:
                return getattr(o, attr)()
            except Exception:
                pass
    return str(o)


def echo_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["type", "name", "task", "modality", "info", "seed", "out"]
    return {k: cfg.get(k) for k in keys if k in cfg}


def discover_yaml_files(configs_dir: Optional[Path], single_config: Optional[Path]) -> List[Path]:
    files: List[Path] = []
    if single_config:
        if single_config.suffix.lower() in {".yml", ".yaml"}:
            files.append(single_config)
        else:
            raise ValueError(f"--config must be a YAML file: {single_config}")
    if configs_dir:
        files.extend(sorted([p for p in configs_dir.iterdir() if p.suffix.lower() in {".yml", ".yaml"}]))
    seen = set()
    uniq: List[Path] = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    return uniq


def resolve_out_dir(cfg: Dict[str, Any], out_root: Path) -> Path:
    if "out" in cfg and cfg["out"]:
        out = Path(cfg["out"])
        if not out.is_absolute():
            out = out_root / out
        return out
    typ = cfg.get("type", "unknown")
    name = cfg.get("name", "unnamed")
    return out_root / str(typ) / str(name)


def write_summary(out_dir: Path, summary: Dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, default=_json_default, indent=2))
    return path


def write_performance(out_dir: Path, performance: Dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "performance.json"
    path.write_text(json.dumps(performance, default=_json_default, indent=2))
    return path


def run_one_config(cfg: Dict[str, Any], out_root: Path, log: logging.Logger, do_benchmark: bool = False):
    typ = cfg.get("type", "unknown")
    name = cfg.get("name", "unnamed")
    out_dir = resolve_out_dir(cfg, out_root)

    log.info(f"run: {typ}/{name}")

    # 1) Load data (must provide at least train/test; valid optional)
    loader = build_loader(cfg)
    splits = loader.get_splits()
    if "train" not in splits or "test" not in splits:
        raise RuntimeError("loader must provide at least 'train' and 'test' splits")

    # 2) Analyze (hygiene, similarity, leakage checks, etc.)
    analyzer = build_analyzer(cfg)
    analysis_result = analyzer.run(splits)
    summary: Dict[str, Any] = {}
    if hasattr(analysis_result, "summary"):
        summary.update(analysis_result.summary)  # type: ignore

    # 3) Echo config
    summary["config"] = echo_config(cfg)

    # 4) Save summary.json
    spath = write_summary(out_dir, summary)
    log.info(f"saved: {spath}")

    # 5) Optional: baselines → performance.json
    if do_benchmark:
        try:
            perf = run_baselines(cfg, splits)
        except Exception as e:
            log.error(f"benchmark failed: {e}")
            perf = {"error": str(e)}
            
        ppath = write_performance(out_dir, perf)
        log.info(f"saved: {ppath}")


def _make_logger(level: str = "INFO") -> logging.Logger:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("runner")
    logger.setLevel(lvl)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(levelname)s | %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def main():
    ap = argparse.ArgumentParser(description="Benchmarking-benchmarks: run analyzer (+ optional baselines)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--configs", type=Path, help="Folder with YAML configs")
    g.add_argument("--config", type=Path, help="Single YAML config")
    ap.add_argument("--out-root", type=Path, default=Path("runs"), help="Output root folder")
    ap.add_argument("--benchmark", action="store_true",
                    help="Train baselines (train-only) and write performance.json")
    ap.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    args = ap.parse_args()

    log = _make_logger(args.log_level)

    files = discover_yaml_files(args.configs, args.config)
    if not files:
        log.info("no configs found")
        return

    for yml in files:
        try:
            cfg = load_yaml(yml)
            run_one_config(cfg, args.out_root, log, do_benchmark=args.benchmark)
        except Exception as e:
            log.error(f"failed: {yml.name}: {e}")


if __name__ == "__main__":
    main()
