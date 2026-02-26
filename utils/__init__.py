from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from .analysis import AnalysisResult, AnalyzerConfig, DTIAnalyzer, SMILESAnalyzer
from .baselines import run_baselines
from .config_models import normalize_runtime_config
from .loader import BaseLoader, DTILoader, PolarisLoader, TabularLoader, TDCLoader


LOGGER_NAME = "benchaudit"
_JSON_INDENT = 2
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")

__all__ = [
    "build_loader",
    "build_analyzer",
    "resolve_output_dir",
    "make_logger",
    "ResultWriter",
    "json_default",
    "run_baselines",
]


def json_default(value: Any):
    """Safe JSON encoder that understands numpy/pandas scalars."""
    import numpy as np  # Local import to avoid import-time overhead

    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    for attr in ("model_dump", "dict", "to_dict", "as_dict"):
        if hasattr(value, attr) and callable(getattr(value, attr)):
            try:
                return getattr(value, attr)()
            except Exception:
                continue
    return str(value)


def make_logger(name: str = LOGGER_NAME, level: str | int = "INFO") -> logging.Logger:
    """Return a logger with a consistent, informative format."""
    logger = logging.getLogger(name)
    level_value = level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(level_value)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def _slugify(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    slug = _SLUG_RE.sub("-", str(text).strip())
    slug = re.sub(r"-{2,}", "-", slug).strip("-._")
    return slug or None


def _preferred_name(cfg: Dict[str, Any], config_path: Optional[Path]) -> str:
    for candidate in (
        cfg.get("name"),
        cfg.get("id"),
        cfg.get("task"),
        config_path.stem if config_path else None,
    ):
        slug = _slugify(candidate)
        if slug:
            return slug
    return "run"


def _config_relative_parent(config_path: Optional[Path], configs_root: Optional[Path]) -> Optional[Path]:
    """Return the config's parent path relative to the configs root, if applicable."""
    if not config_path or not configs_root:
        return None
    try:
        rel_parent = config_path.resolve().parent.relative_to(configs_root.resolve())
    except Exception:
        return None
    return None if rel_parent == Path(".") else rel_parent


def _extend_with_structure(base_path: Path, rel_parent: Optional[Path]) -> Path:
    """Append the config's relative parent path (if any) onto the output base."""
    if not rel_parent or rel_parent == Path("."):
        return base_path

    base_parts = base_path.parts
    rel_parts = list(Path(rel_parent).parts)
    overlap = 0
    max_overlap = min(len(base_parts), len(rel_parts))
    for size in range(max_overlap, 0, -1):
        if base_parts[-size:] == tuple(rel_parts[:size]):
            overlap = size
            break

    if overlap:
        rel_parts = rel_parts[overlap:]
    if rel_parts:
        base_path = base_path.joinpath(*rel_parts)
    return base_path


def resolve_output_dir(
    cfg: Dict[str, Any],
    cli_out_root: Path,
    config_path: Optional[Path] = None,
    configs_root: Optional[Path] = None,
) -> Path:
    """Derive the output folder: <cfg['out'] or cli_root/type>/<relative-config-path>/<config-name>."""
    base = cfg.get("out")
    rel_parent = _config_relative_parent(config_path, configs_root)

    if rel_parent is not None:
        if base and Path(base).is_absolute():
            base_path = Path(base)
        else:
            base_path = cli_out_root / rel_parent
    else:
        if base:
            base_path = Path(base)
        else:
            type_slug = _slugify(cfg.get("type") or cfg.get("modality") or "unknown") or "unknown"
            base_path = cli_out_root / type_slug

    base_path = _extend_with_structure(base_path, rel_parent)

    if not base_path.is_absolute():
        base_path = (Path.cwd() / base_path).resolve()

    run_dir = base_path / _preferred_name(cfg, config_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_loader(cfg: Dict[str, Any]) -> BaseLoader:
    """Factory that instantiates the appropriate loader for the config."""
    cfg = normalize_runtime_config(cfg)
    typ = str(cfg.get("type") or cfg.get("modality") or "").lower()
    modality = str(cfg.get("modality") or "").lower()

    if modality == "dti" or typ == "dti":
        return DTILoader(cfg)
    if typ == "tdc":
        return TDCLoader(cfg)
    if typ == "polaris":
        return PolarisLoader(cfg)
    return TabularLoader(cfg)


def _normalize_label_cols(info: Dict[str, Any]) -> Optional[list[str]]:
    label_cols = info.get("label_cols")
    if label_cols is None:
        return None
    if isinstance(label_cols, str):
        return [label_cols]
    if isinstance(label_cols, Iterable):
        return [str(col) for col in label_cols]
    raise TypeError("info.label_cols must be a string or an iterable of strings")


def _analyzer_typ(cfg: Dict[str, Any]) -> str:
    typ = str(cfg.get("type") or cfg.get("modality") or "tabular").lower()
    if typ in {"tdc", "tabular", "polaris"}:
        return typ
    return "tabular"


def _build_analyzer_config(cfg: Dict[str, Any]) -> AnalyzerConfig:
    cfg = normalize_runtime_config(cfg)
    info = cfg.get("info", {}) or {}
    task = str(cfg.get("task") or "").lower()
    if task not in {"classification", "regression"}:
        raise ValueError("cfg.task must be 'classification' or 'regression'")

    unique_sequences_jsonl = info.get("unique_sequences_jsonl") or cfg.get("unique_sequences_jsonl")
    foldseek_m8_path = info.get("foldseek_m8_path") or cfg.get("foldseek_m8_path")

    return AnalyzerConfig(
        task_type="classification" if task == "classification" else "regression",
        typ=_analyzer_typ(cfg),
        sim_threshold=float(info.get("sim_threshold") or cfg.get("sim_threshold") or 0.9),
        fp_radius=int(info.get("fp_radius") or cfg.get("fp_radius") or 2),
        fp_nbits=int(info.get("fp_nbits") or cfg.get("fp_nbits") or 2048),
        smiles_col=info.get("smiles_col"),
        label_col=info.get("label_col"),
        id_col=info.get("id_col"),
        label_cols=_normalize_label_cols(info),
        sequence_col=info.get("sequence_col"),
        target_id_col=info.get("target_id_col"),
        name=str(cfg.get("name") or cfg.get("id") or "") or None,
        unique_sequences_jsonl=str(unique_sequences_jsonl) if unique_sequences_jsonl else None,
        foldseek_m8_path=str(foldseek_m8_path) if foldseek_m8_path else None,
    )


def build_analyzer(cfg: Dict[str, Any], logger: Optional[logging.Logger] = None):
    """Factory that picks the analyzer (SMILES vs DTI) and configures it."""
    cfg = normalize_runtime_config(cfg)
    modality = str(cfg.get("modality") or cfg.get("type") or "").lower()
    analyzer_cfg = _build_analyzer_config(cfg)
    log = logger or logging.getLogger(LOGGER_NAME)

    if modality == "dti":
        return DTIAnalyzer(analyzer_cfg, log)
    return SMILESAnalyzer(analyzer_cfg, log)


class ResultWriter:
    """Persist analyzer artifacts (summary, tables, drill-down files)."""

    def __init__(self, out_dir: Path, logger: Optional[logging.Logger] = None):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = logger or logging.getLogger(LOGGER_NAME)

    def write_summary(self, summary: Dict[str, Any]) -> Path:
        path = self.out_dir / "summary.json"
        path.write_text(json.dumps(summary, default=json_default, indent=_JSON_INDENT))
        self.log.info("saved summary -> %s", path)
        return path

    def write_performance(self, payload: Dict[str, Any]) -> Path:
        path = self.out_dir / "performance.json"
        path.write_text(json.dumps(payload, default=json_default, indent=_JSON_INDENT))
        self.log.info("saved baseline metrics -> %s", path)
        return path

    def _write_jsonl(self, rows: Iterable[Dict[str, Any]], filename: str) -> Optional[Path]:
        rows = list(rows)
        if not rows:
            self.log.info("skipped %s (no rows)", filename)
            return None
        path = self.out_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=json_default))
                fh.write("\n")
        self.log.info("saved %d %s rows -> %s", len(rows), filename, path)
        return path

    def write_records(self, df: pd.DataFrame) -> Optional[Path]:
        if df is None or df.empty:
            self.log.info("per-record table is empty; skipping records.csv")
            return None
        path = self.out_dir / "records.csv"
        df.to_csv(path, index=False)
        self.log.info("saved %d records -> %s", len(df), path)
        return path

    def write_analysis(self, result: AnalysisResult) -> Dict[str, Optional[Path]]:
        paths: Dict[str, Optional[Path]] = {}
        paths["summary"] = self.write_summary(result.summary)
        paths["records"] = self.write_records(result.per_record_df)
        paths["conflicts"] = self._write_jsonl(result.conflicts_rows, "conflicts.jsonl")
        paths["cliffs"] = self._write_jsonl(result.cliffs_rows, "cliffs.jsonl")
        if result.sequence_alignment_rows is not None:
            paths["sequence_alignments"] = self._write_jsonl(
                result.sequence_alignment_rows, "sequence_alignments.jsonl"
            )
        else:
            paths["sequence_alignments"] = None
        if result.structure_alignment_rows is not None:
            paths["structure_alignments"] = self._write_jsonl(
                result.structure_alignment_rows, "structure_alignments.jsonl"
            )
        else:
            paths["structure_alignments"] = None
        return paths
