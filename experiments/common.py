from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Simple defaults; override in notebooks as needed
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = ROOT / "runs"
FIG_WIDTH_IN = 8.0
FIG_HEIGHT_IN = 4


@dataclass
class RunInfo:
    """Minimal metadata about a run folder."""

    name: str
    path: Path
    summary_path: Path
    records_path: Optional[Path]


def find_runs(run_root: Path | str = DEFAULT_RUNS_ROOT, include_dti: bool = False) -> List[RunInfo]:
    """Return all run folders with a summary.json under run_root."""
    root = Path(run_root).expanduser().resolve()
    runs: List[RunInfo] = []
    for summary_path in root.rglob("summary.json"):
        # Skip aggregate roll-ups (runs/runs/*/summary.json)
        if summary_path.parent.parent.name == "runs":
            continue
        if not include_dti and "dti" in {p.lower() for p in summary_path.parts}:
            continue
        run_dir = summary_path.parent
        records_path = run_dir / "records.csv"
        runs.append(
            RunInfo(
                name=run_dir.name,
                path=run_dir,
                summary_path=summary_path,
                records_path=records_path if records_path.exists() else None,
            )
        )
    runs.sort(key=lambda r: r.name.lower())
    return runs


def load_summary(run: RunInfo) -> Dict:
    with run.summary_path.open("r") as fh:
        return json.load(fh)


def load_records(run: RunInfo) -> pd.DataFrame:
    if run.records_path and run.records_path.exists():
        df = pd.read_csv(run.records_path)
    else:
        raise FileNotFoundError(f"No records.csv found in {run.path}")
    if "split" not in df.columns:
        df["split"] = "unknown"
    return df


def init_plots() -> None:
    """Set a lightweight publication-ish matplotlib/seaborn style."""
    sns.set_theme(context="paper")
    plt.rcParams["figure.figsize"] = (FIG_WIDTH_IN, FIG_HEIGHT_IN)
    sns.despine()


def format_latex_table(df: pd.DataFrame) -> str:
    """Compact LaTeX export for tables."""
    return df.to_latex(index=False, float_format=lambda x: f"{x:.3g}", na_rep="---")
