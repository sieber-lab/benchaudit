#!/usr/bin/env python3
"""Prepare Leak-Proof PDBBind split CSVs for BenchAudit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd


SOURCE_URL = "https://github.com/THGLab/LP-PDBBind/raw/refs/heads/master/dataset/LP_PDBBind.csv"
DEFAULT_OUT_DIR = Path("data/LeakProofPDBBind/CL1_non_covalent")
EXPECTED_DEFAULT_COUNTS = {"train": 7393, "valid": 1891, "test": 4250}

REQUIRED_COLUMNS = [
    "header",
    "smiles",
    "category",
    "seq",
    "resolution",
    "date",
    "type",
    "new_split",
    "CL1",
    "CL2",
    "CL3",
    "remove_for_balancing_val",
    "kd/ki",
    "value",
    "covalent",
]

OUTPUT_COLUMNS = [
    "ID",
    "pdbid",
    "Ligand",
    "Protein",
    "regression_label",
    "lp_header",
    "lp_category",
    "resolution",
    "date",
    "lp_type",
    "new_split",
    "CL1",
    "CL2",
    "CL3",
    "remove_for_balancing_val",
    "binding_measure",
    "covalent",
]

SPLIT_MAP = {
    "train": "train",
    "val": "valid",
    "valid": "valid",
    "test": "test",
}


def _is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"}


def _read_source(source: str) -> tuple[pd.DataFrame, str]:
    if _is_url(source):
        with urlopen(source) as response:
            payload = response.read()
    else:
        payload = Path(source).read_bytes()
    source_sha256 = hashlib.sha256(payload).hexdigest()
    return pd.read_csv(BytesIO(payload), index_col=0), source_sha256


def _require_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"LP-PDBBind source is missing required column(s): {', '.join(missing)}")
    if df.index.hasnans:
        raise ValueError("LP-PDBBind source index contains missing pdbid values")
    if df.index.astype(str).duplicated().any():
        raise ValueError("LP-PDBBind source index contains duplicated pdbid values")


def _coerce_bool(value: Any, *, column: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "t", "1", "yes", "y"}:
            return True
        if token in {"false", "f", "0", "no", "n", ""}:
            return False
    raise ValueError(f"Cannot interpret {value!r} as boolean in column {column!r}")


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].map(lambda value: _coerce_bool(value, column=column)).astype(bool)


def prepare_dataframe(df: pd.DataFrame, *, clean_level: str = "CL1", include_covalent: bool = False) -> pd.DataFrame:
    """Return a normalized LP-PDBBind dataframe ready to split for BenchAudit."""
    _require_columns(df)
    if clean_level not in {"CL1", "CL2", "CL3"}:
        raise ValueError("clean_level must be one of CL1, CL2, or CL3")

    source = df.copy()
    for bool_col in ["CL1", "CL2", "CL3", "remove_for_balancing_val", "covalent"]:
        source[bool_col] = _bool_series(source, bool_col)

    raw_split = source["new_split"].astype("string").str.strip().str.lower()
    canonical_split = raw_split.map(SPLIT_MAP)
    split_mask = canonical_split.notna()
    covalent_mask = pd.Series(True, index=source.index) if include_covalent else ~source["covalent"]
    filtered = source[split_mask & source[clean_level] & covalent_mask].copy()
    filtered["_split"] = canonical_split.loc[filtered.index].astype(str)

    filtered["regression_label"] = pd.to_numeric(filtered["value"], errors="coerce")
    core_missing = {
        "smiles": int(filtered["smiles"].isna().sum()),
        "seq": int(filtered["seq"].isna().sum()),
        "regression_label": int(filtered["regression_label"].isna().sum()),
    }
    bad_core = {key: value for key, value in core_missing.items() if value}
    if bad_core:
        raise ValueError(f"Filtered LP-PDBBind rows contain missing core values: {bad_core}")

    pdbids = pd.Index(filtered.index.astype(str).str.lower(), name="pdbid")
    prepared = pd.DataFrame(
        {
            "ID": [f"id_{pdbid}" for pdbid in pdbids],
            "pdbid": pdbids,
            "Ligand": filtered["smiles"].astype(str).tolist(),
            "Protein": filtered["seq"].astype(str).tolist(),
            "regression_label": filtered["regression_label"].astype(float).tolist(),
            "lp_header": filtered["header"].tolist(),
            "lp_category": filtered["category"].tolist(),
            "resolution": filtered["resolution"].tolist(),
            "date": filtered["date"].tolist(),
            "lp_type": filtered["type"].tolist(),
            "new_split": filtered["new_split"].tolist(),
            "CL1": filtered["CL1"].tolist(),
            "CL2": filtered["CL2"].tolist(),
            "CL3": filtered["CL3"].tolist(),
            "remove_for_balancing_val": filtered["remove_for_balancing_val"].tolist(),
            "binding_measure": filtered["kd/ki"].tolist(),
            "covalent": filtered["covalent"].tolist(),
            "_split": filtered["_split"].tolist(),
        }
    )
    if prepared["ID"].duplicated().any():
        raise ValueError("Prepared LP-PDBBind IDs are not unique")
    return prepared


def _write_split_files(prepared: pd.DataFrame, out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        split_df = prepared[prepared["_split"] == split].copy()
        if split_df.empty:
            raise ValueError(f"Prepared LP-PDBBind data has no rows for split {split!r}")
        split_df[OUTPUT_COLUMNS].to_csv(out_dir / f"{split}.csv", index=False)
        counts[split] = int(len(split_df))
    return counts


def _validate_expected_counts(counts: dict[str, int], *, clean_level: str, include_covalent: bool) -> None:
    if clean_level != "CL1" or include_covalent:
        return
    if counts != EXPECTED_DEFAULT_COUNTS:
        raise ValueError(
            "Unexpected default LP-PDBBind split counts. "
            f"Expected {EXPECTED_DEFAULT_COUNTS}, got {counts}."
        )


def prepare_lp_pdbbind(
    *,
    source: str = SOURCE_URL,
    out_dir: Path = DEFAULT_OUT_DIR,
    clean_level: str = "CL1",
    include_covalent: bool = False,
    check_expected_counts: bool = True,
) -> dict[str, Any]:
    """Load LP-PDBBind, write BenchAudit split CSVs, and return manifest data."""
    source_df, source_sha256 = _read_source(source)
    prepared = prepare_dataframe(source_df, clean_level=clean_level, include_covalent=include_covalent)
    counts = _write_split_files(prepared, out_dir)
    if check_expected_counts:
        _validate_expected_counts(counts, clean_level=clean_level, include_covalent=include_covalent)

    manifest = {
        "dataset": "Leak-Proof PDBBind",
        "source": source,
        "source_sha256": source_sha256,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clean_level": clean_level,
        "include_covalent": include_covalent,
        "filters": {
            "new_split": ["train", "val", "test"],
            clean_level: True,
            "covalent": "any" if include_covalent else False,
        },
        "upstream_rows": int(len(source_df)),
        "filtered_rows": int(len(prepared)),
        "split_counts": counts,
        "output_columns": OUTPUT_COLUMNS,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Leak-Proof PDBBind split CSVs for BenchAudit.")
    parser.add_argument("--source", default=SOURCE_URL, help="Upstream CSV URL or local CSV path.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output split directory.")
    parser.add_argument("--clean-level", choices=["CL1", "CL2", "CL3"], default="CL1")
    parser.add_argument("--include-covalent", action="store_true", help="Do not filter out covalent rows.")
    parser.add_argument(
        "--no-count-check",
        action="store_true",
        help="Skip the built-in CL1 non-covalent split-count check.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    manifest = prepare_lp_pdbbind(
        source=args.source,
        out_dir=args.out_dir,
        clean_level=args.clean_level,
        include_covalent=args.include_covalent,
        check_expected_counts=not args.no_count_check,
    )
    counts = ", ".join(f"{split}={count}" for split, count in manifest["split_counts"].items())
    print(f"Wrote LP-PDBBind splits to {args.out_dir} ({counts})")


if __name__ == "__main__":
    main()
