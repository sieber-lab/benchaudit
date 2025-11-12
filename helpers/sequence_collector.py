#!/usr/bin/env python3
"""Collect unique amino-acid sequences from configured DTI datasets.

Instead of crawling every file, this helper inspects the DTI YAML configs under
``configs/dti`` (customizable via ``--config-dir``), loads the declared CSV
splits, and reads the amino-acid column specified by ``info.sequence_col``. The
sequences are normalized (trimmed + upper-cased), deduplicated, and emitted as
JSON Lines along with the dataset/split/file provenance so each sequence can be
traced back to its source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


@dataclass
class DatasetConfig:
    name: str
    config_path: Path
    sequence_col: str
    splits: Dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect unique amino-acid sequences from DTI datasets.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/dti"),
        help="Directory containing DTI YAML configs (default: configs/dti).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve relative dataset paths (default: current working dir).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("unique_sequences.jsonl"),
        help="Destination JSONL file (default: unique_sequences.jsonl).",
    )
    return parser.parse_args()


def load_dti_configs(config_dir: Path, repo_root: Path) -> List[DatasetConfig]:
    dataset_configs: List[DatasetConfig] = []

    for cfg_path in sorted(config_dir.glob("*.yaml")):
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
        except Exception as exc:
            logging.warning("Skipping %s (invalid YAML): %s", cfg_path, exc)
            continue

        modality = str(cfg.get("modality") or cfg.get("type") or "").lower()
        if modality != "dti":
            continue

        info = cfg.get("info") or {}
        sequence_col = info.get("sequence_col")
        if not sequence_col:
            logging.warning("Skipping %s (missing info.sequence_col).", cfg_path)
            continue

        raw_paths = cfg.get("paths")
        if not raw_paths:
            logging.warning("Skipping %s (missing paths section).", cfg_path)
            continue

        resolved_paths: Dict[str, Path] = {}
        for split_name, split_path in raw_paths.items():
            path_obj = Path(split_path)
            if not path_obj.is_absolute():
                path_obj = (repo_root / path_obj).resolve()
            resolved_paths[split_name] = path_obj

        dataset_configs.append(
            DatasetConfig(
                name=str(cfg.get("name") or cfg_path.stem),
                config_path=cfg_path,
                sequence_col=str(sequence_col),
                splits=resolved_paths,
            )
        )

    if not dataset_configs:
        logging.warning("No DTI configs found under %s.", config_dir)
    return dataset_configs


def read_sequences(csv_path: Path, column: str) -> Iterable[str]:
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise KeyError(f"Column '{column}' not found in {csv_path}.")
        for row in reader:
            value = row.get(column)
            if not value:
                continue
            normalized = "".join(value.split()).upper()
            if normalized:
                yield normalized


def collect_sequences(datasets: List[DatasetConfig], repo_root: Path) -> Dict[str, Dict[str, object]]:
    sequences: Dict[str, Dict[str, object]] = {}

    for ds in datasets:
        for split, file_path in ds.splits.items():
            if not file_path.exists():
                logging.warning("Missing split file %s for dataset %s.", file_path, ds.name)
                continue

            if file_path.is_dir():
                logging.warning("Expected CSV file but got directory: %s", file_path)
                continue

            try:
                sequences_iter = read_sequences(file_path, ds.sequence_col)
            except Exception as exc:
                logging.warning("Failed to read %s (%s split): %s", file_path, ds.name, exc)
                continue

            for seq in sequences_iter:
                if seq not in sequences:
                    seq_hash = hashlib.sha1(seq.encode("utf-8")).hexdigest()[:12]
                    sequence_id = f"{seq_hash}@{ds.name}"
                    sequences[seq] = {
                        "sequence_id": sequence_id,
                        "sequence": seq,
                        "sources": [],
                    }

                source_record = {
                    "dataset": ds.name,
                    "split": split,
                    "file": _rel_path(file_path, repo_root),
                }
                sequences[seq]["sources"].append(source_record)

    return sequences


def _rel_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_sequences(sequences: Dict[str, Dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(sequences.values(), key=lambda item: item["sequence_id"])

    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            entry["sources"] = sorted(
                entry["sources"],
                key=lambda src: (src["dataset"], src["split"], src["file"]),
            )
            handle.write(json.dumps(entry))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()

    datasets = load_dti_configs(config_dir, repo_root)
    sequences = collect_sequences(datasets, repo_root)
    write_sequences(sequences, output)
    print(f"Unique sequences: {len(sequences)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
