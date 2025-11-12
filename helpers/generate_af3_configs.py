#!/usr/bin/env python3
"""Generate AlphaFold3 config files from collected DTI sequences."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create AF3 JSON configs from collected DTI sequences.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("helpers/dti_unique_sequences.jsonl"),
        help="Path to the JSONL file emitted by sequence_collector.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("helpers/af3_configs"),
        help="Directory where AF3 configs will be written.",
    )
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=[1],
        help="Model seeds to embed in each config (default: 1).",
    )
    return parser.parse_args()


def load_sequence_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:  # pragma: no cover - guard
                logging.warning("Skipping line %d in %s: %s", line_no, path, exc)


def build_config(row: Dict[str, object], model_seeds: Sequence[int]) -> Dict[str, object]:
    sequence_id = row["sequence_id"]
    sequence = row["sequence"]
    return {
        "name": sequence_id,
        "modelSeeds": list(model_seeds),
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": sequence,
                },
            },
        ],
        "dialect": "alphafold3",
        "version": 1,
    }


def write_configs(rows: Iterable[Dict[str, object]], out_dir: Path, model_seeds: Sequence[int]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        if "sequence_id" not in row or "sequence" not in row:
            logging.warning("Skipping malformed row without sequence_id/sequence keys.")
            continue
        config = build_config(row, model_seeds)
        file_path = out_dir / f"{row['sequence_id']}.json"
        file_path.write_text(json.dumps(config, indent=2))
        written += 1
    return written


def main() -> None:
    args = parse_args()
    rows = load_sequence_rows(args.input)
    count = write_configs(rows, args.output_dir, args.model_seeds)
    print(f"Wrote {count} AF3 config files to {args.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
