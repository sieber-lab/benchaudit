#!/usr/bin/env python3
"""Convert collected DTI sequences (JSONL) into FASTA format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Dict, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sequence JSONL to FASTA.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("helpers/dti_unique_sequences.jsonl"),
        help="Path to JSONL file emitted by sequence_collector.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("helpers/af_input.fasta"),
        help="Destination FASTA path (default: helpers/af_input.fasta).",
    )
    return parser.parse_args()


def load_rows(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            yield row


def export_fasta(rows: Iterator[Dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            seq_id = row.get("sequence_id")
            sequence = row.get("sequence")
            if not seq_id or not sequence:
                continue
            handle.write(f">{seq_id}\n{sequence}\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    count = export_fasta(rows, args.output)
    print(f"Wrote {count} sequences to {args.output}")


if __name__ == "__main__":
    main()
