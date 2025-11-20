#!/usr/bin/env python3
"""Convert collected DTI sequences (JSONL) into CSV format."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterator, Dict, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sequence JSONL to id,sequence CSV/FASTA.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("helpers/dti_unique_sequences.jsonl"),
        help="Path to JSONL file emitted by sequence_collector.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("helpers/af_input.csv"),
        help="Destination CSV path (default: helpers/af_input.csv).",
    )
    parser.add_argument(
        "--fasta-output",
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


def export_csv(rows: Iterator[Dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sequence"])
        for row in rows:
            seq_id = row.get("sequence_id")
            sequence = row.get("sequence")
            if not seq_id or not sequence:
                continue
            writer.writerow([seq_id, sequence])
            count += 1
    return count


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
    rows_list = list(load_rows(args.input))
    csv_count = export_csv(iter(rows_list), args.output)
    fasta_count = export_fasta(iter(rows_list), args.fasta_output)
    print(
        f"Wrote {csv_count} sequences to {args.output} and {fasta_count} sequences to {args.fasta_output}"
    )


if __name__ == "__main__":
    main()
