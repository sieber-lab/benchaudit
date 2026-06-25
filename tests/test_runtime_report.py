from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from experiments import report_runtimes


def _write_summary(run_dir: Path, summary: dict, mtime: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _write_artifact(run_dir: Path, name: str, mtime: float) -> None:
    path = run_dir / name
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))


class RuntimeReportTests(unittest.TestCase):
    def test_build_runtime_rows_prefers_recorded_then_infers_then_lower_bound(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"

            first = runs_root / "fam" / "first"
            _write_summary(first, {"counts": {"train": 1, "test": 1}}, 100.0)
            _write_artifact(first, "records.csv", 102.0)

            recorded = runs_root / "fam" / "recorded"
            _write_summary(
                recorded,
                {
                    "counts": {"train": 2, "valid": 1, "test": 1},
                    "runtime": {"elapsed_seconds": 12.5},
                },
                200.0,
            )
            _write_artifact(recorded, "records.csv", 203.0)

            inferred = runs_root / "fam" / "inferred"
            _write_summary(inferred, {"counts": {"train": 3, "test": 2}}, 230.0)
            _write_artifact(inferred, "records.csv", 231.0)

            rows = report_runtimes.build_runtime_rows(
                runs_root,
                max_inferred_gap_seconds=100.0,
            )
            by_dataset = {row["dataset"]: row for row in rows}

            self.assertEqual(by_dataset["first"]["estimate_type"], "artifact_write_span_lower_bound")
            self.assertEqual(by_dataset["recorded"]["estimate_type"], "recorded")
            self.assertEqual(by_dataset["recorded"]["estimated_runtime_seconds"], 12.5)
            self.assertEqual(by_dataset["inferred"]["estimate_type"], "inferred_sequential_mtime")
            self.assertEqual(by_dataset["inferred"]["estimated_runtime_seconds"], 27.0)
            self.assertEqual(by_dataset["inferred"]["rows"], 5)


if __name__ == "__main__":
    unittest.main()
