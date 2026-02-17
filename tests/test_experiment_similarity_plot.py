from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments import plot_trainvalid_test_similarity_by_benchmark as plot_mod


class SimilarityPlotTests(unittest.TestCase):
    def _write_summary(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_load_similarity_rows_filters_invalid_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            self._write_summary(
                runs_root / "tdc" / "DatasetA" / "summary.json",
                {"similarity": {"test_to_trainvalid": {"mean": 0.82, "std": 0.05, "n": 12}}},
            )
            self._write_summary(
                runs_root / "tdc" / "DatasetB" / "summary.json",
                {"similarity": {"test_to_trainvalid": {"mean": None, "std": 0.1, "n": 9}}},
            )
            self._write_summary(
                runs_root / "moleculenet" / "DatasetC" / "summary.json",
                {"similarity": {"test_to_trainvalid": "invalid"}},
            )

            rows = plot_mod.load_similarity_rows(runs_root)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["benchmark"], "tdc")
            self.assertEqual(rows[0]["dataset"], "DatasetA")

    def test_load_similarity_rows_raises_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            self._write_summary(
                runs_root / "tdc" / "DatasetX" / "summary.json",
                {"similarity": {"test_to_trainvalid": {"mean": None}}},
            )

            with self.assertRaises(RuntimeError):
                plot_mod.load_similarity_rows(runs_root)

    def test_summarize_by_benchmark_uses_expected_order(self) -> None:
        rows = [
            {"benchmark": "unknown", "similarity_mean": 0.5},
            {"benchmark": "tdc", "similarity_mean": 0.7},
            {"benchmark": "polaris", "similarity_mean": 0.8},
            {"benchmark": "tdc", "similarity_mean": 0.9},
        ]

        stats = plot_mod.summarize_by_benchmark(rows)
        order = [row["benchmark"] for row in stats]

        self.assertEqual(order[:2], ["polaris", "tdc"])
        self.assertEqual(stats[1]["n_datasets"], 2)

    def test_render_svg_writes_svg_file(self) -> None:
        rows = [
            {"benchmark": "polaris", "dataset": "A", "similarity_mean": 0.6},
            {"benchmark": "polaris", "dataset": "B", "similarity_mean": 0.8},
            {"benchmark": "tdc", "dataset": "C", "similarity_mean": 0.4},
        ]

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "plot.svg"
            plot_mod._render_svg(rows, out_path)
            text = out_path.read_text(encoding="utf-8")

            self.assertIn("<svg", text)
            self.assertIn("polaris (n=2)", text)

    def test_render_svg_rejects_non_svg_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                plot_mod._render_svg([], Path(td) / "plot.png")

    def test_write_csv_writes_rows(self) -> None:
        rows = [{"benchmark": "tdc", "dataset": "A", "similarity_mean": 0.75}]

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "rows.csv"
            plot_mod._write_csv(rows, out_path, ["benchmark", "dataset", "similarity_mean"])

            with out_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                loaded = list(reader)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["benchmark"], "tdc")


if __name__ == "__main__":
    unittest.main()
