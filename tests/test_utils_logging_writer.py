from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils import ResultWriter, json_default, make_logger, resolve_output_dir
from utils.analysis import AnalysisResult, AnalyzerConfig


class LoggingAndWriterTests(unittest.TestCase):
    def test_make_logger_reuses_single_handler(self) -> None:
        logger = make_logger("test.logger", "DEBUG")
        logger_again = make_logger("test.logger", "INFO")

        self.assertIs(logger, logger_again)
        self.assertEqual(len(logger.handlers), 1)
        self.assertEqual(logger.level, logging.INFO)
        self.assertFalse(logger.propagate)

    def test_json_default_handles_numpy_scalars_and_arrays(self) -> None:
        payload = {
            "int": np.int64(3),
            "float": np.float32(1.25),
            "arr": np.array([1, 2, 3], dtype=np.int64),
        }
        encoded = json.dumps(payload, default=json_default)
        decoded = json.loads(encoded)

        self.assertEqual(decoded["int"], 3)
        self.assertAlmostEqual(decoded["float"], 1.25)
        self.assertEqual(decoded["arr"], [1, 2, 3])

    def test_resolve_output_dir_respects_config_relative_layout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            configs_root = root / "configs"
            config_path = configs_root / "nested" / "tiny.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("name: x\n", encoding="utf-8")

            out_dir = resolve_output_dir(
                {"type": "tabular", "name": "My Dataset"},
                cli_out_root=root / "runs",
                config_path=config_path,
                configs_root=configs_root,
            )

            self.assertEqual(out_dir.name, "My-Dataset")
            self.assertIn(str(root / "runs" / "nested"), str(out_dir))

    def test_result_writer_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            writer = ResultWriter(out_dir, make_logger("test.writer", "INFO"))

            result = AnalysisResult(
                summary={"ok": True},
                per_record_df=pd.DataFrame(
                    [
                        {"id": "a", "smiles_clean": "CCO", "label_raw": 1, "split": "train"},
                        {"id": "b", "smiles_clean": "CCN", "label_raw": 0, "split": "test"},
                    ]
                ),
                conflicts_rows=[{"kind": "cross_train_test", "id": "a"}],
                cliffs_rows=[{"kind": "cross_tv_test", "id_A": "a", "id_B": "b"}],
                sequence_alignment_rows=[{"split_query": "test", "identity_pct": 85.0}],
                structure_alignment_rows=[],
            )

            paths = writer.write_analysis(result)

            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "records.csv").exists())
            self.assertTrue((out_dir / "conflicts.jsonl").exists())
            self.assertTrue((out_dir / "cliffs.jsonl").exists())
            self.assertTrue((out_dir / "sequence_alignments.jsonl").exists())
            self.assertIsNone(paths["structure_alignments"])

    def test_write_records_returns_none_for_empty_df(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            writer = ResultWriter(Path(td), make_logger("test.writer.empty", "INFO"))
            self.assertIsNone(writer.write_records(pd.DataFrame()))

    def test_analyzer_config_rejects_invalid_similarity_threshold(self) -> None:
        with self.assertRaises(Exception) as ctx:
            AnalyzerConfig(task_type="classification", typ="tabular", sim_threshold=1.5)
        self.assertIn("sim_threshold", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
