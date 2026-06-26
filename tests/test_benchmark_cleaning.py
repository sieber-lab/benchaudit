from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import run as run_module
from utils import clean_benchmark_splits


class BenchmarkCleaningTests(unittest.TestCase):
    def test_clean_benchmark_splits_removes_invalid_conflicts_and_contaminants(self) -> None:
        splits = {
            "train": pd.DataFrame(
                [
                    {"id": "tr_keep", "smiles_clean": "CCO", "label_raw": 1},
                    {"id": "tr_dup_a", "smiles_clean": "CCCC", "label_raw": 0},
                    {"id": "tr_dup_b", "smiles_clean": "CCCC", "label_raw": 0},
                    {"id": "tr_conflict", "smiles_clean": "CCN", "label_raw": 0},
                    {"id": "tr_bad", "smiles_clean": "notasmiles", "label_raw": 1},
                ]
            ),
            "valid": pd.DataFrame(
                [
                    {"id": "va_conflict", "smiles_clean": "CCN", "label_raw": 1},
                    {"id": "va_keep", "smiles_clean": "CCC", "label_raw": 0},
                ]
            ),
            "test": pd.DataFrame(
                [
                    {"id": "te_contaminant", "smiles_clean": "CCO", "label_raw": 1},
                    {"id": "te_conflict", "smiles_clean": "CCN", "label_raw": 0},
                    {"id": "te_keep", "smiles_clean": "CCCl", "label_raw": 0},
                    {"id": "te_bad", "smiles_clean": "", "label_raw": 0},
                ]
            ),
        }

        cleaned, report = clean_benchmark_splits(splits, "classification")

        self.assertEqual(cleaned["train"]["id"].tolist(), ["tr_keep", "tr_dup_a", "tr_dup_b"])
        self.assertEqual(cleaned["valid"]["id"].tolist(), ["va_keep"])
        self.assertEqual(cleaned["test"]["id"].tolist(), ["te_keep"])
        self.assertEqual(report["removed_counts"]["train"]["invalid"], 1)
        self.assertEqual(report["removed_counts"]["train"]["conflict"], 1)
        self.assertEqual(report["removed_counts"]["test"]["conflict"], 1)
        self.assertEqual(report["removed_counts"]["test"]["contaminant"], 1)
        self.assertEqual(report["removed_counts"]["test"]["invalid"], 1)
        self.assertEqual(report["n_conflict_smiles"], 1)
        self.assertEqual(report["n_contaminant_smiles"], 1)

    def test_clean_benchmark_splits_preserves_dti_column_alignment(self) -> None:
        splits = {
            "train": pd.DataFrame(
                [
                    {
                        "id": "tr1",
                        "smiles_clean": "CCO",
                        "label_raw": 1,
                        "sequence_aa": "AAAA",
                        "target_id": "TGT1",
                    }
                ]
            ),
            "test": pd.DataFrame(
                [
                    {
                        "id": "te_contaminant",
                        "smiles_clean": "CCO",
                        "label_raw": 1,
                        "sequence_aa": "BBBB",
                        "target_id": "TGT2",
                    },
                    {
                        "id": "te_keep",
                        "smiles_clean": "CCC",
                        "label_raw": 0,
                        "sequence_aa": "CCCC",
                        "target_id": "TGT3",
                    },
                ]
            ),
        }

        cleaned, report = clean_benchmark_splits(splits, "classification")

        self.assertEqual(cleaned["test"]["id"].tolist(), ["te_keep"])
        self.assertEqual(cleaned["test"]["sequence_aa"].tolist(), ["CCCC"])
        self.assertEqual(cleaned["test"]["target_id"].tolist(), ["TGT3"])
        self.assertEqual(cleaned["test"]["label_raw"].tolist(), [0])
        self.assertEqual(report["removed_counts"]["test"]["contaminant"], 1)

    def test_run_one_config_applies_clean_benchmark_option(self) -> None:
        class _Loader:
            def get_splits(self):
                return {
                    "train": pd.DataFrame(
                        [{"id": "tr_keep", "smiles_clean": "CCO", "label_raw": 1}]
                    ),
                    "valid": pd.DataFrame(
                        [{"id": "va_keep", "smiles_clean": "CCCl", "label_raw": 0}]
                    ),
                    "test": pd.DataFrame(
                        [
                            {"id": "te_contaminant", "smiles_clean": "CCO", "label_raw": 1},
                            {"id": "te_keep", "smiles_clean": "CCC", "label_raw": 0},
                            {"id": "te_bad", "smiles_clean": "notasmiles", "label_raw": 1},
                        ]
                    ),
                }

        cfg = {
            "type": "tabular",
            "name": "Clean Tiny",
            "task": "classification",
            "info": {"clean_benchmark": True},
        }

        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td) / "runs"
            with patch.object(run_module, "build_loader", return_value=_Loader()):
                run_module.run_one_config(
                    cfg,
                    Path("clean_tiny.yaml"),
                    out_root,
                    run_module.make_logger("test.clean.pipeline", "INFO"),
                    do_benchmark=False,
                    force=True,
                )

            out_dir = run_module.resolve_output_dir(
                cfg,
                out_root,
                config_path=Path("clean_tiny.yaml"),
            )
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            records = pd.read_csv(out_dir / "records.csv")

        self.assertEqual(summary["counts"]["test"], 1)
        self.assertEqual(summary["benchmark_cleaning"]["original_counts"]["test"], 3)
        self.assertEqual(summary["benchmark_cleaning"]["cleaned_counts"]["test"], 1)
        self.assertEqual(summary["benchmark_cleaning"]["removed_counts"]["test"]["invalid"], 1)
        self.assertEqual(summary["benchmark_cleaning"]["removed_counts"]["test"]["contaminant"], 1)
        self.assertEqual(records["id"].tolist(), ["tr_keep", "va_keep", "te_keep"])


if __name__ == "__main__":
    unittest.main()
