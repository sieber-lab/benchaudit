from __future__ import annotations

import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import utils.loader as loader_mod
from utils.loader import DTILoader, PolarisLoader, TDCLoader, TabularLoader


DATA_DIR = Path(__file__).resolve().parent / "data"


class LoaderTests(unittest.TestCase):
    def test_tabular_loader_single_file_split(self) -> None:
        cfg = {
            "type": "tabular",
            "task": "classification",
            "path": str(DATA_DIR / "tabular_single.csv"),
            "info": {
                "split_col": "split",
                "smiles_col": "smiles",
                "label_col": "label",
                "id_col": "compound_id",
                "cleaner": "none",
            },
        }

        splits = TabularLoader(cfg).get_splits()

        self.assertEqual(set(splits), {"train", "valid", "test"})
        self.assertEqual(splits["train"]["smiles_clean"].tolist(), ["CCO", "CCN"])
        self.assertEqual(splits["test"]["label_raw"].tolist(), [1, 0])
        self.assertIn("id", splits["valid"].columns)

    def test_tabular_loader_three_paths(self) -> None:
        cfg = {
            "type": "tabular",
            "task": "classification",
            "paths": {
                "train": str(DATA_DIR / "tabular_paths" / "train.csv"),
                "valid": str(DATA_DIR / "tabular_paths" / "valid.csv"),
                "test": str(DATA_DIR / "tabular_paths" / "test.csv"),
            },
            "info": {
                "smiles_col": "Drug",
                "label_col": "Y",
                "id_col": "ID",
                "cleaner": "none",
            },
        }

        splits = TabularLoader(cfg).get_splits()

        self.assertEqual(len(splits["train"]), 3)
        self.assertEqual(len(splits["valid"]), 2)
        self.assertEqual(len(splits["test"]), 2)
        self.assertEqual(splits["test"]["id"].tolist(), [6, 7])

    def test_dti_loader_requires_sequence_column(self) -> None:
        cfg = {
            "type": "dti",
            "modality": "dti",
            "task": "classification",
            "paths": {
                "train": str(DATA_DIR / "tabular_paths" / "train.csv"),
                "valid": str(DATA_DIR / "tabular_paths" / "valid.csv"),
                "test": str(DATA_DIR / "tabular_paths" / "test.csv"),
            },
            "info": {
                "smiles_col": "Drug",
                "label_col": "Y",
                "id_col": "ID",
                "cleaner": "none",
            },
        }

        with self.assertRaises(KeyError):
            DTILoader(cfg).get_splits()

    def test_dti_loader_keeps_sequence_and_target_columns(self) -> None:
        cfg = {
            "type": "dti",
            "modality": "dti",
            "task": "classification",
            "paths": {
                "train": str(DATA_DIR / "dti" / "train.csv"),
                "valid": str(DATA_DIR / "dti" / "valid.csv"),
                "test": str(DATA_DIR / "dti" / "test.csv"),
            },
            "info": {
                "smiles_col": "Ligand",
                "label_col": "classification_label",
                "id_col": "ID",
                "sequence_col": "Protein",
                "target_id_col": "Target_ID",
                "cleaner": "none",
                "keep_invalid": True,
            },
        }

        splits = DTILoader(cfg).get_splits()

        self.assertIn("sequence_aa", splits["train"].columns)
        self.assertIn("target_id", splits["train"].columns)
        self.assertEqual(splits["train"]["sequence_aa"].tolist(), ["AAAA", "BBBB", "CCCC"])
        self.assertEqual(len(splits["test"]), 2)

    def test_tdc_loader_with_stub_dataset(self) -> None:
        class _StubDataset:
            def get_split(self, method=None):
                frame = {
                    "Drug": ["CCO", "CCN"],
                    "Y": [1, 0],
                    "ID": [10, 11],
                }
                return {
                    "train": pd.DataFrame(frame),
                    "valid": pd.DataFrame(frame),
                    "test": pd.DataFrame(frame),
                }

        cfg = {
            "type": "tdc",
            "name": "stub",
            "task": "classification",
            "info": {
                "cleaner": "none",
                "split": "random",
            },
        }

        with patch.object(TDCLoader, "_init_dataset", return_value=_StubDataset()):
            splits = TDCLoader(deepcopy(cfg)).get_splits()

        self.assertEqual(set(splits), {"train", "valid", "test"})
        self.assertEqual(splits["train"]["label_raw"].tolist(), [1, 0])

    def test_polaris_loader_with_stub_benchmark(self) -> None:
        class _Split:
            def __init__(self, inputs, targets):
                self.inputs = inputs
                self.targets = targets

        class _Benchmark:
            def get_train_test_split(self):
                return _Split(["CCO", "CCN"], [1, 0]), _Split(["CCC"], [1])

        fake_po = types.SimpleNamespace(load_benchmark=lambda _: _Benchmark())
        cfg = {
            "type": "polaris",
            "name": "fake/vendor-bench",
            "task": "classification",
            "info": {
                "cleaner": "none",
            },
        }

        with patch.object(loader_mod, "po", fake_po):
            splits = PolarisLoader(deepcopy(cfg)).get_splits()

        self.assertEqual(set(splits), {"train", "test"})
        self.assertIn("id", splits["train"].columns)
        self.assertEqual(splits["test"]["label_raw"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
