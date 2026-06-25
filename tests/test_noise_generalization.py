from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils import noise_generalization as ng


class NoiseGeneralizationTests(unittest.TestCase):
    def test_classification_metrics_include_auc_columns(self) -> None:
        metrics = ng.evaluate_classification(
            np.asarray([0, 1, 1, 0], dtype=float),
            np.asarray([0.05, 0.95, 0.8, 0.1], dtype=float),
        )

        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["pr_auc"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["n_eval_rows"], 4.0)

    def test_regression_metrics_include_full_metric_set(self) -> None:
        metrics = ng.evaluate_regression(
            np.asarray([1.0, 2.0, 3.0], dtype=float),
            np.asarray([1.0, 2.0, 4.0], dtype=float),
        )

        for key in ng.REGRESSION_METRICS:
            self.assertIn(key, metrics)
        self.assertAlmostEqual(metrics["mse"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(1.0 / 3.0))
        self.assertEqual(metrics["n_eval_rows"], 3.0)

    def test_noise_scenarios_for_classification(self) -> None:
        train = pd.DataFrame(
            {
                "smiles_clean": ["CCO", "CCO", "CCN", "CCC"],
                "label_raw": [0, 1, 0, 1],
                "split": ["train"] * 4,
                "sequence_aa": [""] * 4,
            }
        )
        test = pd.DataFrame(
            {
                "smiles_clean": ["CCO", "CCCl"],
                "label_raw": [1, 0],
                "split": ["test", "test"],
                "sequence_aa": ["", ""],
            }
        )

        conflicted, stats = ng.inject_conflicts(train, "classification", 0.5, np.random.default_rng(0))
        self.assertEqual(len(conflicted), 6)
        self.assertEqual(stats["n_conflicts_added"], 2.0)
        self.assertTrue(set(conflicted["label_raw"].astype(int)).issubset({0, 1}))

        contaminated, stats = ng.inject_contamination(train, test, 0.5, np.random.default_rng(1))
        self.assertEqual(len(contaminated), 5)
        self.assertEqual(stats["n_contamination_added"], 1.0)
        self.assertIn("CCO", set(contaminated["smiles_clean"]))

        randomized, stats = ng.inject_random_label_noise(train, 1.0, np.random.default_rng(2))
        self.assertEqual(len(randomized), len(train))
        self.assertEqual(stats["n_random_labels_modified"], 4.0)
        self.assertTrue(set(randomized["label_raw"].astype(int)).issubset({0, 1}))

        cliffed, stats = ng.inject_cliffs(
            train,
            "classification",
            1.0,
            np.random.default_rng(3),
            sim_threshold=0.9,
            fp_radius=2,
            fp_nbits=128,
            candidate_pool=4,
        )
        self.assertEqual(len(cliffed), len(train))
        self.assertGreater(stats["n_cliffs_modified"], 0.0)

    def test_noise_scenarios_for_regression(self) -> None:
        train = pd.DataFrame(
            {
                "smiles_clean": ["CCO", "CCO", "CCN", "CCC"],
                "label_raw": [1.0, 2.0, 3.0, 4.0],
                "split": ["train"] * 4,
                "sequence_aa": [""] * 4,
            }
        )

        conflicted, stats = ng.inject_conflicts(train, "regression", 0.5, np.random.default_rng(4))
        self.assertEqual(len(conflicted), 6)
        self.assertEqual(stats["n_conflicts_added"], 2.0)
        self.assertFalse(conflicted["label_raw"].isna().any())

        cliffed, stats = ng.inject_cliffs(
            train,
            "regression",
            1.0,
            np.random.default_rng(5),
            sim_threshold=0.9,
            fp_radius=2,
            fp_nbits=128,
            candidate_pool=4,
        )
        self.assertEqual(len(cliffed), len(train))
        self.assertGreater(stats["n_cliffs_modified"], 0.0)

    def test_discovery_skips_multitask_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, labels in {
                "SingleTask": ["0", "1", "0"],
                "MultiTask": ["[0, 1]", "[1, 0]", "[0, 0]"],
                "NonBinary": ["0", "0.5", "1"],
            }.items():
                run_dir = root / "moleculenet" / name
                run_dir.mkdir(parents=True)
                (run_dir / "summary.json").write_text(
                    json.dumps({"task": {"type": "classification"}, "config": {"name": name}}),
                    encoding="utf-8",
                )
                pd.DataFrame(
                    {
                        "smiles_clean": ["CCO", "CCN", "CCC"],
                        "label_raw": labels,
                        "split": ["train", "valid", "test"],
                        "valid": [True, True, True],
                    }
                ).to_csv(run_dir / "records.csv", index=False)

            datasets, manifest = ng.discover_noise_datasets(root, ["all"], skip_multitask=True)

            self.assertEqual([ds.bundle.dataset for ds in datasets], ["SingleTask"])
            skipped = [row for row in manifest if row["status"] == "skipped"]
            self.assertEqual(len(skipped), 2)
            self.assertTrue(any("multitask" in row["skip_reason"] for row in skipped))
            self.assertTrue(any("non_binary_labels" in row["skip_reason"] for row in skipped))

    def test_missing_backend_validation_is_clear(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "lightgbm"):
            ng.validate_model_backends(
                ["lgbm"],
                ng.BackendAvailability(have_lgbm=False, have_torch=True, have_lightning=True),
            )
        with self.assertRaisesRegex(RuntimeError, "torch"):
            ng.validate_model_backends(
                ["torch_mlp"],
                ng.BackendAvailability(have_lgbm=True, have_torch=False, have_lightning=True),
            )
        with self.assertRaisesRegex(RuntimeError, "lightning"):
            ng.validate_model_backends(
                ["torch_mlp"],
                ng.BackendAvailability(have_lgbm=True, have_torch=True, have_lightning=False),
            )

    def test_mlp_accelerator_helpers(self) -> None:
        self.assertEqual(ng._normalize_mlp_accelerator("cuda"), "gpu")
        self.assertEqual(ng._normalize_mlp_accelerator("gpu"), "gpu")
        self.assertEqual(ng._normalize_mlp_accelerator("cpu"), "cpu")
        self.assertEqual(ng._lightning_devices_arg("1"), 1)
        self.assertEqual(ng._lightning_devices_arg("auto"), "auto")
        self.assertEqual(ng._lightning_devices_arg("0,1"), [0, 1])
        with self.assertRaisesRegex(ValueError, "mlp-accelerator"):
            ng._normalize_mlp_accelerator("invalid")


if __name__ == "__main__":
    unittest.main()
