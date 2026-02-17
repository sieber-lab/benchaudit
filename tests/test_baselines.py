from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils import baselines


class _DummyClassifier:
    def fit(self, X, y):
        self.p = float(np.clip(np.mean(y), 1e-4, 1 - 1e-4))
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack(
            [
                np.full(n, 1.0 - self.p, dtype=float),
                np.full(n, self.p, dtype=float),
            ]
        )


class _DummyRegressor:
    def fit(self, X, y):
        self.mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean, dtype=float)


class BaselineTests(unittest.TestCase):
    def test_eval_baselines_generic_classification(self) -> None:
        train = pd.DataFrame(
            {
                "f1": [0.1, 0.9, 0.2, 0.8],
                "label_raw": [0, 1, 0, 1],
            }
        )
        test = pd.DataFrame(
            {
                "f1": [0.15, 0.85],
                "label_raw": [0, 1],
            }
        )
        cfg = {"task": "classification", "info": {"label_col": "label_raw"}}

        with patch.object(baselines, "_class_models", return_value={"dummy": _DummyClassifier()}):
            result = baselines.eval_baselines_generic(cfg, {"train": train, "test": test})

        self.assertEqual(result["task"], "classification")
        self.assertIn("dummy", result["models"])
        self.assertEqual(len(result["models"]["dummy"]["predictions"]), 2)
        self.assertIn("roc_auc", result["models"]["dummy"]["metrics"])

    def test_eval_baselines_generic_regression(self) -> None:
        train = pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0],
                "label_raw": [0.1, 0.2, 0.3, 0.4],
            }
        )
        test = pd.DataFrame(
            {
                "f1": [1.5, 2.5],
                "label_raw": [0.25, 0.35],
            }
        )
        cfg = {"task": "regression", "info": {"label_col": "label_raw"}}

        with patch.object(baselines, "_reg_models", return_value={"dummy": _DummyRegressor()}):
            result = baselines.eval_baselines_generic(cfg, {"train": train, "test": test})

        self.assertEqual(result["task"], "regression")
        self.assertIn("dummy", result["models"])
        self.assertEqual(len(result["models"]["dummy"]["predictions"]), 2)
        self.assertIn("mse", result["models"]["dummy"]["metrics"])

    def test_eval_baselines_generic_rejects_multitask_labels(self) -> None:
        train = pd.DataFrame({"f1": [1.0, 2.0], "label_raw": [[0, 1], [1, 0]]})
        test = pd.DataFrame({"f1": [3.0], "label_raw": [[1, 1]]})
        cfg = {"task": "classification", "info": {"label_col": "label_raw"}}

        with self.assertRaises(NotImplementedError):
            baselines.eval_baselines_generic(cfg, {"train": train, "test": test})

    def test_run_baselines_requires_splits_for_non_polaris(self) -> None:
        with self.assertRaises(ValueError):
            baselines.run_baselines({"type": "tabular", "task": "classification"}, splits=None)

    def test_run_baselines_dispatches_polaris_path(self) -> None:
        with patch.object(baselines, "eval_baselines_polaris", return_value={"ok": True}) as mock_eval:
            result = baselines.run_baselines({"type": "polaris", "name": "fake"}, splits=None)

        self.assertEqual(result, {"ok": True})
        mock_eval.assert_called_once()


if __name__ == "__main__":
    unittest.main()
