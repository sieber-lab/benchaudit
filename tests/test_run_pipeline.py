from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pandas as pd

import run as run_module


DATA_DIR = Path(__file__).resolve().parent / "data"
CONFIG_DIR = DATA_DIR / "configs"


def _tabular_cfg() -> Dict:
    return {
        "type": "tabular",
        "name": "Tiny Tabular",
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


class RunPipelineTests(unittest.TestCase):
    def test_discover_yaml_files_deduplicates_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg_file = root / "configs" / "one.yaml"
            cfg_file.parent.mkdir(parents=True)
            cfg_file.write_text("type: tabular\n", encoding="utf-8")

            files = run_module.discover_yaml_files(cfg_file.parent, cfg_file)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].resolve(), cfg_file.resolve())

    def test_run_one_config_writes_analysis_files(self) -> None:
        cfg = _tabular_cfg()
        config_path = CONFIG_DIR / "tabular_single.yaml"

        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td) / "runs"
            logger = run_module.make_logger("test.run.analysis", "INFO")

            run_module.run_one_config(
                cfg,
                config_path,
                out_root,
                logger,
                do_benchmark=False,
                configs_root=CONFIG_DIR,
                force=True,
            )

            out_dir = run_module.resolve_output_dir(
                cfg,
                out_root,
                config_path=config_path,
                configs_root=CONFIG_DIR,
            )
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "records.csv").exists())

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["counts"]["train"], 2)
            self.assertEqual(summary["config"]["name"], "Tiny Tabular")
            self.assertEqual(summary["config"]["type"], "tabular")

    def test_run_one_config_skips_when_outputs_exist(self) -> None:
        cfg = _tabular_cfg()
        config_path = CONFIG_DIR / "tabular_single.yaml"

        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td) / "runs"
            out_dir = run_module.resolve_output_dir(
                cfg,
                out_root,
                config_path=config_path,
                configs_root=CONFIG_DIR,
            )
            (out_dir / "summary.json").write_text("{}", encoding="utf-8")
            (out_dir / "performance.json").write_text("{}", encoding="utf-8")

            with patch.object(run_module, "build_loader", side_effect=AssertionError("loader should not run")):
                run_module.run_one_config(
                    cfg,
                    config_path,
                    out_root,
                    run_module.make_logger("test.run.skip", "INFO"),
                    do_benchmark=True,
                    configs_root=CONFIG_DIR,
                    force=False,
                )

    def test_run_one_config_benchmark_error_is_serialized(self) -> None:
        cfg = _tabular_cfg()
        config_path = CONFIG_DIR / "tabular_single.yaml"

        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td) / "runs"

            with patch.object(run_module, "run_baselines", side_effect=RuntimeError("boom")):
                run_module.run_one_config(
                    cfg,
                    config_path,
                    out_root,
                    run_module.make_logger("test.run.bench.err", "INFO"),
                    do_benchmark=True,
                    configs_root=CONFIG_DIR,
                    force=True,
                )

            out_dir = run_module.resolve_output_dir(
                cfg,
                out_root,
                config_path=config_path,
                configs_root=CONFIG_DIR,
            )
            perf = json.loads((out_dir / "performance.json").read_text(encoding="utf-8"))
            self.assertEqual(perf["error"], "boom")

    def test_run_one_config_benchmark_success_writes_payload(self) -> None:
        cfg = _tabular_cfg()
        config_path = CONFIG_DIR / "tabular_single.yaml"

        payload = {
            "task": "classification",
            "models": {
                "dummy": {
                    "metrics": {"roc_auc": 0.5},
                    "predictions": [0.1, 0.9],
                }
            },
        }

        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td) / "runs"

            with patch.object(run_module, "run_baselines", return_value=payload):
                run_module.run_one_config(
                    cfg,
                    config_path,
                    out_root,
                    run_module.make_logger("test.run.bench.ok", "INFO"),
                    do_benchmark=True,
                    configs_root=CONFIG_DIR,
                    force=True,
                )

            out_dir = run_module.resolve_output_dir(
                cfg,
                out_root,
                config_path=config_path,
                configs_root=CONFIG_DIR,
            )
            perf = json.loads((out_dir / "performance.json").read_text(encoding="utf-8"))
            self.assertEqual(perf["models"]["dummy"]["metrics"]["roc_auc"], 0.5)

    def test_run_one_config_requires_train_and_test_splits(self) -> None:
        cfg = _tabular_cfg()

        class _LoaderMissingTest:
            def get_splits(self):
                return {"train": pd.DataFrame({"smiles_clean": ["CCO"], "label_raw": [1]})}

        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td) / "runs"
            with patch.object(run_module, "build_loader", return_value=_LoaderMissingTest()):
                with self.assertRaises(RuntimeError):
                    run_module.run_one_config(
                        cfg,
                        CONFIG_DIR / "tabular_single.yaml",
                        out_root,
                        run_module.make_logger("test.run.missing", "INFO"),
                        do_benchmark=False,
                        configs_root=CONFIG_DIR,
                        force=True,
                    )


if __name__ == "__main__":
    unittest.main()
