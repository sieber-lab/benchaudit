from __future__ import annotations

import tempfile
import unittest
import os
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils.rank_fragility.audit import audit_dataset
from utils.rank_fragility.chem import standardize_smiles
from utils.rank_fragility.config import AuditConfig, PanelConfig
from utils.rank_fragility.counterfactual import run_counterfactual_evaluation
from utils.rank_fragility.leaderboard import rank_models
from utils.rank_fragility.metrics import higher_is_better
from utils.rank_fragility.panels import generate_counterfactual_panels
from utils.rank_fragility.run import main as cli_main


def _dataset(task: str = "classification") -> pd.DataFrame:
    train = [
        ("tr00", "CCO", 0.0),
        ("tr01", "CCN", 0.0),
        ("tr02", "c1ccccc1", 1.0),
        ("tr03", "C1CCCCC1", 1.0),
        ("tr04", "CC(=O)O", 0.0),
        ("tr05", "CCCl", 0.0),
        ("tr06", "CCC", 0.0),
        ("tr07", "COC", 0.0),
        ("tr08", "CCBr", 1.0),
        ("tr09", "CC(C)O", 0.0),
        ("tr10", "CC(C)N", 1.0),
        ("tr11", "c1ccncc1", 1.0),
        ("tr12", "CC(C)C", 0.0),
        ("tr13", "O=C=O", 0.0),
        ("tr14", "CC(C)(C)O", 0.0),
        ("tr15", "CCO", 0.0),
    ]
    valid = [
        ("va00", "C=CC", 0.0),
        ("va01", "CNC", 0.0),
        ("va02", "CCS", 1.0),
        ("va03", "OCCO", 0.0),
        ("va04", "CC#N", 1.0),
    ]
    test = [
        ("te_exact", "CCO", 0.0),
        ("te_conflict", "CCN", 1.0),
        ("te_near", "CCCO", 0.0),
        ("te_scaffold", "Cc1ccccc1", 1.0),
        ("te_clean0", "C1CCNCC1", 1.0),
        ("te_clean1", "CC(F)(F)F", 0.0),
        ("te_clean2", "N#CCO", 0.0),
        ("te_clean3", "CCS", 1.0),
        ("te_clean4", "C=CCO", 0.0),
        ("te_clean5", "CC(C)F", 0.0),
        ("te_clean6", "NCCO", 1.0),
        ("te_invalid", "not_a_smiles", 0.0),
    ]
    rows = []
    for split, data in [("train", train), ("valid", valid), ("test", test)]:
        for molecule_id, smiles, y in data:
            rows.append({"molecule_id": molecule_id, "smiles": smiles, "y": y, "split": split})
    df = pd.DataFrame(rows)
    if task == "regression":
        df["y"] = np.linspace(0, 5, len(df))
        df.loc[df["molecule_id"] == "tr01", "y"] = 0.0
        df.loc[df["molecule_id"] == "te_conflict", "y"] = 2.0
    return df


def _write_predictions(pred_dir: Path, test_df: pd.DataFrame) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    y = test_df["y"].astype(float).to_numpy()
    model_b = np.where(y > 0.5, 0.9, 0.1)
    model_a = np.asarray([0.2, 0.3, 0.8, 0.4, 0.7, 0.6, 0.5, 0.3, 0.55, 0.45, 0.25, 0.75])
    for name, pred in {"model_a": model_a, "model_b": model_b}.items():
        pd.DataFrame({"molecule_id": test_df["molecule_id"], "y_true": test_df["y"], "y_pred": pred}).to_csv(
            pred_dir / f"{name}.csv", index=False
        )


class ChemRankFragilityTests(unittest.TestCase):
    def test_invalid_smiles_do_not_crash_standardization(self) -> None:
        self.assertIsNone(standardize_smiles("not_a_smiles"))

    def test_audit_detects_classification_flags_and_priority(self) -> None:
        audited = audit_dataset(
            _dataset(),
            AuditConfig(task="classification", primary_near_leak_threshold=0.2),
        )
        by_id = audited.set_index("molecule_id")
        self.assertTrue(bool(by_id.loc["te_exact", "exact_train_test_leak"]))
        self.assertTrue(bool(by_id.loc["te_conflict", "label_conflict"]))
        self.assertTrue(bool(by_id.loc["te_near", "near_train_analogue"]))
        self.assertEqual(by_id.loc["te_conflict", "audit_group"], "label_conflict")

    def test_audit_detects_regression_conflicts_and_same_scaffold(self) -> None:
        audited = audit_dataset(
            _dataset(task="regression"),
            AuditConfig(task="regression", regression_conflict_threshold=1.0, primary_near_leak_threshold=0.99),
        )
        by_id = audited.set_index("molecule_id")
        self.assertTrue(bool(by_id.loc["te_conflict", "label_conflict"]))
        self.assertTrue(bool(by_id.loc["te_scaffold", "same_scaffold_as_train"]))
        self.assertEqual(by_id.loc["te_scaffold", "audit_group"], "same_scaffold")

    def test_metric_direction_and_ranking(self) -> None:
        self.assertTrue(higher_is_better("auroc"))
        self.assertFalse(higher_is_better("rmse"))
        high = rank_models(pd.DataFrame({"model": ["a", "b"], "score": [0.8, 0.7]}), "auroc")
        low = rank_models(pd.DataFrame({"model": ["a", "b"], "score": [0.8, 0.7]}), "rmse")
        self.assertEqual(high.iloc[0]["model"], "a")
        self.assertEqual(low.iloc[0]["model"], "b")

    def test_panels_preserve_target_rates_and_warn_on_infeasible_rates(self) -> None:
        audited_test = pd.DataFrame(
            {
                "molecule_id": [f"m{i}" for i in range(20)],
                "y": [i % 2 for i in range(20)],
                "audit_group": ["exact_leak"] * 5 + ["audit_clean"] * 15,
                "exact_train_test_leak": [True] * 5 + [False] * 15,
                "near_train_analogue": [False] * 20,
                "label_conflict": [False] * 20,
            }
        )
        manifest = generate_counterfactual_panels(
            audited_test,
            PanelConfig(panel_size=12, n_panels=4, target_rates=(0.25,), random_seed=1),
        )
        leak = manifest[manifest["mode"] == "leakage_curve"]
        self.assertFalse(leak.empty)
        self.assertTrue(np.allclose(leak.groupby("panel_id")["observed_rate"].first(), 0.25))

        with self.assertWarns(RuntimeWarning):
            generate_counterfactual_panels(
                audited_test,
                PanelConfig(panel_size=12, n_panels=1, target_rates=(0.75,), random_seed=1),
            )

    def test_sota_baseline_delta_sign_for_lower_is_better_metric(self) -> None:
        pred = pd.DataFrame(
            {
                "model": ["baseline", "baseline", "sota", "sota"],
                "molecule_id": ["a", "b", "a", "b"],
                "y_true": [0, 1, 0, 1],
                "y_pred": [0.4, 0.6, 0.1, 0.9],
                "audit_group": ["audit_clean"] * 4,
                "max_train_tanimoto": [0.1] * 4,
            }
        )
        manifest = pd.DataFrame(
            {
                "panel_id": ["p0", "p0"],
                "mode": ["clean_reference", "clean_reference"],
                "target_rate": [0.0, 0.0],
                "observed_rate": [0.0, 0.0],
                "molecule_id": ["a", "b"],
            }
        )
        out = run_counterfactual_evaluation(pred, manifest, "classification", "log_loss", "baseline", "sota")
        self.assertGreater(out["sota_margin_by_panel"]["delta"].iloc[0], 0)

    def test_cli_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = _dataset()
            data_path = root / "dataset.csv"
            data.to_csv(data_path, index=False)
            pred_dir = root / "predictions"
            _write_predictions(pred_dir, data[data["split"] == "test"])
            out = root / "out"

            cli_main(
                [
                    "--data",
                    str(data_path),
                    "--pred_dir",
                    str(pred_dir),
                    "--task",
                    "classification",
                    "--metric",
                    "auroc",
                    "--baseline_model",
                    "model_a",
                    "--sota_model",
                    "auto",
                    "--primary_near_leak_threshold",
                    "0.8",
                    "--n_panels",
                    "2",
                    "--panel_size",
                    "8",
                    "--target_rates",
                    "0",
                    "0.25",
                    "observed",
                    "--out",
                    str(out),
                ]
            )

            expected = [
                "audit_summary.csv",
                "original_leaderboard.csv",
                "clean_reference_leaderboard.csv",
                "rank_probabilities.csv",
                "sota_margin_by_composition.csv",
                "kendall_tau_by_composition.csv",
                "fragility_summary.csv",
                "advantage_decomposition.csv",
            ]
            for filename in expected:
                self.assertTrue((out / filename).exists(), filename)
            verbose = [
                "audit_annotations.csv",
                "counterfactual_panel_manifest.csv",
                "counterfactual_scores.csv",
                "counterfactual_ranks.csv",
                "sota_margin_by_panel.csv",
                "kendall_tau_by_panel.csv",
                "per_example_advantage.csv",
                "report.md",
                "rank_probability_heatmap.png",
                "rank_probability_heatmap.pdf",
            ]
            for filename in verbose:
                self.assertFalse((out / filename).exists(), filename)

    def test_cli_defaults_to_runs_rank_fragility(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = _dataset()
            data_path = root / "dataset.csv"
            data.to_csv(data_path, index=False)
            pred_dir = root / "predictions"
            _write_predictions(pred_dir, data[data["split"] == "test"])

            old_cwd = Path.cwd()
            os.chdir(root)
            try:
                cli_main(
                    [
                        "--data",
                        str(data_path),
                        "--pred_dir",
                        str(pred_dir),
                        "--task",
                        "classification",
                        "--metric",
                        "auroc",
                        "--baseline_model",
                        "model_a",
                        "--sota_model",
                        "auto",
                        "--n_panels",
                        "1",
                        "--panel_size",
                        "8",
                        "--target_rates",
                        "0",
                    ]
                )
            finally:
                os.chdir(old_cwd)

            out = root / "runs" / "rank_fragility"
            self.assertTrue((out / "fragility_summary.csv").exists())
            self.assertFalse((out / "report.md").exists())

    def test_batch_mode_discovers_records_and_skips_multitask(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runs_root = root / "runs"
            single_dir = runs_root / "moleculenet" / "SingleTask"
            multi_dir = runs_root / "moleculenet" / "MultiTask"
            single_dir.mkdir(parents=True)
            multi_dir.mkdir(parents=True)

            single = _dataset()
            single.rename(columns={"molecule_id": "id", "smiles": "smiles_clean", "y": "label_raw"}).to_csv(
                single_dir / "records.csv", index=False
            )
            (single_dir / "summary.json").write_text(
                '{"config": {"task": "classification", "name": "SingleTask"}}',
                encoding="utf-8",
            )
            multi = single.copy()
            multi["y"] = ["[0, 1]"] * len(multi)
            multi.rename(columns={"molecule_id": "id", "smiles": "smiles_clean", "y": "label_raw"}).to_csv(
                multi_dir / "records.csv", index=False
            )
            (multi_dir / "summary.json").write_text(
                '{"config": {"task": "classification", "name": "MultiTask"}}',
                encoding="utf-8",
            )

            out = root / "rank_fragility"
            cli_main(
                [
                    "--from-runs-root",
                    str(runs_root),
                    "--batch-out-dir",
                    str(out),
                    "--skip-multitask",
                    "--batch-models",
                    "ecfp_linear,ecfp_rf",
                    "--baseline_model",
                    "ecfp_linear",
                    "--n_panels",
                    "1",
                    "--panel_size",
                    "8",
                    "--target_rates",
                    "0",
                    "--rf-estimators",
                    "5",
                    "--n-jobs",
                    "1",
                ]
            )

            manifest = pd.read_csv(out / "batch_manifest.csv")
            self.assertTrue(((manifest["dataset"] == "MultiTask") & (manifest["skip_reason"] == "multitask")).any())
            self.assertTrue((out / "moleculenet" / "SingleTask" / "original_leaderboard.csv").exists())
            self.assertFalse((out / "moleculenet" / "SingleTask" / "_inputs").exists())

    def test_batch_mode_accepts_lgbm_models(self) -> None:
        if importlib.util.find_spec("lightgbm") is None:
            self.skipTest("lightgbm is not installed")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runs_root = root / "runs"
            single_dir = runs_root / "moleculenet" / "SingleTask"
            single_dir.mkdir(parents=True)
            single = _dataset()
            single.rename(columns={"molecule_id": "id", "smiles": "smiles_clean", "y": "label_raw"}).to_csv(
                single_dir / "records.csv", index=False
            )
            (single_dir / "summary.json").write_text(
                '{"config": {"task": "classification", "name": "SingleTask"}}',
                encoding="utf-8",
            )

            out = root / "rank_fragility"
            cli_main(
                [
                    "--from-runs-root",
                    str(runs_root),
                    "--batch-out-dir",
                    str(out),
                    "--skip-multitask",
                    "--batch-models",
                    "ecfp_linear,lgbm_basic,lgbm_advanced",
                    "--baseline_model",
                    "ecfp_linear",
                    "--n_panels",
                    "1",
                    "--panel_size",
                    "8",
                    "--target_rates",
                    "0",
                    "--lgbm-estimators",
                    "3",
                    "--lgbm-advanced-estimators",
                    "4",
                    "--n-jobs",
                    "1",
                ]
            )

            self.assertFalse((out / "moleculenet" / "SingleTask" / "_inputs").exists())
            leaderboard = pd.read_csv(out / "moleculenet" / "SingleTask" / "original_leaderboard.csv")
            self.assertIn("lgbm_basic", set(leaderboard["model"]))
            self.assertIn("lgbm_advanced", set(leaderboard["model"]))

    def test_prune_rank_fragility_outputs_deletes_only_verbose_artifacts(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "prune_rank_fragility_outputs.py"
        spec = importlib.util.spec_from_file_location("prune_rank_fragility_outputs", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules["prune_rank_fragility_outputs"] = module
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "rank_fragility"
            dataset = root / "family" / "dataset"
            inputs = dataset / "_inputs" / "predictions"
            inputs.mkdir(parents=True)
            keep_files = [
                dataset / "audit_summary.csv",
                dataset / "original_leaderboard.csv",
                dataset / "rank_probabilities.csv",
                root / "batch_manifest.csv",
                root / "meta_analysis_tables" / "dataset_original_winners.csv",
            ]
            verbose_files = [
                dataset / "audit_annotations.csv",
                dataset / "counterfactual_panel_manifest.csv",
                dataset / "counterfactual_scores.csv",
                dataset / "counterfactual_ranks.csv",
                dataset / "sota_margin_by_panel.csv",
                dataset / "kendall_tau_by_panel.csv",
                dataset / "per_example_advantage.csv",
                dataset / "report.md",
                dataset / "plot.png",
                dataset / "plot.pdf",
                root / "Thumbs.db",
                inputs / "model.csv",
            ]
            for path in keep_files + verbose_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")
            unrelated = Path(td) / "other" / "counterfactual_scores.csv"
            unrelated.parent.mkdir()
            unrelated.write_text("x", encoding="utf-8")

            module.prune(root, apply=False)
            for path in keep_files + verbose_files:
                self.assertTrue(path.exists(), path)

            module.prune(root, apply=True)
            for path in keep_files:
                self.assertTrue(path.exists(), path)
            for path in verbose_files:
                self.assertFalse(path.exists(), path)
            self.assertTrue(unrelated.exists())

    def test_batch_parser_accepts_torch_mlp_model_options(self) -> None:
        from utils.rank_fragility.run import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--from-runs-root",
                "runs",
                "--batch-models",
                "torch_mlp_basic,torch_mlp_advanced",
                "--mlp-accelerator",
                "cpu",
                "--mlp-devices",
                "1",
                "--mlp-hidden-size",
                "16",
                "--mlp-advanced-hidden-sizes",
                "32,16",
                "--mlp-advanced-max-epochs",
                "2",
            ]
        )
        self.assertEqual(args.batch_models, "torch_mlp_basic,torch_mlp_advanced")
        self.assertEqual(args.mlp_accelerator, "cpu")
        self.assertEqual(args.mlp_advanced_hidden_sizes, "32,16")


if __name__ == "__main__":
    unittest.main()
