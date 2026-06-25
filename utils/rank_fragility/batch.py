"""Batch runner for rank-fragility analysis over BenchAudit artifacts."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit import DataStructs
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .chem import morgan_fingerprint
from .config import RunConfig

LOG = logging.getLogger(__name__)


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text).strip()).strip("-._")
    return value or "dataset"


def _parse_csv(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    out: list[str] = []
    for item in value:
        out.extend(_parse_csv(str(item)))
    return out


def _parse_target_rates(value: str | Sequence[str]) -> tuple[float | str, ...]:
    out: list[float | str] = []
    for token in _parse_csv(value):
        out.append("observed" if token.lower() == "observed" else float(token))
    return tuple(out)


def _parse_hidden_sizes(value: int | str | Sequence[int | str]) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, str):
        tokens = _parse_csv(value)
    else:
        tokens = []
        for item in value:
            tokens.extend(_parse_csv(str(item)))
    hidden = tuple(int(token) for token in tokens)
    if not hidden or any(size <= 0 for size in hidden):
        raise ValueError("MLP hidden sizes must be one or more positive integers")
    return hidden


def _normalize_model_name(model_name: str) -> str:
    name = str(model_name).strip()
    aliases = {
        "mlp": "torch_mlp_basic",
        "mlp_basic": "torch_mlp_basic",
        "mlp_advanced": "torch_mlp_advanced",
        "torch_mlp": "torch_mlp_basic",
        "lgbm": "lgbm_basic",
    }
    return aliases.get(name, name)


def _requires_lgbm(model_name: str) -> bool:
    return _normalize_model_name(model_name) in {"lgbm_basic", "lgbm_advanced"}


def _requires_torch_mlp(model_name: str) -> bool:
    return _normalize_model_name(model_name) in {"torch_mlp_basic", "torch_mlp_advanced"}


def _normalize_mlp_accelerator(accelerator: str) -> str:
    value = str(accelerator).strip().lower()
    if value == "cuda":
        return "gpu"
    if value not in {"auto", "cpu", "gpu", "mps"}:
        raise ValueError("--mlp-accelerator must be one of: auto, cpu, gpu, cuda, or mps")
    return value


def _lightning_devices_arg(raw: Any) -> Any:
    text = str(raw).strip()
    if text.lower() == "auto":
        return "auto"
    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if parts and all(part.lstrip("-").isdigit() for part in parts):
            return [int(part) for part in parts]
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def validate_batch_model_backends(models: Sequence[str], args: argparse.Namespace) -> None:
    """Validate optional backend availability for requested batch models."""
    requested = [_normalize_model_name(model) for model in models]
    unsupported = sorted(
        set(requested)
        - {
            "ecfp_linear",
            "ecfp_rf",
            "lgbm_basic",
            "lgbm_advanced",
            "torch_mlp_basic",
            "torch_mlp_advanced",
        }
    )
    if unsupported:
        raise ValueError(f"unsupported batch prediction model(s): {', '.join(unsupported)}")

    missing: list[str] = []
    if any(model in {"lgbm_basic", "lgbm_advanced"} for model in requested):
        if importlib.util.find_spec("lightgbm") is None:
            missing.append("lightgbm")
    if any(model in {"torch_mlp_basic", "torch_mlp_advanced"} for model in requested):
        if importlib.util.find_spec("torch") is None:
            missing.append("torch")
        if importlib.util.find_spec("lightning") is None:
            missing.append("lightning")
    if missing:
        raise RuntimeError(
            "Missing required model dependencies for rank-fragility batch predictions: "
            + ", ".join(sorted(set(missing)))
        )

    accelerator = _normalize_mlp_accelerator(getattr(args, "mlp_accelerator", "auto"))
    if any(model in {"torch_mlp_basic", "torch_mlp_advanced"} for model in requested) and accelerator in {"gpu", "mps"}:
        import torch

        if accelerator == "gpu" and not bool(torch.cuda.is_available()):
            raise RuntimeError(
                "torch MLP requested --mlp-accelerator gpu, but torch.cuda.is_available() is False. "
                "Use a CUDA-enabled environment or pass --mlp-accelerator cpu/auto."
            )
        if accelerator == "mps":
            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            if mps_backend is None or not bool(mps_backend.is_available()):
                raise RuntimeError("torch MLP requested --mlp-accelerator mps, but MPS is not available.")


def _parse_label(value: Any) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        raw = list(value)
    elif value is None:
        raw = [np.nan]
    elif isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            raw = [np.nan]
        elif text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                raw = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
            except Exception:
                raw = [np.nan]
        else:
            raw = [text]
    else:
        raw = [value]

    labels: list[float] = []
    for item in raw:
        if item is None:
            labels.append(np.nan)
            continue
        try:
            if pd.isna(item):
                labels.append(np.nan)
                continue
        except TypeError:
            pass
        try:
            labels.append(float(item))
        except Exception:
            labels.append(np.nan)
    return labels or [np.nan]


def _label_widths(labels: pd.Series) -> pd.Series:
    return labels.map(lambda value: len(_parse_label(value)))


def _single_label(value: Any) -> float:
    labels = _parse_label(value)
    return float(labels[0]) if labels else np.nan


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_rank_fragility_datasets(
    runs_root: Path,
    datasets: str = "all",
    skip_multitask: bool = True,
    include_dti: bool = False,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Discover eligible rank-fragility datasets from existing run artifacts."""
    filters = set() if str(datasets).strip().lower() == "all" else set(_parse_csv(datasets))
    found: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for records_path in sorted(runs_root.glob("*/*/records.csv")):
        run_dir = records_path.parent
        family = run_dir.parent.name
        dataset = run_dir.name
        if filters and dataset not in filters and f"{family}/{dataset}" not in filters:
            continue
        row = {"family": family, "dataset": dataset, "run_dir": str(run_dir), "status": "included", "skip_reason": ""}
        if family == "dti" and not include_dti:
            row.update(status="skipped", skip_reason="dti_pair_dataset_not_molecular_property")
            manifest.append(row)
            continue
        summary = _load_summary(run_dir / "summary.json")
        task = str(summary.get("config", {}).get("task") or summary.get("task", {}).get("type") or "").lower()
        if task not in {"classification", "regression"}:
            row.update(status="skipped", skip_reason="missing_or_unknown_task")
            manifest.append(row)
            continue
        try:
            label_frame = pd.read_csv(records_path, usecols=["split", "label_raw"])
        except Exception as exc:
            row.update(status="skipped", skip_reason=f"cannot_read_labels: {exc}")
            manifest.append(row)
            continue
        label_frame["split"] = label_frame["split"].astype(str).str.lower()
        train_like = label_frame["split"].isin({"train", "valid"})
        test_like = label_frame["split"].eq("test")
        train_label_count = int(label_frame.loc[train_like, "label_raw"].notna().sum())
        test_label_count = int(label_frame.loc[test_like, "label_raw"].notna().sum())
        row["train_label_count"] = train_label_count
        row["test_label_count"] = test_label_count
        if train_label_count == 0:
            row.update(status="skipped", skip_reason="no_train_labels")
            manifest.append(row)
            continue
        if test_label_count == 0:
            row.update(status="skipped", skip_reason="no_test_labels")
            manifest.append(row)
            continue
        labels = label_frame["label_raw"]
        max_width = int(_label_widths(labels).max()) if len(labels) else 0
        row["label_task_count"] = max_width
        if skip_multitask and max_width > 1:
            row.update(status="skipped", skip_reason="multitask")
            manifest.append(row)
            continue
        found.append({"family": family, "dataset": dataset, "run_dir": run_dir, "records_path": records_path, "task": task})
        manifest.append(row)
    return found, pd.DataFrame(manifest)


def _records_to_dataset(records_path: Path, task: str) -> pd.DataFrame:
    records = pd.read_csv(records_path)
    required = {"smiles_clean", "label_raw", "split"}
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"records.csv missing required column(s): {', '.join(missing)}")
    df = records.copy()
    df["split"] = df["split"].astype(str).str.lower()
    df = df[df["split"].isin({"train", "valid", "test"})].copy()
    if "valid" in df.columns:
        df = df[df["valid"].fillna(False).astype(bool)].copy()
    df["y"] = df["label_raw"].map(_single_label)
    df = df[np.isfinite(df["y"].astype(float))].copy()
    df["smiles"] = df["smiles_clean"].astype("string").fillna("").astype(str)
    df = df[df["smiles"].str.strip() != ""].copy()
    if "id" in df.columns and not df["id"].astype(str).duplicated().any():
        df["molecule_id"] = df["id"].astype(str)
    else:
        df["molecule_id"] = [f"row_{idx}" for idx in range(len(df))]
    if task == "classification":
        labels = set(df["y"].dropna().astype(float).unique().tolist())
        if not labels.issubset({0.0, 1.0}):
            raise ValueError(f"non-binary classification labels: {sorted(labels)[:10]}")
    return df[["molecule_id", "smiles", "y", "split"]].reset_index(drop=True)


def _fingerprint_matrix(smiles: Sequence[str], n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for idx, smi in enumerate(smiles):
        fp = morgan_fingerprint(str(smi), n_bits=n_bits)
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, arr[idx])
    return arr


def _constant_class_prediction(y_train: np.ndarray, n_test: int) -> np.ndarray | None:
    labels = np.unique(y_train[np.isfinite(y_train)])
    if len(labels) < 2:
        return np.full(n_test, float(labels[0]) if len(labels) else 0.0, dtype=float)
    return None


def _fit_predict_ecfp_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    task: str,
    seed: int,
) -> np.ndarray:
    if task == "classification":
        constant = _constant_class_prediction(y_train, len(x_test))
        if constant is not None:
            return constant
        model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=1000, random_state=seed, n_jobs=1),
        )
        model.fit(x_train, y_train.astype(int))
        return model.predict_proba(x_test)[:, 1].astype(float)
    model = make_pipeline(StandardScaler(with_mean=False), Ridge(random_state=seed))
    model.fit(x_train, y_train)
    return model.predict(x_test).astype(float)


def _fit_predict_ecfp_rf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    task: str,
    seed: int,
    n_jobs: int,
    rf_estimators: int,
) -> np.ndarray:
    if task == "classification":
        constant = _constant_class_prediction(y_train, len(x_test))
        if constant is not None:
            return constant
        model = RandomForestClassifier(n_estimators=rf_estimators, random_state=seed, n_jobs=n_jobs)
        model.fit(x_train, y_train.astype(int))
        return model.predict_proba(x_test)[:, 1].astype(float)
    model = RandomForestRegressor(n_estimators=rf_estimators, random_state=seed, n_jobs=n_jobs)
    model.fit(x_train, y_train)
    return model.predict(x_test).astype(float)


def _fit_predict_lgbm(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    task: str,
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    import lightgbm as lgb

    canonical = _normalize_model_name(model_name)
    if canonical == "lgbm_advanced":
        params = {
            "n_estimators": args.lgbm_advanced_estimators,
            "learning_rate": args.lgbm_advanced_learning_rate,
            "num_leaves": args.lgbm_advanced_num_leaves,
            "min_child_samples": args.lgbm_advanced_min_child_samples,
            "subsample": args.lgbm_advanced_subsample,
            "colsample_bytree": args.lgbm_advanced_colsample,
            "reg_alpha": args.lgbm_advanced_reg_alpha,
            "reg_lambda": args.lgbm_advanced_reg_lambda,
        }
    else:
        params = {
            "n_estimators": args.lgbm_estimators,
            "learning_rate": args.lgbm_learning_rate,
            "num_leaves": args.lgbm_basic_num_leaves,
            "min_child_samples": args.lgbm_basic_min_child_samples,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        }

    common = {
        **params,
        "random_state": seed,
        "n_jobs": args.n_jobs,
        "verbosity": -1,
        "device_type": args.lgbm_device_type,
    }
    if task == "classification":
        constant = _constant_class_prediction(y_train, len(x_test))
        if constant is not None:
            return constant
        model = lgb.LGBMClassifier(**common)
        model.fit(x_train, np.rint(y_train).astype(int))
        proba = model.predict_proba(x_test)
        if proba.shape[1] == 1:
            return np.full(len(x_test), float(model.classes_[0]), dtype=float)
        pos_idx = int(np.where(model.classes_ == 1)[0][0]) if 1 in model.classes_ else -1
        return np.asarray(proba[:, pos_idx], dtype=float)

    model = lgb.LGBMRegressor(**common)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def _lightning_imports():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    try:
        import lightning.pytorch as pl
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("lightning is required for torch MLP batch predictions") from exc
    return torch, TensorDataset, DataLoader, pl


def _mlp_config(model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    canonical = _normalize_model_name(model_name)
    if canonical == "torch_mlp_advanced":
        return {
            "hidden_sizes": _parse_hidden_sizes(args.mlp_advanced_hidden_sizes),
            "max_epochs": args.mlp_advanced_max_epochs,
            "lr": args.mlp_advanced_lr,
            "weight_decay": args.mlp_advanced_weight_decay,
            "dropout": args.mlp_advanced_dropout,
        }
    return {
        "hidden_sizes": (int(args.mlp_hidden_size),),
        "max_epochs": args.mlp_max_epochs,
        "lr": args.mlp_lr,
        "weight_decay": args.mlp_weight_decay,
        "dropout": 0.0,
    }


def _fit_predict_torch_mlp(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    task: str,
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    torch, TensorDataset, DataLoader, pl = _lightning_imports()
    cfg = _mlp_config(model_name, args)
    accelerator = _normalize_mlp_accelerator(args.mlp_accelerator)
    trainer_devices = _lightning_devices_arg(args.mlp_devices)

    if task == "classification":
        constant = _constant_class_prediction(y_train, len(x_test))
        if constant is not None:
            return constant

    pl.seed_everything(seed, workers=True)
    scaler = StandardScaler(with_mean=False)
    x_tr = scaler.fit_transform(x_train).astype(np.float32)
    x_te = scaler.transform(x_test).astype(np.float32)
    y_tr = y_train.astype(np.float32)

    class LitMLP(pl.LightningModule):
        def __init__(self, in_dim: int) -> None:
            super().__init__()
            layers: list[Any] = []
            prev = in_dim
            for hidden in cfg["hidden_sizes"]:
                layers.append(torch.nn.Linear(prev, int(hidden)))
                layers.append(torch.nn.ReLU())
                if float(cfg["dropout"]) > 0:
                    layers.append(torch.nn.Dropout(float(cfg["dropout"])))
                prev = int(hidden)
            layers.append(torch.nn.Linear(prev, 1))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            if task == "classification":
                return torch.nn.functional.binary_cross_entropy_with_logits(pred, y)
            return torch.nn.functional.mse_loss(pred, y)

        def configure_optimizers(self):
            return torch.optim.Adam(
                self.parameters(),
                lr=float(cfg["lr"]),
                weight_decay=float(cfg["weight_decay"]),
            )

    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    train_loader = DataLoader(train_ds, batch_size=min(args.mlp_batch_size, max(1, len(train_ds))), shuffle=True)
    model = LitMLP(x_tr.shape[1])
    trainer = pl.Trainer(
        max_epochs=int(cfg["max_epochs"]),
        accelerator=accelerator,
        devices=trainer_devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
    )
    trainer.fit(model, train_loader)
    model.eval()
    device = model.device
    pred_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_te)),
        batch_size=min(args.mlp_batch_size, max(1, len(x_te))),
    )
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for (x_batch,) in pred_loader:
            chunks.append(model(x_batch.to(device)).detach().cpu().numpy())
    raw = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
    if task == "classification":
        return (1.0 / (1.0 + np.exp(-raw))).astype(float)
    return raw.astype(float)


def _write_predictions(
    pred_dir: Path,
    dataset_df: pd.DataFrame,
    train_splits: Sequence[str],
    task: str,
    models: Sequence[str],
    args: argparse.Namespace,
) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    for old_prediction in pred_dir.glob("*.csv"):
        old_prediction.unlink()
    train = dataset_df[dataset_df["split"].isin(train_splits)].copy()
    test = dataset_df[dataset_df["split"].eq("test")].copy()
    if train.empty or test.empty:
        raise ValueError("empty train/test after split filtering")
    x_train = _fingerprint_matrix(train["smiles"].tolist(), n_bits=args.fp_nbits)
    x_test = _fingerprint_matrix(test["smiles"].tolist(), n_bits=args.fp_nbits)
    y_train = train["y"].astype(float).to_numpy()
    for model_name in models:
        canonical = _normalize_model_name(model_name)
        if canonical == "ecfp_linear":
            pred = _fit_predict_ecfp_linear(x_train, y_train, x_test, task, args.random_seed)
        elif canonical == "ecfp_rf":
            pred = _fit_predict_ecfp_rf(
                x_train, y_train, x_test, task, args.random_seed, args.n_jobs, args.rf_estimators
            )
        elif _requires_lgbm(canonical):
            pred = _fit_predict_lgbm(canonical, x_train, y_train, x_test, task, args.random_seed, args)
        elif _requires_torch_mlp(canonical):
            pred = _fit_predict_torch_mlp(canonical, x_train, y_train, x_test, task, args.random_seed, args)
        else:
            raise ValueError(f"unsupported batch prediction model: {model_name}")
        pd.DataFrame(
            {
                "molecule_id": test["molecule_id"].astype(str).to_numpy(),
                "y_true": test["y"].astype(float).to_numpy(),
                "y_pred": pred,
            }
        ).to_csv(pred_dir / f"{model_name}.csv", index=False)


def run_batch_rank_fragility(args: argparse.Namespace) -> None:
    """Run rank-fragility analysis for all discovered batch datasets."""
    from .run import run_analysis

    runs_root = Path(args.from_runs_root).resolve()
    out_root = Path(args.batch_out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    models = _parse_csv(args.batch_models)
    validate_batch_model_backends(models, args)
    train_splits = tuple(_parse_csv(args.train_splits))
    discovered, manifest = discover_rank_fragility_datasets(
        runs_root,
        datasets=args.datasets,
        skip_multitask=args.skip_multitask,
        include_dti=args.include_dti,
    )
    selected = discovered[: int(args.max_datasets)] if args.max_datasets else discovered
    selected_keys = {(item["family"], item["dataset"]) for item in selected}
    manifest_rows: list[dict[str, Any]] = []
    for row in manifest.to_dict("records"):
        key = (row.get("family"), row.get("dataset"))
        if row.get("status") == "skipped":
            manifest_rows.append(row)
        elif args.max_datasets and key not in selected_keys:
            row = dict(row)
            row.update(status="skipped", skip_reason="max_datasets_limit")
            manifest_rows.append(row)

    for item in selected:
        family = item["family"]
        dataset = item["dataset"]
        task = item["task"]
        dataset_out = out_root / family / _slug(dataset)
        if args.skip_existing and (dataset_out / "fragility_summary.csv").exists():
            LOG.info("skipping existing rank-fragility output: %s", dataset_out)
            metric = args.classification_metric if task == "classification" else args.regression_metric
            manifest_rows.append(
                {
                    "family": family,
                    "dataset": dataset,
                    "run_dir": str(item["run_dir"]),
                    "status": "completed",
                    "skip_reason": "existing_output",
                    "output_dir": str(dataset_out),
                    "task": task,
                    "metric": metric,
                    "models": ",".join(models),
                    "baseline_model": args.baseline_model,
                }
            )
            continue
        try:
            dataset_df = _records_to_dataset(item["records_path"], task=task)
            dataset_out.mkdir(parents=True, exist_ok=True)
            metric = args.classification_metric if task == "classification" else args.regression_metric
            with tempfile.TemporaryDirectory(prefix="rank_fragility_inputs_") as tmp:
                input_dir = Path(tmp)
                pred_dir = input_dir / "predictions"
                dataset_csv = input_dir / "dataset.csv"
                dataset_df.to_csv(dataset_csv, index=False)
                _write_predictions(
                    pred_dir=pred_dir,
                    dataset_df=dataset_df,
                    train_splits=train_splits,
                    task=task,
                    models=models,
                    args=args,
                )
                run_analysis(
                    RunConfig(
                        data=dataset_csv,
                        pred_dir=pred_dir,
                        task=task,
                        metric=metric,
                        baseline_model=args.baseline_model,
                        sota_model=args.sota_model,
                        near_leak_thresholds=tuple(args.near_leak_thresholds),
                        primary_near_leak_threshold=args.primary_near_leak_threshold,
                        regression_conflict_threshold=args.regression_conflict_threshold,
                        regression_conflict_threshold_sensitivity=args.regression_conflict_threshold_sensitivity,
                        random_seed=args.random_seed,
                        panel_size=args.panel_size,
                        n_panels=args.n_panels,
                        target_rates=_parse_target_rates(args.target_rates),
                        output_dir=dataset_out,
                    )
                )
            manifest_rows.append(
                {
                    "family": family,
                    "dataset": dataset,
                    "run_dir": str(item["run_dir"]),
                    "status": "completed",
                    "skip_reason": "",
                    "output_dir": str(dataset_out),
                    "task": task,
                    "metric": metric,
                    "models": ",".join(models),
                    "baseline_model": args.baseline_model,
                }
            )
        except Exception as exc:
            LOG.exception("rank-fragility batch failed for %s/%s", family, dataset)
            manifest_rows.append(
                {
                    "family": family,
                    "dataset": dataset,
                    "run_dir": str(item["run_dir"]),
                    "status": "failed",
                    "skip_reason": str(exc),
                    "output_dir": str(dataset_out),
                    "task": task,
                }
            )
    manifest_out = pd.DataFrame(manifest_rows)
    manifest_out.to_csv(out_root / "batch_manifest.csv", index=False)
    LOG.info("wrote rank-fragility batch manifest -> %s", out_root / "batch_manifest.csv")
