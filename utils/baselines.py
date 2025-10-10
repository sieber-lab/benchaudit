from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
)
from scipy.stats import pearsonr

try:
    import lightgbm as lgb
    _HAVE_LGBM = True
except Exception:
    _HAVE_LGBM = False


@dataclass
class BaselineParams:
    seed: int = 0
    fp_radius: int = 2
    fp_nbits: int = 2048
    # MLP defaults
    mlp_hidden: tuple = (256, 128)
    mlp_max_iter: int = 300
    # RF defaults
    rf_estimators: int = 500
    # LGBM defaults
    lgbm_estimators: int = 800
    lgbm_lr: float = 0.05
    lgbm_leaves: int = 31
    lgbm_subsample: float = 0.8
    lgbm_colsample: float = 0.8


def _cfg_param(cfg: Dict[str, Any], *path, default=None):
    d = cfg
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _task_kind(cfg: Dict[str, Any], y: np.ndarray) -> str:
    kind = cfg.get("task")
    if kind in ("classification", "regression"):
        return kind
    # Fallback heuristic
    y_clean = y[~pd.isna(y)]
    uniq = np.unique(y_clean)
    if len(uniq) <= 10 and set(uniq.tolist()).issubset({0, 1}):
        return "classification"
    return "regression"


def _get_ycol(df: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    info = cfg.get("info", {})
    for c in (info.get("label_col"), "label", "label_raw"):
        if c and c in df.columns:
            return df[c].to_numpy()
    raise KeyError("No label column found in dataframe")


def _morgan_fps(smiles: List[Optional[str]], radius: int, nbits: int):
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if pd.notna(s) else None
        if m is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits))
    return fps


def _fps_to_numpy(fps):
    if not fps:
        return np.zeros((0, 0), dtype=np.float32)
    n_bits = fps[0].GetNumBits() if fps[0] is not None else 2048
    arr = np.zeros((len(fps), n_bits), dtype=np.float32)
    for i, fp in enumerate(fps):
        if fp is None:
            continue
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def _X_from_df(df: pd.DataFrame, cfg: Dict[str, Any], params: BaselineParams) -> np.ndarray:
    info = cfg.get("info", {})
    smi_col = info.get("smiles_col")
    if smi_col is None:
        # prefer cleaned column if present
        smi_col = "smiles_clean" if "smiles_clean" in df.columns else "smiles"
    if smi_col in df.columns:
        fps = _morgan_fps(
            df[smi_col].astype(str).tolist(),
            radius=int(_cfg_param(cfg, "info", "fp_radius", default=params.fp_radius)),
            nbits=int(_cfg_param(cfg, "info", "fp_nbits", default=params.fp_nbits)),
        )
        return _fps_to_numpy(fps)

    # Tabular fallback: all numeric columns except id/labels/smiles
    drop = {info.get("id_col", "id"), info.get("label_col", "label"), "label_raw", "smiles", "smiles_clean"}
    cols = [c for c in df.columns if c not in drop and np.issubdtype(df[c].dtype, np.number)]
    return df[cols].to_numpy(dtype=np.float32)


def _class_models(p: BaselineParams):
    models = {
        "mlp": make_pipeline(StandardScaler(with_mean=False),
                             MLPClassifier(hidden_layer_sizes=p.mlp_hidden, max_iter=p.mlp_max_iter,
                                           random_state=p.seed)),
        "rf": RandomForestClassifier(n_estimators=p.rf_estimators, random_state=p.seed, n_jobs=-1),
    }
    if _HAVE_LGBM:
        models["lgbm"] = lgb.LGBMClassifier(
            n_estimators=p.lgbm_estimators,
            learning_rate=p.lgbm_lr,
            num_leaves=p.lgbm_leaves,
            subsample=p.lgbm_subsample,
            colsample_bytree=p.lgbm_colsample,
            random_state=p.seed,
        )
    return models


def _reg_models(p: BaselineParams):
    models = {
        "mlp": make_pipeline(StandardScaler(with_mean=False),
                             MLPRegressor(hidden_layer_sizes=p.mlp_hidden, max_iter=p.mlp_max_iter,
                                          random_state=p.seed)),
        "rf": RandomForestRegressor(n_estimators=p.rf_estimators, random_state=p.seed, n_jobs=-1),
    }
    if _HAVE_LGBM:
        models["lgbm"] = lgb.LGBMRegressor(
            n_estimators=p.lgbm_estimators,
            learning_rate=p.lgbm_lr,
            num_leaves=p.lgbm_leaves,
            subsample=p.lgbm_subsample,
            colsample_bytree=p.lgbm_colsample,
            random_state=p.seed,
        )
    return models


def eval_baselines_generic(cfg: Dict[str, Any], splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Train on train only, evaluate on test using standard metrics."""
    p = BaselineParams(
        seed=int(cfg.get("seed", 0)),
        fp_radius=int(_cfg_param(cfg, "info", "fp_radius", default=2)),
        fp_nbits=int(_cfg_param(cfg, "info", "fp_nbits", default=2048)),
    )
    train = splits["train"]
    test = splits["test"]

    X_tr = _X_from_df(train, cfg, p)
    y_tr = _get_ycol(train, cfg)
    X_te = _X_from_df(test, cfg, p)
    # test labels may not exist (e.g., Polaris). If present we compute local metrics.
    y_te = None
    try:
        y_te = _get_ycol(test, cfg)
    except Exception:
        pass

    kind = _task_kind(cfg, y_tr)
    results: Dict[str, Any] = {"task": kind, "models": {}}

    if kind == "classification":
        models = _class_models(p)
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            # Use probabilities for metrics
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_te)
                y_pred = prob[:, 1] if prob.ndim == 2 and prob.shape[1] >= 2 else prob.ravel()
            else:
                # fall back to scaled decision function
                dec = model.decision_function(X_te)
                y_pred = (dec - dec.min()) / (dec.max() - dec.min() + 1e-12)

            metrics = {}
            if y_te is not None:
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_te, y_pred))
                except Exception:
                    metrics["roc_auc"] = None
                try:
                    metrics["average_precision"] = float(average_precision_score(y_te, y_pred))
                except Exception:
                    metrics["average_precision"] = None
            results["models"][name] = {"metrics": metrics, "predictions": y_pred.tolist()}
    else:
        models = _reg_models(p)
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            metrics = {}
            if y_te is not None:
                try:
                    metrics["mse"] = float(mean_squared_error(y_te, y_pred))
                except Exception:
                    metrics["mse"] = None
                try:
                    pr = pearsonr(y_te, y_pred)
                    metrics["pearsonr"] = float(getattr(pr, "statistic", pr[0]))
                except Exception:
                    metrics["pearsonr"] = None
            results["models"][name] = {"metrics": metrics, "predictions": y_pred.astype(float).tolist()}

    return results


def eval_baselines_polaris(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Train on train only, get predictions for test, and evaluate via Polaris' API."""
    import polaris as po

    p = BaselineParams(seed=int(cfg.get("seed", 0)))
    bench_name = cfg.get("name")
    if not bench_name:
        raise ValueError("Polaris config must include 'name' (e.g., 'tdcommons/pgp-broccatelli').")

    benchmark = po.load_benchmark(bench_name)
    train, test = benchmark.get_train_test_split()

    # Polaris loaders expose .inputs/.targets; fall back to iteration if needed
    X_tr = np.asarray(getattr(train, "inputs", [x for x, _ in train]))
    y_tr = np.asarray(getattr(train, "targets", [y for _, y in train]))
    X_te = np.asarray(getattr(test, "inputs", [x for x in test]))

    kind = _task_kind(cfg, y_tr)
    out: Dict[str, Any] = {"task": kind, "models": {}}

    if kind == "classification":
        models = _class_models(p)
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_te)
                y_pred = prob[:, 1] if prob.ndim == 2 and prob.shape[1] >= 2 else prob.ravel()
            else:
                dec = model.decision_function(X_te)
                y_pred = (dec - dec.min()) / (dec.max() - dec.min() + 1e-12)

            results = benchmark.evaluate(list(map(float, y_pred)))
            try:
                metrics = results.to_dict() if hasattr(results, "to_dict") else dict(results)
            except Exception:
                metrics = {"polaris_evaluate": True}
            out["models"][name] = {"metrics": metrics, "predictions": list(map(float, y_pred))}
    else:
        models = _reg_models(p)
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            results = benchmark.evaluate(list(map(float, y_pred)))
            try:
                metrics = results.to_dict() if hasattr(results, "to_dict") else dict(results)
            except Exception:
                metrics = {"polaris_evaluate": True}
            out["models"][name] = {"metrics": metrics, "predictions": list(map(float, y_pred))}

    return out


def run_baselines(cfg: Dict[str, Any], splits: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
    """Public entry point. Uses Polaris path when cfg['type']=='polaris', else generic."""
    if cfg.get("type") == "polaris":
        return eval_baselines_polaris(cfg)
    if splits is None:
        raise ValueError("Non-Polaris baselines require preloaded data splits")
    return eval_baselines_generic(cfg, splits)
