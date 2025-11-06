from __future__ import annotations
from typing import Dict, Any, List, Optional
import importlib
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import polaris as po
except ImportError:  # pragma: no cover - optional dependency
    po = None


class BaseLoader:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.info = cfg.get("info", {})

    def get_splits(self) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    def _import_from_str(self, dotted: str):
        mod, _, attr = dotted.rpartition(".")
        if not mod:
            raise ImportError(dotted)
        return getattr(importlib.import_module(mod), attr)

    def _maybe_clean(self, smiles: List[str]) -> pd.DataFrame:
        # Default cleaner is SMILESCleaner unless explicitly disabled
        cleaner_path = self.info.get("cleaner", "utils.cleaner.SMILESCleaner")
        keep_invalid = bool(self.info.get("keep_invalid", True))
        if cleaner_path and str(cleaner_path).lower() not in {"none", "null", "false"}:
            Cleaner = self._import_from_str(cleaner_path)
            cleaner = Cleaner(smiles)
            # Prefer valid rows by default; keep all if you want to audit failures too
            df = (cleaner.get_valid() if not keep_invalid else cleaner.get_data()).copy()
        else:
            df = pd.DataFrame({"smiles": smiles})
        if "smiles_clean" not in df.columns and "smiles" in df.columns:
            df = df.rename(columns={"smiles": "smiles_clean"})
        return df

    
class TDCLoader(BaseLoader):
    def _init_dataset(self):
        name = self.cfg["name"]
        path = self.cfg.get("data_path", "data/tdc/")
        from tdc.single_pred import ADME, Tox, HTS, QM
        for cls in (ADME, Tox, HTS, QM):
            try:
                ds = cls(name=name, path=path)
                _ = ds.y  # probe
                return ds
            except Exception:
                continue
        raise RuntimeError(f"TDC dataset not found: {name}")

    def _pick(self, cols: List[str], frame: pd.DataFrame) -> str:
        for c in cols:
            if c in frame.columns:
                return c
        raise KeyError(f"none of {cols} in {list(frame.columns)}")

    def get_splits(self) -> Dict[str, pd.DataFrame]:
        ds = self._init_dataset()
        method = self.info.get("split")
        raw = ds.get_split(method=method) if method else ds.get_split()

        out = {}
        for split in ("train", "valid", "test"):
            part = raw[split]
            smiles_col = self.info.get("smiles_col") or self._pick(["Drug", "SMILES", "smiles"], part)
            label_col = self.info.get("label_col") or self._pick(["Y", "y", "label"], part)
            id_col = self.info.get("id_col") if self.info.get("id_col") in part.columns else None

            df_clean = self._maybe_clean(part[smiles_col].tolist())
            df_clean["label_raw"] = part[label_col].tolist()
            if id_col:
                df_clean["id"] = part[id_col].tolist()
            out[split] = df_clean
        return out
    

class TabularLoader(BaseLoader):
    DEFAULT_SMILES_COLS = ["smiles", "SMILES", "drug", "Drug"]
    DEFAULT_LABEL_COLS = ["label_raw", "label", "Label", "y", "Y"]
    DEFAULT_ID_COLS = ["id", "ID", "compound_id", "compoundID"]
    DEFAULT_SEQUENCE_COLS = [
        "sequence_aa",
        "sequence",
        "Sequence",
        "protein_sequence",
        "ProteinSequence",
        "target_sequence",
        "TargetSequence",
        "AASequence",
    ]
    DEFAULT_TARGET_ID_COLS = [
        "target_id",
        "target",
        "TargetID",
        "protein_id",
        "ProteinID",
    ]

    def _read_like(self, path: Path) -> pd.DataFrame:
        s = path.suffix.lower()
        if s in {".csv", ".tsv"}:
            return pd.read_csv(path, sep="," if s == ".csv" else "\t")
        if s == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"unsupported file: {path}")

    def _resolve_column(self, df: pd.DataFrame, key: str, candidates: List[str]) -> Optional[str]:
        info_val = self.info.get(key)
        if info_val and info_val in df.columns:
            return info_val
        lower_map = {col.lower(): col for col in df.columns}
        for cand in candidates:
            if cand in df.columns:
                return cand
            cli = cand.lower()
            if cli in lower_map:
                return lower_map[cli]
        return None

    def _standardize_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        smiles_col = self._resolve_column(df, "smiles_col", self.DEFAULT_SMILES_COLS)
        if smiles_col and smiles_col != "smiles":
            df = df.rename(columns={smiles_col: "smiles"})
        if "smiles" not in df.columns:
            raise KeyError("Could not determine SMILES column. Set info.smiles_col in the config.")

        label_col = self._resolve_column(df, "label_col", self.DEFAULT_LABEL_COLS)
        if label_col and label_col != "label_raw":
            df = df.rename(columns={label_col: "label_raw"})
        if "label_raw" not in df.columns:
            raise KeyError("Could not determine label column. Set info.label_col in the config.")

        id_col = self._resolve_column(df, "id_col", self.DEFAULT_ID_COLS)
        if id_col and id_col != "id":
            df = df.rename(columns={id_col: "id"})

        seq_col = self._resolve_column(df, "sequence_col", self.DEFAULT_SEQUENCE_COLS)
        if seq_col and seq_col != "sequence_aa":
            df = df.rename(columns={seq_col: "sequence_aa"})

        target_col = self._resolve_column(df, "target_id_col", self.DEFAULT_TARGET_ID_COLS)
        if target_col and target_col != "target_id":
            df = df.rename(columns={target_col: "target_id"})
        return df

    def get_splits(self) -> Dict[str, pd.DataFrame]:
        # three files
        if "paths" in self.cfg:
            out = {}
            for split in ("train", "valid", "test"):
                df = self._read_like(Path(self.cfg["paths"][split]))
                df = self._standardize_cols(df)
                df_clean = self._maybe_clean(df["smiles"].tolist())
                df_clean["label_raw"] = df["label_raw"].tolist()
                if "id" in df.columns:
                    df_clean["id"] = df["id"].tolist()
                if "sequence_aa" in df.columns:
                    if len(df_clean) != len(df):
                        raise ValueError("Sequence-aware tabular loader expects keep_invalid=True to retain row alignment.")
                    df_clean["sequence_aa"] = df["sequence_aa"].tolist()
                if "target_id" in df.columns:
                    if len(df_clean) != len(df):
                        raise ValueError("Sequence-aware tabular loader expects keep_invalid=True to retain row alignment.")
                    df_clean["target_id"] = df["target_id"].tolist()
                out[split] = df_clean
            return out

        # single file + split column
        if "path" in self.cfg:
            df = self._read_like(Path(self.cfg["path"]))
            df = self._standardize_cols(df)
            split_col = self.info.get("split_col", "split")
            if split_col not in df.columns:
                raise KeyError(f"missing split_col '{split_col}' in {self.cfg['path']}")
            df[split_col] = df[split_col].str.lower().map({"train": "train", "val": "valid", "valid": "valid", "test": "test"})
            out = {}
            for split in ("train", "valid", "test"):
                part = df[df[split_col] == split]
                if part.empty:
                    raise ValueError(f"no rows for split '{split}' in {self.cfg['path']}")
                df_clean = self._maybe_clean(part["smiles"].tolist())
                df_clean["label_raw"] = part["label_raw"].tolist()
                if "id" in part.columns:
                    df_clean["id"] = part["id"].tolist()
                if "sequence_aa" in part.columns:
                    if len(df_clean) != len(part):
                        raise ValueError("Sequence-aware tabular loader expects keep_invalid=True to retain row alignment.")
                    df_clean["sequence_aa"] = part["sequence_aa"].tolist()
                if "target_id" in part.columns:
                    if len(df_clean) != len(part):
                        raise ValueError("Sequence-aware tabular loader expects keep_invalid=True to retain row alignment.")
                    df_clean["target_id"] = part["target_id"].tolist()
                out[split] = df_clean
            return out

        raise ValueError("tabular loader needs 'paths' or 'path'")
    
    
class PolarisLoader(BaseLoader):
    """Minimal Polaris loader.
    Expects cfg = {"type": "polaris", "name": "<vendor/benchmark-id>"}.
    Returns only {'train', 'test'} with columns: smiles_clean, label_raw, id.
    """
    def get_splits(self) -> Dict[str, pd.DataFrame]:
        if po is None:
            raise ImportError(
                "polaris-lib is required for Polaris datasets. Install with 'pip install polaris-lib'."
            )
        bench = po.load_benchmark(self.cfg["name"])
        train, test = bench.get_train_test_split()

        def _to_df(loader) -> pd.DataFrame:
            smiles = loader.inputs
            try:
                y = loader.targets      # TODO: Handle multitask labels...
            except:
                y = [None] * len(smiles)
            if smiles is None or y is None:
                raise ValueError("Missing SMILES or labels")
            df = self._maybe_clean(smiles)
            df["label_raw"] = y
            if "id" not in df.columns:
                df["id"] = np.arange(len(df), dtype=np.int64)
            return df

        return {"train": _to_df(train), "test": _to_df(test)}


class DTILoader(TabularLoader):
    """DTI loader built on TabularLoader with sensible defaults."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        if "keep_invalid" not in self.info:
            # Keep invalid molecules so that auxiliary columns (sequence/target) remain aligned.
            self.info["keep_invalid"] = True

    def _standardize_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super()._standardize_cols(df)
        if "sequence_aa" not in df.columns:
            raise KeyError(
                "DTI loader requires an amino-acid sequence column. "
                "Set info.sequence_col or name the column one of "
                f"{self.DEFAULT_SEQUENCE_COLS}."
            )
        return df
