from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Optional

from .pydantic_compat import (
    ConfigDict,
    Field,
    HAVE_PYDANTIC,
    PydanticBaseModel,
    PydanticValidationError,
    pydantic_model_dump,
    pydantic_model_validate,
)


def _ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping, got {type(value).__name__}")
    return {str(k): v for k, v in value.items()}


def _normalize_label_cols(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        val = value.strip()
        return [val] if val else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise TypeError("info.label_cols must be a string or a list/tuple of strings")


def _normalize_split_fracs(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError("info.split_fracs must be a 3-element list/tuple")
    if len(value) != 3:
        raise ValueError("info.split_fracs must contain exactly 3 fractions")
    out = [float(v) for v in value]
    if any(v < 0 for v in out):
        raise ValueError("info.split_fracs entries must be non-negative")
    total = sum(out)
    if total <= 0:
        raise ValueError("info.split_fracs must sum to a positive value")
    return out


def _normalize_split_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip().lower()
    else:
        token = str(value).strip().lower()
    mapping = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
    return mapping.get(token)


def normalize_split_column(series) -> Any:
    """Normalize split labels to train/valid/test while preserving pandas semantics."""
    # Keep pandas import local to avoid import-time dependency in lightweight callers.
    import pandas as pd

    normalized = series.map(_normalize_split_name)
    missing_mask = normalized.isna()
    if missing_mask.any():
        bad_values = [
            _repr_value(v)
            for v in pd.Series(series)[missing_mask].drop_duplicates().tolist()
        ]
        raise ValueError(
            "Unsupported split labels. Expected one of "
            "{train, valid, val, test}; got "
            + ", ".join(bad_values[:5])
            + (" ..." if len(bad_values) > 5 else "")
        )
    return normalized


def _repr_value(value: Any) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<{type(value).__name__}>"
    return text


if HAVE_PYDANTIC:  # pragma: no cover - exercised when pydantic is installed
    class LoaderConfigSchema(PydanticBaseModel):
        type: Optional[str] = None
        modality: Optional[str] = None
        task: Optional[str] = None
        name: Optional[str] = None
        id: Optional[str] = None
        path: Optional[str] = None
        paths: Optional[dict[str, Any]] = None
        info: dict[str, Any] = Field(default_factory=dict) if Field is not None else {}
        out: Optional[str] = None
        seed: Optional[int] = None

        if ConfigDict is not None:  # pydantic v2
            model_config = ConfigDict(extra="allow")
        else:  # pydantic v1
            class Config:
                extra = "allow"

    class RunConfigEchoSchema(PydanticBaseModel):
        type: Optional[str] = None
        name: Optional[str] = None
        task: Optional[str] = None
        modality: Optional[str] = None
        info: Optional[dict[str, Any]] = None
        seed: Optional[int] = None
        out: Optional[str] = None

        if ConfigDict is not None:  # pydantic v2
            model_config = ConfigDict(extra="allow")
        else:  # pydantic v1
            class Config:
                extra = "allow"


def validate_yaml_mapping(data: Any, *, source: Optional[Path] = None) -> dict[str, Any]:
    label = f"YAML file '{source}'" if source is not None else "YAML payload"
    if data is None:
        raise ValueError(f"{label} is empty; expected a mapping at the document root")
    return _ensure_mapping(data, context=f"{label} root")


def _normalize_config(cfg: Any, *, require_loader_inputs: bool) -> dict[str, Any]:
    cfg_map = _ensure_mapping(cfg, context="cfg")

    if HAVE_PYDANTIC:  # pragma: no cover - exercised when pydantic is installed
        try:
            parsed = pydantic_model_validate(LoaderConfigSchema, cfg_map)
        except PydanticValidationError as exc:
            raise ValueError(f"Invalid config: {exc}") from exc
        normalized = pydantic_model_dump(parsed, exclude_none=False)
    else:
        normalized = copy.deepcopy(cfg_map)

    normalized = copy.deepcopy(normalized)
    info = normalized.get("info", {})
    if info is None:
        info = {}
    info = _ensure_mapping(info, context="cfg.info")
    normalized["info"] = info

    if "label_cols" in info:
        info["label_cols"] = _normalize_label_cols(info.get("label_cols"))
    if "split_fracs" in info:
        info["split_fracs"] = _normalize_split_fracs(info.get("split_fracs"))

    path = normalized.get("path")
    paths = normalized.get("paths")
    if path is not None and paths is not None:
        raise ValueError("cfg must not define both 'path' and 'paths'")
    if path is not None:
        normalized["path"] = str(path)
    if paths is not None:
        paths_map = _ensure_mapping(paths, context="cfg.paths")
        missing = [k for k in ("train", "valid", "test") if k not in paths_map]
        if missing:
            raise KeyError(f"cfg.paths missing required splits: {missing}")
        normalized["paths"] = {str(k): str(v) for k, v in paths_map.items()}

    loader_kind = str(normalized.get("modality") or normalized.get("type") or "tabular").lower()
    if require_loader_inputs and loader_kind in {"tabular", "dti"} and path is None and paths is None:
        raise ValueError("tabular/dti config must define either 'path' or 'paths'")

    return normalized


def normalize_loader_config(cfg: Any) -> dict[str, Any]:
    """Validate and normalize a loader config without mutating the caller's dict."""
    return _normalize_config(cfg, require_loader_inputs=True)


def normalize_runtime_config(cfg: Any) -> dict[str, Any]:
    """Validate a runtime config passed into run_one_config/builders."""
    return _normalize_config(cfg, require_loader_inputs=False)


def normalize_echo_config(cfg: Any) -> dict[str, Any]:
    cfg_map = _ensure_mapping(cfg, context="cfg")
    if HAVE_PYDANTIC:  # pragma: no cover - exercised when pydantic is installed
        try:
            parsed = pydantic_model_validate(RunConfigEchoSchema, cfg_map)
        except PydanticValidationError as exc:
            raise ValueError(f"Invalid config echo payload: {exc}") from exc
        return pydantic_model_dump(parsed, exclude_unset=True)
    return copy.deepcopy(cfg_map)
