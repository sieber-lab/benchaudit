from __future__ import annotations

from typing import Any, Type, TypeVar

try:  # pragma: no cover - exercised when pydantic is installed
    import pydantic as _pydantic
except Exception:  # pragma: no cover - local/dev environments may not have pydantic
    _pydantic = None


HAVE_PYDANTIC = _pydantic is not None
PYDANTIC_V2 = bool(HAVE_PYDANTIC and hasattr(_pydantic.BaseModel, "model_validate"))

if HAVE_PYDANTIC:  # pragma: no cover - import-path shim
    PydanticBaseModel = _pydantic.BaseModel
    PydanticValidationError = _pydantic.ValidationError
    ConfigDict = getattr(_pydantic, "ConfigDict", None)
    Field = getattr(_pydantic, "Field", None)
else:  # pragma: no cover - fallback types
    PydanticBaseModel = object
    PydanticValidationError = ValueError
    ConfigDict = None
    Field = None


T = TypeVar("T")


def pydantic_model_validate(model_cls: Type[T], payload: Any) -> T:
    """Compatibility wrapper for pydantic v1/v2 model validation."""
    if not HAVE_PYDANTIC:  # pragma: no cover - guarded by callers
        raise RuntimeError("pydantic is not installed")
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)  # type: ignore[attr-defined]
    return model_cls.parse_obj(payload)  # type: ignore[attr-defined]


def pydantic_model_dump(instance: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility wrapper for pydantic v1/v2 model dumps."""
    if not HAVE_PYDANTIC:  # pragma: no cover - guarded by callers
        raise RuntimeError("pydantic is not installed")
    if hasattr(instance, "model_dump"):
        return instance.model_dump(**kwargs)  # type: ignore[attr-defined]
    return instance.dict(**kwargs)  # type: ignore[attr-defined]
