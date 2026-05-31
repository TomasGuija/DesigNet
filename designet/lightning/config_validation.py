from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def ensure_matching_config(
    *,
    model_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    pairs: Iterable[tuple[str, str]],
) -> None:
    """Fail early when model and data settings that define tensor shapes differ."""
    mismatches = []
    for model_key, data_key in pairs:
        model_value = _normalize_value(model_config[model_key])
        data_value = _normalize_value(data_config[data_key])
        if model_value != data_value:
            mismatches.append((model_key, data_key, model_value, data_value))

    if not mismatches:
        return

    details = ", ".join(
        f"model.{model_key}={model_value!r} != data.{data_key}={data_value!r}"
        for model_key, data_key, model_value, data_value in mismatches
    )
    raise ValueError(f"Model/data configuration mismatch: {details}")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value
