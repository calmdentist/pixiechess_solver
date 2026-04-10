from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def to_primitive(value: Any) -> JsonValue:
    """Convert nested dataclasses and enums into JSON-safe primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return {
            field.name: to_primitive(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): to_primitive(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    raise TypeError(f"Unsupported value for JSON serialization: {type(value)!r}")


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    """Serialize a value using deterministic key ordering."""
    return json.dumps(
        to_primitive(value),
        ensure_ascii=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    )
