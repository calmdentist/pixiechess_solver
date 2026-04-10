from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_piece_program_text(
    text: str, *, format_hint: str | None = None
) -> dict[str, Any]:
    normalized_hint = (format_hint or "").lower().strip()
    if normalized_hint in {"", "json"}:
        return json.loads(text)

    if normalized_hint in {"yaml", "yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "YAML parsing requires PyYAML; use JSON or install PyYAML."
            ) from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError("Piece program must deserialize into a mapping")
        return loaded

    raise ValueError(f"Unsupported format_hint: {format_hint!r}")


def load_piece_program(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        format_hint = "json"
    elif suffix in {".yaml", ".yml"}:
        format_hint = "yaml"
    else:
        raise ValueError(
            f"Unsupported piece program file extension {suffix!r}. "
            "Use .json, .yaml, or .yml."
        )

    return parse_piece_program_text(file_path.read_text(encoding="utf-8"), format_hint=format_hint)
