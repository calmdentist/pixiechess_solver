from __future__ import annotations

from typing import Any


def repair_piece_program(*, description: str, current_program: dict[str, Any], diff: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError(
        "Mismatch-driven repair is reserved for the M3 LLM integration path. "
        f"Received description: {description!r}"
    )
