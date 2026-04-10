from __future__ import annotations


def compile_piece_from_text(description: str) -> dict[str, object]:
    raise NotImplementedError(
        "English-to-DSL compilation is reserved for the M3 LLM integration path. "
        f"Received description: {description!r}"
    )
