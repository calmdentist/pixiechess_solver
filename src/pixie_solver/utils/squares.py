from __future__ import annotations

import re

SQUARE_PATTERN = re.compile(r"^[a-h][1-8]$")


def normalize_square(square: str | None) -> str | None:
    if square is None:
        return None

    normalized = square.strip().lower()
    if not SQUARE_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid square: {square!r}")
    return normalized


def is_valid_square(square: str | None) -> bool:
    try:
        normalize_square(square)
    except ValueError:
        return False
    return True


def square_to_coords(square: str) -> tuple[int, int]:
    normalized = normalize_square(square)
    return ord(normalized[0]) - ord("a"), int(normalized[1]) - 1


def coords_to_square(file_index: int, rank_index: int) -> str | None:
    if not (0 <= file_index < 8 and 0 <= rank_index < 8):
        return None
    return f"{chr(ord('a') + file_index)}{rank_index + 1}"
