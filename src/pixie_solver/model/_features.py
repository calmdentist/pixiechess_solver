from __future__ import annotations

import hashlib
import math

from pixie_solver.core.piece import BasePieceType, Color
from pixie_solver.utils.serialization import JsonValue
from pixie_solver.utils.squares import square_to_coords

BASE_PIECE_TYPE_IDS = {
    BasePieceType.PAWN: 1,
    BasePieceType.KNIGHT: 2,
    BasePieceType.BISHOP: 3,
    BasePieceType.ROOK: 4,
    BasePieceType.QUEEN: 5,
    BasePieceType.KING: 6,
}
COLOR_IDS = {
    Color.WHITE: 1,
    Color.BLACK: 2,
}
FIELD_TYPE_IDS = {
    "bool": 1,
    "int": 2,
    "float": 3,
    "str": 4,
}
PROMOTION_PIECE_TYPE_IDS = {
    None: 0,
    "knight": 1,
    "bishop": 2,
    "rook": 3,
    "queen": 4,
}
COMMON_MOVE_KIND_IDS = {
    "move": 1,
    "capture": 2,
    "push_capture": 3,
    "castle": 4,
    "en_passant_capture": 5,
}
BASE_PIECE_VALUES = {
    BasePieceType.PAWN: 1.0,
    BasePieceType.KNIGHT: 3.0,
    BasePieceType.BISHOP: 3.25,
    BasePieceType.ROOK: 5.0,
    BasePieceType.QUEEN: 9.0,
    BasePieceType.KING: 0.0,
}


def stable_bucket(namespace: str, value: str, bucket_count: int) -> int:
    if bucket_count < 1:
        raise ValueError("bucket_count must be positive")
    digest = hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % bucket_count + 1


def square_index(square: str | None) -> int:
    if square is None:
        return 0
    file_index, rank_index = square_to_coords(square)
    return rank_index * 8 + file_index + 1


def move_kind_id(move_kind: str, *, extra_buckets: int = 32) -> int:
    known_id = COMMON_MOVE_KIND_IDS.get(move_kind)
    if known_id is not None:
        return known_id
    return len(COMMON_MOVE_KIND_IDS) + stable_bucket(
        "move_kind", move_kind, extra_buckets
    )


def normalize_scalar(value: float, *, scale: float = 8.0) -> float:
    return math.tanh(float(value) / scale)


def json_scalar_features(value: JsonValue) -> tuple[float, float, float, float, float]:
    if value is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    if isinstance(value, bool):
        return (1.0 if value else -1.0, 0.125, 1.0, 0.0, 0.0)
    if isinstance(value, (int, float)):
        numeric = float(value)
        return (
            normalize_scalar(numeric, scale=8.0),
            normalize_scalar(abs(numeric), scale=16.0),
            0.0,
            1.0 if numeric < 0 else 0.0,
            0.0,
        )
    if isinstance(value, str):
        return (
            0.0,
            normalize_scalar(len(value), scale=8.0),
            0.0,
            0.0,
            1.0,
        )
    return (0.0, 0.0, 0.0, 0.0, 0.0)


def json_value_token(value: JsonValue, *, namespace: str, bucket_count: int) -> int:
    if value is None:
        return 0
    return stable_bucket(namespace, repr(value), bucket_count)


def material_score(state, color: Color) -> float:
    total = 0.0
    for piece in state.active_pieces():
        if piece.color != color:
            continue
        piece_class = state.piece_classes[piece.piece_class_id]
        total += BASE_PIECE_VALUES[piece_class.base_piece_type]
    return total
