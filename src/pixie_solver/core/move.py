from __future__ import annotations

from dataclasses import dataclass, field

from pixie_solver.utils.serialization import JsonValue
from pixie_solver.utils.squares import normalize_square


@dataclass(frozen=True, slots=True)
class Move:
    piece_id: str
    from_square: str
    to_square: str
    move_kind: str = "move"
    captured_piece_id: str | None = None
    promotion_piece_type: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.piece_id:
            raise ValueError("move piece_id must not be empty")
        if not self.move_kind:
            raise ValueError("move_kind must not be empty")
        object.__setattr__(self, "from_square", normalize_square(self.from_square))
        object.__setattr__(self, "to_square", normalize_square(self.to_square))
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def stable_id(self) -> str:
        return stable_move_id(self)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "piece_id": self.piece_id,
            "from_square": self.from_square,
            "to_square": self.to_square,
            "move_kind": self.move_kind,
            "captured_piece_id": self.captured_piece_id,
            "promotion_piece_type": self.promotion_piece_type,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "Move":
        return cls(
            piece_id=str(data["piece_id"]),
            from_square=str(data["from_square"]),
            to_square=str(data["to_square"]),
            move_kind=str(data.get("move_kind", "move")),
            captured_piece_id=(
                str(data["captured_piece_id"])
                if data.get("captured_piece_id") is not None
                else None
            ),
            promotion_piece_type=(
                str(data["promotion_piece_type"])
                if data.get("promotion_piece_type") is not None
                else None
            ),
            tags=tuple(str(tag) for tag in data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )


def stable_move_id(move: Move) -> str:
    from pixie_solver.core.hash import stable_digest

    return stable_digest(move.to_dict())
