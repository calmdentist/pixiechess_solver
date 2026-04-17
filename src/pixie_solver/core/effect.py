from __future__ import annotations

import re
from dataclasses import dataclass, field

from pixie_solver.utils.serialization import JsonValue
from pixie_solver.utils.squares import normalize_square

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_identifier(label: str, value: str) -> None:
    if not value:
        raise ValueError(f"{label} must not be empty")
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must match {_IDENTIFIER_PATTERN.pattern!r}, got {value!r}"
        )


@dataclass(frozen=True, slots=True)
class TransitionEffect:
    effect_kind: str
    actor_piece_id: str | None = None
    target_piece_id: str | None = None
    target_square: str | None = None
    payload: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    sequence: int = 0

    def __post_init__(self) -> None:
        _validate_identifier("effect kind", self.effect_kind)
        object.__setattr__(self, "target_square", normalize_square(self.target_square))
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "effect_kind": self.effect_kind,
            "actor_piece_id": self.actor_piece_id,
            "target_piece_id": self.target_piece_id,
            "target_square": self.target_square,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "TransitionEffect":
        return cls(
            effect_kind=str(data["effect_kind"]),
            actor_piece_id=(
                str(data["actor_piece_id"])
                if data.get("actor_piece_id") is not None
                else None
            ),
            target_piece_id=(
                str(data["target_piece_id"])
                if data.get("target_piece_id") is not None
                else None
            ),
            target_square=(
                str(data["target_square"])
                if data.get("target_square") is not None
                else None
            ),
            payload=dict(data.get("payload", {})),
            metadata=dict(data.get("metadata", {})),
            sequence=int(data.get("sequence", 0)),
        )
