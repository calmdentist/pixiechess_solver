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
class ActionIntent:
    action_kind: str
    actor_piece_id: str
    from_square: str | None = None
    to_square: str | None = None
    target_piece_id: str | None = None
    promotion_piece_type: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    params: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("action kind", self.action_kind)
        _validate_identifier("actor piece id", self.actor_piece_id)
        object.__setattr__(self, "from_square", normalize_square(self.from_square))
        object.__setattr__(self, "to_square", normalize_square(self.to_square))
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def stable_id(self) -> str:
        return stable_action_id(self)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "action_kind": self.action_kind,
            "actor_piece_id": self.actor_piece_id,
            "from_square": self.from_square,
            "to_square": self.to_square,
            "target_piece_id": self.target_piece_id,
            "promotion_piece_type": self.promotion_piece_type,
            "tags": list(self.tags),
            "params": dict(self.params),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "ActionIntent":
        return cls(
            action_kind=str(data["action_kind"]),
            actor_piece_id=str(data["actor_piece_id"]),
            from_square=(
                str(data["from_square"])
                if data.get("from_square") is not None
                else None
            ),
            to_square=(
                str(data["to_square"])
                if data.get("to_square") is not None
                else None
            ),
            target_piece_id=(
                str(data["target_piece_id"])
                if data.get("target_piece_id") is not None
                else None
            ),
            promotion_piece_type=(
                str(data["promotion_piece_type"])
                if data.get("promotion_piece_type") is not None
                else None
            ),
            tags=tuple(str(tag) for tag in data.get("tags", [])),
            params=dict(data.get("params", {})),
            metadata=dict(data.get("metadata", {})),
        )


def stable_action_id(action: ActionIntent) -> str:
    from pixie_solver.core.hash import stable_digest

    return stable_digest(action.to_dict())


def action_intent_from_move(move: "Move") -> ActionIntent:
    params = dict(move.metadata)
    if move.captured_piece_id is not None:
        params.setdefault("captured_piece_id", move.captured_piece_id)
    target_piece_id = move.captured_piece_id
    if target_piece_id is None and move.metadata.get("target_piece_id") is not None:
        target_piece_id = str(move.metadata["target_piece_id"])
    return ActionIntent(
        action_kind=move.move_kind,
        actor_piece_id=move.piece_id,
        from_square=move.from_square,
        to_square=move.to_square,
        target_piece_id=target_piece_id,
        promotion_piece_type=move.promotion_piece_type,
        tags=move.tags,
        params=params,
    )


def move_from_action_intent(action: ActionIntent) -> "Move":
    from pixie_solver.core.move import Move

    if action.from_square is None or action.to_square is None:
        raise ValueError("action intent must define from_square and to_square")
    metadata = dict(action.params)
    captured_piece_id = (
        str(metadata["captured_piece_id"])
        if metadata.get("captured_piece_id") is not None
        else None
    )
    if (
        captured_piece_id is None
        and action.action_kind in {"capture", "en_passant_capture"}
        and action.target_piece_id is not None
    ):
        captured_piece_id = action.target_piece_id
    if (
        action.target_piece_id is not None
        and "target_piece_id" not in metadata
        and action.target_piece_id != captured_piece_id
    ):
        metadata["target_piece_id"] = action.target_piece_id
    metadata.pop("captured_piece_id", None)
    return Move(
        piece_id=action.actor_piece_id,
        from_square=action.from_square,
        to_square=action.to_square,
        move_kind=action.action_kind,
        captured_piece_id=captured_piece_id,
        promotion_piece_type=action.promotion_piece_type,
        tags=action.tags,
        metadata=metadata,
    )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pixie_solver.core.move import Move
