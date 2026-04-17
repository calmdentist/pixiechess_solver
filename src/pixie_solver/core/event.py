from __future__ import annotations

from dataclasses import dataclass, field

from pixie_solver.core.action import ActionIntent
from pixie_solver.core.effect import TransitionEffect
from pixie_solver.core.move import Move
from pixie_solver.utils.serialization import JsonValue


@dataclass(frozen=True, slots=True)
class Event:
    event_type: str
    actor_piece_id: str | None = None
    target_piece_id: str | None = None
    payload: dict[str, JsonValue] = field(default_factory=dict)
    source_cause: str = "engine"
    sequence: int = 0
    source_action_id: str | None = None
    source_frame_id: int | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_type:
            raise ValueError("event_type must not be empty")
        if not self.source_cause:
            raise ValueError("source_cause must not be empty")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "event_type": self.event_type,
            "actor_piece_id": self.actor_piece_id,
            "target_piece_id": self.target_piece_id,
            "payload": dict(self.payload),
            "source_cause": self.source_cause,
            "sequence": self.sequence,
            "source_action_id": self.source_action_id,
            "source_frame_id": self.source_frame_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "Event":
        return cls(
            event_type=str(data["event_type"]),
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
            payload=dict(data.get("payload", {})),
            source_cause=str(data.get("source_cause", "engine")),
            sequence=int(data.get("sequence", 0)),
            source_action_id=(
                str(data["source_action_id"])
                if data.get("source_action_id") is not None
                else None
            ),
            source_frame_id=(
                int(data["source_frame_id"])
                if data.get("source_frame_id") is not None
                else None
            ),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class StateDelta:
    move: Move | None = None
    action: ActionIntent | None = None
    events: tuple[Event, ...] = ()
    effects: tuple[TransitionEffect, ...] = ()
    changed_piece_ids: tuple[str, ...] = ()
    created_piece_ids: tuple[str, ...] = ()
    removed_piece_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    trace: "TransitionTrace | None" = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "changed_piece_ids", tuple(self.changed_piece_ids))
        object.__setattr__(self, "created_piece_ids", tuple(self.created_piece_ids))
        object.__setattr__(self, "removed_piece_ids", tuple(self.removed_piece_ids))
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "move": self.move.to_dict() if self.move is not None else None,
            "action": self.action.to_dict() if self.action is not None else None,
            "events": [event.to_dict() for event in self.events],
            "effects": [effect.to_dict() for effect in self.effects],
            "changed_piece_ids": list(self.changed_piece_ids),
            "created_piece_ids": list(self.created_piece_ids),
            "removed_piece_ids": list(self.removed_piece_ids),
            "notes": list(self.notes),
            "trace": self.trace.to_dict() if self.trace is not None else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "StateDelta":
        from pixie_solver.core.trace import TransitionTrace

        return cls(
            move=Move.from_dict(data["move"]) if data.get("move") is not None else None,
            action=(
                ActionIntent.from_dict(dict(data["action"]))
                if data.get("action") is not None
                else None
            ),
            events=tuple(Event.from_dict(item) for item in data.get("events", [])),
            effects=tuple(
                TransitionEffect.from_dict(dict(item))
                for item in data.get("effects", [])
            ),
            changed_piece_ids=tuple(
                str(piece_id) for piece_id in data.get("changed_piece_ids", [])
            ),
            created_piece_ids=tuple(
                str(piece_id) for piece_id in data.get("created_piece_ids", [])
            ),
            removed_piece_ids=tuple(
                str(piece_id) for piece_id in data.get("removed_piece_ids", [])
            ),
            notes=tuple(str(note) for note in data.get("notes", [])),
            trace=(
                TransitionTrace.from_dict(dict(data["trace"]))
                if data.get("trace") is not None
                else None
            ),
            metadata=dict(data.get("metadata", {})),
        )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pixie_solver.core.trace import TransitionTrace
