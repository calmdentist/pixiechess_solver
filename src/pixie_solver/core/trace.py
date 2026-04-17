from __future__ import annotations

from dataclasses import dataclass, field

from pixie_solver.core.effect import TransitionEffect
from pixie_solver.utils.serialization import JsonValue


@dataclass(frozen=True, slots=True)
class TraceFrame:
    frame_id: int
    phase: str
    event: "Event | None" = None
    effects: tuple[TransitionEffect, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if not self.phase:
            raise ValueError("phase must not be empty")
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "frame_id": self.frame_id,
            "phase": self.phase,
            "event": self.event.to_dict() if self.event is not None else None,
            "effects": [effect.to_dict() for effect in self.effects],
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "TraceFrame":
        from pixie_solver.core.event import Event

        return cls(
            frame_id=int(data["frame_id"]),
            phase=str(data["phase"]),
            event=Event.from_dict(dict(data["event"])) if data.get("event") is not None else None,
            effects=tuple(
                TransitionEffect.from_dict(dict(item))
                for item in data.get("effects", [])
            ),
            notes=tuple(str(note) for note in data.get("notes", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class TransitionTrace:
    action_id: str | None = None
    frames: tuple[TraceFrame, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "action_id": self.action_id,
            "frames": [frame.to_dict() for frame in self.frames],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "TransitionTrace":
        return cls(
            action_id=(
                str(data["action_id"])
                if data.get("action_id") is not None
                else None
            ),
            frames=tuple(
                TraceFrame.from_dict(dict(item))
                for item in data.get("frames", [])
            ),
            metadata=dict(data.get("metadata", {})),
        )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pixie_solver.core.event import Event
