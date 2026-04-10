from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from pixie_solver.utils.serialization import JsonValue
from pixie_solver.utils.squares import normalize_square

IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
STATE_FIELD_TYPES = frozenset({"bool", "int", "float", "str"})


def _validate_identifier(label: str, value: str) -> None:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must match {IDENTIFIER_PATTERN.pattern!r}, got {value!r}"
        )


def _matches_field_type(field_type: str, value: JsonValue) -> bool:
    if field_type == "bool":
        return isinstance(value, bool)
    if field_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if field_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if field_type == "str":
        return isinstance(value, str)
    return False


class Color(StrEnum):
    WHITE = "white"
    BLACK = "black"


class BasePieceType(StrEnum):
    PAWN = "pawn"
    KNIGHT = "knight"
    BISHOP = "bishop"
    ROOK = "rook"
    QUEEN = "queen"
    KING = "king"


@dataclass(frozen=True, slots=True)
class Modifier:
    op: str
    args: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("modifier op", self.op)
        object.__setattr__(self, "args", dict(self.args))

    def to_dict(self) -> dict[str, JsonValue]:
        return {"op": self.op, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "Modifier":
        return cls(op=str(data["op"]), args=dict(data.get("args", {})))


@dataclass(frozen=True, slots=True)
class Condition:
    op: str
    args: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("condition op", self.op)
        object.__setattr__(self, "args", dict(self.args))

    def to_dict(self) -> dict[str, JsonValue]:
        return {"op": self.op, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "Condition":
        return cls(op=str(data["op"]), args=dict(data.get("args", {})))


@dataclass(frozen=True, slots=True)
class Effect:
    op: str
    args: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("effect op", self.op)
        object.__setattr__(self, "args", dict(self.args))

    def to_dict(self) -> dict[str, JsonValue]:
        return {"op": self.op, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "Effect":
        return cls(op=str(data["op"]), args=dict(data.get("args", {})))


@dataclass(frozen=True, slots=True)
class Hook:
    event: str
    conditions: tuple[Condition, ...] = ()
    effects: tuple[Effect, ...] = ()
    priority: int = 0
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("hook event", self.event)
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "event": self.event,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "effects": [effect.to_dict() for effect in self.effects],
            "priority": self.priority,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "Hook":
        return cls(
            event=str(data["event"]),
            conditions=tuple(
                Condition.from_dict(item) for item in data.get("conditions", [])
            ),
            effects=tuple(Effect.from_dict(item) for item in data.get("effects", [])),
            priority=int(data.get("priority", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class StateField:
    name: str
    field_type: str
    default: JsonValue = None
    description: str = ""

    def __post_init__(self) -> None:
        _validate_identifier("state field name", self.name)
        if self.field_type not in STATE_FIELD_TYPES:
            raise ValueError(
                f"state field type must be one of {sorted(STATE_FIELD_TYPES)!r}, "
                f"got {self.field_type!r}"
            )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "type": self.field_type,
            "default": self.default,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "StateField":
        return cls(
            name=str(data["name"]),
            field_type=str(data["type"]),
            default=data.get("default"),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True, slots=True)
class PieceClass:
    class_id: str
    name: str
    base_piece_type: BasePieceType
    movement_modifiers: tuple[Modifier, ...] = ()
    capture_modifiers: tuple[Modifier, ...] = ()
    hooks: tuple[Hook, ...] = ()
    instance_state_schema: tuple[StateField, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("piece class id", self.class_id)
        if not self.name.strip():
            raise ValueError("piece class name must not be empty")
        object.__setattr__(self, "movement_modifiers", tuple(self.movement_modifiers))
        object.__setattr__(self, "capture_modifiers", tuple(self.capture_modifiers))
        object.__setattr__(self, "hooks", tuple(self.hooks))
        object.__setattr__(
            self, "instance_state_schema", tuple(self.instance_state_schema)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "class_id": self.class_id,
            "name": self.name,
            "base_piece_type": self.base_piece_type.value,
            "movement_modifiers": [
                modifier.to_dict() for modifier in self.movement_modifiers
            ],
            "capture_modifiers": [
                modifier.to_dict() for modifier in self.capture_modifiers
            ],
            "hooks": [hook.to_dict() for hook in self.hooks],
            "instance_state_schema": [
                field_spec.to_dict() for field_spec in self.instance_state_schema
            ],
            "metadata": dict(self.metadata),
        }

    @property
    def state_fields_by_name(self) -> dict[str, StateField]:
        return {field_spec.name: field_spec for field_spec in self.instance_state_schema}

    def normalize_instance_state(
        self, state: dict[str, JsonValue] | None = None
    ) -> dict[str, JsonValue]:
        incoming_state = {} if state is None else dict(state)
        unknown_fields = sorted(set(incoming_state) - set(self.state_fields_by_name))
        if unknown_fields:
            raise ValueError(
                f"Unknown state fields for piece class {self.class_id!r}: "
                f"{', '.join(unknown_fields)}"
            )

        normalized_state: dict[str, JsonValue] = {}
        for field_spec in self.instance_state_schema:
            value = incoming_state.get(field_spec.name, field_spec.default)
            if not _matches_field_type(field_spec.field_type, value):
                raise ValueError(
                    f"State field {field_spec.name!r} for piece class "
                    f"{self.class_id!r} must match declared type "
                    f"{field_spec.field_type!r}, got {value!r}"
                )
            normalized_state[field_spec.name] = value
        return normalized_state

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "PieceClass":
        return cls(
            class_id=str(data["class_id"]),
            name=str(data["name"]),
            base_piece_type=BasePieceType(str(data["base_piece_type"])),
            movement_modifiers=tuple(
                Modifier.from_dict(item) for item in data.get("movement_modifiers", [])
            ),
            capture_modifiers=tuple(
                Modifier.from_dict(item) for item in data.get("capture_modifiers", [])
            ),
            hooks=tuple(Hook.from_dict(item) for item in data.get("hooks", [])),
            instance_state_schema=tuple(
                StateField.from_dict(item)
                for item in data.get("instance_state_schema", [])
            ),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class PieceInstance:
    instance_id: str
    piece_class_id: str
    color: Color
    square: str | None
    state: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("piece instance id", self.instance_id)
        _validate_identifier("piece class id", self.piece_class_id)
        object.__setattr__(self, "square", normalize_square(self.square))
        object.__setattr__(self, "state", dict(self.state))

    @property
    def is_active(self) -> bool:
        return self.square is not None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "instance_id": self.instance_id,
            "piece_class_id": self.piece_class_id,
            "color": self.color.value,
            "square": self.square,
            "state": dict(self.state),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "PieceInstance":
        return cls(
            instance_id=str(data["instance_id"]),
            piece_class_id=str(data["piece_class_id"]),
            color=Color(str(data["color"])),
            square=str(data["square"]) if data.get("square") is not None else None,
            state=dict(data.get("state", {})),
        )
