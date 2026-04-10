from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from pixie_solver.core.event import StateDelta
    from pixie_solver.core.move import Move
    from pixie_solver.core.state import GameState

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def to_primitive(value: Any) -> JsonValue:
    """Convert nested dataclasses and enums into JSON-safe primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return {
            field.name: to_primitive(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): to_primitive(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    raise TypeError(f"Unsupported value for JSON serialization: {type(value)!r}")


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    """Serialize a value using deterministic key ordering."""
    return json.dumps(
        to_primitive(value),
        ensure_ascii=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    )


def write_jsonl(path: str | Path, rows: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL record on line {line_number} must be an object"
                )
            records.append(record)
    return records

@dataclass(frozen=True, slots=True)
class ReplayStep:
    ply: int
    move: "Move"
    delta: "StateDelta"
    before_state_hash: str
    after_state_hash: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "ply": self.ply,
            "move": self.move.to_dict(),
            "delta": self.delta.to_dict(),
            "before_state_hash": self.before_state_hash,
            "after_state_hash": self.after_state_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "ReplayStep":
        from pixie_solver.core.event import StateDelta
        from pixie_solver.core.move import Move

        return cls(
            ply=int(data["ply"]),
            move=Move.from_dict(dict(data["move"])),
            delta=StateDelta.from_dict(dict(data["delta"])),
            before_state_hash=str(data["before_state_hash"]),
            after_state_hash=str(data["after_state_hash"]),
        )


@dataclass(frozen=True, slots=True)
class ReplayTrace:
    initial_state: "GameState"
    steps: tuple[ReplayStep, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def final_state_hash(self) -> str:
        if self.steps:
            return self.steps[-1].after_state_hash
        return self.initial_state.state_hash()

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "initial_state": self.initial_state.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "metadata": {
                **dict(self.metadata),
                "final_state_hash": self.final_state_hash,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "ReplayTrace":
        from pixie_solver.core.state import GameState

        metadata = dict(data.get("metadata", {}))
        metadata.pop("final_state_hash", None)
        return cls(
            initial_state=GameState.from_dict(dict(data["initial_state"])),
            steps=tuple(ReplayStep.from_dict(dict(item)) for item in data.get("steps", [])),
            metadata=metadata,
        )


def build_replay_trace(
    initial_state: "GameState",
    moves: list["Move"] | tuple["Move", ...],
    *,
    metadata: dict[str, JsonValue] | None = None,
) -> ReplayTrace:
    from pixie_solver.simulator.engine import apply_move

    state = initial_state
    steps: list[ReplayStep] = []
    for ply, move in enumerate(moves, start=1):
        before_state_hash = state.state_hash()
        state, delta = apply_move(state, move)
        after_state_hash = state.state_hash()
        _assert_delta_hashes(
            delta=delta,
            before_state_hash=before_state_hash,
            after_state_hash=after_state_hash,
        )
        steps.append(
            ReplayStep(
                ply=ply,
                move=move,
                delta=delta,
                before_state_hash=before_state_hash,
                after_state_hash=after_state_hash,
            )
        )
    return ReplayTrace(initial_state=initial_state, steps=tuple(steps), metadata=metadata or {})


def replay_trace(trace: ReplayTrace) -> "GameState":
    from pixie_solver.simulator.engine import apply_move

    state = trace.initial_state
    for step in trace.steps:
        if state.state_hash() != step.before_state_hash:
            raise ValueError(
                f"Replay diverged before ply {step.ply}: "
                f"{state.state_hash()!r} != {step.before_state_hash!r}"
            )
        next_state, delta = apply_move(state, step.move)
        _assert_delta_hashes(
            delta=delta,
            before_state_hash=step.before_state_hash,
            after_state_hash=step.after_state_hash,
        )
        if delta != step.delta:
            raise ValueError(f"Replay delta mismatch at ply {step.ply}")
        state = next_state
    return state


def _assert_delta_hashes(
    *,
    delta: "StateDelta",
    before_state_hash: str,
    after_state_hash: str,
) -> None:
    if delta.metadata.get("before_state_hash") != before_state_hash:
        raise ValueError("delta before_state_hash does not match replay trace")
    if delta.metadata.get("after_state_hash") != after_state_hash:
        raise ValueError("delta after_state_hash does not match replay trace")
