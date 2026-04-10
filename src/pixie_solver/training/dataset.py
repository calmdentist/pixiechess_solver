from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pixie_solver.core.move import Move
from pixie_solver.core.state import GameState
from pixie_solver.utils.serialization import JsonValue, ReplayTrace, read_jsonl, write_jsonl


@dataclass(slots=True)
class SelfPlayExample:
    state: GameState
    legal_moves: tuple[Move, ...] = ()
    legal_move_ids: tuple[str, ...] = ()
    visit_distribution: dict[str, float] = field(default_factory=dict)
    visit_counts: dict[str, int] = field(default_factory=dict)
    selected_move_id: str | None = None
    root_value: float = 0.0
    outcome: float = 0.0
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.legal_moves = tuple(self.legal_moves)
        self.legal_move_ids = tuple(str(move_id) for move_id in self.legal_move_ids)
        self.visit_distribution = {
            str(move_id): float(value)
            for move_id, value in sorted(self.visit_distribution.items())
        }
        self.visit_counts = {
            str(move_id): int(value)
            for move_id, value in sorted(self.visit_counts.items())
        }
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "state": self.state.to_dict(),
            "legal_moves": [move.to_dict() for move in self.legal_moves],
            "legal_move_ids": list(self.legal_move_ids),
            "visit_distribution": dict(self.visit_distribution),
            "visit_counts": dict(self.visit_counts),
            "selected_move_id": self.selected_move_id,
            "root_value": self.root_value,
            "outcome": self.outcome,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "SelfPlayExample":
        return cls(
            state=GameState.from_dict(dict(data["state"])),
            legal_moves=tuple(
                Move.from_dict(dict(move_data)) for move_data in data.get("legal_moves", [])
            ),
            legal_move_ids=tuple(
                str(move_id) for move_id in data.get("legal_move_ids", [])
            ),
            visit_distribution={
                str(move_id): float(value)
                for move_id, value in dict(data.get("visit_distribution", {})).items()
            },
            visit_counts={
                str(move_id): int(value)
                for move_id, value in dict(data.get("visit_counts", {})).items()
            },
            selected_move_id=(
                str(data["selected_move_id"])
                if data.get("selected_move_id") is not None
                else None
            ),
            root_value=float(data.get("root_value", 0.0)),
            outcome=float(data.get("outcome", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class SelfPlayGame:
    replay_trace: ReplayTrace
    examples: tuple[SelfPlayExample, ...] = ()
    outcome: str = "draw"
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.examples = tuple(self.examples)
        self.metadata = dict(self.metadata)

    @property
    def final_state_hash(self) -> str:
        return self.replay_trace.final_state_hash

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "replay_trace": self.replay_trace.to_dict(),
            "examples": [example.to_dict() for example in self.examples],
            "outcome": self.outcome,
            "metadata": {
                **dict(self.metadata),
                "final_state_hash": self.final_state_hash,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "SelfPlayGame":
        metadata = dict(data.get("metadata", {}))
        metadata.pop("final_state_hash", None)
        return cls(
            replay_trace=ReplayTrace.from_dict(dict(data["replay_trace"])),
            examples=tuple(
                SelfPlayExample.from_dict(dict(example_data))
                for example_data in data.get("examples", [])
            ),
            outcome=str(data.get("outcome", "draw")),
            metadata=metadata,
        )


def write_selfplay_games_jsonl(
    path: str | Path,
    games: Iterable[SelfPlayGame],
) -> None:
    write_jsonl(path, (game.to_dict() for game in games))


def read_selfplay_games_jsonl(path: str | Path) -> list[SelfPlayGame]:
    return [SelfPlayGame.from_dict(record) for record in read_jsonl(path)]


def write_selfplay_examples_jsonl(
    path: str | Path,
    examples: Iterable[SelfPlayExample],
) -> None:
    write_jsonl(path, (example.to_dict() for example in examples))


def read_selfplay_examples_jsonl(path: str | Path) -> list[SelfPlayExample]:
    return [SelfPlayExample.from_dict(record) for record in read_jsonl(path)]
