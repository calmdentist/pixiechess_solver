from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pixie_solver.core.event import Event
from pixie_solver.core.piece import PieceInstance
from pixie_solver.core.state import GameState
from pixie_solver.core.trace import TransitionTrace


@dataclass(frozen=True, slots=True)
class ActionContext:
    state: GameState
    piece: PieceInstance
    program: Mapping[str, Any]
    occupancy: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "occupancy", self.state.occupancy())

    @property
    def entity(self) -> PieceInstance:
        return self.piece


@dataclass(frozen=True, slots=True)
class QueryContext:
    state: GameState
    piece: PieceInstance
    program: Mapping[str, Any]
    query_kind: str | None = None
    occupancy: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "occupancy", self.state.occupancy())
        if self.query_kind is not None:
            object.__setattr__(self, "query_kind", str(self.query_kind))

    @property
    def entity(self) -> PieceInstance:
        return self.piece


ThreatContext = QueryContext


@dataclass(frozen=True, slots=True)
class EventContext:
    state: GameState
    piece_instances: Mapping[str, PieceInstance]
    hook_owner_id: str
    program: Mapping[str, Any]
    event: Event
    trace: TransitionTrace | None = None

    @property
    def hook_owner(self) -> PieceInstance:
        return self.piece_instances[self.hook_owner_id]

    @property
    def entity_instances(self) -> Mapping[str, PieceInstance]:
        return self.piece_instances

    @property
    def owner(self) -> PieceInstance:
        return self.hook_owner
