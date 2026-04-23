from __future__ import annotations

from dataclasses import dataclass, field

from pixie_solver.core.event import StateDelta
from pixie_solver.core.move import Move
from pixie_solver.core.piece import Color
from pixie_solver.core.state import GameState


@dataclass(slots=True)
class SearchEdge:
    move: Move
    move_id: str
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    child_state_hash: str | None = None
    state_delta: StateDelta | None = None
    child: SearchNode | None = None

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass(slots=True)
class SearchNode:
    state: GameState
    state_hash: str
    to_play: Color
    visit_count: int = 0
    value_sum: float = 0.0
    is_expanded: bool = False
    is_terminal: bool = False
    terminal_value: float | None = None
    model_uncertainty: float | None = None
    legal_moves: tuple[Move, ...] = ()
    move_ids: tuple[str, ...] = ()
    children_by_move_id: dict[str, SearchEdge] = field(default_factory=dict)
    policy_logits: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: GameState) -> "SearchNode":
        return cls(
            state=state,
            state_hash=state.state_hash(),
            to_play=state.side_to_move,
        )

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
