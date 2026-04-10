from __future__ import annotations

from dataclasses import dataclass, field

from pixie_solver.core.state import GameState


@dataclass(slots=True)
class SelfPlayExample:
    state: GameState
    legal_move_ids: tuple[str, ...] = ()
    visit_distribution: dict[str, float] = field(default_factory=dict)
    outcome: float = 0.0
