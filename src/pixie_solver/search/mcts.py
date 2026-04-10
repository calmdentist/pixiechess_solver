from __future__ import annotations

from dataclasses import dataclass, field

from pixie_solver.core.move import Move
from pixie_solver.core.state import GameState


@dataclass(slots=True)
class SearchResult:
    selected_move: Move | None
    visit_distribution: dict[str, float] = field(default_factory=dict)
    root_value: float = 0.0


def run_mcts(state: GameState, *, simulations: int = 0) -> SearchResult:
    raise NotImplementedError(
        f"MCTS lands in milestone M4. Requested simulations={simulations}."
    )
