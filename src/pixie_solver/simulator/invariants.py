from __future__ import annotations

from pixie_solver.core.state import GameState


def assert_state_invariants(state: GameState) -> None:
    state.validate()
