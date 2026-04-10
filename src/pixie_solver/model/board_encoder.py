from __future__ import annotations

from pixie_solver.core.state import GameState


class BoardEncoder:
    def encode_state(self, state: GameState) -> list[float]:
        raise NotImplementedError("The board encoder lands in milestone M5.")
