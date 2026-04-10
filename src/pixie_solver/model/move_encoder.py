from __future__ import annotations

from pixie_solver.core.move import Move
from pixie_solver.core.state import GameState


class MoveEncoder:
    def encode_move(self, state: GameState, move: Move) -> list[float]:
        raise NotImplementedError("The move encoder lands in milestone M5.")
