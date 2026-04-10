from __future__ import annotations

from pixie_solver.training.dataset import SelfPlayExample


def generate_selfplay_games(*, games: int = 0) -> list[SelfPlayExample]:
    raise NotImplementedError(f"Self-play generation lands in milestone M4/M6. Requested games={games}.")
