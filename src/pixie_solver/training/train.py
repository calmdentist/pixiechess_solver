from __future__ import annotations

from collections.abc import Sequence

from pixie_solver.training.dataset import SelfPlayExample


def train_from_replays(replays: Sequence[SelfPlayExample]) -> None:
    raise NotImplementedError(
        f"Training lands in milestone M5/M6. Received {len(replays)} replay examples."
    )
