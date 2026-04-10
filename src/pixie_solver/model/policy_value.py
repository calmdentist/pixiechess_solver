from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pixie_solver.core.move import Move
from pixie_solver.core.state import GameState


@dataclass(slots=True)
class PolicyValueOutput:
    policy_logits: dict[str, float] = field(default_factory=dict)
    value: float = 0.0


class PolicyValueModel:
    def infer(
        self,
        state: GameState,
        legal_moves: Sequence[Move],
    ) -> PolicyValueOutput:
        raise NotImplementedError("The policy/value model lands in milestone M5.")
