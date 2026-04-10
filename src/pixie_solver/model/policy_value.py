from __future__ import annotations

from dataclasses import dataclass, field

from pixie_solver.core.state import GameState


@dataclass(slots=True)
class PolicyValueOutput:
    policy_logits: dict[str, float] = field(default_factory=dict)
    value: float = 0.0


class PolicyValueModel:
    def infer(self, state: GameState) -> PolicyValueOutput:
        raise NotImplementedError("The policy/value model lands in milestone M5.")
