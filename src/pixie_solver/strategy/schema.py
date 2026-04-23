from __future__ import annotations

from dataclasses import dataclass, field

from pixie_solver.utils.serialization import JsonValue


@dataclass(frozen=True, slots=True)
class StrategyHypothesis:
    strategy_id: str
    summary: str
    confidence: float = 0.5
    scope: str = "generic"
    subgoals: tuple[str, ...] = ()
    action_biases: tuple[str, ...] = ()
    avoid_biases: tuple[str, ...] = ()
    success_predicates: tuple[str, ...] = ()
    failure_triggers: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_id", str(self.strategy_id))
        object.__setattr__(self, "summary", str(self.summary))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "scope", str(self.scope))
        object.__setattr__(self, "subgoals", tuple(str(item) for item in self.subgoals))
        object.__setattr__(
            self,
            "action_biases",
            tuple(str(item) for item in self.action_biases),
        )
        object.__setattr__(
            self,
            "avoid_biases",
            tuple(str(item) for item in self.avoid_biases),
        )
        object.__setattr__(
            self,
            "success_predicates",
            tuple(str(item) for item in self.success_predicates),
        )
        object.__setattr__(
            self,
            "failure_triggers",
            tuple(str(item) for item in self.failure_triggers),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "strategy_id": self.strategy_id,
            "summary": self.summary,
            "confidence": self.confidence,
            "scope": self.scope,
            "subgoals": list(self.subgoals),
            "action_biases": list(self.action_biases),
            "avoid_biases": list(self.avoid_biases),
            "success_predicates": list(self.success_predicates),
            "failure_triggers": list(self.failure_triggers),
            "metadata": dict(self.metadata),
        }
