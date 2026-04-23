from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from pixie_solver.core.hash import stable_digest
from pixie_solver.strategy.schema import StrategyHypothesis
from pixie_solver.strategy.validator import validate_strategy_hypothesis
from pixie_solver.utils.serialization import JsonValue


def canonicalize_strategy_hypothesis(
    strategy: StrategyHypothesis | Mapping[str, Any],
) -> dict[str, JsonValue]:
    candidate = _coerce_strategy(strategy)
    validate_strategy_hypothesis(candidate)
    return {
        "strategy_id": candidate.strategy_id,
        "summary": candidate.summary.strip(),
        "confidence": round(float(candidate.confidence), 6),
        "scope": candidate.scope.strip(),
        "subgoals": [item.strip() for item in candidate.subgoals],
        "action_biases": [item.strip() for item in candidate.action_biases],
        "avoid_biases": [item.strip() for item in candidate.avoid_biases],
        "success_predicates": [item.strip() for item in candidate.success_predicates],
        "failure_triggers": [item.strip() for item in candidate.failure_triggers],
        "metadata": deepcopy(dict(candidate.metadata)),
    }


def strategy_digest(strategy: StrategyHypothesis | Mapping[str, Any]) -> str:
    return stable_digest(canonicalize_strategy_hypothesis(strategy))


def _coerce_strategy(
    strategy: StrategyHypothesis | Mapping[str, Any],
) -> StrategyHypothesis:
    if isinstance(strategy, StrategyHypothesis):
        return strategy
    if not isinstance(strategy, Mapping):
        raise TypeError(
            "strategy must be a StrategyHypothesis or a mapping compatible with StrategyHypothesis"
        )
    return StrategyHypothesis(
        strategy_id=str(strategy.get("strategy_id", "")),
        summary=str(strategy.get("summary", "")),
        confidence=float(strategy.get("confidence", 0.0)),
        scope=str(strategy.get("scope", "generic")),
        subgoals=tuple(str(item) for item in strategy.get("subgoals", ())),
        action_biases=tuple(str(item) for item in strategy.get("action_biases", ())),
        avoid_biases=tuple(str(item) for item in strategy.get("avoid_biases", ())),
        success_predicates=tuple(
            str(item) for item in strategy.get("success_predicates", ())
        ),
        failure_triggers=tuple(
            str(item) for item in strategy.get("failure_triggers", ())
        ),
        metadata=dict(strategy.get("metadata", {})),
    )
