from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pixie_solver.strategy.schema import StrategyHypothesis

IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_STRATEGY_LIST_LENGTH = 16


class StrategyValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def collect_strategy_validation_errors(
    strategy: StrategyHypothesis | Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    candidate = _coerce_strategy(strategy)

    if not candidate.strategy_id or not IDENTIFIER_PATTERN.fullmatch(candidate.strategy_id):
        errors.append("strategy_id must be a lowercase identifier")
    if not candidate.summary.strip():
        errors.append("summary must be a non-empty string")
    if not 0.0 <= candidate.confidence <= 1.0:
        errors.append("confidence must be in [0, 1]")
    if not candidate.scope.strip():
        errors.append("scope must be a non-empty string")

    _validate_unique_string_list(errors, candidate.subgoals, field_name="subgoals")
    _validate_unique_string_list(
        errors,
        candidate.action_biases,
        field_name="action_biases",
    )
    _validate_unique_string_list(
        errors,
        candidate.avoid_biases,
        field_name="avoid_biases",
    )
    _validate_unique_string_list(
        errors,
        candidate.success_predicates,
        field_name="success_predicates",
    )
    _validate_unique_string_list(
        errors,
        candidate.failure_triggers,
        field_name="failure_triggers",
    )
    return errors


def validate_strategy_hypothesis(
    strategy: StrategyHypothesis | Mapping[str, Any],
) -> None:
    errors = collect_strategy_validation_errors(strategy)
    if errors:
        raise StrategyValidationError(errors)


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
        subgoals=_string_sequence(strategy.get("subgoals", ())),
        action_biases=_string_sequence(strategy.get("action_biases", ())),
        avoid_biases=_string_sequence(strategy.get("avoid_biases", ())),
        success_predicates=_string_sequence(strategy.get("success_predicates", ())),
        failure_triggers=_string_sequence(strategy.get("failure_triggers", ())),
        metadata=dict(strategy.get("metadata", {})),
    )


def _string_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    raise TypeError("strategy lists must be sequences of strings")


def _validate_unique_string_list(
    errors: list[str],
    values: tuple[str, ...],
    *,
    field_name: str,
) -> None:
    if len(values) > MAX_STRATEGY_LIST_LENGTH:
        errors.append(
            f"{field_name} may contain at most {MAX_STRATEGY_LIST_LENGTH} items"
        )
    if any(not value.strip() for value in values):
        errors.append(f"{field_name} may not contain empty strings")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        errors.append(f"{field_name} values must be unique: " + ", ".join(duplicates))
