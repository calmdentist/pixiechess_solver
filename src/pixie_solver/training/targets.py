from __future__ import annotations

from pixie_solver.training.dataset import SelfPlayExample


def blended_value_target_for_example(
    example: SelfPlayExample,
    *,
    root_value_weight: float,
    outcome_weight: float,
) -> float:
    total_weight = root_value_weight + outcome_weight
    if total_weight <= 0.0:
        raise ValueError("At least one blended value target weight must be positive")
    return (
        root_value_weight * float(example.root_value)
        + outcome_weight * float(example.outcome)
    ) / total_weight


def uncertainty_target_for_example(example: SelfPlayExample) -> float:
    disagreement = abs(float(example.root_value) - float(example.outcome)) / 2.0
    return max(0.0, min(1.0, disagreement))
