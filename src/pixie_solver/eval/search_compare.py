from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pixie_solver.core.state import GameState
from pixie_solver.model.policy_value import PolicyValueModel
from pixie_solver.search import SearchResult, StateEvaluator, run_mcts
from pixie_solver.utils.serialization import JsonValue


@dataclass(frozen=True, slots=True)
class SearchComparisonConfig:
    baseline_simulations: int = 64
    guided_simulations: int = 16
    c_puct: float = 1.5


@dataclass(slots=True)
class SearchComparisonCase:
    state_hash: str
    baseline: SearchResult
    guided: SearchResult
    move_agreement: bool


@dataclass(slots=True)
class SearchComparisonSummary:
    config: SearchComparisonConfig
    cases: tuple[SearchComparisonCase, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    @property
    def positions(self) -> int:
        return len(self.cases)

    @property
    def move_agreement_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for case in self.cases if case.move_agreement) / len(self.cases)

    @property
    def average_abs_root_value_delta(self) -> float:
        if not self.cases:
            return 0.0
        return sum(
            abs(case.baseline.root_value - case.guided.root_value)
            for case in self.cases
        ) / len(self.cases)


def compare_search_modes(
    states: Sequence[GameState],
    *,
    policy_value_model: PolicyValueModel,
    config: SearchComparisonConfig | None = None,
    evaluator: StateEvaluator | None = None,
) -> SearchComparisonSummary:
    if not states:
        raise ValueError("states must contain at least one GameState")

    active_config = SearchComparisonConfig() if config is None else config
    if active_config.baseline_simulations < 1:
        raise ValueError("baseline_simulations must be at least 1")
    if active_config.guided_simulations < 1:
        raise ValueError("guided_simulations must be at least 1")

    cases: list[SearchComparisonCase] = []
    for state in states:
        baseline = run_mcts(
            state,
            simulations=active_config.baseline_simulations,
            evaluator=evaluator,
            c_puct=active_config.c_puct,
        )
        guided = run_mcts(
            state,
            simulations=active_config.guided_simulations,
            policy_value_model=policy_value_model,
            evaluator=evaluator,
            c_puct=active_config.c_puct,
        )
        cases.append(
            SearchComparisonCase(
                state_hash=state.state_hash(),
                baseline=baseline,
                guided=guided,
                move_agreement=baseline.selected_move_id == guided.selected_move_id,
            )
        )

    return SearchComparisonSummary(
        config=active_config,
        cases=tuple(cases),
        metadata={
            "guided_budget_ratio": active_config.guided_simulations
            / active_config.baseline_simulations,
            "average_baseline_expanded_nodes": _average_metadata_value(
                cases,
                side="baseline",
                key="expanded_nodes",
            ),
            "average_guided_expanded_nodes": _average_metadata_value(
                cases,
                side="guided",
                key="expanded_nodes",
            ),
            "average_baseline_heuristic_evaluations": _average_metadata_value(
                cases,
                side="baseline",
                key="heuristic_evaluations",
            ),
            "average_guided_model_inference_calls": _average_metadata_value(
                cases,
                side="guided",
                key="model_inference_calls",
            ),
        },
    )


def _average_metadata_value(
    cases: Sequence[SearchComparisonCase],
    *,
    side: str,
    key: str,
) -> float:
    if not cases:
        return 0.0
    total = 0.0
    for case in cases:
        result = case.baseline if side == "baseline" else case.guided
        total += float(result.metadata.get(key, 0.0))
    return total / len(cases)
