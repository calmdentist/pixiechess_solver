from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pixie_solver.core.state import GameState
from pixie_solver.model.policy_value import PolicyValueModel
from pixie_solver.search import StateEvaluator
from pixie_solver.training.dataset import SelfPlayExample, SelfPlayGame
from pixie_solver.training.selfplay import (
    SelfPlayConfig,
    flatten_selfplay_examples,
    generate_selfplay_games,
)
from pixie_solver.training.train import TrainingConfig, TrainingMetrics, train_from_replays


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    bootstrap_games: int = 8
    guided_games: int = 8
    selfplay_config: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass(slots=True)
class BootstrapRunResult:
    model: PolicyValueModel
    bootstrap_games: tuple[SelfPlayGame, ...]
    bootstrap_examples: tuple[SelfPlayExample, ...]
    guided_games: tuple[SelfPlayGame, ...]
    guided_examples: tuple[SelfPlayExample, ...]
    training_metrics: TrainingMetrics


def bootstrap_policy_value_model(
    initial_states: Sequence[GameState],
    *,
    config: BootstrapConfig | None = None,
    evaluator: StateEvaluator | None = None,
) -> BootstrapRunResult:
    if not initial_states:
        raise ValueError("initial_states must contain at least one GameState")

    active_config = BootstrapConfig() if config is None else config
    if active_config.bootstrap_games < 1:
        raise ValueError("bootstrap_games must be at least 1")
    if active_config.guided_games < 0:
        raise ValueError("guided_games must be non-negative")

    bootstrap_games = tuple(
        generate_selfplay_games(
            initial_states,
            games=active_config.bootstrap_games,
            config=active_config.selfplay_config,
            evaluator=evaluator,
        )
    )
    bootstrap_examples = tuple(flatten_selfplay_examples(bootstrap_games))
    training_run = train_from_replays(
        bootstrap_examples,
        config=active_config.training_config,
    )

    guided_games: tuple[SelfPlayGame, ...] = ()
    guided_examples: tuple[SelfPlayExample, ...] = ()
    if active_config.guided_games > 0:
        guided_games = tuple(
            generate_selfplay_games(
                initial_states,
                games=active_config.guided_games,
                config=active_config.selfplay_config,
                policy_value_model=training_run.model,
                evaluator=evaluator,
            )
        )
        guided_examples = tuple(flatten_selfplay_examples(guided_games))

    return BootstrapRunResult(
        model=training_run.model,
        bootstrap_games=bootstrap_games,
        bootstrap_examples=bootstrap_examples,
        guided_games=guided_games,
        guided_examples=guided_examples,
        training_metrics=training_run.metrics,
    )
