from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pixie_solver.core import Color, GameState, Move, StateDelta, stable_move_id
from pixie_solver.model.policy_value import PolicyValueModel
from pixie_solver.search import (
    DirichletRootNoise,
    HeuristicEvaluator,
    SearchResult,
    StateEvaluator,
    run_mcts,
)
from pixie_solver.strategy import (
    StrategyHypothesis,
    StrategyProvider,
    StrategyRequest,
    strategy_digest as compute_strategy_digest,
)
from pixie_solver.simulator.engine import apply_move, result
from pixie_solver.training.dataset import SelfPlayExample, SelfPlayGame
from pixie_solver.training.benchmark_metadata import (
    SEARCH_ONLY_MODEL_ARCHITECTURE,
    benchmark_metadata_for_state,
)
from pixie_solver.training.checkpoint import load_training_checkpoint
from pixie_solver.training.inference_service import (
    BatchedInferenceClient,
    BatchedInferenceConfig,
    BatchedInferenceService,
    BatchedInferenceStats,
)
from pixie_solver.utils import build_replay_trace
from pixie_solver.utils.serialization import JsonValue, to_primitive

SELFPLAY_SEED_STRIDE = 1_000_003
_WORKER_MODEL_CACHE: dict[tuple[str, str | None], PolicyValueModel] = {}


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    simulations: int = 64
    max_plies: int = 256
    opening_temperature: float = 1.0
    final_temperature: float = 0.0
    temperature_drop_after_ply: int = 12
    c_puct: float = 1.5
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    seed: int = 0
    adjudicate_max_plies: bool = True
    adjudication_threshold: float = 0.2
    strategy: StrategyHypothesis | dict[str, JsonValue] | None = None
    adaptive_search: bool = False
    adaptive_min_simulations: int | None = None
    adaptive_max_simulations: int | None = None
    strategy_refresh_on_uncertainty: bool = False
    strategy_refresh_uncertainty_threshold: float = 0.75

    def __post_init__(self) -> None:
        if self.max_plies < 0:
            raise ValueError("max_plies must be non-negative")
        if self.root_dirichlet_alpha <= 0.0:
            raise ValueError("root_dirichlet_alpha must be positive")
        if self.root_exploration_fraction < 0.0 or self.root_exploration_fraction > 1.0:
            raise ValueError("root_exploration_fraction must be in [0, 1]")
        if self.adjudication_threshold < 0.0 or self.adjudication_threshold > 1.0:
            raise ValueError("adjudication_threshold must be in [0, 1]")
        if self.adaptive_min_simulations is not None and self.adaptive_min_simulations < 1:
            raise ValueError("adaptive_min_simulations must be at least 1")
        if self.adaptive_max_simulations is not None and self.adaptive_max_simulations < 1:
            raise ValueError("adaptive_max_simulations must be at least 1")
        if (
            self.adaptive_min_simulations is not None
            and self.adaptive_max_simulations is not None
            and self.adaptive_max_simulations < self.adaptive_min_simulations
        ):
            raise ValueError(
                "adaptive_max_simulations must be greater than or equal to adaptive_min_simulations"
            )
        if self.strategy is not None:
            object.__setattr__(self, "strategy", dict(to_primitive(self.strategy)))
        if (
            self.strategy_refresh_uncertainty_threshold < 0.0
            or self.strategy_refresh_uncertainty_threshold > 1.0
        ):
            raise ValueError("strategy_refresh_uncertainty_threshold must be in [0, 1]")

    def temperature_for_ply(self, ply: int) -> float:
        if ply < self.temperature_drop_after_ply:
            return self.opening_temperature
        return self.final_temperature

    def root_noise_for_selfplay(self) -> DirichletRootNoise | None:
        if self.root_exploration_fraction <= 0.0:
            return None
        return DirichletRootNoise(
            alpha=self.root_dirichlet_alpha,
            exploration_fraction=self.root_exploration_fraction,
        )


@dataclass(frozen=True, slots=True)
class SelfPlayProgress:
    event: str
    game_index: int
    games_total: int
    ply: int | None = None
    selected_move_id: str | None = None
    legal_move_count: int | None = None
    plies_played: int | None = None
    outcome: str | None = None
    termination_reason: str | None = None
    used_model: bool = False


@dataclass(frozen=True, slots=True)
class SelfPlayTraceEvent:
    event: str
    game_index: int
    games_total: int
    ply: int | None = None
    before_state: GameState | None = None
    after_state: GameState | None = None
    move: Move | None = None
    selected_move_id: str | None = None
    delta: StateDelta | None = None
    search_result: SearchResult | None = None
    outcome: str | None = None
    termination_reason: str | None = None
    used_model: bool = False
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CutoffAdjudication:
    outcome: str
    score: float
    threshold: float
    method: str = "heuristic_material_mobility_check"

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "method": self.method,
            "outcome": self.outcome,
            "score": self.score,
            "threshold": self.threshold,
        }


def generate_selfplay_games(
    initial_states: Sequence[GameState],
    *,
    games: int,
    config: SelfPlayConfig | None = None,
    policy_value_model: PolicyValueModel | None = None,
    evaluator: StateEvaluator | None = None,
    strategy_provider: StrategyProvider | None = None,
    progress_callback: Callable[[SelfPlayProgress], None] | None = None,
    trace_callback: Callable[[SelfPlayTraceEvent], None] | None = None,
) -> list[SelfPlayGame]:
    if games < 1:
        raise ValueError("games must be at least 1")
    if not initial_states:
        raise ValueError("initial_states must contain at least one GameState")

    active_config = SelfPlayConfig() if config is None else config
    generated_games: list[SelfPlayGame] = []
    for game_index in range(games):
        seed_state = initial_states[game_index % len(initial_states)]
        game_seed = seed_for_game(active_config.seed, game_index)
        generated_games.append(
            _play_single_game(
                initial_state=_clone_state(seed_state),
                game_index=game_index,
                games_total=games,
                config=active_config,
                game_seed=game_seed,
                rng=random.Random(game_seed),
                policy_value_model=policy_value_model,
                evaluator=evaluator,
                strategy_provider=strategy_provider,
                progress_callback=progress_callback,
                trace_callback=trace_callback,
            )
        )
    return generated_games


def generate_selfplay_games_parallel(
    initial_states: Sequence[GameState],
    *,
    games: int,
    workers: int,
    config: SelfPlayConfig | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    batched_inference_config: BatchedInferenceConfig | None = None,
    inference_device: str | None = None,
    strategy_provider: StrategyProvider | None = None,
    progress_callback: Callable[[SelfPlayProgress], None] | None = None,
) -> tuple[list[SelfPlayGame], BatchedInferenceStats | None]:
    if games < 1:
        raise ValueError("games must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not initial_states:
        raise ValueError("initial_states must contain at least one GameState")

    active_config = SelfPlayConfig() if config is None else config
    if workers == 1:
        model = (
            load_training_checkpoint(checkpoint_path, device=device).model
            if checkpoint_path is not None
            else None
        )
        return (
            generate_selfplay_games(
                initial_states,
                games=games,
                config=active_config,
                policy_value_model=model,
                strategy_provider=strategy_provider,
                progress_callback=progress_callback,
            ),
            None,
        )
    if strategy_provider is not None and active_config.strategy_refresh_on_uncertainty:
        raise ValueError(
            "parallel self-play does not support uncertainty-triggered strategy refresh"
        )

    config_payload = asdict(active_config)
    checkpoint_payload = str(checkpoint_path) if checkpoint_path is not None else None
    inference_stats = None
    if checkpoint_payload is not None and batched_inference_config is not None:
        with BatchedInferenceService(
            checkpoint_payload,
            device=inference_device or device,
            config=batched_inference_config,
        ) as inference_service:
            games_out = _run_parallel_selfplay_jobs(
                initial_states=initial_states,
                games=games,
                workers=workers,
                config_payload=config_payload,
                strategy_payloads=_parallel_strategy_payloads(
                    initial_states=initial_states,
                    games=games,
                    config=active_config,
                    strategy_provider=strategy_provider,
                ),
                checkpoint_payload=None,
                device=device,
                inference_client=inference_service.client(),
                progress_callback=progress_callback,
            )
            inference_stats = inference_service.stats()
        return games_out, inference_stats

    return (
        _run_parallel_selfplay_jobs(
            initial_states=initial_states,
            games=games,
            workers=workers,
            config_payload=config_payload,
            strategy_payloads=_parallel_strategy_payloads(
                initial_states=initial_states,
                games=games,
                config=active_config,
                strategy_provider=strategy_provider,
            ),
            checkpoint_payload=checkpoint_payload,
            device=device,
            inference_client=None,
            progress_callback=progress_callback,
        ),
        inference_stats,
    )


def _run_parallel_selfplay_jobs(
    *,
    initial_states: Sequence[GameState],
    games: int,
    workers: int,
    config_payload: dict[str, object],
    strategy_payloads: Sequence[dict[str, JsonValue] | None],
    checkpoint_payload: str | None,
    device: str | None,
    inference_client: BatchedInferenceClient | None,
    progress_callback: Callable[[SelfPlayProgress], None] | None,
) -> list[SelfPlayGame]:
    jobs = [
        {
            "initial_state": initial_states[game_index % len(initial_states)].to_dict(),
            "game_index": game_index,
            "games_total": games,
            "config": {
                **config_payload,
                "strategy": strategy_payloads[game_index],
            },
            "checkpoint_path": checkpoint_payload,
            "device": device,
            "inference_client": inference_client,
        }
        for game_index in range(games)
    ]

    completed: dict[int, SelfPlayGame] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_parallel_selfplay_worker, job)
            for job in jobs
        ]
        for future in as_completed(futures):
            game = SelfPlayGame.from_dict(future.result())
            completed[int(game.metadata["game_index"])] = game
            if progress_callback is not None:
                progress_callback(
                    SelfPlayProgress(
                        event="game_completed",
                        game_index=int(game.metadata["game_index"]),
                        games_total=games,
                        plies_played=int(game.metadata["plies_played"]),
                        outcome=game.outcome,
                        termination_reason=str(game.metadata["termination_reason"]),
                        used_model=checkpoint_payload is not None,
                    )
                )

    return [
        completed[game_index]
        for game_index in range(games)
    ]


def flatten_selfplay_examples(games: Sequence[SelfPlayGame]) -> list[SelfPlayExample]:
    examples: list[SelfPlayExample] = []
    for game in games:
        examples.extend(game.examples)
    return examples


def adjudicate_cutoff(
    state: GameState,
    *,
    threshold: float,
    evaluator: StateEvaluator | None = None,
) -> CutoffAdjudication:
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")
    evaluator_impl = HeuristicEvaluator() if evaluator is None else evaluator
    white_state = _with_side_to_move(state, Color.WHITE)
    score = max(-1.0, min(1.0, float(evaluator_impl.evaluate(white_state))))
    if score > threshold:
        outcome = Color.WHITE.value
    elif score < -threshold:
        outcome = Color.BLACK.value
    else:
        outcome = "draw"
    return CutoffAdjudication(
        outcome=outcome,
        score=score,
        threshold=threshold,
    )


def seed_for_game(base_seed: int, game_index: int) -> int:
    if game_index < 0:
        raise ValueError("game_index must be non-negative")
    return int(base_seed) + game_index * SELFPLAY_SEED_STRIDE


def _play_single_game(
    *,
    initial_state: GameState,
    game_index: int,
    games_total: int,
    config: SelfPlayConfig,
    game_seed: int,
    rng: random.Random,
    policy_value_model: PolicyValueModel | None,
    evaluator: StateEvaluator | None,
    strategy_provider: StrategyProvider | None,
    progress_callback: Callable[[SelfPlayProgress], None] | None,
    trace_callback: Callable[[SelfPlayTraceEvent], None] | None,
) -> SelfPlayGame:
    state = initial_state
    examples: list[SelfPlayExample] = []
    moves: list[Move] = []
    termination_reason = "max_plies"
    game_result: str | None = None
    adjudication: CutoffAdjudication | None = None
    model_architecture = _model_architecture(policy_value_model)
    initial_strategy_payload, strategy_provider_metadata = _resolve_strategy(
        state=state,
        strategy=config.strategy,
        strategy_provider=strategy_provider,
        phase="game_start",
        metadata={
            "game_index": game_index,
            "games_total": games_total,
        },
    )
    current_strategy_payload = initial_strategy_payload
    strategy_digest = _strategy_digest(current_strategy_payload)
    strategy_scope = _strategy_scope(current_strategy_payload)
    strategy_refreshes = 0
    if progress_callback is not None:
        progress_callback(
            SelfPlayProgress(
                event="game_started",
                game_index=game_index,
                games_total=games_total,
                used_model=policy_value_model is not None,
            )
        )
    if trace_callback is not None:
        trace_callback(
            SelfPlayTraceEvent(
                event="game_started",
                game_index=game_index,
                games_total=games_total,
                ply=0,
                after_state=state,
                used_model=policy_value_model is not None,
            )
        )

    for ply in range(config.max_plies):
        game_result = result(state)
        if game_result is not None:
            termination_reason = "terminal"
            break

        search_result = run_mcts(
            state,
            simulations=config.simulations,
            policy_value_model=policy_value_model,
            evaluator=evaluator,
            c_puct=config.c_puct,
            root_noise=config.root_noise_for_selfplay(),
            rng=rng,
            strategy=current_strategy_payload,
            adaptive_search=config.adaptive_search,
            adaptive_min_simulations=config.adaptive_min_simulations,
            adaptive_max_simulations=config.adaptive_max_simulations,
        )
        if (
            strategy_provider is not None
            and config.strategy_refresh_on_uncertainty
            and float(search_result.metadata.get("root_uncertainty", 0.0))
            >= config.strategy_refresh_uncertainty_threshold
        ):
            refreshed_strategy, refresh_metadata = _resolve_strategy(
                state=state,
                strategy=current_strategy_payload,
                strategy_provider=strategy_provider,
                phase="high_uncertainty",
                metadata={
                    "game_index": game_index,
                    "games_total": games_total,
                    "ply": ply,
                    "root_uncertainty": search_result.metadata.get("root_uncertainty"),
                },
            )
            refreshed_digest = _strategy_digest(refreshed_strategy)
            if refreshed_digest != strategy_digest:
                current_strategy_payload = refreshed_strategy
                strategy_digest = refreshed_digest
                strategy_scope = _strategy_scope(current_strategy_payload)
                strategy_refreshes += 1
                strategy_provider_metadata = {
                    **dict(strategy_provider_metadata),
                    f"refresh_{strategy_refreshes}": refresh_metadata,
                }
                search_result = run_mcts(
                    state,
                    simulations=config.simulations,
                    policy_value_model=policy_value_model,
                    evaluator=evaluator,
                    c_puct=config.c_puct,
                    root_noise=config.root_noise_for_selfplay(),
                    rng=rng,
                    strategy=current_strategy_payload,
                    adaptive_search=config.adaptive_search,
                    adaptive_min_simulations=config.adaptive_min_simulations,
                    adaptive_max_simulations=config.adaptive_max_simulations,
                )
        chosen_move, chosen_move_id = _select_move_for_selfplay(
            search_result=search_result,
            temperature=config.temperature_for_ply(ply),
            rng=rng,
        )
        if chosen_move is None or chosen_move_id is None:
            termination_reason = "no_legal_moves"
            break

        if progress_callback is not None:
            progress_callback(
                SelfPlayProgress(
                    event="ply_completed",
                    game_index=game_index,
                    games_total=games_total,
                    ply=ply + 1,
                    selected_move_id=chosen_move_id,
                    legal_move_count=len(search_result.legal_moves),
                    used_model=policy_value_model is not None,
                )
            )
        examples.append(
            SelfPlayExample(
                state=state,
                legal_moves=search_result.legal_moves,
                legal_move_ids=tuple(stable_move_id(move) for move in search_result.legal_moves),
                visit_distribution=search_result.visit_distribution,
                visit_counts=search_result.visit_counts,
                selected_move_id=chosen_move_id,
                root_value=search_result.root_value,
                metadata=benchmark_metadata_for_state(
                    state,
                    {
                        "game_index": game_index,
                        "ply": ply,
                        "temperature": config.temperature_for_ply(ply),
                        "strategy": current_strategy_payload,
                        "strategy_digest": strategy_digest,
                        "strategy_scope": strategy_scope,
                        "model_architecture": model_architecture,
                        **dict(search_result.metadata),
                    },
                    search_budget=_coerce_search_budget(
                        search_result.metadata.get("simulations_used"),
                        default=config.simulations,
                    ),
                    model_architecture=model_architecture,
                ),
            )
        )
        moves.append(chosen_move)
        before_state = state
        state, delta = apply_move(before_state, chosen_move)
        if trace_callback is not None:
            trace_callback(
                SelfPlayTraceEvent(
                    event="ply_completed",
                    game_index=game_index,
                    games_total=games_total,
                    ply=ply + 1,
                    before_state=before_state,
                    after_state=state,
                    move=chosen_move,
                    selected_move_id=chosen_move_id,
                    delta=delta,
                    search_result=search_result,
                    used_model=policy_value_model is not None,
                    metadata={
                        "temperature": config.temperature_for_ply(ply),
                        "legal_move_count": len(search_result.legal_moves),
                        "strategy_digest": strategy_digest,
                    },
                )
            )
    else:
        game_result = result(state)

    if game_result is None:
        if termination_reason == "max_plies" and config.adjudicate_max_plies:
            adjudication = adjudicate_cutoff(
                state,
                threshold=config.adjudication_threshold,
                evaluator=evaluator,
            )
            outcome = adjudication.outcome
        else:
            outcome = "draw"
    else:
        outcome = game_result
    for example in examples:
        example.outcome = _outcome_for_perspective(outcome, example.state.side_to_move)
        if adjudication is not None:
            example.metadata["cutoff_adjudication"] = adjudication.to_dict()

    replay_trace = build_replay_trace(
        initial_state,
        moves,
        metadata=benchmark_metadata_for_state(
            initial_state,
            {
                "game_index": game_index,
                "base_seed": config.seed,
                "seed": game_seed,
                "simulations": config.simulations,
                "max_plies": config.max_plies,
                "opening_temperature": config.opening_temperature,
                "final_temperature": config.final_temperature,
                "temperature_drop_after_ply": config.temperature_drop_after_ply,
                "c_puct": config.c_puct,
                "root_dirichlet_alpha": config.root_dirichlet_alpha,
                "root_exploration_fraction": config.root_exploration_fraction,
                "adjudicate_max_plies": config.adjudicate_max_plies,
                "adjudication_threshold": config.adjudication_threshold,
                "initial_strategy": initial_strategy_payload,
                "strategy": current_strategy_payload,
                "strategy_digest": strategy_digest,
                "strategy_scope": strategy_scope,
                "strategy_refreshes": strategy_refreshes,
                "strategy_provider": strategy_provider_metadata,
                "strategy_refresh_on_uncertainty": (
                    config.strategy_refresh_on_uncertainty
                ),
                "strategy_refresh_uncertainty_threshold": (
                    config.strategy_refresh_uncertainty_threshold
                ),
                "adaptive_search": config.adaptive_search,
                "adaptive_min_simulations": config.adaptive_min_simulations,
                "adaptive_max_simulations": config.adaptive_max_simulations,
                "cutoff_adjudication": (
                    adjudication.to_dict() if adjudication is not None else None
                ),
                "termination_reason": termination_reason,
                "outcome": outcome,
                "model_architecture": model_architecture,
            },
            search_budget=config.simulations,
            model_architecture=model_architecture,
        ),
    )
    game = SelfPlayGame(
        replay_trace=replay_trace,
        examples=tuple(examples),
        outcome=outcome,
        metadata=benchmark_metadata_for_state(
            initial_state,
            {
                "game_index": game_index,
                "base_seed": config.seed,
                "seed": game_seed,
                "plies_played": len(moves),
                "termination_reason": termination_reason,
                "cutoff_adjudication": (
                    adjudication.to_dict() if adjudication is not None else None
                ),
                "initial_strategy": initial_strategy_payload,
                "strategy": current_strategy_payload,
                "strategy_digest": strategy_digest,
                "strategy_scope": strategy_scope,
                "strategy_refreshes": strategy_refreshes,
                "strategy_provider": strategy_provider_metadata,
                "strategy_refresh_on_uncertainty": (
                    config.strategy_refresh_on_uncertainty
                ),
                "strategy_refresh_uncertainty_threshold": (
                    config.strategy_refresh_uncertainty_threshold
                ),
                "adaptive_search": config.adaptive_search,
                "adaptive_min_simulations": config.adaptive_min_simulations,
                "adaptive_max_simulations": config.adaptive_max_simulations,
                "model_architecture": model_architecture,
            },
            search_budget=config.simulations,
            model_architecture=model_architecture,
        ),
    )
    if trace_callback is not None:
        trace_callback(
            SelfPlayTraceEvent(
                event="game_completed",
                game_index=game_index,
                games_total=games_total,
                ply=len(moves),
                after_state=state,
                outcome=outcome,
                termination_reason=termination_reason,
                used_model=policy_value_model is not None,
                metadata={
                    "plies_played": len(moves),
                    "cutoff_adjudication": (
                        adjudication.to_dict() if adjudication is not None else None
                    ),
                },
            )
        )
    if progress_callback is not None:
        progress_callback(
            SelfPlayProgress(
                event="game_completed",
                game_index=game_index,
                games_total=games_total,
                plies_played=len(moves),
                outcome=outcome,
                termination_reason=termination_reason,
                used_model=policy_value_model is not None,
            )
        )
    return game


def _select_move_for_selfplay(
    *,
    search_result: SearchResult,
    temperature: float,
    rng: random.Random,
) -> tuple[Move | None, str | None]:
    if search_result.selected_move is None or search_result.selected_move_id is None:
        return None, None
    if temperature <= 0.0:
        return search_result.selected_move, search_result.selected_move_id

    move_by_id = {
        stable_move_id(move): move for move in search_result.legal_moves
    }
    weighted_ids: list[str] = []
    weights: list[float] = []
    power = 1.0 / temperature
    for move_id in sorted(move_by_id):
        if move_id not in search_result.visit_counts:
            continue
        count = search_result.visit_counts[move_id]
        if count <= 0:
            continue
        weighted_ids.append(move_id)
        weights.append(float(count) ** power)

    if not weighted_ids:
        return search_result.selected_move, search_result.selected_move_id
    chosen_index = rng.choices(range(len(weighted_ids)), weights=weights, k=1)[0]
    chosen_move_id = weighted_ids[chosen_index]
    return move_by_id[chosen_move_id], chosen_move_id


def _outcome_for_perspective(outcome: str, perspective: Color) -> float:
    if outcome == "draw":
        return 0.0
    if outcome == perspective.value:
        return 1.0
    return -1.0


def _with_side_to_move(state: GameState, side_to_move: Color) -> GameState:
    return GameState(
        piece_classes=state.piece_classes,
        piece_instances=state.piece_instances,
        side_to_move=side_to_move,
        castling_rights=state.castling_rights,
        en_passant_square=state.en_passant_square,
        halfmove_clock=state.halfmove_clock,
        fullmove_number=state.fullmove_number,
        repetition_counts=state.repetition_counts,
        pending_events=state.pending_events,
        metadata=state.metadata,
    )


def _clone_state(state: GameState) -> GameState:
    return GameState.from_dict(state.to_dict())


def _parallel_selfplay_worker(job: dict[str, object]) -> dict[str, JsonValue]:
    config = SelfPlayConfig(**dict(job["config"]))
    game_index = int(job["game_index"])
    game_seed = seed_for_game(config.seed, game_index)
    checkpoint_path = job.get("checkpoint_path")
    inference_client = job.get("inference_client")
    model = None
    if inference_client is not None:
        model = inference_client
    elif checkpoint_path is not None:
        device = str(job["device"]) if job.get("device") is not None else None
        cache_key = (str(checkpoint_path), device)
        model = _WORKER_MODEL_CACHE.get(cache_key)
        if model is None:
            model = load_training_checkpoint(str(checkpoint_path), device=device).model
            _WORKER_MODEL_CACHE[cache_key] = model
    game = _play_single_game(
        initial_state=GameState.from_dict(dict(job["initial_state"])),
        game_index=game_index,
        games_total=int(job["games_total"]),
        config=config,
        game_seed=game_seed,
        rng=random.Random(game_seed),
        policy_value_model=model,
        evaluator=None,
        strategy_provider=None,
        progress_callback=None,
        trace_callback=None,
    )
    game.metadata["parallel_worker"] = True
    game.replay_trace.metadata["parallel_worker"] = True
    return game.to_dict()


def _parallel_strategy_payloads(
    *,
    initial_states: Sequence[GameState],
    games: int,
    config: SelfPlayConfig,
    strategy_provider: StrategyProvider | None,
) -> list[dict[str, JsonValue] | None]:
    if strategy_provider is None:
        return [_strategy_payload(config.strategy) for _ in range(games)]
    payloads: list[dict[str, JsonValue] | None] = []
    for game_index in range(games):
        state = _clone_state(initial_states[game_index % len(initial_states)])
        strategy_payload, _ = _resolve_strategy(
            state=state,
            strategy=config.strategy,
            strategy_provider=strategy_provider,
            phase="game_start",
            metadata={
                "game_index": game_index,
                "games_total": games,
            },
        )
        payloads.append(strategy_payload)
    return payloads


def _resolve_strategy(
    *,
    state: GameState,
    strategy: StrategyHypothesis | dict[str, JsonValue] | None,
    strategy_provider: StrategyProvider | None,
    phase: str,
    metadata: dict[str, JsonValue],
) -> tuple[dict[str, JsonValue] | None, dict[str, JsonValue]]:
    strategy_payload = _strategy_payload(strategy)
    if strategy_provider is None:
        return strategy_payload, {"provider": None, "phase": phase}
    request = StrategyRequest(
        state=state.to_dict(),
        world_summary=_world_summary(state),
        phase=phase,
        prior_strategy=strategy_payload,
        metadata=metadata,
    )
    response = strategy_provider.propose_strategy(request)
    return dict(response.strategy), {
        "provider": type(strategy_provider).__name__,
        "phase": phase,
        "explanation": response.explanation,
        **dict(response.metadata),
    }


def _strategy_payload(
    strategy: StrategyHypothesis | dict[str, JsonValue] | None,
) -> dict[str, JsonValue] | None:
    if strategy is None:
        return None
    return dict(to_primitive(strategy))


def _strategy_digest(strategy: dict[str, JsonValue] | None) -> str | None:
    if strategy is None:
        return None
    return compute_strategy_digest(strategy)


def _strategy_scope(strategy: dict[str, JsonValue] | None) -> str | None:
    if strategy is None:
        return None
    scope = strategy.get("scope")
    return None if scope is None else str(scope)


def _model_architecture(policy_value_model: PolicyValueModel | None) -> str:
    if policy_value_model is None:
        return SEARCH_ONLY_MODEL_ARCHITECTURE
    config = getattr(policy_value_model, "config", None)
    architecture = getattr(config, "architecture", None)
    if architecture is None:
        return type(policy_value_model).__name__
    return str(architecture)


def _coerce_search_budget(value: JsonValue | None, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                return int(stripped)
            except ValueError:
                return default
    return default


def _world_summary(state: GameState) -> dict[str, JsonValue]:
    return {
        "side_to_move": state.side_to_move.value,
        "piece_classes": {
            class_id: dict(to_primitive(piece_class))
            for class_id, piece_class in sorted(state.piece_classes.items())
        },
        "active_piece_count": len(state.active_pieces()),
    }
