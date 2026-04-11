from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pixie_solver.core import Color, GameState, Move, StateDelta, stable_move_id
from pixie_solver.model.policy_value import PolicyValueModel
from pixie_solver.search import (
    DirichletRootNoise,
    HeuristicEvaluator,
    SearchResult,
    StateEvaluator,
    run_mcts,
)
from pixie_solver.simulator.engine import apply_move, result
from pixie_solver.training.dataset import SelfPlayExample, SelfPlayGame
from pixie_solver.utils import build_replay_trace
from pixie_solver.utils.serialization import JsonValue


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

    def __post_init__(self) -> None:
        if self.max_plies < 0:
            raise ValueError("max_plies must be non-negative")
        if self.root_dirichlet_alpha <= 0.0:
            raise ValueError("root_dirichlet_alpha must be positive")
        if self.root_exploration_fraction < 0.0 or self.root_exploration_fraction > 1.0:
            raise ValueError("root_exploration_fraction must be in [0, 1]")
        if self.adjudication_threshold < 0.0 or self.adjudication_threshold > 1.0:
            raise ValueError("adjudication_threshold must be in [0, 1]")

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
    progress_callback: Callable[[SelfPlayProgress], None] | None = None,
    trace_callback: Callable[[SelfPlayTraceEvent], None] | None = None,
) -> list[SelfPlayGame]:
    if games < 1:
        raise ValueError("games must be at least 1")
    if not initial_states:
        raise ValueError("initial_states must contain at least one GameState")

    active_config = SelfPlayConfig() if config is None else config
    rng = random.Random(active_config.seed)
    generated_games: list[SelfPlayGame] = []
    for game_index in range(games):
        seed_state = initial_states[game_index % len(initial_states)]
        generated_games.append(
            _play_single_game(
                initial_state=_clone_state(seed_state),
                game_index=game_index,
                games_total=games,
                config=active_config,
                rng=rng,
                policy_value_model=policy_value_model,
                evaluator=evaluator,
                progress_callback=progress_callback,
                trace_callback=trace_callback,
            )
        )
    return generated_games


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


def _play_single_game(
    *,
    initial_state: GameState,
    game_index: int,
    games_total: int,
    config: SelfPlayConfig,
    rng: random.Random,
    policy_value_model: PolicyValueModel | None,
    evaluator: StateEvaluator | None,
    progress_callback: Callable[[SelfPlayProgress], None] | None,
    trace_callback: Callable[[SelfPlayTraceEvent], None] | None,
) -> SelfPlayGame:
    state = initial_state
    examples: list[SelfPlayExample] = []
    moves: list[Move] = []
    termination_reason = "max_plies"
    game_result: str | None = None
    adjudication: CutoffAdjudication | None = None
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
                metadata={
                    "game_index": game_index,
                    "ply": ply,
                    "temperature": config.temperature_for_ply(ply),
                    **dict(search_result.metadata),
                },
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
        metadata={
            "game_index": game_index,
            "seed": config.seed,
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
            "cutoff_adjudication": (
                adjudication.to_dict() if adjudication is not None else None
            ),
            "termination_reason": termination_reason,
            "outcome": outcome,
        },
    )
    game = SelfPlayGame(
        replay_trace=replay_trace,
        examples=tuple(examples),
        outcome=outcome,
        metadata={
            "game_index": game_index,
            "plies_played": len(moves),
            "termination_reason": termination_reason,
            "cutoff_adjudication": (
                adjudication.to_dict() if adjudication is not None else None
            ),
        },
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
