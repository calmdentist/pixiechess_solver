from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from pixie_solver.core import Color, GameState, Move, stable_move_id
from pixie_solver.model.policy_value import PolicyValueModel
from pixie_solver.search import SearchResult, StateEvaluator, run_mcts
from pixie_solver.simulator.engine import apply_move, result
from pixie_solver.training.dataset import SelfPlayExample, SelfPlayGame
from pixie_solver.utils import build_replay_trace


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    simulations: int = 64
    max_plies: int = 256
    opening_temperature: float = 1.0
    final_temperature: float = 0.0
    temperature_drop_after_ply: int = 12
    c_puct: float = 1.5
    seed: int = 0

    def temperature_for_ply(self, ply: int) -> float:
        if ply < self.temperature_drop_after_ply:
            return self.opening_temperature
        return self.final_temperature


def generate_selfplay_games(
    initial_states: Sequence[GameState],
    *,
    games: int,
    config: SelfPlayConfig | None = None,
    policy_value_model: PolicyValueModel | None = None,
    evaluator: StateEvaluator | None = None,
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
                config=active_config,
                rng=rng,
                policy_value_model=policy_value_model,
                evaluator=evaluator,
            )
        )
    return generated_games


def flatten_selfplay_examples(games: Sequence[SelfPlayGame]) -> list[SelfPlayExample]:
    examples: list[SelfPlayExample] = []
    for game in games:
        examples.extend(game.examples)
    return examples


def _play_single_game(
    *,
    initial_state: GameState,
    game_index: int,
    config: SelfPlayConfig,
    rng: random.Random,
    policy_value_model: PolicyValueModel | None,
    evaluator: StateEvaluator | None,
) -> SelfPlayGame:
    state = initial_state
    examples: list[SelfPlayExample] = []
    moves: list[Move] = []
    termination_reason = "max_plies"
    game_result: str | None = None

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
        )
        chosen_move, chosen_move_id = _select_move_for_selfplay(
            search_result=search_result,
            temperature=config.temperature_for_ply(ply),
            rng=rng,
        )
        if chosen_move is None or chosen_move_id is None:
            termination_reason = "no_legal_moves"
            break

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
        state, _ = apply_move(state, chosen_move)
    else:
        game_result = result(state)

    outcome = "draw" if game_result is None else game_result
    for example in examples:
        example.outcome = _outcome_for_perspective(outcome, example.state.side_to_move)

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
            "termination_reason": termination_reason,
            "outcome": outcome,
        },
    )
    return SelfPlayGame(
        replay_trace=replay_trace,
        examples=tuple(examples),
        outcome=outcome,
        metadata={
            "game_index": game_index,
            "plies_played": len(moves),
            "termination_reason": termination_reason,
        },
    )


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


def _clone_state(state: GameState) -> GameState:
    return GameState.from_dict(state.to_dict())
