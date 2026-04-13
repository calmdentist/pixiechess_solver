from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from pixie_solver.core import BasePieceType, GameState, Move
from pixie_solver.simulator.engine import apply_move, result
from pixie_solver.simulator.movegen import legal_moves
from pixie_solver.training.selfplay import seed_for_game
from pixie_solver.utils import build_replay_trace, replay_trace
from pixie_solver.utils.serialization import JsonValue


@dataclass(frozen=True, slots=True)
class SimulatorStressConfig:
    games: int = 32
    max_plies: int = 64
    seed: int = 0
    verify_all_legal_moves: bool = True

    def __post_init__(self) -> None:
        if self.games < 1:
            raise ValueError("games must be at least 1")
        if self.max_plies < 0:
            raise ValueError("max_plies must be non-negative")


@dataclass(slots=True)
class SimulatorStressSummary:
    config: SimulatorStressConfig
    games_started: int = 0
    games_completed: int = 0
    plies_played: int = 0
    legal_moves_checked: int = 0
    replay_hash_failures: int = 0
    invariant_failures: int = 0
    illegal_apply_failures: int = 0
    king_sanity_failures: int = 0
    failures: list[dict[str, JsonValue]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.replay_hash_failures == 0
            and self.invariant_failures == 0
            and self.illegal_apply_failures == 0
            and self.king_sanity_failures == 0
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "ok": self.ok,
            "config": {
                "games": self.config.games,
                "max_plies": self.config.max_plies,
                "seed": self.config.seed,
                "verify_all_legal_moves": self.config.verify_all_legal_moves,
            },
            "games_started": self.games_started,
            "games_completed": self.games_completed,
            "plies_played": self.plies_played,
            "legal_moves_checked": self.legal_moves_checked,
            "replay_hash_failures": self.replay_hash_failures,
            "invariant_failures": self.invariant_failures,
            "illegal_apply_failures": self.illegal_apply_failures,
            "king_sanity_failures": self.king_sanity_failures,
            "failures": list(self.failures),
        }


def run_simulator_stress(
    initial_states: Sequence[GameState],
    *,
    config: SimulatorStressConfig | None = None,
) -> SimulatorStressSummary:
    if not initial_states:
        raise ValueError("initial_states must contain at least one GameState")

    active_config = SimulatorStressConfig() if config is None else config
    summary = SimulatorStressSummary(config=active_config)
    for game_index in range(active_config.games):
        initial_state = GameState.from_dict(
            initial_states[game_index % len(initial_states)].to_dict()
        )
        summary.games_started += 1
        game_seed = seed_for_game(active_config.seed, game_index)
        rng = random.Random(game_seed)
        moves: list[Move] = []
        state = initial_state

        if not _has_one_king_per_side(state):
            summary.king_sanity_failures += 1
            summary.failures.append(
                {
                    "game_index": game_index,
                    "seed": game_seed,
                    "stage": "initial_king_sanity",
                    "state_hash": state.state_hash(),
                }
            )
            continue

        for ply in range(active_config.max_plies):
            try:
                state.validate()
            except Exception as exc:
                summary.invariant_failures += 1
                summary.failures.append(
                    _failure(
                        game_index=game_index,
                        seed=game_seed,
                        ply=ply,
                        stage="state_validate",
                        state=state,
                        error=exc,
                    )
                )
                break

            if result(state) is not None:
                break

            try:
                current_moves = legal_moves(state)
            except Exception as exc:
                summary.invariant_failures += 1
                summary.failures.append(
                    _failure(
                        game_index=game_index,
                        seed=game_seed,
                        ply=ply,
                        stage="legal_moves",
                        state=state,
                        error=exc,
                    )
                )
                break

            if active_config.verify_all_legal_moves:
                for candidate in current_moves:
                    summary.legal_moves_checked += 1
                    try:
                        apply_move(state, candidate)
                    except Exception as exc:
                        summary.illegal_apply_failures += 1
                        summary.failures.append(
                            {
                                **_failure(
                                    game_index=game_index,
                                    seed=game_seed,
                                    ply=ply,
                                    stage="apply_legal_move",
                                    state=state,
                                    error=exc,
                                ),
                                "move": candidate.to_dict(),
                            }
                        )
                        break
                if summary.illegal_apply_failures:
                    break

            if not current_moves:
                break

            move = rng.choice(current_moves)
            try:
                state, _ = apply_move(state, move)
            except Exception as exc:
                summary.illegal_apply_failures += 1
                summary.failures.append(
                    {
                        **_failure(
                            game_index=game_index,
                            seed=game_seed,
                            ply=ply,
                            stage="apply_selected_move",
                            state=state,
                            error=exc,
                        ),
                        "move": move.to_dict(),
                    }
                )
                break
            moves.append(move)
            summary.plies_played += 1
        else:
            summary.games_completed += 1

        if summary.failures and summary.failures[-1].get("game_index") == game_index:
            continue

        try:
            trace = build_replay_trace(
                initial_state,
                moves,
                metadata={"mode": "simulator_stress", "seed": game_seed},
            )
            replayed = replay_trace(trace)
            if replayed.state_hash() != state.state_hash():
                raise ValueError("final replay hash differs from stress state")
        except Exception as exc:
            summary.replay_hash_failures += 1
            summary.failures.append(
                _failure(
                    game_index=game_index,
                    seed=game_seed,
                    ply=len(moves),
                    stage="replay",
                    state=state,
                    error=exc,
                )
            )
            continue

        if len(moves) < active_config.max_plies:
            summary.games_completed += 1

    return summary


def _has_one_king_per_side(state: GameState) -> bool:
    counts = {"white": 0, "black": 0}
    for piece in state.active_pieces():
        piece_class = state.piece_classes[piece.piece_class_id]
        if piece_class.base_piece_type == BasePieceType.KING:
            counts[piece.color.value] += 1
    return counts == {"white": 1, "black": 1}


def _failure(
    *,
    game_index: int,
    seed: int,
    ply: int,
    stage: str,
    state: GameState,
    error: Exception,
) -> dict[str, JsonValue]:
    return {
        "game_index": game_index,
        "seed": seed,
        "ply": ply,
        "stage": stage,
        "state_hash": state.state_hash(),
        "error": str(error),
    }
