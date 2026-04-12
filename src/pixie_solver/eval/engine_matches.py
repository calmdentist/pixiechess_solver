from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pixie_solver.core import Color, GameState, Move
from pixie_solver.model.policy_value import PolicyValueModel
from pixie_solver.search import StateEvaluator, run_mcts
from pixie_solver.simulator.engine import apply_move, result
from pixie_solver.training import load_training_checkpoint
from pixie_solver.training.selfplay import CutoffAdjudication, adjudicate_cutoff
from pixie_solver.utils import build_replay_trace
from pixie_solver.utils.serialization import JsonValue, ReplayTrace, read_jsonl, write_jsonl


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    games: int = 2
    simulations: int = 16
    max_plies: int = 80
    c_puct: float = 1.5
    seed: int = 0
    alternate_colors: bool = True
    adjudicate_max_plies: bool = True
    adjudication_threshold: float = 0.2

    def __post_init__(self) -> None:
        if self.games < 1:
            raise ValueError("games must be at least 1")
        if self.simulations < 1:
            raise ValueError("simulations must be at least 1")
        if self.max_plies < 0:
            raise ValueError("max_plies must be non-negative")
        if self.adjudication_threshold < 0.0 or self.adjudication_threshold > 1.0:
            raise ValueError("adjudication_threshold must be in [0, 1]")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "games": self.games,
            "simulations": self.simulations,
            "max_plies": self.max_plies,
            "c_puct": self.c_puct,
            "seed": self.seed,
            "alternate_colors": self.alternate_colors,
            "adjudicate_max_plies": self.adjudicate_max_plies,
            "adjudication_threshold": self.adjudication_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "ArenaConfig":
        return cls(
            games=int(data.get("games", 2)),
            simulations=int(data.get("simulations", 16)),
            max_plies=int(data.get("max_plies", 80)),
            c_puct=float(data.get("c_puct", 1.5)),
            seed=int(data.get("seed", 0)),
            alternate_colors=bool(data.get("alternate_colors", True)),
            adjudicate_max_plies=bool(data.get("adjudicate_max_plies", True)),
            adjudication_threshold=float(data.get("adjudication_threshold", 0.2)),
        )


@dataclass(frozen=True, slots=True)
class ArenaProgress:
    event: str
    game_index: int
    games_total: int
    ply: int | None = None
    candidate_color: str | None = None
    outcome: str | None = None
    candidate_score: float | None = None
    termination_reason: str | None = None


@dataclass(slots=True)
class ArenaGameResult:
    game_index: int
    seed: int
    candidate_color: Color
    baseline_color: Color
    outcome: str
    winner: str
    candidate_score: float
    plies_played: int
    termination_reason: str
    final_state_hash: str
    selected_move_ids: tuple[str, ...] = ()
    replay_trace: ReplayTrace | None = None
    cutoff_adjudication: CutoffAdjudication | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.selected_move_ids = tuple(str(move_id) for move_id in self.selected_move_ids)
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "game_index": self.game_index,
            "seed": self.seed,
            "candidate_color": self.candidate_color.value,
            "baseline_color": self.baseline_color.value,
            "outcome": self.outcome,
            "winner": self.winner,
            "candidate_score": self.candidate_score,
            "plies_played": self.plies_played,
            "termination_reason": self.termination_reason,
            "final_state_hash": self.final_state_hash,
            "selected_move_ids": list(self.selected_move_ids),
            "replay_trace": (
                self.replay_trace.to_dict() if self.replay_trace is not None else None
            ),
            "cutoff_adjudication": (
                self.cutoff_adjudication.to_dict()
                if self.cutoff_adjudication is not None
                else None
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "ArenaGameResult":
        cutoff_payload = data.get("cutoff_adjudication")
        replay_payload = data.get("replay_trace")
        return cls(
            game_index=int(data["game_index"]),
            seed=int(data["seed"]),
            candidate_color=Color(str(data["candidate_color"])),
            baseline_color=Color(str(data["baseline_color"])),
            outcome=str(data["outcome"]),
            winner=str(data["winner"]),
            candidate_score=float(data["candidate_score"]),
            plies_played=int(data["plies_played"]),
            termination_reason=str(data["termination_reason"]),
            final_state_hash=str(data["final_state_hash"]),
            selected_move_ids=tuple(
                str(move_id) for move_id in data.get("selected_move_ids", [])
            ),
            replay_trace=(
                ReplayTrace.from_dict(dict(replay_payload))
                if replay_payload is not None
                else None
            ),
            cutoff_adjudication=(
                CutoffAdjudication(
                    outcome=str(dict(cutoff_payload)["outcome"]),
                    score=float(dict(cutoff_payload)["score"]),
                    threshold=float(dict(cutoff_payload)["threshold"]),
                    method=str(
                        dict(cutoff_payload).get(
                            "method",
                            "heuristic_material_mobility_check",
                        )
                    ),
                )
                if cutoff_payload is not None
                else None
            ),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class ArenaSummary:
    candidate_id: str
    baseline_id: str
    config: ArenaConfig
    games: tuple[ArenaGameResult, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.games = tuple(self.games)
        self.metadata = dict(self.metadata)

    @property
    def games_played(self) -> int:
        return len(self.games)

    @property
    def candidate_wins(self) -> int:
        return sum(1 for game in self.games if game.winner == "candidate")

    @property
    def baseline_wins(self) -> int:
        return sum(1 for game in self.games if game.winner == "baseline")

    @property
    def draws(self) -> int:
        return sum(1 for game in self.games if game.winner == "draw")

    @property
    def candidate_points(self) -> float:
        return sum(game.candidate_score for game in self.games)

    @property
    def candidate_score_rate(self) -> float:
        if not self.games:
            return 0.0
        return self.candidate_points / len(self.games)

    @property
    def candidate_win_rate(self) -> float:
        if not self.games:
            return 0.0
        return self.candidate_wins / len(self.games)

    @property
    def baseline_win_rate(self) -> float:
        if not self.games:
            return 0.0
        return self.baseline_wins / len(self.games)

    @property
    def draw_rate(self) -> float:
        if not self.games:
            return 0.0
        return self.draws / len(self.games)

    @property
    def average_plies(self) -> float:
        if not self.games:
            return 0.0
        return sum(game.plies_played for game in self.games) / len(self.games)

    @property
    def score_rate_ci95(self) -> dict[str, float]:
        if not self.games:
            return {"lower": 0.0, "upper": 0.0}
        scores = [game.candidate_score for game in self.games]
        mean = sum(scores) / len(scores)
        if len(scores) == 1:
            return {"lower": mean, "upper": mean}
        variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
        margin = 1.96 * math.sqrt(variance / len(scores))
        return {
            "lower": max(0.0, mean - margin),
            "upper": min(1.0, mean + margin),
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "candidate_id": self.candidate_id,
            "baseline_id": self.baseline_id,
            "config": self.config.to_dict(),
            "games_played": self.games_played,
            "candidate_wins": self.candidate_wins,
            "draws": self.draws,
            "candidate_losses": self.baseline_wins,
            "baseline_wins": self.baseline_wins,
            "candidate_points": self.candidate_points,
            "candidate_score_rate": self.candidate_score_rate,
            "candidate_score_rate_ci95": self.score_rate_ci95,
            "candidate_win_rate": self.candidate_win_rate,
            "draw_rate": self.draw_rate,
            "baseline_win_rate": self.baseline_win_rate,
            "average_plies": self.average_plies,
            "outcomes": _counter_dict(game.outcome for game in self.games),
            "termination_reasons": _counter_dict(
                game.termination_reason for game in self.games
            ),
            "candidate_color_counts": _counter_dict(
                game.candidate_color.value for game in self.games
            ),
            "games": [game.to_dict() for game in self.games],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "ArenaSummary":
        return cls(
            candidate_id=str(data["candidate_id"]),
            baseline_id=str(data["baseline_id"]),
            config=ArenaConfig.from_dict(dict(data["config"])),
            games=tuple(
                ArenaGameResult.from_dict(dict(game_data))
                for game_data in data.get("games", [])
            ),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promoted: bool
    reason: str
    candidate_checkpoint: str
    champion_checkpoint: str | None
    selected_checkpoint: str
    candidate_score_rate: float | None = None
    threshold: float | None = None
    arena_summary: dict[str, JsonValue] | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "promoted": self.promoted,
            "reason": self.reason,
            "candidate_checkpoint": self.candidate_checkpoint,
            "champion_checkpoint": self.champion_checkpoint,
            "selected_checkpoint": self.selected_checkpoint,
            "candidate_score_rate": self.candidate_score_rate,
            "threshold": self.threshold,
            "arena_summary": self.arena_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "PromotionDecision":
        return cls(
            promoted=bool(data["promoted"]),
            reason=str(data["reason"]),
            candidate_checkpoint=str(data["candidate_checkpoint"]),
            champion_checkpoint=(
                str(data["champion_checkpoint"])
                if data.get("champion_checkpoint") is not None
                else None
            ),
            selected_checkpoint=str(data["selected_checkpoint"]),
            candidate_score_rate=(
                float(data["candidate_score_rate"])
                if data.get("candidate_score_rate") is not None
                else None
            ),
            threshold=(
                float(data["threshold"]) if data.get("threshold") is not None else None
            ),
            arena_summary=(
                dict(data["arena_summary"])
                if data.get("arena_summary") is not None
                else None
            ),
        )


def run_engine_match(*, white: str, black: str) -> dict[str, object]:
    raise NotImplementedError(
        f"Use `pixie arena` for checkpoint-vs-checkpoint evaluation. Requested white={white!r}, black={black!r}."
    )


def run_checkpoint_arena(
    *,
    candidate_model: PolicyValueModel,
    baseline_model: PolicyValueModel,
    initial_states: Sequence[GameState],
    candidate_id: str,
    baseline_id: str,
    config: ArenaConfig | None = None,
    evaluator: StateEvaluator | None = None,
    progress_callback: Callable[[ArenaProgress], None] | None = None,
) -> ArenaSummary:
    if not initial_states:
        raise ValueError("initial_states must contain at least one GameState")

    active_config = ArenaConfig() if config is None else config
    games: list[ArenaGameResult] = []
    for game_index in range(active_config.games):
        candidate_color = (
            Color.WHITE
            if not active_config.alternate_colors or game_index % 2 == 0
            else Color.BLACK
        )
        game_seed = active_config.seed + game_index
        seed_state = initial_states[game_index % len(initial_states)]
        games.append(
            _play_arena_game(
                initial_state=_clone_state(seed_state),
                game_index=game_index,
                games_total=active_config.games,
                game_seed=game_seed,
                candidate_color=candidate_color,
                candidate_model=candidate_model,
                baseline_model=baseline_model,
                candidate_id=candidate_id,
                baseline_id=baseline_id,
                config=active_config,
                evaluator=evaluator,
                progress_callback=progress_callback,
            )
        )

    return ArenaSummary(
        candidate_id=candidate_id,
        baseline_id=baseline_id,
        config=active_config,
        games=tuple(games),
    )


def run_checkpoint_arena_from_paths(
    *,
    candidate_checkpoint: str | Path,
    baseline_checkpoint: str | Path,
    initial_states: Sequence[GameState],
    config: ArenaConfig | None = None,
    device: str | None = None,
    evaluator: StateEvaluator | None = None,
    progress_callback: Callable[[ArenaProgress], None] | None = None,
) -> ArenaSummary:
    candidate_path = Path(candidate_checkpoint)
    baseline_path = Path(baseline_checkpoint)
    candidate = load_training_checkpoint(candidate_path, device=device)
    baseline = load_training_checkpoint(baseline_path, device=device)
    return run_checkpoint_arena(
        candidate_model=candidate.model,
        baseline_model=baseline.model,
        initial_states=initial_states,
        candidate_id=str(candidate_path),
        baseline_id=str(baseline_path),
        config=config,
        evaluator=evaluator,
        progress_callback=progress_callback,
    )


def decide_promotion(
    *,
    candidate_checkpoint: str | Path,
    champion_checkpoint: str | Path | None,
    threshold: float,
    arena_summary: ArenaSummary | None = None,
) -> PromotionDecision:
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")
    candidate_path = str(candidate_checkpoint)
    if champion_checkpoint is None:
        return PromotionDecision(
            promoted=True,
            reason="initial_candidate",
            candidate_checkpoint=candidate_path,
            champion_checkpoint=None,
            selected_checkpoint=candidate_path,
            threshold=threshold,
        )
    champion_path = str(champion_checkpoint)
    if arena_summary is None:
        raise ValueError("arena_summary is required when champion_checkpoint is set")

    score_rate = arena_summary.candidate_score_rate
    promoted = score_rate >= threshold
    return PromotionDecision(
        promoted=promoted,
        reason=(
            "candidate_beat_champion"
            if promoted
            else "candidate_did_not_clear_gate"
        ),
        candidate_checkpoint=candidate_path,
        champion_checkpoint=champion_path,
        selected_checkpoint=candidate_path if promoted else champion_path,
        candidate_score_rate=score_rate,
        threshold=threshold,
        arena_summary=arena_summary.to_dict(),
    )


def write_arena_games_jsonl(
    path: str | Path,
    games: Iterable[ArenaGameResult],
) -> None:
    write_jsonl(path, (game.to_dict() for game in games))


def read_arena_games_jsonl(path: str | Path) -> list[ArenaGameResult]:
    return [ArenaGameResult.from_dict(record) for record in read_jsonl(path)]


def _play_arena_game(
    *,
    initial_state: GameState,
    game_index: int,
    games_total: int,
    game_seed: int,
    candidate_color: Color,
    candidate_model: PolicyValueModel,
    baseline_model: PolicyValueModel,
    candidate_id: str,
    baseline_id: str,
    config: ArenaConfig,
    evaluator: StateEvaluator | None,
    progress_callback: Callable[[ArenaProgress], None] | None,
) -> ArenaGameResult:
    state = initial_state
    moves: list[Move] = []
    selected_move_ids: list[str] = []
    termination_reason = "max_plies"
    game_result: str | None = None
    adjudication: CutoffAdjudication | None = None
    rng = random.Random(game_seed)
    baseline_color = _other_color(candidate_color)
    if progress_callback is not None:
        progress_callback(
            ArenaProgress(
                event="game_started",
                game_index=game_index,
                games_total=games_total,
                candidate_color=candidate_color.value,
            )
        )

    for ply in range(config.max_plies):
        game_result = result(state)
        if game_result is not None:
            termination_reason = "terminal"
            break

        model = (
            candidate_model
            if state.side_to_move == candidate_color
            else baseline_model
        )
        search_result = run_mcts(
            state,
            simulations=config.simulations,
            policy_value_model=model,
            evaluator=evaluator,
            c_puct=config.c_puct,
            root_noise=None,
            rng=rng,
        )
        if search_result.selected_move is None or search_result.selected_move_id is None:
            termination_reason = "no_legal_moves"
            break

        selected_move_ids.append(search_result.selected_move_id)
        moves.append(search_result.selected_move)
        state, _ = apply_move(state, search_result.selected_move)
        if progress_callback is not None:
            progress_callback(
                ArenaProgress(
                    event="ply_completed",
                    game_index=game_index,
                    games_total=games_total,
                    ply=ply + 1,
                    candidate_color=candidate_color.value,
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

    candidate_score = _candidate_score(outcome, candidate_color)
    winner = _winner_for_outcome(outcome, candidate_color)
    replay_trace = build_replay_trace(
        initial_state,
        moves,
        metadata={
            "mode": "checkpoint_arena",
            "game_index": game_index,
            "seed": game_seed,
            "candidate_id": candidate_id,
            "baseline_id": baseline_id,
            "candidate_color": candidate_color.value,
            "baseline_color": baseline_color.value,
            "outcome": outcome,
            "winner": winner,
            "candidate_score": candidate_score,
            "termination_reason": termination_reason,
            "simulations": config.simulations,
            "max_plies": config.max_plies,
            "c_puct": config.c_puct,
            "root_noise_applied": False,
            "adjudicate_max_plies": config.adjudicate_max_plies,
            "adjudication_threshold": config.adjudication_threshold,
            "cutoff_adjudication": (
                adjudication.to_dict() if adjudication is not None else None
            ),
        },
    )
    game = ArenaGameResult(
        game_index=game_index,
        seed=game_seed,
        candidate_color=candidate_color,
        baseline_color=baseline_color,
        outcome=outcome,
        winner=winner,
        candidate_score=candidate_score,
        plies_played=len(moves),
        termination_reason=termination_reason,
        final_state_hash=state.state_hash(),
        selected_move_ids=tuple(selected_move_ids),
        replay_trace=replay_trace,
        cutoff_adjudication=adjudication,
        metadata={
            "candidate_id": candidate_id,
            "baseline_id": baseline_id,
        },
    )
    if progress_callback is not None:
        progress_callback(
            ArenaProgress(
                event="game_completed",
                game_index=game_index,
                games_total=games_total,
                ply=len(moves),
                candidate_color=candidate_color.value,
                outcome=outcome,
                candidate_score=candidate_score,
                termination_reason=termination_reason,
            )
        )
    return game


def _candidate_score(outcome: str, candidate_color: Color) -> float:
    if outcome == "draw":
        return 0.5
    if outcome == candidate_color.value:
        return 1.0
    return 0.0


def _winner_for_outcome(outcome: str, candidate_color: Color) -> str:
    if outcome == "draw":
        return "draw"
    if outcome == candidate_color.value:
        return "candidate"
    return "baseline"


def _other_color(color: Color) -> Color:
    return Color.BLACK if color == Color.WHITE else Color.WHITE


def _clone_state(state: GameState) -> GameState:
    return GameState.from_dict(state.to_dict())


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))
