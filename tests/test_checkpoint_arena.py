from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Color, Modifier, Move, PieceClass, PieceInstance
from pixie_solver.core.state import GameState
from pixie_solver.eval import (
    ArenaConfig,
    ArenaGameResult,
    ArenaSummary,
    PromotionDecision,
    decide_promotion,
    run_checkpoint_arena,
)
from pixie_solver.model import PolicyValueConfig, PolicyValueModel, PolicyValueOutput
from pixie_solver.utils.serialization import canonical_json


class _CapturePreferenceModel(PolicyValueModel):
    def __init__(self, *, prefer_capture: bool) -> None:
        super().__init__(
            PolicyValueConfig(
                d_model=32,
                num_heads=4,
                num_layers=1,
                dropout=0.0,
                feedforward_multiplier=2,
            ),
            device="cpu",
        )
        self.prefer_capture = prefer_capture

    def infer(self, state: GameState, legal_moves: tuple[Move, ...]) -> PolicyValueOutput:
        return PolicyValueOutput(
            policy_logits={
                move.stable_id(): self._logit(move)
                for move in legal_moves
            },
            value=0.0,
        )

    def _logit(self, move: Move) -> float:
        is_capture = move.captured_piece_id is not None
        preferred = is_capture if self.prefer_capture else not is_capture
        return 10.0 if preferred else -10.0


class CheckpointArenaTest(unittest.TestCase):
    def setUp(self) -> None:
        king = PieceClass(
            class_id="baseline_king",
            name="Baseline King",
            base_piece_type=BasePieceType.KING,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
        )
        rook = PieceClass(
            class_id="baseline_rook",
            name="Baseline Rook",
            base_piece_type=BasePieceType.ROOK,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
        )
        self.initial_state = GameState(
            piece_classes={
                king.class_id: king,
                rook.class_id: rook,
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=king.class_id,
                    color=Color.WHITE,
                    square="a1",
                ),
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=rook.class_id,
                    color=Color.WHITE,
                    square="e7",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=king.class_id,
                    color=Color.BLACK,
                    square="e8",
                ),
            },
            side_to_move=Color.WHITE,
        )

    def test_checkpoint_arena_is_reproducible_and_json_round_trips(self) -> None:
        config = ArenaConfig(
            games=2,
            simulations=1,
            max_plies=1,
            seed=41,
            adjudicate_max_plies=False,
        )
        candidate = _CapturePreferenceModel(prefer_capture=True)
        baseline = _CapturePreferenceModel(prefer_capture=False)

        first = run_checkpoint_arena(
            candidate_model=candidate,
            baseline_model=baseline,
            initial_states=[self.initial_state],
            candidate_id="candidate",
            baseline_id="baseline",
            config=config,
        )
        second = run_checkpoint_arena(
            candidate_model=candidate,
            baseline_model=baseline,
            initial_states=[self.initial_state],
            candidate_id="candidate",
            baseline_id="baseline",
            config=config,
        )

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(2, first.games_played)
        self.assertEqual({"black": 1, "white": 1}, first.to_dict()["candidate_color_counts"])
        self.assertEqual(
            first.games_played,
            first.candidate_wins + first.draws + first.baseline_wins,
        )
        interval = first.score_rate_ci95
        self.assertLessEqual(interval["lower"], first.candidate_score_rate)
        self.assertLessEqual(first.candidate_score_rate, interval["upper"])
        self.assertTrue(all(game.replay_trace is not None for game in first.games))
        self.assertTrue(all(game.final_state_hash for game in first.games))

        round_tripped = ArenaSummary.from_dict(
            json.loads(canonical_json(first.to_dict()))
        )
        self.assertEqual(first.to_dict(), round_tripped.to_dict())

    def test_promotion_gate_accepts_only_candidate_that_beats_champion(self) -> None:
        winning_summary = ArenaSummary(
            candidate_id="candidate",
            baseline_id="champion",
            config=ArenaConfig(games=2),
            games=(
                _arena_game(candidate_score=1.0, winner="candidate"),
                _arena_game(candidate_score=1.0, winner="candidate", game_index=1),
            ),
        )
        winning_decision = decide_promotion(
            candidate_checkpoint="candidate.pt",
            champion_checkpoint="champion.pt",
            threshold=0.55,
            arena_summary=winning_summary,
        )

        self.assertTrue(winning_decision.promoted)
        self.assertEqual("candidate_beat_champion", winning_decision.reason)
        self.assertEqual("candidate.pt", winning_decision.selected_checkpoint)

        tied_summary = ArenaSummary(
            candidate_id="candidate",
            baseline_id="champion",
            config=ArenaConfig(games=2),
            games=(
                _arena_game(candidate_score=1.0, winner="candidate"),
                _arena_game(candidate_score=0.0, winner="baseline", game_index=1),
            ),
        )
        tied_decision = decide_promotion(
            candidate_checkpoint="candidate.pt",
            champion_checkpoint="champion.pt",
            threshold=0.55,
            arena_summary=tied_summary,
        )

        self.assertFalse(tied_decision.promoted)
        self.assertEqual("candidate_did_not_clear_gate", tied_decision.reason)
        self.assertEqual("champion.pt", tied_decision.selected_checkpoint)
        round_tripped = PromotionDecision.from_dict(
            json.loads(canonical_json(tied_decision.to_dict()))
        )
        self.assertEqual(tied_decision.to_dict(), round_tripped.to_dict())


def _arena_game(
    *,
    candidate_score: float,
    winner: str,
    game_index: int = 0,
) -> ArenaGameResult:
    outcome = "draw"
    if winner == "candidate":
        outcome = "white"
    elif winner == "baseline":
        outcome = "black"
    return ArenaGameResult(
        game_index=game_index,
        seed=game_index,
        candidate_color=Color.WHITE,
        baseline_color=Color.BLACK,
        outcome=outcome,
        winner=winner,
        candidate_score=candidate_score,
        plies_played=1,
        termination_reason="terminal",
        final_state_hash=f"hash-{game_index}",
    )


if __name__ == "__main__":
    unittest.main()
