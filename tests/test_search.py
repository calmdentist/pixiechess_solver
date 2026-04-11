from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Color, Modifier, Move, PieceClass, PieceInstance
from pixie_solver.core import stable_move_id
from pixie_solver.core.state import GameState
from pixie_solver.model import PolicyValueModel, PolicyValueOutput
from pixie_solver.search import DirichletRootNoise, run_mcts
from pixie_solver.simulator.movegen import legal_moves


class _BiasedPolicyValueModel(PolicyValueModel):
    def __init__(self, preferred_move_id: str) -> None:
        self.preferred_move_id = preferred_move_id

    def infer(self, state: GameState, legal_moves: tuple[Move, ...]) -> PolicyValueOutput:
        return PolicyValueOutput(
            policy_logits={
                stable_move_id(move): (
                    10.0 if stable_move_id(move) == self.preferred_move_id else -10.0
                )
                for move in legal_moves
            },
            value=0.0,
        )


class SearchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.king = PieceClass(
            class_id="baseline_king",
            name="Baseline King",
            base_piece_type=BasePieceType.KING,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
        )
        self.rook = PieceClass(
            class_id="baseline_rook",
            name="Baseline Rook",
            base_piece_type=BasePieceType.ROOK,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
        )
        self.queen = PieceClass(
            class_id="baseline_queen",
            name="Baseline Queen",
            base_piece_type=BasePieceType.QUEEN,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
        )

    def test_stable_move_id_round_trips(self) -> None:
        move = Move(
            piece_id="white_queen",
            from_square="a7",
            to_square="a8",
            promotion_piece_type="queen",
            metadata={"promotion_class_id": "baseline_queen"},
        )
        round_tripped = Move.from_dict(move.to_dict())

        self.assertEqual(stable_move_id(move), stable_move_id(round_tripped))
        self.assertNotEqual(
            stable_move_id(move),
            stable_move_id(
                Move(
                    piece_id="white_queen",
                    from_square="a7",
                    to_square="b8",
                    promotion_piece_type="queen",
                    metadata={"promotion_class_id": "baseline_queen"},
                )
            ),
        )

    def test_run_mcts_returns_terminal_root_without_moves(self) -> None:
        state = GameState(
            piece_classes={
                self.king.class_id: self.king,
                self.queen.class_id: self.queen,
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=self.king.class_id,
                    color=Color.WHITE,
                    square="h1",
                ),
                "black_queen": PieceInstance(
                    instance_id="black_queen",
                    piece_class_id=self.queen.class_id,
                    color=Color.BLACK,
                    square="g2",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="f3",
                ),
            },
            side_to_move=Color.WHITE,
        )

        result = run_mcts(state, simulations=4)

        self.assertIsNone(result.selected_move)
        self.assertIsNone(result.selected_move_id)
        self.assertEqual((), result.legal_moves)
        self.assertEqual({}, result.visit_distribution)
        self.assertEqual({}, result.visit_counts)
        self.assertEqual(-1.0, result.root_value)

    def test_run_mcts_selects_legal_mating_capture(self) -> None:
        state = GameState(
            piece_classes={
                self.king.class_id: self.king,
                self.rook.class_id: self.rook,
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=self.king.class_id,
                    color=Color.WHITE,
                    square="a1",
                ),
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=self.rook.class_id,
                    color=Color.WHITE,
                    square="e7",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="e8",
                ),
            },
            side_to_move=Color.WHITE,
        )

        search_result = run_mcts(state, simulations=50, c_puct=3.0)

        self.assertIsNotNone(search_result.selected_move)
        self.assertEqual("e8", search_result.selected_move.to_square)
        self.assertEqual("capture", search_result.selected_move.move_kind)
        self.assertEqual(
            stable_move_id(search_result.selected_move),
            search_result.selected_move_id,
        )
        self.assertIn(search_result.selected_move, search_result.legal_moves)
        self.assertAlmostEqual(1.0, sum(search_result.visit_distribution.values()))

    def test_model_priors_are_respected_when_visits_are_tied(self) -> None:
        state = GameState(
            piece_classes={
                self.king.class_id: self.king,
                self.rook.class_id: self.rook,
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=self.king.class_id,
                    color=Color.WHITE,
                    square="e1",
                ),
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=self.rook.class_id,
                    color=Color.WHITE,
                    square="a1",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="e8",
                ),
            },
            side_to_move=Color.WHITE,
        )
        preferred_move = next(
            move for move in legal_moves(state) if move.piece_id == "white_rook" and move.to_square == "a8"
        )
        preferred_move_id = stable_move_id(preferred_move)

        result = run_mcts(
            state,
            simulations=1,
            policy_value_model=_BiasedPolicyValueModel(preferred_move_id),
        )

        self.assertEqual(preferred_move_id, result.selected_move_id)
        self.assertEqual("a8", result.selected_move.to_square)
        self.assertIn(preferred_move_id, result.policy_logits)
        self.assertAlmostEqual(1.0, sum(result.visit_distribution.values()))
        self.assertEqual(
            {stable_move_id(move) for move in result.legal_moves},
            set(result.visit_distribution),
        )

    def test_root_dirichlet_noise_mixes_root_priors_deterministically(self) -> None:
        state = GameState(
            piece_classes={
                self.king.class_id: self.king,
                self.rook.class_id: self.rook,
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=self.king.class_id,
                    color=Color.WHITE,
                    square="e1",
                ),
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=self.rook.class_id,
                    color=Color.WHITE,
                    square="a1",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="e8",
                ),
            },
            side_to_move=Color.WHITE,
        )

        first = run_mcts(
            state,
            simulations=1,
            root_noise=DirichletRootNoise(alpha=0.3, exploration_fraction=1.0),
            rng=random.Random(123),
        )
        second = run_mcts(
            state,
            simulations=1,
            root_noise=DirichletRootNoise(alpha=0.3, exploration_fraction=1.0),
            rng=random.Random(123),
        )
        without_noise = run_mcts(state, simulations=1)

        self.assertTrue(first.metadata["root_noise_applied"])
        self.assertEqual(first.visit_distribution, second.visit_distribution)
        self.assertNotEqual(without_noise.visit_distribution, first.visit_distribution)
        self.assertAlmostEqual(1.0, sum(first.metadata["root_noise"].values()))
        self.assertEqual(
            first.metadata["root_noise"],
            first.metadata["root_priors_after_noise"],
        )


if __name__ == "__main__":
    unittest.main()
