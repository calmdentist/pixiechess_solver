from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Color, PieceClass, PieceInstance, standard_initial_state
from pixie_solver.core.move import Move, stable_move_id
from pixie_solver.core.state import GameState
from pixie_solver.model import ActionTokenEncoderV2, action_token_specs
from pixie_solver.simulator.movegen import legal_moves


class ActionEncoderV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.rook = PieceClass(
            class_id="sumo_rook",
            name="Sumo Rook",
            base_piece_type=BasePieceType.ROOK,
        )
        self.knight = PieceClass(
            class_id="baseline_knight",
            name="Baseline Knight",
            base_piece_type=BasePieceType.KNIGHT,
        )
        self.king = PieceClass(
            class_id="baseline_king",
            name="Baseline King",
            base_piece_type=BasePieceType.KING,
        )
        self.state = GameState(
            piece_classes={
                self.rook.class_id: self.rook,
                self.knight.class_id: self.knight,
                self.king.class_id: self.king,
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
                "black_knight": PieceInstance(
                    instance_id="black_knight",
                    piece_class_id=self.knight.class_id,
                    color=Color.BLACK,
                    square="a3",
                ),
            },
            side_to_move=Color.WHITE,
        )

    def test_action_token_specs_align_with_move_ids_and_fields(self) -> None:
        moves = (
            Move(
                piece_id="white_rook",
                from_square="a1",
                to_square="a2",
            ),
            Move(
                piece_id="white_rook",
                from_square="a1",
                to_square="a3",
                move_kind="push_capture",
                tags=("line_break", "aggressive"),
                metadata={
                    "target_piece_id": "black_knight",
                    "distance": 1,
                    "edge_behavior": "block",
                    "conditions": {"requires_charge": True},
                    "path": ["a2", "a3"],
                },
            ),
        )

        specs = action_token_specs(self.state, moves)

        self.assertEqual(tuple(stable_move_id(move) for move in moves), tuple(spec.move_id for spec in specs))
        self.assertEqual("sumo_rook", specs[0].actor_class_id)
        self.assertEqual("baseline_knight", specs[1].target_class_id)
        self.assertEqual(("aggressive", "line_break"), specs[1].tags)
        self.assertIn("params.conditions.requires_charge", {spec.path for spec in specs[1].param_specs})
        self.assertIn("params.path[1]", {spec.path for spec in specs[1].param_specs})

    def test_action_token_specs_are_deterministic(self) -> None:
        move = Move(
            piece_id="white_rook",
            from_square="a1",
            to_square="a3",
            move_kind="push_capture",
            metadata={"target_piece_id": "black_knight", "distance": 1},
        )

        first = action_token_specs(self.state, (move,))
        second = action_token_specs(self.state, (move,))

        self.assertEqual(first, second)

    def test_action_token_encoder_batches_without_simulator_rollouts(self) -> None:
        encoder = ActionTokenEncoderV2(d_model=32)
        moves = (
            Move(piece_id="white_rook", from_square="a1", to_square="a2"),
            Move(
                piece_id="white_rook",
                from_square="a1",
                to_square="a3",
                move_kind="push_capture",
                tags=("line_break", "aggressive"),
                metadata={"target_piece_id": "black_knight", "distance": 1},
            ),
        )

        with patch("pixie_solver.simulator.engine.apply_move", side_effect=AssertionError("apply_move should not run")):
            with patch("pixie_solver.simulator.movegen.is_in_check", side_effect=AssertionError("is_in_check should not run")):
                encoded, metrics = encoder.encode_moves_with_metrics(self.state, moves)

        self.assertEqual(tuple(stable_move_id(move) for move in moves), encoded.move_ids)
        self.assertEqual((2, 32), tuple(encoded.candidate_embeddings.shape))
        self.assertEqual(2, metrics.actions_encoded)
        self.assertEqual(2, metrics.total_tag_count)
        self.assertGreater(metrics.total_param_tokens, 0)

    def test_action_token_encoder_aligns_with_real_legal_moves(self) -> None:
        encoder = ActionTokenEncoderV2(d_model=32)
        state = standard_initial_state()
        moves = tuple(legal_moves(state))

        encoded = encoder.encode_moves(state, moves)

        self.assertEqual(tuple(stable_move_id(move) for move in moves), encoded.move_ids)
        self.assertEqual(len(moves), len(encoded.action_specs))
        self.assertEqual(len(moves), encoded.candidate_embeddings.shape[0])


if __name__ == "__main__":
    unittest.main()
