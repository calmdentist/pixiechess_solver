from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Color, PieceClass, PieceInstance
from pixie_solver.core.state import GameState
from pixie_solver.model import BoardEncoder, DSLFeatureEncoder


class BoardContextEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.king = PieceClass(
            class_id="baseline_king",
            name="Baseline King",
            base_piece_type=BasePieceType.KING,
        )
        self.rook = PieceClass(
            class_id="sumo_rook",
            name="Sumo Rook",
            base_piece_type=BasePieceType.ROOK,
        )
        self.state = GameState(
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

    def test_board_encoder_returns_context_tokens_and_mappings(self) -> None:
        encoder = BoardEncoder(d_model=32, dsl_encoder=DSLFeatureEncoder(d_model=32))

        encoded = encoder.encode_state(self.state)

        self.assertEqual(("baseline_king", "sumo_rook"), encoded.program_ids)
        self.assertEqual(2, len(encoded.program_token_specs))
        self.assertEqual(32, encoded.program_tokens.shape[1])
        self.assertEqual(len(encoded.probe_specs), encoded.probe_tokens.shape[0])
        self.assertIn("capture_control:white", encoded.context_probe_index_by_id)
        self.assertIn("capture_control:black", encoded.context_probe_index_by_id)
        self.assertIn("class_summary:baseline_king", encoded.context_probe_index_by_id)
        self.assertIn("class_summary:sumo_rook", encoded.context_probe_index_by_id)
        self.assertEqual(0, encoded.context_global_index)
        self.assertEqual(3, encoded.context_piece_index_by_id["white_rook"])
        self.assertEqual(4, encoded.context_program_token_span_by_class_id["baseline_king"][0])
        self.assertEqual(
            1 + len(encoded.piece_ids) + encoded.program_tokens.shape[0] + encoded.probe_tokens.shape[0],
            encoded.context_tokens.shape[0],
        )
        self.assertEqual(
            encoded.context_tokens.shape[0],
            encoded.context_token_type_ids.shape[0],
        )
        self.assertEqual([0, 1, 1, 1], encoded.context_token_type_ids[:4].tolist())
        self.assertTrue(all(token_type in {0, 1, 2, 3} for token_type in encoded.context_token_type_ids.tolist()))


if __name__ == "__main__":
    unittest.main()
