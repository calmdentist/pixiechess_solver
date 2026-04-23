from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import (
    BasePieceType,
    Color,
    PieceClass,
    PieceInstance,
    stable_move_id,
    standard_initial_state,
)
from pixie_solver.core.state import GameState
from pixie_solver.model import (
    PolicyValueConfig,
    PolicyValueModelV2,
    WORLD_CONDITIONED_MODEL_ARCHITECTURE,
)
from pixie_solver.simulator.movegen import legal_moves
from pixie_solver.training import load_training_checkpoint, save_training_checkpoint


class PolicyValueV2Test(unittest.TestCase):
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
        self.config = PolicyValueConfig(
            architecture=WORLD_CONDITIONED_MODEL_ARCHITECTURE,
            d_model=32,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            feedforward_multiplier=2,
        )

    def test_v2_infer_scores_legal_candidates(self) -> None:
        model = PolicyValueModelV2(self.config, device="cpu")
        moves = tuple(legal_moves(self.state))

        forward_output = model(self.state, moves)
        infer_output, metrics = model.infer_batch_with_metrics(((self.state, moves),))

        self.assertEqual(tuple(stable_move_id(move) for move in moves), forward_output.move_ids)
        self.assertEqual((len(moves),), tuple(forward_output.policy_logits.shape))
        self.assertEqual(0.0, float(forward_output.uncertainty.detach().cpu().item()))
        self.assertGreaterEqual(infer_output[0].value, -1.0)
        self.assertLessEqual(infer_output[0].value, 1.0)
        self.assertEqual(0.0, infer_output[0].uncertainty)
        self.assertEqual(0.0, metrics.consequence_total_ms)
        self.assertGreaterEqual(metrics.move_encode_ms, 0.0)

    def test_v2_forward_batch_supports_variable_move_counts(self) -> None:
        model = PolicyValueModelV2(self.config, device="cpu")
        opening_state = standard_initial_state()
        opening_moves = tuple(legal_moves(opening_state))
        sparse_moves = tuple(legal_moves(self.state))

        outputs = model.forward_batch(
            (
                (opening_state, opening_moves),
                (self.state, sparse_moves),
            )
        )

        self.assertEqual(len(opening_moves), len(outputs[0].move_ids))
        self.assertEqual(len(sparse_moves), len(outputs[1].move_ids))

    def test_v2_checkpoint_round_trip(self) -> None:
        model = PolicyValueModelV2(self.config, device="cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "world_conditioned.pt"
            save_training_checkpoint(checkpoint_path, model=model)
            loaded = load_training_checkpoint(checkpoint_path, device="cpu")

        self.assertEqual(
            WORLD_CONDITIONED_MODEL_ARCHITECTURE,
            loaded.model_config.architecture,
        )
        self.assertIsInstance(loaded.model, PolicyValueModelV2)


if __name__ == "__main__":
    unittest.main()
