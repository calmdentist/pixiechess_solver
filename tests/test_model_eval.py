from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Color, Modifier, PieceClass, PieceInstance
from pixie_solver.core import stable_move_id
from pixie_solver.core.state import GameState
from pixie_solver.eval import evaluate_policy_value_model
from pixie_solver.model import PolicyValueForwardOutput
from pixie_solver.simulator.movegen import legal_moves
from pixie_solver.training import SelfPlayExample


class _StrategySensitiveEvalModel:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> "_StrategySensitiveEvalModel":
        self.training = False
        return self

    def train(self) -> "_StrategySensitiveEvalModel":
        self.training = True
        return self

    def __call__(self, state, moves, *, strategy=None) -> PolicyValueForwardOutput:
        move_ids = tuple(stable_move_id(move) for move in moves)
        preferred_index = 0 if strategy is not None else len(move_ids) - 1
        logits = torch.full((len(move_ids),), -1.0, dtype=torch.float32)
        logits[preferred_index] = 1.0
        return PolicyValueForwardOutput(
            move_ids=move_ids,
            policy_logits=logits,
            value=torch.tensor(0.0),
            uncertainty=torch.tensor(0.0),
        )


class ModelEvalTest(unittest.TestCase):
    def test_evaluate_policy_value_model_passes_strategy_metadata(self) -> None:
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
        state = GameState(
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
                    square="d4",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=king.class_id,
                    color=Color.BLACK,
                    square="h8",
                ),
            },
            side_to_move=Color.WHITE,
        )
        moves = tuple(legal_moves(state))
        self.assertGreaterEqual(len(moves), 2)
        example = SelfPlayExample(
            state=state,
            legal_moves=moves,
            legal_move_ids=tuple(stable_move_id(move) for move in moves),
            visit_distribution={stable_move_id(moves[0]): 1.0},
            outcome=0.0,
            metadata={
                "strategy": {
                    "strategy_id": "activate_rook",
                    "summary": "activate the rook quickly",
                    "confidence": 0.9,
                    "scope": "game_start",
                }
            },
        )

        metrics = evaluate_policy_value_model(
            model=_StrategySensitiveEvalModel(),
            examples=[example],
        )

        self.assertEqual(1.0, metrics.top1_agreement)
        self.assertGreaterEqual(metrics.average_predicted_uncertainty, 0.0)
        self.assertLessEqual(metrics.average_predicted_uncertainty, 1.0)


if __name__ == "__main__":
    unittest.main()
