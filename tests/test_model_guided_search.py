from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Color, Modifier, Move, PieceClass, PieceInstance
from pixie_solver.core.state import GameState
from pixie_solver.eval import SearchComparisonConfig, compare_search_modes
from pixie_solver.model import PolicyValueConfig, PolicyValueModel, PolicyValueOutput
from pixie_solver.search import run_mcts
from pixie_solver.training import (
    BootstrapConfig,
    SelfPlayConfig,
    TrainingConfig,
    bootstrap_policy_value_model,
)


class _BiasedPolicyValueModel(PolicyValueModel):
    def __init__(self, preferred_move_id: str) -> None:
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
        self.preferred_move_id = preferred_move_id

    def infer(self, state: GameState, legal_moves: tuple[Move, ...]) -> PolicyValueOutput:
        return PolicyValueOutput(
            policy_logits={
                move.stable_id(): (
                    10.0 if move.stable_id() == self.preferred_move_id else -10.0
                )
                for move in legal_moves
            },
            value=0.75,
        )


class ModelGuidedSearchTest(unittest.TestCase):
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
        self.initial_state = GameState(
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

    def test_bootstrap_policy_value_model_generates_guided_selfplay(self) -> None:
        result = bootstrap_policy_value_model(
            [self.initial_state],
            config=BootstrapConfig(
                bootstrap_games=1,
                guided_games=1,
                selfplay_config=SelfPlayConfig(
                    simulations=4,
                    max_plies=4,
                    opening_temperature=0.0,
                    final_temperature=0.0,
                    temperature_drop_after_ply=0,
                    c_puct=2.0,
                    seed=11,
                ),
                training_config=TrainingConfig(
                    epochs=1,
                    batch_size=1,
                    shuffle=False,
                    device="cpu",
                    seed=11,
                    model_config=PolicyValueConfig(
                        d_model=32,
                        num_heads=4,
                        num_layers=1,
                        dropout=0.0,
                        feedforward_multiplier=2,
                    ),
                ),
            ),
        )

        self.assertEqual(1, len(result.bootstrap_games))
        self.assertEqual(1, len(result.guided_games))
        self.assertEqual(len(result.bootstrap_examples), result.training_metrics.examples_seen)
        self.assertTrue(result.bootstrap_examples)
        self.assertTrue(result.guided_examples)
        self.assertTrue(
            all(example.metadata.get("used_model") is False for example in result.bootstrap_examples)
        )
        self.assertTrue(
            all(example.metadata.get("used_model") is True for example in result.guided_examples)
        )

    def test_compare_search_modes_reports_guided_diagnostics(self) -> None:
        baseline = run_mcts(self.initial_state, simulations=24, c_puct=3.0)
        preferred_move_id = baseline.selected_move_id
        biased_model = _BiasedPolicyValueModel(preferred_move_id)

        summary = compare_search_modes(
            [self.initial_state],
            policy_value_model=biased_model,
            config=SearchComparisonConfig(
                baseline_simulations=24,
                guided_simulations=6,
                c_puct=3.0,
            ),
        )

        self.assertEqual(1, summary.positions)
        self.assertEqual(1.0, summary.move_agreement_rate)
        self.assertGreaterEqual(summary.average_abs_root_value_delta, 0.0)
        self.assertAlmostEqual(0.25, summary.metadata["guided_budget_ratio"])
        self.assertGreater(summary.metadata["average_baseline_expanded_nodes"], 0.0)
        self.assertGreater(summary.metadata["average_guided_expanded_nodes"], 0.0)
        self.assertGreater(summary.metadata["average_baseline_heuristic_evaluations"], 0.0)
        self.assertGreater(summary.metadata["average_guided_model_inference_calls"], 0.0)
        self.assertTrue(summary.cases[0].guided.metadata["used_model"])


if __name__ == "__main__":
    unittest.main()
