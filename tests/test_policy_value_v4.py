from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

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
)
from pixie_solver.core.state import GameState
from pixie_solver.model import (
    HYPERNETWORK_MODEL_ARCHITECTURE,
    PolicyValueConfig,
    PolicyValueModelV4,
)
from pixie_solver.simulator.movegen import legal_moves
from pixie_solver.strategy import StrategyHypothesis
from pixie_solver.training import load_training_checkpoint, save_training_checkpoint


class PolicyValueV4Test(unittest.TestCase):
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
            architecture=HYPERNETWORK_MODEL_ARCHITECTURE,
            d_model=32,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            feedforward_multiplier=2,
        )

    def test_v4_infer_scores_legal_candidates(self) -> None:
        model = PolicyValueModelV4(self.config, device="cpu")
        moves = tuple(legal_moves(self.state))

        forward_output = model(self.state, moves)
        infer_output, metrics = model.infer_batch_with_metrics(((self.state, moves),))

        self.assertEqual(tuple(stable_move_id(move) for move in moves), forward_output.move_ids)
        self.assertEqual((len(moves),), tuple(forward_output.policy_logits.shape))
        self.assertGreaterEqual(infer_output[0].value, -1.0)
        self.assertLessEqual(infer_output[0].value, 1.0)
        self.assertGreaterEqual(float(forward_output.uncertainty.detach().cpu().item()), 0.0)
        self.assertLessEqual(float(forward_output.uncertainty.detach().cpu().item()), 1.0)
        self.assertGreaterEqual(infer_output[0].uncertainty, 0.0)
        self.assertLessEqual(infer_output[0].uncertainty, 1.0)
        self.assertGreaterEqual(metrics.board_encode_ms, 0.0)

    def test_v4_compiler_can_change_executor_outputs(self) -> None:
        model = PolicyValueModelV4(self.config, device="cpu")
        moves = tuple(legal_moves(self.state))
        baseline = model(self.state, moves)

        with torch.no_grad():
            model.world_compiler.scale_head.bias.fill_(10.0)
            model.world_compiler.shift_head.bias.fill_(10.0)
            model.world_compiler.gate_head.bias.fill_(10.0)
        model.clear_adapter_cache()
        adapted = model(self.state, moves)

        self.assertFalse(torch.allclose(baseline.policy_logits, adapted.policy_logits))
        self.assertFalse(torch.allclose(baseline.value, adapted.value))

    def test_v4_checkpoint_round_trip(self) -> None:
        model = PolicyValueModelV4(self.config, device="cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "hypernetwork_conditioned.pt"
            save_training_checkpoint(checkpoint_path, model=model)
            loaded = load_training_checkpoint(checkpoint_path, device="cpu")

        self.assertEqual(
            HYPERNETWORK_MODEL_ARCHITECTURE,
            loaded.model_config.architecture,
        )
        self.assertIsInstance(loaded.model, PolicyValueModelV4)

    def test_v4_checkpoint_loader_accepts_legacy_shim_payload(self) -> None:
        model = PolicyValueModelV4(self.config, device="cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "legacy_hypernetwork_conditioned.pt"
            save_training_checkpoint(checkpoint_path, model=model)
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            payload["model_state_dict"] = {
                key: value
                for key, value in dict(payload["model_state_dict"]).items()
                if not key.startswith("world_compiler.")
            }
            torch.save(payload, checkpoint_path)
            loaded = load_training_checkpoint(checkpoint_path, device="cpu")

        self.assertEqual(
            HYPERNETWORK_MODEL_ARCHITECTURE,
            loaded.model_config.architecture,
        )
        self.assertIsInstance(loaded.model, PolicyValueModelV4)

    def test_v4_reuses_adapter_cache_for_repeated_worlds(self) -> None:
        model = PolicyValueModelV4(self.config, device="cpu")
        compile_calls = 0
        original_compile = model.world_compiler.compile_from_state_with_metrics

        def counting_compile(*args, **kwargs):
            nonlocal compile_calls
            compile_calls += 1
            return original_compile(*args, **kwargs)

        model.world_compiler.compile_from_state_with_metrics = counting_compile
        moves = tuple(legal_moves(self.state))

        model(self.state, moves)
        model(self.state, moves)

        self.assertEqual(1, compile_calls)
        stats = model.adapter_cache_stats()
        self.assertEqual(1, stats.hits)
        self.assertEqual(1, stats.misses)

    def test_v4_strategy_digest_invalidates_compile_cache(self) -> None:
        model = PolicyValueModelV4(self.config, device="cpu")
        compile_calls = 0
        original_compile = model.world_compiler.compile_from_state_with_metrics

        def counting_compile(*args, **kwargs):
            nonlocal compile_calls
            compile_calls += 1
            return original_compile(*args, **kwargs)

        model.world_compiler.compile_from_state_with_metrics = counting_compile
        moves = tuple(legal_moves(self.state))

        model.set_active_strategy_digest("strategy_a")
        model(self.state, moves)
        model(self.state, moves)
        model.set_active_strategy_digest("strategy_b")
        model(self.state, moves)

        self.assertEqual(2, compile_calls)
        stats = model.adapter_cache_stats()
        self.assertEqual(1, stats.hits)
        self.assertEqual(2, stats.misses)

    def test_v4_clear_adapter_cache_forces_recompile(self) -> None:
        model = PolicyValueModelV4(self.config, device="cpu")
        compile_calls = 0
        original_compile = model.world_compiler.compile_from_state_with_metrics

        def counting_compile(*args, **kwargs):
            nonlocal compile_calls
            compile_calls += 1
            return original_compile(*args, **kwargs)

        model.world_compiler.compile_from_state_with_metrics = counting_compile

        model.prepare_adapter_bundle_for_state(self.state)
        model.clear_adapter_cache()
        model.prepare_adapter_bundle_for_state(self.state)

        self.assertEqual(2, compile_calls)

    def test_v4_structured_strategy_changes_bundle_digest(self) -> None:
        model = PolicyValueModelV4(self.config, device="cpu")
        aggressive = StrategyHypothesis(
            strategy_id="aggressive",
            summary="activate the rook aggressively",
            confidence=0.8,
            action_biases=("open file",),
        )
        cautious = StrategyHypothesis(
            strategy_id="cautious",
            summary="avoid exposing the rook",
            confidence=0.8,
            avoid_biases=("open file",),
        )

        aggressive_bundle = model.compile_adapter_bundle_for_state(
            self.state,
            strategy=aggressive,
        )
        cautious_bundle = model.compile_adapter_bundle_for_state(
            self.state,
            strategy=cautious,
        )

        self.assertNotEqual(
            aggressive_bundle.strategy_digest,
            cautious_bundle.strategy_digest,
        )
        self.assertNotEqual(
            aggressive_bundle.bundle_id,
            cautious_bundle.bundle_id,
        )


if __name__ == "__main__":
    unittest.main()
