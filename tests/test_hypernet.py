from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Color, PieceClass, PieceInstance
from pixie_solver.core.state import GameState
from pixie_solver.hypernet import (
    AdapterBundleCache,
    AdapterCacheStats,
    AdapterBundle,
    AdapterBundleValidationError,
    AttentionBias,
    DEFAULT_ADAPTER_TARGET_LAYERS,
    GatingValues,
    LayerModulation,
    WorldCompilerHypernetwork,
    adapter_bundle_digest,
    apply_layer_adapter,
    canonicalize_adapter_bundle,
    prepare_adapter_bundle,
    validate_adapter_bundle,
)
from pixie_solver.strategy import StrategyHypothesis


class HypernetTest(unittest.TestCase):
    def setUp(self) -> None:
        king = PieceClass(
            class_id="baseline_king",
            name="Baseline King",
            base_piece_type=BasePieceType.KING,
        )
        rook = PieceClass(
            class_id="sumo_rook",
            name="Sumo Rook",
            base_piece_type=BasePieceType.ROOK,
        )
        self.state = GameState(
            piece_classes={
                king.class_id: king,
                rook.class_id: rook,
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=king.class_id,
                    color=Color.WHITE,
                    square="e1",
                ),
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=rook.class_id,
                    color=Color.WHITE,
                    square="a1",
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

    def test_adapter_bundle_canonicalization_is_deterministic(self) -> None:
        bundle = AdapterBundle(
            bundle_id="bundle_alpha",
            world_digest="world123",
            strategy_digest="strategy456",
            layer_modulations=(
                LayerModulation(
                    layer_name="context_block_1",
                    scale=(1.0, -0.0, 0.333333333333),
                    shift=(0.25, -0.25, 0.0),
                ),
                LayerModulation(
                    layer_name="action_block_0",
                    scale=(0.5,),
                    shift=(0.1,),
                ),
            ),
            attention_biases=(
                AttentionBias(layer_name="context_attn", values=(0.1, 0.2)),
            ),
            gating_values=(
                GatingValues(layer_name="action_gate", values=(0.9,)),
            ),
            metadata={"novelty_tier": "tier_2", "admission_cycle": 3},
        )

        first = canonicalize_adapter_bundle(bundle)
        second = canonicalize_adapter_bundle(bundle)

        self.assertEqual(first, second)
        self.assertEqual(
            ["action_block_0", "context_block_1"],
            [item["layer_name"] for item in first["layer_modulations"]],
        )
        self.assertEqual(0.0, first["layer_modulations"][1]["scale"][1])

    def test_adapter_bundle_digest_ignores_component_order(self) -> None:
        first = AdapterBundle(
            bundle_id="bundle_beta",
            world_digest="world123",
            strategy_digest="strategy456",
            layer_modulations=(
                LayerModulation(layer_name="b", scale=(1.0,), shift=(2.0,)),
                LayerModulation(layer_name="a", scale=(3.0,), shift=(4.0,)),
            ),
        )
        second = AdapterBundle(
            bundle_id="bundle_beta",
            world_digest="world123",
            strategy_digest="strategy456",
            layer_modulations=(
                LayerModulation(layer_name="a", scale=(3.0,), shift=(4.0,)),
                LayerModulation(layer_name="b", scale=(1.0,), shift=(2.0,)),
            ),
        )

        self.assertEqual(adapter_bundle_digest(first), adapter_bundle_digest(second))

    def test_validate_adapter_bundle_rejects_duplicate_layers(self) -> None:
        bundle = AdapterBundle(
            bundle_id="bundle_gamma",
            world_digest="world123",
            layer_modulations=(
                LayerModulation(layer_name="shared", scale=(1.0,), shift=(0.0,)),
                LayerModulation(layer_name="shared", scale=(0.5,), shift=(0.1,)),
            ),
        )

        with self.assertRaisesRegex(
            AdapterBundleValidationError,
            "layer_modulations layer_name values must be unique",
        ):
            validate_adapter_bundle(bundle)

    def test_validate_adapter_bundle_rejects_mismatched_modulation_shapes(self) -> None:
        bundle = AdapterBundle(
            bundle_id="bundle_delta",
            world_digest="world123",
            layer_modulations=(
                LayerModulation(layer_name="context", scale=(1.0, 2.0), shift=(0.1,)),
            ),
        )

        with self.assertRaisesRegex(
            AdapterBundleValidationError,
            "scale and shift lengths must match",
        ):
            validate_adapter_bundle(bundle)

    def test_prepare_and_apply_layer_adapter_modulates_tokens(self) -> None:
        bundle = AdapterBundle(
            bundle_id="bundle_eps",
            world_digest="world123",
            layer_modulations=(
                LayerModulation(
                    layer_name="context_input",
                    scale=(2.0, 0.5),
                    shift=(1.0, -1.0),
                ),
            ),
            gating_values=(
                GatingValues(layer_name="context_input", values=(0.5, 2.0)),
            ),
        )
        prepared = prepare_adapter_bundle(
            bundle,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        tokens = torch.tensor([[1.0, 3.0]], dtype=torch.float32)

        modulated = apply_layer_adapter(
            tokens,
            prepared,
            layer_name="context_input",
        )

        self.assertTrue(
            torch.allclose(
                modulated,
                torch.tensor([[1.5, 1.0]], dtype=torch.float32),
            )
        )

    def test_world_compiler_is_deterministic(self) -> None:
        compiler = WorldCompilerHypernetwork(d_model=32, hidden_dim=64)

        first_bundle, first_metrics = compiler.compile_from_state_with_metrics(self.state)
        second_bundle, second_metrics = compiler.compile_from_state_with_metrics(self.state)

        self.assertEqual(first_bundle.world_digest, second_bundle.world_digest)
        self.assertEqual(first_bundle.bundle_id, second_bundle.bundle_id)
        self.assertEqual(
            adapter_bundle_digest(first_bundle),
            adapter_bundle_digest(second_bundle),
        )
        self.assertEqual(
            len(DEFAULT_ADAPTER_TARGET_LAYERS),
            len(first_bundle.layer_modulations),
        )
        self.assertEqual(2, first_metrics.program_count)
        self.assertEqual(2, second_metrics.program_count)

    def test_world_compiler_changes_bundle_for_different_strategies(self) -> None:
        compiler = WorldCompilerHypernetwork(d_model=32, hidden_dim=64)
        strategy_a = StrategyHypothesis(
            strategy_id="aggressive",
            summary="attack quickly",
            confidence=0.7,
            action_biases=("open file",),
        )
        strategy_b = StrategyHypothesis(
            strategy_id="defensive",
            summary="stabilize first",
            confidence=0.7,
            avoid_biases=("open file",),
        )

        bundle_a = compiler.compile_from_state(self.state, strategy=strategy_a)
        bundle_b = compiler.compile_from_state(self.state, strategy=strategy_b)

        self.assertNotEqual(bundle_a.bundle_id, bundle_b.bundle_id)
        self.assertNotEqual(bundle_a.strategy_digest, bundle_b.strategy_digest)

    def test_adapter_bundle_cache_tracks_hits_and_lru_eviction(self) -> None:
        cache = AdapterBundleCache[str](max_size=2)

        self.assertIsNone(cache.get(("world_a", None)))
        cache.put(("world_a", None), "bundle_a")
        cache.put(("world_b", None), "bundle_b")
        self.assertEqual("bundle_a", cache.get(("world_a", None)))
        cache.put(("world_c", None), "bundle_c")

        self.assertIsNone(cache.get(("world_b", None)))
        self.assertEqual("bundle_a", cache.get(("world_a", None)))
        self.assertEqual("bundle_c", cache.get(("world_c", None)))

        stats = cache.stats()
        self.assertIsInstance(stats, AdapterCacheStats)
        self.assertEqual(3, stats.hits)
        self.assertEqual(2, stats.misses)
        self.assertEqual(2, stats.size)
        self.assertEqual(2, stats.max_size)


if __name__ == "__main__":
    unittest.main()
