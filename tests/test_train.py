from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import stable_move_id, standard_initial_state
from pixie_solver.core.state import GameState
from pixie_solver.model import PolicyValueConfig, PolicyValueModel
from pixie_solver.model import PolicyValueForwardOutput
from pixie_solver.simulator.movegen import legal_moves
from pixie_solver.training.dataset import SelfPlayExample
from pixie_solver.training.train import (
    BUCKET_BALANCED_REPLAY_SAMPLING_STRATEGY,
    COMPOSITION_REPLAY_BUCKET,
    FOUNDATION_REPLAY_BUCKET,
    KNOWN_MECHANIC_REPLAY_BUCKET,
    RECENT_REPLAY_BUCKET,
    TrainingConfig,
    _build_replay_sampler,
    replay_bucket_for_example,
    summarize_replay_buckets,
    train_from_replays,
)


class StrategyRecordingPolicyValueModel(PolicyValueModel):
    def __init__(self) -> None:
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
        self.seen_strategies: list[str | None] = []

    def forward_batch(self, requests, *, strategy=None):
        strategy_id = None
        if isinstance(strategy, dict):
            strategy_id = str(strategy.get("strategy_id"))
        self.seen_strategies.append(strategy_id)
        return super().forward_batch(requests, strategy=strategy)


class FixedValuePolicyValueModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.uncertainty_bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward_batch(self, requests, *, strategy=None):
        del strategy
        outputs = []
        for _, moves in requests:
            outputs.append(
                PolicyValueForwardOutput(
                    move_ids=tuple(stable_move_id(move) for move in moves),
                    policy_logits=torch.zeros(
                        len(moves),
                        dtype=torch.float32,
                        device=self.bias.device,
                    ),
                    value=self.bias.squeeze(),
                    uncertainty=torch.sigmoid(self.uncertainty_bias),
                )
            )
        return tuple(outputs)


class TrainingCurriculumTest(unittest.TestCase):
    def test_replay_bucket_for_example_distinguishes_recent_known_composition_and_foundation(self) -> None:
        recent_example = _example(cycle=5)
        verified_example = _example(cycle=3, verified_piece_id="verified_piece")
        composition_example = _example(
            cycle=4,
            verified_piece_id="composition_a",
            metadata={
                "family_id": "composition",
                "split": "mixed",
                "novelty_tier": "composition",
                "world_family_ids": ["capture_sprint", "phase_rook"],
            },
        )
        foundation_example = _example(cycle=2)

        self.assertEqual(
            RECENT_REPLAY_BUCKET,
            replay_bucket_for_example(
                recent_example,
                reference_cycle=5,
                recent_cycle_window=1,
            ),
        )
        self.assertEqual(
            KNOWN_MECHANIC_REPLAY_BUCKET,
            replay_bucket_for_example(
                verified_example,
                reference_cycle=5,
                recent_cycle_window=1,
            ),
        )
        self.assertEqual(
            COMPOSITION_REPLAY_BUCKET,
            replay_bucket_for_example(
                composition_example,
                reference_cycle=5,
                recent_cycle_window=1,
            ),
        )
        self.assertEqual(
            FOUNDATION_REPLAY_BUCKET,
            replay_bucket_for_example(
                foundation_example,
                reference_cycle=5,
                recent_cycle_window=1,
            ),
        )

    def test_summarize_replay_buckets_respects_reference_cycle(self) -> None:
        config = TrainingConfig(
            sampling_strategy=BUCKET_BALANCED_REPLAY_SAMPLING_STRATEGY,
            recent_cycle_window=1,
            sampling_reference_cycle=4,
        )
        counts = summarize_replay_buckets(
            (
                _example(cycle=1),
                _example(cycle=2, verified_piece_id="verified_piece"),
                _example(cycle=4, verified_piece_id="recent_verified_piece"),
            ),
            config=config,
        )

        self.assertEqual(
            {
                FOUNDATION_REPLAY_BUCKET: 1,
                KNOWN_MECHANIC_REPLAY_BUCKET: 1,
                RECENT_REPLAY_BUCKET: 1,
                COMPOSITION_REPLAY_BUCKET: 0,
            },
            counts,
        )

    def test_active_verified_pool_metadata_does_not_mark_example_as_verified(self) -> None:
        foundation_example = _example(
            cycle=2,
            active_verified_piece_id="verified_piece",
        )

        self.assertEqual(
            FOUNDATION_REPLAY_BUCKET,
            replay_bucket_for_example(
                foundation_example,
                reference_cycle=5,
                recent_cycle_window=1,
            ),
        )

    def test_bucket_balanced_sampler_uses_bucket_weights(self) -> None:
        config = TrainingConfig(
            sampling_strategy=BUCKET_BALANCED_REPLAY_SAMPLING_STRATEGY,
            recent_cycle_window=1,
            recent_bucket_weight=5.0,
            known_bucket_weight=3.0,
            composition_bucket_weight=7.0,
            foundation_bucket_weight=2.0,
            sampling_reference_cycle=4,
        )
        sampler = _build_replay_sampler(
            (
                _example(cycle=1),
                _example(cycle=2, verified_piece_id="verified_piece"),
                _example(
                    cycle=2,
                    verified_piece_id="composition_piece",
                    metadata={
                        "family_id": "composition",
                        "split": "mixed",
                        "novelty_tier": "composition",
                        "world_family_ids": ["capture_sprint", "phase_rook"],
                    },
                ),
                _example(cycle=4),
            ),
            config,
        )

        self.assertIsNotNone(sampler)
        self.assertEqual(
            [2.0, 3.0, 7.0, 5.0],
            [float(value) for value in sampler.weights.tolist()],
        )

    def test_train_from_replays_groups_batches_by_strategy(self) -> None:
        model = StrategyRecordingPolicyValueModel()
        state = standard_initial_state()
        moves = tuple(legal_moves(state))
        selected_move_id = stable_move_id(moves[0])
        examples = [
            SelfPlayExample(
                state=state,
                legal_moves=moves,
                legal_move_ids=tuple(stable_move_id(move) for move in moves),
                visit_distribution={selected_move_id: 1.0},
                visit_counts={selected_move_id: 1},
                selected_move_id=selected_move_id,
                outcome=1.0,
            ),
            SelfPlayExample(
                state=state,
                legal_moves=moves,
                legal_move_ids=tuple(stable_move_id(move) for move in moves),
                visit_distribution={selected_move_id: 1.0},
                visit_counts={selected_move_id: 1},
                selected_move_id=selected_move_id,
                outcome=1.0,
                metadata={
                    "strategy": {
                        "strategy_id": "aggressive_plan",
                        "summary": "attack quickly",
                        "confidence": 0.8,
                        "scope": "game_start",
                    }
                },
            ),
        ]

        result = train_from_replays(
            examples,
            model=model,
            config=TrainingConfig(
                epochs=1,
                batch_size=2,
                shuffle=False,
                device="cpu",
            ),
        )

        self.assertEqual(2, result.metrics.examples_seen)
        self.assertEqual([None, "aggressive_plan"], model.seen_strategies)

    def test_train_from_replays_can_blend_root_value_and_outcome_targets(self) -> None:
        state = standard_initial_state()
        moves = tuple(legal_moves(state))
        selected_move_id = stable_move_id(moves[0])
        example = SelfPlayExample(
            state=state,
            legal_moves=moves,
            legal_move_ids=tuple(stable_move_id(move) for move in moves),
            visit_distribution={selected_move_id: 1.0},
            visit_counts={selected_move_id: 1},
            selected_move_id=selected_move_id,
            root_value=0.75,
            outcome=-1.0,
        )

        root_only_result = train_from_replays(
            [example],
            model=FixedValuePolicyValueModel(),
            config=TrainingConfig(
                epochs=1,
                batch_size=1,
                shuffle=False,
                device="cpu",
                policy_weight=0.0,
                value_weight=1.0,
                root_value_target_weight=1.0,
                outcome_target_weight=0.0,
            ),
        )
        outcome_only_result = train_from_replays(
            [example],
            model=FixedValuePolicyValueModel(),
            config=TrainingConfig(
                epochs=1,
                batch_size=1,
                shuffle=False,
                device="cpu",
                policy_weight=0.0,
                value_weight=1.0,
                root_value_target_weight=0.0,
                outcome_target_weight=1.0,
            ),
        )
        blended_result = train_from_replays(
            [example],
            model=FixedValuePolicyValueModel(),
            config=TrainingConfig(
                epochs=1,
                batch_size=1,
                shuffle=False,
                device="cpu",
                policy_weight=0.0,
                value_weight=1.0,
                root_value_target_weight=1.0,
                outcome_target_weight=1.0,
            ),
        )

        self.assertAlmostEqual(0.75**2, root_only_result.metrics.average_value_loss)
        self.assertAlmostEqual(1.0, outcome_only_result.metrics.average_value_loss)
        self.assertAlmostEqual(0.125**2, blended_result.metrics.average_value_loss)

    def test_train_from_replays_can_train_uncertainty_from_root_outcome_disagreement(self) -> None:
        state = standard_initial_state()
        moves = tuple(legal_moves(state))
        selected_move_id = stable_move_id(moves[0])
        example = SelfPlayExample(
            state=state,
            legal_moves=moves,
            legal_move_ids=tuple(stable_move_id(move) for move in moves),
            visit_distribution={selected_move_id: 1.0},
            visit_counts={selected_move_id: 1},
            selected_move_id=selected_move_id,
            root_value=1.0,
            outcome=-1.0,
        )

        result = train_from_replays(
            [example],
            model=FixedValuePolicyValueModel(),
            config=TrainingConfig(
                epochs=1,
                batch_size=1,
                shuffle=False,
                device="cpu",
                policy_weight=0.0,
                value_weight=0.0,
                uncertainty_weight=1.0,
            ),
        )

        self.assertAlmostEqual(0.25, result.metrics.average_uncertainty_loss)


def _example(
    *,
    cycle: int,
    verified_piece_id: str | None = None,
    active_verified_piece_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> SelfPlayExample:
    payload: dict[str, object] = {"cycle": cycle, **dict(metadata or {})}
    if verified_piece_id is not None:
        payload["verified_piece_digests"] = {
            verified_piece_id: {
                "version": 1,
                "dsl_digest": f"digest:{verified_piece_id}",
                "source": "test",
            }
        }
    if active_verified_piece_id is not None:
        payload["active_verified_piece_digests"] = {
            active_verified_piece_id: {
                "version": 1,
                "dsl_digest": f"digest:{active_verified_piece_id}",
                "source": "test",
            }
        }
    return SelfPlayExample(
        state=GameState.empty(),
        metadata=payload,
    )


if __name__ == "__main__":
    unittest.main()
