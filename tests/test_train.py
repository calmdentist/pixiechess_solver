from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import stable_move_id, standard_initial_state
from pixie_solver.core.state import GameState
from pixie_solver.model import PolicyValueConfig, PolicyValueModel
from pixie_solver.simulator.movegen import legal_moves
from pixie_solver.training.dataset import SelfPlayExample
from pixie_solver.training.train import (
    BUCKET_BALANCED_REPLAY_SAMPLING_STRATEGY,
    FOUNDATION_REPLAY_BUCKET,
    RECENT_REPLAY_BUCKET,
    VERIFIED_REPLAY_BUCKET,
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


class TrainingCurriculumTest(unittest.TestCase):
    def test_replay_bucket_for_example_distinguishes_recent_verified_and_foundation(self) -> None:
        recent_example = _example(cycle=5)
        verified_example = _example(cycle=3, verified_piece_id="verified_piece")
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
            VERIFIED_REPLAY_BUCKET,
            replay_bucket_for_example(
                verified_example,
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
                VERIFIED_REPLAY_BUCKET: 1,
                RECENT_REPLAY_BUCKET: 1,
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
            verified_bucket_weight=3.0,
            foundation_bucket_weight=2.0,
            sampling_reference_cycle=4,
        )
        sampler = _build_replay_sampler(
            (
                _example(cycle=1),
                _example(cycle=2, verified_piece_id="verified_piece"),
                _example(cycle=4),
            ),
            config,
        )

        self.assertIsNotNone(sampler)
        self.assertEqual(
            [2.0, 3.0, 5.0],
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


def _example(
    *,
    cycle: int,
    verified_piece_id: str | None = None,
    active_verified_piece_id: str | None = None,
) -> SelfPlayExample:
    metadata: dict[str, object] = {"cycle": cycle}
    if verified_piece_id is not None:
        metadata["verified_piece_digests"] = {
            verified_piece_id: {
                "version": 1,
                "dsl_digest": f"digest:{verified_piece_id}",
                "source": "test",
            }
        }
    if active_verified_piece_id is not None:
        metadata["active_verified_piece_digests"] = {
            active_verified_piece_id: {
                "version": 1,
                "dsl_digest": f"digest:{active_verified_piece_id}",
                "source": "test",
            }
        }
    return SelfPlayExample(
        state=GameState.empty(),
        metadata=metadata,
    )


if __name__ == "__main__":
    unittest.main()
