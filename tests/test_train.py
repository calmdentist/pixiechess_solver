from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core.state import GameState
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
)


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
