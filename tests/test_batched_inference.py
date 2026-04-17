from __future__ import annotations

import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import stable_move_id, standard_initial_state
from pixie_solver.model import (
    PolicyValueConfig,
    PolicyValueModel,
    build_policy_value_model,
)
from pixie_solver.simulator.movegen import legal_moves
from pixie_solver.training import (
    BatchedInferenceConfig,
    BatchedInferenceService,
    save_training_checkpoint,
)


class BatchedInferenceTest(unittest.TestCase):
    def test_model_infer_batch_matches_single_request_shape(self) -> None:
        state = standard_initial_state()
        moves = tuple(legal_moves(state))
        model = PolicyValueModel(
            PolicyValueConfig(
                d_model=32,
                num_heads=4,
                num_layers=1,
                dropout=0.0,
                feedforward_multiplier=2,
            ),
            device="cpu",
        )

        single = model.infer(state, moves)
        batched = model.infer_batch(((state, moves), (state, moves)))

        self.assertEqual(2, len(batched))
        self.assertEqual(set(single.policy_logits), set(batched[0].policy_logits))
        self.assertEqual(set(single.policy_logits), set(batched[1].policy_logits))
        self.assertGreaterEqual(single.value, -1.0)
        self.assertLessEqual(single.value, 1.0)

    def test_batched_inference_service_batches_concurrent_requests(self) -> None:
        state = standard_initial_state()
        moves = tuple(legal_moves(state))
        model = PolicyValueModel(
            PolicyValueConfig(
                d_model=32,
                num_heads=4,
                num_layers=1,
                dropout=0.0,
                feedforward_multiplier=2,
            ),
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            save_training_checkpoint(checkpoint_path, model=model)
            with BatchedInferenceService(
                checkpoint_path,
                device="cpu",
                config=BatchedInferenceConfig(max_batch_size=4, max_wait_ms=50.0),
            ) as service:
                client = service.client()
                with ThreadPoolExecutor(max_workers=2) as executor:
                    outputs = list(
                        executor.map(
                            lambda _: client.infer(state, moves),
                            range(2),
                        )
                    )
                stats = service.stats()

                self.assertEqual(2, len(outputs))
        self.assertEqual(set(outputs[0].policy_logits), set(outputs[1].policy_logits))
        self.assertEqual(2, stats.requests_completed)
        self.assertGreaterEqual(stats.batches_completed, 1)
        self.assertGreaterEqual(stats.max_batch_size_seen, 2)
        self.assertGreater(stats.total_legal_moves, 0)
        self.assertGreaterEqual(stats.model_total_ms, 0.0)
        self.assertGreaterEqual(stats.request_latency_ms_total, 0.0)
        self.assertGreaterEqual(stats.queue_wait_ms_total, 0.0)
        self.assertGreaterEqual(stats.to_dict()["average_batch_size"], 1.0)
        self.assertGreaterEqual(stats.to_dict()["requests_per_second"], 0.0)

    def test_batched_inference_service_supports_world_conditioned_v2_checkpoint(self) -> None:
        state = standard_initial_state()
        moves = tuple(legal_moves(state))
        model = build_policy_value_model(
            PolicyValueConfig(
                architecture="world_conditioned_v2",
                d_model=32,
                num_heads=4,
                num_layers=1,
                dropout=0.0,
                feedforward_multiplier=2,
            ),
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "world_conditioned.pt"
            save_training_checkpoint(checkpoint_path, model=model)
            with BatchedInferenceService(
                checkpoint_path,
                device="cpu",
                config=BatchedInferenceConfig(max_batch_size=2, max_wait_ms=20.0),
            ) as service:
                output = service.client().infer(state, moves)
                stats = service.stats()

        self.assertEqual(set(stable_move_id(move) for move in moves), set(output.policy_logits))
        self.assertGreaterEqual(output.value, -1.0)
        self.assertLessEqual(output.value, 1.0)
        self.assertEqual(1, stats.requests_completed)


if __name__ == "__main__":
    unittest.main()
