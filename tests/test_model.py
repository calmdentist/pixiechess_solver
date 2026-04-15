from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import (
    BasePieceType,
    Color,
    Condition,
    Effect,
    Hook,
    Modifier,
    PieceClass,
    PieceInstance,
    StateField,
    stable_move_id,
)
from pixie_solver.core.state import GameState
from pixie_solver.model import (
    BoardEncoder,
    DSLFeatureEncoder,
    PolicyValueConfig,
    PolicyValueModel,
)
from pixie_solver.simulator.movegen import legal_moves
from pixie_solver.training import (
    SelfPlayConfig,
    TrainingConfig,
    flatten_selfplay_examples,
    generate_selfplay_games,
    load_training_checkpoint,
    save_training_checkpoint,
    train_from_replays,
)


class CountingPolicyValueModel(PolicyValueModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.forward_calls = 0
        self.forward_batch_calls = 0

    def forward(self, state, legal_moves):
        self.forward_calls += 1
        return super().forward(state, legal_moves)

    def forward_batch(self, requests):
        self.forward_batch_calls += 1
        return super().forward_batch(requests)


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.king = PieceClass(
            class_id="baseline_king",
            name="Baseline King",
            base_piece_type=BasePieceType.KING,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
        )
        self.rook = PieceClass(
            class_id="sumo_rook",
            name="Sumo Rook",
            base_piece_type=BasePieceType.ROOK,
            movement_modifiers=(
                Modifier(op="inherit_base"),
                Modifier(op="phase_through_allies"),
            ),
            capture_modifiers=(
                Modifier(
                    op="replace_capture_with_push",
                    args={"distance": 1, "edge_behavior": "fail"},
                ),
            ),
            hooks=(
                Hook(
                    event="turn_start",
                    effects=(
                        Effect(
                            op="set_state",
                            args={"piece": "self", "name": "charge", "value": 1},
                        ),
                    ),
                ),
            ),
            instance_state_schema=(
                StateField(name="charge", field_type="int", default=0),
            ),
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
                    state={"charge": 2},
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="e8",
                ),
            },
            side_to_move=Color.WHITE,
            castling_rights={"white": ("king",), "black": ("queen",)},
            fullmove_number=7,
        )

    def test_dsl_encoder_is_deterministic(self) -> None:
        encoder = DSLFeatureEncoder(d_model=32)
        first = encoder.encode_piece_class(self.rook)
        second = encoder.encode_piece_class(self.rook)

        self.assertEqual((32,), tuple(first.shape))
        self.assertTrue(first.dtype.is_floating_point)
        self.assertTrue((first == second).all().item())

    def test_board_encoder_returns_piece_and_global_tokens(self) -> None:
        board_encoder = BoardEncoder(d_model=32, dsl_encoder=DSLFeatureEncoder(d_model=32))
        encoded = board_encoder.encode_state(self.state)

        self.assertEqual(("black_king", "white_king", "white_rook"), encoded.piece_ids)
        self.assertEqual((3, 32), tuple(encoded.piece_tokens.shape))
        self.assertEqual((32,), tuple(encoded.global_token.shape))
        self.assertEqual(2, encoded.piece_index_by_id["white_rook"])

    def test_policy_value_infer_scores_legal_candidates(self) -> None:
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
        moves = tuple(legal_moves(self.state))

        forward_output = model(self.state, moves)
        infer_output = model.infer(self.state, moves)

        self.assertEqual(len(moves), len(forward_output.move_ids))
        self.assertEqual((len(moves),), tuple(forward_output.policy_logits.shape))
        self.assertEqual(
            {stable_move_id(move) for move in moves},
            set(infer_output.policy_logits),
        )
        self.assertGreaterEqual(infer_output.value, -1.0)
        self.assertLessEqual(infer_output.value, 1.0)

    def test_train_from_replays_smoke(self) -> None:
        games = generate_selfplay_games(
            [self.state],
            games=1,
            config=SelfPlayConfig(
                simulations=2,
                max_plies=4,
                opening_temperature=0.0,
                final_temperature=0.0,
                seed=7,
            ),
        )
        examples = flatten_selfplay_examples(games)

        run_result = train_from_replays(
            examples,
            config=TrainingConfig(
                epochs=1,
                batch_size=2,
                device="cpu",
                shuffle=False,
                model_config=PolicyValueConfig(
                    d_model=32,
                    num_heads=4,
                    num_layers=1,
                    dropout=0.0,
                    feedforward_multiplier=2,
                ),
            ),
        )

        self.assertEqual(len(examples), run_result.metrics.examples_seen)
        self.assertGreater(run_result.metrics.batches_completed, 0)
        self.assertGreaterEqual(run_result.metrics.average_policy_loss, 0.0)
        self.assertGreaterEqual(run_result.metrics.average_value_loss, 0.0)
        self.assertGreaterEqual(run_result.metrics.average_total_loss, 0.0)

    def test_train_from_replays_uses_batched_forward_path(self) -> None:
        games = generate_selfplay_games(
            [self.state],
            games=1,
            config=SelfPlayConfig(
                simulations=2,
                max_plies=4,
                opening_temperature=0.0,
                final_temperature=0.0,
                seed=11,
            ),
        )
        examples = flatten_selfplay_examples(games)
        model = CountingPolicyValueModel(
            PolicyValueConfig(
                d_model=32,
                num_heads=4,
                num_layers=1,
                dropout=0.0,
                feedforward_multiplier=2,
            ),
            device="cpu",
        )

        run_result = train_from_replays(
            examples,
            model=model,
            config=TrainingConfig(
                epochs=1,
                batch_size=2,
                device="cpu",
                shuffle=False,
                model_config=model.config,
            ),
        )

        self.assertEqual(len(examples), run_result.metrics.examples_seen)
        self.assertGreater(model.forward_batch_calls, 0)
        self.assertEqual(0, model.forward_calls)

    def test_training_checkpoint_round_trip(self) -> None:
        games = generate_selfplay_games(
            [self.state],
            games=1,
            config=SelfPlayConfig(
                simulations=2,
                max_plies=4,
                opening_temperature=0.0,
                final_temperature=0.0,
                seed=5,
            ),
        )
        examples = flatten_selfplay_examples(games)
        run_result = train_from_replays(
            examples,
            config=TrainingConfig(
                epochs=1,
                batch_size=1,
                device="cpu",
                shuffle=False,
                seed=5,
                model_config=PolicyValueConfig(
                    d_model=32,
                    num_heads=4,
                    num_layers=1,
                    dropout=0.0,
                    feedforward_multiplier=2,
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            save_training_checkpoint(
                checkpoint_path,
                model=run_result.model,
                training_config=TrainingConfig(
                    epochs=1,
                    batch_size=1,
                    device="cpu",
                    shuffle=False,
                    seed=5,
                    model_config=PolicyValueConfig(
                        d_model=32,
                        num_heads=4,
                        num_layers=1,
                        dropout=0.0,
                        feedforward_multiplier=2,
                    ),
                ),
                training_metrics=run_result.metrics,
                optimizer_state_dict=run_result.optimizer_state_dict,
                metadata={"source": "unit_test"},
            )
            loaded = load_training_checkpoint(checkpoint_path, device="cpu")

        self.assertEqual(32, loaded.model_config.d_model)
        self.assertEqual("unit_test", loaded.metadata["source"])
        self.assertIsNotNone(loaded.training_metrics)
        self.assertEqual(
            run_result.metrics.examples_seen,
            loaded.training_metrics.examples_seen,
        )
        self.assertIsNotNone(loaded.optimizer_state_dict)
        original_state = run_result.model.state_dict()
        loaded_state = loaded.model.state_dict()
        self.assertEqual(set(original_state), set(loaded_state))
        self.assertTrue(
            all(
                original_state[name].equal(loaded_state[name])
                for name in original_state
            )
        )


if __name__ == "__main__":
    unittest.main()
