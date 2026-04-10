from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Color, Modifier, PieceClass, PieceInstance
from pixie_solver.core import stable_move_id
from pixie_solver.core.state import GameState
from pixie_solver.simulator.engine import result
from pixie_solver.training import (
    SelfPlayConfig,
    flatten_selfplay_examples,
    generate_selfplay_games,
    read_selfplay_examples_jsonl,
    read_selfplay_games_jsonl,
    write_selfplay_examples_jsonl,
    write_selfplay_games_jsonl,
)
from pixie_solver.utils import replay_trace


class SelfPlayTest(unittest.TestCase):
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
        self.config = SelfPlayConfig(
            simulations=50,
            max_plies=4,
            opening_temperature=0.0,
            final_temperature=0.0,
            temperature_drop_after_ply=0,
            c_puct=3.0,
            seed=17,
        )

    def test_generate_selfplay_games_is_reproducible(self) -> None:
        games_a = generate_selfplay_games([self.initial_state], games=2, config=self.config)
        games_b = generate_selfplay_games([self.initial_state], games=2, config=self.config)

        self.assertEqual(
            [game.to_dict() for game in games_a],
            [game.to_dict() for game in games_b],
        )

    def test_selfplay_game_contains_replay_and_labeled_examples(self) -> None:
        game = generate_selfplay_games([self.initial_state], games=1, config=self.config)[0]

        self.assertEqual("white", game.outcome)
        self.assertEqual(1, len(game.examples))
        self.assertEqual(1, len(game.replay_trace.steps))
        example = game.examples[0]
        chosen_move = game.replay_trace.steps[0].move
        self.assertEqual(stable_move_id(chosen_move), example.selected_move_id)
        self.assertEqual(1.0, example.outcome)

        final_state = replay_trace(game.replay_trace)
        self.assertEqual("white", result(final_state))
        self.assertEqual(game.final_state_hash, final_state.state_hash())

    def test_selfplay_jsonl_round_trip(self) -> None:
        games = generate_selfplay_games([self.initial_state], games=2, config=self.config)
        examples = flatten_selfplay_examples(games)

        with tempfile.TemporaryDirectory() as temp_dir:
            games_path = Path(temp_dir) / "games.jsonl"
            examples_path = Path(temp_dir) / "examples.jsonl"

            write_selfplay_games_jsonl(games_path, games)
            write_selfplay_examples_jsonl(examples_path, examples)

            round_tripped_games = read_selfplay_games_jsonl(games_path)
            round_tripped_examples = read_selfplay_examples_jsonl(examples_path)

        self.assertEqual(
            [game.to_dict() for game in games],
            [game.to_dict() for game in round_tripped_games],
        )
        self.assertEqual(
            [example.to_dict() for example in examples],
            [example.to_dict() for example in round_tripped_examples],
        )


if __name__ == "__main__":
    unittest.main()
