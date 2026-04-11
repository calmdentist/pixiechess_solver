from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Color, Modifier, PieceClass, PieceInstance
from pixie_solver.core import sample_standard_initial_state, stable_move_id, standard_initial_state
from pixie_solver.core.state import GameState
from pixie_solver.dsl import compile_piece_file
from pixie_solver.simulator.engine import result
from pixie_solver.simulator.movegen import legal_moves
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
        self.phasing_rook = compile_piece_file(
            ROOT / "data/pieces/handauthored/phasing_rook.json"
        )
        self.sumo_rook = compile_piece_file(
            ROOT / "data/pieces/handauthored/sumo_rook.json"
        )
        self.war_automaton = compile_piece_file(
            ROOT / "data/pieces/handauthored/war_automaton.json"
        )
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
        self.queen = PieceClass(
            class_id="baseline_queen",
            name="Baseline Queen",
            base_piece_type=BasePieceType.QUEEN,
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

    def test_max_ply_cutoff_adjudicates_clear_material_edge(self) -> None:
        state = GameState(
            piece_classes={
                self.king.class_id: self.king,
                self.queen.class_id: self.queen,
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=self.king.class_id,
                    color=Color.WHITE,
                    square="a1",
                ),
                "white_queen": PieceInstance(
                    instance_id="white_queen",
                    piece_class_id=self.queen.class_id,
                    color=Color.WHITE,
                    square="d2",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="h8",
                ),
            },
            side_to_move=Color.WHITE,
        )
        config = SelfPlayConfig(
            simulations=4,
            max_plies=1,
            opening_temperature=0.0,
            final_temperature=0.0,
            temperature_drop_after_ply=0,
            seed=23,
            adjudicate_max_plies=True,
            adjudication_threshold=0.2,
        )

        game = generate_selfplay_games([state], games=1, config=config)[0]

        self.assertEqual("white", game.outcome)
        self.assertEqual("max_plies", game.metadata["termination_reason"])
        self.assertIsNotNone(game.metadata["cutoff_adjudication"])
        self.assertGreater(
            game.metadata["cutoff_adjudication"]["score"],
            game.metadata["cutoff_adjudication"]["threshold"],
        )
        self.assertEqual(1.0, game.examples[0].outcome)
        self.assertEqual(
            "heuristic_material_mobility_check",
            game.examples[0].metadata["cutoff_adjudication"]["method"],
        )

    def test_max_ply_cutoff_can_be_left_as_draw(self) -> None:
        state = GameState(
            piece_classes={
                self.king.class_id: self.king,
                self.queen.class_id: self.queen,
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=self.king.class_id,
                    color=Color.WHITE,
                    square="a1",
                ),
                "white_queen": PieceInstance(
                    instance_id="white_queen",
                    piece_class_id=self.queen.class_id,
                    color=Color.WHITE,
                    square="d2",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="h8",
                ),
            },
            side_to_move=Color.WHITE,
        )
        config = SelfPlayConfig(
            simulations=4,
            max_plies=1,
            opening_temperature=0.0,
            final_temperature=0.0,
            temperature_drop_after_ply=0,
            seed=23,
            adjudicate_max_plies=False,
        )

        game = generate_selfplay_games([state], games=1, config=config)[0]

        self.assertEqual("draw", game.outcome)
        self.assertEqual(0.0, game.examples[0].outcome)
        self.assertIsNone(game.metadata["cutoff_adjudication"])

    def test_standard_initial_state_matches_orthodox_layout(self) -> None:
        state = standard_initial_state()

        self.assertEqual(32, len(state.active_pieces()))
        self.assertEqual(Color.WHITE, state.side_to_move)
        self.assertEqual(("king", "queen"), state.castling_rights["white"])
        self.assertEqual(("king", "queen"), state.castling_rights["black"])
        self.assertEqual("e1", state.piece_instances["white_king"].square)
        self.assertEqual("d8", state.piece_instances["black_queen"].square)
        self.assertEqual("a2", state.piece_instances["white_pawn_a"].square)
        self.assertEqual("h7", state.piece_instances["black_pawn_h"].square)
        self.assertEqual(20, len(legal_moves(state)))

    def test_sample_standard_initial_state_places_specials_on_eligible_slots(self) -> None:
        state = sample_standard_initial_state(
            random.Random(9),
            special_piece_classes=(
                self.phasing_rook,
                self.sumo_rook,
                self.war_automaton,
            ),
            inclusion_probability=1.0,
        )

        white_rook_specials = [
            piece
            for piece in state.active_pieces()
            if piece.color == Color.WHITE and piece.piece_class_id in {"phasing_rook", "sumo_rook"}
        ]
        black_rook_specials = [
            piece
            for piece in state.active_pieces()
            if piece.color == Color.BLACK and piece.piece_class_id in {"phasing_rook", "sumo_rook"}
        ]
        white_war_automata = [
            piece
            for piece in state.active_pieces()
            if piece.color == Color.WHITE and piece.piece_class_id == "war_automaton"
        ]
        black_war_automata = [
            piece
            for piece in state.active_pieces()
            if piece.color == Color.BLACK and piece.piece_class_id == "war_automaton"
        ]

        self.assertEqual(2, len(white_rook_specials))
        self.assertEqual(2, len(black_rook_specials))
        self.assertEqual({"a", "h"}, {piece.square[0] for piece in white_rook_specials})
        self.assertEqual({"a", "h"}, {piece.square[0] for piece in black_rook_specials})
        self.assertEqual(1, len(white_war_automata))
        self.assertEqual(1, len(black_war_automata))
        self.assertEqual(white_war_automata[0].square[0], black_war_automata[0].square[0])
        self.assertEqual("2", white_war_automata[0].square[1])
        self.assertEqual("7", black_war_automata[0].square[1])


if __name__ == "__main__":
    unittest.main()
