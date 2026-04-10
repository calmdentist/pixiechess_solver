from __future__ import annotations

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
    Effect,
    Hook,
    Modifier,
    PieceClass,
    PieceInstance,
    StateField,
)
from pixie_solver.core.state import GameState
from pixie_solver.dsl import compile_piece_file
from pixie_solver.simulator.engine import apply_move, is_terminal, result
from pixie_solver.simulator.movegen import is_in_check, legal_moves


class SimulatorTest(unittest.TestCase):
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
        self.counter_piece = PieceClass(
            class_id="counter_pawn",
            name="Counter Pawn",
            base_piece_type=BasePieceType.PAWN,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
            hooks=(
                Hook(
                    event="turn_start",
                    effects=(
                        Effect(
                            op="increment_state",
                            args={"piece": "self", "name": "charges", "amount": 1},
                        ),
                    ),
                    priority=0,
                ),
            ),
            instance_state_schema=(StateField(name="charges", field_type="int", default=0),),
        )

    def test_phasing_rook_moves_through_allies(self) -> None:
        state = GameState(
            piece_classes={
                self.phasing_rook.class_id: self.phasing_rook,
                self.war_automaton.class_id: self.war_automaton,
                self.king.class_id: self.king,
            },
            piece_instances={
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=self.phasing_rook.class_id,
                    color=Color.WHITE,
                    square="a1",
                ),
                "white_pawn": PieceInstance(
                    instance_id="white_pawn",
                    piece_class_id=self.war_automaton.class_id,
                    color=Color.WHITE,
                    square="a2",
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

        rook_moves = [move.to_square for move in legal_moves(state) if move.piece_id == "white_rook"]
        self.assertIn("a3", rook_moves)
        self.assertNotIn("a2", rook_moves)

    def test_sumo_rook_pushes_instead_of_capturing(self) -> None:
        state = GameState(
            piece_classes={
                self.sumo_rook.class_id: self.sumo_rook,
                self.war_automaton.class_id: self.war_automaton,
            },
            piece_instances={
                "white_sumo": PieceInstance(
                    instance_id="white_sumo",
                    piece_class_id=self.sumo_rook.class_id,
                    color=Color.WHITE,
                    square="a1",
                ),
                "black_target": PieceInstance(
                    instance_id="black_target",
                    piece_class_id=self.war_automaton.class_id,
                    color=Color.BLACK,
                    square="a3",
                ),
            },
            side_to_move=Color.WHITE,
        )

        move = next(
            candidate
            for candidate in legal_moves(state)
            if candidate.piece_id == "white_sumo" and candidate.to_square == "a3"
        )
        next_state, delta = apply_move(state, move)

        self.assertEqual("a3", next_state.piece_instances["white_sumo"].square)
        self.assertEqual("a4", next_state.piece_instances["black_target"].square)
        self.assertNotIn("piece_captured", [event.event_type for event in delta.events])

    def test_war_automaton_advances_after_capture(self) -> None:
        state = GameState(
            piece_classes={
                self.phasing_rook.class_id: self.phasing_rook,
                self.war_automaton.class_id: self.war_automaton,
            },
            piece_instances={
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=self.phasing_rook.class_id,
                    color=Color.WHITE,
                    square="h1",
                ),
                "white_war_auto": PieceInstance(
                    instance_id="white_war_auto",
                    piece_class_id=self.war_automaton.class_id,
                    color=Color.WHITE,
                    square="a2",
                ),
                "black_target": PieceInstance(
                    instance_id="black_target",
                    piece_class_id=self.war_automaton.class_id,
                    color=Color.BLACK,
                    square="h3",
                ),
            },
            side_to_move=Color.WHITE,
        )

        move = next(
            candidate
            for candidate in legal_moves(state)
            if candidate.piece_id == "white_rook" and candidate.to_square == "h3"
        )
        next_state, delta = apply_move(state, move)

        self.assertEqual("a3", next_state.piece_instances["white_war_auto"].square)
        self.assertIsNone(next_state.piece_instances["black_target"].square)
        self.assertEqual(Color.BLACK, next_state.side_to_move)
        self.assertIn("piece_captured", [event.event_type for event in delta.events])

    def test_turn_start_hook_updates_piece_state(self) -> None:
        state = GameState(
            piece_classes={
                self.rook.class_id: self.rook,
                self.counter_piece.class_id: self.counter_piece,
                self.king.class_id: self.king,
            },
            piece_instances={
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=self.rook.class_id,
                    color=Color.WHITE,
                    square="a1",
                ),
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=self.king.class_id,
                    color=Color.WHITE,
                    square="h1",
                ),
                "black_counter": PieceInstance(
                    instance_id="black_counter",
                    piece_class_id=self.counter_piece.class_id,
                    color=Color.BLACK,
                    square="a7",
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

        move = next(
            candidate
            for candidate in legal_moves(state)
            if candidate.piece_id == "white_rook" and candidate.to_square == "a2"
        )
        next_state, _ = apply_move(state, move)
        self.assertEqual(1, next_state.piece_instances["black_counter"].state["charges"])

    def test_legal_moves_filter_out_self_check(self) -> None:
        state = GameState(
            piece_classes={
                self.rook.class_id: self.rook,
                self.king.class_id: self.king,
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
                    square="e2",
                ),
                "black_rook": PieceInstance(
                    instance_id="black_rook",
                    piece_class_id=self.rook.class_id,
                    color=Color.BLACK,
                    square="e8",
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

        rook_moves = [move.to_square for move in legal_moves(state) if move.piece_id == "white_rook"]
        self.assertNotIn("a2", rook_moves)
        self.assertIn("e3", rook_moves)

    def test_checkmate_is_terminal(self) -> None:
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
                    square="h1",
                ),
                "black_queen": PieceInstance(
                    instance_id="black_queen",
                    piece_class_id=self.queen.class_id,
                    color=Color.BLACK,
                    square="g2",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="f3",
                ),
            },
            side_to_move=Color.WHITE,
        )

        self.assertTrue(is_in_check(state, Color.WHITE))
        self.assertEqual([], legal_moves(state))
        self.assertTrue(is_terminal(state))
        self.assertEqual("black", result(state))

    def test_stalemate_is_draw(self) -> None:
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
                    square="h1",
                ),
                "black_queen": PieceInstance(
                    instance_id="black_queen",
                    piece_class_id=self.queen.class_id,
                    color=Color.BLACK,
                    square="g3",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="f2",
                ),
            },
            side_to_move=Color.WHITE,
        )

        self.assertFalse(is_in_check(state, Color.WHITE))
        self.assertEqual([], legal_moves(state))
        self.assertTrue(is_terminal(state))
        self.assertEqual("draw", result(state))


if __name__ == "__main__":
    unittest.main()
