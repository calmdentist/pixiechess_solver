from __future__ import annotations

import json
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
    stable_position_hash,
)
from pixie_solver.core.state import GameState
from pixie_solver.dsl import compile_piece_file
from pixie_solver.simulator.engine import apply_move, is_terminal, result
from pixie_solver.simulator.movegen import is_in_check, legal_moves
from pixie_solver.utils import ReplayTrace, build_replay_trace, canonical_json, replay_trace


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
        self.bishop = PieceClass(
            class_id="baseline_bishop",
            name="Baseline Bishop",
            base_piece_type=BasePieceType.BISHOP,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
        )
        self.knight = PieceClass(
            class_id="baseline_knight",
            name="Baseline Knight",
            base_piece_type=BasePieceType.KNIGHT,
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
        self.pawn = PieceClass(
            class_id="baseline_pawn",
            name="Baseline Pawn",
            base_piece_type=BasePieceType.PAWN,
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

    def test_threefold_repetition_is_draw(self) -> None:
        state = GameState(
            piece_classes={
                self.king.class_id: self.king,
                self.knight.class_id: self.knight,
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=self.king.class_id,
                    color=Color.WHITE,
                    square="e1",
                ),
                "white_knight": PieceInstance(
                    instance_id="white_knight",
                    piece_class_id=self.knight.class_id,
                    color=Color.WHITE,
                    square="b1",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="e8",
                ),
                "black_knight": PieceInstance(
                    instance_id="black_knight",
                    piece_class_id=self.knight.class_id,
                    color=Color.BLACK,
                    square="b8",
                ),
            },
            side_to_move=Color.WHITE,
        )
        initial_position_hash = stable_position_hash(state)

        for _ in range(2):
            state = self._play_to(state, "white_knight", "c3")
            state = self._play_to(state, "black_knight", "c6")
            state = self._play_to(state, "white_knight", "b1")
            state = self._play_to(state, "black_knight", "b8")

        self.assertEqual(3, state.repetition_counts[initial_position_hash])
        self.assertTrue(is_terminal(state))
        self.assertEqual("draw", result(state))

    def test_fifty_move_rule_is_draw(self) -> None:
        state = GameState(
            piece_classes={
                self.king.class_id: self.king,
                self.knight.class_id: self.knight,
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=self.king.class_id,
                    color=Color.WHITE,
                    square="e1",
                ),
                "white_knight": PieceInstance(
                    instance_id="white_knight",
                    piece_class_id=self.knight.class_id,
                    color=Color.WHITE,
                    square="b1",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="e8",
                ),
            },
            side_to_move=Color.WHITE,
            halfmove_clock=100,
        )

        self.assertTrue(legal_moves(state))
        self.assertEqual("draw", result(state))

    def test_historical_repetition_does_not_draw_different_current_position(self) -> None:
        state = GameState(
            piece_classes={
                self.king.class_id: self.king,
                self.knight.class_id: self.knight,
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=self.king.class_id,
                    color=Color.WHITE,
                    square="e1",
                ),
                "white_knight": PieceInstance(
                    instance_id="white_knight",
                    piece_class_id=self.knight.class_id,
                    color=Color.WHITE,
                    square="b1",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="e8",
                ),
            },
            side_to_move=Color.WHITE,
            repetition_counts={"some_other_position": 3},
        )

        self.assertIsNone(result(state))

    def test_kingside_castle_is_generated_and_applied(self) -> None:
        state = GameState(
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
                "white_rook_h": PieceInstance(
                    instance_id="white_rook_h",
                    piece_class_id=self.rook.class_id,
                    color=Color.WHITE,
                    square="h1",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="e8",
                ),
            },
            side_to_move=Color.WHITE,
            castling_rights={"white": ("king",)},
        )

        move = next(
            candidate
            for candidate in legal_moves(state)
            if candidate.move_kind == "castle" and candidate.to_square == "g1"
        )
        next_state, delta = apply_move(state, move)

        self.assertEqual("g1", next_state.piece_instances["white_king"].square)
        self.assertEqual("f1", next_state.piece_instances["white_rook_h"].square)
        self.assertNotIn("white", next_state.castling_rights)
        self.assertEqual(state.state_hash(), delta.metadata["before_state_hash"])
        self.assertEqual(next_state.state_hash(), delta.metadata["after_state_hash"])

    def test_castle_is_blocked_when_path_square_is_attacked(self) -> None:
        state = GameState(
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
                "white_rook_h": PieceInstance(
                    instance_id="white_rook_h",
                    piece_class_id=self.rook.class_id,
                    color=Color.WHITE,
                    square="h1",
                ),
                "black_rook": PieceInstance(
                    instance_id="black_rook",
                    piece_class_id=self.rook.class_id,
                    color=Color.BLACK,
                    square="f8",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="a8",
                ),
            },
            side_to_move=Color.WHITE,
            castling_rights={"white": ("king",)},
        )

        castle_moves = [move for move in legal_moves(state) if move.move_kind == "castle"]
        self.assertEqual([], castle_moves)

    def test_en_passant_is_generated_and_applied(self) -> None:
        state = GameState(
            piece_classes={
                self.king.class_id: self.king,
                self.pawn.class_id: self.pawn,
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=self.king.class_id,
                    color=Color.WHITE,
                    square="h1",
                ),
                "white_pawn": PieceInstance(
                    instance_id="white_pawn",
                    piece_class_id=self.pawn.class_id,
                    color=Color.WHITE,
                    square="e5",
                ),
                "black_pawn": PieceInstance(
                    instance_id="black_pawn",
                    piece_class_id=self.pawn.class_id,
                    color=Color.BLACK,
                    square="d7",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="a8",
                ),
            },
            side_to_move=Color.BLACK,
        )

        black_double_step = next(
            candidate
            for candidate in legal_moves(state)
            if candidate.piece_id == "black_pawn" and candidate.to_square == "d5"
        )
        state_after_black, _ = apply_move(state, black_double_step)
        self.assertEqual("d6", state_after_black.en_passant_square)

        en_passant = next(
            candidate
            for candidate in legal_moves(state_after_black)
            if candidate.move_kind == "en_passant_capture"
        )
        next_state, _ = apply_move(state_after_black, en_passant)

        self.assertEqual("d6", next_state.piece_instances["white_pawn"].square)
        self.assertIsNone(next_state.piece_instances["black_pawn"].square)
        self.assertIsNone(next_state.en_passant_square)

    def test_pawn_promotion_creates_promoted_piece_class(self) -> None:
        state = GameState(
            piece_classes={
                self.king.class_id: self.king,
                self.queen.class_id: self.queen,
                self.rook.class_id: self.rook,
                self.bishop.class_id: self.bishop,
                self.knight.class_id: self.knight,
                self.pawn.class_id: self.pawn,
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=self.king.class_id,
                    color=Color.WHITE,
                    square="h1",
                ),
                "white_pawn": PieceInstance(
                    instance_id="white_pawn",
                    piece_class_id=self.pawn.class_id,
                    color=Color.WHITE,
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

        promotion_moves = [
            candidate
            for candidate in legal_moves(state)
            if candidate.piece_id == "white_pawn" and candidate.to_square == "a8"
        ]
        self.assertEqual(
            {"queen", "rook", "bishop", "knight"},
            {move.promotion_piece_type for move in promotion_moves},
        )

        queen_promotion = next(
            candidate
            for candidate in promotion_moves
            if candidate.promotion_piece_type == "queen"
        )
        next_state, _ = apply_move(state, queen_promotion)

        promoted_piece = next_state.piece_instances["white_pawn"]
        self.assertEqual("a8", promoted_piece.square)
        self.assertEqual(self.queen.class_id, promoted_piece.piece_class_id)
        self.assertEqual({}, promoted_piece.state)

    def test_replay_trace_round_trips_and_replays_deterministically(self) -> None:
        initial_state = GameState(
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

        first_move = next(
            candidate
            for candidate in legal_moves(initial_state)
            if candidate.piece_id == "white_rook" and candidate.to_square == "a2"
        )
        state_after_first, _ = apply_move(initial_state, first_move)
        second_move = next(
            candidate
            for candidate in legal_moves(state_after_first)
            if candidate.piece_id == "black_king" and candidate.to_square == "g8"
        )

        trace = build_replay_trace(initial_state, [first_move, second_move])
        round_tripped = ReplayTrace.from_dict(
            json.loads(canonical_json(trace.to_dict()))
        )
        replayed_final_state = replay_trace(round_tripped)

        self.assertEqual(trace.final_state_hash, replayed_final_state.state_hash())
        self.assertEqual(2, replayed_final_state.piece_instances["black_counter"].state["charges"])

    def _play_to(self, state: GameState, piece_id: str, to_square: str) -> GameState:
        move = next(
            candidate
            for candidate in legal_moves(state)
            if candidate.piece_id == piece_id and candidate.to_square == to_square
        )
        next_state, _ = apply_move(state, move)
        return next_state


if __name__ == "__main__":
    unittest.main()
