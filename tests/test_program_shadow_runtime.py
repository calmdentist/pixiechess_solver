from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Color, Modifier, PieceClass, PieceInstance, StateField
from pixie_solver.dsl import compile_piece_file
from pixie_solver.simulator import apply_action_shadow, enumerate_shadow_legal_actions
from pixie_solver.simulator.engine import apply_move_legacy
from pixie_solver.simulator.movegen import legacy_legal_moves


class ProgramShadowRuntimeTest(unittest.TestCase):
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
        self.counter_piece = PieceClass(
            class_id="counter_pawn",
            name="Counter Pawn",
            base_piece_type=BasePieceType.PAWN,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
            hooks=(),
            instance_state_schema=(StateField(name="charges", field_type="int", default=0),),
        )

    def test_shadow_actiongen_matches_legal_moves(self) -> None:
        from pixie_solver.core.state import GameState

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
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=self.king.class_id,
                    color=Color.WHITE,
                    square="h1",
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

        shadow_ids = {
            action.stable_id()
            for action in enumerate_shadow_legal_actions(state)
        }
        legacy_ids = {
            move.to_action_intent().stable_id()
            for move in legacy_legal_moves(state)
        }

        self.assertEqual(legacy_ids, shadow_ids)

    def test_shadow_apply_action_matches_war_automaton_transition(self) -> None:
        from pixie_solver.core.state import GameState

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
            for candidate in legacy_legal_moves(state)
            if candidate.piece_id == "white_rook" and candidate.to_square == "h3"
        )
        shadow_state, shadow_delta = apply_action_shadow(state, move.to_action_intent())
        legacy_state, legacy_delta = apply_move_legacy(state, move)

        self.assertEqual(legacy_state.to_dict(), shadow_state.to_dict())
        self.assertEqual(
            [event.event_type for event in legacy_delta.events],
            [event.event_type for event in shadow_delta.events],
        )
        self.assertGreater(len(shadow_delta.effects), 0)
        self.assertIsNotNone(shadow_delta.trace)

    def test_shadow_apply_action_matches_turn_start_hook_transition(self) -> None:
        from pixie_solver.core import Effect, Hook
        from pixie_solver.core.state import GameState

        counter_piece = PieceClass(
            class_id="counter_pawn_hooked",
            name="Counter Pawn Hooked",
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
                ),
            ),
            instance_state_schema=(StateField(name="charges", field_type="int", default=0),),
        )
        state = GameState(
            piece_classes={
                self.rook.class_id: self.rook,
                counter_piece.class_id: counter_piece,
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
                    piece_class_id=counter_piece.class_id,
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
            for candidate in legacy_legal_moves(state)
            if candidate.piece_id == "white_rook" and candidate.to_square == "a2"
        )
        shadow_state, shadow_delta = apply_action_shadow(state, move)
        legacy_state, _ = apply_move_legacy(state, move)

        self.assertEqual(legacy_state.to_dict(), shadow_state.to_dict())
        self.assertEqual(1, shadow_state.piece_instances["black_counter"].state["charges"])
        self.assertTrue(
            any(effect.effect_kind == "increment_state" for effect in shadow_delta.effects)
        )

    def test_shadow_trace_is_deterministic(self) -> None:
        from pixie_solver.core.state import GameState

        state = GameState(
            piece_classes={
                self.rook.class_id: self.rook,
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
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=self.king.class_id,
                    color=Color.BLACK,
                    square="h8",
                ),
            },
            side_to_move=Color.WHITE,
        )

        action = next(iter(enumerate_shadow_legal_actions(state)))
        _, first_delta = apply_action_shadow(state, action)
        _, second_delta = apply_action_shadow(state, action)

        self.assertEqual(first_delta.to_dict(), second_delta.to_dict())


if __name__ == "__main__":
    unittest.main()
