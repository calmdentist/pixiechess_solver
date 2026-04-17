from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Color, Modifier, PieceClass, PieceInstance
from pixie_solver.core.state import GameState
from pixie_solver.dsl import compile_piece_file, compile_piece_file_ir
from pixie_solver.simulator.movegen import is_in_check, is_square_attacked
from pixie_solver.simulator.query import (
    CAPTURE_CONTROL_QUERY_KIND,
    enumerate_query_facts_for_piece,
    query_fact_exists,
)


class QueryRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.phasing_rook = compile_piece_file(
            ROOT / "data/pieces/handauthored/phasing_rook.json"
        )
        self.sumo_rook = compile_piece_file(
            ROOT / "data/pieces/handauthored/sumo_rook.json"
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

    def test_legacy_lowering_emits_query_blocks(self) -> None:
        program_ir = compile_piece_file_ir(ROOT / "data/pieces/handauthored/phasing_rook.json")

        self.assertEqual(1, len(program_ir["query_blocks"]))
        self.assertEqual("legacy_capture_control", program_ir["query_blocks"][0]["kind"])
        self.assertEqual(
            program_ir["action_blocks"][0]["params"],
            program_ir["query_blocks"][0]["params"],
        )

    def test_query_runtime_enumerates_capture_control_for_phasing_rook(self) -> None:
        state = GameState(
            piece_classes={
                self.phasing_rook.class_id: self.phasing_rook,
                self.rook.class_id: self.rook,
                self.king.class_id: self.king,
            },
            piece_instances={
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=self.phasing_rook.class_id,
                    color=Color.WHITE,
                    square="a1",
                ),
                "white_blocker": PieceInstance(
                    instance_id="white_blocker",
                    piece_class_id=self.rook.class_id,
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

        targets = {
            fact.target_ref
            for fact in enumerate_query_facts_for_piece(
                state,
                piece_id="white_rook",
                query_kind=CAPTURE_CONTROL_QUERY_KIND,
            )
            if fact.query_kind == CAPTURE_CONTROL_QUERY_KIND
        }

        self.assertIn("a3", targets)
        self.assertNotIn("a2", targets)

    def test_query_runtime_does_not_delegate_to_movegen_attack_helper(self) -> None:
        state = GameState(
            piece_classes={
                self.phasing_rook.class_id: self.phasing_rook,
                self.rook.class_id: self.rook,
            },
            piece_instances={
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=self.phasing_rook.class_id,
                    color=Color.WHITE,
                    square="a1",
                ),
                "white_blocker": PieceInstance(
                    instance_id="white_blocker",
                    piece_class_id=self.rook.class_id,
                    color=Color.WHITE,
                    square="a2",
                ),
            },
            side_to_move=Color.WHITE,
        )

        with patch(
            "pixie_solver.simulator.movegen._piece_attacks_square",
            side_effect=AssertionError("query runtime should not call movegen helper"),
        ):
            targets = {
                fact.target_ref
                for fact in enumerate_query_facts_for_piece(
                    state,
                    piece_id="white_rook",
                    query_kind=CAPTURE_CONTROL_QUERY_KIND,
                )
            }

        self.assertIn("a3", targets)

    def test_query_runtime_respects_push_capture_legality(self) -> None:
        blocked_state = GameState(
            piece_classes={
                self.sumo_rook.class_id: self.sumo_rook,
                self.rook.class_id: self.rook,
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
                    piece_class_id=self.rook.class_id,
                    color=Color.BLACK,
                    square="a3",
                ),
                "white_blocker": PieceInstance(
                    instance_id="white_blocker",
                    piece_class_id=self.rook.class_id,
                    color=Color.BLACK,
                    square="a4",
                ),
            },
            side_to_move=Color.WHITE,
        )
        open_state = GameState(
            piece_classes={
                self.sumo_rook.class_id: self.sumo_rook,
                self.rook.class_id: self.rook,
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
                    piece_class_id=self.rook.class_id,
                    color=Color.BLACK,
                    square="a3",
                ),
            },
            side_to_move=Color.WHITE,
        )

        self.assertFalse(
            query_fact_exists(
                blocked_state,
                target_ref="a3",
                by_color=Color.WHITE,
                query_kind=CAPTURE_CONTROL_QUERY_KIND,
            )
        )
        self.assertTrue(
            query_fact_exists(
                open_state,
                target_ref="a3",
                by_color=Color.WHITE,
                query_kind=CAPTURE_CONTROL_QUERY_KIND,
            )
        )

    def test_movegen_attack_wrapper_delegates_to_query_runtime(self) -> None:
        state = GameState(piece_classes={}, piece_instances={}, side_to_move=Color.WHITE)

        with patch("pixie_solver.simulator.query.is_square_capturable", return_value=True) as mock_fn:
            self.assertTrue(is_square_attacked(state, "e4", by_color=Color.BLACK))
            mock_fn.assert_called_once_with(state, "e4", by_color=Color.BLACK)

    def test_movegen_check_wrapper_delegates_to_query_runtime(self) -> None:
        state = GameState(piece_classes={}, piece_instances={}, side_to_move=Color.WHITE)

        with patch("pixie_solver.simulator.query.is_king_capturable", return_value=True) as mock_fn:
            self.assertTrue(is_in_check(state, Color.WHITE))
            mock_fn.assert_called_once_with(state, Color.WHITE)


if __name__ == "__main__":
    unittest.main()
