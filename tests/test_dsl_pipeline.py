from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType
from pixie_solver.dsl import (
    PieceValidationError,
    canonicalize_piece_program,
    compile_piece_file,
    load_piece_program,
    validate_piece_program,
)


class DSLPipelineTest(unittest.TestCase):
    def test_compile_phasing_rook(self) -> None:
        piece_class = compile_piece_file(
            ROOT / "data/pieces/handauthored/phasing_rook.json"
        )
        self.assertEqual("phasing_rook", piece_class.class_id)
        self.assertEqual(BasePieceType.ROOK, piece_class.base_piece_type)
        self.assertEqual(
            ["inherit_base", "phase_through_allies"],
            [modifier.op for modifier in piece_class.movement_modifiers],
        )

    def test_compile_war_automaton_hook(self) -> None:
        piece_class = compile_piece_file(
            ROOT / "data/pieces/handauthored/war_automaton.json"
        )
        self.assertEqual(1, len(piece_class.hooks))
        self.assertEqual("piece_captured", piece_class.hooks[0].event)
        self.assertEqual(
            ["self_on_board", "square_empty"],
            [condition.op for condition in piece_class.hooks[0].conditions],
        )

    def test_compile_sumo_rook_capture_modifier(self) -> None:
        piece_class = compile_piece_file(
            ROOT / "data/pieces/handauthored/sumo_rook.json"
        )
        self.assertEqual(
            ["replace_capture_with_push"],
            [modifier.op for modifier in piece_class.capture_modifiers],
        )
        self.assertEqual(
            "remove_if_pushed_off_board",
            piece_class.capture_modifiers[0].args["edge_behavior"],
        )

    def test_validator_rejects_unknown_effect(self) -> None:
        program = load_piece_program(ROOT / "data/pieces/handauthored/war_automaton.json")
        program["hooks"][0]["effects"][0]["op"] = "summon_dragon"
        with self.assertRaises(PieceValidationError):
            validate_piece_program(program)

    def test_validator_rejects_invalid_square_reference(self) -> None:
        program = load_piece_program(ROOT / "data/pieces/handauthored/war_automaton.json")
        program["hooks"][0]["conditions"][1]["args"]["square"]["offset"] = [0, "north"]
        with self.assertRaises(PieceValidationError):
            validate_piece_program(program)

    def test_canonicalizer_fills_optional_fields_and_normalizes_squares(self) -> None:
        program = load_piece_program(ROOT / "data/pieces/handauthored/war_automaton.json")
        program["hooks"][0]["conditions"][1]["args"]["square"] = {"absolute": "E4"}
        canonical = canonicalize_piece_program(program)
        self.assertEqual({}, canonical["metadata"])
        self.assertEqual(0, canonical["hooks"][0]["priority"])
        self.assertEqual({}, canonical["hooks"][0]["metadata"])
        self.assertEqual(
            {"absolute": "e4"},
            canonical["hooks"][0]["conditions"][1]["args"]["square"],
        )


if __name__ == "__main__":
    unittest.main()
