from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Modifier, PieceClass
from pixie_solver.dsl import (
    compile_piece_artifacts,
    compile_piece_file_ir,
    load_piece_program,
)
from pixie_solver.program import (
    ProgramValidationError,
    canonicalize_program_ir,
    compile_program_file,
    lower_legacy_piece_class,
    lower_legacy_piece_program,
    validate_program_ir,
)


class ProgramIRTest(unittest.TestCase):
    def test_compile_file_ir_lowers_phasing_rook(self) -> None:
        program_ir = compile_piece_file_ir(
            ROOT / "data/pieces/handauthored/phasing_rook.json"
        )

        self.assertEqual("phasing_rook", program_ir["program_id"])
        self.assertEqual("rook", program_ir["base_archetype"])
        self.assertEqual(1, len(program_ir["action_blocks"]))
        self.assertEqual("legacy_base_actions", program_ir["action_blocks"][0]["kind"])
        self.assertEqual(
            ["inherit_base", "phase_through_allies"],
            [
                modifier["op"]
                for modifier in program_ir["action_blocks"][0]["params"]["movement_modifiers"]
            ],
        )

    def test_lowering_war_automaton_emits_reaction_blocks(self) -> None:
        program_ir = lower_legacy_piece_program(
            load_piece_program(ROOT / "data/pieces/handauthored/war_automaton.json")
        )

        self.assertEqual(1, len(program_ir["reaction_blocks"]))
        reaction = program_ir["reaction_blocks"][0]
        self.assertEqual("event_reaction", reaction["kind"])
        self.assertEqual("piece_captured", reaction["trigger"]["event_type"])
        self.assertEqual(
            ["self_on_board", "square_empty"],
            [condition["condition_kind"] for condition in reaction["conditions"]],
        )

    def test_compile_piece_artifacts_emits_piece_class_and_program_ir(self) -> None:
        artifacts = compile_piece_artifacts(
            load_piece_program(ROOT / "data/pieces/handauthored/sumo_rook.json")
        )

        self.assertEqual(artifacts.piece_class.class_id, artifacts.program_ir["program_id"])
        self.assertEqual(
            artifacts.piece_class.base_piece_type.value,
            artifacts.program_ir["base_archetype"],
        )

    def test_canonicalize_program_ir_normalizes_nested_square_refs(self) -> None:
        program_ir = compile_piece_file_ir(
            ROOT / "data/pieces/handauthored/war_automaton.json"
        )
        reaction = dict(program_ir["reaction_blocks"][0])
        reaction["conditions"] = list(reaction["conditions"])
        reaction["conditions"][1] = {
            "condition_kind": "square_empty",
            "args": {"square": {"absolute": "E4"}},
        }
        candidate = {
            **program_ir,
            "reaction_blocks": [reaction],
        }

        canonical = canonicalize_program_ir(candidate)

        self.assertEqual(
            {"absolute": "e4"},
            canonical["reaction_blocks"][0]["conditions"][1]["args"]["square"],
        )

    def test_validator_rejects_unknown_action_block_kind(self) -> None:
        program_ir = compile_piece_file_ir(
            ROOT / "data/pieces/handauthored/phasing_rook.json"
        )
        broken = {
            **program_ir,
            "action_blocks": [
                {
                    **program_ir["action_blocks"][0],
                    "kind": "mystery_actions",
                }
            ],
        }

        with self.assertRaises(ProgramValidationError):
            validate_program_ir(broken)

    def test_program_compiler_accepts_legacy_piece_files(self) -> None:
        program_ir = compile_program_file(
            ROOT / "data/pieces/handauthored/sumo_rook.json"
        )

        self.assertEqual("sumo_rook", program_ir["program_id"])
        self.assertEqual(
            "replace_capture_with_push",
            program_ir["action_blocks"][0]["params"]["capture_modifiers"][0]["op"],
        )

    def test_validator_rejects_duplicate_block_ids(self) -> None:
        program_ir = compile_piece_file_ir(
            ROOT / "data/pieces/handauthored/war_automaton.json"
        )
        broken = {
            **program_ir,
            "reaction_blocks": [
                {
                    **program_ir["reaction_blocks"][0],
                    "block_id": program_ir["action_blocks"][0]["block_id"],
                }
            ],
        }

        with self.assertRaises(ProgramValidationError):
            validate_program_ir(broken)

    def test_lower_legacy_piece_class_normalizes_fail_push_edge_behavior(self) -> None:
        piece_class = PieceClass(
            class_id="compat_sumo",
            name="Compat Sumo",
            base_piece_type=BasePieceType.ROOK,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(
                Modifier(
                    op="replace_capture_with_push",
                    args={"distance": 1, "edge_behavior": "fail"},
                ),
            ),
        )

        program_ir = lower_legacy_piece_class(piece_class)

        self.assertEqual(
            "block",
            program_ir["action_blocks"][0]["params"]["capture_modifiers"][0]["args"][
                "edge_behavior"
            ],
        )


if __name__ == "__main__":
    unittest.main()
