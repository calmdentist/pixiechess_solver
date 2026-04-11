from __future__ import annotations

from copy import deepcopy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import Color, PieceInstance
from pixie_solver.dsl import compile_piece_program, load_piece_program
from pixie_solver.rules import (
    StaticRepairProvider,
    append_verified_piece_version,
    build_state_mismatch,
    load_verified_piece_classes,
    load_verified_piece_records,
    repair_and_verify_piece,
)
from pixie_solver.rules.mismatch import replace_piece_program
from pixie_solver.simulator.engine import apply_move
from pixie_solver.simulator.movegen import legal_moves
from pixie_solver.core.state import GameState


class RuleRepairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.phasing_rook_program = load_piece_program(
            ROOT / "data/pieces/handauthored/phasing_rook.json"
        )
        self.current_program = load_piece_program(
            ROOT / "data/pieces/handauthored/war_automaton.json"
        )
        self.patched_program = _war_automaton_with_forward_offset(2)
        self.phasing_rook = compile_piece_program(self.phasing_rook_program)
        self.current_war_automaton = compile_piece_program(self.current_program)
        self.before_state = GameState(
            piece_classes={
                self.phasing_rook.class_id: self.phasing_rook,
                self.current_war_automaton.class_id: self.current_war_automaton,
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
                    piece_class_id=self.current_war_automaton.class_id,
                    color=Color.WHITE,
                    square="a2",
                ),
                "black_target": PieceInstance(
                    instance_id="black_target",
                    piece_class_id=self.current_war_automaton.class_id,
                    color=Color.BLACK,
                    square="h3",
                ),
            },
            side_to_move=Color.WHITE,
        )
        self.move = next(
            move
            for move in legal_moves(self.before_state)
            if move.piece_id == "white_rook" and move.to_square == "h3"
        )
        teacher_before = replace_piece_program(self.before_state, self.patched_program)
        self.observed_state, _ = apply_move(teacher_before, self.move)

    def test_repair_accepts_patch_that_matches_observed_transition(self) -> None:
        mismatch = build_state_mismatch(
            before_state=self.before_state,
            move=self.move,
            observed_state=self.observed_state,
            current_program=self.current_program,
        )

        result = repair_and_verify_piece(
            mismatch,
            description="A pawn that surges forward after a capture.",
            current_program=self.current_program,
            provider=StaticRepairProvider(self.patched_program),
        )

        self.assertTrue(result.accepted, result.verification_errors)
        self.assertEqual(1, result.verified_cases)
        self.assertIsNone(result.primary_diff_after_repair)
        self.assertIn("white_war_auto", mismatch.diff.changed_piece_ids)
        self.assertEqual(
            "a3",
            mismatch.predicted_state.piece_instances["white_war_auto"].square,
        )
        self.assertEqual(
            "a4",
            self.observed_state.piece_instances["white_war_auto"].square,
        )

    def test_repair_rejects_patch_that_still_mismatches(self) -> None:
        mismatch = build_state_mismatch(
            before_state=self.before_state,
            move=self.move,
            observed_state=self.observed_state,
            current_program=self.current_program,
        )

        result = repair_and_verify_piece(
            mismatch,
            description="A pawn that surges forward after a capture.",
            current_program=self.current_program,
            provider=StaticRepairProvider(self.current_program),
        )

        self.assertFalse(result.accepted)
        self.assertIn(
            "primary_mismatch: patched transition still differs from observed state",
            result.verification_errors,
        )
        self.assertIsNotNone(result.primary_diff_after_repair)

    def test_repair_rejects_piece_id_change(self) -> None:
        changed_id = deepcopy(self.patched_program)
        changed_id["piece_id"] = "different_war_automaton"
        mismatch = build_state_mismatch(
            before_state=self.before_state,
            move=self.move,
            observed_state=self.observed_state,
            current_program=self.current_program,
        )

        result = repair_and_verify_piece(
            mismatch,
            description="A pawn that surges forward after a capture.",
            current_program=self.current_program,
            provider=StaticRepairProvider(changed_id),
        )

        self.assertFalse(result.accepted)
        self.assertIn("changed piece_id", result.verification_errors[0])

    def test_registry_persists_latest_verified_piece_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "registry.json"
            out_dir = temp_path / "repaired"

            first = append_verified_piece_version(
                registry_path=registry_path,
                out_dir=out_dir,
                program=self.current_program,
                description="Original one-square automaton.",
                source="handoff",
                verified_cases=1,
            )
            second = append_verified_piece_version(
                registry_path=registry_path,
                out_dir=out_dir,
                program=self.patched_program,
                description="Repaired two-square automaton.",
                source="repair",
                parent_digest=first.dsl_digest,
                verified_cases=2,
            )

            records = load_verified_piece_records(registry_path)
            classes = load_verified_piece_classes(registry_path)

            self.assertEqual(1, first.version)
            self.assertEqual(2, second.version)
            self.assertEqual(1, len(records))
            self.assertEqual(second.dsl_digest, records[0].dsl_digest)
            self.assertEqual("war_automaton", classes[0].class_id)
            self.assertEqual(2, records[0].verified_cases)


def _war_automaton_with_forward_offset(offset: int) -> dict[str, object]:
    program = load_piece_program(ROOT / "data/pieces/handauthored/war_automaton.json")
    hook = program["hooks"][0]
    hook["conditions"][1]["args"]["square"]["offset"] = [0, offset]
    hook["effects"][0]["args"]["to"]["offset"] = [0, offset]
    return program


if __name__ == "__main__":
    unittest.main()
