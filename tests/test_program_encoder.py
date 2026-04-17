from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.dsl.compiler import compile_piece_file
from pixie_solver.dsl.parser import load_piece_program
from pixie_solver.model import ProgramIRTokenEncoder, program_token_specs
from pixie_solver.program.compiler import compile_program_file, compile_program_ir


class ProgramEncoderTest(unittest.TestCase):
    def test_program_token_specs_are_deterministic_for_handauthored_piece(self) -> None:
        program_path = ROOT / "data/pieces/handauthored/war_automaton.json"
        program = load_piece_program(program_path)

        first = program_token_specs(program)
        second = program_token_specs(program)

        self.assertEqual(first, second)
        self.assertEqual("program", first[0].role)
        self.assertEqual("program", first[0].path)
        self.assertIn(
            "reaction_blocks[0].conditions[1].args.square.offset[1]",
            {token.path for token in first},
        )

    def test_program_token_specs_match_piece_class_and_program_ir_sources(self) -> None:
        program_path = ROOT / "data/pieces/handauthored/sumo_rook.json"
        piece_class = compile_piece_file(program_path)
        program_ir = compile_program_file(program_path)

        from_piece_class = program_token_specs(piece_class)
        from_program_ir = program_token_specs(program_ir)

        self.assertEqual(from_program_ir, from_piece_class)

    def test_program_token_specs_sort_mapping_keys_deterministically(self) -> None:
        program_a = compile_program_ir(
            {
                "program_id": "constant_sort",
                "name": "Constant Sort",
                "schema_version": 1,
                "base_archetype": "bishop",
                "state_schema": [],
                "constants": {
                    "zeta": {"later": 2, "earlier": 1},
                    "alpha": ["x", "y"],
                },
                "action_blocks": [],
                "query_blocks": [],
                "reaction_blocks": [],
                "metadata": {},
            }
        )
        program_b = compile_program_ir(
            {
                "program_id": "constant_sort",
                "name": "Constant Sort",
                "schema_version": 1,
                "base_archetype": "bishop",
                "state_schema": [],
                "constants": {
                    "alpha": ["x", "y"],
                    "zeta": {"earlier": 1, "later": 2},
                },
                "action_blocks": [],
                "query_blocks": [],
                "reaction_blocks": [],
                "metadata": {},
            }
        )

        self.assertEqual(program_token_specs(program_a), program_token_specs(program_b))

    def test_program_ir_token_encoder_batches_and_pads(self) -> None:
        encoder = ProgramIRTokenEncoder(d_model=32)
        short_program = compile_program_file(ROOT / "data/pieces/handauthored/phasing_rook.json")
        long_program = compile_program_file(ROOT / "data/pieces/handauthored/war_automaton.json")

        encoded = encoder((short_program, long_program))

        self.assertEqual(("phasing_rook", "war_automaton"), encoded.program_ids)
        self.assertEqual(2, encoded.token_embeddings.shape[0])
        self.assertEqual(32, encoded.token_embeddings.shape[2])
        self.assertEqual(encoded.token_embeddings.shape[:2], encoded.padding_mask.shape)
        self.assertTrue(encoded.padding_mask[0].any().item())
        self.assertFalse(encoded.padding_mask[1].all().item())
        self.assertGreater(len(encoded.token_specs[1]), len(encoded.token_specs[0]))


if __name__ == "__main__":
    unittest.main()
