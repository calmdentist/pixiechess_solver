from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import sample_standard_initial_state, standard_initial_state
from pixie_solver.dsl import compile_piece_file
from pixie_solver.simulator.engine import apply_move, apply_move_legacy
from pixie_solver.simulator.movegen import legal_moves, legacy_legal_moves


class ProgramEquivalenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.special_piece_classes = [
            compile_piece_file(ROOT / "data/pieces/handauthored/phasing_rook.json"),
            compile_piece_file(ROOT / "data/pieces/handauthored/sumo_rook.json"),
            compile_piece_file(ROOT / "data/pieces/handauthored/war_automaton.json"),
        ]

    def test_standard_random_playout_matches_legacy_engine(self) -> None:
        rng = random.Random(7)
        legacy_state = standard_initial_state()
        program_state = standard_initial_state()

        for _ in range(6):
            self._assert_state_equivalent(legacy_state, program_state)
            legacy_moves_by_id = self._moves_by_id(legacy_legal_moves(legacy_state))
            if not legacy_moves_by_id:
                break
            chosen_move_id = rng.choice(sorted(legacy_moves_by_id))
            program_moves_by_id = self._moves_by_id(legal_moves(program_state))
            legacy_state, _ = apply_move_legacy(
                legacy_state,
                legacy_moves_by_id[chosen_move_id],
            )
            program_state, _ = apply_move(
                program_state,
                program_moves_by_id[chosen_move_id],
            )
            self.assertEqual(legacy_state.to_dict(), program_state.to_dict())

    def test_special_piece_random_playouts_match_legacy_engine(self) -> None:
        rng = random.Random(11)
        for _ in range(3):
            legacy_state = sample_standard_initial_state(
                rng,
                special_piece_classes=self.special_piece_classes,
                inclusion_probability=1.0,
            )
            program_state = legacy_state
            for _ in range(4):
                self._assert_state_equivalent(legacy_state, program_state)
                legacy_moves_by_id = self._moves_by_id(legacy_legal_moves(legacy_state))
                if not legacy_moves_by_id:
                    break
                chosen_move_id = rng.choice(sorted(legacy_moves_by_id))
                program_moves_by_id = self._moves_by_id(legal_moves(program_state))
                legacy_state, _ = apply_move_legacy(
                    legacy_state,
                    legacy_moves_by_id[chosen_move_id],
                )
                program_state, _ = apply_move(
                    program_state,
                    program_moves_by_id[chosen_move_id],
                )
                self.assertEqual(legacy_state.to_dict(), program_state.to_dict())

    def _assert_state_equivalent(self, legacy_state, program_state) -> None:
        self.assertEqual(legacy_state.to_dict(), program_state.to_dict())
        legacy_moves_by_id = self._moves_by_id(legacy_legal_moves(legacy_state))
        program_moves_by_id = self._moves_by_id(legal_moves(program_state))
        self.assertEqual(
            set(legacy_moves_by_id),
            set(program_moves_by_id),
            {"legacy": sorted(legacy_moves_by_id), "program": sorted(program_moves_by_id)},
        )

        for move_id in sorted(legacy_moves_by_id):
            legacy_next_state, legacy_delta = apply_move_legacy(
                legacy_state,
                legacy_moves_by_id[move_id],
            )
            program_next_state, program_delta = apply_move(
                program_state,
                program_moves_by_id[move_id],
            )
            self.assertEqual(
                legacy_next_state.to_dict(),
                program_next_state.to_dict(),
                move_id,
            )
            self.assertEqual(
                [self._event_signature(event) for event in legacy_delta.events],
                [self._event_signature(event) for event in program_delta.events],
                move_id,
            )
            self.assertEqual(
                legacy_delta.changed_piece_ids,
                program_delta.changed_piece_ids,
                move_id,
            )
            self.assertEqual(
                legacy_delta.metadata.get("after_state_hash"),
                program_delta.metadata.get("after_state_hash"),
                move_id,
            )

    @staticmethod
    def _moves_by_id(moves) -> dict[str, object]:
        return {move.stable_id(): move for move in moves}

    @staticmethod
    def _event_signature(event) -> dict[str, object]:
        return {
            "event_type": event.event_type,
            "actor_piece_id": event.actor_piece_id,
            "target_piece_id": event.target_piece_id,
            "payload": dict(event.payload),
            "source_cause": event.source_cause,
            "sequence": event.sequence,
        }


if __name__ == "__main__":
    unittest.main()
