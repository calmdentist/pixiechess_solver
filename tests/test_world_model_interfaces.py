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
    Event,
    GameState,
    Modifier,
    PieceClass,
    PieceInstance,
)
from pixie_solver.program import ActionContext, EventContext, QueryContext, ThreatContext
from pixie_solver.world_model import ExecutableWorldModelSpec, ObjectiveSpec


class WorldModelInterfaceTest(unittest.TestCase):
    def test_objective_spec_round_trip(self) -> None:
        objective = ObjectiveSpec(
            objective_id="capture_king_v1",
            win_condition="capture_opponent_king",
            legality_mode="king_safety",
            terminal_timing="after_event_resolution",
            metadata={"scope": "pixiechess"},
        )

        round_tripped = ObjectiveSpec.from_dict(objective.to_dict())

        self.assertEqual(objective.to_dict(), round_tripped.to_dict())

    def test_world_model_spec_round_trip(self) -> None:
        world_model = ExecutableWorldModelSpec(
            world_model_id="pixiechess_capture_king_v1",
            world_schema={"topology": "rect_grid", "width": 8, "height": 8},
            entity_programs={
                "phasing_rook": {
                    "program_id": "phasing_rook",
                    "kind": "entity_program",
                }
            },
            global_rule_programs={
                "alternating_turns": {
                    "program_id": "alternating_turns",
                    "kind": "global_rule_program",
                }
            },
            objective=ObjectiveSpec(
                objective_id="capture_king_v1",
                win_condition="capture_opponent_king",
                legality_mode="king_safety",
                terminal_timing="after_event_resolution",
            ),
            query_programs={
                "threat_query": {
                    "program_id": "threat_query",
                    "kind": "query_program",
                }
            },
            constants={"board_files": 8},
            metadata={"source": "unit_test"},
        )

        round_tripped = ExecutableWorldModelSpec.from_dict(world_model.to_dict())

        self.assertEqual(world_model.to_dict(), round_tripped.to_dict())

    def test_program_contexts_expose_entity_aliases(self) -> None:
        piece_class = PieceClass(
            class_id="baseline_rook",
            name="Baseline Rook",
            base_piece_type=BasePieceType.ROOK,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
        )
        piece = PieceInstance(
            instance_id="white_rook",
            piece_class_id=piece_class.class_id,
            color=Color.WHITE,
            square="a1",
        )
        state = GameState(
            piece_classes={piece_class.class_id: piece_class},
            piece_instances={piece.instance_id: piece},
            side_to_move=Color.WHITE,
        )
        program = {"program_id": "baseline_rook"}

        action_context = ActionContext(state=state, piece=piece, program=program)
        query_context = QueryContext(
            state=state,
            piece=piece,
            program=program,
            query_kind="control_relation",
        )
        threat_context = ThreatContext(state=state, piece=piece, program=program)
        event_context = EventContext(
            state=state,
            piece_instances=state.piece_instances,
            hook_owner_id=piece.instance_id,
            program=program,
            event=Event(event_type="turn_start"),
        )

        self.assertEqual(piece, action_context.entity)
        self.assertEqual(piece, query_context.entity)
        self.assertEqual("control_relation", query_context.query_kind)
        self.assertEqual(piece, threat_context.entity)
        self.assertEqual(state.piece_instances, event_context.entity_instances)
        self.assertEqual(piece, event_context.owner)


if __name__ == "__main__":
    unittest.main()
