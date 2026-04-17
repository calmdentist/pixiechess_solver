from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import (
    ActionIntent,
    Event,
    Move,
    QueryFact,
    StateDelta,
    ThreatMark,
    TraceFrame,
    TransitionEffect,
    TransitionTrace,
    action_intent_from_move,
    move_from_action_intent,
    stable_action_id,
    stable_query_fact_id,
    stable_threat_id,
)


class RuntimeContractTest(unittest.TestCase):
    def test_action_intent_round_trip_and_stable_id(self) -> None:
        action = ActionIntent(
            action_kind="push_capture",
            actor_piece_id="white_rook",
            from_square="a1",
            to_square="a4",
            target_piece_id="black_knight",
            tags=("tactical",),
            params={"push_distance": 2, "push_direction": [0, 1]},
            metadata={"source": "unit_test"},
        )

        round_tripped = ActionIntent.from_dict(action.to_dict())

        self.assertEqual(action.to_dict(), round_tripped.to_dict())
        self.assertEqual(stable_action_id(action), stable_action_id(round_tripped))

    def test_move_conversion_round_trips_through_action_intent(self) -> None:
        move = Move(
            piece_id="white_rook",
            from_square="a1",
            to_square="a4",
            move_kind="push_capture",
            metadata={
                "target_piece_id": "black_knight",
                "push_distance": 2,
                "push_direction": [0, 1],
            },
            tags=("push",),
        )

        action = action_intent_from_move(move)
        round_tripped = move_from_action_intent(action)

        self.assertEqual(move.to_dict(), round_tripped.to_dict())
        self.assertEqual(action.to_dict(), move.to_action_intent().to_dict())
        self.assertEqual(move.to_dict(), Move.from_action_intent(action).to_dict())

    def test_transition_trace_round_trip(self) -> None:
        event = Event(
            event_type="piece_moved",
            actor_piece_id="white_rook",
            source_action_id="action_1",
            source_frame_id=0,
            metadata={"phase": "commit"},
        )
        effect = TransitionEffect(
            effect_kind="move_piece",
            actor_piece_id="white_rook",
            target_square="a4",
            payload={"reason": "commit"},
            sequence=1,
        )
        trace = TransitionTrace(
            action_id="action_1",
            frames=(
                TraceFrame(
                    frame_id=0,
                    phase="commit",
                    event=event,
                    effects=(effect,),
                    notes=("piece moved",),
                ),
            ),
            metadata={"engine_version": "phase0"},
        )

        round_tripped = TransitionTrace.from_dict(trace.to_dict())

        self.assertEqual(trace.to_dict(), round_tripped.to_dict())

    def test_state_delta_round_trip_with_new_runtime_fields(self) -> None:
        move = Move(piece_id="white_rook", from_square="a1", to_square="a4")
        action = ActionIntent(
            action_kind="move",
            actor_piece_id="white_rook",
            from_square="a1",
            to_square="a4",
        )
        effect = TransitionEffect(
            effect_kind="move_piece",
            actor_piece_id="white_rook",
            target_square="a4",
        )
        trace = TransitionTrace(
            action_id=action.stable_id(),
            frames=(
                TraceFrame(
                    frame_id=0,
                    phase="commit",
                    effects=(effect,),
                ),
            ),
        )
        delta = StateDelta(
            move=move,
            action=action,
            events=(Event(event_type="move_committed"),),
            effects=(effect,),
            changed_piece_ids=("white_rook",),
            created_piece_ids=(),
            removed_piece_ids=("black_pawn",),
            notes=("ok",),
            trace=trace,
            metadata={"before_state_hash": "before", "after_state_hash": "after"},
        )

        round_tripped = StateDelta.from_dict(delta.to_dict())

        self.assertEqual(delta.to_dict(), round_tripped.to_dict())

    def test_state_delta_from_legacy_payload_defaults_new_fields(self) -> None:
        payload = {
            "move": {
                "piece_id": "white_rook",
                "from_square": "a1",
                "to_square": "a4",
                "move_kind": "move",
                "captured_piece_id": None,
                "promotion_piece_type": None,
                "tags": [],
                "metadata": {},
            },
            "events": [{"event_type": "move_committed"}],
            "changed_piece_ids": ["white_rook"],
            "notes": ["legacy"],
            "metadata": {"before_state_hash": "before", "after_state_hash": "after"},
        }

        delta = StateDelta.from_dict(payload)

        self.assertIsNone(delta.action)
        self.assertEqual((), delta.effects)
        self.assertEqual((), delta.created_piece_ids)
        self.assertEqual((), delta.removed_piece_ids)
        self.assertIsNone(delta.trace)

    def test_threat_mark_round_trip_and_stable_id(self) -> None:
        threat = ThreatMark(
            threat_kind="capture_window",
            target_square="E4",
            source_entity_id="white_rook",
            tags=("line",),
            params={"ray_length": 4},
            metadata={"query": "threat_logic"},
        )

        round_tripped = ThreatMark.from_dict(threat.to_dict())

        self.assertEqual(threat.to_dict(), round_tripped.to_dict())
        self.assertEqual(stable_threat_id(threat), stable_threat_id(round_tripped))

    def test_query_fact_round_trip_and_stable_id(self) -> None:
        query_fact = QueryFact(
            query_kind="control_relation",
            subject_ref="white_rook",
            target_ref="square:e4",
            tags=("line",),
            params={"distance": 4},
            metadata={"query_program": "control_query"},
        )

        round_tripped = QueryFact.from_dict(query_fact.to_dict())

        self.assertEqual(query_fact.to_dict(), round_tripped.to_dict())
        self.assertEqual(
            stable_query_fact_id(query_fact),
            stable_query_fact_id(round_tripped),
        )

    def test_threat_mark_converts_to_and_from_query_fact(self) -> None:
        threat = ThreatMark(
            threat_kind="capture_window",
            target_square="e4",
            source_entity_id="white_rook",
            tags=("line",),
            params={"distance": 4},
            metadata={"query_program": "control_query"},
        )

        query_fact = threat.to_query_fact()
        round_tripped = ThreatMark.from_query_fact(query_fact)

        self.assertEqual("capture_window", query_fact.query_kind)
        self.assertEqual("white_rook", query_fact.subject_ref)
        self.assertEqual("e4", query_fact.target_ref)
        self.assertEqual(threat.to_dict(), round_tripped.to_dict())


if __name__ == "__main__":
    unittest.main()
