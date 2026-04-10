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
    Effect,
    Event,
    GameState,
    Hook,
    Modifier,
    PieceClass,
    PieceInstance,
    StateField,
    stable_state_hash,
)


class GameStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rook_class = PieceClass(
            class_id="phasing_rook",
            name="Phasing Rook",
            base_piece_type=BasePieceType.ROOK,
            movement_modifiers=(
                Modifier(op="inherit_base"),
                Modifier(op="phase_through_allies"),
            ),
            capture_modifiers=(Modifier(op="inherit_base"),),
        )
        self.king_class = PieceClass(
            class_id="baseline_king",
            name="Baseline King",
            base_piece_type=BasePieceType.KING,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
        )

    def test_round_trip_and_hash_are_stable(self) -> None:
        state = GameState(
            piece_classes={
                self.rook_class.class_id: self.rook_class,
                self.king_class.class_id: self.king_class,
            },
            piece_instances={
                "white_rook_1": PieceInstance(
                    instance_id="white_rook_1",
                    piece_class_id="phasing_rook",
                    color=Color.WHITE,
                    square="a1",
                ),
                "black_king_1": PieceInstance(
                    instance_id="black_king_1",
                    piece_class_id="baseline_king",
                    color=Color.BLACK,
                    square="h8",
                ),
            },
            side_to_move=Color.WHITE,
            pending_events=(Event(event_type="turn_start"),),
            metadata={"variant": "pixiechess"},
        )

        round_tripped = GameState.from_dict(state.to_dict())
        self.assertEqual(state.to_dict(), round_tripped.to_dict())
        self.assertEqual(stable_state_hash(state), stable_state_hash(round_tripped))
        self.assertEqual("white_rook_1", state.occupancy()["a1"])
        self.assertEqual("h8", state.piece_on("h8").square)

    def test_duplicate_occupancy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GameState(
                piece_classes={
                    self.rook_class.class_id: self.rook_class,
                    self.king_class.class_id: self.king_class,
                },
                piece_instances={
                    "white_rook_1": PieceInstance(
                        instance_id="white_rook_1",
                        piece_class_id="phasing_rook",
                        color=Color.WHITE,
                        square="a1",
                    ),
                    "black_king_1": PieceInstance(
                        instance_id="black_king_1",
                        piece_class_id="baseline_king",
                        color=Color.BLACK,
                        square="a1",
                    ),
                },
            )

    def test_state_defaults_are_applied_from_piece_schema(self) -> None:
        charged_piece = PieceClass(
            class_id="charged_pawn",
            name="Charged Pawn",
            base_piece_type=BasePieceType.PAWN,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
            hooks=(
                Hook(
                    event="turn_start",
                    effects=(Effect(op="increment_state", args={"piece": "self", "name": "charges", "amount": 1}),),
                ),
            ),
            instance_state_schema=(
                StateField(name="charges", field_type="int", default=0),
                StateField(name="mode", field_type="str", default="idle"),
            ),
        )
        state = GameState(
            piece_classes={charged_piece.class_id: charged_piece},
            piece_instances={
                "charged_1": PieceInstance(
                    instance_id="charged_1",
                    piece_class_id=charged_piece.class_id,
                    color=Color.WHITE,
                    square="a2",
                )
            },
        )
        self.assertEqual(
            {"charges": 0, "mode": "idle"},
            state.piece_instances["charged_1"].state,
        )

    def test_invalid_piece_state_type_is_rejected(self) -> None:
        charged_piece = PieceClass(
            class_id="charged_pawn",
            name="Charged Pawn",
            base_piece_type=BasePieceType.PAWN,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
            instance_state_schema=(StateField(name="charges", field_type="int", default=0),),
        )
        with self.assertRaises(ValueError):
            GameState(
                piece_classes={charged_piece.class_id: charged_piece},
                piece_instances={
                    "charged_1": PieceInstance(
                        instance_id="charged_1",
                        piece_class_id=charged_piece.class_id,
                        color=Color.WHITE,
                        square="a2",
                        state={"charges": "full"},
                    )
                },
            )


if __name__ == "__main__":
    unittest.main()
