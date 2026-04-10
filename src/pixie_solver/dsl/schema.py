from __future__ import annotations

from pixie_solver.core.piece import BasePieceType

SCHEMA_VERSION = 1

REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "piece_id",
        "name",
        "base_piece_type",
        "instance_state_schema",
        "movement_modifiers",
        "capture_modifiers",
        "hooks",
    }
)

OPTIONAL_TOP_LEVEL_FIELDS = frozenset({"metadata"})
TOP_LEVEL_FIELDS = REQUIRED_TOP_LEVEL_FIELDS | OPTIONAL_TOP_LEVEL_FIELDS

BASE_PIECE_TYPES = frozenset(piece_type.value for piece_type in BasePieceType)
PIECE_REFS = frozenset({"self", "event_source", "event_target"})
EDGE_BEHAVIORS = frozenset({"block", "remove_if_pushed_off_board"})

MOVEMENT_MODIFIERS = frozenset(
    {
        "inherit_base",
        "phase_through_allies",
        "extend_range",
        "limit_range",
    }
)

CAPTURE_MODIFIERS = frozenset(
    {
        "inherit_base",
        "replace_capture_with_push",
    }
)

HOOK_EVENTS = frozenset(
    {
        "move_committed",
        "piece_captured",
        "turn_start",
        "turn_end",
    }
)

CONDITION_OPERATORS = frozenset(
    {
        "self_on_board",
        "square_empty",
        "square_occupied",
        "state_field_eq",
        "state_field_gte",
        "state_field_lte",
        "piece_color_is",
    }
)

EFFECT_OPERATORS = frozenset(
    {
        "move_piece",
        "capture_piece",
        "set_state",
        "increment_state",
        "emit_event",
    }
)

STATE_FIELD_TYPES = frozenset({"bool", "int", "float", "str"})
