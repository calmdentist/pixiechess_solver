from __future__ import annotations

from pixie_solver.core.piece import BasePieceType
from pixie_solver.dsl.schema import CONDITION_OPERATORS, EFFECT_OPERATORS

PROGRAM_SCHEMA_VERSION = 1

PROGRAM_REQUIRED_FIELDS = frozenset(
    {
        "program_id",
        "name",
        "schema_version",
        "base_archetype",
        "state_schema",
        "constants",
        "action_blocks",
        "reaction_blocks",
    }
)
PROGRAM_OPTIONAL_FIELDS = frozenset({"metadata", "query_blocks"})
PROGRAM_TOP_LEVEL_FIELDS = PROGRAM_REQUIRED_FIELDS | PROGRAM_OPTIONAL_FIELDS

ACTION_BLOCK_REQUIRED_FIELDS = frozenset({"block_id", "kind", "params"})
ACTION_BLOCK_OPTIONAL_FIELDS = frozenset({"metadata"})
ACTION_BLOCK_FIELDS = ACTION_BLOCK_REQUIRED_FIELDS | ACTION_BLOCK_OPTIONAL_FIELDS

QUERY_BLOCK_REQUIRED_FIELDS = frozenset({"block_id", "kind", "params"})
QUERY_BLOCK_OPTIONAL_FIELDS = frozenset({"metadata"})
QUERY_BLOCK_FIELDS = QUERY_BLOCK_REQUIRED_FIELDS | QUERY_BLOCK_OPTIONAL_FIELDS

REACTION_BLOCK_REQUIRED_FIELDS = frozenset(
    {"block_id", "kind", "trigger", "conditions", "effects"}
)
REACTION_BLOCK_OPTIONAL_FIELDS = frozenset({"priority", "metadata"})
REACTION_BLOCK_FIELDS = REACTION_BLOCK_REQUIRED_FIELDS | REACTION_BLOCK_OPTIONAL_FIELDS

TRIGGER_REQUIRED_FIELDS = frozenset({"event_type"})
TRIGGER_OPTIONAL_FIELDS = frozenset({"payload_filter"})
TRIGGER_FIELDS = TRIGGER_REQUIRED_FIELDS | TRIGGER_OPTIONAL_FIELDS

CONDITION_REQUIRED_FIELDS = frozenset({"condition_kind", "args"})
CONDITION_FIELDS = CONDITION_REQUIRED_FIELDS

EFFECT_REQUIRED_FIELDS = frozenset({"effect_kind", "args"})
EFFECT_FIELDS = EFFECT_REQUIRED_FIELDS

ACTION_BLOCK_KINDS = frozenset({"legacy_base_actions"})
QUERY_BLOCK_KINDS = frozenset({"legacy_capture_control"})
REACTION_BLOCK_KINDS = frozenset({"event_reaction"})

BASE_ARCHETYPES = frozenset(piece_type.value for piece_type in BasePieceType)
REACTION_CONDITION_KINDS = CONDITION_OPERATORS
REACTION_EFFECT_KINDS = EFFECT_OPERATORS
