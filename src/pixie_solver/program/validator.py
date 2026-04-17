from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pixie_solver.dsl.validator import collect_validation_errors
from pixie_solver.program.schema import (
    ACTION_BLOCK_FIELDS,
    ACTION_BLOCK_KINDS,
    ACTION_BLOCK_REQUIRED_FIELDS,
    BASE_ARCHETYPES,
    CONDITION_FIELDS,
    EFFECT_FIELDS,
    PROGRAM_REQUIRED_FIELDS,
    PROGRAM_SCHEMA_VERSION,
    PROGRAM_TOP_LEVEL_FIELDS,
    QUERY_BLOCK_FIELDS,
    QUERY_BLOCK_KINDS,
    QUERY_BLOCK_REQUIRED_FIELDS,
    REACTION_BLOCK_FIELDS,
    REACTION_BLOCK_KINDS,
    REACTION_BLOCK_REQUIRED_FIELDS,
    TRIGGER_FIELDS,
    TRIGGER_REQUIRED_FIELDS,
)

IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class ProgramValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def _is_sequence_of_mappings(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(
        isinstance(item, Mapping) for item in value
    )


def collect_program_validation_errors(program: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []

    missing_fields = sorted(PROGRAM_REQUIRED_FIELDS - set(program))
    if missing_fields:
        errors.append(f"Missing required program fields: {', '.join(missing_fields)}")

    unknown_fields = sorted(set(program) - PROGRAM_TOP_LEVEL_FIELDS)
    if unknown_fields:
        errors.append(
            "Unknown top-level fields are not allowed in ProgramIR: "
            + ", ".join(unknown_fields)
        )

    program_id = program.get("program_id")
    if not isinstance(program_id, str) or not IDENTIFIER_PATTERN.fullmatch(program_id):
        errors.append("program_id must be a lowercase identifier")

    name = program.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("name must be a non-empty string")

    if program.get("schema_version") != PROGRAM_SCHEMA_VERSION:
        errors.append(f"schema_version must equal {PROGRAM_SCHEMA_VERSION}")

    base_archetype = program.get("base_archetype")
    if base_archetype not in BASE_ARCHETYPES:
        errors.append(f"base_archetype must be one of {sorted(BASE_ARCHETYPES)!r}")

    constants = program.get("constants", {})
    if not isinstance(constants, Mapping):
        errors.append("constants must be a mapping")

    metadata = program.get("metadata", {})
    if not isinstance(metadata, Mapping):
        errors.append("metadata must be a mapping when present")

    state_schema = program.get("state_schema")
    synthetic_piece = {
        "piece_id": "validation_piece",
        "name": "Validation Piece",
        "base_piece_type": base_archetype if base_archetype in BASE_ARCHETYPES else "pawn",
        "instance_state_schema": state_schema,
        "movement_modifiers": [{"op": "inherit_base", "args": {}}],
        "capture_modifiers": [{"op": "inherit_base", "args": {}}],
        "hooks": [],
    }
    for error in collect_validation_errors(synthetic_piece):
        if error.startswith("instance_state_schema"):
            errors.append(f"state_schema: {error}")

    action_blocks = program.get("action_blocks")
    if not _is_sequence_of_mappings(action_blocks):
        errors.append("action_blocks must be a list of mappings")
    else:
        for index, block in enumerate(action_blocks):
            _validate_action_block(errors, block=block, index=index)

    query_blocks = program.get("query_blocks", [])
    if not _is_sequence_of_mappings(query_blocks):
        errors.append("query_blocks must be a list of mappings")
    else:
        for index, block in enumerate(query_blocks):
            _validate_query_block(errors, block=block, index=index)

    reaction_blocks = program.get("reaction_blocks")
    if not _is_sequence_of_mappings(reaction_blocks):
        errors.append("reaction_blocks must be a list of mappings")
    else:
        for index, block in enumerate(reaction_blocks):
            _validate_reaction_block(
                errors,
                block=block,
                index=index,
                base_archetype=(
                    base_archetype if base_archetype in BASE_ARCHETYPES else "pawn"
                ),
                state_schema=state_schema if _is_sequence_of_mappings(state_schema) else [],
            )
    all_blocks: list[Mapping[str, Any]] = []
    if _is_sequence_of_mappings(action_blocks):
        all_blocks.extend(action_blocks)
    if _is_sequence_of_mappings(query_blocks):
        all_blocks.extend(query_blocks)
    if _is_sequence_of_mappings(reaction_blocks):
        all_blocks.extend(reaction_blocks)
    block_ids = [block.get("block_id") for block in all_blocks]
    duplicate_block_ids = sorted(
        {
            str(block_id)
            for block_id in block_ids
            if isinstance(block_id, str) and block_ids.count(block_id) > 1
        }
    )
    if duplicate_block_ids:
        errors.append(
            "block_id values must be unique across action_blocks, query_blocks, and reaction_blocks: "
            + ", ".join(duplicate_block_ids)
        )

    return errors


def _validate_action_block(
    errors: list[str],
    *,
    block: Mapping[str, Any],
    index: int,
) -> None:
    missing_fields = sorted(ACTION_BLOCK_REQUIRED_FIELDS - set(block))
    if missing_fields:
        errors.append(
            f"action_blocks[{index}] missing required fields: {', '.join(missing_fields)}"
        )
    unknown_fields = sorted(set(block) - ACTION_BLOCK_FIELDS)
    if unknown_fields:
        errors.append(
            f"action_blocks[{index}] contains unknown fields: {', '.join(unknown_fields)}"
        )

    block_id = block.get("block_id")
    if not isinstance(block_id, str) or not IDENTIFIER_PATTERN.fullmatch(block_id):
        errors.append(f"action_blocks[{index}].block_id must be a lowercase identifier")

    kind = block.get("kind")
    if kind not in ACTION_BLOCK_KINDS:
        errors.append(
            f"action_blocks[{index}].kind must be one of {sorted(ACTION_BLOCK_KINDS)!r}"
        )

    params = block.get("params", {})
    if not isinstance(params, Mapping):
        errors.append(f"action_blocks[{index}].params must be a mapping")
        return

    metadata = block.get("metadata", {})
    if not isinstance(metadata, Mapping):
        errors.append(f"action_blocks[{index}].metadata must be a mapping")

    if kind == "legacy_base_actions":
        _validate_legacy_piece_params(errors, params=params, prefix=f"action_blocks[{index}]")


def _validate_query_block(
    errors: list[str],
    *,
    block: Mapping[str, Any],
    index: int,
) -> None:
    missing_fields = sorted(QUERY_BLOCK_REQUIRED_FIELDS - set(block))
    if missing_fields:
        errors.append(
            f"query_blocks[{index}] missing required fields: {', '.join(missing_fields)}"
        )
    unknown_fields = sorted(set(block) - QUERY_BLOCK_FIELDS)
    if unknown_fields:
        errors.append(
            f"query_blocks[{index}] contains unknown fields: {', '.join(unknown_fields)}"
        )

    block_id = block.get("block_id")
    if not isinstance(block_id, str) or not IDENTIFIER_PATTERN.fullmatch(block_id):
        errors.append(f"query_blocks[{index}].block_id must be a lowercase identifier")

    kind = block.get("kind")
    if kind not in QUERY_BLOCK_KINDS:
        errors.append(
            f"query_blocks[{index}].kind must be one of {sorted(QUERY_BLOCK_KINDS)!r}"
        )

    params = block.get("params", {})
    if not isinstance(params, Mapping):
        errors.append(f"query_blocks[{index}].params must be a mapping")
        return

    metadata = block.get("metadata", {})
    if not isinstance(metadata, Mapping):
        errors.append(f"query_blocks[{index}].metadata must be a mapping")

    if kind == "legacy_capture_control":
        _validate_legacy_piece_params(errors, params=params, prefix=f"query_blocks[{index}]")


def _validate_legacy_piece_params(
    errors: list[str],
    *,
    params: Mapping[str, Any],
    prefix: str,
) -> None:
    exact_keys = {"base_archetype", "movement_modifiers", "capture_modifiers"}
    unknown_param_keys = sorted(set(params) - exact_keys)
    missing_param_keys = sorted(exact_keys - set(params))
    if missing_param_keys:
        errors.append(f"{prefix}.params missing required fields: {', '.join(missing_param_keys)}")
    if unknown_param_keys:
        errors.append(
            f"{prefix}.params contains unknown fields: " + ", ".join(unknown_param_keys)
        )
    synthetic_piece = {
        "piece_id": "validation_piece",
        "name": "Validation Piece",
        "base_piece_type": params.get("base_archetype", "pawn"),
        "instance_state_schema": [],
        "movement_modifiers": params.get("movement_modifiers", []),
        "capture_modifiers": params.get("capture_modifiers", []),
        "hooks": [],
    }
    for error in collect_validation_errors(synthetic_piece):
        if (
            error.startswith("base_piece_type")
            or error.startswith("movement_modifiers")
            or error.startswith("capture_modifiers")
        ):
            errors.append(f"{prefix}: {error}")


def _validate_reaction_block(
    errors: list[str],
    *,
    block: Mapping[str, Any],
    index: int,
    base_archetype: str,
    state_schema: Sequence[Mapping[str, Any]],
) -> None:
    missing_fields = sorted(REACTION_BLOCK_REQUIRED_FIELDS - set(block))
    if missing_fields:
        errors.append(
            "reaction_blocks"
            f"[{index}] missing required fields: {', '.join(missing_fields)}"
        )
    unknown_fields = sorted(set(block) - REACTION_BLOCK_FIELDS)
    if unknown_fields:
        errors.append(
            f"reaction_blocks[{index}] contains unknown fields: {', '.join(unknown_fields)}"
        )

    block_id = block.get("block_id")
    if not isinstance(block_id, str) or not IDENTIFIER_PATTERN.fullmatch(block_id):
        errors.append(f"reaction_blocks[{index}].block_id must be a lowercase identifier")

    kind = block.get("kind")
    if kind not in REACTION_BLOCK_KINDS:
        errors.append(
            f"reaction_blocks[{index}].kind must be one of {sorted(REACTION_BLOCK_KINDS)!r}"
        )

    trigger = block.get("trigger", {})
    if not isinstance(trigger, Mapping):
        errors.append(f"reaction_blocks[{index}].trigger must be a mapping")
    else:
        missing_trigger_fields = sorted(TRIGGER_REQUIRED_FIELDS - set(trigger))
        if missing_trigger_fields:
            errors.append(
                "reaction_blocks"
                f"[{index}].trigger missing required fields: {', '.join(missing_trigger_fields)}"
            )
        unknown_trigger_fields = sorted(set(trigger) - TRIGGER_FIELDS)
        if unknown_trigger_fields:
            errors.append(
                f"reaction_blocks[{index}].trigger contains unknown fields: "
                + ", ".join(unknown_trigger_fields)
            )

    conditions = block.get("conditions")
    if not _is_sequence_of_mappings(conditions):
        errors.append(f"reaction_blocks[{index}].conditions must be a list of mappings")
        conditions = []
    effects = block.get("effects")
    if not _is_sequence_of_mappings(effects):
        errors.append(f"reaction_blocks[{index}].effects must be a list of mappings")
        effects = []

    for condition_index, condition in enumerate(conditions):
        missing_condition_fields = sorted(CONDITION_FIELDS - set(condition))
        if missing_condition_fields:
            errors.append(
                "reaction_blocks"
                f"[{index}].conditions[{condition_index}] missing required fields: "
                + ", ".join(missing_condition_fields)
            )
        unknown_condition_fields = sorted(set(condition) - CONDITION_FIELDS)
        if unknown_condition_fields:
            errors.append(
                "reaction_blocks"
                f"[{index}].conditions[{condition_index}] contains unknown fields: "
                + ", ".join(unknown_condition_fields)
            )
        if not isinstance(condition.get("args", {}), Mapping):
            errors.append(
                f"reaction_blocks[{index}].conditions[{condition_index}].args must be a mapping"
            )

    for effect_index, effect in enumerate(effects):
        missing_effect_fields = sorted(EFFECT_FIELDS - set(effect))
        if missing_effect_fields:
            errors.append(
                "reaction_blocks"
                f"[{index}].effects[{effect_index}] missing required fields: "
                + ", ".join(missing_effect_fields)
            )
        unknown_effect_fields = sorted(set(effect) - EFFECT_FIELDS)
        if unknown_effect_fields:
            errors.append(
                "reaction_blocks"
                f"[{index}].effects[{effect_index}] contains unknown fields: "
                + ", ".join(unknown_effect_fields)
            )
        if not isinstance(effect.get("args", {}), Mapping):
            errors.append(
                f"reaction_blocks[{index}].effects[{effect_index}].args must be a mapping"
            )

    priority = block.get("priority", 0)
    if not isinstance(priority, int) or isinstance(priority, bool):
        errors.append(f"reaction_blocks[{index}].priority must be an integer")
    metadata = block.get("metadata", {})
    if not isinstance(metadata, Mapping):
        errors.append(f"reaction_blocks[{index}].metadata must be a mapping")

    synthetic_hook = {
        "event": trigger.get("event_type"),
        "conditions": [
            {
                "op": condition.get("condition_kind"),
                "args": dict(condition.get("args", {})),
            }
            for condition in conditions
            if isinstance(condition, Mapping)
        ],
        "effects": [
            {
                "op": effect.get("effect_kind"),
                "args": dict(effect.get("args", {})),
            }
            for effect in effects
            if isinstance(effect, Mapping)
        ],
        "priority": priority,
        "metadata": metadata if isinstance(metadata, Mapping) else {},
    }
    synthetic_piece = {
        "piece_id": "validation_piece",
        "name": "Validation Piece",
        "base_piece_type": base_archetype,
        "instance_state_schema": list(state_schema),
        "movement_modifiers": [{"op": "inherit_base", "args": {}}],
        "capture_modifiers": [{"op": "inherit_base", "args": {}}],
        "hooks": [synthetic_hook],
    }
    for error in collect_validation_errors(synthetic_piece):
        if error.startswith("hooks[0]"):
            errors.append(
                f"reaction_blocks[{index}]{error.removeprefix('hooks[0]')}"
            )


def validate_program_ir(program: Mapping[str, Any]) -> None:
    errors = collect_program_validation_errors(program)
    if errors:
        raise ProgramValidationError(errors)
