from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from pixie_solver.dsl.canonicalize import canonicalize_piece_program
from pixie_solver.program.validator import validate_program_ir


def canonicalize_program_ir(program: Mapping[str, Any]) -> dict[str, Any]:
    validate_program_ir(program)
    return {
        "program_id": str(program["program_id"]),
        "name": str(program["name"]).strip(),
        "schema_version": int(program["schema_version"]),
        "base_archetype": str(program["base_archetype"]),
        "state_schema": [
            {
                "name": str(field_spec["name"]),
                "type": str(field_spec["type"]),
                "default": deepcopy(field_spec["default"]),
                "description": str(field_spec.get("description", "")),
            }
            for field_spec in program.get("state_schema", [])
        ],
        "constants": deepcopy(dict(program.get("constants", {}))),
        "action_blocks": [
            _canonicalize_action_block(block)
            for block in program.get("action_blocks", [])
        ],
        "query_blocks": [
            _canonicalize_query_block(block)
            for block in program.get("query_blocks", [])
        ],
        "reaction_blocks": [
            _canonicalize_reaction_block(
                block,
                base_archetype=str(program["base_archetype"]),
                state_schema=program.get("state_schema", []),
            )
            for block in program.get("reaction_blocks", [])
        ],
        "metadata": deepcopy(dict(program.get("metadata", {}))),
    }


def _canonicalize_action_block(block: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "block_id": str(block["block_id"]),
        "kind": str(block["kind"]),
        "params": _canonicalize_legacy_piece_params(dict(block["params"])),
        "metadata": deepcopy(dict(block.get("metadata", {}))),
    }


def _canonicalize_query_block(block: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "block_id": str(block["block_id"]),
        "kind": str(block["kind"]),
        "params": _canonicalize_legacy_piece_params(dict(block["params"])),
        "metadata": deepcopy(dict(block.get("metadata", {}))),
    }


def _canonicalize_legacy_piece_params(params: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(params)
    legacy_piece = canonicalize_piece_program(
        {
            "piece_id": "canonicalize_piece",
            "name": "Canonicalize Piece",
            "base_piece_type": str(params["base_archetype"]),
            "instance_state_schema": [],
            "movement_modifiers": params.get("movement_modifiers", []),
            "capture_modifiers": params.get("capture_modifiers", []),
            "hooks": [],
        }
    )
    return {
        "base_archetype": str(params["base_archetype"]),
        "movement_modifiers": legacy_piece["movement_modifiers"],
        "capture_modifiers": legacy_piece["capture_modifiers"],
    }


def _canonicalize_reaction_block(
    block: Mapping[str, Any],
    *,
    base_archetype: str,
    state_schema: Any,
) -> dict[str, Any]:
    legacy_piece = canonicalize_piece_program(
        {
            "piece_id": "canonicalize_piece",
            "name": "Canonicalize Piece",
            "base_piece_type": base_archetype,
            "instance_state_schema": list(state_schema),
            "movement_modifiers": [{"op": "inherit_base", "args": {}}],
            "capture_modifiers": [{"op": "inherit_base", "args": {}}],
            "hooks": [
                {
                    "event": str(block["trigger"]["event_type"]),
                    "conditions": [
                        {
                            "op": str(condition["condition_kind"]),
                            "args": deepcopy(dict(condition.get("args", {}))),
                        }
                        for condition in block.get("conditions", [])
                    ],
                    "effects": [
                        {
                            "op": str(effect["effect_kind"]),
                            "args": deepcopy(dict(effect.get("args", {}))),
                        }
                        for effect in block.get("effects", [])
                    ],
                    "priority": int(block.get("priority", 0)),
                    "metadata": deepcopy(dict(block.get("metadata", {}))),
                }
            ],
        }
    )
    canonical_hook = legacy_piece["hooks"][0]
    return {
        "block_id": str(block["block_id"]),
        "kind": str(block["kind"]),
        "trigger": {"event_type": str(canonical_hook["event"])},
        "conditions": [
            {
                "condition_kind": str(condition["op"]),
                "args": deepcopy(dict(condition.get("args", {}))),
            }
            for condition in canonical_hook.get("conditions", [])
        ],
        "effects": [
            {
                "effect_kind": str(effect["op"]),
                "args": deepcopy(dict(effect.get("args", {}))),
            }
            for effect in canonical_hook.get("effects", [])
        ],
        "priority": int(canonical_hook.get("priority", 0)),
        "metadata": deepcopy(dict(canonical_hook.get("metadata", {}))),
    }
