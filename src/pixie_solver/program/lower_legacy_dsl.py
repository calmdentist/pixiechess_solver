from __future__ import annotations

from typing import Any, Mapping

from pixie_solver.core.piece import PieceClass
from pixie_solver.dsl.canonicalize import canonicalize_piece_program
from pixie_solver.dsl.schema import SCHEMA_VERSION as LEGACY_DSL_SCHEMA_VERSION
from pixie_solver.program.canonicalize import canonicalize_program_ir
from pixie_solver.program.schema import PROGRAM_SCHEMA_VERSION


def lower_legacy_piece_program(program: Mapping[str, Any]) -> dict[str, Any]:
    canonical_piece = canonicalize_piece_program(program)
    return canonicalize_program_ir(
        {
            "program_id": str(canonical_piece["piece_id"]),
            "name": str(canonical_piece["name"]),
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "base_archetype": str(canonical_piece["base_piece_type"]),
            "state_schema": list(canonical_piece["instance_state_schema"]),
            "constants": {},
            "action_blocks": [
                {
                    "block_id": "legacy_base_actions",
                    "kind": "legacy_base_actions",
                    "params": {
                        "base_archetype": str(canonical_piece["base_piece_type"]),
                        "movement_modifiers": list(canonical_piece["movement_modifiers"]),
                        "capture_modifiers": list(canonical_piece["capture_modifiers"]),
                    },
                    "metadata": {"source": "dsl_v1"},
                }
            ],
            "query_blocks": [
                {
                    "block_id": "legacy_capture_control",
                    "kind": "legacy_capture_control",
                    "params": {
                        "base_archetype": str(canonical_piece["base_piece_type"]),
                        "movement_modifiers": list(canonical_piece["movement_modifiers"]),
                        "capture_modifiers": list(canonical_piece["capture_modifiers"]),
                    },
                    "metadata": {"source": "dsl_v1"},
                }
            ],
            "reaction_blocks": [
                {
                    "block_id": f"hook_{index:03d}_{hook['event']}",
                    "kind": "event_reaction",
                    "trigger": {"event_type": str(hook["event"])},
                    "conditions": [
                        {
                            "condition_kind": str(condition["op"]),
                            "args": dict(condition.get("args", {})),
                        }
                        for condition in hook.get("conditions", [])
                    ],
                    "effects": [
                        {
                            "effect_kind": str(effect["op"]),
                            "args": dict(effect.get("args", {})),
                        }
                        for effect in hook.get("effects", [])
                    ],
                    "priority": int(hook.get("priority", 0)),
                    "metadata": {
                        "source": "dsl_v1",
                        "source_hook_index": index,
                        **dict(hook.get("metadata", {})),
                    },
                }
                for index, hook in enumerate(canonical_piece.get("hooks", []))
            ],
            "metadata": {
                **dict(canonical_piece.get("metadata", {})),
                "source_format": "dsl_v1",
                "source_schema_version": LEGACY_DSL_SCHEMA_VERSION,
            },
        }
    )


def lower_legacy_piece_class(piece_class: PieceClass) -> dict[str, Any]:
    return canonicalize_program_ir(
        {
            "program_id": piece_class.class_id,
            "name": piece_class.name,
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "base_archetype": piece_class.base_piece_type.value,
            "state_schema": [
                {
                    "name": field_spec.name,
                    "type": field_spec.field_type,
                    "default": field_spec.default,
                    "description": field_spec.description,
                }
                for field_spec in piece_class.instance_state_schema
            ],
            "constants": {},
            "action_blocks": [
                {
                    "block_id": "legacy_base_actions",
                    "kind": "legacy_base_actions",
                    "params": {
                        "base_archetype": piece_class.base_piece_type.value,
                        "movement_modifiers": [
                            _piece_class_modifier_to_dict(modifier)
                            for modifier in piece_class.movement_modifiers
                        ],
                        "capture_modifiers": [
                            _piece_class_modifier_to_dict(modifier)
                            for modifier in piece_class.capture_modifiers
                        ],
                    },
                    "metadata": {"source": "piece_class_legacy"},
                }
            ],
            "query_blocks": [
                {
                    "block_id": "legacy_capture_control",
                    "kind": "legacy_capture_control",
                    "params": {
                        "base_archetype": piece_class.base_piece_type.value,
                        "movement_modifiers": [
                            _piece_class_modifier_to_dict(modifier)
                            for modifier in piece_class.movement_modifiers
                        ],
                        "capture_modifiers": [
                            _piece_class_modifier_to_dict(modifier)
                            for modifier in piece_class.capture_modifiers
                        ],
                    },
                    "metadata": {"source": "piece_class_legacy"},
                }
            ],
            "reaction_blocks": [
                {
                    "block_id": f"hook_{index:03d}_{hook.event}",
                    "kind": "event_reaction",
                    "trigger": {"event_type": hook.event},
                    "conditions": [
                        {
                            "condition_kind": condition.op,
                            "args": dict(condition.args),
                        }
                        for condition in hook.conditions
                    ],
                    "effects": [
                        {
                            "effect_kind": effect.op,
                            "args": dict(effect.args),
                        }
                        for effect in hook.effects
                    ],
                    "priority": hook.priority,
                    "metadata": {
                        "source": "piece_class_legacy",
                        "source_hook_index": index,
                        **dict(hook.metadata),
                    },
                }
                for index, hook in enumerate(piece_class.hooks)
            ],
            "metadata": {
                **dict(piece_class.metadata),
                "source_format": "piece_class_legacy",
                "source_schema_version": LEGACY_DSL_SCHEMA_VERSION,
            },
        }
    )


def _piece_class_modifier_to_dict(modifier: Any) -> dict[str, Any]:
    serialized = modifier.to_dict()
    if (
        serialized["op"] == "replace_capture_with_push"
        and serialized["args"].get("edge_behavior") == "fail"
    ):
        serialized["args"] = {
            **dict(serialized["args"]),
            "edge_behavior": "block",
        }
    return serialized
