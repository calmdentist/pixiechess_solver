from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from pixie_solver.dsl.validator import validate_piece_program
from pixie_solver.utils.squares import normalize_square


def canonicalize_piece_program(program: Mapping[str, Any]) -> dict[str, Any]:
    validate_piece_program(program)
    return {
        "piece_id": str(program["piece_id"]),
        "name": str(program["name"]).strip(),
        "base_piece_type": str(program["base_piece_type"]),
        "instance_state_schema": [
            {
                "name": str(field_spec["name"]),
                "type": str(field_spec["type"]),
                "default": deepcopy(field_spec["default"]),
                "description": str(field_spec.get("description", "")),
            }
            for field_spec in program.get("instance_state_schema", [])
        ],
        "movement_modifiers": [
            {
                "op": str(modifier["op"]),
                "args": deepcopy(dict(modifier.get("args", {}))),
            }
            for modifier in program.get("movement_modifiers", [])
        ],
        "capture_modifiers": [
            {
                "op": str(modifier["op"]),
                "args": deepcopy(dict(modifier.get("args", {}))),
            }
            for modifier in program.get("capture_modifiers", [])
        ],
        "hooks": [_canonicalize_hook(hook) for hook in program.get("hooks", [])],
        "metadata": deepcopy(dict(program.get("metadata", {}))),
    }


def _canonicalize_hook(hook: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event": str(hook["event"]),
        "conditions": [
            _canonicalize_condition(condition)
            for condition in hook.get("conditions", [])
        ],
        "effects": [_canonicalize_effect(effect) for effect in hook.get("effects", [])],
        "priority": int(hook.get("priority", 0)),
        "metadata": deepcopy(dict(hook.get("metadata", {}))),
    }


def _canonicalize_condition(condition: Mapping[str, Any]) -> dict[str, Any]:
    args = deepcopy(dict(condition.get("args", {})))
    if condition["op"] in {"square_empty", "square_occupied"}:
        args["square"] = _canonicalize_square_ref(args["square"])
    return {"op": str(condition["op"]), "args": args}


def _canonicalize_effect(effect: Mapping[str, Any]) -> dict[str, Any]:
    args = deepcopy(dict(effect.get("args", {})))
    if effect["op"] == "move_piece":
        args["to"] = _canonicalize_square_ref(args["to"])
    return {"op": str(effect["op"]), "args": args}


def _canonicalize_square_ref(square_ref: Mapping[str, Any]) -> dict[str, Any]:
    if "absolute" in square_ref:
        return {"absolute": normalize_square(str(square_ref["absolute"]))}
    return {
        "relative_to": str(square_ref["relative_to"]),
        "offset": [int(square_ref["offset"][0]), int(square_ref["offset"][1])],
    }
