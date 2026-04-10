from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pixie_solver.dsl.schema import (
    BASE_PIECE_TYPES,
    CAPTURE_MODIFIERS,
    CONDITION_OPERATORS,
    EDGE_BEHAVIORS,
    EFFECT_OPERATORS,
    HOOK_EVENTS,
    MOVEMENT_MODIFIERS,
    PIECE_REFS,
    REQUIRED_TOP_LEVEL_FIELDS,
    STATE_FIELD_TYPES,
    TOP_LEVEL_FIELDS,
)
from pixie_solver.utils.squares import is_valid_square

IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class PieceValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def _is_sequence_of_mappings(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(
        isinstance(item, Mapping) for item in value
    )


def collect_validation_errors(program: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []

    missing_fields = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(program))
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")

    unknown_fields = sorted(set(program) - TOP_LEVEL_FIELDS)
    if unknown_fields:
        errors.append(
            f"Unknown top-level fields are not allowed in DSL v1: {', '.join(unknown_fields)}"
        )

    piece_id = program.get("piece_id")
    if not isinstance(piece_id, str) or not IDENTIFIER_PATTERN.fullmatch(piece_id):
        errors.append("piece_id must be a lowercase identifier like phasing_rook")

    name = program.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("name must be a non-empty string")

    base_piece_type = program.get("base_piece_type")
    if base_piece_type not in BASE_PIECE_TYPES:
        errors.append(
            f"base_piece_type must be one of {sorted(BASE_PIECE_TYPES)!r}"
        )

    metadata = program.get("metadata", {})
    if not isinstance(metadata, Mapping):
        errors.append("metadata must be a mapping when present")

    schema_field_names = _validate_state_schema(errors, program.get("instance_state_schema"))
    _validate_movement_modifiers(errors, program.get("movement_modifiers"))
    _validate_capture_modifiers(errors, program.get("capture_modifiers"))
    _validate_hooks(errors, program.get("hooks"), schema_field_names)
    return errors


def _validate_modifier_shape(
    errors: list[str], *, label: str, value: Any, allowed_ops: frozenset[str]
) -> list[Mapping[str, Any]]:
    if not _is_sequence_of_mappings(value):
        errors.append(f"{label} must be a list of mappings")
        return []

    items: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        items.append(item)
        unknown_fields = sorted(set(item) - {"op", "args"})
        if unknown_fields:
            errors.append(
                f"{label}[{index}] contains unknown fields: {', '.join(unknown_fields)}"
            )
        op = item.get("op")
        if op not in allowed_ops:
            errors.append(
                f"{label}[{index}].op must be one of {sorted(allowed_ops)!r}, got {op!r}"
            )
        args = item.get("args", {})
        if not isinstance(args, Mapping):
            errors.append(f"{label}[{index}].args must be a mapping")
    return items


def _validate_state_schema(errors: list[str], value: Any) -> set[str]:
    if not _is_sequence_of_mappings(value):
        errors.append("instance_state_schema must be a list of mappings")
        return set()

    seen_names: set[str] = set()
    for index, item in enumerate(value):
        unknown_fields = sorted(set(item) - {"name", "type", "default", "description"})
        if unknown_fields:
            errors.append(
                f"instance_state_schema[{index}] contains unknown fields: "
                f"{', '.join(unknown_fields)}"
            )
        field_name = item.get("name")
        field_type = item.get("type")
        if not isinstance(field_name, str) or not IDENTIFIER_PATTERN.fullmatch(field_name):
            errors.append(
                f"instance_state_schema[{index}].name must be a lowercase identifier"
            )
        elif field_name in seen_names:
            errors.append(f"duplicate instance state field name {field_name!r}")
        else:
            seen_names.add(field_name)

        if field_type not in STATE_FIELD_TYPES:
            errors.append(
                f"instance_state_schema[{index}].type must be one of "
                f"{sorted(STATE_FIELD_TYPES)!r}"
            )
        if "default" not in item:
            errors.append(f"instance_state_schema[{index}].default is required")
        else:
            _validate_typed_scalar(
                errors,
                label=f"instance_state_schema[{index}].default",
                value=item.get("default"),
                field_type=field_type,
            )
        description = item.get("description", "")
        if not isinstance(description, str):
            errors.append(f"instance_state_schema[{index}].description must be a string")
    return seen_names


def _validate_movement_modifiers(errors: list[str], value: Any) -> None:
    modifiers = _validate_modifier_shape(
        errors,
        label="movement_modifiers",
        value=value,
        allowed_ops=MOVEMENT_MODIFIERS,
    )
    if not modifiers:
        return

    if modifiers[0].get("op") != "inherit_base":
        errors.append("movement_modifiers[0].op must be 'inherit_base' in DSL v1")

    for index, item in enumerate(modifiers):
        args = item.get("args", {})
        op = item.get("op")
        if not isinstance(args, Mapping):
            continue
        if op in {"inherit_base", "phase_through_allies"}:
            _validate_exact_keys(errors, f"movement_modifiers[{index}].args", args, set())
        elif op == "extend_range":
            _validate_exact_keys(
                errors,
                f"movement_modifiers[{index}].args",
                args,
                {"extra_steps"},
            )
            _validate_positive_int(
                errors,
                f"movement_modifiers[{index}].args.extra_steps",
                args.get("extra_steps"),
            )
        elif op == "limit_range":
            _validate_exact_keys(
                errors,
                f"movement_modifiers[{index}].args",
                args,
                {"max_steps"},
            )
            _validate_positive_int(
                errors,
                f"movement_modifiers[{index}].args.max_steps",
                args.get("max_steps"),
            )


def _validate_capture_modifiers(errors: list[str], value: Any) -> None:
    modifiers = _validate_modifier_shape(
        errors,
        label="capture_modifiers",
        value=value,
        allowed_ops=CAPTURE_MODIFIERS,
    )
    if not modifiers:
        return

    if len(modifiers) != 1:
        errors.append("capture_modifiers must contain exactly one modifier in DSL v1")

    for index, item in enumerate(modifiers):
        args = item.get("args", {})
        op = item.get("op")
        if not isinstance(args, Mapping):
            continue
        if op == "inherit_base":
            _validate_exact_keys(errors, f"capture_modifiers[{index}].args", args, set())
        elif op == "replace_capture_with_push":
            _validate_exact_keys(
                errors,
                f"capture_modifiers[{index}].args",
                args,
                {"distance", "edge_behavior"},
            )
            _validate_positive_int(
                errors,
                f"capture_modifiers[{index}].args.distance",
                args.get("distance"),
            )
            edge_behavior = args.get("edge_behavior")
            if edge_behavior not in EDGE_BEHAVIORS:
                errors.append(
                    "capture_modifiers"
                    f"[{index}].args.edge_behavior must be one of "
                    f"{sorted(EDGE_BEHAVIORS)!r}"
                )


def _validate_hooks(errors: list[str], value: Any, state_field_names: set[str]) -> None:
    if not _is_sequence_of_mappings(value):
        errors.append("hooks must be a list of mappings")
        return

    for index, hook in enumerate(value):
        unknown_fields = sorted(set(hook) - {"event", "conditions", "effects", "priority", "metadata"})
        if unknown_fields:
            errors.append(
                f"hooks[{index}] contains unknown fields: {', '.join(unknown_fields)}"
            )
        event = hook.get("event")
        if event not in HOOK_EVENTS:
            errors.append(
                f"hooks[{index}].event must be one of {sorted(HOOK_EVENTS)!r}"
            )
        priority = hook.get("priority", 0)
        if not isinstance(priority, int) or isinstance(priority, bool):
            errors.append(f"hooks[{index}].priority must be an integer")
        metadata = hook.get("metadata", {})
        if not isinstance(metadata, Mapping):
            errors.append(f"hooks[{index}].metadata must be a mapping")

        conditions = hook.get("conditions", [])
        if not _is_sequence_of_mappings(conditions):
            errors.append(f"hooks[{index}].conditions must be a list of mappings")
        else:
            for condition_index, condition in enumerate(conditions):
                unknown_fields = sorted(set(condition) - {"op", "args"})
                if unknown_fields:
                    errors.append(
                        f"hooks[{index}].conditions[{condition_index}] contains unknown fields: "
                        f"{', '.join(unknown_fields)}"
                    )
                op = condition.get("op")
                if op not in CONDITION_OPERATORS:
                    errors.append(
                        f"hooks[{index}].conditions[{condition_index}].op must be one of "
                        f"{sorted(CONDITION_OPERATORS)!r}"
                    )
                args = condition.get("args", {})
                if not isinstance(args, Mapping):
                    errors.append(
                        f"hooks[{index}].conditions[{condition_index}].args must be a mapping"
                    )
                    continue
                _validate_condition_args(
                    errors,
                    label=f"hooks[{index}].conditions[{condition_index}]",
                    op=op,
                    args=args,
                    state_field_names=state_field_names,
                )

        effects = hook.get("effects", [])
        if not _is_sequence_of_mappings(effects):
            errors.append(f"hooks[{index}].effects must be a list of mappings")
        else:
            if not effects:
                errors.append(f"hooks[{index}].effects must not be empty")
            for effect_index, effect in enumerate(effects):
                unknown_fields = sorted(set(effect) - {"op", "args"})
                if unknown_fields:
                    errors.append(
                        f"hooks[{index}].effects[{effect_index}] contains unknown fields: "
                        f"{', '.join(unknown_fields)}"
                    )
                op = effect.get("op")
                if op not in EFFECT_OPERATORS:
                    errors.append(
                        f"hooks[{index}].effects[{effect_index}].op must be one of "
                        f"{sorted(EFFECT_OPERATORS)!r}"
                    )
                args = effect.get("args", {})
                if not isinstance(args, Mapping):
                    errors.append(
                        f"hooks[{index}].effects[{effect_index}].args must be a mapping"
                    )
                    continue
                _validate_effect_args(
                    errors,
                    label=f"hooks[{index}].effects[{effect_index}]",
                    op=op,
                    args=args,
                    state_field_names=state_field_names,
                )


def _validate_condition_args(
    errors: list[str],
    *,
    label: str,
    op: Any,
    args: Mapping[str, Any],
    state_field_names: set[str],
) -> None:
    if op == "self_on_board":
        _validate_exact_keys(errors, f"{label}.args", args, set())
    elif op in {"square_empty", "square_occupied"}:
        _validate_exact_keys(errors, f"{label}.args", args, {"square"})
        _validate_square_ref(errors, f"{label}.args.square", args.get("square"))
    elif op in {"state_field_eq", "state_field_gte", "state_field_lte"}:
        _validate_exact_keys(errors, f"{label}.args", args, {"name", "value"})
        _validate_state_field_name(
            errors,
            f"{label}.args.name",
            args.get("name"),
            state_field_names,
        )
        value = args.get("value")
        if op == "state_field_eq":
            if not _is_scalar(value):
                errors.append(f"{label}.args.value must be a scalar")
        else:
            if not _is_number(value):
                errors.append(f"{label}.args.value must be numeric")
    elif op == "piece_color_is":
        _validate_exact_keys(errors, f"{label}.args", args, {"piece", "color"})
        _validate_piece_ref(errors, f"{label}.args.piece", args.get("piece"))
        if args.get("color") not in {"white", "black"}:
            errors.append(f"{label}.args.color must be 'white' or 'black'")


def _validate_effect_args(
    errors: list[str],
    *,
    label: str,
    op: Any,
    args: Mapping[str, Any],
    state_field_names: set[str],
) -> None:
    if op == "move_piece":
        _validate_exact_keys(errors, f"{label}.args", args, {"piece", "to"})
        _validate_piece_ref(errors, f"{label}.args.piece", args.get("piece"))
        _validate_square_ref(errors, f"{label}.args.to", args.get("to"))
    elif op == "capture_piece":
        _validate_exact_keys(errors, f"{label}.args", args, {"piece"})
        _validate_piece_ref(errors, f"{label}.args.piece", args.get("piece"))
    elif op == "set_state":
        _validate_exact_keys(errors, f"{label}.args", args, {"piece", "name", "value"})
        piece_ref = args.get("piece")
        _validate_piece_ref(errors, f"{label}.args.piece", piece_ref)
        _validate_state_field_name(
            errors,
            f"{label}.args.name",
            args.get("name"),
            state_field_names if piece_ref == "self" else None,
        )
        if not _is_scalar(args.get("value")):
            errors.append(f"{label}.args.value must be a scalar")
    elif op == "increment_state":
        _validate_exact_keys(errors, f"{label}.args", args, {"piece", "name", "amount"})
        piece_ref = args.get("piece")
        _validate_piece_ref(errors, f"{label}.args.piece", piece_ref)
        _validate_state_field_name(
            errors,
            f"{label}.args.name",
            args.get("name"),
            state_field_names if piece_ref == "self" else None,
        )
        if not _is_number(args.get("amount")):
            errors.append(f"{label}.args.amount must be numeric")
    elif op == "emit_event":
        _validate_exact_keys(errors, f"{label}.args", args, {"event"})
        if args.get("event") not in HOOK_EVENTS:
            errors.append(
                f"{label}.args.event must be one of {sorted(HOOK_EVENTS)!r}"
            )


def _validate_exact_keys(
    errors: list[str], label: str, args: Mapping[str, Any], expected_keys: set[str]
) -> None:
    unknown_keys = sorted(set(args) - expected_keys)
    missing_keys = sorted(expected_keys - set(args))
    if missing_keys:
        errors.append(f"{label} is missing required keys: {', '.join(missing_keys)}")
    if unknown_keys:
        errors.append(f"{label} contains unknown keys: {', '.join(unknown_keys)}")


def _validate_square_ref(errors: list[str], label: str, value: Any) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be a square reference mapping")
        return
    keys = set(value)
    if keys == {"absolute"}:
        absolute = value.get("absolute")
        if not isinstance(absolute, str) or not is_valid_square(absolute):
            errors.append(f"{label}.absolute must be a square like 'e4'")
        return
    if keys == {"relative_to", "offset"}:
        _validate_piece_ref(errors, f"{label}.relative_to", value.get("relative_to"))
        offset = value.get("offset")
        if (
            not isinstance(offset, Sequence)
            or isinstance(offset, (str, bytes))
            or len(offset) != 2
            or any(not _is_int(component) for component in offset)
        ):
            errors.append(f"{label}.offset must be a two-integer list like [0, 1]")
        return
    errors.append(
        f"{label} must use either {{absolute}} or {{relative_to, offset}} form"
    )


def _validate_piece_ref(errors: list[str], label: str, value: Any) -> None:
    if value not in PIECE_REFS:
        errors.append(f"{label} must be one of {sorted(PIECE_REFS)!r}")


def _validate_state_field_name(
    errors: list[str],
    label: str,
    value: Any,
    known_names: set[str] | None,
) -> None:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        errors.append(f"{label} must be a lowercase identifier")
        return
    if known_names is not None and value not in known_names:
        errors.append(f"{label} must refer to a declared state field")


def _validate_positive_int(errors: list[str], label: str, value: Any) -> None:
    if not _is_int(value) or int(value) <= 0:
        errors.append(f"{label} must be a positive integer")


def _validate_typed_scalar(
    errors: list[str], *, label: str, value: Any, field_type: Any
) -> None:
    type_checks = {
        "bool": lambda candidate: isinstance(candidate, bool),
        "int": _is_int,
        "float": lambda candidate: _is_number(candidate),
        "str": lambda candidate: isinstance(candidate, str),
    }
    validator = type_checks.get(field_type)
    if validator is None:
        return
    if not validator(value):
        errors.append(f"{label} must match declared type {field_type!r}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool))


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def validate_piece_program(program: Mapping[str, Any]) -> None:
    errors = collect_validation_errors(program)
    if errors:
        raise PieceValidationError(errors)
