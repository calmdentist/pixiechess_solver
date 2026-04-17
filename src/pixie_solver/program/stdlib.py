from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from pixie_solver.core.piece import (
    BasePieceType,
    Color,
    Modifier,
    PieceClass,
    PieceInstance,
    StateField,
)
from pixie_solver.core.state import GameState
from pixie_solver.utils.squares import coords_to_square, square_to_coords


def resolve_piece_ref(
    *,
    hook_owner_id: str,
    event: "Event",
    piece_ref: str,
) -> str | None:
    if piece_ref == "self":
        return hook_owner_id
    if piece_ref == "event_source":
        return event.actor_piece_id
    if piece_ref == "event_target":
        return event.target_piece_id
    return None


def resolve_square_ref(
    *,
    state: GameState,
    piece_instances: MutableMapping[str, PieceInstance] | Mapping[str, PieceInstance],
    hook_owner_id: str,
    event: "Event",
    square_ref: object,
) -> str | None:
    if not isinstance(square_ref, Mapping):
        raise ValueError("square reference must be a mapping")
    if "absolute" in square_ref:
        return str(square_ref["absolute"])

    reference_piece_id = resolve_piece_ref(
        hook_owner_id=hook_owner_id,
        event=event,
        piece_ref=str(square_ref["relative_to"]),
    )
    if reference_piece_id is None:
        return None

    reference_piece = piece_instances.get(reference_piece_id)
    if reference_piece is None or reference_piece.square is None:
        return None

    file_index, rank_index = square_to_coords(reference_piece.square)
    dx, dy = square_ref["offset"]
    absolute_dx, absolute_dy = absolute_offset(
        dx=int(dx),
        dy=int(dy),
        color=reference_piece.color,
    )
    return coords_to_square(file_index + absolute_dx, rank_index + absolute_dy)


def absolute_offset(*, dx: int, dy: int, color: Color) -> tuple[int, int]:
    if color == Color.WHITE:
        return dx, dy
    return -dx, -dy


def build_legacy_piece_class_for_actions(
    *,
    program: Mapping[str, Any],
    action_block: Mapping[str, Any],
) -> PieceClass:
    params = dict(action_block["params"])
    return PieceClass(
        class_id=str(program["program_id"]),
        name=str(program["name"]),
        base_piece_type=BasePieceType(str(params["base_archetype"])),
        movement_modifiers=tuple(
            Modifier(op=str(item["op"]), args=dict(item.get("args", {})))
            for item in params.get("movement_modifiers", [])
        ),
        capture_modifiers=tuple(
            Modifier(op=str(item["op"]), args=dict(item.get("args", {})))
            for item in params.get("capture_modifiers", [])
        ),
        instance_state_schema=tuple(
            StateField(
                name=str(field_spec["name"]),
                field_type=str(field_spec["type"]),
                default=field_spec.get("default"),
                description=str(field_spec.get("description", "")),
            )
            for field_spec in program.get("state_schema", [])
        ),
        metadata=dict(program.get("metadata", {})),
    )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pixie_solver.core.event import Event
