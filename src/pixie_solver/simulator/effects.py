from __future__ import annotations

from collections.abc import Callable, MutableMapping

from pixie_solver.core.event import Event
from pixie_solver.core.piece import Color, PieceInstance
from pixie_solver.core.piece import Effect as PieceEffect
from pixie_solver.core.state import GameState
from pixie_solver.utils.squares import coords_to_square, square_to_coords


def resolve_piece_ref(
    *, hook_owner_id: str, event: Event, piece_ref: str
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
    piece_instances: MutableMapping[str, PieceInstance],
    hook_owner_id: str,
    event: Event,
    square_ref: object,
) -> str | None:
    if not isinstance(square_ref, dict):
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
    absolute_dx, absolute_dy = _absolute_offset(
        dx=int(dx),
        dy=int(dy),
        color=reference_piece.color,
    )
    return coords_to_square(file_index + absolute_dx, rank_index + absolute_dy)


def apply_effect(
    *,
    state: GameState,
    piece_instances: MutableMapping[str, PieceInstance],
    hook_owner_id: str,
    effect: PieceEffect,
    event: Event,
    next_sequence: Callable[[], int],
) -> tuple[Event, ...]:
    if effect.op == "move_piece":
        return _apply_move_piece(
            state=state,
            piece_instances=piece_instances,
            hook_owner_id=hook_owner_id,
            effect=effect,
            event=event,
        )
    if effect.op == "capture_piece":
        return _apply_capture_piece(
            piece_instances=piece_instances,
            hook_owner_id=hook_owner_id,
            effect=effect,
            event=event,
            next_sequence=next_sequence,
        )
    if effect.op == "set_state":
        _apply_set_state(
            state=state,
            piece_instances=piece_instances,
            hook_owner_id=hook_owner_id,
            effect=effect,
            event=event,
        )
        return ()
    if effect.op == "increment_state":
        _apply_increment_state(
            state=state,
            piece_instances=piece_instances,
            hook_owner_id=hook_owner_id,
            effect=effect,
            event=event,
        )
        return ()
    if effect.op == "emit_event":
        return (
            Event(
                event_type=str(effect.args["event"]),
                actor_piece_id=hook_owner_id,
                source_cause="hook",
                sequence=next_sequence(),
            ),
        )
    raise ValueError(f"Unsupported effect op: {effect.op!r}")


def _apply_move_piece(
    *,
    state: GameState,
    piece_instances: MutableMapping[str, PieceInstance],
    hook_owner_id: str,
    effect: PieceEffect,
    event: Event,
) -> tuple[Event, ...]:
    piece_id = resolve_piece_ref(
        hook_owner_id=hook_owner_id,
        event=event,
        piece_ref=str(effect.args["piece"]),
    )
    if piece_id is None:
        raise ValueError("move_piece effect could not resolve piece reference")
    piece = piece_instances.get(piece_id)
    if piece is None or piece.square is None:
        raise ValueError(f"move_piece effect refers to inactive piece {piece_id!r}")

    destination = resolve_square_ref(
        state=state,
        piece_instances=piece_instances,
        hook_owner_id=hook_owner_id,
        event=event,
        square_ref=effect.args["to"],
    )
    if destination is None:
        raise ValueError("move_piece effect resolved to an off-board square")
    if any(
        other.instance_id != piece_id and other.square == destination
        for other in piece_instances.values()
    ):
        raise ValueError(f"move_piece effect destination {destination!r} is occupied")

    piece_instances[piece_id] = PieceInstance(
        instance_id=piece.instance_id,
        piece_class_id=piece.piece_class_id,
        color=piece.color,
        square=destination,
        state=piece.state,
    )
    return ()


def _apply_capture_piece(
    *,
    piece_instances: MutableMapping[str, PieceInstance],
    hook_owner_id: str,
    effect: PieceEffect,
    event: Event,
    next_sequence: Callable[[], int],
) -> tuple[Event, ...]:
    piece_id = resolve_piece_ref(
        hook_owner_id=hook_owner_id,
        event=event,
        piece_ref=str(effect.args["piece"]),
    )
    if piece_id is None:
        raise ValueError("capture_piece effect could not resolve piece reference")
    piece = piece_instances.get(piece_id)
    if piece is None or piece.square is None:
        raise ValueError(f"capture_piece effect refers to inactive piece {piece_id!r}")

    piece_instances[piece_id] = PieceInstance(
        instance_id=piece.instance_id,
        piece_class_id=piece.piece_class_id,
        color=piece.color,
        square=None,
        state=piece.state,
    )
    return (
        Event(
            event_type="piece_captured",
            actor_piece_id=hook_owner_id,
            target_piece_id=piece_id,
            source_cause="hook",
            sequence=next_sequence(),
        ),
    )


def _apply_set_state(
    *,
    state: GameState,
    piece_instances: MutableMapping[str, PieceInstance],
    hook_owner_id: str,
    effect: PieceEffect,
    event: Event,
) -> None:
    piece_id = resolve_piece_ref(
        hook_owner_id=hook_owner_id,
        event=event,
        piece_ref=str(effect.args["piece"]),
    )
    if piece_id is None:
        raise ValueError("set_state effect could not resolve piece reference")
    piece = piece_instances[piece_id]
    _ensure_state_field_exists(
        state=state,
        piece=piece,
        field_name=str(effect.args["name"]),
    )
    new_state = dict(piece.state)
    new_state[str(effect.args["name"])] = effect.args["value"]
    piece_instances[piece_id] = PieceInstance(
        instance_id=piece.instance_id,
        piece_class_id=piece.piece_class_id,
        color=piece.color,
        square=piece.square,
        state=new_state,
    )


def _apply_increment_state(
    *,
    state: GameState,
    piece_instances: MutableMapping[str, PieceInstance],
    hook_owner_id: str,
    effect: PieceEffect,
    event: Event,
) -> None:
    piece_id = resolve_piece_ref(
        hook_owner_id=hook_owner_id,
        event=event,
        piece_ref=str(effect.args["piece"]),
    )
    if piece_id is None:
        raise ValueError("increment_state effect could not resolve piece reference")
    piece = piece_instances[piece_id]
    field_name = str(effect.args["name"])
    _ensure_state_field_exists(state=state, piece=piece, field_name=field_name)
    current_value = piece.state.get(field_name, 0)
    if not isinstance(current_value, (int, float)) or isinstance(current_value, bool):
        raise ValueError(f"State field {field_name!r} is not numeric")
    amount = effect.args["amount"]
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        raise ValueError("increment_state amount must be numeric")
    new_state = dict(piece.state)
    new_state[field_name] = current_value + amount
    piece_instances[piece_id] = PieceInstance(
        instance_id=piece.instance_id,
        piece_class_id=piece.piece_class_id,
        color=piece.color,
        square=piece.square,
        state=new_state,
    )


def _ensure_state_field_exists(
    *, state: GameState, piece: PieceInstance, field_name: str
) -> None:
    piece_class = state.piece_classes[piece.piece_class_id]
    if field_name not in {field.name for field in piece_class.instance_state_schema}:
        raise ValueError(
            f"State field {field_name!r} is not declared for piece class "
            f"{piece.piece_class_id!r}"
        )


def _absolute_offset(*, dx: int, dy: int, color: Color) -> tuple[int, int]:
    if color == Color.WHITE:
        return dx, dy
    return -dx, -dy
