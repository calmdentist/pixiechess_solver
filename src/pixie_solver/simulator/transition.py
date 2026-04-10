from __future__ import annotations

from pixie_solver.core.event import Event, StateDelta
from pixie_solver.core.move import Move
from pixie_solver.core.piece import BasePieceType, Color, PieceInstance
from pixie_solver.core.state import GameState
from pixie_solver.simulator.fixed_point import resolve_event_cascade
from pixie_solver.simulator.invariants import assert_state_invariants
from pixie_solver.utils.squares import coords_to_square, square_to_coords


def apply_move_unchecked(state: GameState, move: Move) -> tuple[GameState, StateDelta]:
    piece_instances = dict(state.piece_instances)
    moving_piece = piece_instances[move.piece_id]
    before_state_hash = state.state_hash()
    target_piece_id = move.captured_piece_id or move.metadata.get("target_piece_id")
    changed_piece_ids: set[str] = {move.piece_id}
    directly_moved_piece_ids: set[str] = {move.piece_id}
    seed_events: list[Event] = []
    sequence = 0

    def next_event(
        event_type: str, *, target: str | None = None, source_cause: str = "move"
    ) -> Event:
        nonlocal sequence
        event = Event(
            event_type=event_type,
            actor_piece_id=move.piece_id,
            target_piece_id=target,
            source_cause=source_cause,
            sequence=sequence,
            payload={
                "from": move.from_square,
                "to": move.to_square,
                "move_kind": move.move_kind,
                "promotion_piece_type": move.promotion_piece_type,
                "tags": list(move.tags),
                **dict(move.metadata),
            },
        )
        sequence += 1
        return event

    removed_piece_id: str | None = None
    if move.move_kind == "move":
        piece_instances[move.piece_id] = _move_piece_to_square(
            state=state,
            piece=moving_piece,
            move=move,
            square=move.to_square,
        )
    elif move.move_kind == "capture":
        removed_piece_id = move.captured_piece_id
        if removed_piece_id is None:
            raise ValueError("capture move missing captured_piece_id")
        captured_piece = piece_instances[removed_piece_id]
        piece_instances[removed_piece_id] = _replace_piece_square(captured_piece, None)
        piece_instances[move.piece_id] = _move_piece_to_square(
            state=state,
            piece=moving_piece,
            move=move,
            square=move.to_square,
        )
        changed_piece_ids.add(removed_piece_id)
    elif move.move_kind == "en_passant_capture":
        removed_piece_id = move.captured_piece_id
        if removed_piece_id is None:
            raise ValueError("en passant move missing captured_piece_id")
        captured_piece = piece_instances[removed_piece_id]
        captured_square = str(move.metadata.get("captured_square", ""))
        if captured_piece.square != captured_square:
            raise ValueError("en passant captured_square does not match captured piece")
        piece_instances[removed_piece_id] = _replace_piece_square(captured_piece, None)
        piece_instances[move.piece_id] = _move_piece_to_square(
            state=state,
            piece=moving_piece,
            move=move,
            square=move.to_square,
        )
        changed_piece_ids.add(removed_piece_id)
    elif move.move_kind == "push_capture":
        target_piece_id = str(move.metadata["target_piece_id"])
        push_direction = tuple(int(component) for component in move.metadata["push_direction"])
        push_distance = int(move.metadata["push_distance"])
        push_edge_behavior = str(move.metadata["push_edge_behavior"])
        target_piece = piece_instances[target_piece_id]
        pushed_square, removed_piece_id = _resolve_push(
            piece_instances=piece_instances,
            target_piece=target_piece,
            direction=push_direction,
            distance=push_distance,
            edge_behavior=push_edge_behavior,
        )
        if removed_piece_id is not None:
            changed_piece_ids.add(removed_piece_id)
        piece_instances[target_piece_id] = _replace_piece_square(target_piece, pushed_square)
        piece_instances[move.piece_id] = _move_piece_to_square(
            state=state,
            piece=moving_piece,
            move=move,
            square=move.to_square,
        )
        changed_piece_ids.add(target_piece_id)
        directly_moved_piece_ids.add(target_piece_id)
    elif move.move_kind == "castle":
        rook_piece_id = str(move.metadata["rook_piece_id"])
        rook_piece = piece_instances[rook_piece_id]
        rook_to_square = str(move.metadata["rook_to_square"])
        piece_instances[rook_piece_id] = _replace_piece_square(rook_piece, rook_to_square)
        piece_instances[move.piece_id] = _move_piece_to_square(
            state=state,
            piece=moving_piece,
            move=move,
            square=move.to_square,
        )
        changed_piece_ids.add(rook_piece_id)
        directly_moved_piece_ids.add(rook_piece_id)
    else:
        raise ValueError(f"Unsupported move kind: {move.move_kind!r}")

    seed_events.append(
        next_event(
            "move_committed",
            target=target_piece_id if isinstance(target_piece_id, str) else None,
        )
    )
    if removed_piece_id is not None:
        seed_events.append(next_event("piece_captured", target=removed_piece_id))

    updated_state = GameState(
        piece_classes=state.piece_classes,
        piece_instances=piece_instances,
        side_to_move=state.side_to_move,
        castling_rights=_next_castling_rights(
            state=state,
            move=move,
            moving_piece=moving_piece,
            removed_piece_id=removed_piece_id,
            directly_moved_piece_ids=directly_moved_piece_ids,
        ),
        en_passant_square=_next_en_passant_square(
            state=state,
            move=move,
            moving_piece=moving_piece,
        ),
        halfmove_clock=_next_halfmove_clock(state, moving_piece, removed_piece_id is not None),
        fullmove_number=_next_fullmove_number(state),
        repetition_counts=state.repetition_counts,
        pending_events=(),
        metadata=state.metadata,
    )

    after_primary_state, primary_events = resolve_event_cascade(updated_state, tuple(seed_events))

    turn_end_state, turn_end_events = resolve_event_cascade(
        after_primary_state,
        (
            Event(
                event_type="turn_end",
                actor_piece_id=move.piece_id,
                source_cause="engine",
                sequence=len(primary_events),
                payload={"side": state.side_to_move.value},
            ),
        ),
    )

    next_side = other_color(state.side_to_move)
    start_state = GameState(
        piece_classes=turn_end_state.piece_classes,
        piece_instances=turn_end_state.piece_instances,
        side_to_move=next_side,
        castling_rights=turn_end_state.castling_rights,
        en_passant_square=turn_end_state.en_passant_square,
        halfmove_clock=turn_end_state.halfmove_clock,
        fullmove_number=turn_end_state.fullmove_number,
        repetition_counts=turn_end_state.repetition_counts,
        pending_events=(),
        metadata=turn_end_state.metadata,
    )
    final_state, turn_start_events = resolve_event_cascade(
        start_state,
        (
            Event(
                event_type="turn_start",
                actor_piece_id=move.piece_id,
                source_cause="engine",
                sequence=len(primary_events) + len(turn_end_events),
                payload={"side": next_side.value},
            ),
        ),
    )

    assert_state_invariants(final_state)
    after_state_hash = final_state.state_hash()
    changed_piece_ids.update(diff_piece_ids(state, final_state))
    return final_state, StateDelta(
        move=move,
        events=primary_events + turn_end_events + turn_start_events,
        changed_piece_ids=tuple(sorted(changed_piece_ids)),
        notes=(),
        metadata={
            "before_state_hash": before_state_hash,
            "after_state_hash": after_state_hash,
        },
    )


def other_color(color: Color) -> Color:
    return Color.BLACK if color == Color.WHITE else Color.WHITE


def diff_piece_ids(before: GameState, after: GameState) -> set[str]:
    changed: set[str] = set()
    for piece_id in set(before.piece_instances) | set(after.piece_instances):
        if before.piece_instances.get(piece_id) != after.piece_instances.get(piece_id):
            changed.add(piece_id)
    return changed


def replace_piece_square(piece: PieceInstance, square: str | None) -> PieceInstance:
    return _replace_piece_square(piece, square)


def _replace_piece_square(piece: PieceInstance, square: str | None) -> PieceInstance:
    return PieceInstance(
        instance_id=piece.instance_id,
        piece_class_id=piece.piece_class_id,
        color=piece.color,
        square=square,
        state=piece.state,
    )


def _move_piece_to_square(
    *,
    state: GameState,
    piece: PieceInstance,
    move: Move,
    square: str | None,
) -> PieceInstance:
    next_piece_class_id = piece.piece_class_id
    next_state = piece.state
    if move.promotion_piece_type is not None:
        next_piece_class_id = str(
            move.metadata.get("promotion_class_id")
            or _resolve_promotion_class_id(state, move.promotion_piece_type)
        )
        promoted_class = state.piece_classes[next_piece_class_id]
        if promoted_class.base_piece_type.value != move.promotion_piece_type:
            raise ValueError("promotion_class_id does not match promotion_piece_type")
        next_state = promoted_class.normalize_instance_state({})
    return PieceInstance(
        instance_id=piece.instance_id,
        piece_class_id=next_piece_class_id,
        color=piece.color,
        square=square,
        state=next_state,
    )


def _resolve_push(
    *,
    piece_instances: dict[str, PieceInstance],
    target_piece: PieceInstance,
    direction: tuple[int, int],
    distance: int,
    edge_behavior: str,
) -> tuple[str | None, str | None]:
    file_index, rank_index = square_to_coords(target_piece.square)
    current_file, current_rank = file_index, rank_index
    for _ in range(distance):
        current_file += direction[0]
        current_rank += direction[1]
        candidate_square = coords_to_square(current_file, current_rank)
        if candidate_square is None:
            if edge_behavior == "remove_if_pushed_off_board":
                return None, target_piece.instance_id
            raise ValueError("Push capture would leave the board with blocking edge behavior")
        if any(
            other.instance_id != target_piece.instance_id and other.square == candidate_square
            for other in piece_instances.values()
        ):
            raise ValueError("Push capture destination is occupied")
    return candidate_square, None


def _next_halfmove_clock(
    state: GameState, moving_piece: PieceInstance, was_capture: bool
) -> int:
    piece_class = state.piece_classes[moving_piece.piece_class_id]
    if was_capture or piece_class.base_piece_type == BasePieceType.PAWN:
        return 0
    return state.halfmove_clock + 1


def _next_fullmove_number(state: GameState) -> int:
    if state.side_to_move == Color.BLACK:
        return state.fullmove_number + 1
    return state.fullmove_number


def _next_en_passant_square(
    *,
    state: GameState,
    move: Move,
    moving_piece: PieceInstance,
) -> str | None:
    piece_class = state.piece_classes[moving_piece.piece_class_id]
    if piece_class.base_piece_type != BasePieceType.PAWN or move.move_kind != "move":
        return None

    from_file, from_rank = square_to_coords(move.from_square)
    to_file, to_rank = square_to_coords(move.to_square)
    if from_file != to_file or abs(to_rank - from_rank) != 2:
        return None
    return coords_to_square(from_file, (from_rank + to_rank) // 2)


def _next_castling_rights(
    *,
    state: GameState,
    move: Move,
    moving_piece: PieceInstance,
    removed_piece_id: str | None,
    directly_moved_piece_ids: set[str],
) -> dict[str, tuple[str, ...]]:
    rights = {
        color: set(sides)
        for color, sides in state.castling_rights.items()
    }
    for color in (Color.WHITE.value, Color.BLACK.value):
        rights.setdefault(color, set())

    for piece_id in directly_moved_piece_ids:
        piece_before_move = state.piece_instances[piece_id]
        piece_class = state.piece_classes[piece_before_move.piece_class_id]
        if piece_class.base_piece_type == BasePieceType.KING:
            rights[piece_before_move.color.value].clear()
            continue
        if piece_class.base_piece_type != BasePieceType.ROOK or piece_before_move.square is None:
            continue
        side = _castling_side_for_rook_square(
            color=piece_before_move.color,
            square=piece_before_move.square,
        )
        if side is not None:
            rights[piece_before_move.color.value].discard(side)

    if removed_piece_id is not None:
        removed_piece = state.piece_instances[removed_piece_id]
        removed_class = state.piece_classes[removed_piece.piece_class_id]
        side = None
        if removed_class.base_piece_type == BasePieceType.ROOK and removed_piece.square is not None:
            side = _castling_side_for_rook_square(
                color=removed_piece.color,
                square=removed_piece.square,
            )
        if side is not None:
            rights[removed_piece.color.value].discard(side)

    return {
        color: tuple(side for side in ("king", "queen") if side in sides)
        for color, sides in rights.items()
        if sides
    }


def _castling_side_for_rook_square(*, color: Color, square: str) -> str | None:
    home_rank = "1" if color == Color.WHITE else "8"
    if square == f"a{home_rank}":
        return "queen"
    if square == f"h{home_rank}":
        return "king"
    return None


def _resolve_promotion_class_id(state: GameState, promotion_piece_type: str) -> str:
    preferred_ids = (
        f"baseline_{promotion_piece_type}",
        f"orthodox_{promotion_piece_type}",
        promotion_piece_type,
    )
    for preferred_id in preferred_ids:
        piece_class = state.piece_classes.get(preferred_id)
        if piece_class is not None and piece_class.base_piece_type.value == promotion_piece_type:
            return preferred_id

    candidates = sorted(
        class_id
        for class_id, piece_class in state.piece_classes.items()
        if piece_class.base_piece_type.value == promotion_piece_type
    )
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Could not resolve unique promotion class for {promotion_piece_type!r}: "
        f"{candidates!r}"
    )
