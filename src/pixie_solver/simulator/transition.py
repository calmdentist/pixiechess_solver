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
    target_piece_id = move.captured_piece_id or move.metadata.get("target_piece_id")
    changed_piece_ids: set[str] = {move.piece_id}
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
            },
        )
        sequence += 1
        return event

    removed_piece_id: str | None = None
    if move.move_kind == "move":
        piece_instances[move.piece_id] = _replace_piece_square(moving_piece, move.to_square)
    elif move.move_kind == "capture":
        removed_piece_id = move.captured_piece_id
        if removed_piece_id is None:
            raise ValueError("capture move missing captured_piece_id")
        captured_piece = piece_instances[removed_piece_id]
        piece_instances[removed_piece_id] = _replace_piece_square(captured_piece, None)
        piece_instances[move.piece_id] = _replace_piece_square(moving_piece, move.to_square)
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
        piece_instances[move.piece_id] = _replace_piece_square(moving_piece, move.to_square)
        changed_piece_ids.add(target_piece_id)
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
        castling_rights=state.castling_rights,
        en_passant_square=None,
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
    changed_piece_ids.update(diff_piece_ids(state, final_state))
    return final_state, StateDelta(
        move=move,
        events=primary_events + turn_end_events + turn_start_events,
        changed_piece_ids=tuple(sorted(changed_piece_ids)),
        notes=(),
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
