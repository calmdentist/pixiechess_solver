from __future__ import annotations

from dataclasses import dataclass

from pixie_solver.core.action import ActionIntent
from pixie_solver.core.event import Event
from pixie_solver.core.move import Move
from pixie_solver.core.piece import PieceInstance
from pixie_solver.core.state import GameState
from pixie_solver.simulator.transition import (
    _move_piece_to_square,
    _next_castling_rights,
    _next_en_passant_square,
    _next_fullmove_number,
    _next_halfmove_clock,
    _resolve_push,
)


@dataclass(frozen=True, slots=True)
class PrimaryCommitResult:
    action: ActionIntent
    move: Move
    state_after_commit: GameState
    seed_events: tuple[Event, ...]
    before_state_hash: str
    changed_piece_ids: tuple[str, ...]
    directly_moved_piece_ids: tuple[str, ...]
    removed_piece_id: str | None = None
    target_piece_id: str | None = None


def commit_action_unchecked(
    state: GameState,
    action: ActionIntent | Move,
) -> PrimaryCommitResult:
    move = action if isinstance(action, Move) else Move.from_action_intent(action)
    canonical_action = move.to_action_intent()

    piece_instances = dict(state.piece_instances)
    moving_piece = piece_instances[move.piece_id]
    before_state_hash = state.state_hash()
    target_piece_id = move.captured_piece_id or move.metadata.get("target_piece_id")
    changed_piece_ids: set[str] = {move.piece_id}
    directly_moved_piece_ids: set[str] = {move.piece_id}
    seed_events: list[Event] = []
    sequence = 0

    def next_event(
        event_type: str,
        *,
        target: str | None = None,
        source_cause: str = "move",
    ) -> Event:
        nonlocal sequence
        event = Event(
            event_type=event_type,
            actor_piece_id=move.piece_id,
            target_piece_id=target,
            source_cause=source_cause,
            sequence=sequence,
            source_action_id=canonical_action.stable_id(),
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

    return PrimaryCommitResult(
        action=canonical_action,
        move=move,
        state_after_commit=updated_state,
        seed_events=tuple(seed_events),
        before_state_hash=before_state_hash,
        changed_piece_ids=tuple(sorted(changed_piece_ids)),
        directly_moved_piece_ids=tuple(sorted(directly_moved_piece_ids)),
        removed_piece_id=removed_piece_id,
        target_piece_id=str(target_piece_id) if target_piece_id is not None else None,
    )


def _replace_piece_square(piece: PieceInstance, square: str | None) -> PieceInstance:
    return PieceInstance(
        instance_id=piece.instance_id,
        piece_class_id=piece.piece_class_id,
        color=piece.color,
        square=square,
        state=piece.state,
    )
