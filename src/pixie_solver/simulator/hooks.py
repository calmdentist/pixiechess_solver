from __future__ import annotations

from collections import deque

from pixie_solver.core.event import Event
from pixie_solver.core.piece import Hook
from pixie_solver.core.state import GameState
from pixie_solver.simulator.effects import apply_effect, resolve_piece_ref, resolve_square_ref

MAX_EVENT_CASCADE = 128


def resolve_hooks(
    state: GameState, events: tuple[Event, ...]
) -> tuple[GameState, tuple[Event, ...]]:
    piece_instances = dict(state.piece_instances)
    emitted_events: list[Event] = []
    queue: deque[Event] = deque()
    sequence = max((event.sequence for event in events), default=-1) + 1

    for event in events:
        queue.append(event)

    def next_sequence() -> int:
        nonlocal sequence
        current = sequence
        sequence += 1
        return current

    while queue:
        if len(emitted_events) > MAX_EVENT_CASCADE:
            raise RuntimeError("Hook/event cascade exceeded safety limit")
        event = queue.popleft()
        emitted_events.append(event)
        current_state = _rebuild_state(state, piece_instances=piece_instances)
        for hook_owner_id, hook in _matching_hooks(current_state, event):
            if _conditions_match(
                state=current_state,
                piece_instances=piece_instances,
                hook_owner_id=hook_owner_id,
                hook=hook,
                event=event,
            ):
                for effect in hook.effects:
                    current_state = _rebuild_state(state, piece_instances=piece_instances)
                    new_events = apply_effect(
                        state=current_state,
                        piece_instances=piece_instances,
                        hook_owner_id=hook_owner_id,
                        effect=effect,
                        event=event,
                        next_sequence=next_sequence,
                    )
                    queue.extend(new_events)

    return _rebuild_state(state, piece_instances=piece_instances), tuple(emitted_events)


def _matching_hooks(state: GameState, event: Event) -> list[tuple[str, Hook]]:
    matches: list[tuple[str, int, int, Hook]] = []
    for piece_id in sorted(state.piece_instances):
        piece = state.piece_instances[piece_id]
        piece_class = state.piece_classes[piece.piece_class_id]
        for hook_index, hook in enumerate(piece_class.hooks):
            if hook.event == event.event_type:
                matches.append((piece_id, hook.priority, hook_index, hook))
    matches.sort(key=lambda item: (item[1], item[0], item[2]))
    return [(piece_id, hook) for piece_id, _, _, hook in matches]


def _conditions_match(
    *,
    state: GameState,
    piece_instances: dict[str, object],
    hook_owner_id: str,
    hook: Hook,
    event: Event,
) -> bool:
    for condition in hook.conditions:
        if not _condition_matches(
            state=state,
            piece_instances=piece_instances,
            hook_owner_id=hook_owner_id,
            condition_op=condition.op,
            args=condition.args,
            event=event,
        ):
            return False
    return True


def _condition_matches(
    *,
    state: GameState,
    piece_instances: dict[str, object],
    hook_owner_id: str,
    condition_op: str,
    args: dict[str, object],
    event: Event,
) -> bool:
    hook_owner = state.piece_instances[hook_owner_id]
    if condition_op == "self_on_board":
        return hook_owner.square is not None
    if condition_op == "square_empty":
        square = resolve_square_ref(
            state=state,
            piece_instances=piece_instances,
            hook_owner_id=hook_owner_id,
            event=event,
            square_ref=args["square"],
        )
        return square is not None and state.piece_on(square) is None
    if condition_op == "square_occupied":
        square = resolve_square_ref(
            state=state,
            piece_instances=piece_instances,
            hook_owner_id=hook_owner_id,
            event=event,
            square_ref=args["square"],
        )
        return square is not None and state.piece_on(square) is not None
    if condition_op == "state_field_eq":
        return hook_owner.state.get(str(args["name"])) == args["value"]
    if condition_op == "state_field_gte":
        value = hook_owner.state.get(str(args["name"]), 0)
        return isinstance(value, (int, float)) and value >= args["value"]
    if condition_op == "state_field_lte":
        value = hook_owner.state.get(str(args["name"]), 0)
        return isinstance(value, (int, float)) and value <= args["value"]
    if condition_op == "piece_color_is":
        piece_id = resolve_piece_ref(
            hook_owner_id=hook_owner_id,
            event=event,
            piece_ref=str(args["piece"]),
        )
        if piece_id is None or piece_id not in state.piece_instances:
            return False
        return state.piece_instances[piece_id].color.value == args["color"]
    raise ValueError(f"Unsupported condition op: {condition_op!r}")


def _rebuild_state(
    state: GameState,
    *,
    piece_instances: dict[str, object],
    side_to_move: object | None = None,
) -> GameState:
    return GameState(
        piece_classes=state.piece_classes,
        piece_instances=piece_instances,
        side_to_move=state.side_to_move if side_to_move is None else side_to_move,
        castling_rights=state.castling_rights,
        en_passant_square=state.en_passant_square,
        halfmove_clock=state.halfmove_clock,
        fullmove_number=state.fullmove_number,
        repetition_counts=state.repetition_counts,
        pending_events=(),
        metadata=state.metadata,
    )
