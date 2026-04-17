from __future__ import annotations

from collections import deque
from typing import Any, Mapping, MutableMapping

from pixie_solver.core import stable_action_id
from pixie_solver.core.action import ActionIntent
from pixie_solver.core.effect import TransitionEffect
from pixie_solver.core.event import Event, StateDelta
from pixie_solver.core.move import Move
from pixie_solver.core.piece import PieceInstance
from pixie_solver.core.state import GameState
from pixie_solver.core.trace import TraceFrame, TransitionTrace
from pixie_solver.program.contexts import EventContext
from pixie_solver.program.lower_legacy_dsl import lower_legacy_piece_class
from pixie_solver.program.stdlib import resolve_piece_ref, resolve_square_ref
from pixie_solver.simulator.commit import commit_action_unchecked
from pixie_solver.simulator.invariants import assert_state_invariants
from pixie_solver.simulator.transition import (
    _next_repetition_counts,
    _with_repetition_counts,
    diff_piece_ids,
    other_color,
)

MAX_PROGRAM_EVENT_CASCADE = 128


def apply_action_shadow(
    state: GameState,
    action: ActionIntent | Move,
    *,
    program_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[GameState, StateDelta]:
    canonical_action = (
        action if isinstance(action, ActionIntent) else action.to_action_intent()
    )
    legal_actions = {
        stable_action_id(candidate): candidate
        for candidate in enumerate_shadow_legal_actions(
            state,
            program_registry=program_registry,
        )
    }
    action_id = stable_action_id(canonical_action)
    if action_id not in legal_actions:
        raise ValueError(f"Illegal action: {canonical_action.to_dict()!r}")
    return apply_action_shadow_unchecked(
        state,
        legal_actions[action_id],
        program_registry=program_registry,
    )


def apply_action_shadow_unchecked(
    state: GameState,
    action: ActionIntent | Move,
    *,
    program_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[GameState, StateDelta]:
    canonical_action = (
        action if isinstance(action, ActionIntent) else action.to_action_intent()
    )

    commit = commit_action_unchecked(state, canonical_action)
    action_id = stable_action_id(commit.action)

    after_primary_state, primary_events, primary_effects, primary_trace = resolve_program_event_cascade(
        commit.state_after_commit,
        commit.seed_events,
        program_registry=program_registry,
        action_id=action_id,
        start_frame_id=0,
    )

    turn_end_seed = (
        Event(
            event_type="turn_end",
            actor_piece_id=commit.move.piece_id,
            source_cause="engine",
            source_action_id=action_id,
            sequence=len(primary_events),
            payload={"side": state.side_to_move.value},
        ),
    )
    turn_end_state, turn_end_events, turn_end_effects, turn_end_trace = resolve_program_event_cascade(
        after_primary_state,
        turn_end_seed,
        program_registry=program_registry,
        action_id=action_id,
        start_frame_id=len(primary_trace.frames),
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
    turn_start_seed = (
        Event(
            event_type="turn_start",
            actor_piece_id=commit.move.piece_id,
            source_cause="engine",
            source_action_id=action_id,
            sequence=len(primary_events) + len(turn_end_events),
            payload={"side": next_side.value},
        ),
    )
    final_state, turn_start_events, turn_start_effects, turn_start_trace = resolve_program_event_cascade(
        start_state,
        turn_start_seed,
        program_registry=program_registry,
        action_id=action_id,
        start_frame_id=len(primary_trace.frames) + len(turn_end_trace.frames),
    )
    final_state = _with_repetition_counts(
        final_state,
        _next_repetition_counts(state, final_state),
    )

    assert_state_invariants(final_state)
    after_state_hash = final_state.state_hash()
    changed_piece_ids = set(commit.changed_piece_ids)
    changed_piece_ids.update(diff_piece_ids(state, final_state))
    all_events = primary_events + turn_end_events + turn_start_events
    all_effects = primary_effects + turn_end_effects + turn_start_effects
    trace = TransitionTrace(
        action_id=action_id,
        frames=primary_trace.frames + turn_end_trace.frames + turn_start_trace.frames,
        metadata={
            "phases": ["primary", "turn_end", "turn_start"],
            "engine": "program_shadow",
        },
    )
    return final_state, StateDelta(
        move=commit.move,
        action=commit.action,
        events=all_events,
        effects=all_effects,
        changed_piece_ids=tuple(sorted(changed_piece_ids)),
        created_piece_ids=(),
        removed_piece_ids=((commit.removed_piece_id,) if commit.removed_piece_id is not None else ()),
        notes=(),
        trace=trace,
        metadata={
            "before_state_hash": commit.before_state_hash,
            "after_state_hash": after_state_hash,
        },
    )


def resolve_program_event_cascade(
    state: GameState,
    seed_events: tuple[Event, ...],
    *,
    program_registry: Mapping[str, Mapping[str, Any]] | None = None,
    action_id: str | None = None,
    start_frame_id: int = 0,
) -> tuple[GameState, tuple[Event, ...], tuple[TransitionEffect, ...], TransitionTrace]:
    piece_instances = dict(state.piece_instances)
    emitted_events: list[Event] = []
    emitted_effects: list[TransitionEffect] = []
    frames: list[TraceFrame] = []
    queue: deque[Event] = deque(seed_events)
    sequence = max((event.sequence for event in seed_events), default=-1) + 1
    frame_id = start_frame_id

    def next_sequence() -> int:
        nonlocal sequence
        current = sequence
        sequence += 1
        return current

    while queue:
        if len(emitted_events) > MAX_PROGRAM_EVENT_CASCADE:
            raise RuntimeError("ProgramIR event cascade exceeded safety limit")
        event = queue.popleft()
        emitted_events.append(event)
        current_state = _rebuild_state(state, piece_instances=piece_instances)
        frame_effects: list[TransitionEffect] = []
        for hook_owner_id, block in _matching_reaction_blocks(
            current_state,
            event,
            program_registry=program_registry,
        ):
            context = EventContext(
                state=current_state,
                piece_instances=piece_instances,
                hook_owner_id=hook_owner_id,
                program=_resolve_program_for_piece(
                    current_state,
                    piece_id=hook_owner_id,
                    program_registry=program_registry,
                ),
                event=event,
            )
            if not _conditions_match(context=context, block=block):
                continue
            for effect in block["effects"]:
                current_state = _rebuild_state(state, piece_instances=piece_instances)
                transition_effect, new_events = _apply_program_effect(
                    state=current_state,
                    piece_instances=piece_instances,
                    hook_owner_id=hook_owner_id,
                    effect=effect,
                    event=event,
                    next_sequence=next_sequence,
                    action_id=action_id,
                )
                frame_effects.append(transition_effect)
                emitted_effects.append(transition_effect)
                queue.extend(new_events)
        frames.append(
            TraceFrame(
                frame_id=frame_id,
                phase="event_resolution",
                event=event,
                effects=tuple(frame_effects),
                metadata={"action_id": action_id, "event_type": event.event_type},
            )
        )
        frame_id += 1

    return (
        _rebuild_state(state, piece_instances=piece_instances),
        tuple(emitted_events),
        tuple(emitted_effects),
        TransitionTrace(
            action_id=action_id,
            frames=tuple(frames),
            metadata={"engine": "program_shadow"},
        ),
    )


def _matching_reaction_blocks(
    state: GameState,
    event: Event,
    *,
    program_registry: Mapping[str, Mapping[str, Any]] | None,
) -> list[tuple[str, Mapping[str, Any]]]:
    matches: list[tuple[str, int, str, Mapping[str, Any]]] = []
    for piece_id in sorted(state.piece_instances):
        piece = state.piece_instances[piece_id]
        program = _resolve_program_for_piece(
            state,
            piece_id=piece_id,
            program_registry=program_registry,
        )
        for block in program.get("reaction_blocks", []):
            if block["trigger"]["event_type"] == event.event_type:
                matches.append(
                    (
                        piece_id,
                        int(block.get("priority", 0)),
                        str(block["block_id"]),
                        block,
                    )
                )
    matches.sort(key=lambda item: (item[1], item[0], item[2]))
    return [(piece_id, block) for piece_id, _, _, block in matches]


def _conditions_match(
    *,
    context: EventContext,
    block: Mapping[str, Any],
) -> bool:
    for condition in block.get("conditions", []):
        if not _condition_matches(
            context=context,
            condition_kind=str(condition["condition_kind"]),
            args=dict(condition.get("args", {})),
        ):
            return False
    return True


def _condition_matches(
    *,
    context: EventContext,
    condition_kind: str,
    args: Mapping[str, Any],
) -> bool:
    hook_owner = context.hook_owner
    if condition_kind == "self_on_board":
        return hook_owner.square is not None
    if condition_kind == "square_empty":
        square = resolve_square_ref(
            state=context.state,
            piece_instances=context.piece_instances,
            hook_owner_id=context.hook_owner_id,
            event=context.event,
            square_ref=args["square"],
        )
        return square is not None and context.state.piece_on(square) is None
    if condition_kind == "square_occupied":
        square = resolve_square_ref(
            state=context.state,
            piece_instances=context.piece_instances,
            hook_owner_id=context.hook_owner_id,
            event=context.event,
            square_ref=args["square"],
        )
        return square is not None and context.state.piece_on(square) is not None
    if condition_kind == "state_field_eq":
        return hook_owner.state.get(str(args["name"])) == args["value"]
    if condition_kind == "state_field_gte":
        value = hook_owner.state.get(str(args["name"]), 0)
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= args["value"]
    if condition_kind == "state_field_lte":
        value = hook_owner.state.get(str(args["name"]), 0)
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value <= args["value"]
    if condition_kind == "piece_color_is":
        piece_id = resolve_piece_ref(
            hook_owner_id=context.hook_owner_id,
            event=context.event,
            piece_ref=str(args["piece"]),
        )
        if piece_id is None or piece_id not in context.state.piece_instances:
            return False
        return context.state.piece_instances[piece_id].color.value == args["color"]
    raise ValueError(f"Unsupported condition kind: {condition_kind!r}")


def _apply_program_effect(
    *,
    state: GameState,
    piece_instances: MutableMapping[str, PieceInstance],
    hook_owner_id: str,
    effect: Mapping[str, Any],
    event: Event,
    next_sequence,
    action_id: str | None,
) -> tuple[TransitionEffect, tuple[Event, ...]]:
    effect_kind = str(effect["effect_kind"])
    args = dict(effect.get("args", {}))
    if effect_kind == "move_piece":
        return _apply_move_piece(
            state=state,
            piece_instances=piece_instances,
            hook_owner_id=hook_owner_id,
            args=args,
            event=event,
            action_id=action_id,
        )
    if effect_kind == "capture_piece":
        return _apply_capture_piece(
            piece_instances=piece_instances,
            hook_owner_id=hook_owner_id,
            args=args,
            event=event,
            next_sequence=next_sequence,
            action_id=action_id,
        )
    if effect_kind == "set_state":
        transition_effect = _apply_set_state(
            state=state,
            piece_instances=piece_instances,
            hook_owner_id=hook_owner_id,
            args=args,
            event=event,
            action_id=action_id,
        )
        return transition_effect, ()
    if effect_kind == "increment_state":
        transition_effect = _apply_increment_state(
            state=state,
            piece_instances=piece_instances,
            hook_owner_id=hook_owner_id,
            args=args,
            event=event,
            action_id=action_id,
        )
        return transition_effect, ()
    if effect_kind == "emit_event":
        emitted_event = Event(
            event_type=str(args["event"]),
            actor_piece_id=hook_owner_id,
            source_cause="hook",
            source_action_id=action_id,
            sequence=next_sequence(),
        )
        return (
            TransitionEffect(
                effect_kind="emit_event",
                actor_piece_id=hook_owner_id,
                payload={"event_type": str(args["event"])},
                metadata={"applied": True},
            ),
            (emitted_event,),
        )
    raise ValueError(f"Unsupported effect kind: {effect_kind!r}")


def _apply_move_piece(
    *,
    state: GameState,
    piece_instances: MutableMapping[str, PieceInstance],
    hook_owner_id: str,
    args: Mapping[str, Any],
    event: Event,
    action_id: str | None,
) -> tuple[TransitionEffect, tuple[Event, ...]]:
    piece_id = resolve_piece_ref(
        hook_owner_id=hook_owner_id,
        event=event,
        piece_ref=str(args["piece"]),
    )
    if piece_id is None:
        raise ValueError("move_piece effect could not resolve piece reference")
    piece = piece_instances.get(piece_id)
    if piece is None or piece.square is None:
        return (
            TransitionEffect(
                effect_kind="move_piece",
                actor_piece_id=hook_owner_id,
                target_piece_id=piece_id,
                payload={"piece_ref": str(args["piece"])},
                metadata={"applied": False, "reason": "inactive_piece", "action_id": action_id},
            ),
            (),
        )

    destination = resolve_square_ref(
        state=state,
        piece_instances=piece_instances,
        hook_owner_id=hook_owner_id,
        event=event,
        square_ref=args["to"],
    )
    if destination is None:
        return (
            TransitionEffect(
                effect_kind="move_piece",
                actor_piece_id=hook_owner_id,
                target_piece_id=piece_id,
                payload={"piece_ref": str(args["piece"])},
                metadata={"applied": False, "reason": "unresolved_destination", "action_id": action_id},
            ),
            (),
        )
    if any(
        other.instance_id != piece_id and other.square == destination
        for other in piece_instances.values()
    ):
        return (
            TransitionEffect(
                effect_kind="move_piece",
                actor_piece_id=hook_owner_id,
                target_piece_id=piece_id,
                target_square=destination,
                payload={"piece_ref": str(args["piece"])},
                metadata={"applied": False, "reason": "occupied_destination", "action_id": action_id},
            ),
            (),
        )

    piece_instances[piece_id] = PieceInstance(
        instance_id=piece.instance_id,
        piece_class_id=piece.piece_class_id,
        color=piece.color,
        square=destination,
        state=piece.state,
    )
    return (
        TransitionEffect(
            effect_kind="move_piece",
            actor_piece_id=hook_owner_id,
            target_piece_id=piece_id,
            target_square=destination,
            payload={"piece_ref": str(args["piece"])},
            metadata={"applied": True, "action_id": action_id},
        ),
        (),
    )


def _apply_capture_piece(
    *,
    piece_instances: MutableMapping[str, PieceInstance],
    hook_owner_id: str,
    args: Mapping[str, Any],
    event: Event,
    next_sequence,
    action_id: str | None,
) -> tuple[TransitionEffect, tuple[Event, ...]]:
    piece_id = resolve_piece_ref(
        hook_owner_id=hook_owner_id,
        event=event,
        piece_ref=str(args["piece"]),
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
    emitted_event = Event(
        event_type="piece_captured",
        actor_piece_id=hook_owner_id,
        target_piece_id=piece_id,
        source_cause="hook",
        source_action_id=action_id,
        sequence=next_sequence(),
    )
    return (
        TransitionEffect(
            effect_kind="capture_piece",
            actor_piece_id=hook_owner_id,
            target_piece_id=piece_id,
            metadata={"applied": True, "action_id": action_id},
        ),
        (emitted_event,),
    )


def _apply_set_state(
    *,
    state: GameState,
    piece_instances: MutableMapping[str, PieceInstance],
    hook_owner_id: str,
    args: Mapping[str, Any],
    event: Event,
    action_id: str | None,
) -> TransitionEffect:
    piece_id = resolve_piece_ref(
        hook_owner_id=hook_owner_id,
        event=event,
        piece_ref=str(args["piece"]),
    )
    if piece_id is None:
        raise ValueError("set_state effect could not resolve piece reference")
    piece = piece_instances[piece_id]
    field_name = str(args["name"])
    _ensure_state_field_exists(state=state, piece=piece, field_name=field_name)
    new_state = dict(piece.state)
    new_state[field_name] = args["value"]
    piece_instances[piece_id] = PieceInstance(
        instance_id=piece.instance_id,
        piece_class_id=piece.piece_class_id,
        color=piece.color,
        square=piece.square,
        state=new_state,
    )
    return TransitionEffect(
        effect_kind="set_state",
        actor_piece_id=hook_owner_id,
        target_piece_id=piece_id,
        payload={"name": field_name, "value": args["value"]},
        metadata={"applied": True, "action_id": action_id},
    )


def _apply_increment_state(
    *,
    state: GameState,
    piece_instances: MutableMapping[str, PieceInstance],
    hook_owner_id: str,
    args: Mapping[str, Any],
    event: Event,
    action_id: str | None,
) -> TransitionEffect:
    piece_id = resolve_piece_ref(
        hook_owner_id=hook_owner_id,
        event=event,
        piece_ref=str(args["piece"]),
    )
    if piece_id is None:
        raise ValueError("increment_state effect could not resolve piece reference")
    piece = piece_instances[piece_id]
    field_name = str(args["name"])
    _ensure_state_field_exists(state=state, piece=piece, field_name=field_name)
    current_value = piece.state.get(field_name, 0)
    if not isinstance(current_value, (int, float)) or isinstance(current_value, bool):
        raise ValueError(f"State field {field_name!r} is not numeric")
    amount = args["amount"]
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
    return TransitionEffect(
        effect_kind="increment_state",
        actor_piece_id=hook_owner_id,
        target_piece_id=piece_id,
        payload={"name": field_name, "amount": amount},
        metadata={"applied": True, "action_id": action_id},
    )


def _ensure_state_field_exists(
    *,
    state: GameState,
    piece: PieceInstance,
    field_name: str,
) -> None:
    piece_class = state.piece_classes[piece.piece_class_id]
    if field_name not in {field.name for field in piece_class.instance_state_schema}:
        raise ValueError(
            f"State field {field_name!r} is not declared for piece class "
            f"{piece.piece_class_id!r}"
        )


def _resolve_program_for_piece(
    state: GameState,
    *,
    piece_id: str,
    program_registry: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    piece = state.piece_instances[piece_id]
    if program_registry is not None and piece.piece_class_id in program_registry:
        return program_registry[piece.piece_class_id]
    return lower_legacy_piece_class(state.piece_classes[piece.piece_class_id])


def _rebuild_state(
    state: GameState,
    *,
    piece_instances: MutableMapping[str, PieceInstance],
    side_to_move=None,
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


from pixie_solver.simulator.actiongen import enumerate_shadow_legal_actions
