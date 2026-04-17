from __future__ import annotations

from typing import Any, Mapping

from pixie_solver.core.action import ActionIntent, stable_action_id
from pixie_solver.core.state import GameState
from pixie_solver.program.contexts import ActionContext
from pixie_solver.program.lower_legacy_dsl import lower_legacy_piece_class
from pixie_solver.program.stdlib import build_legacy_piece_class_for_actions
from pixie_solver.simulator.movegen import _generate_piece_moves, is_in_check, king_for_color


def enumerate_shadow_pseudo_legal_actions_for_piece(
    state: GameState,
    *,
    piece_id: str,
    program: Mapping[str, Any] | None = None,
) -> tuple[ActionIntent, ...]:
    piece = state.piece_instances[piece_id]
    active_program = _resolve_program_for_piece(
        state,
        piece_id=piece_id,
        program=program,
    )
    context = ActionContext(state=state, piece=piece, program=active_program)

    actions: list[ActionIntent] = []
    for action_block in active_program.get("action_blocks", []):
        if action_block["kind"] != "legacy_base_actions":
            raise ValueError(f"Unsupported action block kind: {action_block['kind']!r}")
        shadow_piece_class = build_legacy_piece_class_for_actions(
            program=active_program,
            action_block=action_block,
        )
        moves = _generate_piece_moves(
            state=context.state,
            occupancy=context.occupancy,
            piece=context.piece,
            piece_class=shadow_piece_class,
        )
        for move in moves:
            actions.append(move.to_action_intent())
    return tuple(sorted(actions, key=stable_action_id))


def enumerate_shadow_legal_actions(
    state: GameState,
    *,
    program_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[ActionIntent, ...]:
    actions: list[ActionIntent] = []
    for piece_id in sorted(state.piece_instances):
        piece = state.piece_instances[piece_id]
        if piece.square is None or piece.color != state.side_to_move:
            continue
        actions.extend(
            enumerate_shadow_pseudo_legal_actions_for_piece(
                state,
                piece_id=piece_id,
                program=(
                    program_registry.get(piece.piece_class_id)
                    if program_registry is not None
                    else None
                ),
            )
        )

    king = king_for_color(state, state.side_to_move)
    if king is None:
        return tuple(sorted(actions, key=stable_action_id))

    from pixie_solver.simulator.resolution import apply_action_shadow_unchecked

    filtered: list[ActionIntent] = []
    for action in actions:
        next_state, _ = apply_action_shadow_unchecked(
            state,
            action,
            program_registry=program_registry,
        )
        next_king = king_for_color(next_state, state.side_to_move)
        if next_king is None:
            continue
        if is_in_check(next_state, state.side_to_move):
            continue
        filtered.append(action)
    return tuple(sorted(filtered, key=stable_action_id))


def _resolve_program_for_piece(
    state: GameState,
    *,
    piece_id: str,
    program: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if program is not None:
        return program
    piece = state.piece_instances[piece_id]
    piece_class = state.piece_classes[piece.piece_class_id]
    return lower_legacy_piece_class(piece_class)
