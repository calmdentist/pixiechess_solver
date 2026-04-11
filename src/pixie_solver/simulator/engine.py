from __future__ import annotations

from pixie_solver.core.event import StateDelta
from pixie_solver.core.hash import stable_position_hash
from pixie_solver.core.move import Move
from pixie_solver.core.piece import BasePieceType, Color
from pixie_solver.core.state import GameState
from pixie_solver.simulator.movegen import is_in_check, king_for_color, legal_moves
from pixie_solver.simulator.transition import apply_move_unchecked, other_color

REPETITION_DRAW_COUNT = 3
FIFTY_MOVE_RULE_HALFMOVES = 100


def apply_move(state: GameState, move: Move) -> tuple[GameState, StateDelta]:
    legal = legal_moves(state)
    if move not in legal:
        raise ValueError(f"Illegal move: {move.to_dict()!r}")
    return apply_move_unchecked(state, move)


def is_terminal(state: GameState) -> bool:
    return result(state) is not None


def result(state: GameState) -> str | None:
    has_white_king = False
    has_black_king = False
    for piece in state.active_pieces():
        piece_class = state.piece_classes[piece.piece_class_id]
        if piece_class.base_piece_type != BasePieceType.KING:
            continue
        if piece.color == Color.WHITE:
            has_white_king = True
        else:
            has_black_king = True
    if has_white_king and not has_black_king:
        return "white"
    if has_black_king and not has_white_king:
        return "black"
    if not has_white_king and not has_black_king:
        return "draw"
    moves = legal_moves(state)
    if moves:
        if is_draw_by_repetition(state) or is_draw_by_fifty_move_rule(state):
            return "draw"
        return None
    if king_for_color(state, state.side_to_move) is None:
        return other_color(state.side_to_move).value
    if is_in_check(state, state.side_to_move):
        return other_color(state.side_to_move).value
    return "draw"


def is_draw_by_repetition(state: GameState) -> bool:
    return (
        state.repetition_counts.get(stable_position_hash(state), 0)
        >= REPETITION_DRAW_COUNT
    )


def is_draw_by_fifty_move_rule(state: GameState) -> bool:
    return state.halfmove_clock >= FIFTY_MOVE_RULE_HALFMOVES
