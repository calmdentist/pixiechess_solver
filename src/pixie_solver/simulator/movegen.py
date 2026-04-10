from __future__ import annotations

from typing import Iterable

from pixie_solver.core.move import Move
from pixie_solver.core.piece import BasePieceType, Color, PieceClass, PieceInstance
from pixie_solver.core.state import GameState
from pixie_solver.simulator.transition import apply_move_unchecked, other_color
from pixie_solver.utils.squares import coords_to_square, square_to_coords

ORTHOGONAL_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONAL_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KING_DIRECTIONS = ORTHOGONAL_DIRECTIONS + DIAGONAL_DIRECTIONS
KNIGHT_OFFSETS = (
    (1, 2),
    (2, 1),
    (-1, 2),
    (-2, 1),
    (1, -2),
    (2, -1),
    (-1, -2),
    (-2, -1),
)


def legal_moves(state: GameState) -> list[Move]:
    moves = pseudo_legal_moves(state, color=state.side_to_move)
    king = king_for_color(state, state.side_to_move)
    if king is None:
        return _sorted_moves(moves)

    filtered_moves: list[Move] = []
    for move in moves:
        next_state, _ = apply_move_unchecked(state, move)
        next_king = king_for_color(next_state, state.side_to_move)
        if next_king is None:
            continue
        if is_in_check(next_state, state.side_to_move):
            continue
        filtered_moves.append(move)
    return _sorted_moves(filtered_moves)


def pseudo_legal_moves(state: GameState, *, color: Color | None = None) -> list[Move]:
    moves: list[Move] = []
    occupancy = state.occupancy()
    active_color = state.side_to_move if color is None else color
    for piece_id in sorted(state.piece_instances):
        piece = state.piece_instances[piece_id]
        if piece.square is None or piece.color != active_color:
            continue
        piece_class = state.piece_classes[piece.piece_class_id]
        moves.extend(
            _generate_piece_moves(
                state=state,
                occupancy=occupancy,
                piece=piece,
                piece_class=piece_class,
            )
        )
    return _sorted_moves(moves)


def is_in_check(state: GameState, color: Color) -> bool:
    king = king_for_color(state, color)
    if king is None or king.square is None:
        return False
    enemy_color = other_color(color)
    for move in pseudo_legal_moves(state, color=enemy_color):
        if _move_targets_piece(move, king.instance_id, king.square):
            return True
    return False


def king_for_color(state: GameState, color: Color) -> PieceInstance | None:
    for piece in state.active_pieces():
        if piece.color != color:
            continue
        piece_class = state.piece_classes[piece.piece_class_id]
        if piece_class.base_piece_type == BasePieceType.KING:
            return piece
    return None


def _generate_piece_moves(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    piece_class: PieceClass,
) -> list[Move]:
    if piece_class.base_piece_type == BasePieceType.PAWN:
        return _generate_pawn_moves(
            state=state,
            occupancy=occupancy,
            piece=piece,
            piece_class=piece_class,
        )
    return _generate_orthodox_moves(
        state=state,
        occupancy=occupancy,
        piece=piece,
        piece_class=piece_class,
    )


def _generate_orthodox_moves(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    piece_class: PieceClass,
) -> list[Move]:
    directions: tuple[tuple[int, int], ...]
    max_steps = _max_steps_for_piece(piece_class)
    if piece_class.base_piece_type == BasePieceType.ROOK:
        directions = ORTHOGONAL_DIRECTIONS
    elif piece_class.base_piece_type == BasePieceType.BISHOP:
        directions = DIAGONAL_DIRECTIONS
    elif piece_class.base_piece_type == BasePieceType.QUEEN:
        directions = KING_DIRECTIONS
    elif piece_class.base_piece_type == BasePieceType.KING:
        directions = KING_DIRECTIONS
        max_steps = 1
    elif piece_class.base_piece_type == BasePieceType.KNIGHT:
        return _generate_leaper_moves(
            state=state,
            occupancy=occupancy,
            piece=piece,
            offsets=KNIGHT_OFFSETS,
            piece_class=piece_class,
        )
    else:
        return []

    phase_through_allies = any(
        modifier.op == "phase_through_allies"
        for modifier in piece_class.movement_modifiers
    )
    return _generate_ray_moves(
        state=state,
        occupancy=occupancy,
        piece=piece,
        directions=directions,
        max_steps=max_steps,
        piece_class=piece_class,
        phase_through_allies=phase_through_allies,
    )


def _generate_ray_moves(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    directions: Iterable[tuple[int, int]],
    max_steps: int,
    piece_class: PieceClass,
    phase_through_allies: bool,
) -> list[Move]:
    moves: list[Move] = []
    file_index, rank_index = square_to_coords(piece.square)
    capture_config = _capture_config(piece_class)
    for dx, dy in directions:
        for step in range(1, max_steps + 1):
            square = coords_to_square(file_index + dx * step, rank_index + dy * step)
            if square is None:
                break
            occupant_id = occupancy.get(square)
            if occupant_id is None:
                moves.append(
                    Move(piece_id=piece.instance_id, from_square=piece.square, to_square=square)
                )
                continue

            occupant = state.piece_instances[occupant_id]
            if occupant.color == piece.color:
                if phase_through_allies:
                    continue
                break

            capture_move = _build_capture_move(
                state=state,
                piece=piece,
                piece_class=piece_class,
                target_piece=occupant,
                to_square=square,
                direction=(dx, dy),
                capture_config=capture_config,
            )
            if capture_move is not None:
                moves.append(capture_move)
            break
    return moves


def _generate_leaper_moves(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    offsets: Iterable[tuple[int, int]],
    piece_class: PieceClass,
) -> list[Move]:
    moves: list[Move] = []
    file_index, rank_index = square_to_coords(piece.square)
    capture_config = _capture_config(piece_class)
    for dx, dy in offsets:
        square = coords_to_square(file_index + dx, rank_index + dy)
        if square is None:
            continue
        occupant_id = occupancy.get(square)
        if occupant_id is None:
            moves.append(
                Move(piece_id=piece.instance_id, from_square=piece.square, to_square=square)
            )
            continue
        occupant = state.piece_instances[occupant_id]
        if occupant.color == piece.color:
            continue
        capture_move = _build_capture_move(
            state=state,
            piece=piece,
            piece_class=piece_class,
            target_piece=occupant,
            to_square=square,
            direction=(dx, dy),
            capture_config=capture_config,
        )
        if capture_move is not None:
            moves.append(capture_move)
    return moves


def _generate_pawn_moves(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    piece_class: PieceClass,
) -> list[Move]:
    moves: list[Move] = []
    file_index, rank_index = square_to_coords(piece.square)
    forward = 1 if piece.color == Color.WHITE else -1
    start_rank = 1 if piece.color == Color.WHITE else 6
    capture_config = _capture_config(piece_class)

    one_forward = coords_to_square(file_index, rank_index + forward)
    if one_forward is not None and one_forward not in occupancy:
        moves.append(
            Move(
                piece_id=piece.instance_id,
                from_square=piece.square,
                to_square=one_forward,
            )
        )
        two_forward = coords_to_square(file_index, rank_index + 2 * forward)
        if rank_index == start_rank and two_forward is not None and two_forward not in occupancy:
            moves.append(
                Move(
                    piece_id=piece.instance_id,
                    from_square=piece.square,
                    to_square=two_forward,
                )
            )

    for file_delta in (-1, 1):
        target_square = coords_to_square(file_index + file_delta, rank_index + forward)
        if target_square is None:
            continue
        target_id = occupancy.get(target_square)
        if target_id is None:
            continue
        target_piece = state.piece_instances[target_id]
        if target_piece.color == piece.color:
            continue
        capture_move = _build_capture_move(
            state=state,
            piece=piece,
            piece_class=piece_class,
            target_piece=target_piece,
            to_square=target_square,
            direction=(file_delta, forward),
            capture_config=capture_config,
        )
        if capture_move is not None:
            moves.append(capture_move)
    return moves


def _build_capture_move(
    *,
    state: GameState,
    piece: PieceInstance,
    piece_class: PieceClass,
    target_piece: PieceInstance,
    to_square: str,
    direction: tuple[int, int],
    capture_config: dict[str, object],
) -> Move | None:
    if capture_config["op"] == "inherit_base":
        return Move(
            piece_id=piece.instance_id,
            from_square=piece.square,
            to_square=to_square,
            move_kind="capture",
            captured_piece_id=target_piece.instance_id,
        )

    if capture_config["op"] == "replace_capture_with_push":
        if not _push_is_legal(
            state=state,
            target_piece=target_piece,
            direction=direction,
            distance=int(capture_config["distance"]),
            edge_behavior=str(capture_config["edge_behavior"]),
        ):
            return None
        return Move(
            piece_id=piece.instance_id,
            from_square=piece.square,
            to_square=to_square,
            move_kind="push_capture",
            metadata={
                "target_piece_id": target_piece.instance_id,
                "push_direction": list(direction),
                "push_distance": int(capture_config["distance"]),
                "push_edge_behavior": str(capture_config["edge_behavior"]),
            },
        )

    raise ValueError(f"Unsupported capture op for {piece_class.class_id!r}")


def _push_is_legal(
    *,
    state: GameState,
    target_piece: PieceInstance,
    direction: tuple[int, int],
    distance: int,
    edge_behavior: str,
) -> bool:
    file_index, rank_index = square_to_coords(target_piece.square)
    occupancy = state.occupancy()
    current_file, current_rank = file_index, rank_index
    for _ in range(distance):
        current_file += direction[0]
        current_rank += direction[1]
        pushed_square = coords_to_square(current_file, current_rank)
        if pushed_square is None:
            return edge_behavior == "remove_if_pushed_off_board"
        if occupancy.get(pushed_square) is not None:
            return False
    return True


def _max_steps_for_piece(piece_class: PieceClass) -> int:
    base_steps = 7
    if piece_class.base_piece_type == BasePieceType.KING:
        base_steps = 1
    for modifier in piece_class.movement_modifiers:
        if modifier.op == "extend_range":
            base_steps += int(modifier.args["extra_steps"])
        elif modifier.op == "limit_range":
            base_steps = min(base_steps, int(modifier.args["max_steps"]))
    return max(base_steps, 1)


def _capture_config(piece_class: PieceClass) -> dict[str, object]:
    if not piece_class.capture_modifiers:
        return {"op": "inherit_base"}
    modifier = piece_class.capture_modifiers[0]
    if modifier.op == "inherit_base":
        return {"op": "inherit_base"}
    if modifier.op == "replace_capture_with_push":
        return {
            "op": modifier.op,
            "distance": int(modifier.args["distance"]),
            "edge_behavior": str(modifier.args["edge_behavior"]),
        }
    raise ValueError(f"Unsupported capture modifier {modifier.op!r}")


def _move_targets_piece(move: Move, piece_id: str, square: str) -> bool:
    if move.captured_piece_id == piece_id:
        return True
    if move.metadata.get("target_piece_id") == piece_id:
        return True
    return move.move_kind in {"capture", "push_capture"} and move.to_square == square


def _sorted_moves(moves: list[Move]) -> list[Move]:
    return sorted(
        moves,
        key=lambda move: (
            move.piece_id,
            move.from_square,
            move.to_square,
            move.move_kind,
            move.captured_piece_id or "",
        ),
    )
