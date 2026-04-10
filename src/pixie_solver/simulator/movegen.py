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
PROMOTION_PIECE_TYPES = ("queen", "rook", "bishop", "knight")
CASTLE_SPECS = {
    "king": {
        "rook_file": 7,
        "rook_to_file": 5,
        "king_to_file": 6,
        "clear_files": (5, 6),
        "safe_files": (5, 6),
    },
    "queen": {
        "rook_file": 0,
        "rook_to_file": 3,
        "king_to_file": 2,
        "clear_files": (1, 2, 3),
        "safe_files": (3, 2),
    },
}


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
    return is_square_attacked(state, king.square, by_color=other_color(color))


def is_square_attacked(state: GameState, square: str, *, by_color: Color) -> bool:
    occupancy = state.occupancy()
    for piece_id in sorted(state.piece_instances):
        piece = state.piece_instances[piece_id]
        if piece.square is None or piece.color != by_color:
            continue
        piece_class = state.piece_classes[piece.piece_class_id]
        if _piece_attacks_square(
            state=state,
            occupancy=occupancy,
            piece=piece,
            piece_class=piece_class,
            target_square=square,
        ):
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
    moves = _generate_ray_moves(
        state=state,
        occupancy=occupancy,
        piece=piece,
        directions=directions,
        max_steps=max_steps,
        piece_class=piece_class,
        phase_through_allies=phase_through_allies,
    )
    if piece_class.base_piece_type == BasePieceType.KING:
        moves.extend(_generate_castling_moves(state=state, occupancy=occupancy, piece=piece))
    return moves


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
        moves.extend(
            _promotion_moves(
                state=state,
                moving_piece=piece,
                move=Move(
                    piece_id=piece.instance_id,
                    from_square=piece.square,
                    to_square=one_forward,
                ),
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
        if target_id is not None:
            target_piece = state.piece_instances[target_id]
            if target_piece.color != piece.color:
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
                    moves.extend(
                        _promotion_moves(
                            state=state,
                            moving_piece=piece,
                            move=capture_move,
                        )
                    )
            continue

        if target_square == state.en_passant_square:
            captured_square = coords_to_square(file_index + file_delta, rank_index)
            captured_piece_id = occupancy.get(captured_square)
            if captured_piece_id is None:
                continue
            captured_piece = state.piece_instances[captured_piece_id]
            captured_class = state.piece_classes[captured_piece.piece_class_id]
            if (
                captured_piece.color != piece.color
                and captured_class.base_piece_type == BasePieceType.PAWN
            ):
                moves.append(
                    Move(
                        piece_id=piece.instance_id,
                        from_square=piece.square,
                        to_square=target_square,
                        move_kind="en_passant_capture",
                        captured_piece_id=captured_piece_id,
                        metadata={"captured_square": captured_square},
                    )
                )
    return moves


def _generate_castling_moves(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
) -> list[Move]:
    rights = state.castling_rights.get(piece.color.value, ())
    if not rights:
        return []

    home_rank = 0 if piece.color == Color.WHITE else 7
    home_square = coords_to_square(4, home_rank)
    if piece.square != home_square or is_square_attacked(
        state, piece.square, by_color=other_color(piece.color)
    ):
        return []

    moves: list[Move] = []
    for side in rights:
        spec = CASTLE_SPECS.get(side)
        if spec is None:
            continue
        rook_square = coords_to_square(int(spec["rook_file"]), home_rank)
        rook_piece_id = occupancy.get(rook_square)
        if rook_piece_id is None:
            continue
        rook_piece = state.piece_instances[rook_piece_id]
        rook_class = state.piece_classes[rook_piece.piece_class_id]
        if (
            rook_piece.color != piece.color
            or rook_piece.square != rook_square
            or rook_class.base_piece_type != BasePieceType.ROOK
        ):
            continue
        if any(
            occupancy.get(coords_to_square(file_index, home_rank)) is not None
            for file_index in spec["clear_files"]
        ):
            continue
        if any(
            is_square_attacked(
                state,
                coords_to_square(file_index, home_rank),
                by_color=other_color(piece.color),
            )
            for file_index in spec["safe_files"]
        ):
            continue
        destination = coords_to_square(int(spec["king_to_file"]), home_rank)
        rook_to_square = coords_to_square(int(spec["rook_to_file"]), home_rank)
        moves.append(
            Move(
                piece_id=piece.instance_id,
                from_square=piece.square,
                to_square=destination,
                move_kind="castle",
                tags=(f"castle_{side}",),
                metadata={
                    "side": side,
                    "rook_piece_id": rook_piece_id,
                    "rook_from_square": rook_square,
                    "rook_to_square": rook_to_square,
                },
            )
        )
    return moves


def _promotion_moves(
    *,
    state: GameState,
    moving_piece: PieceInstance,
    move: Move,
) -> list[Move]:
    _, rank_index = square_to_coords(move.to_square)
    promotion_rank = 7 if moving_piece.color == Color.WHITE else 0
    if rank_index != promotion_rank:
        return [move]

    moves: list[Move] = []
    for promotion_piece_type in PROMOTION_PIECE_TYPES:
        moves.append(
            Move(
                piece_id=move.piece_id,
                from_square=move.from_square,
                to_square=move.to_square,
                move_kind=move.move_kind,
                captured_piece_id=move.captured_piece_id,
                promotion_piece_type=promotion_piece_type,
                tags=move.tags + (f"promote_{promotion_piece_type}",),
                metadata={
                    **move.metadata,
                    "promotion_class_id": _resolve_promotion_class_id(
                        state, promotion_piece_type
                    ),
                },
            )
        )
    return moves


def _resolve_promotion_class_id(state: GameState, promotion_piece_type: str) -> str:
    target_type = BasePieceType(promotion_piece_type)
    preferred_ids = (
        f"baseline_{promotion_piece_type}",
        f"orthodox_{promotion_piece_type}",
        promotion_piece_type,
    )
    for preferred_id in preferred_ids:
        piece_class = state.piece_classes.get(preferred_id)
        if piece_class is not None and piece_class.base_piece_type == target_type:
            return preferred_id

    candidates = sorted(
        class_id
        for class_id, piece_class in state.piece_classes.items()
        if piece_class.base_piece_type == target_type
    )
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Could not resolve unique promotion class for {promotion_piece_type!r}: "
        f"{candidates!r}"
    )


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


def _piece_attacks_square(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    piece_class: PieceClass,
    target_square: str,
) -> bool:
    if piece.square == target_square:
        return False

    capture_config = _capture_config(piece_class)
    if piece_class.base_piece_type == BasePieceType.PAWN:
        return _pawn_attacks_square(
            state=state,
            occupancy=occupancy,
            piece=piece,
            target_square=target_square,
            capture_config=capture_config,
        )
    if piece_class.base_piece_type == BasePieceType.KNIGHT:
        return _leaper_attacks_square(
            state=state,
            occupancy=occupancy,
            piece=piece,
            target_square=target_square,
            offsets=KNIGHT_OFFSETS,
            capture_config=capture_config,
        )
    if piece_class.base_piece_type == BasePieceType.KING:
        return _leaper_attacks_square(
            state=state,
            occupancy=occupancy,
            piece=piece,
            target_square=target_square,
            offsets=KING_DIRECTIONS,
            capture_config=capture_config,
        )

    phase_through_allies = any(
        modifier.op == "phase_through_allies"
        for modifier in piece_class.movement_modifiers
    )
    if piece_class.base_piece_type == BasePieceType.ROOK:
        directions = ORTHOGONAL_DIRECTIONS
    elif piece_class.base_piece_type == BasePieceType.BISHOP:
        directions = DIAGONAL_DIRECTIONS
    elif piece_class.base_piece_type == BasePieceType.QUEEN:
        directions = KING_DIRECTIONS
    else:
        return False
    return _ray_attacks_square(
        state=state,
        occupancy=occupancy,
        piece=piece,
        target_square=target_square,
        directions=directions,
        max_steps=_max_steps_for_piece(piece_class),
        phase_through_allies=phase_through_allies,
        capture_config=capture_config,
    )


def _pawn_attacks_square(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    target_square: str,
    capture_config: dict[str, object],
) -> bool:
    file_index, rank_index = square_to_coords(piece.square)
    forward = 1 if piece.color == Color.WHITE else -1
    for file_delta in (-1, 1):
        candidate_square = coords_to_square(file_index + file_delta, rank_index + forward)
        if candidate_square == target_square:
            return _capture_reaches_square(
                state=state,
                occupancy=occupancy,
                piece=piece,
                target_square=target_square,
                direction=(file_delta, forward),
                capture_config=capture_config,
            )
    return False


def _leaper_attacks_square(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    target_square: str,
    offsets: Iterable[tuple[int, int]],
    capture_config: dict[str, object],
) -> bool:
    file_index, rank_index = square_to_coords(piece.square)
    for dx, dy in offsets:
        candidate_square = coords_to_square(file_index + dx, rank_index + dy)
        if candidate_square != target_square:
            continue
        return _capture_reaches_square(
            state=state,
            occupancy=occupancy,
            piece=piece,
            target_square=target_square,
            direction=(dx, dy),
            capture_config=capture_config,
        )
    return False


def _ray_attacks_square(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    target_square: str,
    directions: Iterable[tuple[int, int]],
    max_steps: int,
    phase_through_allies: bool,
    capture_config: dict[str, object],
) -> bool:
    file_index, rank_index = square_to_coords(piece.square)
    for dx, dy in directions:
        for step in range(1, max_steps + 1):
            candidate_square = coords_to_square(
                file_index + dx * step, rank_index + dy * step
            )
            if candidate_square is None:
                break
            occupant_id = occupancy.get(candidate_square)
            if candidate_square == target_square:
                if occupant_id is not None:
                    occupant = state.piece_instances[occupant_id]
                    if occupant.color == piece.color:
                        return False
                return _capture_reaches_square(
                    state=state,
                    occupancy=occupancy,
                    piece=piece,
                    target_square=target_square,
                    direction=(dx, dy),
                    capture_config=capture_config,
                )
            if occupant_id is None:
                continue
            occupant = state.piece_instances[occupant_id]
            if occupant.color == piece.color and phase_through_allies:
                continue
            break
    return False


def _capture_reaches_square(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    target_square: str,
    direction: tuple[int, int],
    capture_config: dict[str, object],
) -> bool:
    occupant_id = occupancy.get(target_square)
    if occupant_id is not None:
        occupant = state.piece_instances[occupant_id]
        if occupant.color == piece.color:
            return False
    if capture_config["op"] == "inherit_base":
        return True
    return _push_target_is_legal(
        occupancy=occupancy,
        target_square=target_square,
        target_piece_id=occupant_id,
        direction=direction,
        distance=int(capture_config["distance"]),
        edge_behavior=str(capture_config["edge_behavior"]),
    )


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


def _push_target_is_legal(
    *,
    occupancy: dict[str, str],
    target_square: str,
    target_piece_id: str | None,
    direction: tuple[int, int],
    distance: int,
    edge_behavior: str,
) -> bool:
    file_index, rank_index = square_to_coords(target_square)
    current_file, current_rank = file_index, rank_index
    for _ in range(distance):
        current_file += direction[0]
        current_rank += direction[1]
        pushed_square = coords_to_square(current_file, current_rank)
        if pushed_square is None:
            return edge_behavior == "remove_if_pushed_off_board"
        occupant_id = occupancy.get(pushed_square)
        if occupant_id is not None and occupant_id != target_piece_id:
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


def _sorted_moves(moves: list[Move]) -> list[Move]:
    return sorted(
        moves,
        key=lambda move: (
            move.piece_id,
            move.from_square,
            move.to_square,
            move.move_kind,
            move.captured_piece_id or "",
            move.promotion_piece_type or "",
        ),
    )
