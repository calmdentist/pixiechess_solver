from __future__ import annotations

from typing import Any, Iterable, Mapping

from pixie_solver.core.piece import Color, PieceInstance
from pixie_solver.core.query import QueryFact, stable_query_fact_id
from pixie_solver.core.state import GameState
from pixie_solver.program.contexts import QueryContext
from pixie_solver.program.lower_legacy_dsl import lower_legacy_piece_class
from pixie_solver.utils.squares import coords_to_square, normalize_square, square_to_coords

CAPTURE_CONTROL_QUERY_KIND = "capture_control"
LEGACY_CAPTURE_CONTROL_BLOCK_KIND = "legacy_capture_control"
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



def enumerate_query_facts(
    state: GameState,
    *,
    by_color: Color | None = None,
    query_kind: str | None = None,
    program_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[QueryFact, ...]:
    facts: list[QueryFact] = []
    for piece_id in sorted(state.piece_instances):
        piece = state.piece_instances[piece_id]
        if piece.square is None:
            continue
        if by_color is not None and piece.color != by_color:
            continue
        facts.extend(
            enumerate_query_facts_for_piece(
                state,
                piece_id=piece_id,
                query_kind=query_kind,
                program=(
                    program_registry.get(piece.piece_class_id)
                    if program_registry is not None
                    else None
                ),
            )
        )
    return tuple(sorted(facts, key=stable_query_fact_id))


def enumerate_query_facts_for_piece(
    state: GameState,
    *,
    piece_id: str,
    query_kind: str | None = None,
    program: Mapping[str, Any] | None = None,
) -> tuple[QueryFact, ...]:
    piece = state.piece_instances[piece_id]
    if piece.square is None:
        return ()
    active_program = _resolve_program_for_piece(
        state,
        piece_id=piece_id,
        program=program,
    )
    context = QueryContext(
        state=state,
        piece=piece,
        program=active_program,
        query_kind=query_kind,
    )
    facts: list[QueryFact] = []
    for query_block in _iter_query_blocks(active_program):
        facts.extend(_enumerate_query_block_facts(context=context, query_block=query_block))
    return tuple(sorted(facts, key=stable_query_fact_id))


def query_fact_exists(
    state: GameState,
    *,
    target_ref: str,
    by_color: Color | None = None,
    query_kind: str,
    program_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    normalized_target = _normalize_target_ref(target_ref, query_kind=query_kind)
    if normalized_target is None:
        return False
    for piece_id in sorted(state.piece_instances):
        piece = state.piece_instances[piece_id]
        if piece.square is None:
            continue
        if by_color is not None and piece.color != by_color:
            continue
        active_program = _resolve_program_for_piece(
            state,
            piece_id=piece_id,
            program=(
                program_registry.get(piece.piece_class_id)
                if program_registry is not None
                else None
            ),
        )
        context = QueryContext(
            state=state,
            piece=piece,
            program=active_program,
            query_kind=query_kind,
        )
        for query_block in _iter_query_blocks(active_program):
            if _query_block_matches_target(
                context=context,
                query_block=query_block,
                query_kind=query_kind,
                target_ref=normalized_target,
            ):
                return True
    return False


def is_square_capturable(
    state: GameState,
    square: str,
    *,
    by_color: Color,
    program_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    normalized_square = normalize_square(square)
    if normalized_square is None:
        return False
    return query_fact_exists(
        state,
        target_ref=normalized_square,
        by_color=by_color,
        query_kind=CAPTURE_CONTROL_QUERY_KIND,
        program_registry=program_registry,
    )


def is_king_capturable(
    state: GameState,
    color: Color,
    *,
    program_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    from pixie_solver.simulator.movegen import king_for_color
    from pixie_solver.simulator.transition import other_color

    king = king_for_color(state, color)
    if king is None or king.square is None:
        return False
    return is_square_capturable(
        state,
        king.square,
        by_color=other_color(color),
        program_registry=program_registry,
    )


def _enumerate_query_block_facts(
    *,
    context: QueryContext,
    query_block: Mapping[str, Any],
) -> list[QueryFact]:
    if query_block["kind"] != LEGACY_CAPTURE_CONTROL_BLOCK_KIND:
        raise ValueError(f"Unsupported query block kind: {query_block['kind']!r}")
    if context.query_kind not in (None, CAPTURE_CONTROL_QUERY_KIND):
        return []

    return [
        QueryFact(
            query_kind=CAPTURE_CONTROL_QUERY_KIND,
            subject_ref=context.piece.instance_id,
            target_ref=square,
            metadata={
                "program_id": str(context.program["program_id"]),
                "query_block_id": str(query_block["block_id"]),
            },
        )
        for square in _enumerate_legacy_capture_control_targets(
            state=context.state,
            occupancy=context.occupancy,
            piece=context.piece,
            query_params=dict(query_block["params"]),
        )
    ]


def _query_block_matches_target(
    *,
    context: QueryContext,
    query_block: Mapping[str, Any],
    query_kind: str,
    target_ref: str,
) -> bool:
    if query_block["kind"] != LEGACY_CAPTURE_CONTROL_BLOCK_KIND:
        raise ValueError(f"Unsupported query block kind: {query_block['kind']!r}")
    if query_kind != CAPTURE_CONTROL_QUERY_KIND:
        return False

    return _legacy_piece_controls_square(
        state=context.state,
        occupancy=context.occupancy,
        piece=context.piece,
        query_params=dict(query_block["params"]),
        target_square=target_ref,
    )


def _enumerate_legacy_capture_control_targets(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    query_params: Mapping[str, Any],
) -> tuple[str, ...]:
    controlled_squares: list[str] = []
    for file_index in range(8):
        for rank_index in range(8):
            square = coords_to_square(file_index, rank_index)
            if square is None:
                continue
            if _legacy_piece_controls_square(
                state=state,
                occupancy=occupancy,
                piece=piece,
                query_params=query_params,
                target_square=square,
            ):
                controlled_squares.append(square)
    return tuple(controlled_squares)


def _legacy_piece_controls_square(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    query_params: Mapping[str, Any],
    target_square: str,
) -> bool:
    if piece.square == target_square:
        return False

    base_archetype = str(query_params["base_archetype"])
    capture_config = _capture_config(query_params)
    if base_archetype == "pawn":
        return _pawn_controls_square(
            state=state,
            occupancy=occupancy,
            piece=piece,
            target_square=target_square,
            capture_config=capture_config,
        )
    if base_archetype == "knight":
        return _leaper_controls_square(
            state=state,
            occupancy=occupancy,
            piece=piece,
            target_square=target_square,
            offsets=KNIGHT_OFFSETS,
            capture_config=capture_config,
        )
    if base_archetype == "king":
        return _leaper_controls_square(
            state=state,
            occupancy=occupancy,
            piece=piece,
            target_square=target_square,
            offsets=KING_DIRECTIONS,
            capture_config=capture_config,
        )

    if base_archetype == "rook":
        directions = ORTHOGONAL_DIRECTIONS
    elif base_archetype == "bishop":
        directions = DIAGONAL_DIRECTIONS
    elif base_archetype == "queen":
        directions = KING_DIRECTIONS
    else:
        return False

    return _ray_controls_square(
        state=state,
        occupancy=occupancy,
        piece=piece,
        target_square=target_square,
        directions=directions,
        max_steps=_max_steps(query_params),
        phase_through_allies=_has_modifier(query_params, "phase_through_allies"),
        capture_config=capture_config,
    )


def _iter_query_blocks(program: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    query_blocks = tuple(program.get("query_blocks", ()))
    if query_blocks:
        return query_blocks
    # Compatibility fallback for older ProgramIR snapshots created before query_blocks.
    return tuple(
        {
            "block_id": str(action_block["block_id"]).replace(
                "legacy_base_actions",
                "legacy_capture_control",
            ),
            "kind": LEGACY_CAPTURE_CONTROL_BLOCK_KIND,
            "params": dict(action_block["params"]),
            "metadata": dict(action_block.get("metadata", {})),
        }
        for action_block in program.get("action_blocks", ())
        if action_block.get("kind") == "legacy_base_actions"
    )


def _normalize_target_ref(target_ref: str, *, query_kind: str) -> str | None:
    if query_kind == CAPTURE_CONTROL_QUERY_KIND:
        return normalize_square(target_ref)
    return str(target_ref)


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


def _pawn_controls_square(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    target_square: str,
    capture_config: Mapping[str, object],
) -> bool:
    file_index, rank_index = square_to_coords(piece.square)
    forward = 1 if piece.color == Color.WHITE else -1
    for file_delta in (-1, 1):
        candidate_square = coords_to_square(file_index + file_delta, rank_index + forward)
        if candidate_square != target_square:
            continue
        return _capture_reaches_square(
            state=state,
            occupancy=occupancy,
            piece=piece,
            target_square=target_square,
            direction=(file_delta, forward),
            capture_config=capture_config,
        )
    return False


def _leaper_controls_square(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    target_square: str,
    offsets: Iterable[tuple[int, int]],
    capture_config: Mapping[str, object],
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


def _ray_controls_square(
    *,
    state: GameState,
    occupancy: dict[str, str],
    piece: PieceInstance,
    target_square: str,
    directions: Iterable[tuple[int, int]],
    max_steps: int,
    phase_through_allies: bool,
    capture_config: Mapping[str, object],
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
    capture_config: Mapping[str, object],
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


def _max_steps(query_params: Mapping[str, Any]) -> int:
    base_steps = 7
    if str(query_params["base_archetype"]) == "king":
        base_steps = 1
    for modifier in query_params.get("movement_modifiers", []):
        op = str(modifier["op"])
        args = dict(modifier.get("args", {}))
        if op == "extend_range":
            base_steps += int(args["extra_steps"])
        elif op == "limit_range":
            base_steps = min(base_steps, int(args["max_steps"]))
    return max(base_steps, 1)


def _has_modifier(query_params: Mapping[str, Any], op_name: str) -> bool:
    return any(
        str(modifier["op"]) == op_name
        for modifier in query_params.get("movement_modifiers", [])
    )


def _capture_config(query_params: Mapping[str, Any]) -> dict[str, object]:
    capture_modifiers = list(query_params.get("capture_modifiers", []))
    if not capture_modifiers:
        return {"op": "inherit_base"}
    modifier = dict(capture_modifiers[0])
    op = str(modifier["op"])
    args = dict(modifier.get("args", {}))
    if op == "inherit_base":
        return {"op": "inherit_base"}
    if op == "replace_capture_with_push":
        return {
            "op": op,
            "distance": int(args["distance"]),
            "edge_behavior": str(args["edge_behavior"]),
        }
    raise ValueError(f"Unsupported capture modifier {op!r}")
